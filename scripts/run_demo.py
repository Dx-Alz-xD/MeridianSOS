"""Start the whole MeridianSOS demo: flood dashboard + SOS control room.

Two servers, one command:

    Meridian Flood Watch   http://localhost:8731   static, Python
    MeridianSOS control    http://localhost:8787   Node, meridian-sos-local/

They are linked to each other in both directions, and both are driven by the
same city model -- the SOS app's districts, gauges, shelters and alerts are
generated out of the simulator by `scripts/09_export_sos_seed.py`.

    python scripts/run_demo.py                 # start both
    python scripts/run_demo.py --scenario 5    # seed the SOS app at 190 mm
    python scripts/run_demo.py --reseed        # rebuild the SOS seed first

`--reseed` regenerates the seed AND deletes `meridian-data.json`, which throws
away anything typed into the chat app. Without it the app keeps whatever
state it already has, which is what you want between demo runs.

Ctrl+C stops both.
"""
from __future__ import annotations
import argparse, os, shutil, signal, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "outputs" / "web"
SOS = ROOT / "meridian-sos-local"
FLOOD_PORT = 8731
SOS_PORT = 8787


def lan_ip() -> str:
    """Best-effort LAN address, for the phones in the room."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))      # no packet is actually sent
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def port_busy(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3,
                    help="rainfall level the SOS app is seeded at (0-5)")
    ap.add_argument("--reseed", action="store_true",
                    help="regenerate the SOS seed and wipe its saved state")
    ap.add_argument("--no-sos", action="store_true",
                    help="flood dashboard only")
    args = ap.parse_args()

    if not (WEB / "index.html").exists():
        raise SystemExit("outputs/web/index.html is missing — run "
                         "scripts/07_export_interactive.py first")

    env = dict(os.environ, PYTHONPATH=str(ROOT))

    if args.reseed and not args.no_sos:
        print("regenerating the SOS seed from the city model...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "09_export_sos_seed.py"),
                        "--scenario", str(args.scenario)],
                       cwd=ROOT, env=env, check=True)
        data = SOS / "meridian-data.json"
        if data.exists():
            data.unlink()
            print("  cleared meridian-data.json so the new seed loads")

    procs = []

    if port_busy(FLOOD_PORT):
        print(f"port {FLOOD_PORT} is already serving - reusing it")
    else:
        procs.append(("Flood Watch", subprocess.Popen(
            [sys.executable, "-m", "http.server", str(FLOOD_PORT),
             "--directory", str(WEB)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))

    if not args.no_sos:
        node = shutil.which("node")
        if not node:
            print("\n  node was not found on PATH, so the SOS control room is "
                  "not starting.\n  Install Node 18+ from https://nodejs.org, "
                  "or run with --no-sos.\n")
        elif port_busy(SOS_PORT):
            print(f"port {SOS_PORT} is already serving - reusing it")
        else:
            procs.append(("SOS control room", subprocess.Popen(
                [node, "server.js"], cwd=str(SOS),
                env=dict(env, PORT=str(SOS_PORT)))))
            time.sleep(1.2)     # let it print its admin claim code first

    ip = lan_ip()
    bar = "-" * 58
    print(f"\n  MeridianSOS demo\n  {bar}")
    print(f"  Flood Watch dashboard   http://localhost:{FLOOD_PORT}")
    print(f"  SOS control room        http://localhost:{SOS_PORT}")
    print(f"\n  On this WiFi (phones):  http://{ip}:{SOS_PORT}")
    print(f"                          http://{ip}:{FLOOD_PORT}")
    print(f"  {bar}")
    print("  The two are linked to each other in the top bar / sidebar.")
    print("  The admin claim code for the SOS app is printed above.")
    print("  Ctrl+C stops everything.\n")

    try:
        while True:
            for name, p in procs:
                if p.poll() is not None:
                    print(f"  {name} exited with code {p.returncode}")
                    raise KeyboardInterrupt
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  stopping...")
        for _, p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for _, p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("  stopped.")


if __name__ == "__main__":
    main()
