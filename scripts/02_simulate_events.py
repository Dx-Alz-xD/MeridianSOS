"""Generate the training corpus: N rainfall events, each run through the
physics simulator, saved as hourly rasters.

Each event stores
    rain_obs_h [H, n, n]  observed rainfall, hourly mean mm/h
    rain_fc_h  [H, n, n]  the forecast that was available beforehand
    depth_h    [H, n, n]  simulated surface water depth, m
plus scalar metadata. Stored as float16: depths are metres to ~1 mm precision
and rainfall to ~0.01 mm/h, which is far finer than the model needs, and it
keeps the whole corpus small enough to sit in memory during training.

Usage:  python scripts/02_simulate_events.py --n 80
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

from meridian.config import CFG, EVENT_DIR
from meridian.worldgen.city import load_city
from meridian.sim.storms import generate_event
from meridian.sim.flood import simulate_event


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="number of events")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    city = load_city()
    land = ~city["sea"]
    oro = np.clip(city["dem"], 0, None)
    rng = np.random.default_rng(args.seed)
    # steps per recorded hour
    rec = int(round(60 / CFG.storm.timestep_min))

    index = []
    t_start = time.time()
    for i in range(args.start, args.start + args.n):
        ev = generate_event(CFG, rng, orography=oro)
        t0 = time.time()
        sim = simulate_event(city, ev["rain_obs"], ev["dt_h"], CFG,
                             record_every=rec)
        dt = time.time() - t0

        nt, n = ev["nt"], CFG.grid.n
        nh = nt // rec
        # hourly-mean rainfall
        ro = ev["rain_obs"][:nh * rec].reshape(nh, rec, n, n).mean(1)
        rf = ev["rain_fc"][:nh * rec].reshape(nh, rec, n, n).mean(1)
        depth = sim["frames"][:nh]

        hm = sim["h_max"]
        fl = (hm > 0.10) & land
        meta = {
            "id": i, "severity": ev["severity"],
            "total_mm": ev["total_mm"], "max_point_mm": ev["max_point_mm"],
            "peak_mmh": ev["peak_mmh"], "bearing_deg": ev["bearing_deg"],
            "speed_kmh": ev["speed_kmh"],
            "frac_flooded": float(fl.sum() / land.sum()),
            "people_affected": float(city["population"][fl].sum()),
            "max_depth_m": float(hm.max()),
            "runoff_coeff": sim["runoff_coeff"],
            "sewer_saturated_h": sim["sewer_saturated_h"],
            "mass_error": sim["mass_error_frac"],
            "sim_seconds": dt,
        }
        np.savez_compressed(
            EVENT_DIR / f"event_{i:04d}.npz",
            rain_obs_h=ro.astype(np.float16),
            rain_fc_h=rf.astype(np.float16),
            depth_h=depth.astype(np.float16),
            h_max=hm.astype(np.float16),
            meta=np.array([meta], dtype=object),
        )
        index.append(meta)
        el = time.time() - t_start
        done = i - args.start + 1
        eta = el / done * (args.n - done)
        print(f"[{done:3d}/{args.n}] {ev['severity']:<9} "
              f"{ev['total_mm']:6.1f} mm -> flooded {100*meta['frac_flooded']:5.2f}% "
              f"| {meta['people_affected']/1000:7.1f}k people "
              f"| {dt:5.1f}s | ETA {eta/60:4.1f} min", flush=True)

    (EVENT_DIR / "index.json").write_text(json.dumps(index, indent=1),
                                          encoding="utf-8")
    sev = {}
    for m in index:
        sev[m["severity"]] = sev.get(m["severity"], 0) + 1
    print(f"\ndone: {len(index)} events in {(time.time()-t_start)/60:.1f} min")
    print("severity mix:", sev)
    fr = np.array([m["frac_flooded"] for m in index])
    print(f"flooded fraction: median {np.median(fr):.4f}  "
          f"p90 {np.quantile(fr,0.9):.4f}  max {fr.max():.4f}")
    print(f"events with any flooding: {(fr>0.001).sum()}/{len(fr)}")


if __name__ == "__main__":
    main()
