"""Check that the dashboard's JavaScript router still agrees with Python.

`outputs/web/template_interactive.html` re-implements
`meridian/routing.py::route()` in JavaScript so the page can route between
two points the user picks, with no server. Two copies of one cost model is
exactly the kind of thing that drifts silently, so this script runs the
PYTHON router over the same quantised rasters the browser sees and prints
the numbers for comparison.

It deliberately routes on the PNG-decoded depth field, not on the float32
simulator output, because that is what the page has. Any difference between
this script and the browser is an algorithm difference, not a quantisation
difference -- quantisation is already baked into both sides.

Usage:
    python scripts/98_verify_web_routing.py            # print the pairs
    python scripts/98_verify_web_routing.py --json     # machine-readable

Then run the JS half in the page console (the script prints the snippet)
and compare. Travel times should agree to well under 0.1 min; paths can
differ where two routes tie, which is why length and depth are printed too.
"""
from __future__ import annotations
import argparse, base64, io, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meridian.config import OUT
from meridian import routing as RT

WEB = OUT / "web"


def decode(uri: str, n: int) -> np.ndarray:
    raw = base64.b64decode(uri.split(",", 1)[1])
    img = plt.imread(io.BytesIO(raw))
    if img.ndim == 3:
        img = img[..., 0]
    v = np.round(img * 255) if img.max() <= 1.0 else img
    return v.astype(np.int32).reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--pairs", type=int, default=6)
    args = ap.parse_args()

    payload = json.loads((WEB / "payload.json").read_text(encoding="utf-8"))
    n = int(payload["city"]["n"])
    cell_m = float(payload["city"]["cell_m"])

    # ---- rebuild exactly what the browser builds -------------------------
    road_id = decode(payload["city"]["road_id"], n) - 1      # stored as id+1
    pot_raw = decode(payload["city"]["potholes"], n)
    seg = {s["id"]: s for s in payload["roads"]["segments"]}
    classes = payload["roads"]["classes"]

    rqi = np.full((n, n), np.nan, np.float32)
    kmh = np.zeros((n, n), np.float32)
    on = road_id >= 0
    ys, xs = np.nonzero(on)
    for y, x in zip(ys, xs):
        s = seg.get(int(road_id[y, x]))
        if s is None:
            on[y, x] = False
            continue
        rqi[y, x] = s["rqi"]
        kmh[y, x] = classes.get(s["cls"], {"speed": 30})["speed"]
    graph = {"road_id": road_id, "rqi": rqi, "design_kmh": kmh,
             "pothole": np.round(pot_raw / 20.0).astype(np.float32),
             "on_road": on}

    # depth field: the +6 h operational forecast at the mid scenario, which
    # is what the dashboard opens on
    si = min(3, len(payload["scenarios"]) - 1)
    lead = payload["model"]["leads"][-1]
    codec = payload["codec"]["depth"]
    dv = decode(payload["scenarios"][si]["leads"][f"fc_{lead}"]["pred_depth"], n)
    depth = (codec["max"] * (dv / 255.0) ** codec["gamma"]).astype(np.float32)

    # ---- pairs: control centre -> the first few household access points ---
    ctl = tuple(payload["control"]["yx"])
    hh = payload["registry"]["households"][: args.pairs]
    rows = []
    for h in hh:
        goal = (int(h["ry"]), int(h["rx"]))
        r = RT.route(graph, depth, ctl, goal, cell_m=cell_m)
        rows.append({
            "code": h["code"],
            "start": [int(ctl[0]), int(ctl[1])],
            "goal": [goal[0], goal[1]],
            "ok": bool(r["ok"]),
            "minutes": round(r.get("minutes", 0.0), 3),
            "km": round(r.get("km", 0.0), 3),
            "cells": len(r.get("path", [])),
            "max_depth_cm": round(100 * r.get("max_depth_m", 0.0), 1),
            "submerged": r.get("submerged_potholes", 0),
            "min_rqi": round(r.get("min_rqi", 0.0), 1),
        })

    if args.json:
        print(json.dumps(rows, indent=1))
        return

    print(f"scenario {si} ({payload['scenarios'][si]['total_mm']:.0f} mm), "
          f"+{lead} h operational forecast, {int(on.sum())} road cells")
    print(f"control centre at {ctl}\n")
    print(f"{'house':<14}{'ok':<4}{'minutes':>9}{'km':>7}{'cells':>7}"
          f"{'maxcm':>7}{'subm':>6}{'minRQI':>8}")
    for r in rows:
        print(f"{r['code']:<14}{str(r['ok']):<4}{r['minutes']:>9.3f}"
              f"{r['km']:>7.2f}{r['cells']:>7}{r['max_depth_cm']:>7.1f}"
              f"{r['submerged']:>6}{r['min_rqi']:>8.1f}")

    js = ("const dep=depthField('pred');"
          "JSON.stringify(" + json.dumps([[r["goal"][0], r["goal"][1]] for r in rows])
          + ".map(g=>{const r=dijkstra(DATA.control.yx[0]*"
          + str(n) + "+DATA.control.yx[1], g[0]*" + str(n)
          + "+g[1], dep, false, false);"
          "return r.ok?[+r.minutes.toFixed(3),+r.km.toFixed(2),r.path.length,"
          "+(100*r.maxD).toFixed(1),r.submerged,+r.minRqi.toFixed(1)]:null;}))")
    print("\nRun this in the dashboard console and compare row by row:\n")
    print(js)


if __name__ == "__main__":
    main()
