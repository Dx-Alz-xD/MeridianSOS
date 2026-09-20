"""Export a self-contained HTML dashboard for the flood surrogate.

Runs the trained model over several held-out storms, renders the maps as
transparent PNG overlays on a shared base map, and writes everything --
images base64-inlined, tables as JSON -- into a single file:

    outputs/web/index.html

No server, no build step, no network. Open it in a browser.

Usage:  python scripts/06_export_web.py
"""
from __future__ import annotations
import base64, io, json, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from meridian.config import CFG, OUT, LU_NAME
from meridian.worldgen.city import load_city
from meridian.ml.dataset import (load_index, load_event, event_frames,
                                 Router, static_features, FEATURE_NAMES,
                                 FLOOD_THRESHOLD_M, LEADS)
from meridian.ml import metrics as M
from meridian import risk, viz

WEB = OUT / "web"
WEB.mkdir(parents=True, exist_ok=True)


def png_b64(rgba: np.ndarray) -> str:
    """Encode an HxWx4 uint8 array as a base64 PNG data URI."""
    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def base_map(city) -> str:
    """Hillshaded terrain tinted by land use, as an opaque RGB image."""
    dem, sea = city["dem"], city["sea"]
    ls = LightSource(azdeg=315, altdeg=42)
    sh = ls.hillshade(np.where(sea, 0, dem), vert_exag=9,
                      dx=CFG.grid.cell_size_m, dy=CFG.grid.cell_size_m)
    lu_rgb = viz.LU_CMAP(viz.LU_NORM(city["landuse"]))[..., :3]
    img = 0.45 * lu_rgb + 0.55 * sh[..., None]
    img[sea] = np.array([0.07, 0.22, 0.36])
    rgba = np.concatenate([np.clip(img, 0, 1),
                           np.ones(img.shape[:2] + (1,))], axis=-1)
    return png_b64((rgba * 255).astype(np.uint8))


def depth_overlay(depth, sea, vmax=3.0) -> str:
    """Flood depth as a transparent overlay: dry cells are fully clear."""
    d = np.clip(depth, 0, None)
    wet = (d > 0.05) & ~sea
    norm = np.clip(np.log1p(d / 0.05) / np.log1p(vmax / 0.05), 0, 1)
    rgba = plt.get_cmap("turbo")(norm)
    rgba[..., 3] = np.where(wet, 0.88, 0.0)
    return png_b64((rgba * 255).astype(np.uint8))


def overlay_static(mask, color, alpha=0.9) -> str:
    rgba = np.zeros(mask.shape + (4,))
    rgba[..., :3] = np.array(color)
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return png_b64((rgba * 255).astype(np.uint8))


def expected_exposure(city, prob_grid):
    """Expected number of people in >10 cm, summed over per-cell probabilities.

    Taking this from the classifier rather than the depth regressor is not a
    convenience. The regressor is fit on log1p(depth) under squared error, and
    back-transforming through expm1 is biased low -- on a test storm it
    underestimated people at risk roughly fivefold. The classifier is trained
    directly on the question being asked ("will this cell exceed 10 cm?"), so
    the probability-weighted sum is both better calibrated and threshold-free.
    The regressor is still what colours the depth map, where magnitude, not
    headcount, is the point.
    """
    import numpy as _np
    return float((city["population"] * prob_grid * (~city["sea"])).sum())


def main():
    city = load_city()
    sea, land = city["sea"], ~city["sea"]
    land_flat = land.ravel()
    with open(OUT / "models" / "surrogate.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models, test_ids = bundle["models"], bundle["test_ids"]
    index = {m["id"]: m for m in load_index()}
    router, static = Router(city), static_features(city)

    # pick the most severe unseen event of each severity we have
    picks, used = [], set()
    for sev in ["extreme", "heavy", "moderate"]:
        cand = [i for i in test_ids if i in index
                and index[i]["severity"] == sev and i not in used]
        if not cand:
            continue
        eid = max(cand, key=lambda i: index[i]["frac_flooded"])
        picks.append(eid); used.add(eid)
    if not picks:
        picks = sorted(test_ids)[:2]
    print(f"scenarios: {picks}")

    scenarios = []
    for eid in picks:
        ev = load_event(eid)
        meta = index[eid]
        nh = ev["depth_h"].shape[0]
        t0 = max(issue for issue in [6, 7, 8] if issue + 6 < nh)
        t = time.time()
        X, Y = event_frames(city, router, static, ev, t0)
        Xf = X.reshape(X.shape[0], -1)[:, land_flat].T
        feat_ms = 1000 * (time.time() - t)

        leads = {}
        for li, lead in enumerate(LEADS):
            t = time.time()
            prob = models[lead]["clf"].predict_proba(Xf)[:, 1]
            dep = np.clip(np.expm1(models[lead]["reg"].predict(Xf)) / 100.0, 0, None)
            infer_ms = 1000 * (time.time() - t)
            g = np.zeros(land.size, np.float32); g[land_flat] = dep
            pred = g.reshape(land.shape)
            gp = np.zeros(land.size, np.float32); gp[land_flat] = prob
            prob_grid = gp.reshape(land.shape)
            truth = ev["depth_h"][min(t0 + lead, nh - 1)]
            r = M.summarise(prob, Y[li].ravel()[land_flat], name=f"+{lead}h")
            sp, st = risk.city_summary(city, pred), risk.city_summary(city, truth)
            leads[str(lead)] = {
                "pred_img": depth_overlay(pred, sea),
                "true_img": depth_overlay(truth, sea),
                "csi": r["CSI"], "pod": r["POD"], "far": r["FAR"],
                "auc": r["roc_auc"],
                "pred_people": expected_exposure(city, prob_grid),
                "true_people": st["people_affected"],
                "pred_area": sp["area_km2_affected"],
                "true_area": st["area_km2_affected"],
                "infer_ms": infer_ms,
                "districts": [
                    {"name": d["district"], "people": d["exposed_10cm"],
                     "frac": d["exposed_frac"], "depth": d["max_depth_m"],
                     "informal": d["informal_frac"]}
                    for d in risk.district_table(city, pred, prob=prob_grid)[:8]],
                "equity": [
                    {"type": e["landuse"].replace("_", " "),
                     "rate": e["exposed_rate"], "ratio": e["exposure_ratio"],
                     "drain": e["drain_mm_h"], "hand": e["median_hand_m"]}
                    for e in risk.equity_report(city, truth)
                    if e["population"] > 5000],
            }
        scenarios.append({
            "id": int(eid), "severity": meta["severity"],
            "total_mm": meta["total_mm"], "max_point_mm": meta["max_point_mm"],
            "t0": t0, "feat_ms": feat_ms,
            "rain_img": png_b64((plt.get_cmap("YlGnBu")(
                np.clip((ev["rain_obs_h"].sum(0) /
                         max(ev["rain_obs_h"].sum(0).max(), 1e-6)), 0, 1)
            ) * 255).astype(np.uint8)),
            "leads": leads,
        })
        print(f"  event {eid} ({meta['severity']}, {meta['total_mm']:.0f} mm) done")

    payload = {
        "city": {
            "name": "Meridian City",
            "population": float(city["population"].sum()),
            "extent_km": CFG.grid.extent_km,
            "area_km2": CFG.grid.area_km2,
            "n_districts": len(city["district_names"]),
            "base_img": base_map(city),
            "informal_img": overlay_static(city["landuse"] == 4, (0.95, 0.15, 0.25)),
            "ghost_img": overlay_static(city["ghost"], (1.0, 0.1, 0.9)),
            "channel_img": overlay_static(city["channels"], (0.35, 0.75, 1.0), 0.75),
        },
        "model": {
            "n_features": len(FEATURE_NAMES),
            "threshold_m": FLOOD_THRESHOLD_M,
            "leads": list(LEADS),
            "n_events": len(index),
            "n_test": len(test_ids),
        },
        "scenarios": scenarios,
    }
    try:
        payload["eval"] = json.loads((OUT / "eval_results.json").read_text())
    except Exception:
        payload["eval"] = {}

    tpl = (WEB / "template.html")
    html = tpl.read_text(encoding="utf-8")
    out = html.replace("/*__DATA__*/null",
                       json.dumps(payload, default=float))
    (WEB / "index.html").write_text(out, encoding="utf-8")
    mb = len(out.encode()) / 1e6
    print(f"\nwrote {WEB/'index.html'}  ({mb:.1f} MB, self-contained)")


if __name__ == "__main__":
    main()
