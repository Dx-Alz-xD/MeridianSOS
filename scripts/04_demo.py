"""End-to-end demonstration: forecast a held-out storm and issue warnings.

Loads the trained surrogate, picks a test event it has never seen, and at a
chosen issue time produces the 6-hour forecast, the district alert list, the
equity breakdown, and a figure comparing prediction to the physics truth.

Also times the surrogate against the simulator, which is the whole argument
for the approach: the simulator is the better model and always will be, but
it cannot run fast enough, or often enough, to warn anyone.

Usage:  python scripts/04_demo.py [--event ID] [--t0 HOURS]
"""
from __future__ import annotations
import argparse, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from meridian.config import CFG, OUT
from meridian.worldgen.city import load_city
from meridian.ml.dataset import (load_index, load_event, event_frames,
                                 Router, static_features, FEATURE_NAMES,
                                 FLOOD_THRESHOLD_M, LEADS)
from meridian.ml import metrics as M
from meridian import risk, viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=int, default=None)
    ap.add_argument("--t0", type=int, default=7)
    args = ap.parse_args()

    city = load_city()
    land = ~city["sea"]
    land_flat = land.ravel()
    with open(OUT / "models" / "surrogate.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models, test_ids = bundle["models"], bundle["test_ids"]

    index = {m["id"]: m for m in load_index()}
    if args.event is None:
        cand = [i for i in test_ids if i in index]
        args.event = max(cand, key=lambda i: index[i]["frac_flooded"])
    seen = "TEST (unseen)" if args.event in test_ids else "TRAIN"
    meta = index[args.event]
    print(f"event {args.event} [{seen}]  {meta['severity']}  "
          f"{meta['total_mm']:.0f} mm  ->  {100*meta['frac_flooded']:.2f}% flooded")

    ev = load_event(args.event)
    router = Router(city)
    static = static_features(city)
    t0 = args.t0

    t = time.time()
    X, Y = event_frames(city, router, static, ev, t0)
    Xf = X.reshape(X.shape[0], -1)[:, land_flat].T
    feat_s = time.time() - t

    print(f"\nforecast issued at t = {t0} h")
    print(f"  feature construction {1000*feat_s:6.0f} ms")
    preds = {}
    for li, lead in enumerate(LEADS):
        t = time.time()
        p = models[lead]["clf"].predict_proba(Xf)[:, 1]
        d = np.expm1(models[lead]["reg"].predict(Xf)) / 100.0
        infer_s = time.time() - t
        grid_p = np.zeros(land.size, np.float32); grid_p[land_flat] = p
        grid_d = np.zeros(land.size, np.float32); grid_d[land_flat] = np.clip(d, 0, None)
        preds[lead] = (grid_p.reshape(land.shape), grid_d.reshape(land.shape))
        obs = Y[li].ravel()[land_flat] > FLOOD_THRESHOLD_M
        r = M.summarise(p, Y[li].ravel()[land_flat], name=f"+{lead}h")
        print(f"  +{lead}h  inference {1000*infer_s:5.0f} ms   "
              f"CSI {r['CSI']:.3f}  POD {r['POD']:.3f}  FAR {r['FAR']:.3f}  "
              f"AUC {r['roc_auc']:.4f}   ({obs.sum():,} cells truly wet)")

    # ---- warnings from the +6h forecast -----------------------------------
    lead = LEADS[-1]
    pred_depth = preds[lead][1]
    truth = Y[-1]
    print(f"\n--- WARNING PRODUCT, +{lead} h ---")
    s_pred = risk.city_summary(city, pred_depth)
    s_true = risk.city_summary(city, truth)
    print(f"  people >10cm   predicted {s_pred['people_affected']:>9,.0f}   "
          f"actual {s_true['people_affected']:>9,.0f}")
    print(f"  area km2       predicted {s_pred['area_km2_affected']:>9.1f}   "
          f"actual {s_true['area_km2_affected']:>9.1f}")
    rows_p = risk.district_table(city, pred_depth)
    rows_t = risk.district_table(city, truth)
    print("\n  predicted top districts:")
    print("   " + risk.format_alerts(rows_p, 6).replace("\n", "\n   "))
    top_p = [r["district"] for r in rows_p[:6]]
    top_t = [r["district"] for r in rows_t[:6]]
    hit = len(set(top_p) & set(top_t))
    print(f"\n  top-6 district overlap with truth: {hit}/6  -> {sorted(set(top_p)&set(top_t))}")

    print("\n  equity (from the predicted map):")
    for r in risk.equity_report(city, pred_depth):
        print(f"    {r['landuse']:<13} exposed {100*r['exposed_rate']:5.1f}%  "
              f"({r['exposure_ratio']:.2f}x citywide average)")

    # ---- figure ------------------------------------------------------------
    sh = viz.hillshade(city["dem"], city["sea"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    vmax = max(float(truth.max()), 0.5)

    def depth_panel(ax, d, title):
        ax.imshow(sh, cmap="gray"); viz.draw_sea(ax, city["sea"])
        ax.imshow(np.ma.masked_where((d < 0.05) | city["sea"], d), cmap="turbo",
                  norm=LogNorm(vmin=0.05, vmax=vmax))
        viz.clean(ax, title, 11)

    for j, lead in enumerate(LEADS):
        depth_panel(axes[0, j], preds[lead][1], f"PREDICTED depth, +{lead} h")
    nh = ev["depth_h"].shape[0]
    for j, lead in enumerate(LEADS):
        depth_panel(axes[1, j], ev["depth_h"][min(t0 + lead, nh - 1)],
                    f"PHYSICS TRUTH, +{lead} h")

    plt.suptitle(f"Meridian City flood surrogate — unseen {meta['severity']} event "
                 f"({meta['total_mm']:.0f} mm), forecast issued t={t0} h",
                 fontsize=14, y=.995)
    plt.tight_layout()
    plt.savefig(OUT / "fig_forecast.png", dpi=104, bbox_inches="tight")
    print(f"\nsaved {OUT/'fig_forecast.png'}")

    # ---- speed argument ----------------------------------------------------
    from meridian.sim.flood import simulate_event
    from meridian.sim.storms import generate_event
    rng = np.random.default_rng(0)
    e2 = generate_event(CFG, rng, severity="heavy",
                        orography=np.clip(city["dem"], 0, None))
    t = time.time()
    simulate_event(city, e2["rain_obs"], e2["dt_h"], CFG, record_every=4)
    sim_s = time.time() - t
    ml_s = feat_s + 3 * infer_s
    # Be precise about what is being compared. The simulator produces a full
    # 18 h event; the surrogate produces one citywide forecast at three lead
    # times. The operational claim is not "N times faster at the same job" --
    # it is that the surrogate is cheap enough to re-run on every radar
    # refresh, and to run as an ensemble, which the simulator is not.
    print("\nspeed")
    print(f"  physics simulator {sim_s:7.1f} s  to simulate one 18 h event")
    print(f"  surrogate         {ml_s:7.2f} s  for a citywide forecast at "
          f"+1/+3/+6 h from one issue time")
    print(f"  re-forecasting every 5 min through an 18 h storm costs "
          f"{ml_s*12*18/60:.1f} min of compute with the surrogate")
    print(f"  the simulator would need {sim_s*12*18/60:.0f} min for the same "
          f"cadence, i.e. it cannot keep up with real time")


if __name__ == "__main__":
    main()
