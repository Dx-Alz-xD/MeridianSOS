"""Train the flood surrogate and verify it honestly.

Trains, per lead time, a gradient-boosted classifier for P(depth > 10 cm) and
a regressor for depth, then scores them on held-out EVENTS against three
baselines:

  persistence    depth stays exactly as it is now. The bar any nowcast must
                 clear, and a surprisingly strong one at short lead times.
  HAND threshold the standard GIS flood-susceptibility proxy -- static, takes
                 no account of the storm at all.
  no-routing     the same model without upstream-accumulated rainfall, which
                 isolates what catchment routing is worth.

Usage:  python scripts/03_train.py
"""
from __future__ import annotations
import json, time, pickle
import numpy as np
import xgboost as xgb

from meridian.config import CFG, OUT
from meridian.worldgen.city import load_city
from meridian.ml.dataset import (load_index, load_event, build_table,
                                 split_events, event_frames, issue_times,
                                 Router, static_features,
                                 FEATURE_NAMES, FLOOD_THRESHOLD_M, LEADS)
from meridian.ml import metrics as M

MODEL_DIR = OUT / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

ROUTED = ["up_fc_1h", "up_fc_3h", "up_fc_6h", "up_past_3h",
          "up_cum_event", "up_depth_now"]


def full_raster_eval(model, cols, city, router, static, test_ids, li):
    """Predict every land cell of every held-out event at lead index li."""
    land_flat = (~city["sea"]).ravel()
    scores, obs, now = [], [], []
    for eid in test_ids:
        ev = load_event(eid)
        nh = ev["depth_h"].shape[0]
        for t0 in issue_times(nh)[::3]:
            X, Y = event_frames(city, router, static, ev, t0)
            Xf = X.reshape(X.shape[0], -1)[:, land_flat].T
            scores.append(model.predict_proba(Xf[:, cols])[:, 1].astype(np.float32))
            obs.append(Y[li].ravel()[land_flat].astype(np.float32))
            now.append(X[FEATURE_NAMES.index("depth_now")].ravel()[land_flat])
    return (np.concatenate(scores), np.concatenate(obs), np.concatenate(now))


def main():
    t_start = time.time()
    city = load_city()
    index = load_index()
    if not index:
        raise SystemExit("no events found - run scripts/02_simulate_events.py first")
    tr, va, te = split_events(index)
    print(f"events: {len(index)} total -> train {len(tr)} / val {len(va)} / test {len(te)}")

    router = Router(city)
    static = static_features(city)
    rng = np.random.default_rng(11)

    print("\nbuilding training table...")
    Xtr, Ytr, etr, ctr = build_table(tr, city, router, static, rng)
    Xva, Yva, eva, cva = build_table(va, city, router, static, rng)
    print(f"  train {Xtr.shape}  val {Xva.shape}  "
          f"wet rate {100*(Ytr[:,-1]>FLOOD_THRESHOLD_M).mean():.1f}%")

    all_cols = np.arange(len(FEATURE_NAMES))
    no_route = np.array([i for i, nm in enumerate(FEATURE_NAMES) if nm not in ROUTED])

    results, models = {}, {}
    for li, lead in enumerate(LEADS):
        ytr = (Ytr[:, li] > FLOOD_THRESHOLD_M).astype(int)
        yva = (Yva[:, li] > FLOOD_THRESHOLD_M).astype(int)
        print(f"\n=== lead +{lead} h  (train wet {100*ytr.mean():.1f}%) ===")

        clf = xgb.XGBClassifier(
            n_estimators=350, max_depth=7, learning_rate=0.09,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=4,
            reg_lambda=1.5, eval_metric="aucpr", early_stopping_rounds=25,
            n_jobs=-1, tree_method="hist", random_state=0)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=7, learning_rate=0.09,
            subsample=0.85, colsample_bytree=0.8, min_child_weight=4,
            eval_metric="rmse", early_stopping_rounds=25,
            n_jobs=-1, tree_method="hist", random_state=0)
        reg.fit(Xtr, np.log1p(Ytr[:, li] * 100.0),
                eval_set=[(Xva, np.log1p(Yva[:, li] * 100.0))], verbose=False)

        abl = None
        if lead == LEADS[-1]:          # ablation only at the headline lead
            abl = xgb.XGBClassifier(
                n_estimators=350, max_depth=7, learning_rate=0.09,
                subsample=0.85, colsample_bytree=0.8, min_child_weight=4,
                reg_lambda=1.5, eval_metric="aucpr", early_stopping_rounds=25,
                n_jobs=-1, tree_method="hist", random_state=0)
            abl.fit(Xtr[:, no_route], ytr,
                    eval_set=[(Xva[:, no_route], yva)], verbose=False)

        print(f"  trained in {time.time()-t_start:.0f}s cumulative; scoring test set...")
        score, obs, depth_now = full_raster_eval(clf, all_cols, city, router,
                                                 static, te, li)
        rows = [M.summarise(score, obs, name=f"surrogate +{lead}h")]
        rows.append(M.summarise(depth_now, obs, pred_depth=depth_now,
                                name=f"persistence +{lead}h"))
        hand = np.tile(-city["hand"].ravel()[(~city["sea"]).ravel()],
                       len(score) // int((~city["sea"]).sum()))
        rows.append(M.summarise(hand, obs, name=f"HAND proxy +{lead}h"))
        if abl is not None:
            sc_ab, _, _ = full_raster_eval(abl, no_route, city, router,
                                           static, te, li)
            rows.append(M.summarise(sc_ab, obs, name=f"no-routing +{lead}h"))

        for r in rows:
            print("   " + M.fmt_row(r))
        results[f"lead_{lead}h"] = rows
        models[lead] = {"clf": clf, "reg": reg}

        imp = clf.get_booster().get_score(importance_type="gain")
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:8]
        named = [(FEATURE_NAMES[int(k[1:])], v) for k, v in top]
        print("   top features by gain: "
              + ", ".join(f"{n}({v:.0f})" for n, v in named))

    with open(MODEL_DIR / "surrogate.pkl", "wb") as fh:
        pickle.dump({"models": models, "feature_names": FEATURE_NAMES,
                     "leads": LEADS, "threshold": FLOOD_THRESHOLD_M,
                     "test_ids": te}, fh)
    (OUT / "eval_results.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8")
    print(f"\nsaved models -> {MODEL_DIR/'surrogate.pkl'}")
    print(f"total {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
