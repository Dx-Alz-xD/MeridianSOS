"""Verification metrics for flood forecasts.

Deliberately uses the contingency-table scores the flood-forecasting
community actually reports (CSI, POD, FAR, frequency bias) rather than
accuracy. With under 2% of cells wet, a model that predicts "never floods"
scores 98% accuracy and is worthless -- CSI scores it 0.
"""
from __future__ import annotations
import numpy as np


def contingency(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Binary skill scores from a boolean prediction/observation pair."""
    pred = pred.astype(bool).ravel()
    obs = obs.astype(bool).ravel()
    tp = int(np.count_nonzero(pred & obs))
    fp = int(np.count_nonzero(pred & ~obs))
    fn = int(np.count_nonzero(~pred & obs))
    tn = int(np.count_nonzero(~pred & ~obs))
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "CSI": tp / max(tp + fp + fn, 1),        # critical success index
        "POD": tp / max(tp + fn, 1),             # hit rate / recall
        "FAR": fp / max(tp + fp, 1),             # false alarm ratio
        "bias": (tp + fp) / max(tp + fn, 1),     # over/under-forecasting
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def best_csi_threshold(score: np.ndarray, obs: np.ndarray,
                       grid: np.ndarray | None = None, n_grid: int = 60) -> tuple:
    """Threshold on a continuous score that maximises CSI.

    The grid is taken from quantiles of the score itself, not a fixed 0-1
    range. Baselines are not all probabilities -- a HAND-based score runs
    from -60 to 0, and a fixed [0,1] grid silently scores it CSI 0, making a
    reasonable baseline look useless.
    """
    score = np.asarray(score).ravel()
    if grid is None:
        qs = np.linspace(0.50, 0.9995, n_grid)
        grid = np.unique(np.quantile(score, qs))
    best = (float(grid[0]) if len(grid) else 0.5, -1.0)
    for t in grid:
        c = contingency(score >= t, obs)
        if c["CSI"] > best[1]:
            best = (float(t), c["CSI"])
    return best


def depth_errors(pred_m: np.ndarray, obs_m: np.ndarray,
                 wet_thresh: float = 0.10) -> dict:
    """Depth accuracy overall and restricted to genuinely wet cells."""
    p, o = np.asarray(pred_m).ravel(), np.asarray(obs_m).ravel()
    wet = o > wet_thresh
    out = {
        "rmse_all_cm": 100 * float(np.sqrt(np.mean((p - o) ** 2))),
        "mae_all_cm": 100 * float(np.mean(np.abs(p - o))),
    }
    if wet.any():
        out["rmse_wet_cm"] = 100 * float(np.sqrt(np.mean((p[wet] - o[wet]) ** 2)))
        out["mae_wet_cm"] = 100 * float(np.mean(np.abs(p[wet] - o[wet])))
        out["bias_wet_cm"] = 100 * float(np.mean(p[wet] - o[wet]))
    return out


def roc_auc(score: np.ndarray, obs: np.ndarray) -> float:
    """AUC via the rank identity; returns nan if one class is absent."""
    s = np.asarray(score).ravel()
    y = np.asarray(obs).astype(bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(score: np.ndarray, obs: np.ndarray) -> float:
    """Average precision. More informative than ROC-AUC under heavy imbalance."""
    s = np.asarray(score).ravel()
    y = np.asarray(obs).astype(bool).ravel()
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def summarise(score, obs_depth, pred_depth=None, thresh=0.10, name=""):
    """One-line evaluation bundle for a lead time."""
    obs = obs_depth > thresh
    out = {"name": name, "n": int(obs.size), "n_wet": int(obs.sum()),
           "base_rate": float(obs.mean())}
    out["roc_auc"] = roc_auc(score, obs)
    out["pr_auc"] = pr_auc(score, obs)
    t, csi = best_csi_threshold(np.asarray(score), obs)
    out["best_thresh"] = t
    out.update(contingency(np.asarray(score) >= t, obs))
    if pred_depth is not None:
        out.update(depth_errors(pred_depth, obs_depth, thresh))
    return out


def fmt_row(r: dict) -> str:
    return (f"{r['name']:<22} CSI {r['CSI']:.3f}  POD {r['POD']:.3f}  "
            f"FAR {r['FAR']:.3f}  bias {r['bias']:.2f}  "
            f"AUC {r['roc_auc']:.4f}  PR-AUC {r['pr_auc']:.3f}")
