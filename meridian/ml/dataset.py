"""Turn simulated events into a supervised learning table.

Sampling and splitting decisions that actually matter:

* Split by EVENT, never by cell. Cells within one event are strongly
  spatially correlated, so a random cell split leaks neighbours between
  train and test and inflates every score.

* Cells are subsampled for TRAINING only (keeping all wet cells plus a
  ratio of dry ones, because <2% of cells flood). Evaluation always runs
  on the full raster of held-out events, so the reported scores are on the
  real class balance rather than the resampled one.

* The model is given the FORECAST rainfall, never the observed field, so
  the forecast error it must tolerate is baked into training.
"""
from __future__ import annotations
import json
import numpy as np

from ..config import EVENT_DIR
from .features import Router, static_features, build_features, FEATURE_NAMES

FLOOD_THRESHOLD_M = 0.10        # operational "this street is impassable"
LEADS = (1, 3, 6)               # hours ahead


def load_index() -> list:
    """Event metadata. Falls back to scanning the directory so a partially
    complete simulation run is still usable for pipeline testing."""
    p = EVENT_DIR / "index.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    out = []
    for f in sorted(EVENT_DIR.glob("event_*.npz")):
        try:
            m = dict(np.load(f, allow_pickle=True)["meta"][0])
            out.append(m)
        except Exception:
            continue          # file still being written
    return out


def load_event(i: int) -> dict:
    z = np.load(EVENT_DIR / f"event_{i:04d}.npz", allow_pickle=True)
    return {
        "rain_obs_h": z["rain_obs_h"].astype(np.float32),
        "rain_fc_h": z["rain_fc_h"].astype(np.float32),
        "depth_h": z["depth_h"].astype(np.float32),
        "h_max": z["h_max"].astype(np.float32),
        "meta": z["meta"][0],
    }


def issue_times(nh: int, max_lead: int = max(LEADS), first: int = 2) -> list:
    """Hours at which a forecast could be issued with all leads available."""
    return list(range(first, nh - max_lead))


def event_frames(city, router, static, ev, t0):
    """Full-raster features [F, n, n] and targets [L, n, n] at issue time t0."""
    X = build_features(city, router, static, ev["rain_fc_h"],
                       ev["rain_obs_h"], ev["depth_h"], t0)
    nh = ev["depth_h"].shape[0]
    Y = np.stack([ev["depth_h"][min(t0 + L, nh - 1)] for L in LEADS])
    return X, Y


def sample_cells(Y, land, rng, neg_ratio=3.0, min_neg=1500, max_pos=20000):
    """Indices of cells to train on: all wet, plus a ratio of dry ones."""
    wet = (Y > FLOOD_THRESHOLD_M).any(axis=0) & land
    dry = land & ~wet
    wi = np.flatnonzero(wet.ravel())
    di = np.flatnonzero(dry.ravel())
    if len(wi) > max_pos:
        wi = rng.choice(wi, max_pos, replace=False)
    n_neg = int(max(min_neg, neg_ratio * len(wi)))
    n_neg = min(n_neg, len(di))
    di = rng.choice(di, n_neg, replace=False)
    return np.concatenate([wi, di])


def build_table(event_ids, city, router=None, static=None, rng=None,
                stride: int = 2, neg_ratio: float = 3.0,
                keep_dry_events: float = 0.25, verbose: bool = True):
    """Assemble (X, Y, event_id, cell_idx) across many events."""
    rng = rng or np.random.default_rng(0)
    router = router or Router(city)
    static = static_features(city) if static is None else static
    land = ~city["sea"]

    Xs, Ys, eids, cids = [], [], [], []
    for k, eid in enumerate(event_ids):
        ev = load_event(eid)
        nh = ev["depth_h"].shape[0]
        wet_event = float(ev["h_max"].max()) > FLOOD_THRESHOLD_M
        if not wet_event and rng.random() > keep_dry_events:
            continue                      # keep a few dry events for calibration
        for t0 in issue_times(nh)[::stride]:
            X, Y = event_frames(city, router, static, ev, t0)
            idx = sample_cells(Y, land, rng, neg_ratio=neg_ratio)
            Xs.append(X.reshape(X.shape[0], -1)[:, idx].T)
            Ys.append(Y.reshape(Y.shape[0], -1)[:, idx].T)
            eids.append(np.full(len(idx), eid, dtype=np.int32))
            cids.append(idx.astype(np.int32))
        if verbose and (k + 1) % 10 == 0:
            print(f"    tabulated {k+1}/{len(event_ids)} events, "
                  f"{sum(len(a) for a in Xs):,} rows", flush=True)

    if not Xs:
        f, l = len(FEATURE_NAMES), len(LEADS)
        return (np.zeros((0, f), np.float32), np.zeros((0, l), np.float32),
                np.zeros(0, np.int32), np.zeros(0, np.int32))
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    return X, Y, np.concatenate(eids), np.concatenate(cids)


def split_events(index, frac=(0.70, 0.15, 0.15), seed=7):
    """Stratify the train/val/test event split by severity."""
    rng = np.random.default_rng(seed)
    by_sev = {}
    for m in index:
        by_sev.setdefault(m["severity"], []).append(m["id"])
    tr, va, te = [], [], []
    for sev, ids in by_sev.items():
        ids = np.array(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n >= 3:
            # guarantee val and test are non-empty for every severity, so a
            # rare class cannot silently vanish from the held-out sets
            a = max(1, min(n - 2, int(round(frac[0] * n))))
            b = max(a + 1, min(n - 1, a + int(round(frac[1] * n))))
        else:
            a, b = n, n
        tr += list(ids[:a]); va += list(ids[a:b]); te += list(ids[b:])
    return sorted(tr), sorted(va), sorted(te)


__all__ = ["FEATURE_NAMES", "FLOOD_THRESHOLD_M", "LEADS", "load_index",
           "load_event", "build_table", "split_events", "event_frames",
           "issue_times", "Router", "static_features"]
