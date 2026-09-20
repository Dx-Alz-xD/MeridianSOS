"""Feature construction for the flood surrogate.

The single most important design choice here is catchment routing. What
floods a cell is not the rain that falls on it -- it is the rain that falls
anywhere upstream of it and arrives later. A model given only local rainfall
has to infer the drainage network from terrain proxies; a model given
upstream-accumulated rainfall gets the physics for free and needs far less
data to learn the rest.

Feature groups
  static    terrain, hydrology and urban form (fixed for the city)
  dynamic   rainfall forecast and recent history, both local and routed
            upstream, plus the current observed water state
  global    citywide rainfall rate, broadcast to every cell -- this is what
            lets the model see trunk-sewer saturation, which is a
            system-level threshold no local feature can express
"""
from __future__ import annotations
import numpy as np

from ..sim.flood import downstream_index

# feature names in the exact order build_features returns them
STATIC_NAMES = [
    "elevation", "hand", "slope", "twi", "curvature",
    "log_acc", "dist_channel", "dist_coast", "dist_ghost", "dist_inland",
    "imperviousness", "curve_number", "drain_capacity", "manning",
    "chan_depth", "landuse",
]
DYNAMIC_NAMES = [
    "rain_fc_1h", "rain_fc_3h", "rain_fc_6h",
    "rain_past_1h", "rain_past_3h", "rain_past_6h", "rain_cum_event",
    "up_fc_1h", "up_fc_3h", "up_fc_6h", "up_past_3h", "up_cum_event",
    "depth_now", "up_depth_now",
]
GLOBAL_NAMES = ["city_rain_now", "city_rain_fc_6h", "city_depth_mean", "hours_elapsed"]
FEATURE_NAMES = STATIC_NAMES + DYNAMIC_NAMES + GLOBAL_NAMES


class Router:
    """Precomputed D8 topology for fast repeated upstream accumulation."""

    def __init__(self, city: dict):
        self.shape = city["dem"].shape
        self.n = self.shape[0]
        self.sea = city["sea"]
        ds = downstream_index(city["fdir"])
        self.ds = ds
        self.valid = ds >= 0
        self.target = ds[self.valid]
        # process cells from high to low so a cell is always resolved before
        # the cell it drains into
        self.order = np.argsort(city["filled"].ravel())[::-1]
        self.order = self.order[~self.sea.ravel()[self.order]]

    def accumulate(self, fields: np.ndarray) -> np.ndarray:
        """Route [K, n, n] fields down the flow network. Returns [K, n, n].

        Each cell ends up holding its own value plus everything upstream.
        """
        single = fields.ndim == 2
        if single:
            fields = fields[None]
        k = fields.shape[0]
        acc = fields.reshape(k, -1).astype(np.float32).copy()
        acc[:, self.sea.ravel()] = 0.0
        ds = self.ds
        for idx in self.order:
            j = ds[idx]
            if j >= 0:
                acc[:, j] += acc[:, idx]
        out = acc.reshape((k,) + self.shape)
        return out[0] if single else out


def static_features(city: dict) -> np.ndarray:
    """[S, n, n] stack of time-invariant features."""
    eps = 1.0
    stack = [
        city["dem"],
        np.minimum(city["hand"], 60.0),
        city["slope"],
        city["twi"],
        np.clip(city["curvature"], -0.02, 0.02),
        np.log10(city["acc"] + eps),
        np.minimum(city["dist_channel"], 5000.0),
        np.minimum(city["dist_coast"], 25000.0),
        np.minimum(city["dist_ghost"], 8000.0),
        city["dist_inland"],
        city["imperv"],
        city["cn"],
        city["drain_mm_h"],
        city["manning"],
        city["chan_d"],
        city["landuse"].astype(np.float32),
    ]
    return np.stack([np.asarray(a, dtype=np.float32) for a in stack])


def _window_mean(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of arr over hour indices [lo, hi), clipped to bounds."""
    lo = max(lo, 0)
    hi = min(hi, arr.shape[0])
    if hi <= lo:
        return np.zeros(arr.shape[1:], dtype=np.float32)
    return arr[lo:hi].mean(0).astype(np.float32)


def dynamic_features(router: Router, rain_fc_h: np.ndarray,
                     rain_obs_h: np.ndarray, depth_h: np.ndarray,
                     t0: int) -> np.ndarray:
    """[D, n, n] stack of features available at issue time t0 (hours)."""
    fc1 = _window_mean(rain_fc_h, t0, t0 + 1)
    fc3 = _window_mean(rain_fc_h, t0, t0 + 3)
    fc6 = _window_mean(rain_fc_h, t0, t0 + 6)
    p1 = _window_mean(rain_obs_h, t0 - 1, t0)
    p3 = _window_mean(rain_obs_h, t0 - 3, t0)
    p6 = _window_mean(rain_obs_h, t0 - 6, t0)
    cum = rain_obs_h[:max(t0, 1)].sum(0).astype(np.float32)
    now = depth_h[min(t0, depth_h.shape[0] - 1)].astype(np.float32)

    # Route through the drainage network and keep the TOTAL, not the mean.
    # The catchment mean of a spatially smooth rain field is almost exactly
    # the local value, so it carries no extra information. The total is
    # proportional to discharge, which is what actually floods a cell, and it
    # scales with catchment area -- so it separates a big river from a gutter
    # under identical rainfall. Log-compressed: upstream area spans 1 to
    # ~55,000 cells across this domain.
    routed = np.log1p(router.accumulate(np.stack([fc1, fc3, fc6, p3, cum, now])))

    return np.stack([fc1, fc3, fc6, p1, p3, p6, cum,
                     routed[0], routed[1], routed[2], routed[3],
                     routed[4], now, routed[5]]).astype(np.float32)


def global_features(rain_fc_h, rain_obs_h, depth_h, t0, land) -> np.ndarray:
    """[G, n, n] scalars broadcast over the grid."""
    shape = land.shape
    city_now = float(_window_mean(rain_obs_h, t0 - 1, t0)[land].mean())
    city_fc6 = float(_window_mean(rain_fc_h, t0, t0 + 6)[land].mean())
    d = depth_h[min(t0, depth_h.shape[0] - 1)]
    city_depth = float(d[land].mean())
    vals = [city_now, city_fc6, city_depth, float(t0)]
    return np.stack([np.full(shape, v, dtype=np.float32) for v in vals])


def build_features(city: dict, router: Router, static: np.ndarray,
                   rain_fc_h, rain_obs_h, depth_h, t0: int) -> np.ndarray:
    """Full [F, n, n] feature stack at issue time t0."""
    land = ~city["sea"]
    dyn = dynamic_features(router, rain_fc_h, rain_obs_h, depth_h, t0)
    glo = global_features(rain_fc_h, rain_obs_h, depth_h, t0, land)
    return np.concatenate([static, dyn, glo], axis=0)
