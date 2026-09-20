"""Build the complete Meridian City state and cache it to disk.

Pipeline:
    fractal terrain -> stream-power erosion -> hydrology -> bury the historical
    watercourse -> re-run hydrology on the modern surface -> urban form

The buried watercourse ("the Ghost Channel") is the interesting part. A
watercourse that ran through what is now the city centre was culverted during
20th-century expansion. It is invisible in the modern DEM -- the surface was
levelled over it -- but it remains the preferential flow path, and its culvert
has a finite capacity. When that capacity is exceeded the channel surcharges
and floods a corridor with no surface watercourse on any map.

This gives the ML model something genuinely non-obvious to learn: a
flood-prone corridor that terrain alone does not explain.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.ndimage import binary_dilation, grey_dilation

from ..config import Config, CFG, CITY_DIR
from . import hydrology as H
from .terrain import generate_terrain, erode_landscape
from .landuse import generate_landuse


def _run_hydrology(dem, sea, cfg):
    cs = cfg.grid.cell_size_m
    filled = H.fill_depressions(dem, sea)
    fdir = H.d8_flow_direction(filled, sea, cs)
    acc = H.flow_accumulation(filled, fdir, sea)
    ch = (acc > cfg.hydro.channel_accum_threshold) & (~sea)
    hand = H.compute_hand(filled, fdir, ch, sea)
    return filled, fdir, acc, ch, hand


def _find_ghost_channel(acc, ch, fdir, dist_inland, cbd_yx, plain_frac=0.30):
    """The historical watercourse: the stem passing nearest the modern CBD."""
    from scipy.ndimage import binary_dilation as _bd
    cy, cx = cbd_yx
    cand = np.nonzero(ch & (dist_inland < plain_frac))
    if not len(cand[0]):
        return np.zeros_like(ch)
    d = (cand[0] - cy) ** 2 + (cand[1] - cx) ** 2
    # among the 40 channel cells nearest the CBD, start from the largest stream
    near = np.argsort(d)[:40]
    a = acc[cand[0][near], cand[1][near]]
    start = (int(cand[0][near][np.argmax(a)]), int(cand[1][near][np.argmax(a)]))

    path = set(H.trace_upstream_maxacc(start, fdir, acc, min_acc=40))
    path |= set(H.trace_downstream(start, fdir))
    ghost = np.zeros_like(ch)
    for (y, x) in path:
        ghost[y, x] = True
    # confine to the built-up plain: a culverted urban watercourse does not
    # run out into the hinterland hills
    return ghost & (dist_inland < plain_frac)


def build_city(cfg: Config = CFG, seed: int | None = None,
               verbose: bool = True) -> dict:
    t_start = time.time()
    if seed is not None:
        cfg.terrain.seed = seed
    cs = cfg.grid.cell_size_m
    rng = np.random.default_rng(cfg.terrain.seed + 101)
    log = (lambda m: print(m)) if verbose else (lambda m: None)

    log("  [1/6] fractal terrain")
    T = generate_terrain(cfg)
    dem, sea = T["dem"], T["sea"]
    land = ~sea

    log(f"  [2/6] stream-power erosion ({cfg.terrain.erosion_iters} iters)")
    dem = erode_landscape(dem, sea, cs, n_iter=cfg.terrain.erosion_iters,
                          target_max_m=cfg.terrain.max_elev_m)

    log("  [3/6] hydrology (natural surface)")
    filled, fdir, acc, ch, hand = _run_hydrology(dem, sea, cfg)

    log("  [4/6] burying the historical watercourse")
    cbd_yx = (cfg.city.cbd_xy[1] * cfg.grid.n, cfg.city.cbd_xy[0] * cfg.grid.n)
    ghost = np.zeros_like(ch)
    if cfg.hydro.ghost_channel_enabled:
        ghost = _find_ghost_channel(acc, ch, fdir, T["dist_inland"], cbd_yx)
        if ghost.any():
            # Raise the old channel bed almost to the surrounding grade,
            # leaving the shallow depression a levelled-over valley really
            # keeps. Water still collects here; what it no longer has is an
            # open watercourse to carry it away -- only a fixed-capacity pipe.
            surround = grey_dilation(dem, size=7)
            dem = dem.copy()
            dem[ghost] = surround[ghost] - cfg.hydro.ghost_burial_depth_m
            log(f"        culverted {int(ghost.sum())} cells "
                f"(capacity {cfg.hydro.ghost_channel_capacity_m3s:.0f} m3/s, "
                f"{cfg.hydro.ghost_burial_depth_m:.1f} m below grade)")

    log("  [5/6] hydrology (modern built surface)")
    filled, fdir, acc, ch, hand = _run_hydrology(dem, sea, cfg)
    # The culverted reach is NOT an open channel: it gets no bankfull
    # conveyance, only the pipe. This is what makes it fail catastrophically
    # once the pipe is full, while looking unremarkable on a terrain map.
    if ghost.any():
        ch = ch & ~ghost
    slope = H.slope_pct(filled, cs)
    twi = H.twi(acc, slope, cs)
    curv = H.curvature(filled, cs)
    d_chan = H.distance_to(ch, cs)
    d_coast = H.distance_to(sea, cs)
    d_ghost = H.distance_to(ghost, cs) if ghost.any() else np.full_like(d_chan, 1e5)

    # --- sub-grid channel geometry -----------------------------------------
    # A 100 m cell cannot resolve a 15 m-wide watercourse. Treating the whole
    # cell as channel gives it ~7x too much conveyance, so water drains off
    # the floodplain instead of overtopping onto it. Downstream hydraulic
    # geometry (width and depth as power laws of upstream area) gives each
    # channel cell a realistic bankfull capacity instead.
    area_km2 = acc * (cs * cs) / 1e6
    chan_w = np.clip(2.0 * np.power(np.maximum(area_km2, 1e-3), 0.45), 2.0, 45.0)
    chan_d = np.clip(0.27 * np.power(np.maximum(area_km2, 1e-3), 0.30), 0.35, 4.0)
    chan_w = np.where(ch, chan_w, 0.0).astype(np.float32)
    chan_d = np.where(ch, chan_d, 0.0).astype(np.float32)
    # in-bank storage volume per cell, and bankfull discharge by Manning
    chan_store = (chan_w * cs * chan_d).astype(np.float32)          # m3
    s_bed = np.maximum(H.downstream_slope(filled, fdir, cs), 1e-4)
    chan_qbf = ((1.0 / 0.035) * (chan_w * chan_d)
                * np.power(np.maximum(chan_d, 1e-3), 2.0 / 3.0)
                * np.sqrt(s_bed)).astype(np.float32)                # m3/s
    chan_qbf = np.where(ch, chan_qbf, 0.0).astype(np.float32)
    log(f"        channel width {chan_w[ch].min():.0f}-{chan_w[ch].max():.0f} m, "
        f"bankfull {chan_qbf[ch].max():.0f} m3/s at the outlet")

    log("  [6/6] urban form, population, districts")
    L = generate_landuse(cfg, dem, sea, hand, acc, ch, rng)

    city = {
        "dem": dem, "filled": filled, "sea": sea, "fdir": fdir,
        "acc": acc, "channels": ch, "hand": hand, "slope": slope,
        "twi": twi, "curvature": curv, "dist_channel": d_chan,
        "dist_coast": d_coast, "dist_ghost": d_ghost, "ghost": ghost,
        "chan_w": chan_w, "chan_d": chan_d, "chan_store": chan_store,
        "chan_qbf": chan_qbf,
        "dist_inland": T["dist_inland"], "coast_col": T["coast_col"],
        "breach_rows": T["breach_rows"],
        **L,
    }
    log(f"  built in {time.time()-t_start:.1f}s   "
        f"({land.sum():,} land cells, {L['population'].sum():,.0f} people)")
    return city


def save_city(city: dict, path=None) -> str:
    path = path or (CITY_DIR / "meridian.npz")
    names = city.pop("district_names", None)
    arrs = {k: v for k, v in city.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in city.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, **arrs,
                        _district_names=np.array(names or [], dtype=object),
                        _meta=np.array([meta], dtype=object))
    if names is not None:
        city["district_names"] = names
    return str(path)


def load_city(path=None) -> dict:
    path = path or (CITY_DIR / "meridian.npz")
    z = np.load(path, allow_pickle=True)
    city = {k: z[k] for k in z.files if not k.startswith("_")}
    city["district_names"] = list(z["_district_names"])
    city.update(z["_meta"][0])
    return city
