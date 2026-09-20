"""Procedural terrain for Meridian City.

Builds a low-relief Atlantic coastal plain: near-flat near the shore, rising
to inland hills, with a coastal dune barrier that impedes drainage to the sea.
Low relief is the point -- flat coastal cities flood because water has nowhere
to go, not because it falls from a height.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

from ..config import Config


def fractal_noise(n: int, octaves: int, persistence: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Value-noise fBm in [0,1]: summed octaves of smoothly upsampled noise."""
    total = np.zeros((n, n), dtype=np.float64)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        res = max(2, 2 ** (o + 1))
        if res > n:
            break
        coarse = rng.random((res, res))
        layer = zoom(coarse, n / res, order=3)[:n, :n]
        if layer.shape != (n, n):                      # zoom rounding guard
            pad = [(0, n - layer.shape[0]), (0, n - layer.shape[1])]
            layer = np.pad(layer, pad, mode="edge")
        total += amp * layer
        norm += amp
        amp *= persistence
    total /= norm
    return (total - total.min()) / (np.ptp(total) + 1e-12)


def _coast_profile(xn: np.ndarray, plain_frac: float,
                   plain_top: float, max_elev: float) -> np.ndarray:
    """Cross-shore elevation profile: flat plain, then rising hinterland."""
    out = np.empty_like(xn)
    inner = xn < plain_frac
    t_in = np.clip(xn / plain_frac, 0, 1)
    out[inner] = plain_top * t_in[inner] ** 1.25
    t_out = np.clip((xn - plain_frac) / (1.0 - plain_frac), 0, 1)
    out[~inner] = plain_top + (max_elev - plain_top) * t_out[~inner] ** 1.55
    return out


def generate_terrain(cfg: Config) -> dict:
    """Return dict with dem (m), sea mask, and the shoreline column index."""
    g, t = cfg.grid, cfg.terrain
    n = g.n
    rng = np.random.default_rng(t.seed)

    yy, xx = np.mgrid[0:n, 0:n]

    # --- irregular shoreline: wobble the coast column with low-freq noise ---
    coast_wobble = gaussian_filter(rng.random(n), sigma=9.0)
    coast_wobble = (coast_wobble - coast_wobble.mean()) / (coast_wobble.std() + 1e-9)
    coast_col = 10.0 + coast_wobble * 5.5          # per-row shoreline position
    coast_col = np.clip(coast_col, 3, 26)

    # distance inland from the local shoreline, normalised 0..1
    dist_inland = (xx - coast_col[:, None]) / (n - coast_col[:, None])
    xn = np.clip(dist_inland, 0.0, 1.0)

    plain_top = 0.115 * t.max_elev_m
    base = _coast_profile(xn, t.plain_fraction, plain_top, t.max_elev_m)

    # --- fractal relief, scaled up inland (coast stays flat) ---
    relief = fractal_noise(n, t.octaves, t.roughness, rng)
    relief = (relief - 0.5) * 2.0                                   # [-1,1]
    relief_amp = 3.0 + 46.0 * xn ** 1.4                             # metres
    dem = base + relief * relief_amp

    # --- coastal dune barrier, cut by river mouths ---
    # A continuous dune would dam the whole plain; real dune-barred coasts are
    # breached at river mouths and tidal inlets. Those breaches set where the
    # drainage network reaches the sea, so they define the catchment structure.
    dune_centre = coast_col[:, None] + t.dune_width_cells
    dune = t.dune_ridge_m * np.exp(-((xx - dune_centre) ** 2) /
                                   (2.0 * (t.dune_width_cells * 0.62) ** 2))
    dune *= 0.55 + 0.45 * gaussian_filter(rng.random((n, n)), 7.0) / 0.5

    breach_rows = (np.array([0.13, 0.36, 0.58, 0.81]) * n
                   + rng.integers(-9, 10, size=4))
    breach_mask = np.ones((n, n))
    for r, sig in zip(breach_rows, rng.uniform(3.5, 6.5, size=len(breach_rows))):
        breach_mask *= 1.0 - np.exp(-((yy - r) ** 2) / (2.0 * sig ** 2))
    dune *= breach_mask
    dem += np.clip(dune, 0, None)

    # --- ocean: everything seaward of the shoreline ---
    sea = xx < coast_col[:, None]
    dem[sea] = -2.5 - 1.2 * (coast_col[:, None] - xx)[sea]

    dem = gaussian_filter(dem, 0.8)
    dem[sea] = np.minimum(dem[sea], -0.5)

    return {
        "dem": dem.astype(np.float32),
        "sea": sea,
        "coast_col": coast_col.astype(np.float32),
        "dist_inland": xn.astype(np.float32),
        "breach_rows": np.asarray(breach_rows, dtype=int),
    }


def erode_landscape(dem: np.ndarray, sea: np.ndarray, cell_size: float,
                    n_iter: int = 80, K: float = 6.0e-3, m_exp: float = 0.5,
                    n_exp: float = 1.0, diffusion: float = 0.02,
                    uplift_m: float = 0.22, target_max_m: float | None = None,
                    verbose: bool = False
                    ) -> np.ndarray:
    """Stream-power landscape evolution -> realistic dendritic valley networks.

    Detachment-limited incision (Howard 1994):   dz/dt = U - K * A^m * S^n
    plus linear hillslope diffusion. Channels (large drainage area A) incise
    far faster than hillslopes, which is what produces branching valley
    networks instead of the parallel rills a smooth regional gradient gives.
    """
    from . import hydrology as _H

    z = dem.astype(np.float64).copy()
    cell_area = cell_size ** 2
    land = ~sea
    # uplift tapers to zero at the coast so the plain stays low and flat
    ny_, nx_ = z.shape
    _, xx = np.mgrid[0:ny_, 0:nx_]
    uplift = uplift_m * np.clip(xx / nx_, 0, 1) ** 1.3

    for it in range(n_iter):
        filled = _H.fill_depressions(z.astype(np.float32), sea)
        fdir = _H.d8_flow_direction(filled, sea, cell_size)
        acc = _H.flow_accumulation(filled, fdir, sea)
        slope = _H.downstream_slope(filled, fdir, cell_size)

        A = np.maximum(acc, 1.0) * cell_area
        incision = K * (A ** m_exp) * (slope ** n_exp)

        z = filled.astype(np.float64)
        z[land] -= incision[land]
        z[land] += uplift[land]

        # hillslope diffusion. NOTE: explicit 5-point Laplacian is only stable
        # for coefficient <= 0.25; above that it grows a checkerboard mode that
        # blows the surface up and forces all flow onto diagonals.
        zp = np.pad(z, 1, mode="edge")          # edge-replicate, NOT wrap
        lap = (zp[:-2, 1:-1] + zp[2:, 1:-1] +
               zp[1:-1, :-2] + zp[1:-1, 2:] - 4.0 * z)
        z[land] += diffusion * lap[land]

        z[land] = np.maximum(z[land], 0.15)      # never erode below sea level
        z[sea] = dem[sea]
        if verbose and (it + 1) % 10 == 0:
            print(f"    erosion iter {it+1:3d}/{n_iter}  "
                  f"relief {z[land].min():.1f}..{z[land].max():.1f} m")

    if target_max_m is not None:                # renormalise relief
        hi = float(np.percentile(z[land], 99.5))
        z[land] *= target_max_m / max(hi, 1e-6)
        z[land] = np.clip(z[land], 0.15, None)
    return z.astype(np.float32)
