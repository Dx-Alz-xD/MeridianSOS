"""Hydrological conditioning and terrain derivatives.

Implements the standard DEM-hydrology chain used in flood-susceptibility work:
  priority-flood depression filling -> D8 flow direction -> flow accumulation
  -> channel extraction -> HAND -> TWI

HAND (Height Above Nearest Drainage) is consistently the single strongest
predictor of inundation in the literature, so it is computed exactly rather
than approximated.
"""
from __future__ import annotations
import heapq
import numpy as np

SQRT2 = float(np.sqrt(2.0))
# 8 neighbours: (dy, dx, distance_multiplier)
D8 = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2),
      (0, -1, 1.0),                  (0, 1, 1.0),
      (1, -1, SQRT2),  (1, 0, 1.0),  (1, 1, SQRT2)]


def fill_depressions(dem: np.ndarray, sea: np.ndarray,
                     eps: float = 1e-3) -> np.ndarray:
    """Priority-flood (Barnes et al. 2014) seeded from the ocean.

    Guarantees every land cell has a strictly descending path to the sea,
    which is what makes D8 routing and HAND well defined.
    """
    n, m = dem.shape
    filled = dem.copy().astype(np.float64)
    closed = np.zeros((n, m), dtype=bool)
    heap: list = []

    ys, xs = np.nonzero(sea)
    for y, x in zip(ys, xs):
        closed[y, x] = True
        heapq.heappush(heap, (filled[y, x], int(y), int(x)))
    # NOTE: grid borders are deliberately NOT seeded. The ocean is the only
    # outlet, so every land cell is forced to find a descending path to the
    # sea instead of leaking off the domain edge.

    while heap:
        e, y, x = heapq.heappop(heap)
        for dy, dx, _ in D8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < m and not closed[ny, nx]:
                closed[ny, nx] = True
                ne = max(filled[ny, nx], e + eps)
                filled[ny, nx] = ne
                heapq.heappush(heap, (ne, ny, nx))
    return filled.astype(np.float32)


def d8_flow_direction(dem: np.ndarray, sea: np.ndarray,
                      cell_size: float) -> np.ndarray:
    """Index 0..7 into D8 of the steepest-descent neighbour; -1 = sink/sea."""
    n, m = dem.shape
    fdir = np.full((n, m), -1, dtype=np.int8)
    d = dem.astype(np.float64)
    best = np.zeros((n, m), dtype=np.float64)

    for k, (dy, dx, dist) in enumerate(D8):
        shifted = np.full((n, m), np.inf)
        ys = slice(max(0, -dy), n - max(0, dy))
        xs = slice(max(0, -dx), m - max(0, dx))
        yd = slice(max(0, dy), n - max(0, -dy))
        xd = slice(max(0, dx), m - max(0, -dx))
        shifted[ys, xs] = d[yd, xd]
        drop = (d - shifted) / (dist * cell_size)
        better = (drop > best) & np.isfinite(drop)
        best[better] = drop[better]
        fdir[better] = k

    fdir[sea] = -1
    return fdir


def flow_accumulation(dem: np.ndarray, fdir: np.ndarray,
                      sea: np.ndarray) -> np.ndarray:
    """Number of upslope cells draining through each cell (D8, single-flow)."""
    n, m = dem.shape
    acc = np.ones((n, m), dtype=np.float64)
    acc[sea] = 0.0
    order = np.argsort(dem.ravel())[::-1]          # high -> low
    for idx in order:
        y, x = divmod(int(idx), m)
        k = fdir[y, x]
        if k < 0:
            continue
        dy, dx, _ = D8[k]
        ny, nx = y + dy, x + dx
        if 0 <= ny < n and 0 <= nx < m:
            acc[ny, nx] += acc[y, x]
    return acc.astype(np.float32)


def trace_downstream(start: tuple, fdir: np.ndarray, max_steps: int = 100000):
    """Yield cells along the D8 flow path from `start` until a sink."""
    n, m = fdir.shape
    y, x = start
    for _ in range(max_steps):
        yield (y, x)
        k = fdir[y, x]
        if k < 0:
            return
        dy, dx, _ = D8[k]
        y, x = y + dy, x + dx
        if not (0 <= y < n and 0 <= x < m):
            return


def trace_upstream_maxacc(start: tuple, fdir: np.ndarray, acc: np.ndarray,
                          min_acc: float = 1.0) -> list:
    """Walk upstream from `start`, always taking the highest-accumulation
    contributor. Returns the main-stem cell list (downstream -> upstream)."""
    n, m = fdir.shape
    path = [start]
    y, x = start
    while True:
        best, bk = None, -1.0
        for k, (dy, dx, _) in enumerate(D8):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < n and 0 <= nx < m):
                continue
            if fdir[ny, nx] < 0:
                continue
            # does (ny,nx) drain into (y,x)?
            ddy, ddx, _ = D8[fdir[ny, nx]]
            if ny + ddy == y and nx + ddx == x and acc[ny, nx] > bk:
                bk, best = acc[ny, nx], (ny, nx)
        if best is None or bk < min_acc:
            return path
        path.append(best)
        y, x = best


def compute_hand(dem: np.ndarray, fdir: np.ndarray, channels: np.ndarray,
                 sea: np.ndarray) -> np.ndarray:
    """Height Above Nearest Drainage, following D8 flow paths.

    Processed in ascending elevation so each cell's downstream neighbour
    (strictly lower after depression filling) is already resolved.
    """
    n, m = dem.shape
    ref = np.full((n, m), np.nan, dtype=np.float64)
    ref[sea] = 0.0
    ref[channels] = dem[channels]

    order = np.argsort(dem.ravel())                 # low -> high
    for idx in order:
        y, x = divmod(int(idx), m)
        if not np.isnan(ref[y, x]):
            continue
        k = fdir[y, x]
        if k < 0:
            ref[y, x] = dem[y, x]
            continue
        dy, dx, _ = D8[k]
        ny, nx = y + dy, x + dx
        if 0 <= ny < n and 0 <= nx < m and not np.isnan(ref[ny, nx]):
            ref[y, x] = ref[ny, nx]
        else:
            ref[y, x] = dem[y, x]

    hand = np.clip(dem - ref, 0.0, None)
    hand[sea] = 0.0
    return hand.astype(np.float32)


def slope_pct(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Slope as rise/run (not degrees), Horn's method."""
    gy, gx = np.gradient(dem.astype(np.float64), cell_size)
    return np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)


def twi(acc: np.ndarray, slope: np.ndarray, cell_size: float) -> np.ndarray:
    """Topographic Wetness Index = ln(a / tan(beta))."""
    a = (acc + 1.0) * cell_size                     # upslope area per unit width
    return np.log(a / np.maximum(slope, 1e-3)).astype(np.float32)


def curvature(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Laplacian curvature; negative = concave (water collects)."""
    d = dem.astype(np.float64)
    lap = (np.roll(d, 1, 0) + np.roll(d, -1, 0) +
           np.roll(d, 1, 1) + np.roll(d, -1, 1) - 4.0 * d) / (cell_size ** 2)
    return lap.astype(np.float32)


def distance_to(mask: np.ndarray, cell_size: float) -> np.ndarray:
    """Euclidean distance (metres) to the nearest True cell in `mask`."""
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return np.full(mask.shape, 1e5, dtype=np.float32)
    return (distance_transform_edt(~mask) * cell_size).astype(np.float32)


def downstream_slope(dem: np.ndarray, fdir: np.ndarray,
                     cell_size: float) -> np.ndarray:
    """Slope along the D8 flow path (rise/run). Sinks get 0."""
    n, m = dem.shape
    out = np.zeros((n, m), dtype=np.float64)
    d = dem.astype(np.float64)
    for k, (dy, dx, dist) in enumerate(D8):
        sel = fdir == k
        if not sel.any():
            continue
        ys, xs = np.nonzero(sel)
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < n) & (nx >= 0) & (nx < m)
        ys, xs, ny, nx = ys[ok], xs[ok], ny[ok], nx[ok]
        out[ys, xs] = (d[ys, xs] - d[ny, nx]) / (dist * cell_size)
    return np.clip(out, 0.0, None)
