"""Urban form, districts and population for Meridian City.

Urban land use is grown from an accessibility-potential surface (attractors at
the CBD, port, airport and secondary centres) rather than painted by hand, so
the city has an organic gradient from core to fringe.

One structural choice matters for the whole project: informal settlements are
sited on *marginal* land -- low-lying, near watercourses, cheap precisely
because it floods. Combined with their much lower storm-drain capacity
(3 mm/h vs 26 mm/h in the CBD, see config.LANDUSE), this reproduces the real
pattern where the least-resourced districts carry the most flood risk. The
model is never told this; it emerges from the simulation.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation

from ..config import (Config, LU_CN, LU_IMPERV, LU_MANNING,
                      LU_DRAIN, LU_POP_WEIGHT, LU_FRAGILITY)

DISTRICT_NAMES = [
    "Harbour Point", "Old Port", "Saltmarsh", "The Corniche", "Beacon Hill",
    "Meridian Centre", "Exchange District", "Textile Quarter", "Rivergate",
    "Lower Ford", "Kiln Fields", "Northgate", "Eastbrook", "Highfield",
    "Sablon", "Quarry Rise", "Verdant Park", "Aqueduct", "Stonebridge",
    "Clayhill", "New Dawn", "Sunrise Camp", "Canal Side", "Foundry Row",
    "Airport Zone", "Olive Grove", "Westmead", "Palm Court", "Southbank",
    "Terrace Hill", "Marsh End", "Cedar Gate",
]


def _attractor(shape, cy, cx, sigma, kind="gauss"):
    """Distance-decay field around an urban centre.

    kind="exp" is Clark's negative-exponential density law, which is how real
    urban density actually falls off. A Gaussian decays far too fast and
    produces an implausibly compact city ringed by empty land.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    if kind == "exp":
        return np.exp(-np.sqrt(d2) / sigma)
    return np.exp(-d2 / (2.0 * sigma ** 2))


def generate_landuse(cfg: Config, dem, sea, hand, acc, channels,
                     rng: np.random.Generator) -> dict:
    n = cfg.grid.n
    land = ~sea
    c = cfg.city

    cbd = (c.cbd_xy[1] * n, c.cbd_xy[0] * n)
    port = (c.port_xy[1] * n, c.port_xy[0] * n)
    airport = (c.airport_xy[1] * n, c.airport_xy[0] * n)

    # ---------------------------------------------- accessibility potential --
    pot = np.zeros((n, n))
    pot += 1.00 * _attractor((n, n), *cbd, 0.30 * n, "exp")
    pot += 0.50 * _attractor((n, n), *port, 0.16 * n, "exp")
    pot += 0.34 * _attractor((n, n), *airport, 0.13 * n, "exp")
    for _ in range(5):                       # secondary centres
        sy, sx = rng.uniform(0.10, 0.94) * n, rng.uniform(0.08, 0.52) * n
        pot += rng.uniform(0.26, 0.46) * _attractor((n, n), sy, sx, 0.11 * n, "exp")
    # coastal corridor bonus: development hugs the shore
    _, xxg = np.mgrid[0:n, 0:n]
    pot += 0.22 * np.exp(-np.clip(xxg - 12, 0, None) / (0.34 * n))
    pot += 0.15 * gaussian_filter(rng.random((n, n)), 6.0) / 0.5
    slope_pen = np.clip(np.hypot(*np.gradient(dem)) / 6.0, 0, 1)
    pot *= (1.0 - 0.55 * slope_pen)          # steep ground resists building
    pot[sea] = -1.0
    pot = (pot - pot[land].min()) / (np.ptp(pot[land]) + 1e-9)

    # -------------------------------------------------------- classification -
    lu = np.full((n, n), 8, dtype=np.int8)   # default: bare scrub
    lu[sea] = 0
    # thresholds are quantiles of the potential over land, so the urban
    # footprint is stable regardless of how the potential surface scales
    q = lambda f: float(np.quantile(pot[land], f))
    lu[land & (pot > q(0.26))] = 7           # agriculture     ~ 24% of land
    lu[land & (pot > q(0.50))] = 3           # suburban        ~ 26%
    lu[land & (pot > q(0.76))] = 2           # dense resid.    ~ 18%
    lu[land & (pot > q(0.945))] = 1          # CBD             ~  5%

    big_river = channels & (acc > 8 * cfg.hydro.channel_accum_threshold)
    lu[big_river] = 0

    wet = land & (dem < 3.0) & (hand < 1.0) & (pot < 0.55)
    lu[wet] = 9                              # coastal wetland

    # ------------------------------------------------------------ industry ---
    ind_sites = [port]
    for _ in range(c.n_industrial - 1):
        ind_sites.append((rng.uniform(0.15, 0.9) * n, rng.uniform(0.05, 0.35) * n))
    for (iy, ix) in ind_sites:
        sel = land & (_attractor((n, n), iy, ix, rng.uniform(5.5, 9.0)) > 0.45)
        lu[sel & (lu != 0)] = 5

    # -------------------------------------------- informal settlements -------
    # Settlements form on the worst ground available inside the urban area:
    # low-lying, hard against a watercourse. Seeds are drawn with probability
    # weighted toward low HAND so they land in the true flood corridor, not
    # merely somewhere on the urban fringe.
    near_channel = binary_dilation(channels, iterations=3)
    marginal = (land & (hand < 2.2) & (pot > q(0.45)) & (pot < q(0.93))
                & (lu != 0) & (lu != 9))
    pool = marginal & near_channel
    if pool.sum() < 250:
        pool = marginal
    cy_, cx_ = np.nonzero(pool)
    informal_seeds = []
    if len(cy_):
        w = 1.0 / (hand[cy_, cx_] + 0.35)          # prefer the lowest ground
        w = w / w.sum()
        for _ in range(c.n_informal):
            j = int(rng.choice(len(cy_), p=w))
            iy, ix = int(cy_[j]), int(cx_[j])
            informal_seeds.append((iy, ix))
            sel = land & (_attractor((n, n), iy, ix, rng.uniform(4.0, 7.0)) > 0.40)
            # settlements spread along the low ground, not over the hills
            lu[sel & (lu != 0) & (hand < 6.0)] = 4

    # ------------------------------------------------------------- parks -----
    urban_y, urban_x = np.nonzero(np.isin(lu, [1, 2, 3]))
    for _ in range(22):
        if not len(urban_y):
            break
        j = int(rng.integers(len(urban_y)))
        sel = land & (_attractor((n, n), urban_y[j], urban_x[j],
                                 rng.uniform(2.4, 4.6)) > 0.42)
        lu[sel & np.isin(lu, [2, 3])] = 6

    # ------------------------------------------------------- derived layers --
    def mapcode(table):
        out = np.zeros((n, n), dtype=np.float32)
        for k, v in table.items():
            out[lu == k] = v
        return out

    cn, imperv = mapcode(LU_CN), mapcode(LU_IMPERV)
    manning, fragility = mapcode(LU_MANNING), mapcode(LU_FRAGILITY)
    drain = mapcode(LU_DRAIN)

    # maintenance declines with distance from the core
    dist_core = np.hypot(*(np.mgrid[0:n, 0:n] - np.array(cbd).reshape(2, 1, 1)))
    upkeep = np.clip(1.15 - 0.45 * (dist_core / (0.55 * n)), 0.35, 1.15)
    drain = (drain * upkeep).astype(np.float32)

    # ------------------------------------------------------------ population -
    w = mapcode(LU_POP_WEIGHT)
    w *= (0.55 + 0.9 * pot).astype(np.float32)
    w *= (0.75 + 0.5 * gaussian_filter(rng.random((n, n)), 3.0) / 0.5).astype(np.float32)
    w[sea] = 0
    pop = (w / max(w.sum(), 1e-9) * cfg.city.population).astype(np.float32)

    # ------------------------------------------------------------- districts -
    built = land & np.isin(lu, [1, 2, 3, 4, 5])
    by, bx = np.nonzero(built)
    k = min(len(DISTRICT_NAMES), 28)
    idx = rng.choice(len(by), size=k, replace=False)
    seeds = np.stack([by[idx], bx[idx]], axis=1)
    yy, xx = np.mgrid[0:n, 0:n]
    d2 = ((yy[None] - seeds[:, 0][:, None, None]) ** 2 +
          (xx[None] - seeds[:, 1][:, None, None]) ** 2)
    districts = np.argmin(d2, axis=0).astype(np.int16)
    districts[~land] = -1

    return {
        "landuse": lu, "potential": pot.astype(np.float32),
        "cn": cn, "imperv": imperv, "manning": manning,
        "drain_mm_h": drain, "fragility": fragility, "population": pop,
        "districts": districts, "district_names": DISTRICT_NAMES[:k],
        "cbd": cbd, "port": port, "airport": airport,
        "informal_seeds": informal_seeds,
    }
