"""From water depth to human impact.

A depth map is not a warning. This module turns predicted depth into the
three things an emergency operations centre actually acts on: who is
exposed, how badly, and which districts to send resources to first.

    risk = hazard (depth)  x  exposure (people)  x  fragility (capacity to cope)

The fragility term is what separates this from a pure hazard map. Two
districts under the same 40 cm of water are not in the same trouble: one has
drained streets, solid construction and somewhere to go, the other does not.
Fragility weights come from config.LU_FRAGILITY and are applied on top of
the physical hazard, never mixed into it, so the hazard model stays
physically interpretable and the value judgements stay visible and
adjustable.

Depth thresholds follow standard flood-hazard practice:
    0.10 m  streets impassable to pedestrians in places, disruption begins
    0.30 m  cars float and stall; most vehicle evacuation becomes impossible
    0.70 m  dangerous to adults on foot; single-storey property inundated
    1.50 m  life-threatening; structural damage to light construction
"""
from __future__ import annotations
import numpy as np

DEPTH_BANDS = [
    (0.10, 0.30, "disruption"),
    (0.30, 0.70, "serious"),
    (0.70, 1.50, "dangerous"),
    (1.50, 1e9, "life-threatening"),
]
BAND_WEIGHT = {"disruption": 0.15, "serious": 0.45,
               "dangerous": 0.80, "life-threatening": 1.00}


def hazard_severity(depth_m: np.ndarray) -> np.ndarray:
    """Map depth to a 0-1 severity using the band thresholds above.

    Piecewise-linear between band edges rather than a step function, so the
    ranking degrades smoothly instead of jumping as cells cross a threshold.
    """
    d = np.asarray(depth_m, dtype=np.float32)
    knots_d = np.array([0.0, 0.10, 0.30, 0.70, 1.50, 3.0], dtype=np.float32)
    knots_s = np.array([0.0, 0.05, 0.30, 0.65, 0.92, 1.0], dtype=np.float32)
    return np.interp(d, knots_d, knots_s).astype(np.float32)


def impact_map(city: dict, depth_m: np.ndarray) -> dict:
    """Per-cell exposure and risk rasters."""
    pop = city["population"]
    frag = city["fragility"]
    sev = hazard_severity(depth_m)
    land = ~city["sea"]

    exposed = pop * (depth_m > 0.10) * land
    risk = pop * sev * (0.45 + 0.55 * frag) * land
    return {"severity": sev, "exposed": exposed.astype(np.float32),
            "risk": risk.astype(np.float32)}


def district_table(city: dict, depth_m: np.ndarray, top: int | None = None,
                   prob: np.ndarray | None = None):
    """Rank districts by population-weighted risk.

    When `prob` (a per-cell probability of exceeding the 10 cm threshold) is
    supplied, headcounts are the probability-weighted expectation rather than
    a hard threshold on predicted depth. See expected_exposure in the web
    exporter for why that matters.
    """
    names = list(city["district_names"])
    dis = city["districts"]
    pop = city["population"]
    im = impact_map(city, depth_m)
    rows = []
    for k, nm in enumerate(names):
        m = dis == k
        if not m.any():
            continue
        p = float(pop[m].sum())
        if p < 1.0:
            continue
        d = depth_m[m]
        exposed = (float((pop[m] * prob[m]).sum()) if prob is not None
                   else float(im["exposed"][m].sum()))
        rows.append({
            "district": nm,
            "population": p,
            "exposed_10cm": exposed,
            "exposed_frac": exposed / p,
            "exposed_30cm": float((pop[m] * (d > 0.30)).sum()),
            "exposed_70cm": float((pop[m] * (d > 0.70)).sum()),
            "max_depth_m": float(d.max()),
            "mean_depth_cm": float(100 * d[d > 0.01].mean()) if (d > 0.01).any() else 0.0,
            "risk_score": float(im["risk"][m].sum()),
            "informal_frac": float((city["landuse"][m] == 4).mean()),
            "drain_mm_h": float(city["drain_mm_h"][m].mean()),
        })
    rows.sort(key=lambda r: -r["risk_score"])
    return rows[:top] if top else rows


def city_summary(city: dict, depth_m: np.ndarray) -> dict:
    """Headline numbers for the whole city."""
    pop = city["population"]
    land = ~city["sea"]
    out = {"total_population": float(pop.sum())}
    for lo, hi, label in DEPTH_BANDS:
        m = land & (depth_m >= lo) & (depth_m < hi)
        out[f"people_{label}"] = float(pop[m].sum())
        out[f"area_km2_{label}"] = float(m.sum() * 0.01)
    out["people_affected"] = float(pop[land & (depth_m > 0.10)].sum())
    out["area_km2_affected"] = float((land & (depth_m > 0.10)).sum() * 0.01)
    out["max_depth_m"] = float(depth_m[land].max())
    return out


def equity_report(city: dict, depth_m: np.ndarray) -> list:
    """Exposure broken down by settlement type.

    The point of this table: informal settlements occupy the lowest ground
    and have a fraction of the drainage, so they carry exposure far out of
    proportion to their share of the population. Reporting a citywide
    average would hide exactly the group that needs the warning most.
    """
    from .config import LU_NAME
    pop, land = city["population"], ~city["sea"]
    tot_pop = float(pop[land].sum())
    tot_exp = float(pop[land & (depth_m > 0.10)].sum())
    rows = []
    for k in [1, 2, 3, 4, 5]:
        m = (city["landuse"] == k) & land
        if not m.any():
            continue
        p = float(pop[m].sum())
        e = float(pop[m & (depth_m > 0.10)].sum())
        rows.append({
            "landuse": LU_NAME[k],
            "population": p,
            "pop_share": p / max(tot_pop, 1),
            "exposed": e,
            "exposed_rate": e / max(p, 1),
            "share_of_exposed": e / max(tot_exp, 1),
            # >1 means this group carries more exposure than its size warrants
            "exposure_ratio": (e / max(p, 1)) / max(tot_exp / max(tot_pop, 1), 1e-9),
            "drain_mm_h": float(city["drain_mm_h"][m].mean()),
            "median_hand_m": float(np.median(city["hand"][m])),
        })
    return rows


def format_alerts(rows: list, n: int = 8) -> str:
    """Operations-centre style ranked alert list."""
    out = [f"{'#':<3}{'district':<18}{'people >10cm':>13}{'% of pop':>9}"
           f"{'max depth':>10}{'risk':>9}"]
    out.append("-" * len(out[0]))
    for i, r in enumerate(rows[:n], 1):
        out.append(f"{i:<3}{r['district']:<18}{r['exposed_10cm']:>13,.0f}"
                   f"{100*r['exposed_frac']:>8.1f}%{r['max_depth_m']:>9.2f}m"
                   f"{r['risk_score']/1000:>9.1f}k")
    return "\n".join(out)
