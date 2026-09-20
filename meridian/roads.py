"""Meridian City road network and Road Quality Index.

Two things live here:

1. A named road network, built by least-cost pathfinding across the terrain
   rather than drawn by hand, so roads follow valleys, avoid steep ground and
   bridge watercourses where a real alignment would.

2. The Road Quality Index (RQI) -- 74 variables in 7 weighted categories.

    RQI = 100 x [ 0.25*visual + 0.20*structural + 0.15*drainage
                + 0.15*traffic + 0.10*safety + 0.10*environment
                + 0.05*maintenance ]

    category   = SUM(weight_i * normalised_i)
    normalised = clip((X - X_bad) / (X_good - X_bad), 0, 1)

X_bad may be greater than X_good; that is how "more is worse" variables
(drain blockage, pothole count, axle load) are handled without a separate
sign convention.

The point of doing this properly rather than inventing an RQI number: the
DRAINAGE category is not synthetic filler. Standing-water depth and duration,
flooding frequency, drainage capacity, catchment area and runoff coefficient
all come straight out of the flood simulator, and the AI_VISUAL category is
driven by pothole statistics calibrated to the real YOLOv8 detector's output
distribution. So road quality degrades where the city actually floods, and
the two hazard models are genuinely coupled rather than merely displayed side
by side.
"""
from __future__ import annotations
import heapq
import math
import numpy as np

# ---------------------------------------------------------------- classes ---
# lanes, carriageway width (m), design ESAL (millions), base thickness (mm)
ROAD_CLASSES = {
    "highway":   dict(lanes=6, width_m=22.0, design_msa=60.0, thick=430, speed=90),
    "arterial":  dict(lanes=4, width_m=15.0, design_msa=25.0, thick=340, speed=60),
    "collector": dict(lanes=2, width_m=9.5,  design_msa=6.0,  thick=240, speed=45),
    "local":     dict(lanes=2, width_m=7.0,  design_msa=1.5,  thick=160, speed=30),
}

# Pixel-to-ground conversion for the pothole detector's measurements. The demo
# set is dashcam-style; 0.7 cm/px at the pothole is the nominal geometry for
# pothole_measure.py --cam-height-m 1.2 --pitch-deg 25 --hfov-deg 70.
CM_PER_PX = 0.7
# real detector output, demo_output/potholes.csv (109 detections)
POTHOLE_RADIUS_PX = (25.0, 48.1, 201.3)      # p10, p50, p90

ROAD_NAMES = [
    ("N1 Atlantic Highway", "highway"), ("Corniche Boulevard", "arterial"),
    ("Port Approach Road", "arterial"), ("Meridian Ring Road", "highway"),
    ("Kasbah Avenue", "arterial"), ("Exchange Street", "arterial"),
    ("Textile Mill Road", "collector"), ("Rivergate Causeway", "arterial"),
    ("Lower Ford Lane", "local"), ("Kiln Fields Road", "collector"),
    ("Northgate Avenue", "arterial"), ("Eastbrook Way", "collector"),
    ("Highfield Rise", "collector"), ("Sablon Street", "local"),
    ("Quarry Haul Road", "collector"), ("Parkside Drive", "collector"),
    ("Aqueduct Road", "collector"), ("Stonebridge Way", "arterial"),
    ("Clayhill Track", "local"), ("New Dawn Lane", "local"),
    ("Sunrise Camp Road", "local"), ("Canal Side Street", "local"),
    ("Foundry Row", "collector"), ("Airport Link Road", "arterial"),
    ("Olive Grove Road", "collector"), ("Westmead Avenue", "collector"),
    ("Palm Court Street", "local"), ("Southbank Road", "arterial"),
    ("Terrace Hill Climb", "local"), ("Marsh End Lane", "local"),
    ("Cedar Gate Road", "collector"), ("Saltmarsh Causeway", "collector"),
    ("Harbour Wall Road", "collector"), ("Beacon Hill Road", "collector"),
]

# ------------------------------------------------------- variable spec ------
# name: (X_bad, X_good). X_bad > X_good means "more is worse".
VARSPEC = {
    # ---- structural (9) ----
    "pavement_thickness_mm":      (120.0, 450.0),
    "subgrade_cbr_pct":           (2.0, 18.0),
    "pavement_deflection_mm":     (1.60, 0.25),
    "elastic_modulus_mpa":        (900.0, 5200.0),
    "layer_strength_index":       (0.25, 1.00),
    "subgrade_moisture_pct":      (28.0, 9.0),
    "soil_type_index":            (0.15, 0.95),
    "soil_plasticity_pi":         (38.0, 6.0),
    "pavement_age_yr":            (32.0, 1.0),
    # ---- drainage (16) ----
    "rainfall_intensity_mmh":     (95.0, 5.0),
    "cumulative_rainfall_mm":     (220.0, 15.0),
    "antecedent_rainfall_mm":     (85.0, 2.0),
    "drainage_capacity_mmh":      (1.0, 28.0),
    "drain_blockage_pct":         (85.0, 3.0),
    # Scales below are set from the observed distribution ON ROADS, not from
    # textbook ranges. Roads sit on relatively good ground, so a 0-9/yr
    # flooding scale put every road at ~0.98 and erased the flood signal
    # entirely -- the coupling to the flood model came out positive.
    "standing_water_depth_cm":    (35.0, 0.0),
    "standing_water_hours":       (8.0, 0.0),
    "flooding_frequency_per_yr":  (3.0, 0.0),
    "road_elevation_m":           (1.0, 60.0),
    "cross_slope_pct":            (0.4, 2.6),
    "longitudinal_slope_pct":     (0.1, 3.5),
    "groundwater_depth_m":        (0.4, 12.0),
    "soil_permeability_mmh":      (0.6, 45.0),
    "infiltration_rate_mmh":      (1.0, 38.0),
    "catchment_area_ha":          (450.0, 3.0),
    "runoff_coefficient":         (0.95, 0.30),
    # ---- traffic (11) ----
    "adt_vehicles_day":           (85000.0, 1500.0),
    "heavy_vehicle_pct":          (34.0, 2.0),
    "axle_load_t":                (13.5, 6.0),
    "esal_msa":                   (55.0, 0.8),
    "overloaded_vehicle_pct":     (30.0, 1.0),
    "traffic_growth_pct_yr":      (9.5, 1.0),
    "peak_hour_factor":           (0.19, 0.08),
    "average_speed_kmh":          (12.0, 60.0),
    "speed_variation_kmh":        (26.0, 5.0),
    "braking_events_per_km":      (45.0, 4.0),
    "congestion_index":           (0.92, 0.12),
    # ---- safety (11) ----
    "skid_resistance_sn":         (28.0, 68.0),
    "lane_marking_condition":     (0.10, 0.97),
    "road_sign_condition":        (0.12, 0.96),
    "street_lighting_lux":        (1.5, 32.0),
    "road_geometry_index":        (0.20, 0.95),
    "sight_distance_m":           (35.0, 240.0),
    "shoulder_condition":         (0.10, 0.95),
    "median_condition":           (0.05, 0.95),
    "guardrail_coverage":         (0.02, 0.95),
    "pedestrian_facility_index":  (0.03, 0.95),
    "accident_freq_per_km_yr":    (14.0, 0.3),
    # ---- environment (10) ----
    "mean_temperature_c":         (33.0, 17.0),
    "temperature_range_c":        (26.0, 7.0),
    "annual_rainfall_mm":         (900.0, 260.0),
    "humidity_pct":               (88.0, 45.0),
    "solar_radiation_mj":         (26.0, 14.0),
    "uv_index":                   (11.5, 4.0),
    "wind_speed_kmh":             (38.0, 8.0),
    "air_pollution_pm10":         (145.0, 18.0),
    "dust_deposition_gm2":        (22.0, 2.0),
    "extreme_weather_per_yr":     (11.0, 0.5),
    # ---- maintenance (7) ----
    "months_since_maintenance":   (96.0, 2.0),
    "maintenance_freq_per_yr":    (0.05, 2.2),
    "patching_pct":               (34.0, 0.5),
    "repair_quality_index":       (0.15, 0.95),
    "crack_sealing_index":        (0.05, 0.95),
    "drain_cleaning_per_yr":      (0.1, 4.0),
    "deterioration_rate_pct_yr":  (13.0, 1.0),
    # ---- AI visual, from the detector (10) ----
    "pothole_count_per_km":       (48.0, 0.0),
    "pothole_area_pct":           (9.0, 0.0),
    "crack_density_m_per_m2":     (5.5, 0.05),
    "rutting_depth_mm":           (28.0, 1.0),
    "ravelling_pct":              (35.0, 0.5),
    "surface_deformation_mm":     (42.0, 2.0),
    "patch_area_pct":             (30.0, 0.5),
    "edge_deterioration_pct":     (40.0, 1.0),
    "standing_water_pct":         (35.0, 0.0),
    "marking_visibility_pct":     (15.0, 97.0),
    "surface_texture_mm":         (0.35, 1.30),
}

CATEGORY_VARS = {
    "structural": ["pavement_thickness_mm", "subgrade_cbr_pct",
                   "pavement_deflection_mm", "elastic_modulus_mpa",
                   "layer_strength_index", "subgrade_moisture_pct",
                   "soil_type_index", "soil_plasticity_pi", "pavement_age_yr"],
    "drainage":   ["rainfall_intensity_mmh", "cumulative_rainfall_mm",
                   "antecedent_rainfall_mm", "drainage_capacity_mmh",
                   "drain_blockage_pct", "standing_water_depth_cm",
                   "standing_water_hours", "flooding_frequency_per_yr",
                   "road_elevation_m", "cross_slope_pct",
                   "longitudinal_slope_pct", "groundwater_depth_m",
                   "soil_permeability_mmh", "infiltration_rate_mmh",
                   "catchment_area_ha", "runoff_coefficient"],
    "traffic":    ["adt_vehicles_day", "heavy_vehicle_pct", "axle_load_t",
                   "esal_msa", "overloaded_vehicle_pct", "traffic_growth_pct_yr",
                   "peak_hour_factor", "average_speed_kmh",
                   "speed_variation_kmh", "braking_events_per_km",
                   "congestion_index"],
    "safety":     ["skid_resistance_sn", "lane_marking_condition",
                   "road_sign_condition", "street_lighting_lux",
                   "road_geometry_index", "sight_distance_m",
                   "shoulder_condition", "median_condition",
                   "guardrail_coverage", "pedestrian_facility_index",
                   "accident_freq_per_km_yr"],
    "environment": ["mean_temperature_c", "temperature_range_c",
                    "annual_rainfall_mm", "humidity_pct",
                    "solar_radiation_mj", "uv_index", "wind_speed_kmh",
                    "air_pollution_pm10", "dust_deposition_gm2",
                    "extreme_weather_per_yr"],
    "maintenance": ["months_since_maintenance", "maintenance_freq_per_yr",
                    "patching_pct", "repair_quality_index",
                    "crack_sealing_index", "drain_cleaning_per_yr",
                    "deterioration_rate_pct_yr"],
    "visual":     ["pothole_count_per_km", "pothole_area_pct",
                   "crack_density_m_per_m2", "rutting_depth_mm",
                   "ravelling_pct", "surface_deformation_mm",
                   "patch_area_pct", "edge_deterioration_pct",
                   "standing_water_pct", "marking_visibility_pct",
                   "surface_texture_mm"],
}
CATEGORY_WEIGHTS = {"visual": 0.25, "structural": 0.20, "drainage": 0.15,
                    "traffic": 0.15, "safety": 0.10, "environment": 0.10,
                    "maintenance": 0.05}

# variables carrying extra weight inside their category (all others equal)
VAR_EMPHASIS = {
    "pothole_count_per_km": 2.4, "pothole_area_pct": 2.0,
    "standing_water_depth_cm": 2.2, "standing_water_hours": 1.8,
    "flooding_frequency_per_yr": 1.8, "drainage_capacity_mmh": 1.6,
    "pavement_age_yr": 1.6, "esal_msa": 1.6, "pavement_deflection_mm": 1.5,
    "months_since_maintenance": 1.8, "skid_resistance_sn": 1.6,
}


def normalise(name: str, x: float) -> float:
    bad, good = VARSPEC[name]
    return float(np.clip((x - bad) / (good - bad), 0.0, 1.0))


def category_score(values: dict, cat: str) -> float:
    names = CATEGORY_VARS[cat]
    w = np.array([VAR_EMPHASIS.get(n, 1.0) for n in names], dtype=float)
    w /= w.sum()
    v = np.array([normalise(n, values[n]) for n in names], dtype=float)
    return float((w * v).sum())


def compute_rqi(values: dict) -> dict:
    cats = {c: category_score(values, c) for c in CATEGORY_WEIGHTS}
    rqi = 100.0 * sum(CATEGORY_WEIGHTS[c] * cats[c] for c in cats)
    return {"rqi": rqi, "categories": cats,
            "ai_visual_quality_index": 100.0 * cats["visual"]}


def rqi_grade(rqi: float) -> tuple:
    if rqi >= 80: return "excellent", "#2f7d55"
    if rqi >= 65: return "good", "#6aa84f"
    if rqi >= 50: return "fair", "#b8860f"
    if rqi >= 35: return "poor", "#c25a25"
    return "critical", "#9e1f3d"


# ------------------------------------------------------------ network -------
def _least_cost_path(cost, start, goal):
    """8-connected Dijkstra between two grid cells."""
    n, m = cost.shape
    INF = np.inf
    dist = np.full((n, m), INF)
    prev = np.full((n, m, 2), -1, dtype=np.int32)
    dist[start] = 0.0
    pq = [(0.0, start[0], start[1])]
    nbrs = [(-1, -1, 1.414), (-1, 0, 1), (-1, 1, 1.414), (0, -1, 1),
            (0, 1, 1), (1, -1, 1.414), (1, 0, 1), (1, 1, 1.414)]
    while pq:
        d, y, x = heapq.heappop(pq)
        if (y, x) == goal:
            break
        if d > dist[y, x]:
            continue
        for dy, dx, w in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < m:
                nd = d + w * cost[ny, nx]
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    prev[ny, nx] = (y, x)
                    heapq.heappush(pq, (nd, ny, nx))
    if not np.isfinite(dist[goal]):
        return []
    path, cur = [], goal
    while cur != start and prev[cur[0], cur[1]][0] >= 0:
        path.append(cur)
        cur = tuple(prev[cur[0], cur[1]])
    path.append(start)
    return path[::-1]


def build_network(city: dict, rng: np.random.Generator) -> dict:
    """Route named roads across the city by least-cost pathfinding."""
    n = city["dem"].shape[0]
    sea, land = city["sea"], ~city["sea"]
    slope = city["slope"]
    lu = city["landuse"]

    # Building cost: steep ground is expensive, open water very expensive,
    # dense urban land expensive (demolition), empty land cheap.
    cost = 1.0 + 26.0 * np.clip(slope, 0, 0.35) / 0.35
    cost += np.where(np.isin(lu, [1]), 5.0, 0.0)       # CBD
    cost += np.where(np.isin(lu, [2]), 2.4, 0.0)       # dense residential
    # Informal settlements are cheap to route through, which is precisely why
    # roads get pushed through them. Penalising them heavily kept the network
    # out of the districts whose road quality the model most needs to report.
    cost += np.where(np.isin(lu, [4]), 0.6, 0.0)
    cost += np.where(np.isin(lu, [9]), 7.0, 0.0)       # wetland
    cost[city["channels"]] += 9.0                      # bridges cost
    cost[sea] = 1e4

    cbd = (int(city["cbd"][0]), int(city["cbd"][1]))
    port = (int(city["port"][0]), int(city["port"][1]))
    airport = (int(city["airport"][0]), int(city["airport"][1]))

    # district centroids as network nodes
    dis = city["districts"]
    cents = []
    for k in range(len(city["district_names"])):
        ys, xs = np.nonzero(dis == k)
        if len(ys):
            cents.append((int(ys.mean()), int(xs.mean())))

    pairs = []
    # coastal highway: north to south, hugging the shore
    coast_col = np.clip(city["coast_col"].astype(int) + 5, 3, n - 4)
    pairs.append(((6, int(coast_col[6])), (n - 7, int(coast_col[n - 7])), "highway"))
    # ring road: CBD outwards through four quadrant anchors and back
    ring = [(int(0.18 * n), int(0.34 * n)), (int(0.42 * n), int(0.62 * n)),
            (int(0.74 * n), int(0.52 * n)), (int(0.80 * n), int(0.22 * n))]
    for a, b in zip(ring, ring[1:] + ring[:1]):
        pairs.append((a, b, "highway"))
    # radial arterials from the CBD
    for tgt in [port, airport, (6, int(0.30 * n)), (n - 7, int(0.28 * n)),
                (int(0.5 * n), n - 7), (int(0.2 * n), int(0.75 * n))]:
        pairs.append((cbd, tgt, "arterial"))
    # collectors: link district centroids to their nearest already-served node
    for c in cents:
        d = min(ring + [cbd, port, airport],
                key=lambda p: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)
        pairs.append((c, d, "collector"))

    road_id = np.full((n, n), -1, dtype=np.int16)
    segments = []
    for i, (a, b, cls) in enumerate(pairs):
        if sea[a] or sea[b]:
            continue
        path = _least_cost_path(cost, a, b)
        if len(path) < 6:
            continue
        nm, nm_cls = (ROAD_NAMES[len(segments) % len(ROAD_NAMES)]
                      if len(segments) < len(ROAD_NAMES)
                      else (f"Route {len(segments)+1}", cls))
        cls = cls if len(segments) < len(ROAD_NAMES) else cls
        ys = np.array([p[0] for p in path]); xs = np.array([p[1] for p in path])
        road_id[ys, xs] = len(segments)
        # make arterials and highways two cells wide so they read on the map
        if cls in ("highway", "arterial"):
            for dy, dx in ((0, 1), (1, 0)):
                yy, xx = np.clip(ys + dy, 0, n - 1), np.clip(xs + dx, 0, n - 1)
                m = (road_id[yy, xx] < 0) & land[yy, xx]
                road_id[yy[m], xx[m]] = len(segments)
        segments.append({
            "id": len(segments), "name": nm, "cls": cls,
            "cells": list(zip(ys.tolist(), xs.tolist())),
            "length_km": float(len(path) * 0.1),
        })
        # once built, reuse is cheap -> encourages a realistic shared network
        cost[ys, xs] *= 0.35
    return {"road_id": road_id, "segments": segments}
