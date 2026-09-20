"""Synthesis of the 74 Road Quality Index variables, and the RQI pipeline.

`roads.py` holds the network, the variable specification and the scoring
arithmetic. This module produces the actual values.

The division that matters: anything the city model or the flood simulator
already knows is *read from them*, not invented. Standing-water depth and
duration, flooding frequency, drainage capacity, catchment area, runoff
coefficient, elevation and longitudinal slope all come out of the flood
pipeline, so a road's DRAINAGE score degrades exactly where the city floods.
Only genuinely unmodelled quantities (subgrade CBR, axle loads, sign
condition) are sampled, and each is conditioned on road class, centrality
and neighbourhood upkeep so the variables stay mutually consistent instead
of being independent noise.

Pothole counts and sizes are calibrated to the real YOLOv8 detector output in
demo_output/potholes.csv, so the AI_VISUAL category reflects what the
teammate's model actually reports on road imagery.
"""
from __future__ import annotations
import math
import numpy as np

from .roads import (ROAD_CLASSES, CM_PER_PX, POTHOLE_RADIUS_PX,
                    normalise, compute_rqi, rqi_grade, build_network)


def _agg(city, ys, xs, key, fn=np.mean):
    return float(fn(city[key][ys, xs]))


def segment_variables(city, seg, flood, rng) -> dict:
    """All 74 RQI variables for one road segment."""
    cells = seg["cells"]
    ys = np.array([c[0] for c in cells])
    xs = np.array([c[1] for c in cells])
    cls = ROAD_CLASSES[seg["cls"]]
    n = city["dem"].shape[0]

    hand = _agg(city, ys, xs, "hand")
    elev = _agg(city, ys, xs, "dem")
    slope_pc = 100.0 * _agg(city, ys, xs, "slope")
    drain = _agg(city, ys, xs, "drain_mm_h")
    cn = _agg(city, ys, xs, "cn")
    imperv = _agg(city, ys, xs, "imperv")
    acc_ha = _agg(city, ys, xs, "acc", np.max)
    d_coast_km = _agg(city, ys, xs, "dist_coast") / 1000.0
    lu_mode = int(np.bincount(city["landuse"][ys, xs].astype(int)).argmax())
    informal_frac = float((city["landuse"][ys, xs] == 4).mean())
    pop_near = float(city["population"][ys, xs].mean())

    # Flood exposure is aggregated at the 95th percentile along the road, not
    # the mean. A road is closed by its lowest 100 m, not by its average
    # elevation, and the network deliberately avoids floodplains -- so a mean
    # washes the signal out completely and the RQI stops responding to floods.
    def worst(key):
        return float(np.percentile(flood[key][ys, xs], 95))

    fl_freq = worst("freq")                              # events per year
    fl_depth = worst("depth")                            # design-storm depth, m
    fl_hours = worst("hours")
    fl_exposure = float(np.clip(fl_freq / 3.0, 0, 1))    # 0-1 driver

    # neglect proxy: informal and outlying roads get less of everything.
    # `neglect` is drawn per road with real variance -- without it, averaging
    # 74 variables collapses every road onto the mean and every RQI lands in
    # a narrow "fair" band, which is useless for prioritising repairs.
    neglect = float(np.clip(rng.beta(1.25, 1.45), 0.02, 0.98))
    upkeep = float(np.clip(drain / 22.0, 0.10, 1.0)) * (1.0 - 0.45 * informal_frac)
    upkeep = float(np.clip(0.55 * upkeep + 0.45 * (1.0 - neglect)
                           - 0.25 * fl_exposure + rng.normal(0, 0.05), 0.03, 1.0))
    cls_q = {"highway": 1.0, "arterial": 0.85,
             "collector": 0.6, "local": 0.38}[seg["cls"]]
    q = float(np.clip(0.42 * cls_q + 0.58 * upkeep, 0.03, 1.0))

    def jit(s):
        return float(rng.normal(0, s))

    age = float(np.clip(rng.uniform(2, 30) * (1.25 - 0.5 * q), 1, 34))
    v = {}

    # ---------------- structural (9) ----------------
    v["pavement_thickness_mm"] = cls["thick"] * (0.72 + 0.4 * q) + jit(12)
    v["subgrade_cbr_pct"] = np.clip(3 + 13 * q - 4 * (hand < 2) + jit(1.2), 1.5, 19)
    v["pavement_deflection_mm"] = np.clip(1.5 - 1.15 * q + 0.012 * age + jit(.06), .18, 1.8)
    v["elastic_modulus_mpa"] = np.clip(1100 + 3800 * q - 45 * age + jit(160), 700, 5400)
    v["layer_strength_index"] = np.clip(0.28 + 0.66 * q + jit(.04), .1, 1.0)
    v["subgrade_moisture_pct"] = np.clip(11 + 15 * fl_exposure + 0.35 * max(0, 6 - hand) + jit(1), 7, 30)
    v["soil_type_index"] = np.clip(0.35 + 0.5 * (hand / 20) + jit(.08), .1, .97)
    v["soil_plasticity_pi"] = np.clip(30 - 22 * (hand / 20) + jit(2.5), 5, 40)
    v["pavement_age_yr"] = age

    # ---------------- drainage (16) -- straight from the flood model ------
    v["rainfall_intensity_mmh"] = flood["design_intensity_mmh"] * (1 + 0.05 * (elev / 120)) + jit(1.5)
    v["cumulative_rainfall_mm"] = flood["design_total_mm"] + jit(6)
    v["antecedent_rainfall_mm"] = np.clip(flood["antecedent_mm"] + jit(4), 0, 90)
    v["drainage_capacity_mmh"] = max(0.4, drain)
    v["drain_blockage_pct"] = np.clip(78 - 68 * upkeep + jit(6), 2, 92)
    v["standing_water_depth_cm"] = np.clip(100 * fl_depth, 0, 70)
    v["standing_water_hours"] = np.clip(fl_hours, 0, 16)
    v["flooding_frequency_per_yr"] = np.clip(fl_freq, 0, 12)
    v["road_elevation_m"] = max(0.5, elev)
    v["cross_slope_pct"] = np.clip(0.8 + 1.8 * q + jit(.15), .3, 3.0)
    v["longitudinal_slope_pct"] = np.clip(slope_pc, 0.05, 6.0)
    v["groundwater_depth_m"] = np.clip(0.5 + 0.55 * hand + jit(.4), .3, 14)
    v["soil_permeability_mmh"] = np.clip(48 * (1 - imperv) + jit(3), .4, 50)
    v["infiltration_rate_mmh"] = np.clip(40 * (1 - imperv) * (1 - .5 * fl_exposure) + jit(2.5), .5, 42)
    v["catchment_area_ha"] = np.clip(acc_ha, 1, 500)
    v["runoff_coefficient"] = np.clip((cn - 30) / 70.0, .25, .98)

    # ---------------- traffic (11) ----------------
    centrality = float(np.exp(-np.hypot(ys.mean() - city["cbd"][0],
                                        xs.mean() - city["cbd"][1]) / (0.28 * n)))
    base_adt = {"highway": 52000, "arterial": 24000,
                "collector": 8000, "local": 2200}[seg["cls"]]
    adt = base_adt * (0.55 + 0.95 * centrality) * (0.8 + 0.5 * pop_near / 150)
    v["adt_vehicles_day"] = float(np.clip(adt + jit(600), 800, 92000))
    v["heavy_vehicle_pct"] = np.clip((22 if lu_mode == 5 else 9) * (0.6 + 0.8 * cls_q) + jit(2), 1.5, 36)
    v["axle_load_t"] = np.clip(7.5 + 0.12 * v["heavy_vehicle_pct"] + jit(.5), 5.5, 14)
    v["esal_msa"] = np.clip(cls["design_msa"] * (0.35 + 1.3 * centrality)
                            * (v["heavy_vehicle_pct"] / 12) + jit(1.5), .3, 60)
    v["overloaded_vehicle_pct"] = np.clip(24 - 20 * q + jit(3), .5, 32)
    v["traffic_growth_pct_yr"] = np.clip(3.5 + 4 * centrality + jit(.8), .8, 10)
    v["peak_hour_factor"] = np.clip(.10 + .08 * centrality + jit(.01), .07, .20)
    v["average_speed_kmh"] = np.clip(cls["speed"] * (0.42 + 0.55 * (1 - centrality)) + jit(3), 8, 95)
    v["speed_variation_kmh"] = np.clip(6 + 18 * centrality + jit(2), 4, 28)
    v["braking_events_per_km"] = np.clip(6 + 34 * centrality + 12 * (1 - q) + jit(3), 3, 48)
    v["congestion_index"] = np.clip(.12 + .75 * centrality + jit(.06), .08, .95)

    # ---------------- safety (11) ----------------
    v["skid_resistance_sn"] = np.clip(66 - 26 * (1 - q) - 0.5 * age + jit(2.5), 25, 70)
    v["lane_marking_condition"] = np.clip(0.95 * q + jit(.06), .05, .98)
    v["road_sign_condition"] = np.clip(0.93 * q + jit(.07), .08, .97)
    v["street_lighting_lux"] = np.clip(32 * q * (0.5 + 0.7 * centrality) + jit(1.5), 1, 34)
    v["road_geometry_index"] = np.clip(0.95 - 0.5 * min(1, slope_pc / 5) + jit(.05), .18, .96)
    v["sight_distance_m"] = np.clip(240 * (0.35 + 0.6 * cls_q) - 18 * slope_pc + jit(12), 30, 250)
    v["shoulder_condition"] = np.clip(0.92 * q + jit(.07), .05, .96)
    v["median_condition"] = np.clip((0.9 * q if cls["lanes"] >= 4 else 0.12) + jit(.06), .03, .96)
    wide_or_steep = (slope_pc > 1.5) or (cls["lanes"] >= 4)
    v["guardrail_coverage"] = np.clip((0.9 * q if wide_or_steep else .15) + jit(.07), .01, .96)
    v["pedestrian_facility_index"] = np.clip(0.9 * q * (1 - .5 * informal_frac) + jit(.06), .02, .96)
    v["accident_freq_per_km_yr"] = np.clip(0.5 + 12 * (1 - q) * (0.4 + centrality) + jit(.8), .2, 15)

    # ---------------- environment (10) -- coastal gradient ----------------
    coastal = float(np.exp(-d_coast_km / 6.0))
    v["mean_temperature_c"] = 19.5 + 4.5 * (1 - coastal) + jit(.6)
    v["temperature_range_c"] = 9 + 11 * (1 - coastal) + jit(1)
    v["annual_rainfall_mm"] = 420 + 180 * (elev / 200) + jit(25)
    v["humidity_pct"] = np.clip(58 + 26 * coastal + jit(2.5), 44, 90)
    v["solar_radiation_mj"] = np.clip(18 + 6 * (1 - coastal) + jit(.8), 13, 27)
    v["uv_index"] = np.clip(7.5 + 2.5 * (1 - coastal) + jit(.4), 4, 12)
    v["wind_speed_kmh"] = np.clip(12 + 22 * coastal + jit(2), 7, 40)
    v["air_pollution_pm10"] = np.clip(28 + 95 * centrality + (35 if lu_mode == 5 else 0) + jit(7), 16, 150)
    v["dust_deposition_gm2"] = np.clip(3 + 16 * (1 - q) + jit(1.5), 1.5, 24)
    v["extreme_weather_per_yr"] = np.clip(2 + 7 * coastal + jit(.7), .4, 12)

    # ---------------- maintenance (7) ----------------
    v["months_since_maintenance"] = float(np.clip(96 * (1 - upkeep) * rng.uniform(.6, 1.2) + jit(3), 1, 100))
    v["maintenance_freq_per_yr"] = np.clip(2.2 * upkeep + jit(.15), .03, 2.4)
    v["patching_pct"] = np.clip(34 * (1 - q) * (0.4 + 0.03 * age) + jit(2), .3, 36)
    v["repair_quality_index"] = np.clip(0.93 * q + jit(.07), .12, .96)
    v["crack_sealing_index"] = np.clip(0.92 * upkeep + jit(.07), .03, .96)
    v["drain_cleaning_per_yr"] = np.clip(4 * upkeep + jit(.3), .05, 4.2)
    v["deterioration_rate_pct_yr"] = np.clip(1.5 + 11 * (1 - q) + 4 * fl_exposure + jit(.8), .8, 14)

    v["_meta"] = {"upkeep": upkeep, "q": q, "centrality": centrality,
                  "informal_frac": informal_frac, "hand": hand,
                  "flood_freq": fl_freq, "neglect": neglect}
    return v


def place_potholes(seg, v, rng) -> list:
    """Potholes along a road, sized from the real YOLOv8 detector output.

    Radii are drawn from a lognormal fitted to the p10/p50/p90 of the 109
    detections in demo_output/potholes.csv, converted at CM_PER_PX. Counts
    scale with structural, drainage, loading and maintenance stress.
    Confidence is sampled from the detector own confidence distribution, so
    the map shows what the model would report rather than ground truth.
    """
    p10, p50, p90 = [r * CM_PER_PX / 100.0 for r in POTHOLE_RADIUS_PX]
    mu = math.log(p50)
    sigma = (math.log(p90) - math.log(p10)) / 2.563       # p90/p10 spread
    stress = ((1 - normalise("pavement_age_yr", v["pavement_age_yr"])) * .35
              + (1 - normalise("drainage_capacity_mmh", v["drainage_capacity_mmh"])) * .30
              + (1 - normalise("esal_msa", v["esal_msa"])) * .20
              + (1 - normalise("months_since_maintenance", v["months_since_maintenance"])) * .15)
    lam = max(0.05, 26.0 * stress ** 1.7) * seg["length_km"]
    cells = seg["cells"]
    out = []
    for _ in range(int(rng.poisson(lam))):
        cy, cx = cells[int(rng.integers(len(cells)))]
        r = float(np.clip(rng.lognormal(mu, sigma), 0.06, 2.2))
        out.append({"y": int(cy), "x": int(cx),
                    "radius_m": round(r, 3),
                    "area_m2": round(math.pi * r * r, 3),
                    "conf": round(float(np.clip(rng.beta(2.2, 2.6) * .75 + .22, .25, .95)), 3)})
    return out


def visual_variables(v, seg, potholes, rng) -> None:
    """AI_VISUAL variables (11), driven by the detector output for this road."""
    km = max(seg["length_km"], 0.1)
    area_m2 = km * 1000 * ROAD_CLASSES[seg["cls"]]["width_m"]
    tot_area = sum(p["area_m2"] for p in potholes)
    stress = ((1 - normalise("pavement_age_yr", v["pavement_age_yr"])) * .5
              + (1 - normalise("drainage_capacity_mmh", v["drainage_capacity_mmh"])) * .5)

    def jit(s):
        return float(rng.normal(0, s))

    v["pothole_count_per_km"] = len(potholes) / km
    v["pothole_area_pct"] = 100.0 * tot_area / max(area_m2, 1)
    v["crack_density_m_per_m2"] = np.clip(0.1 + 5.2 * stress + jit(.3), .03, 6)
    v["rutting_depth_mm"] = np.clip(1 + 26 * stress + jit(1.5), .5, 30)
    v["ravelling_pct"] = np.clip(0.5 + 33 * stress + jit(2), .2, 38)
    v["surface_deformation_mm"] = np.clip(2 + 39 * stress + jit(2), 1, 45)
    v["patch_area_pct"] = np.clip(v["patching_pct"] * rng.uniform(.8, 1.2), .3, 36)
    v["edge_deterioration_pct"] = np.clip(1 + 38 * stress + jit(2.5), .5, 42)
    v["standing_water_pct"] = np.clip(2.8 * v["standing_water_hours"] + jit(1.5), 0, 38)
    v["marking_visibility_pct"] = np.clip(100 * v["lane_marking_condition"] + jit(3), 10, 99)
    v["surface_texture_mm"] = np.clip(1.3 - 0.95 * stress + jit(.05), .3, 1.4)


def flood_statistics(city, events) -> dict:
    """Per-cell flood exposure climatology, from the simulated event corpus.

    freq   fraction of storms that put >10 cm on the cell
    depth  mean peak depth across storms that flooded it
    hours  mean hours above 10 cm per flooding storm
    """
    n = city["dem"].shape[0]
    freq = np.zeros((n, n), np.float32)
    hours = np.zeros((n, n), np.float32)
    depths = []
    k = 0
    tot_mm, peak_mmh, ante = [], [], []
    for ev in events:
        hm = ev["h_max"].astype(np.float32)
        freq += (hm > 0.10)
        depths.append(hm)
        hours += (ev["depth_h"] > 0.10).sum(axis=0).astype(np.float32)
        k += 1
        m = ev["meta"]
        tot_mm.append(float(m["total_mm"]))
        peak_mmh.append(float(m["peak_mmh"]))
        ante.append(float(m["total_mm"]) * 0.25)
    k = max(k, 1)
    # Depth as the 90th-percentile storm, not the mean over all storms. Most
    # storms flood nothing, so a mean across the corpus is ~0 everywhere and
    # carries no information; the design storm is what a road is built for.
    depth_p90 = np.percentile(np.stack(depths), 90, axis=0).astype(np.float32)
    # Hours conditional on the cell flooding at all, rather than diluted by
    # every dry storm in the corpus.
    hours_cond = hours / np.maximum(freq, 1.0)
    # The corpus stands for roughly a decade of storms at ~12 events a year.
    EVENTS_PER_YEAR = 12.0
    return {
        "freq": (freq / k) * EVENTS_PER_YEAR,      # flooding events per year
        "depth": depth_p90, "hours": hours_cond,
        "design_total_mm": float(np.percentile(tot_mm, 90)) if tot_mm else 80.0,
        "design_intensity_mmh": float(np.percentile(peak_mmh, 90)) if peak_mmh else 40.0,
        "antecedent_mm": float(np.median(ante)) if ante else 15.0,
        "n_events": k,
    }


def build_road_quality(city, flood, seed=4242) -> dict:
    """Full pipeline: network -> variables -> potholes -> RQI per segment."""
    rng = np.random.default_rng(seed)
    net = build_network(city, rng)
    for seg in net["segments"]:
        v = segment_variables(city, seg, flood, rng)
        ph = place_potholes(seg, v, rng)
        visual_variables(v, seg, ph, rng)
        r = compute_rqi(v)
        seg["vars"] = v
        seg["potholes"] = ph
        seg["rqi"] = r["rqi"]
        seg["categories"] = r["categories"]
        seg["ai_visual_quality_index"] = r["ai_visual_quality_index"]
        seg["grade"] = rqi_grade(r["rqi"])[0]
    return net
