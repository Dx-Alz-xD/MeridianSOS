"""Export the interactive Meridian Flood Watch dashboard.

Unlike 06_export_web.py (three fixed scenarios, pre-coloured images), this
builds a dial: ONE storm pattern scaled across a range of rainfall totals,
each one put through the physics simulator, then through the surrogate under
two forecast qualities.

    rainfall dial   the same storm at 6 totals, physically re-simulated at
                    each -- so the sharp non-linearity at trunk-sewer
                    saturation is something you can slide through, not a
                    claim in a table
    forecast switch the model fed either the degraded operational forecast or
                    a perfect one, which shows how much of the error is the
                    model and how much is not knowing the rain
    hover readout   every raster is shipped as an 8-bit PNG the page decodes
                    per pixel, so hovering reads real per-cell values rather
                    than an approximation

Rasters are encoded to PNG rather than JSON: a 256x256 grid of float32 is
256 KB as JSON and about 25 KB as a PNG, and the browser decodes it for free.
Values are recovered by the inverse of the scaling recorded in `codec`.

Usage:  python scripts/07_export_interactive.py [--levels 6]
"""
from __future__ import annotations
import argparse, base64, io, json, pickle, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

from meridian.config import CFG, OUT, LU_NAME
from meridian.worldgen.city import load_city
from meridian.sim.storms import generate_event, _make_forecast
from meridian.sim.flood import simulate_event
from meridian.ml.features import Router, static_features, build_features
from meridian.ml.dataset import FEATURE_NAMES, FLOOD_THRESHOLD_M, LEADS
from meridian import risk, viz

WEB = OUT / "web"
WEB.mkdir(parents=True, exist_ok=True)

# The dashboard re-implements meridian/routing.py in JavaScript so a user can
# route between two points they pick, live, without a server. These constants
# are shipped rather than hard-coded in the template so the two copies of the
# cost model cannot drift apart silently.
from meridian.routing import IMPASSABLE_DEPTH_M, IMPASSABLE_DEPTH_HEAVY_M
ROUTING_CONST = {
    "impassable_m": IMPASSABLE_DEPTH_M,
    "impassable_heavy_m": IMPASSABLE_DEPTH_HEAVY_M,
    "min_kmh": 3.0,
}

# Meridian is fictional. Anchoring it at a real Atlantic coastal latitude
# keeps distances, bearings and the storm climatology self-consistent.
ORIGIN_LAT, ORIGIN_LON = 33.35, -7.85       # SW corner of the grid

# raster codecs: name -> (max value, gamma). v = 255*(x/max)**(1/gamma)
CODEC = {
    "depth":    (6.0, 2.0),      # metres, gamma 2 for resolution at the shallow end
    "prob":     (1.0, 1.0),
    "elev":     (230.0, 1.0),
    "hand":     (60.0, 1.5),
    "drain":    (30.0, 1.0),
    "pop":      (420.0, 2.0),
    "imperv":   (1.0, 1.0),
    "rain":     (400.0, 1.5),    # event total, mm
    "landuse":  (255.0, 1.0),    # stored raw
    "district": (255.0, 1.0),    # stored raw
}


def enc(arr: np.ndarray, kind: str) -> str:
    """Encode a raster as an 8-bit grayscale PNG data URI."""
    mx, gamma = CODEC[kind]
    if kind in ("landuse", "district"):
        v = np.clip(np.nan_to_num(arr, nan=0) + (1 if kind == "district" else 0),
                    0, 255).astype(np.uint8)
    else:
        x = np.clip(np.nan_to_num(arr, nan=0.0) / mx, 0, 1)
        v = np.round(255.0 * np.power(x, 1.0 / gamma)).astype(np.uint8)
    buf = io.BytesIO()
    plt.imsave(buf, v, format="png", cmap="gray", vmin=0, vmax=255)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def base_map_png(city) -> str:
    dem, sea = city["dem"], city["sea"]
    ls = LightSource(azdeg=315, altdeg=42)
    sh = ls.hillshade(np.where(sea, 0, dem), vert_exag=9,
                      dx=CFG.grid.cell_size_m, dy=CFG.grid.cell_size_m)
    lu = viz.LU_CMAP(viz.LU_NORM(city["landuse"]))[..., :3]
    img = 0.45 * lu + 0.55 * sh[..., None]
    img[sea] = np.array([0.07, 0.22, 0.36])
    rgba = np.concatenate([np.clip(img, 0, 1), np.ones(img.shape[:2] + (1,))], -1)
    buf = io.BytesIO()
    plt.imsave(buf, (rgba * 255).astype(np.uint8), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def turbo_lut(n=48):
    cm = plt.get_cmap("turbo")
    return [[int(255 * c) for c in cm(i / (n - 1))[:3]] for i in range(n)]


def people_layer(city, on_road, n_households=2200):
    """Household registry + control centre. Storm-independent, so this can be
    regenerated under --render-only without touching the simulator."""
    from meridian import households as HH
    reg = HH.build_registry(city, on_road, np.random.default_rng(4114),
                            n_households=n_households)
    ctl = HH.place_control_center(city, on_road)
    print(f"  {len(reg['households']):,} households, {reg['n_people']:,} residents "
          f"enrolled; control centre at {ctl['district']} "
          f"(HAND {ctl['hand_m']:.1f} m)")
    return reg, ctl


def decode_road_mask(data_uri: str, n: int) -> np.ndarray:
    """Recover the on-road boolean mask from the road_id PNG in a payload.

    road_id is stored as (id + 1), so any non-zero byte is a road cell. This
    exists so --render-only can place households without rebuilding the road
    network, which would mean re-reading all 100 event files.
    """
    raw = base64.b64decode(data_uri.split(",", 1)[1])
    img = plt.imread(io.BytesIO(raw))
    if img.ndim == 3:
        img = img[..., 0]
    v = np.round(img * 255).astype(np.int32) if img.max() <= 1.0 else img.astype(np.int32)
    return v.reshape(n, n) > 0


def hourly(x, rec):
    nh = x.shape[0] // rec
    return x[:nh * rec].reshape(nh, rec, x.shape[1], x.shape[2]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=6)
    # Issue the forecast BEFORE the storm peaks. At t0=8 the hyetograph has
    # already crested, so every district reports "impassable now" and the
    # time-to-impact table has nothing to warn about.
    ap.add_argument("--t0", type=int, default=5)
    ap.add_argument("--households", type=int, default=2200,
                    help="size of the MeridianSOS check-in registry")
    ap.add_argument("--render-only", action="store_true",
                    help="re-apply the template to the cached payload "
                         "without re-running the simulations")
    args = ap.parse_args()

    cache = WEB / "payload.json"
    if args.render_only:
        if not cache.exists():
            raise SystemExit("no cached payload; run once without --render-only")
        payload = json.loads(cache.read_text(encoding="utf-8"))
        # The household registry and the control centre do not depend on the
        # storm, so refresh them here rather than making a copy change cost
        # six physics simulations.
        city = load_city()
        on_road = decode_road_mask(payload["city"]["road_id"],
                                   int(payload["city"]["n"]))
        print("rebuilding household registry...")
        reg, ctl = people_layer(city, on_road, args.households)
        payload["registry"] = reg
        payload["control"] = ctl
        from meridian import roads as _RD
        payload.setdefault("roads", {})["classes"] = {
            k: {"speed": v["speed"], "lanes": v["lanes"]}
            for k, v in _RD.ROAD_CLASSES.items()}
        payload["routing"] = ROUTING_CONST
        blob = json.dumps(payload, default=float)
        cache.write_text(blob, encoding="utf-8")
        html = (WEB / "template_interactive.html").read_text(encoding="utf-8")
        out = html.replace("/*__DATA__*/null", blob)
        (WEB / "index.html").write_text(out, encoding="utf-8")
        print(f"re-rendered {WEB/'index.html'} ({len(out.encode())/1e6:.1f} MB)")
        return

    city = load_city()
    sea, land = city["sea"], ~city["sea"]
    land_flat = land.ravel()
    with open(OUT / "models" / "surrogate.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models = bundle["models"]
    router, static = Router(city), static_features(city)
    rec = int(round(60 / CFG.storm.timestep_min))

    # ---- road network + Road Quality Index -------------------------------
    from meridian.ml.dataset import load_index as _li, load_event as _le
    from meridian import rqi as RQ, roads as RD
    print("building road network and RQI...")
    _events = [_le(m["id"]) for m in _li()]
    flood_clim = RQ.flood_statistics(city, _events)
    net = RQ.build_road_quality(city, flood_clim)
    segs = net["segments"]
    road_id = net["road_id"]
    pothole_cell = np.zeros(city["dem"].shape, np.float32)
    for s in segs:
        for p in s["potholes"]:
            pothole_cell[p["y"], p["x"]] += 1
    rq = np.array([s["rqi"] for s in segs])
    print(f"  {len(segs)} roads, RQI {rq.min():.0f}-{rq.max():.0f} "
          f"(median {np.median(rq):.0f}), "
          f"{sum(len(s['potholes']) for s in segs):,} potholes")
    del _events

    from meridian import routing as RT
    graph = RT.build_graph(city, net)
    facilities = RT.place_facilities(city, graph, np.random.default_rng(5))
    hospitals = [f for f in facilities if f["kind"] == "hospital"]
    # three incidents chosen on the roads with the highest flood exposure,
    # which is where an ambulance is most likely to be cut off
    ranked = sorted(segs, key=lambda s: -s["vars"]["flooding_frequency_per_yr"])
    incidents = []
    for s in ranked[:3]:
        yx = RT.nearest_road(graph, s["cells"][len(s["cells"]) // 2])
        incidents.append({"name": s["name"], "yx": [int(yx[0]), int(yx[1])]})
    print(f"  {len(facilities)} facilities, {len(incidents)} incident sites")

    # ---- households + control centre --------------------------------------
    print("placing household registry...")
    registry, control = people_layer(city, graph["on_road"], args.households)

    # --- one storm pattern, scaled across a range of totals ----------------
    rng = np.random.default_rng(20260920)
    base = generate_event(CFG, rng, severity="heavy",
                          orography=np.clip(city["dem"], 0, None))
    base_total = base["total_mm"]
    targets = np.linspace(18.0, 190.0, args.levels)
    print(f"base storm pattern {base_total:.0f} mm -> scaling to "
          + ", ".join(f"{t:.0f}" for t in targets) + " mm")

    scenarios = []
    for target in targets:
        t_run = time.time()
        scale = target / base_total
        rain_obs = (base["rain_obs"] * scale).astype(np.float32)
        rain_fc = _make_forecast(rain_obs, np.random.default_rng(7))

        sim = simulate_event(city, rain_obs, base["dt_h"], CFG, record_every=rec)
        ro, rf = hourly(rain_obs, rec), hourly(rain_fc, rec)
        depth_h = sim["frames"]
        nh = depth_h.shape[0]
        t0 = min(args.t0, nh - max(LEADS) - 1)

        entry = {
            "total_mm": float(target),
            "max_point_mm": float((rain_obs.sum(0) * base["dt_h"]).max()),
            "peak_mmh": float(rain_obs.max()),
            "sewer_saturated_h": sim["sewer_saturated_h"],
            "sim_seconds": time.time() - t_run,
            "rain_img": enc(rain_obs.sum(0) * base["dt_h"], "rain"),
            "t0": int(t0),
            "leads": {},
        }

        # model input: degraded forecast (operational) vs perfect (oracle)
        for mode, rain_in in (("fc", rf), ("perfect", ro)):
            X = build_features(city, router, static, rain_in, ro, depth_h, t0)
            Xf = X.reshape(X.shape[0], -1)[:, land_flat].T
            for li, lead in enumerate(LEADS):
                t = time.time()
                prob = models[lead]["clf"].predict_proba(Xf)[:, 1]
                dep = np.clip(np.expm1(models[lead]["reg"].predict(Xf)) / 100, 0, None)
                ms = 1000 * (time.time() - t)
                pg = np.zeros(land.size, np.float32); pg[land_flat] = prob
                dg = np.zeros(land.size, np.float32); dg[land_flat] = dep
                pg, dg = pg.reshape(land.shape), dg.reshape(land.shape)
                truth = depth_h[min(t0 + lead, nh - 1)]

                from meridian.ml import metrics as M
                r = M.summarise(prob, truth.ravel()[land_flat],
                                name=f"+{lead}h")
                exp_people = float((city["population"] * pg * land).sum())
                key = f"{mode}_{lead}"
                entry["leads"][key] = {
                    "pred_depth": enc(dg, "depth"),
                    "prob": enc(pg, "prob"),
                    "csi": r["CSI"], "pod": r["POD"], "far": r["FAR"],
                    "auc": r["roc_auc"],
                    "pred_people": exp_people,
                    "pred_area": float((pg > 0.5).sum() * 0.01),
                    "infer_ms": ms,
                    "districts": [
                        {"name": d["district"], "people": d["exposed_10cm"],
                         "frac": d["exposed_frac"], "depth": d["max_depth_m"],
                         "informal": d["informal_frac"]}
                        for d in risk.district_table(city, dg, prob=pg)[:8]],
                }
                if mode == "fc":                    # truth is mode-independent
                    st = risk.city_summary(city, truth)
                    # emergency routing on the PREDICTED map: this is the
                    # decision a dispatcher would actually take from it
                    # Route on BOTH the forecast and the physics truth. The
                    # depth regressor is biased low (log1p/expm1), so routing
                    # on it alone reports comfortable journeys down roads that
                    # are actually impassable -- the one failure mode that
                    # would get someone killed. Showing the pair makes the
                    # forecast's optimism visible instead of hiding it.
                    def best_route(depth_map):
                        best = None
                        for h in hospitals:
                            rr = RT.route(graph, depth_map, tuple(h["yx"]),
                                          tuple(inc["yx"]))
                            if rr["ok"] and (best is None
                                             or rr["minutes"] < best["minutes"]):
                                best = dict(rr, origin=h["name"])
                        return best

                    routes = []
                    for inc in incidents:
                        bp = best_route(dg)          # on the AI forecast
                        bt = best_route(truth)       # on the physics truth
                        hv = None
                        if bp is None:
                            hv = RT.route(graph, dg, tuple(hospitals[0]["yx"]),
                                          tuple(inc["yx"]), heavy=True)
                        rrec = {"to": inc["name"], "ok": bp is not None,
                               "heavy_ok": bool(hv.get("ok")) if hv else True,
                               "path": (bp or hv or {}).get("path", []),
                               "minutes": (bp or hv or {}).get("minutes", 0.0),
                               "origin": (bp or {}).get("origin",
                                                        hospitals[0]["name"]),
                               "truth_ok": bt is not None,
                               "truth_minutes": (bt or {}).get("minutes", 0.0)}
                        if bp:
                            rrec.update({"km": bp["km"],
                                         "max_depth_m": bp["max_depth_m"],
                                         "potholes": bp["potholes_on_route"],
                                         "submerged": bp["submerged_potholes"],
                                         "min_rqi": bp["min_rqi"]})
                        if bt:
                            rrec["truth_max_depth_m"] = bt["max_depth_m"]
                            rrec["truth_submerged"] = bt["submerged_potholes"]
                        routes.append(rrec)
                    entry["leads"][key]["routes"] = routes
                    entry["leads"][key].update({
                        "true_depth": enc(truth, "depth"),
                        "true_people": st["people_affected"],
                        "true_area": st["area_km2_affected"],
                        "equity": [
                            {"type": e["landuse"].replace("_", " "),
                             "rate": e["exposed_rate"], "ratio": e["exposure_ratio"],
                             "drain": e["drain_mm_h"], "hand": e["median_hand_m"]}
                            for e in risk.equity_report(city, truth)
                            if e["population"] > 5000],
                    })
        # ---- ensemble: 16 rainfall forecasts, 16 runs of the surrogate ----
        # This is the argument for the whole architecture. Each member costs
        # about a second; the physics simulator costs ~45 s, so an ensemble
        # this size is 15 s of surrogate against 12 minutes of simulation.
        n_mem, lead_e = 16, LEADS[-1]
        t_ens = time.time()
        members = []
        for mi in range(n_mem):
            rfc = _make_forecast(rain_obs, np.random.default_rng(1000 + mi))
            Xe = build_features(city, router, static, hourly(rfc, rec), ro,
                                depth_h, t0)
            Xef = Xe.reshape(Xe.shape[0], -1)[:, land_flat].T
            pe = models[lead_e]["clf"].predict_proba(Xef)[:, 1]
            g = np.zeros(land.size, np.float32); g[land_flat] = pe
            members.append(g.reshape(land.shape))
        ens = np.stack(members)
        people = np.array([float((city["population"] * m * land).sum())
                           for m in ens])
        ens_mean = ens.mean(0)
        entry["ensemble"] = {
            "n_members": n_mem, "lead": lead_e,
            "seconds": round(time.time() - t_ens, 2),
            "mean_img": enc(ens_mean, "prob"),
            "spread_img": enc(ens.std(0) * 3.0, "prob"),   # x3 to use the range
            "people_p10": float(np.percentile(people, 10)),
            "people_p50": float(np.percentile(people, 50)),
            "people_p90": float(np.percentile(people, 90)),
            "people_members": [float(p) for p in people],
            "agreement": float((np.abs(ens_mean - 0.5) * 2).mean()),
        }

        # ---- time to impact, per district --------------------------------
        # First hour at which >2% of a district is standing in >10 cm. This is
        # the number an operations centre acts on; a depth map is not.
        tti = []
        dis, popr = city["districts"], city["population"]
        for k, nm in enumerate(city["district_names"]):
            m = dis == k
            ptot = float(popr[m].sum())
            if ptot < 500:
                continue
            hit = None
            for h in range(t0, min(nh, t0 + 13)):
                frac = float(popr[m & (depth_h[h] > 0.10)].sum()) / ptot
                if frac > 0.02:
                    hit = h - t0
                    break
            if hit is not None:
                pk = max(float(popr[m & (depth_h[h] > 0.10)].sum())
                         for h in range(t0, min(nh, t0 + 13)))
                tti.append({"district": nm, "hours": hit,
                            "people": pk, "pop": ptot})
        tti.sort(key=lambda r: (r["hours"], -r["people"]))
        entry["time_to_impact"] = tti[:10]

        scenarios.append(entry)
        hm = sim["h_max"]
        print(f"  {target:5.0f} mm -> flooded {100*((hm>.1)&land).sum()/land.sum():5.2f}%  "
              f"sewer saturated {sim['sewer_saturated_h']:4.1f} h  "
              f"({time.time()-t_run:.0f}s)", flush=True)

    payload = {
        "city": {
            "name": "Meridian City",
            "population": float(city["population"].sum()),
            "n": int(CFG.grid.n),
            "cell_m": float(CFG.grid.cell_size_m),
            "extent_km": CFG.grid.extent_km,
            "origin": [ORIGIN_LAT, ORIGIN_LON],
            "n_districts": len(city["district_names"]),
            "district_names": list(city["district_names"]),
            "lu_names": {int(k): v for k, v in LU_NAME.items()},
            "lu_colors": {int(k): viz.LU_COLORS[k] for k in viz.LU_COLORS},
            "base_img": base_map_png(city),
            "elev": enc(city["dem"], "elev"),
            "hand": enc(city["hand"], "hand"),
            "drain": enc(city["drain_mm_h"], "drain"),
            "pop": enc(city["population"], "pop"),
            "imperv": enc(city["imperv"], "imperv"),
            "landuse": enc(city["landuse"], "landuse"),
            "district": enc(np.where(city["districts"] < 0, -1,
                                     city["districts"]), "district"),
            "sea": enc((city["sea"] * 255).astype(np.float32), "landuse"),
            "ghost": enc((city["ghost"] * 255).astype(np.float32), "landuse"),
            "channels": enc((city["channels"] * 255).astype(np.float32), "landuse"),
            "road_id": enc((road_id + 1).astype(np.float32), "landuse"),
            "potholes": enc(np.clip(pothole_cell * 20, 0, 255), "landuse"),
        },
        "roads": {
            "segments": [{
                "id": s["id"], "name": s["name"], "cls": s["cls"],
                "length_km": round(s["length_km"], 2),
                "rqi": round(s["rqi"], 1), "grade": s["grade"],
                "ai_visual_quality_index": round(s["ai_visual_quality_index"], 1),
                "categories": {k: round(v, 3) for k, v in s["categories"].items()},
                "n_potholes": len(s["potholes"]),
                "pothole_median_r_m": round(float(np.median(
                    [p["radius_m"] for p in s["potholes"]])), 2) if s["potholes"] else 0.0,
                "vars": {k: (round(float(x), 3) if not isinstance(x, dict) else None)
                         for k, x in s["vars"].items() if k != "_meta"},
            } for s in segs],
            "weights": RD.CATEGORY_WEIGHTS,
            "category_vars": RD.CATEGORY_VARS,
            "varspec": {k: list(v) for k, v in RD.VARSPEC.items()},
            "n_potholes": int(sum(len(s["potholes"]) for s in segs)),
            "total_km": round(float(sum(s["length_km"] for s in segs)), 1),
            "cm_per_px": RD.CM_PER_PX,
            "classes": {k: {"speed": v["speed"], "lanes": v["lanes"]}
                        for k, v in RD.ROAD_CLASSES.items()},
        },
        "registry": registry,
        "control": control,
        "routing": ROUTING_CONST,
        "facilities": [{"kind": f["kind"], "name": f["name"],
                        "yx": [int(f["yx"][0]), int(f["yx"][1])]}
                       for f in facilities],
        "incidents": incidents,
        "codec": {k: {"max": v[0], "gamma": v[1]} for k, v in CODEC.items()},
        "turbo": turbo_lut(),
        "model": {"n_features": len(FEATURE_NAMES), "leads": list(LEADS),
                  "threshold_m": FLOOD_THRESHOLD_M,
                  "n_events": bundle.get("n_events", 39)},
        "scenarios": scenarios,
    }
    try:
        payload["eval"] = json.loads((OUT / "eval_results.json").read_text())
    except Exception:
        payload["eval"] = {}

    blob = json.dumps(payload, default=float)
    cache.write_text(blob, encoding="utf-8")
    html = (WEB / "template_interactive.html").read_text(encoding="utf-8")
    out = html.replace("/*__DATA__*/null", blob)
    (WEB / "index.html").write_text(out, encoding="utf-8")
    print(f"\nwrote {WEB/'index.html'}  ({len(out.encode())/1e6:.1f} MB)")
    print(f"cached payload -> {cache.name}  (use --render-only to re-skin)")


if __name__ == "__main__":
    main()
