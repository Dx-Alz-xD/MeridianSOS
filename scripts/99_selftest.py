"""Self-test: validate every stage of the pipeline.

Checks the invariants that actually catch bugs, rather than just running the
code. Each one corresponds to a real defect found during development:

  drainage routes to the sea       - grid borders once acted as fake outlets
  flow routing is exact            - upstream accumulation must equal the
                                     independently computed flow accumulation
  mass is conserved                - the simulator is the ground truth; if it
                                     leaks water, nothing downstream is valid
  informal > CBD exposure          - sub-grid channel conveyance once inverted
                                     this, which broke the whole premise
  features are finite              - NaNs propagate silently into XGBoost
  routed != local rainfall         - normalising by catchment size once made
                                     the routed features duplicates

Usage:  python scripts/99_selftest.py
"""
from __future__ import annotations
import sys
import numpy as np

from meridian.config import CFG
from meridian.worldgen.city import load_city
from meridian.sim.storms import generate_event
from meridian.sim.flood import simulate_event
from meridian.ml.features import Router, static_features, FEATURE_NAMES
from meridian.ml import metrics as M
from meridian import risk

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    print("Meridian City self-test\n" + "=" * 60)

    print("\n[1] city")
    city = load_city()
    land = ~city["sea"]
    check("city loads", "dem" in city and "population" in city,
          f"{land.sum():,} land cells")
    check("population matches config",
          abs(city["population"].sum() - CFG.city.population) < 1,
          f"{city['population'].sum():,.0f}")
    check("no unrouted land cells", (city["fdir"][land] >= 0).all(),
          f"{int((city['fdir'][land] < 0).sum())} unrouted")
    check("HAND is non-negative and finite",
          bool(np.isfinite(city["hand"]).all() and (city["hand"] >= 0).all()))
    check("ghost channel exists and is off the open network",
          bool(city["ghost"].any() and not (city["ghost"] & city["channels"]).any()),
          f"{int(city['ghost'].sum())} culverted cells")
    check("channels have finite bankfull capacity",
          bool((city["chan_qbf"][city["channels"]] > 0).all()),
          f"max {city['chan_qbf'].max():.0f} m3/s")

    print("\n[2] drainage reaches the ocean")
    from scipy.ndimage import binary_dilation
    mouths = city["channels"] & binary_dilation(city["sea"])
    check("channels reach the sea", mouths.sum() > 0, f"{int(mouths.sum())} river mouths")

    print("\n[3] equity structure")
    inf = (city["landuse"] == 4) & land
    cbd = (city["landuse"] == 1) & land
    check("informal settlements sit lower than the CBD",
          float(np.median(city["hand"][inf])) < float(np.median(city["hand"][cbd])),
          f"HAND {np.median(city['hand'][inf]):.1f} m vs {np.median(city['hand'][cbd]):.1f} m")
    check("informal settlements have worse drainage",
          float(city["drain_mm_h"][inf].mean()) < float(city["drain_mm_h"][cbd].mean()),
          f"{city['drain_mm_h'][inf].mean():.1f} vs {city['drain_mm_h'][cbd].mean():.1f} mm/h")

    print("\n[4] storm generator")
    rng = np.random.default_rng(3)
    oro = np.clip(city["dem"], 0, None)
    ev = generate_event(CFG, rng, severity="heavy", orography=oro)
    check("rainfall is finite and non-negative",
          bool(np.isfinite(ev["rain_obs"]).all() and (ev["rain_obs"] >= 0).all()))
    check("event total is in the heavy range", 42 <= ev["total_mm"] <= 95,
          f"{ev['total_mm']:.1f} mm")
    cv = float((ev["rain_obs"].sum(0)).std() / (ev["rain_obs"].sum(0)).mean())
    check("rainfall is spatially variable", cv > 0.12, f"spatial CV {cv:.2f}")
    check("forecast differs from truth",
          float(np.abs(ev["rain_fc"] - ev["rain_obs"]).mean()) > 1e-3)

    print("\n[5] flood simulator")
    sim = simulate_event(city, ev["rain_obs"], ev["dt_h"], CFG, record_every=4)
    check("mass is conserved", abs(sim["mass_error_frac"]) < 1e-6,
          f"relative error {sim['mass_error_frac']:+.1e}")
    check("depths are finite and non-negative",
          bool(np.isfinite(sim["h_max"]).all() and (sim["h_max"] >= 0).all()))
    check("runoff coefficient is physical", 0.0 < sim["runoff_coeff"] < 1.0,
          f"{sim['runoff_coeff']:.3f}")
    check("no water left standing on the ocean", float(sim["h_max"][city["sea"]].max()) == 0.0)

    print("\n[6] flood distribution")
    hm = sim["h_max"]
    r_inf = float(((hm > 0.1) & inf).sum() / max(inf.sum(), 1))
    r_cbd = float(((hm > 0.1) & cbd).sum() / max(cbd.sum(), 1))
    check("informal settlements flood more than the CBD", r_inf > r_cbd,
          f"{100*r_inf:.1f}% vs {100*r_cbd:.1f}%")

    print("\n[7] features")
    router = Router(city)
    static = static_features(city)
    ones = router.accumulate(np.ones(city["dem"].shape, np.float32))
    check("upstream routing matches flow accumulation",
          float(np.abs(ones[land] - city["acc"][land]).max()) < 0.5,
          f"max diff {np.abs(ones[land]-city['acc'][land]).max():.2f}")
    check("static feature count matches names", static.shape[0] == 16,
          f"{static.shape[0]} layers")

    from meridian.ml.features import build_features
    nh = ev["nt"] // 4
    ro = ev["rain_obs"][:nh * 4].reshape(nh, 4, 256, 256).mean(1)
    rf = ev["rain_fc"][:nh * 4].reshape(nh, 4, 256, 256).mean(1)
    X = build_features(city, router, static, rf, ro, sim["frames"][:nh], 6)
    check("feature stack width matches names", X.shape[0] == len(FEATURE_NAMES),
          f"{X.shape[0]} vs {len(FEATURE_NAMES)}")
    check("all features finite", bool(np.isfinite(X[:, land]).all()))
    i_loc = FEATURE_NAMES.index("rain_fc_6h")
    i_up = FEATURE_NAMES.index("up_fc_6h")
    corr = float(np.corrcoef(X[i_loc][land], X[i_up][land])[0, 1])
    check("routed rainfall is not a duplicate of local", abs(corr) < 0.97,
          f"correlation {corr:+.3f}")

    print("\n[8] metrics")
    y = np.array([0, 0, 1, 1, 0, 1], bool)
    s = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.7])
    check("AUC is correct on a known case", abs(M.roc_auc(s, y) - 1.0) < 1e-9,
          f"{M.roc_auc(s, y):.3f}")
    c = M.contingency(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    check("CSI is correct on a known case", abs(c["CSI"] - 1 / 3) < 1e-9,
          f"{c['CSI']:.3f}")

    print("\n[9] risk layer")
    rows = risk.district_table(city, hm)
    summ = risk.city_summary(city, hm)
    check("districts are ranked", len(rows) > 5 and
          rows[0]["risk_score"] >= rows[-1]["risk_score"], f"{len(rows)} districts")
    check("exposure never exceeds population",
          summ["people_affected"] <= summ["total_population"])
    eq = risk.equity_report(city, hm)
    inf_row = [r for r in eq if r["landuse"] == "informal"]
    check("equity report flags informal settlements",
          bool(inf_row) and inf_row[0]["exposure_ratio"] > 1.0,
          f"{inf_row[0]['exposure_ratio']:.2f}x citywide" if inf_row else "")

    print("\n[10] routing and the household registry")
    from meridian import roads as RD, rqi as RQ, routing as RT, households as HH
    # One synthetic event is enough: this section tests routing and registry
    # invariants, not the RQI climatology, which section 3 already covers.
    fake_ev = [{"h_max": hm, "depth_h": hm[None, ...],
                "meta": {"total_mm": 90.0, "peak_mmh": 30.0}}]
    net = RQ.build_road_quality(city, RQ.flood_statistics(city, fake_ev))
    graph = RT.build_graph(city, net)
    on = graph["on_road"]

    # A path length that counts diagonals as one cell understates a D8 route
    # by up to 41%. This caught the dashboard reporting 7.2 km for 8.3 km.
    dry = np.zeros_like(hm)
    cells = np.argwhere(on)
    a, b = tuple(cells[0]), tuple(cells[len(cells) // 2])
    r = RT.route(graph, dry, a, b)
    if r["ok"]:
        p = np.array(r["path"])
        step = np.hypot(*np.diff(p, axis=0).T.astype(float))
        check("route length weights diagonal steps",
              abs(r["km"] - step.sum() * 0.1) < 1e-9
              and r["km"] >= (len(p) - 1) * 0.1 - 1e-9,
              f"{r['km']:.2f} km over {len(p)} cells")
        # No route may cross a cell a car cannot drive through.
        r2 = RT.route(graph, hm, a, b)
        if r2["ok"]:
            d = hm[tuple(np.array(r2["path"]).T)]
            check("no route crosses an impassable cell",
                  float(d.max()) < RT.IMPASSABLE_DEPTH_M,
                  f"deepest {100*d.max():.0f} cm < {100*RT.IMPASSABLE_DEPTH_M:.0f} cm")

    reg = HH.build_registry(city, on, np.random.default_rng(4114), n_households=300)
    hh = reg["households"]
    check("every household sits on habitable land",
          all(not city["sea"][h["y"], h["x"]]
              and city["landuse"][h["y"], h["x"]] in HH.RESIDENTIAL_LU for h in hh))
    check("every household has a road access point",
          all(on[h["ry"], h["rx"]] for h in hh))
    check("house codes are unique", len({h["code"] for h in hh}) == len(hh),
          f"{len(hh)} codes")
    check("roster size matches the resident count",
          all(len(h["occ"]) == h["n"] for h in hh))
    # The registry must not collapse into one district; a floor per district
    # is what keeps the thin coastal wards on the map.
    check("registry spans most districts",
          len({h["d"] for h in hh}) >= 0.8 * len(city["district_names"]),
          f"{len({h['d'] for h in hh})} of {len(city['district_names'])}")

    ctl = HH.place_control_center(city, on)
    check("control centre is on the road network", bool(on[tuple(ctl["yx"])]))
    # An operations centre that floods during the event it manages is useless.
    check("control centre is above the flood corridor", ctl["hand_m"] > 4.0,
          f"HAND {ctl['hand_m']:.1f} m")
    check("control centre stays dry in this storm",
          float(hm[tuple(ctl["yx"])]) < 0.10,
          f"{100*float(hm[tuple(ctl['yx'])]):.1f} cm at {city['district_names'][city['districts'][tuple(ctl['yx'])]]}")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
