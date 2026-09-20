"""Seed the MeridianSOS chat app from the Meridian city model.

`meridian-sos-local/` ships seeded with Vellore, Tamil Nadu: the Palar river,
Katpadi, Thorapadi, gauges on a basin that has nothing to do with anything
else in this repository. Run it as-is beside the flood dashboard and you have
two apps about two different cities sharing a name.

This script replaces that seed with Meridian's own geography, derived from the
same artefacts the dashboard uses:

    zones      the 28 districts, outlined by contouring the district raster,
               each coloured by its CURRENT exposure in the chosen storm
    river      the main stem, traced upstream from the river mouth by
               following flow accumulation
    gauges     channel depth at three real gauge sites plus the storm total,
               with warning marks taken from the channel's own bankfull depth
    shelters   the facilities `routing.place_facilities` sited, plus the
               MeridianSOS Control Centre
    alerts     written from the district warnings and the time-to-impact table

So when the control room in the chat app says Foundry Row is in trouble, it is
because the simulator put water there -- not because someone typed it in.

The risk levels follow a chosen rainfall scenario, so the SOS app and the
dashboard can be demonstrated at the same storm:

    python scripts/09_export_sos_seed.py --scenario 3

Writes `meridian-sos-local/meridian-seed.json`. The server loads that on a
cold start; delete `meridian-sos-local/meridian-data.json` to re-seed.
"""
from __future__ import annotations
import argparse, base64, io, json, math, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meridian.config import CFG, OUT, LU_NAME
from meridian.worldgen.city import load_city
from meridian import risk

WEB = OUT / "web"
SOS = OUT.parent / "meridian-sos-local"
ORIGIN_LAT, ORIGIN_LON = 33.35, -7.85          # matches 07_export_interactive
SVG_W, SVG_H = 260.0, 210.0                    # the app's map viewBox


def decode(uri: str, n: int) -> np.ndarray:
    raw = base64.b64decode(uri.split(",", 1)[1])
    img = plt.imread(io.BytesIO(raw))
    if img.ndim == 3:
        img = img[..., 0]
    v = np.round(img * 255) if img.max() <= 1.0 else img
    return v.astype(np.int32).reshape(n, n)


def geo_box(n: int, cell_m: float) -> dict:
    """The app stores a lat/lon box and projects into its 260x210 viewBox."""
    km = n * cell_m / 1000.0
    north = ORIGIN_LAT + km / 111.32
    mid = 0.5 * (ORIGIN_LAT + north)
    east = ORIGIN_LON + km / (111.32 * math.cos(math.radians(mid)))
    return {"west": round(ORIGIN_LON, 5), "east": round(east, 5),
            "south": round(ORIGIN_LAT, 5), "north": round(north, 5)}


def cell_to_ll(y, x, n, box):
    """Grid cell -> lat/lon. y=0 is the NORTH edge, matching the dashboard."""
    lon = box["west"] + (x + 0.5) / n * (box["east"] - box["west"])
    lat = box["north"] - (y + 0.5) / n * (box["north"] - box["south"])
    return lat, lon


def cell_to_svg(y, x, n):
    return (x + 0.5) / n * SVG_W, (y + 0.5) / n * SVG_H


def outline(mask: np.ndarray, n: int, max_pts: int = 26) -> str:
    """Zone polygon: contour the district mask and simplify to a few points.

    A convex hull would be simpler but districts here are genuinely concave --
    they wrap around the estuary -- and a hull makes neighbouring zones
    overlap on the map.
    """
    if mask.sum() < 4:
        return ""
    pad = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), float)
    pad[1:-1, 1:-1] = mask.astype(float)
    cs = plt.contour(pad, levels=[0.5])
    paths = [p for c in cs.allsegs for p in c if len(p) >= 4]
    plt.close("all")
    if not paths:
        return ""
    best = max(paths, key=len)
    if len(best) > max_pts:
        idx = np.round(np.linspace(0, len(best) - 1, max_pts)).astype(int)
        best = best[idx]
    pts = []
    for px, py in best:                       # contour returns (col, row)
        sx, sy = cell_to_svg(py - 1, px - 1, n)
        pts.append(f"{sx:.1f},{sy:.1f}")
    return " ".join(pts)


def main_stem(city, n: int, max_pts: int = 22) -> str:
    """Trace the trunk river upstream from its mouth, following accumulation."""
    acc, chan, sea = city["acc"], city["channels"], city["sea"]
    # Start at the channel cell with the greatest upstream area anywhere.
    # Accumulation only grows downstream, so that cell IS the outlet of the
    # largest catchment -- no coastline test needed. Testing adjacency to the
    # sea instead missed the main mouth entirely, because the cell was only
    # diagonally adjacent to water, and traced a southern tributary that ran
    # along the bottom edge of the map looking like a border.
    on = chan & ~sea
    if not on.any():
        return ""
    cy, cx = np.unravel_index(int(np.argmax(np.where(on, acc, -1.0))), acc.shape)
    cy, cx = int(cy), int(cx)
    seen = {(int(cy), int(cx))}
    path = [(int(cy), int(cx))]
    for _ in range(4000):
        best, ba = None, -1.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = int(cy) + dy, int(cx) + dx
                if not (0 <= ny < n and 0 <= nx < n) or (ny, nx) in seen:
                    continue
                if not chan[ny, nx] or sea[ny, nx]:
                    continue
                if acc[ny, nx] > ba:
                    ba, best = float(acc[ny, nx]), (ny, nx)
        if best is None or ba < 1e-9:
            break
        seen.add(best)
        path.append(best)
        cy, cx = best
    if len(path) < 4:
        return ""
    if len(path) > max_pts:
        idx = np.round(np.linspace(0, len(path) - 1, max_pts)).astype(int)
        path = [path[i] for i in idx]
    return " ".join(f"{cell_to_svg(y, x, n)[0]:.0f},{cell_to_svg(y, x, n)[1]:.0f}"
                    for y, x in path)


def risk_of(frac: float) -> str:
    if frac >= 0.20:
        return "danger"
    if frac >= 0.07:
        return "warning"
    if frac >= 0.01:
        return "watch"
    return "normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3,
                    help="rainfall level index to take risk levels from")
    ap.add_argument("--out", default=str(SOS / "meridian-seed.json"))
    args = ap.parse_args()

    payload = json.loads((WEB / "payload.json").read_text(encoding="utf-8"))
    city = load_city()
    n = int(payload["city"]["n"])
    cell_m = float(payload["city"]["cell_m"])
    box = geo_box(n, cell_m)
    si = max(0, min(args.scenario, len(payload["scenarios"]) - 1))
    scen = payload["scenarios"][si]
    lead = payload["model"]["leads"][-1]
    now = int(time.time() * 1000)

    # ---- the physics truth for this storm, straight off the dashboard -----
    codec = payload["codec"]["depth"]
    dv = decode(scen["leads"][f"fc_{lead}"]["true_depth"], n)
    depth = (codec["max"] * (dv / 255.0) ** codec["gamma"]).astype(np.float32)

    rows = {r["district"]: r for r in risk.district_table(city, depth)}
    n_warn = sum(1 for r in rows.values() if r["exposed_frac"] >= 0.07)
    print(f"storm {scen['total_mm']:.0f} mm, +{lead} h; "
          f"{n_warn} districts at warning or worse")

    # ---- zones ------------------------------------------------------------
    districts = city["districts"]
    zones = []
    for k, name in enumerate(city["district_names"]):
        m = districts == k
        if m.sum() < 8:
            continue
        pts = outline(m, n)
        if not pts:
            continue
        ys, xs = np.nonzero(m)
        lx, ly = cell_to_svg(ys.mean(), xs.mean(), n)
        r = rows.get(name, {})
        zones.append({"id": f"z{k}", "name": name,
                      "risk": risk_of(float(r.get("exposed_frac", 0.0))),
                      "pts": pts, "lx": round(lx, 1), "ly": round(ly, 1)})
    print(f"  {len(zones)} zones outlined")

    # ---- gauges: three channel sites, ordered downstream, plus rainfall ----
    acc, chan, sea = city["acc"], city["channels"], city["sea"]
    cand = np.argwhere(chan & ~sea)
    cand = cand[np.argsort(-acc[cand[:, 0], cand[:, 1]])]
    sites, used = [], []
    for y, x in cand:
        if len(sites) >= 3:
            break
        if all((y - uy) ** 2 + (x - ux) ** 2 > 45 ** 2 for uy, ux in used):
            sites.append((int(y), int(x)))
            used.append((int(y), int(x)))
    gauges = []
    for i, (y, x) in enumerate(sites):
        d = int(districts[y, x])
        where = city["district_names"][d] if d >= 0 else "the estuary"
        bank = float(city["chan_d"][y, x]) or 2.0
        lvl = float(depth[y, x]) + bank * 0.45      # stage = bankfull + flood
        gauges.append({"id": f"g{i+1}", "name": f"Channel at {where}",
                       "level": round(lvl, 2), "warn": round(bank * 0.95, 2),
                       "danger": round(bank * 1.35, 2),
                       "prev": round(max(0.0, lvl - 0.6), 2), "unit": "m"})
    gauges.append({"id": "g-rain", "name": "Rainfall, storm total",
                   "level": round(float(scen["total_mm"]), 1), "warn": 90.0,
                   "danger": 130.0,
                   "prev": round(float(payload["scenarios"][max(0, si - 1)]["total_mm"]), 1),
                   "unit": "mm"})

    # ---- shelters: the sited facilities, plus the control centre ----------
    shelters = {}
    ctl = payload.get("control")
    if ctl:
        lat, lon = cell_to_ll(ctl["yx"][0], ctl["yx"][1], n, box)
        sx, sy = cell_to_svg(ctl["yx"][0], ctl["yx"][1], n)
        shelters["shelters/s-control"] = {
            "name": ctl["name"], "zone": ctl["district"], "capacity": 0,
            "occupancy": 0, "status": "open",
            "contact": f"Duty officer · dispatch and coordination · "
                       f"{lat:.4f}N {abs(lon):.4f}W",
            "x": round(sx, 1), "y": round(sy, 1)}
    rng = np.random.default_rng(91)
    for i, f in enumerate(payload.get("facilities", [])):
        if f["kind"] not in ("shelter", "hospital"):
            continue
        y, x = f["yx"]
        sx, sy = cell_to_svg(y, x, n)
        d = int(districts[y, x])
        zone = city["district_names"][d] if d >= 0 else "—"
        cap = 320 if f["kind"] == "shelter" else 180
        occ = int(cap * float(rng.uniform(0.05, 0.98)))
        wet = float(depth[y, x])
        shelters[f"shelters/s{i+1}"] = {
            "name": f["name"], "zone": zone, "capacity": cap,
            "occupancy": min(occ, cap),
            "status": "closed" if wet > 0.30 else
                      ("full" if occ >= cap else "open"),
            "contact": ("Ward warden" if f["kind"] == "shelter"
                        else "Emergency department"),
            "x": round(sx, 1), "y": round(sy, 1)}
    print(f"  {len(shelters)} shelters and facilities")

    # ---- alerts, written from the warnings the model produced -------------
    tti = {t["district"]: t for t in scen.get("time_to_impact", [])}
    worst = sorted(rows.values(), key=lambda r: -r["exposed_frac"])[:4]
    named = [r["district"] for r in worst if r["exposed_frac"] >= 0.07]
    alerts = {}
    if named:
        soon = sorted((tti[d] for d in named if d in tti),
                      key=lambda t: t["hours"])
        lead_txt = (f"The first to go is {soon[0]['district']}, impassable in "
                    f"about {soon[0]['hours']} hours. " if soon else "")
        alerts["alerts/a-flood"] = {
            "level": "warning" if worst[0]["exposed_frac"] < 0.20 else "evacuate",
            "title": f"Flood warning — {scen['total_mm']:.0f} mm storm over Meridian",
            "body": (f"{lead_txt}Move to higher ground now if you are in the "
                     f"districts listed. Keep documents in a waterproof bag, "
                     f"take medication with you, and tell the control room you "
                     f"are moving so nobody is searched for twice. Roads under "
                     f"30 cm of water stall a car — do not drive through them."),
            "zones": named, "ts": now, "by": "system"}
    if ctl:
        alerts["alerts/a-control"] = {
            "level": "watch",
            "title": f"{ctl['name']} is on air",
            "body": (f"Dispatch is running from {ctl['district']}, "
                     f"{ctl['hand_m']:.1f} m above the drainage line — it is "
                     f"sited to stay dry through this storm. Report anything "
                     f"you can see from where you are: blocked roads, water "
                     f"depth, people who need help moving."),
            "zones": [], "ts": now - 1800_000, "by": "system"}

    # ---- a few citizen reports, placed on households actually in water ----
    reg = payload.get("registry", {}).get("households", [])
    inwater = [h for h in reg if depth[h["y"], h["x"]] > 0.15]
    inwater.sort(key=lambda h: -h["lc"])
    reports, samples = {}, inwater[:5]
    kinds = [("Flooded road", "high", "Water across the road, about {cm} cm. "
                                      "Two cars have already turned back."),
             ("Property flooded", "high", "Water is inside the ground floor, "
                                          "roughly {cm} cm and still rising."),
             ("Need help moving", "critical", "{n} of us here and one cannot "
                                              "walk out unaided."),
             ("Blocked drain", "medium", "The drain on the corner is blocked, "
                                         "water is backing up fast."),
             ("Flooded road", "medium", "Passable but only just — about {cm} cm "
                                        "at the dip.")]
    for i, h in enumerate(samples):
        lat, lon = cell_to_ll(h["y"], h["x"], n, box)
        typ, sev, tpl = kinds[i % len(kinds)]
        reports[f"reports/r-seed{i+1}"] = {
            "uid": "seed", "type": typ, "severity": sev,
            "zone": city["district_names"][h["d"]],
            "note": tpl.format(cm=int(100 * depth[h["y"], h["x"]]), n=h["n"])
                    + f" ({h['code']})",
            "lat": round(lat, 5), "lng": round(lon, 5),
            "ts": now - (i + 1) * 900_000, "status": "open"}
    print(f"  {len(alerts)} alerts, {len(reports)} seeded reports")

    # ---- config -----------------------------------------------------------
    top = max((r["exposed_frac"] for r in rows.values()), default=0.0)
    level = ("evacuate" if top >= 0.20 else "warning" if top >= 0.07
             else "watch" if top >= 0.01 else "normal")
    seed = {
        "meta/config": {
            "level": level,
            "announcement": (f"Meridian Flood Watch is forecasting a "
                             f"{scen['total_mm']:.0f} mm storm. "
                             f"{n_warn} district{'s' if n_warn != 1 else ''} "
                             f"at warning or worse. "
                             f"Check in so the control room knows you are safe."),
            "announcedBy": "system", "announcedAt": now,
            "pin": "", "pinOn": False, "updatedAt": now,
            "rooms": [
                {"id": "general", "name": "General",
                 "desc": "Everyone in Meridian"},
                {"id": "control", "name": "Control room",
                 "desc": "MeridianSOS dispatch and coordination"},
                {"id": "field", "name": "Field crews",
                 "desc": "Ambulance, fire and ward wardens"},
            ]},
        "meta/map": dict(box, river=main_stem(city, n), zones=zones,
                         gauges=gauges, updatedAt=now),
    }
    seed.update(shelters)
    seed.update(alerts)
    seed.update(reports)

    out = SOS / "meridian-seed.json" if args.out is None else __import__("pathlib").Path(args.out)
    out.write_text(json.dumps(seed, indent=1), encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB)")
    print("delete meridian-sos-local/meridian-data.json to re-seed the server")


if __name__ == "__main__":
    main()
