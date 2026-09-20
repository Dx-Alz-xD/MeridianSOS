"""Emergency routing over the Meridian road network.

Turns two hazard maps into a decision: given predicted flood depth and the
Road Quality Index, what is the fastest surviving route from a station to an
incident, and when does it stop existing?

Cost is travel TIME, not distance, because that is what an ambulance
dispatcher is actually optimising. Time per cell is length / effective speed,
where effective speed is the road class design speed degraded by:

  standing water  a vehicle slows sharply in water and stops entirely at
                  30 cm, where most cars float and stall
  road quality    a road at RQI 40 is not driven at its design speed
  potholes        a hazard multiplier, and the reason a "passable" flooded
                  road is still dangerous: the potholes are invisible under
                  the water

IMPASSABLE_DEPTH_M is the single most consequential constant here. 0.30 m is
the standard threshold at which passenger vehicles lose traction and float;
emergency vehicles manage more, which is why `heavy` raises it.
"""
from __future__ import annotations
import heapq
import numpy as np

IMPASSABLE_DEPTH_M = 0.30          # cars float
IMPASSABLE_DEPTH_HEAVY_M = 0.55    # fire appliance / high-clearance

NBRS = [(-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142)]


def speed_factor(depth_m: float, rqi: float, limit: float) -> float:
    """Fraction of design speed actually achievable. 0 means impassable."""
    if depth_m >= limit:
        return 0.0
    # water slows a vehicle steeply well before it stops it
    w = max(0.0, 1.0 - (depth_m / limit) ** 0.6)
    # RQI 100 -> full speed, RQI 30 -> about half
    q = 0.35 + 0.65 * np.clip((rqi - 20.0) / 70.0, 0.0, 1.0)
    return float(np.clip(w * q, 0.0, 1.0))


def build_graph(city: dict, net: dict) -> dict:
    """Per-cell road attributes used by the router."""
    n = city["dem"].shape[0]
    road_id = net["road_id"]
    rqi = np.full((n, n), np.nan, np.float32)
    design_kmh = np.zeros((n, n), np.float32)
    pothole = np.zeros((n, n), np.float32)
    from .roads import ROAD_CLASSES
    for s in net["segments"]:
        ys = np.array([c[0] for c in s["cells"]])
        xs = np.array([c[1] for c in s["cells"]])
        rqi[ys, xs] = s["rqi"]
        design_kmh[ys, xs] = ROAD_CLASSES[s["cls"]]["speed"]
        for p in s["potholes"]:
            pothole[p["y"], p["x"]] += 1
    # widened cells of arterials/highways inherit their segment values
    m = (road_id >= 0) & np.isnan(rqi)
    if m.any():
        ys, xs = np.nonzero(m)
        for y, x in zip(ys, xs):
            sid = int(road_id[y, x])
            seg = net["segments"][sid]
            rqi[y, x] = seg["rqi"]
            design_kmh[y, x] = ROAD_CLASSES[seg["cls"]]["speed"]
    return {"road_id": road_id, "rqi": rqi, "design_kmh": design_kmh,
            "pothole": pothole, "on_road": road_id >= 0}


def route(graph: dict, depth: np.ndarray, start: tuple, goal: tuple,
          cell_m: float = 100.0, heavy: bool = False) -> dict:
    """Least-TIME path along roads. Returns path, minutes, and diagnostics."""
    on = graph["on_road"]
    rqi = graph["rqi"]
    kmh = graph["design_kmh"]
    limit = IMPASSABLE_DEPTH_HEAVY_M if heavy else IMPASSABLE_DEPTH_M
    n, m = on.shape

    if not on[start] or not on[goal]:
        return {"ok": False, "reason": "endpoint not on the road network"}

    INF = float("inf")
    dist = np.full((n, m), INF)
    prev = np.full((n, m, 2), -1, np.int32)
    dist[start] = 0.0
    pq = [(0.0, start[0], start[1])]
    while pq:
        d, y, x = heapq.heappop(pq)
        if (y, x) == goal:
            break
        if d > dist[y, x]:
            continue
        for dy, dx, w in NBRS:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < n and 0 <= nx < m) or not on[ny, nx]:
                continue
            f = speed_factor(float(depth[ny, nx]), float(rqi[ny, nx]), limit)
            if f <= 1e-6:
                continue                       # flooded out
            v = max(kmh[ny, nx] * f, 3.0)      # km/h
            minutes = (w * cell_m / 1000.0) / v * 60.0
            nd = d + minutes
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                prev[ny, nx] = (y, x)
                heapq.heappush(pq, (nd, ny, nx))

    if not np.isfinite(dist[goal]):
        return {"ok": False, "reason": "no surviving route"}

    path, cur = [], goal
    while cur != start:
        path.append(cur)
        p = prev[cur[0], cur[1]]
        if p[0] < 0:
            break
        cur = (int(p[0]), int(p[1]))
    path.append(start)
    path.reverse()

    ys = np.array([p[0] for p in path]); xs = np.array([p[1] for p in path])
    d_on = depth[ys, xs]
    # Path length must weight diagonal steps by sqrt(2). Counting every cell
    # as one cell_m understates a diagonal-heavy route by up to 41%, and the
    # D8 network is full of diagonals -- the dashboard was reporting 7.2 km
    # for a journey that is 8.3 km on the ground.
    step = np.hypot(np.diff(ys).astype(float), np.diff(xs).astype(float))
    return {
        "ok": True,
        "path": [[int(a), int(b)] for a, b in path],
        "minutes": float(dist[goal]),
        "km": float(step.sum() * cell_m / 1000.0),
        "max_depth_m": float(d_on.max()),
        "wet_cells": int((d_on > 0.05).sum()),
        "potholes_on_route": int(graph["pothole"][ys, xs].sum()),
        "submerged_potholes": int(graph["pothole"][ys, xs][d_on > 0.10].sum()),
        "min_rqi": float(np.nanmin(graph["rqi"][ys, xs])),
    }


def nearest_road(graph: dict, yx: tuple) -> tuple:
    """Snap a point to the closest cell that is actually on a road."""
    ys, xs = np.nonzero(graph["on_road"])
    d = (ys - yx[0]) ** 2 + (xs - yx[1]) ** 2
    i = int(np.argmin(d))
    return (int(ys[i]), int(xs[i]))


def place_facilities(city: dict, graph: dict, rng) -> list:
    """Hospitals, fire stations and shelters, snapped to the road network.

    Sited the way a city would: hospitals central and on high ground, fire
    stations spread for coverage, shelters on ground that stays dry.
    """
    n = city["dem"].shape[0]
    hand = city["hand"]
    pop = city["population"]
    land = ~city["sea"]
    out = []

    def pick(mask, score, used, min_sep=28):
        cand = np.argwhere(mask)
        if not len(cand):
            return None
        sc = score[mask]
        order = np.argsort(-sc)
        for j in order[:4000]:
            y, x = cand[j]
            if all((y - u[0]) ** 2 + (x - u[1]) ** 2 > min_sep ** 2 for u in used):
                return (int(y), int(x))
        return None

    used = []
    builtup = land & np.isin(city["landuse"], [1, 2, 3, 5])
    # hospitals: high population access, not in the flood corridor
    hosp_score = pop * np.clip(hand / 6.0, 0, 1)
    for i in range(3):
        p = pick(builtup, hosp_score, used)
        if p:
            used.append(p)
            out.append({"kind": "hospital", "name": f"Meridian General {i+1}"
                        if i else "Meridian General", "yx": nearest_road(graph, p)})
    # fire stations: spread across the built-up area
    for i in range(3):
        p = pick(builtup, pop * rng.random(pop.shape), used, min_sep=45)
        if p:
            used.append(p)
            out.append({"kind": "fire", "name": f"Fire Station {i+1}",
                        "yx": nearest_road(graph, p)})
    # shelters: dry ground near dense population
    shel_score = pop * np.clip(hand / 10.0, 0, 1) ** 2
    for i in range(3):
        p = pick(builtup, shel_score, used, min_sep=40)
        if p:
            used.append(p)
            out.append({"kind": "shelter", "name": f"Shelter {chr(65+i)}",
                        "yx": nearest_road(graph, p)})
    return out
