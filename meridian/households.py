"""The MeridianSOS household registry.

Every other layer in this project is a raster: depth, probability, population
density, road quality. A dispatcher does not evacuate a raster. They call a
house, and the house either answers or it does not.

This module places a registry of enrolled households across the built-up city
and gives each one the things an operations centre actually holds: an address
code, who lives there, how many of them, whether any of them cannot walk out
unaided, and when they last checked in. Coupled with the flood layer this
produces the list that matters during a storm -- households in water that have
not been heard from.

Three honesty notes, because this is synthetic data sitting next to physics:

1. The registry is a SAMPLE, not a census. Meridian holds 4.2 M people; the
   registry holds a few thousand households. It is framed in the dashboard as
   an opt-in check-in scheme, which is what a registry like this really is.
2. Household size and composition are drawn per land use. Informal
   settlements get larger households and worse contact rates, consistent with
   the inequity the simulator already produces physically (`LU_DRAIN`).
3. The names are generated from pools. They are not real people, and the
   pools are deliberately mixed to match a coastal metro rather than to
   assert any one community.

Placement is weighted by the population raster, so the registry is dense where
the city is dense, but every populated district is guaranteed a floor of
entries so no district is invisible on the map.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import LU_NAME

# Land uses that can hold a dwelling. Industrial is included at low weight
# because Meridian's industrial belt has caretaker housing in it.
RESIDENTIAL_LU = (1, 2, 3, 4, 5)

# Relative registry enrolment by land use. The CBD is mostly offices, so it
# holds fewer households per resident; informal settlements enrol heavily
# because the scheme is targeted there.
ENROL_WEIGHT = {1: 0.45, 2: 1.00, 3: 0.95, 4: 1.35, 5: 0.25}

# (mean, sd) household size by land use, clipped to [1, 11].
HH_SIZE = {1: (2.1, 1.0), 2: (3.6, 1.5), 3: (3.9, 1.4),
           4: (5.4, 2.0), 5: (2.6, 1.2)}

# Probability a household has gone quiet. Informal settlements have patchier
# coverage and prepaid handsets, so their contact tail is longer.
STALE_BIAS = {1: 0.8, 2: 1.0, 3: 0.85, 4: 1.9, 5: 1.1}

GIVEN_M = [
    "Omar", "Youssef", "Karim", "Idriss", "Hamza", "Bilal", "Nabil", "Tarik",
    "Rachid", "Sofiane", "Anass", "Mehdi", "Ayoub", "Ilyas", "Zakaria",
    "Amine", "Khalid", "Reda", "Othmane", "Badr", "Marwan", "Hicham",
    "Adam", "Noah", "Lucas", "Matheo", "Thomas", "Victor", "Rui", "Tiago",
    "Daniel", "Samir", "Farid", "Jamal", "Aziz", "Mourad",
]
GIVEN_F = [
    "Yasmine", "Salma", "Nour", "Imane", "Hajar", "Meryem", "Sara", "Lina",
    "Ghita", "Zineb", "Kenza", "Rim", "Aya", "Malak", "Douae", "Chaimae",
    "Hind", "Asma", "Widad", "Soukaina", "Amina", "Khadija", "Naima",
    "Clara", "Ines", "Manon", "Sofia", "Elena", "Beatriz", "Carla",
    "Leila", "Dounia", "Rania", "Samira", "Fatima", "Latifa",
]
SURNAME = [
    "Belkacem", "El Amrani", "Bennani", "Cherkaoui", "Ouazzani", "Tazi",
    "Benjelloun", "Lahlou", "Sekkat", "El Fassi", "Berrada", "Alaoui",
    "Idrissi", "Naciri", "Sbai", "Chraibi", "Guessous", "Kettani",
    "Amellal", "Boukhris", "Zerouali", "Hakimi", "Mansouri", "Talbi",
    "Ait Ali", "Ouhadi", "Amrouche", "Izri", "Aznag", "Oubella",
    "Moreno", "Ferreira", "Da Costa", "Marchand", "Leclerc", "Santos",
]

CHANNELS = ["app", "SMS", "voice call", "ward warden", "radio net"]

# Notes that change what a dispatcher does. Kept blunt on purpose: these are
# the reasons a household cannot simply be told to walk to a shelter.
MOBILITY_NOTES = [
    "uses a wheelchair", "reduced mobility", "oxygen concentrator at home",
    "recovering from surgery", "registered blind", "hearing impaired",
]


def _district_code(name: str, taken: set) -> str:
    """Three-letter district tag for the house code, e.g. Old Port -> OLP."""
    letters = [c for c in name.upper() if c.isalpha()]
    words = [w for w in name.upper().split() if w.isalpha()]
    if len(words) >= 3:
        cand = words[0][0] + words[1][0] + words[2][0]
    elif len(words) == 2:
        cand = words[0][:2] + words[1][0]
    else:
        cand = "".join(letters[:3])
    cand = (cand + "XXX")[:3]
    if cand not in taken:
        return cand
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        alt = cand[:2] + ch
        if alt not in taken:
            return alt
    return cand


def _occupants(rng, size: int, lu: int) -> tuple[list, list]:
    """Build a plausible household roster and its vulnerability flags."""
    sur = SURNAME[rng.integers(len(SURNAME))]
    occ, flags = [], []

    def person(age: int, female: bool | None = None) -> dict:
        if female is None:
            female = bool(rng.random() < 0.5)
        pool = GIVEN_F if female else GIVEN_M
        return {"name": f"{pool[rng.integers(len(pool))]} {sur}", "age": int(age)}

    # head of household, then a partner in most households
    head_age = int(np.clip(rng.normal(44, 13), 19, 88))
    occ.append(person(head_age))
    if size >= 2 and rng.random() < 0.85:
        occ.append(person(int(np.clip(head_age + rng.normal(-2, 6), 18, 90)),
                          female=not occ[0]["name"].split()[0] in GIVEN_F))
    # children
    while len(occ) < size and rng.random() < 0.88:
        occ.append(person(int(np.clip(rng.normal(12, 8), 0, 27))))
    # remaining slots are older relatives sharing the dwelling
    while len(occ) < size:
        occ.append(person(int(np.clip(rng.normal(71, 9), 55, 97))))

    for p in occ:
        if p["age"] <= 3:
            p["note"] = "infant"
            flags.append("infant")
        elif p["age"] >= 75:
            p["note"] = "elderly"
            flags.append("elderly")
        elif rng.random() < 0.045:
            p["note"] = MOBILITY_NOTES[rng.integers(len(MOBILITY_NOTES))]
            flags.append("mobility")
        else:
            p["note"] = ""
    # informal settlements: more crowded, so a higher chance of a dependant
    if lu == 4 and not flags and rng.random() < 0.18:
        occ[-1]["note"] = MOBILITY_NOTES[rng.integers(len(MOBILITY_NOTES))]
        flags.append("mobility")
    return occ, sorted(set(flags))


def build_registry(city: dict, on_road: np.ndarray, rng,
                   n_households: int = 2200) -> dict:
    """Place `n_households` enrolled households across the built-up city.

    `on_road` is the boolean road mask; every household is given the nearest
    road cell as its access point so the router always has a valid endpoint.
    Returns a dict with the household list and the district code table.
    """
    lu = city["landuse"]
    pop = city["population"]
    land = ~city["sea"]
    districts = city["districts"]
    names = list(city["district_names"])

    # ---- where dwellings can go ----------------------------------------
    habitable = land & np.isin(lu, RESIDENTIAL_LU) & (pop > 0.5)
    w = np.zeros(pop.shape, np.float64)
    for code, ew in ENROL_WEIGHT.items():
        m = habitable & (lu == code)
        w[m] = pop[m] * ew
    if w.sum() <= 0:
        raise RuntimeError("no habitable cells found")

    # ---- nearest road cell for every cell, in one pass -------------------
    # distance_transform_edt on the INVERSE of the road mask returns, for each
    # cell, the indices of the closest road cell. Doing this per household
    # with a brute-force scan is ~2200 x 65k distance computations; this is one.
    _, (ry, rx) = ndimage.distance_transform_edt(~on_road, return_indices=True)

    # ---- district code table --------------------------------------------
    taken, dcode = set(), {}
    for k, nm in enumerate(names):
        c = _district_code(nm, taken)
        taken.add(c)
        dcode[k] = c

    # ---- allocate households per district, then place within -------------
    # Proportional to district population, with a floor so every populated
    # district appears on the map. Without the floor the thin coastal
    # districts vanish and the registry looks like it only covers downtown.
    dist_pop = {}
    for k in range(len(names)):
        m = habitable & (districts == k)
        if m.sum() and pop[m].sum() > 0:
            dist_pop[k] = float((w * m).sum())
    if not dist_pop:
        raise RuntimeError("no populated districts")
    total = sum(dist_pop.values())
    floor = 8
    alloc = {k: max(floor, int(round(n_households * v / total)))
             for k, v in dist_pop.items()}

    serial = {k: 0 for k in alloc}
    out = []
    flat_idx = np.arange(pop.size)
    for k, want in alloc.items():
        m = (habitable & (districts == k)).ravel()
        wk = (w.ravel() * m)
        if wk.sum() <= 0:
            continue
        p = wk / wk.sum()
        # Sample WITH replacement: a dense cell legitimately holds several
        # registered households, which is what 190 people in a 100 m cell means.
        picks = rng.choice(flat_idx, size=want, replace=True, p=p)
        for fi in picks:
            y, x = int(fi // pop.shape[1]), int(fi % pop.shape[1])
            luc = int(lu[y, x])
            mu, sd = HH_SIZE.get(luc, (3.2, 1.3))
            size = int(np.clip(round(rng.normal(mu, sd)), 1, 11))
            occ, flags = _occupants(rng, size, luc)
            serial[k] += 1
            # Contact recency: a lognormal tail. Most households checked in
            # within the hour; a minority have been silent for most of a day.
            stale = STALE_BIAS.get(luc, 1.0)
            mins = float(np.clip(rng.lognormal(3.6, 1.5) * stale, 1, 1440))
            out.append({
                "code": f"MS-{dcode[k]}-{serial[k]:04d}",
                "y": y, "x": x,
                "ry": int(ry[y, x]), "rx": int(rx[y, x]),
                "d": k, "lu": luc,
                "n": len(occ),
                "occ": [{"name": o["name"], "age": o["age"], "note": o["note"]}
                        for o in occ],
                "flags": flags,
                "lc": round(mins),
                "ch": CHANNELS[int(rng.integers(len(CHANNELS)))],
            })

    out.sort(key=lambda h: h["code"])
    return {
        "households": out,
        "district_codes": {str(k): v for k, v in dcode.items()},
        "n_people": int(sum(h["n"] for h in out)),
        "lu_names": {int(k): LU_NAME[k] for k in RESIDENTIAL_LU},
    }


def place_control_center(city: dict, on_road: np.ndarray) -> dict:
    """Site the MeridianSOS Control Centre.

    Three requirements, in priority order:

      it must not flood      an operations centre that goes under water during
                             the event it exists to manage is worthless, so
                             HAND is a hard-ish gate rather than a preference
      it must reach the city population-weighted access, not geometric centre
      it must be on a road   it dispatches vehicles

    Deterministic: no RNG. The same city always produces the same centre, so
    every route drawn in the dashboard is reproducible.
    """
    hand = city["hand"]
    pop = city["population"]
    land = ~city["sea"]
    lu = city["landuse"]

    # population reachable nearby, smoothed over ~2.5 km
    access = ndimage.gaussian_filter(pop.astype(np.float64), sigma=25.0)
    access /= max(access.max(), 1e-9)

    # dryness: HAND above 8 m is comfortably out of the flood corridor
    dry = np.clip(hand / 8.0, 0, 1)

    # Score ON-ROAD cells only. Scoring anywhere and then snapping to the
    # nearest road puts the centre on a dry hilltop and its actual dispatch
    # point in the valley 300 m below -- which is how the first version of
    # this sited a flood control centre at HAND 3.1 m.
    ok = (land & on_road & np.isin(lu, [1, 2, 3, 5, 6]) & (hand > 6.0))
    if not ok.any():
        ok = land & on_road & (hand > 4.0)
    if not ok.any():
        ok = land & on_road

    score = np.where(ok, access * dry ** 1.5, -1.0)
    cy, cx = (int(v) for v in np.unravel_index(int(np.argmax(score)), score.shape))
    fy, fx = cy, cx
    return {
        "kind": "control",
        "name": "MeridianSOS Control Centre",
        "yx": [cy, cx],
        "site_yx": [int(fy), int(fx)],
        "hand_m": float(hand[cy, cx]),
        "elev_m": float(city["dem"][cy, cx]),
        "district": (city["district_names"][int(city["districts"][cy, cx])]
                     if city["districts"][cy, cx] >= 0 else "—"),
    }
