# Meridian City — real-time urban flood forecasting

A 4.2 M-person fictional Atlantic coastal metropolis, a physics-based flood
simulator, and a machine-learning **surrogate** that reproduces the simulator
fast enough to warn people.

Meridian is synthetic. Its parameters — population, coastal setting, low-relief
plain, Mediterranean/Atlantic autumn-winter rainfall regime — are calibrated
against a real North African Atlantic-coast metro so the hydrology behaves
plausibly.

---

## The methodological problem, and why this is built the way it is

The obvious way to build "an AI that predicts flooding" on synthetic data is to
invent a flood-risk formula, label the map with it, and train a model to
recover it. That is **circular**. The model learns your labelling rule, the
accuracy score measures how well it inverted your own arithmetic, and the
number means nothing.

This project avoids that by never labelling anything:

1. **A physics simulator generates the ground truth.** A storage-cell /
   diffusive-wave inundation model (LISFLOOD-FP family) with SCS-CN runoff, a
   sub-grid channel layer, capacity-limited storm sewers and a culverted
   watercourse. It conserves mass to ~1e-15 relative error. Nobody writes down
   where it floods; it falls out of the equations.

2. **The ML model is trained as a fast surrogate of that simulator.** This is
   an established technique — ML emulation of hydrodynamic models is a real
   research area — and it has a real justification here:

   > The simulator takes **10–90 s per storm scenario**. The surrogate
   > produces a full-city, three-lead-time forecast in **well under a second**.
   > The simulator is the better model and always will be. It simply cannot
   > run fast enough, or often enough, to warn anyone.

3. **The model is trained on forecast rainfall, never observed rainfall.**
   Each event carries both: what actually fell, and the degraded forecast that
   would have been available beforehand (amplitude bias, spatial displacement
   and smoothing, all growing with lead time). Training on the observed field
   would leak information a deployed system never has.

The synthetic data stops being a weakness and becomes the point.

---

## Pipeline

```
01_build_city      fractal terrain -> stream-power erosion -> hydrology
                   -> bury the historical watercourse -> urban form
02_simulate_events storms -> physics simulation -> hourly depth rasters
03_train           features -> gradient boosting -> honest verification
04_demo            forecast an unseen storm -> district warnings
```

```bash
python scripts/01_build_city.py
python scripts/02_simulate_events.py --n 100
python scripts/03_train.py
python scripts/04_demo.py
```

Requires `numpy scipy matplotlib pandas scikit-learn xgboost` (see
`requirements.txt`). Nothing needs a GPU or network access.

---

## The city

25.6 × 25.6 km at 100 m resolution (655 km², 256×256 cells), 4.2 M people.

**Terrain** is not fractal noise. Raw fBm over a coastal gradient produces
dead-straight parallel drainage, because a smooth regional slope dominates
local relief and D8 routing always picks due-west. Meridian runs 80 iterations
of stream-power landscape evolution (`dz/dt = U - K·A^m·S^n` plus hillslope
diffusion) so the valley network **erodes itself** into a properly dendritic,
sinuous form.

**Urban form** grows from an accessibility-potential surface with
negative-exponential distance decay (Clark's law — a Gaussian gives an
implausibly compact city ringed by empty land). Result: ~47% built-up, peak
density ~33,000 people/km².

**The equity mechanism is structural, not asserted.** Informal settlements are
sited by sampling weighted toward the lowest ground near watercourses — land
that is cheap precisely because it floods — and carry a fraction of the
drainage:

| | median HAND | storm-drain capacity |
|---|---|---|
| CBD | 3.0 m | 27.5 mm/h |
| informal settlements | 1.3 m | 2.4 mm/h |

Nothing in the model is told this matters. It emerges from the simulation.

**The Ghost Channel.** A watercourse that ran through what is now the city
centre was culverted during 20th-century expansion. It is a shallow
levelled-over depression with no open channel — only a 42 m³/s pipe. It
behaves perfectly in ordinary storms and surcharges catastrophically in
extreme ones, flooding a corridor that carries no watercourse on any map. It
is provided to the model as a feature (`dist_ghost`); what the model has to
learn is the *threshold*.

---

## The simulator

| component | treatment |
|---|---|
| runoff generation | SCS curve number, cumulative, per land use |
| storm sewers | per-cell mm/h capacity × maintenance, **throttled by a citywide trunk capacity** |
| channels | sub-grid: in-bank water conveyed at up to bankfull discharge, above that it overtops |
| surface routing | Manning flux on the water-surface-elevation gradient, volume-limited |
| boundary | the ocean is an open sink |

Two of these were added after the first version produced physically wrong
behaviour, and both matter:

**Sub-grid channels.** At 100 m resolution, treating a whole cell as channel
gives a 15 m-wide watercourse the conveyance of a 100 m one. Water drained
*off* the floodplain instead of overtopping onto it, and informal settlements
— which sit closest to channels — came out flooding *less* than suburbs. That
inverted the entire premise. Bankfull capacity from downstream hydraulic
geometry fixed it.

**Trunk-sewer saturation.** Individual inlets have local capacity, but once
the trunk network saturates, downstream surcharge propagates back upstream and
inlets stop discharging regardless. This threshold is why flood extent is
sharply non-linear in rainfall — and why a linear model cannot reproduce it:

| severity | rainfall | sewer saturated | land >10 cm | people affected | informal | CBD |
|---|---|---|---|---|---|---|
| light | 7 mm | 0 h | 0.00% | 0 | 0.0% | 0.0% |
| moderate | 27 mm | 0 h | 0.03% | 1,945 | 0.3% | 0.0% |
| heavy | 82 mm | 5.1 h | 2.38% | 45,225 | 5.3% | 2.5% |
| extreme | 185 mm | 10.3 h | 16.14% | 807,790 | 52.9% | 17.5% |

---

## Features

34 features per cell, in three groups.

**Static** — elevation, HAND, slope, TWI, curvature, log upstream area,
distance to channel / coast / ghost channel / inland, imperviousness, curve
number, drain capacity, Manning n, channel depth, land use.

**Dynamic** — forecast rainfall at 1/3/6 h, observed rainfall over the past
1/3/6 h, cumulative event rainfall, current water depth.

**Routed** — the same rainfall fields accumulated down the D8 flow network.
What floods a cell is not the rain that falls on it but the rain falling
anywhere upstream. Kept as log **totals**, not catchment means: the mean of a
spatially smooth rain field is almost exactly the local value and carries no
information, while the total is proportional to discharge and scales with
catchment area. Univariate AUC against +6 h flooding:

| | AUC |
|---|---|
| `rain_fc_6h` (local) | 0.637 |
| `up_fc_6h` (routed) | **0.781** |

**Global** — citywide rainfall rate and mean depth, broadcast to every cell.
These let the model see trunk-sewer saturation, which is a system-level
threshold no local feature can express.

---

## Verification

Held out by **event**, never by cell — cells within one event are strongly
spatially correlated, so a random cell split leaks neighbours between train
and test and inflates every score. Cells are subsampled for *training* only;
evaluation runs on the full raster at the true class balance.

Scored with the contingency metrics the flood-forecasting community uses, not
accuracy. Under 2% of cells flood, so "never floods" scores 98% accuracy and
CSI 0.

Against three baselines:

- **persistence** — depth stays as it is now. The bar any nowcast must clear,
  and a genuinely strong one at short lead times.
- **HAND threshold** — the standard static GIS susceptibility proxy, which
  takes no account of the storm at all.
- **no-routing ablation** — the same model without upstream-accumulated
  rainfall, isolating what catchment routing is worth.

### Results

39 simulated storms — 27 train / 6 validation / 6 test. Every figure below is
on **unseen storms**, scored over the full raster at the true class balance.

| lead | model | CSI | POD | FAR | ROC-AUC |
|---|---|---|---|---|---|
| **+1 h** | **surrogate** | **0.514** | 0.933 | 0.467 | 0.999 |
| | persistence | 0.409 | 0.799 | 0.544 | 0.964 |
| | HAND proxy | 0.045 | 0.454 | 0.952 | 0.810 |
| **+3 h** | **surrogate** | **0.513** | 0.692 | 0.334 | 0.995 |
| | persistence | 0.279 | 0.445 | 0.571 | 0.887 |
| | HAND proxy | 0.065 | 0.403 | 0.928 | 0.813 |
| **+6 h** | **surrogate** | **0.431** | 0.684 | 0.462 | 0.984 |
| | persistence | 0.130 | 0.190 | 0.709 | 0.747 |
| | HAND proxy | 0.092 | 0.381 | 0.892 | 0.828 |
| | no-routing ablation | 0.423 | 0.675 | 0.469 | 0.984 |

### Findings

**The surrogate beats persistence at every lead, and the margin grows with
lead time** — 1.3x the CSI at +1 h, 3.3x at +6 h. That shape is the right one:
at +1 h "the water stays where it is" is genuinely hard to beat, and the model
earns its keep further out, which is where a warning is actually useful.

**AUC is a misleading headline, and the HAND baseline proves it.** HAND scores
ROC-AUC 0.81–0.83, which sounds respectable, while its CSI is 0.05–0.09 and it
raises a false alarm on ~90% of the cells it flags. A static susceptibility map
ranks cells adequately and is still useless as a warning. This is exactly why
the contingency metrics are the headline here and accuracy appears nowhere.

**Catchment routing did not pay off — reported as a negative result.** The
no-routing ablation scores CSI 0.423 against the full model's 0.431 at +6 h: a
gain of 0.008, which is noise at this sample size. The routed feature
`up_fc_6h` *is* the second-strongest feature by gain, but the trees reconstruct
almost the same information from `dist_channel` + `log_acc` + local rainfall
when it is removed. The honest reading is that routing is redundant *at this
spatial rainfall variability* (CV ~0.27); with the spatially concentrated
convective cells real storms produce, local rainfall would be a much worse
proxy for upstream discharge and the routed features should matter more. That
is a prediction the design makes, not a result it has demonstrated.

**What the model actually keys on**, by gain, shifts with lead time — which is
physically sensible:

| lead | top features |
|---|---|
| +1 h | `depth_now`, `up_depth_now`, `dist_channel` — where water already is |
| +3 h | `up_depth_now`, `dist_channel`, `up_past_3h` — where it is heading |
| +6 h | `dist_channel`, `up_fc_6h`, `chan_depth`, `hand` — terrain plus forecast rain |

At +1 h it is essentially tracking existing water; by +6 h that information has
decayed and it falls back on drainage structure and the rainfall forecast.

---

## Warning products

Depth is not a warning. `meridian/risk.py` turns predicted depth into what an
operations centre acts on:

```
risk = hazard (depth) × exposure (people) × fragility (capacity to cope)
```

Fragility is applied *on top of* the physical hazard and never mixed into it,
so the hazard model stays physically interpretable and the value judgements
stay visible and adjustable. Depth thresholds follow standard flood-hazard
practice (0.10 m disruption / 0.30 m vehicles float / 0.70 m dangerous on foot
/ 1.50 m life-threatening).

Outputs: citywide impact summary, ranked district alert list, and an equity
breakdown — because a citywide average hides exactly the group that needs the
warning most.

---

## Road Quality Index, and the coupling to flooding

MeridianSOS carries a second hazard model: a road network with a 74-variable
Road Quality Index, and a pothole layer calibrated to the project's real
YOLOv8 detector.

**The network is routed, not drawn.** 39 named roads over 374 km, laid out by
least-cost pathfinding across the terrain, so alignments follow valleys, avoid
steep ground and pay a penalty to bridge watercourses. Four classes (highway /
arterial / collector / local) with their own design speeds, widths and
pavement thicknesses.

**RQI uses the specified formula exactly:**

```
RQI = 100 x [ 0.25*visual + 0.20*structural + 0.15*drainage + 0.15*traffic
            + 0.10*safety + 0.10*environment + 0.05*maintenance ]

category   = SUM(weight_i * normalised_i)
normalised = clip((X - X_bad) / (X_good - X_bad), 0, 1)
```

`X_bad > X_good` handles "more is worse" variables (drain blockage, axle load,
pothole count) without a separate sign convention.

**The two hazard models are genuinely coupled, not merely co-displayed.** All
16 DRAINAGE variables are read out of the flood simulator — standing-water
depth and duration, flooding frequency, drainage capacity, catchment area,
runoff coefficient, elevation, longitudinal slope. Measured on the built
network:

| | |
|---|---|
| corr(RQI, flood frequency) | **-0.564** |
| corr(RQI, informal settlement fraction) | **-0.393** |

Roads degrade where the city floods and where neglect concentrates. Nothing
asserts this; it falls out of the two models sharing a city.

**Potholes are calibrated to the real detector.** Radii are drawn from a
lognormal fitted to the p10/p50/p90 of the 109 detections in
`demo_output/potholes.csv`, converted at 0.7 cm/px — a 0.31 m median radius
across 3,816 potholes. Confidence is sampled from the detector's own
confidence distribution, so the map shows what the model would *report*, not
ground truth.

Three things had to be fixed to make the coupling real, and each is worth
knowing because the first two produced plausible-looking nonsense:

1. **The coupling came out backwards** (+0.164). Flood frequency on roads is
   about 0.02/yr, so against a textbook 0-9/yr normalisation every road scored
   ~0.98 and the flood signal vanished. Scales are now set from the observed
   on-road distribution.
2. **Then it was flat** (-0.016), because exposure was averaged along each
   road. A road is closed by its lowest 100 m, not its average — aggregating
   at the 95th percentile is both physically right and what restored the
   signal.
3. **Every road graded "fair."** Averaging 74 variables collapses everything
   onto the mean. A per-road `neglect` draw with real variance fixed the
   dynamic range; RQI now spans 42-63 with genuine `poor` roads.

---

## Decision layer

A hazard map is not a decision. Three additions turn the forecast into one.

**Emergency routing** (`meridian/routing.py`). Dijkstra over the road network
minimising *travel time*, not distance. Effective speed is the class design
speed degraded by standing water and by RQI; a road is impassable above
0.30 m, the depth at which a passenger car floats (0.55 m for a fire
appliance). Nine facilities — hospitals on high ground, fire stations spread
for coverage, shelters where the ground stays dry — are sited and snapped to
the network. A worked example, hospital to an incident on the most
flood-exposed road:

| storm | route | time | note |
|---|---|---|---|
| dry | yes | 29.2 min | 16.2 km, 68 potholes on route |
| moderate | yes | 29.2 min | unaffected |
| heavy | yes | 32.3 min | slowed by 21 cm of standing water |
| extreme | **no** | — | no surviving route, even for a fire appliance |

**Ensemble forecasting.** 16 perturbed rainfall forecasts through the
surrogate per scenario, giving a P10-P90 band on people at risk rather than a
single number. This is the argument for the whole architecture made concrete:
16 members cost about 15 seconds of surrogate against roughly 12 minutes of
simulation. The surrogate is not a speed optimisation, it is what makes being
probabilistic affordable.

**Time to impact.** The first hour at which more than 2% of a district is
standing in over 10 cm, ranked. "Canal Side impassable in 3 hours" is the
number an operations centre acts on; a depth map is not.

---

## The operations layer

A depth map tells you where the water is. It does not tell you who is in it,
or whether you can reach them. Four additions close that gap, all of them
live in the dashboard rather than precomputed.

**The MeridianSOS Control Centre** (`meridian/households.py`). Sited
deterministically: population-weighted access, smoothed over ~2.5 km, times a
dryness term in HAND, scored **only over cells that are already on the road
network**. Scoring anywhere and snapping afterwards put the first version on a
dry hilltop with its dispatch point in the valley at HAND 3.1 m — a flood
control centre that floods. `99_selftest.py` now asserts it stays dry.

**A household registry.** 2,200 enrolled households, ~8,300 residents, placed
by the population raster across all 28 districts with a floor per district so
the thin coastal wards do not vanish. Each carries a house code, a roster with
ages and dependency notes, a last check-in time and a contact channel.
Searchable by code, resident name or district. The **At risk** view is the
product that matters: households standing in over 10 cm of predicted water,
sorted by how long they have been silent — 104 of them at 121 mm.

It is a sample, not a census, and the dashboard says so: 2,200 households in a
4.2 M city is an opt-in check-in scheme. Names are drawn from pools and are
not real people. Household size and contact reliability vary by land use, so
informal settlements enrol more heavily and go quiet more often — consistent
with the inequity the simulator already produces physically.

**Selectable roads.** Clicking a road, or picking it from a worst-RQI-first
list, selects the whole segment: its 74 RQI variables, its category scores,
and what the current storm is doing to it — percentage of its length under
water, how many cells exceed the 30 cm float threshold, and its deepest point.

**Routing between any two points.** Click an origin and a destination and the
page solves the path itself. Three solutions are shown every time:

| route | objective |
|---|---|
| fastest | travel minutes, identical to `meridian/routing.py` |
| safest | minutes weighted by exposure to the Route Safety Index |
| physics truth | fastest, but solved on the simulator's depth instead of the forecast's |

The **Route Safety Index** is a per-cell 0-100 score, and deliberately not the
speed factor. It punishes shallow water far harder (exponent 0.45 against
0.6), ignores design speed entirely — a highway under 20 cm of water is fast
*and* dangerous — and triples the pothole penalty once they are submerged and
a driver cannot see them. Without that divergence "safest" would just be
"fastest" wearing a different label.

The third route exists for the reason the rest of this project keeps
re-stating: **the depth regressor reads low.** On a test pair the forecast
reported a comfortable 50-minute drive while the simulator said every road in
was under water. The page leads with that disagreement instead of showing the
friendlier number.

**Two copies of one cost model.** The router is re-implemented in JavaScript
because the user picks the endpoints and there is no server. That is a drift
risk, so `scripts/98_verify_web_routing.py` runs the Python router over the
same PNG-quantised rasters the browser sees and prints the numbers to compare.
They currently agree to three decimal places on travel time, and on cells,
depth, submerged potholes and minimum RQI exactly.

That comparison found a real bug: `routing.py` computed path length as
`len(path) * cell_size`, counting a diagonal step as 100 m when it is 141 m.
The D8 network is full of diagonals, so reported journeys were up to 41% short
— 7.2 km for a route that is 8.3 km on the ground. Fixed, and `99_selftest.py`
now asserts diagonal weighting.


---

## Honest limitations

- Meridian is synthetic. The model is calibrated to a simulator, not to
  reality. Transferring this to a real city means re-fitting on real flood
  observations; the *method* transfers, the weights do not.
- The simulator is a diffusive-wave approximation. It omits pressurised sewer
  hydraulics, sediment, debris blockage, tides and storm surge.
- Rainfall has spatial CV ≈ 0.27. Real convective events are more spatially
  concentrated, which would make catchment routing matter more than it does
  here.
- Flood extent depends strongly on the trunk-sewer capacity parameter, which
  is chosen, not measured.
- `vol_surcharge_m3` accumulates per substep and double-counts water that
  stays put; treat it as a pressure indicator, not a volume.
- The 74 RQI variables are synthetic. They are *derived* where the city model
  knows the answer and *sampled* where it does not, and the sampled ones are
  conditioned to stay mutually consistent — but only the drainage category is
  grounded in simulated physics. The RQI weights themselves are a specified
  policy choice, not an empirical fit.
- Pothole locations are simulated. Only their size distribution and detector
  confidence come from the real YOLOv8 output.
- The registry is generated. Placement follows the population raster, but
  names, ages, household sizes and check-in times are sampled, not observed.
- Meridian models 375 km of *named* road — arterials and above, not the full
  street grid. A household's road access therefore reads long (median 900 m).
  A real deployment would route the last few hundred metres over a complete
  street graph; here every endpoint snaps to the modelled network.
- The Route Safety Index is a specified policy, not an empirical fit. Its
  exponents encode a judgement about how dangerous shallow water is, and that
  judgement is not validated against accident data — there is none.

---

## Layout

```
meridian/
  config.py            all parameters, land-use table
  worldgen/            terrain, hydrology, land use, city assembly
  sim/                 storm generator, flood simulator
  ml/                  features, dataset assembly, metrics
  roads.py             road network, RQI specification and scoring
  rqi.py               synthesis of the 74 RQI variables, pothole placement
  routing.py           emergency routing, facility siting
  risk.py              depth -> human impact, equity reporting
  viz.py               shared plotting
scripts/               01_build_city, 02_simulate_events, 03_train, 04_demo,
                       05_figures, 06_export_web, 07_export_interactive,
                       99_selftest
outputs/               figures, trained models, evaluation results
```
