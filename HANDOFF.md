# HANDOFF — MeridianSOS flood module

Written for whoever (human or model) picks this up next. `README.md` is the
judge-facing document; **this** one is the working state: what exists, what is
wrong with it, what is half-done, and what to do first.

Read the three "Start here" items before touching anything.

---

## Start here

**1. The model is trained on 39 events. There are 100 on disk.**
The corpus simulation was stopped early during a deadline crunch, the model
was trained on what existed at that moment, and the simulation process then
survived the stop and finished all 100. `data/events/index.json` lists 100
events; `outputs/models/surrogate.pkl` was fitted when only ~39 were present
(its `test_ids` has 6 entries). **Retraining on the full corpus is the single
highest-value action available and requires no new code:**

```bash
python scripts/03_train.py          # ~15-25 min on 100 events
python scripts/07_export_interactive.py --levels 6   # rebuild the dashboard
```

A backup of the 39-event model sits at `outputs/models/surrogate_39ev.pkl.bak`
with its metrics in `outputs/eval_results_39ev.json.bak`, so the old numbers
stay reproducible.

**2. The depth regressor is biased low, and it is safety-critical.**
It is fit on `log1p(depth_cm)` under squared error; back-transforming through
`expm1` underestimates. On one test storm it reported 76k people at risk
against an actual 383k — roughly 5x low. Two places already work around it:

- **Headcounts** come from the classifier as `sum(population x P(flood))`, not
  from the regressor. See `expected_exposure()` in `scripts/07_export_interactive.py`.
- **Routing** is computed on the forecast *and* the physics truth, and the
  dashboard flags disagreements, because at 156 mm and 190 mm the forecast
  reports all three ambulance routes open while the truth says all three are
  blocked.

The regressor still colours the depth map. **Fixing it properly is the top
correctness task** — see "What to do next", item 2.

**3. The repo was pulled mid-session; `README.md` now belongs to the pothole
module.** `origin/main` has moved on (`POTHOLE` -> `SOS CHAT` -> a merge) and
somebody ran the pull into this working tree. The effect on the flood work:
**the flood `README.md` was renamed to `README_local.md`** and the pothole
module's README now sits at the root. `README_local.md` is intact and is where
the flood documentation has continued to be written.

Nothing of the flood module is committed. Decide the merge layout before
committing — see "Repository state"; the `flood/` + `pothole/` split suggested
there would resolve the README collision rather than leaving two files whose
names do not say which is which.

---

## What this is

A physics-based flood simulator for a synthetic 4.2M-person coastal city, plus
a machine-learning **surrogate** that reproduces it fast enough to issue
warnings, plus a road-quality model coupled to the flooding, plus a decision
layer (routing, ensembles, time-to-impact) and an interactive dashboard.

The central design argument, which every part is built to support:

> Labelling a synthetic map with a flood-risk formula and training a model to
> recover it is circular. Instead, a physics simulator generates ground truth
> and the ML model is trained as a fast surrogate of that simulator. The
> simulator is the better model; it simply cannot run fast enough to warn
> anyone.

Do not undermine this. If you add a feature that leaks the answer — for
example training on observed rather than forecast rainfall — the whole
argument collapses.

---

## Layout

```
meridian/
  config.py          149  all parameters, land-use table, severity classes
  worldgen/
    terrain.py       162  fractal noise + stream-power erosion
    hydrology.py     213  priority-flood fill, D8, accumulation, HAND, TWI
    landuse.py       174  urban form, districts, population, informal siting
    city.py          172  orchestrator; buries the Ghost Channel; save/load
  sim/
    storms.py        169  advected periodic rain field + forecast degradation
    flood.py         209  LISFLOOD-FP-style inundation, sub-grid channels
  ml/
    features.py      153  34 features, D8 upstream routing (Router class)
    dataset.py       143  event-level splits, sampling, table assembly
    metrics.py       121  CSI/POD/FAR/bias, ROC-AUC, PR-AUC (no sklearn dep)
  roads.py           358  road network + RQI spec (74 vars) + scoring
  rqi.py             283  RQI variable synthesis, potholes, flood climatology
  routing.py         195  time-based Dijkstra, facility siting
  risk.py            160  depth -> people -> district alerts -> equity
  households.py      300  MeridianSOS registry + control-centre siting
  viz.py              47  shared colormaps and plotting helpers

scripts/
  01_build_city.py         build + cache the city (~27 s)
  02_simulate_events.py    simulate N storms (~25 s/event average)
  03_train.py              train + verify against 3 baselines
  04_demo.py               forecast an unseen storm, print warnings
  05_figures.py            regenerate fig_city / fig_flood / fig_equity
  06_export_web.py         OLD static 3-scenario dashboard (superseded)
  07_export_interactive.py CURRENT dashboard exporter
  98_verify_web_routing.py checks the dashboard's JS router against Python
  99_selftest.py           34 invariant checks — run this after any change

data/city/meridian.npz     2.5 MB  the cached city
data/events/*.npz          336 MB  100 simulated storms
outputs/models/            trained surrogate + backup
outputs/web/index.html     the dashboard (self-contained, ~3.7 MB)
outputs/web/payload.json   cached export payload (see --render-only)
```

Environment: Python 3.14.6, numpy 2.5.3, scipy 1.18.1, xgboost 3.4.1,
scikit-learn 1.9.1, matplotlib 3.11.2, pandas 3.0.6. No GPU, no network.
`torch` is in `requirements.txt` but **is not used** by anything — remove it
or use it (see "Ideas deliberately not taken", item 1).

Run anything with `PYTHONPATH` set to the repo root.

---

## Run order

```bash
python scripts/01_build_city.py                       # 27 s
python scripts/02_simulate_events.py --n 100          # ~40 min
python scripts/03_train.py                            # ~15-25 min at n=100
python scripts/04_demo.py                             # 1 min
python scripts/07_export_interactive.py --levels 6    # ~6 min
python scripts/99_selftest.py                         # 2 min, must pass 34/34
```

`02` supports `--start` to extend an existing corpus without redoing it.
`07` supports `--render-only`, which re-applies the HTML template to
`outputs/web/payload.json` in about a second — use it for any template or copy
change instead of re-running six physics simulations.

---

## Current results

**Surrogate vs baselines**, 39 events (27 train / 6 val / 6 test), scored on
full rasters of held-out storms. Re-run `03_train.py` to refresh these on 100.

| lead | model | CSI | POD | FAR | ROC-AUC |
|---|---|---|---|---|---|
| +1 h | surrogate | 0.514 | 0.933 | 0.467 | 0.999 |
| | persistence | 0.409 | 0.799 | 0.544 | 0.964 |
| | HAND proxy | 0.045 | 0.454 | 0.952 | 0.810 |
| +3 h | surrogate | 0.513 | 0.692 | 0.334 | 0.995 |
| | persistence | 0.279 | 0.445 | 0.571 | 0.887 |
| +6 h | surrogate | 0.431 | 0.684 | 0.462 | 0.984 |
| | persistence | 0.130 | 0.190 | 0.709 | 0.747 |
| | no-routing ablation | 0.423 | 0.675 | 0.469 | 0.984 |

**Road Quality Index** (39 roads, 375 km, 3,816 potholes): RQI 42–63, median
53. corr(RQI, flood frequency) **-0.564**; corr(RQI, informal fraction)
**-0.393**.

**Simulator sanity**: mass conserved to ~1e-15 relative. Flood extent is
sharply non-linear in rainfall (0.00% of land at 18 mm, 15.65% at 190 mm),
with the break at trunk-sewer saturation.

---

## The operations layer (added after the original handoff)

Everything here is live in the dashboard, not precomputed, and all of it is
solved in the browser.

- **`meridian/households.py`** — a 2,200-household registry (~8,300 residents)
  placed by the population raster across all 28 districts with a floor per
  district, plus deterministic siting of the **MeridianSOS Control Centre**.
- **Registry panel** — search by house code, resident name or district; an
  **At risk** view listing households in >10 cm of predicted water sorted by
  how long they have been silent (104 of them at 121 mm).
- **Selectable roads** — click a road or pick it from a worst-RQI-first list;
  shows the 74 RQI variables plus what this storm is doing to that road.
- **Route planner** — click any two points; the page solves fastest, safest
  and physics-truth routes and draws all three.

Three things worth knowing before you change any of it:

**The control centre must be scored on road cells only.** The first version
scored dryness anywhere and snapped to the nearest road afterwards, which put
the centre on a dry hilltop and its dispatch point in the valley at HAND
3.1 m — a flood control centre that floods. It is now sited at HAND 8.7 m and
`99_selftest.py` asserts both the HAND floor and that it stays dry in-storm.

**The Route Safety Index is not the speed factor.** If you make it one,
"safest" collapses into "fastest" and the second route becomes decoration.
It punishes shallow water harder (exponent 0.45 vs 0.6), ignores design speed
— a highway under 20 cm is fast *and* dangerous — and triples the pothole
penalty once they are submerged.

**Report the worst point excluding the endpoints.** Every route between the
same pair crosses both endpoint cells, so a min-over-the-whole-path safety
figure is identical for every objective and can never respond to the one the
user picked. The first version shipped that and it looked like a bug in the
optimiser.

### The two copies of the cost model

`routing.py` is re-implemented in JavaScript in the template, because the user
picks the endpoints and there is no server. **That is a drift risk.**
`scripts/98_verify_web_routing.py` runs the Python router over the same
PNG-quantised rasters the browser sees and prints the numbers, along with a JS
snippet to paste into the page console. Run it after touching either copy.

They currently agree to three decimal places on travel time, and exactly on
cells, max depth, submerged potholes and minimum RQI.

That comparison found a real bug in `routing.py`: path length was
`len(path) * cell_size`, counting a diagonal step as 100 m when it is 141 m.
The D8 network is diagonal-heavy, so journeys were reported up to 41% short —
7.2 km for a route that is 8.3 km. **Fixed**, which means any route distance
quoted from before this change is wrong. `99_selftest.py` now asserts it.

---

## Known issues, in priority order

**1. Depth regressor bias (safety-critical).** Described in "Start here".

**2. Test set is small.** Six test events at n=39. Do not quote the third
decimal of any metric. Retraining on 100 gives ~15 test events.

**3. Catchment routing did not pay off.** The no-routing ablation scores CSI
0.423 against 0.431 — noise. Trees reconstruct it from `dist_channel` +
`log_acc` + local rainfall because the storms are only moderately variable in
space (CV ~0.27). This is reported as a negative result in `README.md`; **do
not quietly delete the ablation to make the model look better.** If you raise
the spatial variability of the storms, re-check whether routing starts paying.

**4. `vol_surcharge_m3` is not a volume.** It accumulates per substep and
double-counts water that stays put. It is a pressure indicator. Either fix it
or rename it; do not report it as a volume.

**5. ~~The dashboard is unverified visually.~~ Done — opened, fixed, extended.**
Serve it, do not open it as a `file://` URL: the browser pane refuses those
but loads `http://localhost:8731` fine. `.claude/launch.json` holds the
config; by hand it is `python -m http.server 8731 --directory outputs/web`.

Two defects were visible on first sight and are fixed: the page had **no
`<meta charset>` and no doctype**, so every `·` and `×` rendered as mojibake,
and there was no `<html>/<head>/<body>` at all. Click, hover, road selection,
registry search and live routing have all now been exercised in a browser.

Still unchecked: **narrow viewports.** The layout was only driven at 1320 px
and wider. The `.ops` grid is `minmax(310px,1fr)` so it should reflow, but
nobody has looked.

**6. `06_export_web.py` is dead code.** Superseded by `07`. Delete it, or keep
it only if you want the simpler static three-scenario page.

**7. `torch` in requirements is unused.**

---

## Traps that already cost time

Every one of these produced plausible-looking output while being wrong. They
are encoded as assertions in `99_selftest.py` — **if you change the world
generator or the simulator, run it.**

| trap | symptom | fix |
|---|---|---|
| Grid borders seeded as drainage outlets | All flow exited the south edge; zero river mouths | Only the ocean is an outlet (`hydrology.fill_depressions`) |
| Explicit diffusion coefficient > 0.25 | Terrain blew up to 28 km elevation, checkerboard forced all flow diagonal | Keep `diffusion <= 0.2` in `terrain.erode_landscape` |
| `np.roll` in the Laplacian | Wraparound streak on the right edge of the DEM | Edge-replicated padding |
| Gaussian urban decay | Implausibly compact city ringed by empty land | Exponential decay (Clark's law) in `landuse.py` |
| Whole cell treated as channel | Informal settlements flooded *less* than suburbs — inverted the premise | Sub-grid bankfull channel layer in `sim/flood.py` |
| Normalising routed rainfall by catchment size | Routed features became duplicates of local ones (corr ~1.0) | Keep log **totals**, not means |
| Fixed 0–1 CSI threshold grid | HAND baseline scored CSI 0.000 and looked useless | Threshold grid from the score's own quantiles |
| Textbook 0–9/yr flood normalisation | Every road scored ~0.98; RQI/flood coupling came out **positive** | Set scales from the observed on-road distribution |
| Averaging flood exposure along a road | Coupling flat at -0.016 | Aggregate at the 95th percentile — a road is closed by its lowest 100 m |
| Averaging 74 variables | Every road graded "fair", no dynamic range | Per-road `neglect` draw with real variance |
| Forecast issued at t0=8 | Storm had already peaked; time-to-impact read "+0h" everywhere | Issue at t0=5 |
| Siting the control centre off-road, then snapping | A dry hilltop with its dispatch point at HAND 3.1 m | Score on-road cells only |
| Route safety = the speed factor | "Safest" and "fastest" returned the same path every time | Weight water and potholes differently from speed |
| Min safety over the whole path | Identical for every objective — endpoints are shared | Exclude the two endpoints |
| Path length as `len(path) * cell_m` | Journeys up to 41% short on a diagonal-heavy D8 network | Weight diagonals by sqrt(2) |
| No `<meta charset>` in the dashboard | Every `·` and `×` rendered as mojibake | Add a doctype and charset |

Two process notes: chaining multiple heredocs with `&&` in one shell command
kept failing to parse on this machine — write one file per command, or use the
editor tools. And `TaskStop` killed the task wrapper but **not** the Python
process underneath, which is why 100 events exist.

---

## What to do next

Ordered by value per hour.

**1. Retrain on 100 events, re-export, re-verify.** No new code. See
"Start here". Expect the metrics to improve and the test set to roughly
triple.

**2. Fix the depth regressor.** Options, cheapest first:
   - Fit a scalar bias correction on the validation set (Duan smearing
     estimator is the standard fix for log-transform retransformation bias).
   - Train on raw depth with a Tweedie or gamma objective instead of
     log-transformed MSE — xgboost supports `reg:tweedie`.
   - Predict depth only on cells the classifier says are wet, which removes
     the zero-inflation that is dragging the fit down.
   Then **remove the routing workaround** and route on the forecast alone, and
   confirm the forecast/truth disagreement shrinks.

**3. ~~Verify the dashboard visually.~~ Done.** Two encoding defects found and
fixed; the operations layer was built on top. What remains is **checking it at
narrow viewports**, which nobody has done.

**4. Calibration / reliability diagram.** Are the 70% probabilities right 70%
of the time? Cheap now that the ensemble exists, and almost no competing
project shows it. Pairs naturally with the ensemble panel.

**5. Merge the repo.** See below.

**6. Spatial holdout.** Hold out a geographic region rather than a set of
storms, and check the model still works there. That distinguishes "learned the
physics" from "memorised this city", which is currently an open question — the
model is only ever tested on unseen *storms*, never unseen *ground*.

**7. Submerged-pothole product.** The data is all present (pothole positions,
per-cell predicted depth) and the cell inspector already flags it. What is
missing is a ranked citywide list: every pothole about to go under water,
sorted by road traffic and depth. That is a concrete deliverable for a roads
department and it is mostly assembly.

---

## Repository state

```
origin  https://github.com/Dx-Alz-xD/MeridianSOS.git
origin/main  0f32926  "POTHOLE"  (parth200714)
local        no commits, no local branches; HEAD points at a non-existent main
```

The remote commit contains the pothole module: `best.pt` (YOLOv8n),
`pothole_measure.py`, `make_split.py`, `train.py`, `training_results.csv`,
`demo_output/` (41 annotated images + `potholes.csv`, 109 detections).
Reported: precision 0.66, recall 0.60, mAP50 0.66, mAP50-95 0.31 on 102
held-out images; median radius error 8%, area error 16%.

That module's `make_split.py` is worth reading — it strips Roboflow
augmentation suffixes and groups by source photo before splitting, which is
the same leak-avoidance discipline used here (split by storm, not by cell).

**Suggested merge layout**, since `README.md` collides:

```
README.md          new top-level: what MeridianSOS is, links to both modules
flood/             everything currently at the repo root
pothole/           everything currently in origin/main
HANDOFF.md         this file
```

Nothing has been committed or pushed. That was deliberate — it was never
asked for.

---

## How the two modules connect

This is real coupling, not a shared slide.

- The 16 DRAINAGE variables of the RQI are read straight out of the flood
  simulator, so road quality degrades where the city floods (corr -0.564).
- Pothole sizes are drawn from a lognormal fitted to the p10/p50/p90 of the
  109 real detections in `demo_output/potholes.csv`, converted at 0.7 cm/px
  (`CM_PER_PX` in `roads.py`), giving a 0.31 m median radius. Detector
  confidence is sampled from the real confidence distribution, so the map
  shows what the model would *report*, not ground truth.
- Routing costs combine flood depth and RQI, and the dashboard flags
  **submerged potholes** — the hazard a driver cannot see.

If you re-run the pothole detector on new imagery, update
`POTHOLE_RADIUS_PX` in `roads.py` from the new `potholes.csv` percentiles.

---

## Ideas deliberately not taken

Recorded so nobody re-derives them.

1. **Deep learning (U-Net / ConvLSTM).** At ~100 events it will very likely
   lose to gradient boosting, and it costs hours to find out. Revisit only
   with a much larger corpus.
2. **Transfer to a real city.** Correct long-term instinct, but real flood
   inventories are exactly the data blocker this whole synthetic approach was
   designed to route around. Needs weeks, not hours.
3. **A chatbot over the data.** Adds surface, not substance.
4. **More dashboards.** One good one beats three.

---

## Things that must stay true

If a change breaks any of these, the change is wrong.

- The ML model never sees observed rainfall as an input — only the degraded
  forecast (`rain_fc`).
- Train/test splits are by **event**, never by cell.
- Evaluation runs on full rasters at the true class balance; subsampling is
  for training only.
- Headline metrics are CSI / POD / FAR, not accuracy. Under 2% of cells flood,
  so "never floods" scores 98% accurate.
- The simulator conserves mass to ~1e-15. `99_selftest.py` checks this.
- Negative and awkward results stay in the write-up: the failed routing
  ablation, the HAND baseline's AUC/CSI gap, the regressor bias, the
  forecast/truth routing disagreement. The honesty is a differentiator, and
  removing it to make a cleaner story makes the project worse.
- Every route in the dashboard is solved on the forecast **and** on the
  physics truth, and disagreements are shown, not hidden. Showing only the
  forecast route is the one failure mode here that could get someone killed.
- The household registry is presented as a sample, never as a census. 2,200
  households in a 4.2 M city is an opt-in scheme; the dashboard says so.
- The JS router and `routing.py` stay in agreement. If you change one, run
  `98_verify_web_routing.py` and change the other.
