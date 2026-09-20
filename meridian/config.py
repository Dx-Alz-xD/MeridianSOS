"""
Meridian City - global configuration.

Meridian is a fictional coastal metropolis. Its physical and demographic
parameters are calibrated against a real-world analogue (a large North
African Atlantic-coast metro: ~4.2M people, low-relief coastal plain,
Mediterranean-type rainfall regime with intense autumn/winter convective
storms) so that the simulated hydrology behaves plausibly.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CITY_DIR = DATA / "city"
EVENT_DIR = DATA / "events"
OUT = ROOT / "outputs"
for _d in (DATA, CITY_DIR, EVENT_DIR, OUT):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class GridConfig:
    """Raster grid. 256x256 @ 100 m = 25.6 x 25.6 km = 655 km^2 metro area."""
    n: int = 256                 # cells per side (power of 2 -> CNN friendly)
    cell_size_m: float = 100.0   # metres per cell

    @property
    def extent_km(self) -> float:
        return self.n * self.cell_size_m / 1000.0

    @property
    def area_km2(self) -> float:
        return self.extent_km ** 2

    @property
    def cell_area_m2(self) -> float:
        return self.cell_size_m ** 2


@dataclass
class TerrainConfig:
    """Coastal plain rising inland. Coast runs along the WEST (x=0) edge."""
    seed: int = 20260920
    max_elev_m: float = 210.0     # highest inland ridge
    coast_elev_m: float = 0.0     # sea level at the shore
    plain_fraction: float = 0.42  # fraction of width that is near-flat coastal plain
    roughness: float = 0.82       # fractal persistence (0..1), higher = rougher
    erosion_iters: int = 80       # stream-power landscape-evolution steps
    octaves: int = 7
    dune_ridge_m: float = 7.0     # coastal dune barrier height (traps drainage)
    dune_width_cells: int = 6


@dataclass
class HydroConfig:
    channel_accum_threshold: int = 220   # cells of flow accumulation -> channel
    main_channel_carve_m: float = 9.0    # depth carved for the main river
    trib_carve_m: float = 3.5
    # The "Ghost Channel": a historical watercourse that the city culverted and
    # built over during 20th-century expansion. Invisible in the modern DEM but
    # still the preferential flow path in extreme events. (Modelled on real
    # buried-oued behaviour in North African coastal cities.)
    ghost_channel_enabled: bool = True
    ghost_channel_capacity_m3s: float = 42.0
    # How far the culverted reach sits below the surrounding graded surface.
    # Burying it perfectly flush would stop surface water reaching the culvert
    # at all, so the corridor would never surcharge. In reality the old valley
    # is still a shallow low spot that collects runoff.
    ghost_burial_depth_m: float = 1.2
    # Total trunk-sewer throughput for the whole city. Individual inlets have
    # their own mm/h capacity, but once the trunk network saturates the
    # downstream surcharge propagates back upstream and inlets stop
    # discharging no matter how much local capacity they nominally have.
    # This threshold is why flood extent is strongly non-linear in rainfall,
    # and why a linear model cannot reproduce it.
    sewer_network_capacity_m3s: float = 420.0


@dataclass
class CityConfig:
    """Urban form and demographics."""
    population: int = 4_200_000
    # CBD sits slightly inland of the port, on the north coast stretch
    cbd_xy: tuple = (0.22, 0.38)     # fractional grid coords (x, y)
    port_xy: tuple = (0.06, 0.30)
    airport_xy: tuple = (0.78, 0.80)
    n_informal: int = 5              # informal settlement clusters
    n_industrial: int = 3


@dataclass
class StormConfig:
    """Mediterranean/Atlantic coastal rainfall regime."""
    timestep_min: int = 15
    event_hours: int = 18
    # Storms arrive predominantly from the Atlantic (west/southwest)
    bearing_deg_mean: float = 250.0
    bearing_deg_sd: float = 32.0
    speed_kmh_range: tuple = (8.0, 45.0)   # slow storms flood worst
    peak_intensity_mmh_range: tuple = (6.0, 95.0)


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    hydro: HydroConfig = field(default_factory=HydroConfig)
    city: CityConfig = field(default_factory=CityConfig)
    storm: StormConfig = field(default_factory=StormConfig)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


CFG = Config()

# ---------------------------------------------------------------- land use ---
# code: (name, curve_number, imperviousness, manning_n, base_drain_mm_h)
# CN = SCS curve number (hydrologic soil group B); higher = more runoff.
# base_drain = storm-sewer removal capacity, mm/h. Note the deliberate
# inequity: informal settlements have a fraction of the drainage of the CBD.
LANDUSE = {
    0: ("water",        100, 1.00, 0.035,  0.0),
    1: ("cbd",           94, 0.92, 0.015, 26.0),
    2: ("dense_resid",   90, 0.78, 0.018, 17.0),
    3: ("suburban",      79, 0.45, 0.025, 11.0),
    4: ("informal",      92, 0.70, 0.030,  3.0),
    5: ("industrial",    91, 0.85, 0.016, 13.0),
    6: ("green_park",    61, 0.08, 0.060,  2.0),
    7: ("agriculture",   72, 0.03, 0.055,  0.0),
    8: ("bare_scrub",    77, 0.05, 0.045,  0.0),
    9: ("wetland",       85, 0.02, 0.080,  0.0),
}
LU_NAME = {k: v[0] for k, v in LANDUSE.items()}
LU_CN = {k: v[1] for k, v in LANDUSE.items()}
LU_IMPERV = {k: v[2] for k, v in LANDUSE.items()}
LU_MANNING = {k: v[3] for k, v in LANDUSE.items()}
LU_DRAIN = {k: v[4] for k, v in LANDUSE.items()}

# people per cell at full density, by land use (100x100 m cell)
LU_POP_WEIGHT = {0: 0.0, 1: 55.0, 2: 190.0, 3: 62.0, 4: 240.0,
                 5: 8.0, 6: 2.0, 7: 3.0, 8: 1.0, 9: 0.0}

# Social fragility multiplier: capacity to anticipate, cope with and recover
# from flooding. Higher = more vulnerable for the same water depth.
LU_FRAGILITY = {0: 0.0, 1: 0.55, 2: 0.85, 3: 0.45, 4: 1.00,
                5: 0.50, 6: 0.20, 7: 0.60, 8: 0.30, 9: 0.10}
