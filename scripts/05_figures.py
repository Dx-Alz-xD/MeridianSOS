"""Regenerate the presentation figures from the current city and models.

  fig_city.png     the city: terrain, drainage, HAND, land use, population, drainage
  fig_flood.png    one extreme event through the physics simulator
  fig_equity.png   who is exposed, and why

Usage:  python scripts/05_figures.py
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from meridian.config import CFG, OUT, LU_NAME
from meridian.worldgen.city import load_city
from meridian.sim.storms import generate_event
from meridian.sim.flood import simulate_event
from meridian import risk, viz


def fig_city(city):
    dem, sea, land = city["dem"], city["sea"], ~city["sea"]
    sh = viz.hillshade(dem, sea)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 11.6))

    ax = axes[0, 0]
    ax.imshow(sh, cmap="gray")
    im = ax.imshow(np.ma.masked_where(sea, dem), cmap="terrain", alpha=.62,
                   vmin=0, vmax=190)
    viz.draw_sea(ax, sea); plt.colorbar(im, ax=ax, shrink=.78, label="m")
    viz.clean(ax, "Terrain (80 iters stream-power erosion)")

    ax = axes[0, 1]
    ax.imshow(sh, cmap="gray"); viz.draw_sea(ax, sea)
    ax.imshow(np.ma.masked_where(~city["channels"], np.log10(city["acc"] + 1)),
              cmap="cool", vmin=1.8, vmax=4.9)
    gy, gx = np.nonzero(city["ghost"])
    ax.scatter(gx, gy, s=2.2, c="#ff2d55", marker="s", label="buried watercourse")
    ax.legend(loc="lower right", fontsize=8, framealpha=.9)
    viz.clean(ax, "Drainage network + Ghost Channel")

    ax = axes[0, 2]
    im = ax.imshow(np.ma.masked_where(sea, np.minimum(city["hand"], 20)),
                   cmap="YlGnBu_r")
    viz.draw_sea(ax, sea); plt.colorbar(im, ax=ax, shrink=.78, label="m")
    viz.clean(ax, "HAND — height above nearest drainage")

    ax = axes[1, 0]
    ax.imshow(sh, cmap="gray", alpha=.55)
    ax.imshow(city["landuse"], cmap=viz.LU_CMAP, norm=viz.LU_NORM, alpha=.88,
              interpolation="nearest")
    cy, cx = city["cbd"]; ax.plot(cx, cy, "*", c="gold", ms=17, mec="k", mew=.7)
    ax.legend(handles=viz.lu_legend_handles(), loc="center left",
              bbox_to_anchor=(1.01, .5), fontsize=8.5, frameon=False)
    viz.clean(ax, "Land use (★ CBD)")

    ax = axes[1, 1]
    pop = city["population"]
    im = ax.imshow(np.ma.masked_where(sea | (pop < .5), pop * 100), cmap="inferno",
                   vmax=np.percentile(pop[pop > .5] * 100, 99.3))
    viz.draw_sea(ax, sea); plt.colorbar(im, ax=ax, shrink=.78, label="people / km²")
    viz.clean(ax, f"Population — {pop.sum()/1e6:.2f} M")

    ax = axes[1, 2]
    im = ax.imshow(np.ma.masked_where(sea, city["drain_mm_h"]), cmap="BrBG")
    viz.draw_sea(ax, sea); plt.colorbar(im, ax=ax, shrink=.78, label="mm/h")
    iy, ix = np.nonzero(city["landuse"] == 4)
    ax.scatter(ix, iy, s=.7, c="#ff2d55", alpha=.55)
    viz.clean(ax, "Storm-drain capacity (red = informal)")

    plt.suptitle(f"MERIDIAN CITY — {CFG.grid.extent_km:.0f}×{CFG.grid.extent_km:.0f} km, "
                 f"{pop.sum()/1e6:.1f} M people, Atlantic coast", fontsize=15, y=.995)
    plt.tight_layout()
    plt.savefig(OUT / "fig_city.png", dpi=104, bbox_inches="tight")
    plt.close(fig)
    print("  fig_city.png")


def fig_flood(city, ev, sim):
    sea, land = city["sea"], ~city["sea"]
    sh = viz.hillshade(city["dem"], sea)
    hm = sim["h_max"]
    fig = plt.figure(figsize=(17.5, 10.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1])

    ax = fig.add_subplot(gs[0, :2])
    ax.imshow(sh, cmap="gray")
    ax.imshow(city["landuse"], cmap=viz.LU_CMAP, norm=viz.LU_NORM, alpha=.28,
              interpolation="nearest")
    viz.draw_sea(ax, sea)
    im = ax.imshow(np.ma.masked_where((hm < .05) | sea, hm), cmap="turbo",
                   norm=LogNorm(vmin=.05, vmax=max(hm.max(), .6)))
    plt.colorbar(im, ax=ax, shrink=.82, label="max depth (m)")
    iy, ix = np.nonzero((city["landuse"] == 4) & land)
    ax.scatter(ix, iy, s=.8, c="k", alpha=.35, label="informal settlements")
    viz.clean(ax, f"Peak inundation — {ev['severity']} event, "
                  f"{ev['total_mm']:.0f} mm ({ev['max_point_mm']:.0f} mm max point)", 12)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=.92)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(ev["rain_obs"].sum(0) * ev["dt_h"], cmap="YlGnBu")
    plt.colorbar(im, ax=ax, shrink=.75, label="mm")
    viz.clean(ax, "Event rainfall total", 11)

    ax = fig.add_subplot(gs[0, 3])
    d = (ev["rain_fc"].sum(0) - ev["rain_obs"].sum(0)) * ev["dt_h"]
    im = ax.imshow(d, cmap="RdBu_r", vmin=-60, vmax=60)
    plt.colorbar(im, ax=ax, shrink=.75, label="mm")
    viz.clean(ax, "Forecast error the model\nmust tolerate", 10)

    picks = np.linspace(3, len(sim["frames"]) - 1, 4).astype(int)
    for j, fi in enumerate(picks):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(sh, cmap="gray"); viz.draw_sea(ax, sea)
        fr = sim["frames"][fi]
        ax.imshow(np.ma.masked_where((fr < .05) | sea, fr), cmap="turbo",
                  norm=LogNorm(vmin=.05, vmax=max(hm.max(), .6)))
        hrs = sim["frame_t"][fi] * ev["dt_h"]
        pf = city["population"][(fr > .1) & land].sum()
        viz.clean(ax, f"t = {hrs:.0f} h   {pf/1000:.0f}k people in >10 cm", 10)

    plt.suptitle("MERIDIAN CITY — physics simulation of an extreme rainfall event",
                 fontsize=14, y=.995)
    plt.tight_layout()
    plt.savefig(OUT / "fig_flood.png", dpi=104, bbox_inches="tight")
    plt.close(fig)
    print("  fig_flood.png")


def fig_equity(city, hm):
    eq = risk.equity_report(city, hm)
    eq = [r for r in eq if r["population"] > 5000]
    names = [r["landuse"].replace("_", " ") for r in eq]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.barh(names, [100 * r["exposed_rate"] for r in eq], color="#c0392b")
    ax.axvline(100 * sum(r["exposed"] for r in eq) / sum(r["population"] for r in eq),
               ls="--", c="k", lw=1, label="citywide average")
    ax.set_xlabel("% of residents in >10 cm of water"); ax.legend(fontsize=8)
    ax.set_title("Exposure rate", fontsize=11)

    ax = axes[1]
    ax.barh(names, [r["drain_mm_h"] for r in eq], color="#2980b9")
    ax.set_xlabel("storm-drain capacity (mm/h)")
    ax.set_title("Drainage provision", fontsize=11)

    ax = axes[2]
    ax.barh(names, [r["median_hand_m"] for r in eq], color="#27ae60")
    ax.set_xlabel("median HAND (m) — lower is more flood-prone")
    ax.set_title("Ground they are built on", fontsize=11)

    for a in axes:
        a.grid(axis="x", alpha=.25); a.set_axisbelow(True)
    plt.suptitle("Who floods in Meridian City, and why", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "fig_equity.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("  fig_equity.png")


def main():
    city = load_city()
    print("rendering figures...")
    fig_city(city)
    rng = np.random.default_rng(42)
    for _ in range(3):
        ev = generate_event(CFG, rng, severity="extreme",
                            orography=np.clip(city["dem"], 0, None))
    sim = simulate_event(city, ev["rain_obs"], ev["dt_h"], CFG, record_every=2)
    fig_flood(city, ev, sim)
    fig_equity(city, sim["h_max"])
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
