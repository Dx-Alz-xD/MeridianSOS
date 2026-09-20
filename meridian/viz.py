"""Shared plotting helpers for Meridian City."""
from __future__ import annotations
import numpy as np
import matplotlib
from matplotlib.colors import ListedColormap, LightSource, BoundaryNorm

from .config import LU_NAME

LU_COLORS = {
    0: "#1b4f72",   # water
    1: "#2b2b2b",   # cbd
    2: "#8c4a4a",   # dense residential
    3: "#d9a679",   # suburban
    4: "#e03b3b",   # informal
    5: "#6c5b8c",   # industrial
    6: "#3f8f4f",   # park
    7: "#c9d18a",   # agriculture
    8: "#cfc3ad",   # bare scrub
    9: "#5fa8a0",   # wetland
}
LU_CMAP = ListedColormap([LU_COLORS[k] for k in sorted(LU_COLORS)])
LU_NORM = BoundaryNorm(np.arange(-0.5, 10.5, 1.0), LU_CMAP.N)

FLOOD_CMAP = ListedColormap(
    ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#3f007d", "#54278f"])


def hillshade(dem, sea, cell=100.0, az=315, alt=42, exag=9):
    ls = LightSource(azdeg=az, altdeg=alt)
    return ls.hillshade(np.where(sea, 0, dem), vert_exag=exag, dx=cell, dy=cell)


def draw_sea(ax, sea, color="#12395c"):
    ax.imshow(np.ma.masked_where(~sea, np.ones_like(sea, dtype=float)),
              cmap=ListedColormap([color]), interpolation="nearest")


def lu_legend_handles():
    from matplotlib.patches import Patch
    return [Patch(facecolor=LU_COLORS[k], label=LU_NAME[k].replace("_", " "))
            for k in sorted(LU_COLORS)]


def clean(ax, title=None, fs=11):
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)
