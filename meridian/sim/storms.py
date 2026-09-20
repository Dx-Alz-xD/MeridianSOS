"""Synthetic rainfall events for Meridian City.

A storm is a coherent rain field advected across the city, not a set of
independent blobs. The field is built once on a periodic canvas (convective
cells embedded in a stratiform background) and then translated at the storm
velocity, so a point on the ground sees intermittent bursts as successive
cells pass overhead -- the actual structure of Atlantic/Mediterranean
autumn-winter systems, and a large part of why urban flooding is so unevenly
distributed.

Why a periodic canvas: at a realistic 30 km/h a storm crosses 8 km per 15-min
step, a third of this 25.6 km domain. Discrete blobs seeded inside the domain
simply leave before they rain. Advecting a periodic field keeps rainfall
coherent and continuous for the whole event at no memory cost.

Two products are produced per event:
  observed  -- what falls (drives the physics simulation)
  forecast  -- what the forecaster had beforehand, degraded with amplitude
               bias, spatial displacement and smoothing that grow with lead
               time

The ML model is trained on the FORECAST, because a forecast is all a real
system has at run time. Training on the observed field would leak information
the deployed model could never have.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndshift

from ..config import Config

# severity -> (n convective cells, catchment-mean event total mm, cell radius frac)
SEVERITY = {
    "light":    (3,  (5.0, 16.0),   (0.05, 0.14)),
    "moderate": (5,  (16.0, 42.0),  (0.05, 0.15)),
    "heavy":    (7,  (42.0, 95.0),  (0.04, 0.14)),
    "extreme":  (10, (95.0, 200.0), (0.035, 0.13)),
}
SEVERITY_WEIGHTS = {"light": 0.32, "moderate": 0.33, "heavy": 0.24, "extreme": 0.11}


def _hyetograph(nt: int, peak_frac: float, sharpness: float) -> np.ndarray:
    """Unit-peak temporal profile: fast rise, slower recession."""
    t = np.linspace(0, 1, nt)
    tp = float(np.clip(peak_frac, 0.08, 0.92))
    a = np.where(t <= tp, (t / tp) ** (1.0 + sharpness),
                 np.exp(-sharpness * 2.2 * (t - tp) / (1 - tp + 1e-9)))
    return a / (a.max() + 1e-9)


def _periodic_cells(big, n_cells, rad_frac, n, rng):
    """Stack of [n_cells, big, big] periodic convective-cell fields."""
    yy, xx = np.mgrid[0:big, 0:big].astype(np.float32)
    out = np.empty((n_cells, big, big), dtype=np.float32)
    for k in range(n_cells):
        cy, cx = rng.uniform(0, big, 2)
        rad = rng.uniform(*rad_frac) * n
        elong = rng.uniform(1.0, 2.2)
        th = rng.uniform(0, np.pi)
        # wrapped (toroidal) distance keeps the canvas seamless under rolling
        dy = np.minimum(np.abs(yy - cy), big - np.abs(yy - cy))
        dx = np.minimum(np.abs(xx - cx), big - np.abs(xx - cx))
        ry = dy * np.cos(th) + dx * np.sin(th)
        rx = -dy * np.sin(th) + dx * np.cos(th)
        out[k] = np.exp(-((ry / (rad * elong)) ** 2 + (rx / rad) ** 2))
    return out


def _make_forecast(rain, rng):
    """Degrade the truth into a plausible operational forecast."""
    nt = rain.shape[0]
    fc = np.empty_like(rain)
    amp_bias = rng.normal(1.0, 0.14)
    dy_end, dx_end = rng.normal(0, 4.5, 2)
    for t in range(nt):
        lead = t / max(nt - 1, 1)
        f = ndshift(rain[t], (dy_end * lead, dx_end * lead), order=1, mode="nearest")
        f = f * amp_bias * (1.0 + rng.normal(0, 0.06 + 0.22 * lead))
        fc[t] = gaussian_filter(f, 0.6 + 1.8 * lead)
    return np.clip(fc, 0.0, None).astype(np.float32)


def generate_event(cfg: Config, rng, severity=None, orography=None) -> dict:
    """Return rain_obs / rain_fc arrays [T, n, n] in mm/h plus metadata."""
    g, sc = cfg.grid, cfg.storm
    n = g.n
    nt = int(sc.event_hours * 60 / sc.timestep_min)
    dt_h = sc.timestep_min / 60.0
    big = 2 * n

    if severity is None:
        severity = str(rng.choice(list(SEVERITY_WEIGHTS),
                                  p=list(SEVERITY_WEIGHTS.values())))
    n_cells, total_rng, rad_frac = SEVERITY[severity]
    n_cells = max(1, int(rng.poisson(n_cells)))

    # --- storm motion -------------------------------------------------------
    bearing = np.deg2rad(rng.normal(sc.bearing_deg_mean, sc.bearing_deg_sd))
    speed_kmh = rng.uniform(*sc.speed_kmh_range)
    step = speed_kmh * 1000.0 / 60.0 * sc.timestep_min / g.cell_size_m
    vx = -step * np.sin(bearing)     # bearing = direction the storm comes FROM
    vy = step * np.cos(bearing)

    # --- periodic field: convective cells + stratiform background ----------
    cells = _periodic_cells(big, n_cells, rad_frac, n, rng)
    strat = gaussian_filter(rng.random((big, big)).astype(np.float32),
                            0.09 * big, mode="wrap")
    strat = (strat - strat.min()) / (np.ptp(strat) + 1e-9)
    strat_w = rng.uniform(0.18, 0.42)

    amps = np.zeros((n_cells, nt), dtype=np.float32)
    for k in range(n_cells):
        life = int(np.clip(rng.uniform(0.25, 0.75) * nt, 4, nt))
        t0 = int(rng.uniform(0, max(nt - life * 0.4, 1)))
        prof = _hyetograph(life, rng.uniform(0.25, 0.60), rng.uniform(1.2, 2.8))
        end = min(nt, t0 + life)
        amps[k, t0:end] = prof[:end - t0] * rng.uniform(0.55, 1.0)

    env = _hyetograph(nt, rng.uniform(0.30, 0.60), rng.uniform(1.0, 2.0))

    # --- advect the canvas over the city ------------------------------------
    rain = np.empty((nt, n, n), dtype=np.float32)
    ar = np.arange(n)
    for t in range(nt):
        rows = (ar + int(round(vy * t))) % big
        cols = (ar + int(round(vx * t))) % big
        win = cells[:, rows][:, :, cols]                 # [k, n, n]
        conv = np.tensordot(amps[:, t], win, axes=(0, 0))
        rain[t] = (1.0 - strat_w) * conv + strat_w * env[t] * strat[rows][:, cols]

    tex = gaussian_filter(rng.random((nt, n, n)).astype(np.float32), (1.2, 3.0, 3.0))
    rain *= (0.85 + 0.3 * tex / 0.5 * 0.5)

    # --- stationary storm-track axis ---------------------------------------
    # Advecting a field across the domain smears rainfall almost uniformly
    # (spatial CV ~0.1), which would make flood location a pure function of
    # terrain and reduce the rainfall input to a scalar. Real events have a
    # preferred band where the storm core tracked. This stationary modulation
    # restores realistic spatial variability, and it is what makes WHERE the
    # storm hit matter to the model, not just how much fell.
    yy2, xx2 = np.mgrid[0:n, 0:n].astype(np.float32)
    ay, ax_ = rng.uniform(0.15, 0.85) * n, rng.uniform(0.10, 0.75) * n
    nvy, nvx = np.cos(bearing + np.pi / 2), np.sin(bearing + np.pi / 2)
    perp = (yy2 - ay) * nvy + (xx2 - ax_) * nvx
    width = rng.uniform(0.16, 0.42) * n
    band = np.exp(-(perp ** 2) / (2.0 * width ** 2))
    rain *= (0.30 + 1.45 * band)[None].astype(np.float32)

    if orography is not None:
        oro = 1.0 + 0.45 * np.clip(orography / max(float(orography.max()), 1e-6), 0, 1)
        rain *= oro[None].astype(np.float32)

    # --- calibrate to a realistic catchment-mean event total ---------------
    target_mm = float(rng.uniform(*total_rng))
    cur_mm = float(rain.mean(axis=(1, 2)).sum() * dt_h)
    rain *= target_mm / max(cur_mm, 1e-6)
    rain = np.clip(rain, 0.0, None).astype(np.float32)

    return {
        "rain_obs": rain,
        "rain_fc": _make_forecast(rain, rng),
        "severity": severity, "n_cells": n_cells,
        "bearing_deg": float(np.rad2deg(bearing)) % 360.0,
        "speed_kmh": float(speed_kmh),
        "total_mm": float(rain.mean(axis=(1, 2)).sum() * dt_h),
        "max_point_mm": float((rain.sum(axis=0) * dt_h).max()),
        "peak_mmh": float(rain.max()),
        "dt_h": dt_h, "nt": nt,
    }
