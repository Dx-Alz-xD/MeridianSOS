"""Physics-based urban inundation model for Meridian City.

A storage-cell / diffusive-wave scheme in the LISFLOOD-FP family, with a
sub-grid channel layer:

    runoff generation : SCS-CN (cumulative, event-based)
    sewer removal     : per-cell capacity, mm/h, from land use x maintenance
    channel layer     : in-bank water is conveyed downstream at up to the
                        bankfull discharge and does NOT spread laterally;
                        water above bankfull overtops onto the floodplain
    culvert           : the buried watercourse, capacity-limited -> surcharges
    surface routing   : Manning flux on the WSE gradient, volume-limited
    boundary          : the ocean is an open sink

The sub-grid channel layer is essential at 100 m resolution. Treating a whole
cell as channel gives a 15 m-wide watercourse the conveyance of a 100 m one,
so water drains off the floodplain instead of overtopping onto it, which
inverts exactly the low-lying-settlement flooding the model exists to predict.

This simulator IS the ground truth. The ML model never sees these equations;
it sees inputs and the depths that come out, so its task is to learn a fast
approximation of the whole coupled process, not to invert a labelling rule.
"""
from __future__ import annotations
import numpy as np

G = 9.81
D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def downstream_index(fdir: np.ndarray) -> np.ndarray:
    """Flat index of each D8 downstream neighbour (-1 where there is none)."""
    n, m = fdir.shape
    ds = np.full(n * m, -1, dtype=np.int64)
    for k, (dy, dx) in enumerate(D8):
        ys, xs = np.nonzero(fdir == k)
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < n) & (nx >= 0) & (nx < m)
        ds[ys[ok] * m + xs[ok]] = ny[ok] * m + nx[ok]
    return ds


def _surface_flux(wse, dem, hs, manning, dx, dt, cell_area):
    """Volume exchanged with the +x and +y neighbour (m^3), volume-limited."""
    vx = np.zeros_like(hs)
    vy = np.zeros_like(hs)
    for axis, store in ((1, vx), (0, vy)):
        if axis == 1:
            w_a, w_b = wse[:, :-1], wse[:, 1:]
            z_a, z_b = dem[:, :-1], dem[:, 1:]
            h_a, h_b = hs[:, :-1], hs[:, 1:]
            n_ = 0.5 * (manning[:, :-1] + manning[:, 1:])
        else:
            w_a, w_b = wse[:-1, :], wse[1:, :]
            z_a, z_b = dem[:-1, :], dem[1:, :]
            h_a, h_b = hs[:-1, :], hs[1:, :]
            n_ = 0.5 * (manning[:-1, :] + manning[1:, :])

        dh = w_a - w_b                                  # positive: a -> b
        hflow = np.clip(np.maximum(w_a, w_b) - np.maximum(z_a, z_b), 0.0, None)
        slope = dh / dx
        q = (np.power(hflow, 5.0 / 3.0) * np.sqrt(np.abs(slope))
             * np.sign(slope) / np.maximum(n_, 1e-3))   # m^2/s per unit width
        vol = q * dx * dt

        # never move more than a quarter of the donor water in one substep;
        # this is what keeps the explicit scheme stable on steep gradients
        vol = np.clip(vol, -0.25 * h_b * cell_area, 0.25 * h_a * cell_area)
        if axis == 1:
            store[:, :-1] = vol
        else:
            store[:-1, :] = vol
    return vx, vy


def simulate_event(city: dict, rain_mmh: np.ndarray, dt_h: float, cfg,
                   record_every: int = 4, substep_max_s: float = 120.0,
                   substep_min_s: float = 4.0, cfl: float = 0.6,
                   verbose: bool = False) -> dict:
    """Run one rainfall event. Returns recorded depth frames and diagnostics."""
    dx = cfg.grid.cell_size_m
    cell_area = dx * dx
    dem = city["dem"].astype(np.float64)
    sea = city["sea"]
    land = ~sea
    shape = dem.shape
    manning = city["manning"].astype(np.float64)
    cn = np.clip(city["cn"].astype(np.float64), 30.0, 99.0)
    drain_ms = city["drain_mm_h"].astype(np.float64) / 1000.0 / 3600.0
    ghost = city["ghost"]
    ghost_cap = float(cfg.hydro.ghost_channel_capacity_m3s)
    sewer_net_cap = float(cfg.hydro.sewer_network_capacity_m3s)
    sewer_saturated_s = 0.0

    # sub-grid channel layer
    chan_depth = city["chan_store"].astype(np.float64) / cell_area   # m equiv
    chan_qbf = city["chan_qbf"].astype(np.float64)
    ds_idx = downstream_index(city["fdir"])
    ds_valid = ds_idx >= 0
    ds_target = ds_idx[ds_valid]

    S_max = 25400.0 / cn - 254.0            # SCS-CN retention, mm
    Ia = 0.2 * S_max
    P_cum = np.zeros(shape)
    Q_cum = np.zeros(shape)

    h = np.zeros(shape)
    h_max = np.zeros(shape)

    nt = rain_mmh.shape[0]
    frames, frame_t = [], []
    tot_rain = tot_runoff = tot_drain = tot_sea = tot_surch = 0.0

    for t in range(nt):
        rate = rain_mmh[t].astype(np.float64)
        step_s = dt_h * 3600.0
        elapsed = 0.0

        while elapsed < step_s - 1e-9:
            hs_now = np.maximum(h - chan_depth, 0.0)
            hmax = float(hs_now.max())
            dt = (cfl * dx / np.sqrt(G * hmax)) if hmax > 1e-4 else substep_max_s
            dt = float(np.clip(dt, substep_min_s, substep_max_s))
            dt = min(dt, step_s - elapsed)
            elapsed += dt

            # ---- rainfall -> runoff (SCS-CN) ---------------------------
            dP = rate * (dt / 3600.0)
            P_cum += dP
            eff = np.maximum(P_cum - Ia, 0.0)
            Q_new = eff * eff / (eff + S_max)
            dQ = np.maximum(Q_new - Q_cum, 0.0)
            Q_cum = Q_new
            h += (dQ / 1000.0) * land
            tot_rain += float((dP / 1000.0)[land].sum()) * cell_area
            tot_runoff += float((dQ / 1000.0)[land].sum()) * cell_area

            # ---- storm sewers, throttled by trunk-network capacity -----
            hs = np.maximum(h - chan_depth, 0.0)
            removed = np.minimum(hs, drain_ms * dt)
            demand_m3 = float(removed[land].sum()) * cell_area
            budget_m3 = sewer_net_cap * dt
            if demand_m3 > budget_m3 > 0.0:
                removed *= budget_m3 / demand_m3     # network is surcharged
                sewer_saturated_s += dt
            h -= removed
            tot_drain += float(removed[land].sum()) * cell_area

            # ---- channel layer: convey in-bank water downstream --------
            in_bank = np.minimum(h, chan_depth) * cell_area
            move = np.minimum(in_bank, chan_qbf * dt)
            h -= move / cell_area
            hf = h.ravel().copy()
            np.add.at(hf, ds_target, move.ravel()[ds_valid] / cell_area)
            h = hf.reshape(shape)

            # ---- buried culvert: capacity-limited ----------------------
            if ghost.any():
                want = h[ghost] * cell_area
                demand = float(want.sum())
                if demand > 0:
                    budget = ghost_cap * dt
                    take = min(budget, demand)
                    h[ghost] -= (want * (take / demand)) / cell_area
                    tot_drain += take
                    if demand > budget:
                        tot_surch += demand - budget

            # ---- 2D surface routing (out-of-bank water only) -----------
            hs = np.maximum(h - chan_depth, 0.0)
            wse = dem + hs
            vx, vy = _surface_flux(wse, dem, hs, manning, dx, dt, cell_area)
            dv = np.zeros(shape)
            dv[:, :-1] -= vx[:, :-1]
            dv[:, 1:] += vx[:, :-1]
            dv[:-1, :] -= vy[:-1, :]
            dv[1:, :] += vy[:-1, :]
            h += dv / cell_area
            h = np.clip(h, 0.0, None)

            # ---- open ocean boundary -----------------------------------
            tot_sea += float(h[sea].sum()) * cell_area
            h[sea] = 0.0

        # reported depth is what is visible on the ground: out-of-bank water
        h_surface = np.maximum(h - chan_depth, 0.0)
        h_max = np.maximum(h_max, h_surface)
        if t % record_every == 0:
            frames.append(h_surface.astype(np.float32).copy())
            frame_t.append(t)
        if verbose and t % 12 == 0:
            print(f"    t={t*dt_h:5.2f}h  rain {rate.mean():5.1f} mm/h  "
                  f"max {h_surface.max():5.2f} m  >10cm "
                  f"{100*((h_surface>0.10)&land).sum()/land.sum():5.2f}%")

    storage = float(h[land].sum()) * cell_area
    bal = tot_runoff - tot_drain - tot_sea - storage
    return {
        "frames": np.stack(frames) if frames else np.zeros((0,) + shape, np.float32),
        "frame_t": np.array(frame_t, dtype=np.int32),
        "h_max": h_max.astype(np.float32),
        "h_final": np.maximum(h - chan_depth, 0.0).astype(np.float32),
        "vol_rain_m3": tot_rain, "vol_runoff_m3": tot_runoff,
        "vol_drain_m3": tot_drain, "vol_sea_m3": tot_sea,
        "vol_storage_m3": storage, "vol_surcharge_m3": tot_surch,
        "sewer_saturated_h": sewer_saturated_s / 3600.0,
        "mass_error_frac": bal / max(tot_runoff, 1e-9),
        "runoff_coeff": tot_runoff / max(tot_rain, 1e-9),
    }
