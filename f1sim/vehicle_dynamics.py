"""Quasi-steady-state vehicle dynamics: builds a GG-diagram (max lateral
accel, max traction accel, max brake decel as functions of speed) from the
full car struct — aero, load transfer (front/rear via roll-stiffness split),
and Pacejka tire load sensitivity. This is where the "full vehicle dynamics"
detail actually lives, feeding the simpler QSS lap-time march.
"""
import numpy as np
from . import tire

G = 9.81


def aero_forces(car, v):
    q = 0.5 * car.air_density * v ** 2
    drag = q * car.CdA
    df_front = q * car.ClA_front
    df_rear = q * car.ClA_rear
    return drag, df_front, df_rear


def axle_static_loads(car, df_front, df_rear):
    fz_f = car.static_load_front_N + df_front
    fz_r = car.static_load_rear_N + df_rear
    return fz_f, fz_r


def max_lateral_accel(car, v, iters=25):
    """Solve a_lat such that front+rear max lateral tire force == m*a_lat,
    accounting for lateral load transfer (front/rear split by roll stiffness
    and average track width) and tire load sensitivity."""
    drag, df_f, df_r = aero_forces(car, v)
    fz_f0, fz_r0 = axle_static_loads(car, df_f, df_r)
    track_avg = 0.5 * (car.track_width_front_m + car.track_width_rear_m)

    a_lat = 5.0  # initial guess m/s^2
    for _ in range(iters):
        total_transfer = car.mass_kg * a_lat * car.cg_height_m / track_avg
        dt_f = total_transfer * car.roll_stiffness_front_frac
        dt_r = total_transfer * (1.0 - car.roll_stiffness_front_frac)

        fz_f_outer = np.maximum(fz_f0 / 2 + dt_f / 2, 0)
        fz_f_inner = np.maximum(fz_f0 / 2 - dt_f / 2, 0)
        fz_r_outer = np.maximum(fz_r0 / 2 + dt_r / 2, 0)
        fz_r_inner = np.maximum(fz_r0 / 2 - dt_r / 2, 0)

        f_max = (
            tire.peak_mu_lateral(fz_f_outer, car.tires) * fz_f_outer
            + tire.peak_mu_lateral(fz_f_inner, car.tires) * fz_f_inner
            + tire.peak_mu_lateral(fz_r_outer, car.tires) * fz_r_outer
            + tire.peak_mu_lateral(fz_r_inner, car.tires) * fz_r_inner
        )
        a_new = f_max / car.mass_kg
        if abs(a_new - a_lat) < 1e-4:
            a_lat = a_new
            break
        a_lat = 0.5 * a_lat + 0.5 * a_new
    return max(a_lat, 0.1)


def max_traction_accel(car, v, iters=15):
    """Max forward accel: min(tire-traction-limited, power-limited,
    driveline-cap) minus aero drag."""
    drag, df_f, df_r = aero_forces(car, v)
    fz_f0, fz_r0 = axle_static_loads(car, df_f, df_r)

    a_long = 3.0
    for _ in range(iters):
        long_transfer = car.mass_kg * a_long * car.cg_height_m / car.wheelbase_m
        fz_f = np.maximum(fz_f0 - long_transfer, 0)
        fz_r = np.maximum(fz_r0 + long_transfer, 0)

        if car.drivetrain == "RWD":
            drive_load = fz_r
        elif car.drivetrain == "FWD":
            drive_load = fz_f
        else:  # AWD
            drive_load = fz_f + fz_r

        f_traction = tire.peak_mu_longitudinal(drive_load, car.tires) * drive_load
        a_traction = f_traction / car.mass_kg
        a_new = min(a_traction, car.max_engine_accel_mps2)
        if abs(a_new - a_long) < 1e-4:
            a_long = a_new
            break
        a_long = 0.5 * a_long + 0.5 * a_new

    a_power = (car.max_power_kw * 1000.0) / (car.mass_kg * max(v, 1.0))
    a_traction_capped = min(a_long, a_power)
    a_net = a_traction_capped - drag / car.mass_kg
    return max(a_net, 0.0)


def max_brake_decel(car, v, iters=15):
    """Max deceleration: tire-limited (all 4 wheels), capped by mechanical
    brake limit, plus aero drag assisting."""
    drag, df_f, df_r = aero_forces(car, v)
    fz_f0, fz_r0 = axle_static_loads(car, df_f, df_r)

    a_brake = 5.0
    for _ in range(iters):
        long_transfer = car.mass_kg * a_brake * car.cg_height_m / car.wheelbase_m
        fz_f = np.maximum(fz_f0 + long_transfer, 0)
        fz_r = np.maximum(fz_r0 - long_transfer, 0)

        f_brake = (
            tire.peak_mu_longitudinal(fz_f, car.tires) * fz_f
            + tire.peak_mu_longitudinal(fz_r, car.tires) * fz_r
        )
        a_tire = f_brake / car.mass_kg
        a_new = min(a_tire, car.max_brake_decel_mps2)
        if abs(a_new - a_brake) < 1e-4:
            a_brake = a_new
            break
        a_brake = 0.5 * a_brake + 0.5 * a_new

    return a_brake + drag / car.mass_kg


def build_gg_lookup(car, v_min=5.0, v_max=120.0, n=60):
    """Precompute GG-diagram terms on a speed grid for fast interpolation
    during the lap-time march."""
    v_grid = np.linspace(v_min, v_max, n)
    a_lat = np.array([max_lateral_accel(car, v) for v in v_grid])
    a_acc = np.array([max_traction_accel(car, v) for v in v_grid])
    a_brk = np.array([max_brake_decel(car, v) for v in v_grid])
    return {"v": v_grid, "a_lat": a_lat, "a_acc": a_acc, "a_brk": a_brk}


def interp_gg(gg, v, key):
    return np.interp(v, gg["v"], gg[key])
