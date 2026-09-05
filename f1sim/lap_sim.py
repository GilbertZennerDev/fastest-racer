"""Quasi-steady-state lap-time march: given a curvature profile (from a
candidate line) and a precomputed GG-diagram, compute the speed profile via
corner-speed limit -> forward accel pass -> backward brake pass, then
integrate lap time.

The per-point recurrences (forward/backward pass, corner-speed fixed point)
are inherently sequential, so they're JIT-compiled with numba: each one
degrades to a tight scalar loop with a hand-rolled binary-search interp,
avoiding per-element numpy/np.interp call overhead. This is the dominant
cost during line optimization since it's re-run on every objective/gradient
evaluation. Falls back to pure numpy if numba isn't installed.
"""
import numpy as np
from . import vehicle_dynamics as vd

try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True, fastmath=True)
def _interp1(v, xp, fp):
    n = xp.shape[0]
    if v <= xp[0]:
        return fp[0]
    if v >= xp[n - 1]:
        return fp[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= v:
            lo = mid
        else:
            hi = mid
    x0, x1 = xp[lo], xp[hi]
    f0, f1 = fp[lo], fp[hi]
    t = (v - x0) / (x1 - x0)
    return f0 + t * (f1 - f0)


@njit(cache=True, fastmath=True)
def _corner_speed_limit_core(v_grid, a_lat_grid, kappa, v_max_grid, iters):
    n = kappa.shape[0]
    out = np.empty(n)
    for i in range(n):
        k = abs(kappa[i])
        if k < 1e-6:
            k = 1e-6
        vi = v_max_grid
        for _ in range(iters):
            a = _interp1(vi, v_grid, a_lat_grid)
            v_new = np.sqrt(a / k)
            if v_new > v_max_grid:
                v_new = v_max_grid
            vi = 0.5 * vi + 0.5 * v_new
        out[i] = vi
    return out


@njit(cache=True, fastmath=True)
def _forward_pass_core(v_grid, a_acc_grid, v_corner, ds):
    n = v_corner.shape[0]
    v = v_corner.copy()
    for i in range(n - 1):
        a = _interp1(v[i], v_grid, a_acc_grid)
        v_lim = np.sqrt(v[i] ** 2 + 2 * a * ds[i])
        if v_lim < v[i + 1]:
            v[i + 1] = v_lim
    return v


@njit(cache=True, fastmath=True)
def _backward_pass_core(v_grid, a_brk_grid, v_in, ds):
    n = v_in.shape[0]
    v = v_in.copy()
    for i in range(n - 1, 0, -1):
        a = _interp1(v[i], v_grid, a_brk_grid)
        v_lim = np.sqrt(v[i] ** 2 + 2 * a * ds[i - 1])
        if v_lim < v[i - 1]:
            v[i - 1] = v_lim
    return v


def corner_speed_limit(gg, kappa, v_max_grid=120.0, iters=20):
    """v such that a_lat_max(v) == v^2 * |kappa|, solved per-point via
    fixed-point iteration (a_lat_max is a mild function of v)."""
    kappa = np.ascontiguousarray(kappa, dtype=np.float64)
    return _corner_speed_limit_core(gg["v"], gg["a_lat"], kappa, float(v_max_grid), int(iters))


def forward_pass(gg, v_corner, ds):
    v_corner = np.ascontiguousarray(v_corner, dtype=np.float64)
    ds = np.ascontiguousarray(ds, dtype=np.float64)
    return _forward_pass_core(gg["v"], gg["a_acc"], v_corner, ds)


def backward_pass(gg, v, ds):
    v = np.ascontiguousarray(v, dtype=np.float64)
    ds = np.ascontiguousarray(ds, dtype=np.float64)
    return _backward_pass_core(gg["v"], gg["a_brk"], v, ds)


def warmup(gg):
    """Trigger numba JIT compilation once, up front, so the cost doesn't
    land inside the first optimizer iteration."""
    if not _NUMBA:
        return
    dummy_k = np.array([1e-3, 1e-2])
    dummy_ds = np.array([1.0, 1.0])
    v = corner_speed_limit(gg, dummy_k)
    v = forward_pass(gg, v, dummy_ds)
    backward_pass(gg, v, dummy_ds)


def simulate_lap(track_r, alpha, car, gg, closed_laps=2):
    """Returns dict with speed profile, lap time, path points, curvature.
    For closed tracks, runs the forward/backward pass over `closed_laps`
    concatenated laps and takes the middle lap's time to remove start/finish
    transient effects (a proper cyclic solve would iterate until v wraps
    consistently; this concatenation approximates that cheaply)."""
    kappa, path = track_r.curvature_of_offset(alpha)
    ds = track_r.path_length(alpha)

    v_corner = corner_speed_limit(gg, kappa)

    if track_r.closed:
        reps = max(closed_laps, 2)
        v_c = np.tile(v_corner, reps)
        ds_rep = np.tile(ds, reps)
        v_f = forward_pass(gg, v_c, ds_rep)
        v_b = backward_pass(gg, v_f, ds_rep)
        n = track_r.n
        mid = reps // 2
        v = v_b[mid * n:(mid + 1) * n]
        ds_mid = ds
    else:
        v_f = forward_pass(gg, v_corner, ds)
        v = backward_pass(gg, v_f, ds)
        ds_mid = ds

    dt = ds_mid / np.maximum(v, 0.1)
    lap_time = float(np.sum(dt))

    return {
        "path": path,
        "kappa": kappa,
        "speed": v,
        "ds": ds_mid,
        "dt": dt,
        "lap_time": lap_time,
    }
