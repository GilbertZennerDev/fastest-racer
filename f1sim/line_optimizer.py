"""Outer loop: find the lateral-offset profile (the racing line) that
minimizes lap time, given the track geometry and the car's GG-diagram.

Optimizes over a small number of spline control points rather than one
alpha per track sample. This is the single biggest speed lever: SLSQP's
finite-difference gradient costs (n_vars + 1) objective evaluations per
iteration, so cutting variables from ~300 (one per sample) to ~30-40
control points cuts optimization wall time by roughly the same factor,
while cubic-spline interpolation keeps the resulting line smooth without
needing the explicit curvature-penalty smoothing term as a crutch.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from . import lap_sim


def _make_expander(track_r, n_ctrl):
    """Returns a function mapping n_ctrl control values -> n (full) values,
    via a periodic (closed track) or natural (open track) cubic spline over
    arc length."""
    s = track_r.s
    total = s[-1] + (s[1] - s[0] if len(s) > 1 else 1.0)

    if track_r.closed:
        s_ctrl = np.linspace(0, total, n_ctrl, endpoint=False)
        s_ctrl_wrap = np.append(s_ctrl, s_ctrl[0] + total)

        def expand(ctrl_alpha):
            y_wrap = np.append(ctrl_alpha, ctrl_alpha[0])
            cs = CubicSpline(s_ctrl_wrap, y_wrap, bc_type="periodic")
            return cs(s)
    else:
        s_ctrl = np.linspace(s[0], s[-1], n_ctrl)

        def expand(ctrl_alpha):
            cs = CubicSpline(s_ctrl, ctrl_alpha)
            return cs(s)

    return expand


def _smooth(alpha, weight):
    d2 = np.roll(alpha, -1) - 2 * alpha + np.roll(alpha, 1)
    return weight * np.sum(d2 ** 2)


def optimize_line(track_r, car, gg, smoothing=0.005, maxiter=150, n_ctrl=None, verbose=True):
    n = track_r.n
    if n_ctrl is None:
        n_ctrl = max(15, min(60, n // 5))
    n_ctrl = min(n_ctrl, n)

    expand = _make_expander(track_r, n_ctrl)
    x0 = np.zeros(n_ctrl)

    lap_sim.warmup(gg)  # pay numba JIT compile cost once, outside the loop

    def objective(ctrl_alpha):
        alpha = expand(ctrl_alpha)
        alpha = np.clip(alpha, -0.98, 0.98)
        result = lap_sim.simulate_lap(track_r, alpha, car, gg)
        return result["lap_time"] + _smooth(alpha, smoothing)

    bounds = [(-0.95, 0.95)] * n_ctrl

    history = []

    def callback(xk):
        alpha = np.clip(expand(xk), -0.98, 0.98)
        t = lap_sim.simulate_lap(track_r, alpha, car, gg)["lap_time"]
        history.append(t)
        if verbose and len(history) % 5 == 0:
            print(f"  iter {len(history):4d}  lap_time={t:.3f}s")

    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-6},
        callback=callback,
    )

    alpha_final = np.clip(expand(res.x), -0.98, 0.98)
    final = lap_sim.simulate_lap(track_r, alpha_final, car, gg)
    return alpha_final, final, history
