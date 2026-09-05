"""Pacejka Magic Formula tire force model with simple load sensitivity."""
import numpy as np


def _mu(load_N, nominal_load_N, D_peak, load_sensitivity):
    """Peak friction coefficient falls off as load rises above nominal
    (approximates tire load sensitivity without full Fz-dependent B/C/D)."""
    load_ratio = np.maximum(load_N, 1.0) / nominal_load_N
    return D_peak * (1.0 - load_sensitivity * (load_ratio - 1.0))


def lateral_force(slip_angle_rad, load_N, tp):
    """Magic Formula: F = mu*Fz * sin(C*atan(B*s - E*(B*s - atan(B*s))))"""
    mu = _mu(load_N, tp.nominal_load_N, tp.D_lat, tp.load_sensitivity)
    s = slip_angle_rad
    Bs = tp.B_lat * s
    return mu * load_N * np.sin(tp.C_lat * np.arctan(Bs - tp.E_lat * (Bs - np.arctan(Bs))))


def longitudinal_force(slip_ratio, load_N, tp):
    mu = _mu(load_N, tp.nominal_load_N, tp.D_long, tp.load_sensitivity)
    s = slip_ratio
    Bs = tp.B_long * s
    return mu * load_N * np.sin(tp.C_long * np.arctan(Bs - tp.E_long * (Bs - np.arctan(Bs))))


def peak_mu_lateral(load_N, tp):
    return _mu(load_N, tp.nominal_load_N, tp.D_lat, tp.load_sensitivity)


def peak_mu_longitudinal(load_N, tp):
    return _mu(load_N, tp.nominal_load_N, tp.D_long, tp.load_sensitivity)
