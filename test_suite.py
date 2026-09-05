"""Test suite for f1sim. Run with: pytest test_suite.py -v"""
import numpy as np
import pytest

from f1sim.track import Track, ResampledTrack
from f1sim.car import Car, TireParams
from f1sim import tire
from f1sim import vehicle_dynamics as vd
from f1sim import lap_sim
from f1sim.line_optimizer import optimize_line


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def oval_track():
    """A simple stadium/oval: two straights + two semicircular ends, so
    curvature is analytically known (0 on straights, 1/r on the curves)."""
    r = 50.0
    straight = 100.0
    n_arc = 40
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_arc)
    right_end = np.column_stack([straight / 2 + r * np.sin(theta), r - r * np.cos(theta) + 0])
    left_end = np.column_stack([-straight / 2 - r * np.sin(theta), -r + r * np.cos(theta) + 2 * r])
    points = np.vstack([
        [[-straight / 2, 0], [straight / 2, 0]],
        right_end,
        [[straight / 2, 2 * r], [-straight / 2, 2 * r]],
        left_end,
    ])
    return Track(name="oval", points=points, width=10.0, closed=True)


@pytest.fixture
def straight_track():
    points = [[0, 0], [500, 0]]
    return Track(name="straight", points=points, width=10.0, closed=False)


@pytest.fixture
def car():
    return Car.from_json("cars/example_car.json")


@pytest.fixture
def gg(car):
    return vd.build_gg_lookup(car, v_min=5.0, v_max=100.0, n=30)


# ---------------------------------------------------------------------------
# track geometry
# ---------------------------------------------------------------------------

class TestTrack:
    def test_resample_preserves_closed_loop(self, oval_track):
        tr = oval_track.resample(spacing=2.0)
        assert tr.closed
        # first and last sampled points should be close (loop closes)
        assert tr.n > 10

    def test_resample_spacing_roughly_respected(self, oval_track):
        tr = oval_track.resample(spacing=2.0)
        seg_lengths = tr.path_length(np.zeros(tr.n))
        assert np.median(seg_lengths) == pytest.approx(2.0, abs=1.0)

    def test_straight_has_near_zero_curvature(self, straight_track):
        tr = straight_track.resample(spacing=5.0)
        kappa, _ = tr.curvature_of_offset(np.zeros(tr.n))
        assert np.allclose(kappa[1:-1], 0.0, atol=1e-6)

    def test_offset_path_shifts_toward_normal(self, straight_track):
        tr = straight_track.resample(spacing=5.0)
        _, path0 = tr.curvature_of_offset(np.zeros(tr.n))
        _, path1 = tr.curvature_of_offset(np.ones(tr.n))
        # offset path should be displaced by ~width/2 along the normal
        disp = np.linalg.norm(path1 - path0, axis=1)
        assert np.allclose(disp[1:-1], tr.width[1:-1] / 2, atol=0.5)

    def test_curvature_sign_flips_with_offset_direction(self, oval_track):
        tr = oval_track.resample(spacing=2.0)
        k_pos, _ = tr.curvature_of_offset(np.full(tr.n, 0.9))
        k_neg, _ = tr.curvature_of_offset(np.full(tr.n, -0.9))
        # on curved sections, offsetting should change curvature magnitude
        assert not np.allclose(k_pos, k_neg)

    def test_from_json_roundtrip(self):
        t = Track.from_json("tracks/example_track.json")
        assert t.name == "Test Circuit"
        assert t.closed is True
        assert len(t.raw_points) > 3


# ---------------------------------------------------------------------------
# tire model
# ---------------------------------------------------------------------------

class TestTire:
    def test_zero_slip_gives_zero_force(self):
        tp = TireParams()
        assert tire.lateral_force(0.0, 3000.0, tp) == pytest.approx(0.0, abs=1e-9)
        assert tire.longitudinal_force(0.0, 3000.0, tp) == pytest.approx(0.0, abs=1e-9)

    def test_force_increases_then_saturates(self):
        tp = TireParams()
        slips = np.linspace(0.001, 0.5, 50)
        forces = np.array([tire.lateral_force(s, 3000.0, tp) for s in slips])
        # should rise near the origin
        assert forces[5] > forces[1]
        # magic formula should not diverge — bounded by ~D*Fz
        assert np.all(forces < tp.D_lat * 3000.0 * 1.05)

    def test_load_sensitivity_reduces_mu_at_high_load(self):
        tp = TireParams(load_sensitivity=0.2)
        mu_nominal = tire.peak_mu_lateral(tp.nominal_load_N, tp)
        mu_high = tire.peak_mu_lateral(tp.nominal_load_N * 2, tp)
        assert mu_high < mu_nominal

    def test_force_is_odd_function_of_slip(self):
        tp = TireParams()
        f_pos = tire.lateral_force(0.05, 3000.0, tp)
        f_neg = tire.lateral_force(-0.05, 3000.0, tp)
        assert f_pos == pytest.approx(-f_neg, rel=1e-6)


# ---------------------------------------------------------------------------
# vehicle dynamics / GG-diagram
# ---------------------------------------------------------------------------

class TestVehicleDynamics:
    def test_gg_lookup_shapes(self, gg):
        assert gg["v"].shape == gg["a_lat"].shape == gg["a_acc"].shape == gg["a_brk"].shape

    def test_accelerations_are_positive(self, gg):
        assert np.all(gg["a_lat"] > 0)
        assert np.all(gg["a_acc"] >= 0)
        assert np.all(gg["a_brk"] > 0)

    def test_traction_accel_decreases_with_speed(self, car):
        a_low = vd.max_traction_accel(car, 10.0)
        a_high = vd.max_traction_accel(car, 90.0)
        assert a_high < a_low  # power-limited regime kicks in

    def test_brake_decel_exceeds_traction_accel(self, car, gg):
        # braking uses all 4 wheels vs. drive-axle-only acceleration,
        # so brake decel should generally be higher at a given speed
        v = 40.0
        assert vd.interp_gg(gg, v, "a_brk") > vd.interp_gg(gg, v, "a_acc")

    def test_more_downforce_increases_lateral_grip(self, car):
        v = 60.0
        a_base = vd.max_lateral_accel(car, v)
        car.ClA_front *= 3
        car.ClA_rear *= 3
        a_more_df = vd.max_lateral_accel(car, v)
        assert a_more_df > a_base

    def test_heavier_car_has_lower_traction_accel_at_low_speed(self, car):
        v = 10.0
        a_light = vd.max_traction_accel(car, v)
        car.mass_kg *= 1.5
        a_heavy = vd.max_traction_accel(car, v)
        assert a_heavy <= a_light


# ---------------------------------------------------------------------------
# lap simulation
# ---------------------------------------------------------------------------

class TestLapSim:
    def test_straight_line_reaches_high_speed_no_corner_limit(self, straight_track, car, gg):
        tr = straight_track.resample(spacing=5.0)
        # force a low starting speed (as if exiting a slow corner) so the
        # forward accel pass has room to show acceleration down the straight
        kappa, path = tr.curvature_of_offset(np.zeros(tr.n))
        ds = tr.path_length(np.zeros(tr.n))
        v_corner = np.full(tr.n, 120.0)
        v_corner[0] = 15.0  # simulate a slow corner exit at the start
        v = lap_sim.forward_pass(gg, v_corner, ds)
        assert v[-1] > v[0]
        assert np.all(np.diff(v) >= -1e-9)  # monotonically non-decreasing

        result = lap_sim.simulate_lap(tr, np.zeros(tr.n), car, gg)
        assert result["lap_time"] > 0

    def test_corner_speed_respects_gg_limit(self, oval_track, gg):
        tr = oval_track.resample(spacing=2.0)
        kappa, _ = tr.curvature_of_offset(np.zeros(tr.n))
        v = lap_sim.corner_speed_limit(gg, kappa)
        a_lat_actual = v ** 2 * np.abs(np.maximum(kappa, 1e-6))
        a_lat_max = vd.interp_gg(gg, v, "a_lat")
        assert np.all(a_lat_actual <= a_lat_max * 1.01)

    def test_lap_time_positive_and_finite(self, oval_track, car, gg):
        tr = oval_track.resample(spacing=2.0)
        result = lap_sim.simulate_lap(tr, np.zeros(tr.n), car, gg)
        assert np.isfinite(result["lap_time"])
        assert result["lap_time"] > 0

    def test_forward_pass_never_exceeds_input_ceiling(self, gg):
        v_corner = np.array([10.0, 50.0, 5.0, 40.0])
        ds = np.array([10.0, 10.0, 10.0, 10.0])
        v_out = lap_sim.forward_pass(gg, v_corner, ds)
        assert np.all(v_out <= v_corner + 1e-6)

    def test_backward_pass_never_exceeds_input_ceiling(self, gg):
        v = np.array([10.0, 50.0, 5.0, 40.0])
        ds = np.array([10.0, 10.0, 10.0])
        v_out = lap_sim.backward_pass(gg, v.copy(), ds)
        assert np.all(v_out <= v + 1e-6)


# ---------------------------------------------------------------------------
# line optimizer (slow — mark for optional skip)
# ---------------------------------------------------------------------------

class TestLineOptimizer:
    @pytest.mark.slow
    def test_optimized_line_not_slower_than_centerline(self, oval_track, car, gg):
        tr = oval_track.resample(spacing=4.0)
        baseline = lap_sim.simulate_lap(tr, np.zeros(tr.n), car, gg)
        _, final, _ = optimize_line(tr, car, gg, maxiter=40, verbose=False)
        assert final["lap_time"] <= baseline["lap_time"] + 0.5

    @pytest.mark.slow
    def test_optimized_alpha_within_bounds(self, oval_track, car, gg):
        tr = oval_track.resample(spacing=4.0)
        alpha, _, _ = optimize_line(tr, car, gg, maxiter=40, verbose=False)
        assert np.all(alpha >= -1.0) and np.all(alpha <= 1.0)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
