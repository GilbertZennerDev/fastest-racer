"""Entry point: load track+car, build GG-diagram, optimize the racing line,
report lap time, plot the result, and export data."""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from f1sim.track import Track
from f1sim.car import Car
from f1sim import vehicle_dynamics as vd
from f1sim import lap_sim
from f1sim.line_optimizer import optimize_line


def plot_result(track_r, final, out_path="result.png"):
    path = final["path"]
    speed_kmh = final["speed"] * 3.6

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    left = track_r.points + (track_r.width / 2)[:, None] * track_r.normal
    right = track_r.points - (track_r.width / 2)[:, None] * track_r.normal
    ax.plot(np.append(left[:, 0], left[0, 0]), np.append(left[:, 1], left[0, 1]), "k-", lw=1)
    ax.plot(np.append(right[:, 0], right[0, 0]), np.append(right[:, 1], right[0, 1]), "k-", lw=1)

    pts = np.vstack([path, path[0]])
    sp = np.append(speed_kmh, speed_kmh[0])
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=sp, cmap="RdYlGn", s=8)
    plt.colorbar(sc, ax=ax, label="Speed (km/h)")
    ax.set_aspect("equal")
    ax.set_title("Optimal Racing Line")

    ax2 = axes[1]
    dist = np.cumsum(final["ds"])
    ax2.plot(dist, speed_kmh)
    ax2.set_xlabel("Distance (m)")
    ax2.set_ylabel("Speed (km/h)")
    ax2.set_title(f"Speed Profile — Lap Time: {final['lap_time']:.3f}s")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved plot to {out_path}")


def export_data(track_r, final, out_path="result.json"):
    data = {
        "lap_time_s": final["lap_time"],
        "points": final["path"].tolist(),
        "speed_mps": final["speed"].tolist(),
        "curvature": final["kappa"].tolist(),
        "ds": final["ds"].tolist(),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved data to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="tracks/example_track.json")
    parser.add_argument("--car", default="cars/example_car.json")
    parser.add_argument("--spacing", type=float, default=3.0, help="resample spacing (m)")
    parser.add_argument("--maxiter", type=int, default=150)
    args = parser.parse_args()

    track = Track.from_json(args.track)
    car = Car.from_json(args.car)
    track_r = track.resample(spacing=args.spacing)

    print(f"Track: {track.name}  ({track_r.n} samples, {track_r.s[-1]:.0f} m)")
    print(f"Car:   {car.name}  ({car.mass_kg} kg, {car.max_power_kw} kW, {car.drivetrain})")

    print("Building GG-diagram...")
    gg = vd.build_gg_lookup(car)

    baseline = lap_sim.simulate_lap(track_r, np.zeros(track_r.n), car, gg)
    print(f"Centerline lap time: {baseline['lap_time']:.3f} s")

    print("Optimizing racing line...")
    alpha, final, history = optimize_line(track_r, car, gg, maxiter=args.maxiter)

    print(f"Optimized lap time:  {final['lap_time']:.3f} s "
          f"({baseline['lap_time'] - final['lap_time']:.3f} s faster)")

    plot_result(track_r, final)
    export_data(track_r, final)


if __name__ == "__main__":
    main()
