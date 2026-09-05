"""FastAPI backend for the racing-line web app: wraps f1sim's track/car/lap
simulation and line optimizer behind a small JSON API, and serves the static
frontend that visualizes the result on a canvas.
"""
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1sim.track import Track
from f1sim.car import Car
from f1sim import vehicle_dynamics as vd
from f1sim import lap_sim
from f1sim.line_optimizer import optimize_line

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Fastest Racer")


class SimulateRequest(BaseModel):
    track: dict
    car: dict
    spacing: float = Field(default=3.0, ge=0.5, le=20.0)
    maxiter: int = Field(default=120, ge=1, le=400)
    n_ctrl: int | None = Field(default=None, ge=4, le=200)


def _load_bundled(directory: Path):
    out = {}
    if directory.exists():
        for p in sorted(directory.glob("*.json")):
            out[p.stem] = json.loads(p.read_text())
    return out


BUNDLED_TRACKS = _load_bundled(ROOT_DIR / "tracks")
BUNDLED_CARS = _load_bundled(ROOT_DIR / "cars")


@app.get("/api/tracks")
def list_tracks():
    return {key: val for key, val in BUNDLED_TRACKS.items()}


@app.get("/api/cars")
def list_cars():
    return {key: val for key, val in BUNDLED_CARS.items()}


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    try:
        track = Track.from_dict(req.track)
        car = Car.from_dict(req.car)
    except (KeyError, TypeError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid track/car definition: {e}")

    track_r = track.resample(spacing=req.spacing)
    if track_r.n < 8:
        raise HTTPException(status_code=400, detail="Track too short/coarse after resampling.")

    gg = vd.build_gg_lookup(car)
    baseline = lap_sim.simulate_lap(track_r, np.zeros(track_r.n), car, gg)
    alpha, final, history = optimize_line(
        track_r, car, gg, maxiter=req.maxiter, n_ctrl=req.n_ctrl, verbose=False
    )

    left = track_r.points + (track_r.width / 2)[:, None] * track_r.normal
    right = track_r.points - (track_r.width / 2)[:, None] * track_r.normal

    return {
        "track_name": track.name,
        "car_name": car.name,
        "baseline_lap_time": baseline["lap_time"],
        "optimized_lap_time": final["lap_time"],
        "improvement_s": baseline["lap_time"] - final["lap_time"],
        "path": final["path"].tolist(),
        "speed_mps": final["speed"].tolist(),
        "curvature": final["kappa"].tolist(),
        "distance_m": np.cumsum(final["ds"]).tolist(),
        "left_edge": left.tolist(),
        "right_edge": right.tolist(),
        "history": history,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
