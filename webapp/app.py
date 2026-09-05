"""FastAPI backend for the racing-line web app: wraps f1sim's track/car/lap
simulation and line optimizer behind a small JSON API, and serves the static
frontend that visualizes the result on a canvas.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from f1sim.track import Track
from f1sim.car import Car
from f1sim import vehicle_dynamics as vd
from f1sim import lap_sim
from f1sim.line_optimizer import optimize_line

from . import auth
from . import billing

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Fastest Racer")
auth.init_db()

# Content gating for the SaaS tiers: the synthetic demo track/car are free so
# anyone can try the tool; everything requiring real-world data curation
# (actual F1 circuits, an accuracy-checked F1 car model) is the Pro perk —
# that curation work is the actual product being sold, since the simulator
# code itself runs client-side and is not something a paywall can hide.
FREE_TRACKS = {"example_track"}
FREE_CARS = {"example_car"}


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


def _is_pro(user: Optional[dict]) -> bool:
    return bool(user and user.get("tier") == "pro")


@app.get("/api/tracks")
def list_tracks(user: Optional[dict] = Depends(auth.get_current_user)):
    pro = _is_pro(user)
    out = {}
    for key, val in BUNDLED_TRACKS.items():
        if key in FREE_TRACKS or pro:
            out[key] = val
        else:
            out[key] = {"name": val.get("name", key), "locked": True}
    return out


@app.get("/api/cars")
def list_cars(user: Optional[dict] = Depends(auth.get_current_user)):
    pro = _is_pro(user)
    out = {}
    for key, val in BUNDLED_CARS.items():
        if key in FREE_CARS or pro:
            out[key] = val
        else:
            out[key] = {"name": val.get("name", key), "locked": True}
    return out


# --- Auth -------------------------------------------------------------

class SignupRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: str
    password: str


def _set_session_cookie(response: Response, user_id: int):
    response.set_cookie(
        auth.COOKIE_NAME,
        auth.make_session_cookie(user_id),
        max_age=auth.COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )


@app.post("/api/auth/signup")
def signup(req: SignupRequest, response: Response):
    user_id = auth.create_user(req.email, req.password)
    _set_session_cookie(response, user_id)
    return {"email": req.email.strip().lower(), "tier": "free"}


@app.post("/api/auth/login")
def login(req: LoginRequest, response: Response):
    user = auth.verify_login(req.email, req.password)
    _set_session_cookie(response, user["id"])
    return {"email": user["email"], "tier": user["tier"]}


@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE_NAME)
    return {"ok": True}


@app.get("/api/auth/me")
def me(user: Optional[dict] = Depends(auth.get_current_user)):
    if user is None:
        return {"logged_in": False}
    return {"logged_in": True, "email": user["email"], "tier": user["tier"]}


# --- Billing ------------------------------------------------------------

@app.post("/api/billing/checkout")
def billing_checkout(user: dict = Depends(auth.require_user)):
    try:
        url = billing.create_checkout_session(user)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"url": url}


@app.post("/api/billing/portal")
def billing_portal(user: dict = Depends(auth.require_user)):
    try:
        url = billing.create_portal_session(user)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"url": url}


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event_type = billing.handle_webhook(payload, sig_header)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Webhook error: {e}")
    return {"received": True, "type": event_type}


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
        "track_length_m": float(np.sum(baseline["ds"])),
        "baseline_lap_time": baseline["lap_time"],
        "optimized_lap_time": final["lap_time"],
        "improvement_s": baseline["lap_time"] - final["lap_time"],
        "path": final["path"].tolist(),
        "speed_mps": final["speed"].tolist(),
        "curvature": final["kappa"].tolist(),
        "distance_m": np.cumsum(final["ds"]).tolist(),
        "dt_s": final["dt"].tolist(),
        "baseline_speed_mps": baseline["speed"].tolist(),
        "baseline_distance_m": np.cumsum(baseline["ds"]).tolist(),
        "baseline_dt_s": baseline["dt"].tolist(),
        "left_edge": left.tolist(),
        "right_edge": right.tolist(),
        "history": history,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
