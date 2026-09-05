# syntax=docker/dockerfile:1

FROM python:3.12-slim AS runtime
WORKDIR /app

# numba/llvmlite ship manylinux wheels for this base image's arch, so no
# compiler toolchain is needed here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY f1sim ./f1sim
COPY webapp ./webapp
COPY tracks ./tracks
COPY cars ./cars

# Compile numba's JIT cache once at build time (see f1sim/lap_sim.py
# `warmup`) so the first request on a freshly started container doesn't
# pay that cost. Runs as root (still the active user at this point), before
# the switch below, so the cache files end up root-owned but world-readable
# — fine, since the app only ever reads them again, never recompiles.
RUN python -c "\
from f1sim.car import Car; from f1sim import vehicle_dynamics as vd; from f1sim import lap_sim; \
car = Car.from_json('cars/example_car.json'); gg = vd.build_gg_lookup(car); lap_sim.warmup(gg)"

# Drop root for the actual running process — matches feierblum-networking's
# `USER node` pattern. Almost nothing at runtime needs to write inside /app
# (no uploads, no new numba compiles after the warmup above) — the one
# exception is the SQLite user/subscription database, which needs a
# writable directory owned by appuser (a bind-mounted volume at this path in
# docker-compose.yml persists it across container recreates).
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /data \
    && chown appuser:appuser /data
ENV USERS_DB_PATH=/data/users.db
USER appuser

EXPOSE 8000
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
