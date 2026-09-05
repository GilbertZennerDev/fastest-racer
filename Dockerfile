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
# pay that cost.
RUN python -c "\
from f1sim.car import Car; from f1sim import vehicle_dynamics as vd; from f1sim import lap_sim; \
car = Car.from_json('cars/example_car.json'); gg = vd.build_gg_lookup(car); lap_sim.warmup(gg)"

EXPOSE 8000
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
