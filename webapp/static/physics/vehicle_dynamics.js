// JS port of f1sim/tire.py + f1sim/vehicle_dynamics.py: builds the GG-diagram
// (max lateral accel, max traction accel, max brake decel vs. speed) from
// the car struct via aero load transfer + a simplified load-sensitive
// friction model — mirrors the Python implementation function-for-function.

const G = 9.81;

function tireMu(loadN, nominalLoadN, dPeak, loadSensitivity) {
  const loadRatio = Math.max(loadN, 1.0) / nominalLoadN;
  return dPeak * (1.0 - loadSensitivity * (loadRatio - 1.0));
}

function peakMuLateral(loadN, tp) {
  return tireMu(loadN, tp.nominal_load_N, tp.D_lat, tp.load_sensitivity);
}

function peakMuLongitudinal(loadN, tp) {
  return tireMu(loadN, tp.nominal_load_N, tp.D_long, tp.load_sensitivity);
}

function aeroForces(car, v) {
  const q = 0.5 * car.air_density * v * v;
  return { drag: q * car.CdA, dfFront: q * car.ClA_front, dfRear: q * car.ClA_rear };
}

function axleStaticLoads(car, dfFront, dfRear) {
  return {
    fzF: car.static_load_front_N + dfFront,
    fzR: car.static_load_rear_N + dfRear,
  };
}

function maxLateralAccel(car, v, iters = 25) {
  const { dfFront, dfRear } = aeroForces(car, v);
  const { fzF: fzF0, fzR: fzR0 } = axleStaticLoads(car, dfFront, dfRear);
  const trackAvg = 0.5 * (car.track_width_front_m + car.track_width_rear_m);

  let aLat = 5.0;
  for (let i = 0; i < iters; i++) {
    const totalTransfer = (car.mass_kg * aLat * car.cg_height_m) / trackAvg;
    const dtF = totalTransfer * car.roll_stiffness_front_frac;
    const dtR = totalTransfer * (1.0 - car.roll_stiffness_front_frac);

    const fzFOuter = Math.max(fzF0 / 2 + dtF / 2, 0);
    const fzFInner = Math.max(fzF0 / 2 - dtF / 2, 0);
    const fzROuter = Math.max(fzR0 / 2 + dtR / 2, 0);
    const fzRInner = Math.max(fzR0 / 2 - dtR / 2, 0);

    const fMax =
      peakMuLateral(fzFOuter, car.tires) * fzFOuter +
      peakMuLateral(fzFInner, car.tires) * fzFInner +
      peakMuLateral(fzROuter, car.tires) * fzROuter +
      peakMuLateral(fzRInner, car.tires) * fzRInner;

    const aNew = fMax / car.mass_kg;
    if (Math.abs(aNew - aLat) < 1e-4) { aLat = aNew; break; }
    aLat = 0.5 * aLat + 0.5 * aNew;
  }
  return Math.max(aLat, 0.1);
}

function maxTractionAccel(car, v, iters = 15) {
  const { drag, dfFront, dfRear } = aeroForces(car, v);
  const { fzF: fzF0, fzR: fzR0 } = axleStaticLoads(car, dfFront, dfRear);

  let aLong = 3.0;
  for (let i = 0; i < iters; i++) {
    const longTransfer = (car.mass_kg * aLong * car.cg_height_m) / car.wheelbase_m;
    const fzF = Math.max(fzF0 - longTransfer, 0);
    const fzR = Math.max(fzR0 + longTransfer, 0);

    let driveLoad;
    if (car.drivetrain === "RWD") driveLoad = fzR;
    else if (car.drivetrain === "FWD") driveLoad = fzF;
    else driveLoad = fzF + fzR;

    const fTraction = peakMuLongitudinal(driveLoad, car.tires) * driveLoad;
    const aTraction = fTraction / car.mass_kg;
    const aNew = Math.min(aTraction, car.max_engine_accel_mps2);
    if (Math.abs(aNew - aLong) < 1e-4) { aLong = aNew; break; }
    aLong = 0.5 * aLong + 0.5 * aNew;
  }

  const aPower = (car.max_power_kw * 1000.0) / (car.mass_kg * Math.max(v, 1.0));
  const aTractionCapped = Math.min(aLong, aPower);
  const aNet = aTractionCapped - drag / car.mass_kg;
  return Math.max(aNet, 0.0);
}

function maxBrakeDecel(car, v, iters = 15) {
  const { drag, dfFront, dfRear } = aeroForces(car, v);
  const { fzF: fzF0, fzR: fzR0 } = axleStaticLoads(car, dfFront, dfRear);

  let aBrake = 5.0;
  for (let i = 0; i < iters; i++) {
    const longTransfer = (car.mass_kg * aBrake * car.cg_height_m) / car.wheelbase_m;
    const fzF = Math.max(fzF0 + longTransfer, 0);
    const fzR = Math.max(fzR0 - longTransfer, 0);

    const fBrake =
      peakMuLongitudinal(fzF, car.tires) * fzF + peakMuLongitudinal(fzR, car.tires) * fzR;
    const aTire = fBrake / car.mass_kg;
    const aNew = Math.min(aTire, car.max_brake_decel_mps2);
    if (Math.abs(aNew - aBrake) < 1e-4) { aBrake = aNew; break; }
    aBrake = 0.5 * aBrake + 0.5 * aNew;
  }
  return aBrake + drag / car.mass_kg;
}

/** Normalizes a bundled car JSON (with defaults + derived static loads). */
function prepareCar(carDict) {
  const car = Object.assign({}, carDict);
  car.tires = Object.assign({}, carDict.tires);
  car.weight_dist_rear = 1.0 - car.weight_dist_front;
  car.static_load_front_N = car.mass_kg * G * car.weight_dist_front;
  car.static_load_rear_N = car.mass_kg * G * car.weight_dist_rear;
  return car;
}

function buildGGLookup(car, vMin = 5.0, vMax = 120.0, n = 60) {
  const v = new Float64Array(n);
  const aLat = new Float64Array(n);
  const aAcc = new Float64Array(n);
  const aBrk = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const vi = vMin + ((vMax - vMin) * i) / (n - 1);
    v[i] = vi;
    aLat[i] = maxLateralAccel(car, vi);
    aAcc[i] = maxTractionAccel(car, vi);
    aBrk[i] = maxBrakeDecel(car, vi);
  }
  return { v, a_lat: aLat, a_acc: aAcc, a_brk: aBrk };
}
