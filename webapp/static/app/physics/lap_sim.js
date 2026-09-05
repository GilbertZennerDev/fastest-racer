// JS port of f1sim/lap_sim.py: corner-speed limit -> forward accel pass ->
// backward brake pass, with the friction-ellipse coupling (see f1sim's
// docstring) between lateral and longitudinal grip. Mirrors the Python/
// numba implementation function-for-function.

function interp1(v, xp, fp) {
  const n = xp.length;
  if (v <= xp[0]) return fp[0];
  if (v >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= v) lo = mid; else hi = mid;
  }
  const t = (v - xp[lo]) / (xp[hi] - xp[lo]);
  return fp[lo] + t * (fp[hi] - fp[lo]);
}

function ellipseLongAccel(vi, kappaI, vGrid, aLatGrid, aLongGrid) {
  const aLatMax = interp1(vi, vGrid, aLatGrid);
  const aLongMax = interp1(vi, vGrid, aLongGrid);
  const aLatUsed = vi * vi * Math.abs(kappaI);
  let ratio = aLatMax > 1e-9 ? aLatUsed / aLatMax : 1.0;
  if (ratio > 1.0) ratio = 1.0;
  return aLongMax * Math.sqrt(Math.max(0.0, 1.0 - ratio * ratio));
}

function cornerSpeedLimit(gg, kappa, vMaxGrid = 120.0, iters = 20) {
  const n = kappa.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let k = Math.abs(kappa[i]);
    if (k < 1e-6) k = 1e-6;
    let vi = vMaxGrid;
    for (let it = 0; it < iters; it++) {
      const a = interp1(vi, gg.v, gg.a_lat);
      let vNew = Math.sqrt(a / k);
      if (vNew > vMaxGrid) vNew = vMaxGrid;
      vi = 0.5 * vi + 0.5 * vNew;
    }
    out[i] = vi;
  }
  return out;
}

function forwardPass(gg, vCorner, ds, kappa) {
  const n = vCorner.length;
  const v = vCorner.slice();
  for (let i = 0; i < n - 1; i++) {
    const a = ellipseLongAccel(v[i], kappa[i], gg.v, gg.a_lat, gg.a_acc);
    const vLim = Math.sqrt(v[i] * v[i] + 2 * a * ds[i]);
    if (vLim < v[i + 1]) v[i + 1] = vLim;
  }
  return v;
}

function backwardPass(gg, vIn, ds, kappa) {
  const n = vIn.length;
  const v = vIn.slice();
  for (let i = n - 1; i > 0; i--) {
    const a = ellipseLongAccel(v[i], kappa[i], gg.v, gg.a_lat, gg.a_brk);
    const vLim = Math.sqrt(v[i] * v[i] + 2 * a * ds[i - 1]);
    if (vLim < v[i - 1]) v[i - 1] = vLim;
  }
  return v;
}

/** Full lap: corner-limit -> forward -> backward -> lap_time, with lap
 * tiling on closed tracks to remove start/finish transient (mirrors
 * f1sim.lap_sim.simulate_lap). */
function simulateLap(trackR, alpha, gg, closedLaps = 2) {
  const { kappa, path } = trackR.curvatureOfOffset(alpha);
  const ds = trackR.pathLength(alpha);
  const n = trackR.n;

  let v, dsMid;
  if (trackR.closed) {
    const reps = Math.max(closedLaps, 2);
    const vCornerBase = cornerSpeedLimit(gg, kappa);
    const vC = new Float64Array(n * reps);
    const dsRep = new Float64Array(n * reps);
    const kappaRep = new Float64Array(n * reps);
    for (let r = 0; r < reps; r++) {
      vC.set(vCornerBase, r * n);
      dsRep.set(ds, r * n);
      kappaRep.set(kappa, r * n);
    }
    const vF = forwardPass({ v: gg.v, a_lat: gg.a_lat, a_acc: gg.a_acc }, vC, dsRep, kappaRep);
    const vB = backwardPass({ v: gg.v, a_lat: gg.a_lat, a_brk: gg.a_brk }, vF, dsRep, kappaRep);
    const mid = Math.floor(reps / 2);
    v = vB.slice(mid * n, (mid + 1) * n);
    dsMid = ds;
  } else {
    const vCorner = cornerSpeedLimit(gg, kappa);
    const vF = forwardPass(gg, vCorner, ds, kappa);
    v = backwardPass(gg, vF, ds, kappa);
    dsMid = ds;
  }

  const dt = new Float64Array(n);
  let lapTime = 0;
  for (let i = 0; i < n; i++) {
    dt[i] = dsMid[i] / Math.max(v[i], 0.1);
    lapTime += dt[i];
  }

  return { path, kappa, speed: v, ds: dsMid, dt, lap_time: lapTime };
}
