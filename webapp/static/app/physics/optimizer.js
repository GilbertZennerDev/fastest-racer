// JS port of f1sim/line_optimizer.py's role, adapted for the browser: there
// is no SciPy/SLSQP client-side, so this uses a projected BFGS (quasi-Newton)
// method with a numerical central-difference gradient and Armijo backtracking
// line search instead. Plain gradient descent converged to a noticeably worse
// line on this problem (box-constrained, ~15-60 vars, mildly ill-conditioned
// since different control points have very different sensitivity depending
// on which part of the track they cover) — BFGS's approximate-Hessian scaling
// is what closes that gap and gets within a fraction of a percent of SLSQP.

function makeExpander(trackR, nCtrl) {
  const s = trackR.s;
  const total = s[s.length - 1] + (s.length > 1 ? s[1] - s[0] : 1.0);

  if (trackR.closed) {
    const sCtrl = [];
    for (let i = 0; i < nCtrl; i++) sCtrl.push((total * i) / nCtrl);
    return (ctrlAlpha) => {
      const spline = periodicCubicSpline(sCtrl, ctrlAlpha, total);
      const out = new Float64Array(s.length);
      for (let i = 0; i < s.length; i++) out[i] = spline(s[i]);
      return out;
    };
  }
  const sCtrl = [];
  for (let i = 0; i < nCtrl; i++) sCtrl.push(s[0] + ((s[s.length - 1] - s[0]) * i) / (nCtrl - 1));
  return (ctrlAlpha) => {
    const spline = naturalCubicSpline(sCtrl, ctrlAlpha);
    const out = new Float64Array(s.length);
    for (let i = 0; i < s.length; i++) out[i] = spline(s[i]);
    return out;
  };
}

function smoothPenalty(alpha, weight) {
  const n = alpha.length;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const prev = alpha[(i - 1 + n) % n], next = alpha[(i + 1) % n];
    const d2 = next - 2 * alpha[i] + prev;
    sum += d2 * d2;
  }
  return weight * sum;
}

function clip(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function clipVec(x, lo, hi) {
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = clip(x[i], lo, hi);
  return out;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function matVec(H, v) {
  const n = v.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const row = H[i];
    for (let j = 0; j < n; j++) s += row[j] * v[j];
    out[i] = s;
  }
  return out;
}

function identity(n) {
  const H = new Array(n);
  for (let i = 0; i < n; i++) {
    H[i] = new Float64Array(n);
    H[i][i] = 1.0;
  }
  return H;
}

/**
 * Optimizes the racing line via projected BFGS. onProgress(iter, lapTime) is
 * called after each accepted step; return false from it to stop early.
 */
function optimizeLine(trackR, gg, { smoothing = 0.005, maxiter = 150, nCtrl = null, onProgress = null } = {}) {
  const n = trackR.n;
  if (nCtrl == null) nCtrl = Math.max(15, Math.min(60, Math.floor(n / 5)));
  nCtrl = Math.min(nCtrl, n);

  const expand = makeExpander(trackR, nCtrl);
  const lo = -0.95, hi = 0.95;
  const eps = 2e-3;

  function objective(ctrlAlpha) {
    const alphaRaw = expand(ctrlAlpha);
    const alpha = new Float64Array(alphaRaw.length);
    for (let i = 0; i < alpha.length; i++) alpha[i] = clip(alphaRaw[i], -0.98, 0.98);
    const result = simulateLap(trackR, alpha, gg);
    return { value: result.lap_time + smoothPenalty(alpha, smoothing), result };
  }

  function gradient(x, fx) {
    const grad = new Float64Array(nCtrl);
    for (let j = 0; j < nCtrl; j++) {
      const xPlus = x.slice(); xPlus[j] = clip(xPlus[j] + eps, lo, hi);
      const xMinus = x.slice(); xMinus[j] = clip(xMinus[j] - eps, lo, hi);
      const denom = xPlus[j] - xMinus[j];
      if (denom === 0) { grad[j] = 0; continue; }
      grad[j] = (objective(xPlus).value - objective(xMinus).value) / denom;
    }
    return grad;
  }

  let x = new Float64Array(nCtrl); // start from the centerline, like the Python version
  let { value: fx } = objective(x);
  const history = [fx];
  let grad = gradient(x, fx);
  let H = identity(nCtrl);

  for (let iter = 0; iter < maxiter; iter++) {
    let p = matVec(H, grad);
    for (let j = 0; j < nCtrl; j++) p[j] = -p[j];

    // If the quasi-Newton direction isn't a descent direction (can happen
    // once H drifts from PD due to the bound projections), fall back to
    // plain steepest descent for this step.
    if (dot(p, grad) >= 0) {
      p = grad.slice();
      for (let j = 0; j < nCtrl; j++) p[j] = -p[j];
    }

    let step = 1.0;
    let accepted = false;
    let xNew, fNew, result;
    for (let bt = 0; bt < 15; bt++) {
      xNew = clipVec(x.map((xi, j) => xi + step * p[j]), lo, hi);
      ({ value: fNew, result } = objective(xNew));
      if (fNew < fx - 1e-4 * step * Math.abs(dot(grad, p) || 1e-9)) {
        accepted = true;
        break;
      }
      step *= 0.5;
    }

    if (!accepted) break; // converged (no improving step found)

    const s = new Float64Array(nCtrl);
    for (let j = 0; j < nCtrl; j++) s[j] = xNew[j] - x[j];
    const gradNew = gradient(xNew, fNew);
    const y = new Float64Array(nCtrl);
    for (let j = 0; j < nCtrl; j++) y[j] = gradNew[j] - grad[j];

    const sy = dot(s, y);
    if (sy > 1e-10) {
      // Standard BFGS inverse-Hessian update.
      const Hy = matVec(H, y);
      const yHy = dot(y, Hy);
      const rho = 1 / sy;
      const Hnew = identity(nCtrl);
      for (let i = 0; i < nCtrl; i++) {
        for (let j = 0; j < nCtrl; j++) {
          Hnew[i][j] = H[i][j] + (1 + rho * yHy) * rho * s[i] * s[j] - rho * (s[i] * Hy[j] + Hy[i] * s[j]);
        }
      }
      H = Hnew;
    }

    x = xNew; fx = fNew; grad = gradNew;
    history.push(fx);
    if (onProgress && onProgress(iter, result.lap_time) === false) {
      return finalize(x, expand, trackR, gg, history);
    }
  }

  return finalize(x, expand, trackR, gg, history);
}

function finalize(x, expand, trackR, gg, history) {
  const alphaRaw = expand(x);
  const alpha = new Float64Array(alphaRaw.length);
  for (let i = 0; i < alpha.length; i++) alpha[i] = clip(alphaRaw[i], -0.98, 0.98);
  const final = simulateLap(trackR, alpha, gg);
  return { alpha, final, history };
}
