// Cubic-spline helpers mirroring scipy.interpolate.CubicSpline's two modes
// used by the Python line_optimizer: natural boundary conditions for open
// tracks, periodic (cyclic) boundary conditions for closed ones. Both solve
// for the second derivatives at each knot via a tridiagonal system, then
// evaluate the piecewise cubic — standard textbook natural-cubic-spline
// construction, just written out explicitly since there's no scipy here.

function thomasSolve(a, b, c, d) {
  // Solves a tridiagonal system with sub-diagonal a, diagonal b,
  // super-diagonal c, right-hand side d (all length n; a[0] and c[n-1]
  // unused). Returns the solution vector.
  const n = b.length;
  const cp = new Float64Array(n);
  const dp = new Float64Array(n);
  cp[0] = c[0] / b[0];
  dp[0] = d[0] / b[0];
  for (let i = 1; i < n; i++) {
    const m = b[i] - a[i] * cp[i - 1];
    cp[i] = c[i] / m;
    dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
  }
  const x = new Float64Array(n);
  x[n - 1] = dp[n - 1];
  for (let i = n - 2; i >= 0; i--) {
    x[i] = dp[i] - cp[i] * x[i + 1];
  }
  return x;
}

function cyclicTridiagSolve(a, b, c, d, alpha, beta) {
  // Sherman-Morrison solve for a cyclic tridiagonal system, i.e. a normal
  // tridiagonal system (a,b,c,d) plus a correction alpha in the top-right
  // corner and beta in the bottom-left corner (the wraparound terms).
  const n = b.length;
  const gamma = -b[0];
  const bMod = b.slice();
  bMod[0] -= gamma;
  bMod[n - 1] -= alpha * beta / gamma;

  const y = thomasSolve(a, bMod, c, d);

  const u = new Float64Array(n);
  u[0] = gamma;
  u[n - 1] = alpha;
  const z = thomasSolve(a, bMod, c, u);

  const fact = (y[0] + beta * y[n - 1] / gamma) / (1 + z[0] + beta * z[n - 1] / gamma);
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) x[i] = y[i] - fact * z[i];
  return x;
}

/** Natural cubic spline over knots (xs, ys). Returns eval(x). */
function naturalCubicSpline(xs, ys) {
  const n = xs.length;
  if (n < 3) {
    // Degenerate: linear interpolation fallback.
    return (x) => {
      if (x <= xs[0]) return ys[0];
      if (x >= xs[n - 1]) return ys[n - 1];
      for (let i = 0; i < n - 1; i++) {
        if (x <= xs[i + 1]) {
          const t = (x - xs[i]) / (xs[i + 1] - xs[i]);
          return ys[i] + t * (ys[i + 1] - ys[i]);
        }
      }
      return ys[n - 1];
    };
  }

  const h = new Float64Array(n - 1);
  for (let i = 0; i < n - 1; i++) h[i] = xs[i + 1] - xs[i];

  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), d = new Float64Array(n);
  b[0] = 1; d[0] = 0;
  b[n - 1] = 1; d[n - 1] = 0;
  for (let i = 1; i < n - 1; i++) {
    a[i] = h[i - 1];
    b[i] = 2 * (h[i - 1] + h[i]);
    c[i] = h[i];
    d[i] = 6 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
  }
  const M = thomasSolve(a, b, c, d);

  return (x) => {
    let lo = 0, hi = n - 1;
    if (x <= xs[0]) { lo = 0; hi = 1; }
    else if (x >= xs[n - 1]) { lo = n - 2; hi = n - 1; }
    else {
      while (hi - lo > 1) {
        const mid = (lo + hi) >> 1;
        if (xs[mid] <= x) lo = mid; else hi = mid;
      }
    }
    const hi_ = xs[hi] - xs[lo];
    const A = (xs[hi] - x) / hi_;
    const B = (x - xs[lo]) / hi_;
    return A * ys[lo] + B * ys[hi]
      + ((A * A * A - A) * M[lo] + (B * B * B - B) * M[hi]) * (hi_ * hi_) / 6;
  };
}

/** Periodic cubic spline over n knots spanning [0, total) with wraparound. */
function periodicCubicSpline(xs, ys, total) {
  const n = xs.length;
  const h = new Float64Array(n);
  for (let i = 0; i < n; i++) h[i] = (i === n - 1 ? total - xs[i] + xs[0] : xs[i + 1] - xs[i]);

  const a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n), d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const im1 = (i - 1 + n) % n, ip1 = (i + 1) % n;
    const hPrev = h[im1], hCur = h[i];
    a[i] = hPrev;
    b[i] = 2 * (hPrev + hCur);
    c[i] = hCur;
    d[i] = 6 * ((ys[ip1] - ys[i]) / hCur - (ys[i] - ys[im1]) / hPrev);
  }
  const M = cyclicTridiagSolve(a, b, c, d, h[n - 1], h[n - 1]);

  return (xQuery) => {
    let x = xQuery % total;
    if (x < 0) x += total;
    let lo = 0;
    for (let i = 0; i < n; i++) {
      if (x >= xs[i]) lo = i; else break;
    }
    const hi = (lo + 1) % n;
    const hSeg = h[lo];
    let xHi = xs[hi];
    if (hi === 0) xHi = xs[lo] + hSeg;
    const A = (xHi - x) / hSeg;
    const B = (x - xs[lo]) / hSeg;
    return A * ys[lo] + B * ys[hi]
      + ((A * A * A - A) * M[lo] + (B * B * B - B) * M[hi]) * (hSeg * hSeg) / 6;
  };
}
