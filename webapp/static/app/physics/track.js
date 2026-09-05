// JS port of f1sim/track.py: resample a raw centerline to uniform arc-length
// spacing, and compute curvature/path-length for an offset (racing-line)
// path. Mirrors the Python implementation function-for-function.

function resampleTrack(trackDict, spacing = 2.0) {
  const closed = trackDict.closed !== false;
  let pts = trackDict.points.map((p) => [p[0], p[1]]);
  const rawWidth = trackDict.width ?? 12.0;
  let w = Array.isArray(rawWidth) ? rawWidth.slice() : new Array(pts.length).fill(rawWidth);

  if (closed) {
    pts = [...pts, pts[0]];
    w = [...w, w[0]];
  }

  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    const dx = pts[i][0] - pts[i - 1][0];
    const dy = pts[i][1] - pts[i - 1][1];
    cum.push(cum[i - 1] + Math.hypot(dx, dy));
  }
  const totalLen = cum[cum.length - 1];
  const nSamples = Math.max(Math.floor(totalLen / spacing), 10);

  const sNew = [];
  if (closed) {
    for (let i = 0; i < nSamples; i++) sNew.push((i / nSamples) * totalLen);
  } else {
    for (let i = 0; i < nSamples; i++) sNew.push((i / (nSamples - 1)) * totalLen);
  }

  function interpAlong(cumArr, valsArr, sArr) {
    const out = new Float64Array(sArr.length);
    let idx = 0;
    for (let k = 0; k < sArr.length; k++) {
      const s = sArr[k];
      while (idx < cumArr.length - 2 && cumArr[idx + 1] < s) idx++;
      const t = (s - cumArr[idx]) / (cumArr[idx + 1] - cumArr[idx] || 1);
      out[k] = valsArr[idx] + t * (valsArr[idx + 1] - valsArr[idx]);
    }
    return out;
  }

  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const x = interpAlong(cum, xs, sNew);
  const y = interpAlong(cum, ys, sNew);
  const wNew = interpAlong(cum, w, sNew);

  const points = [];
  for (let i = 0; i < nSamples; i++) points.push([x[i], y[i]]);

  return new ResampledTrack(points, wNew, closed, sNew);
}

class ResampledTrack {
  constructor(points, width, closed, s) {
    this.points = points;
    this.width = width;
    this.closed = closed;
    this.s = s;
    this.n = points.length;
    this._computeFrames();
  }

  _computeFrames() {
    const n = this.n;
    const tangent = new Array(n);
    const normal = new Array(n);
    for (let i = 0; i < n; i++) {
      let prev, next;
      if (this.closed) {
        prev = this.points[(i - 1 + n) % n];
        next = this.points[(i + 1) % n];
      } else {
        prev = this.points[Math.max(i - 1, 0)];
        next = this.points[Math.min(i + 1, n - 1)];
      }
      let dx = next[0] - prev[0], dy = next[1] - prev[1];
      const norm = Math.hypot(dx, dy) || 1e-9;
      dx /= norm; dy /= norm;
      tangent[i] = [dx, dy];
      normal[i] = [-dy, dx]; // left-hand normal (90deg CCW)
    }
    this.tangent = tangent;
    this.normal = normal;
  }

  /** Curvature of the path offset by alpha (fraction of half-width, -1..1). */
  curvatureOfOffset(alpha) {
    const n = this.n;
    const path = new Array(n);
    for (let i = 0; i < n; i++) {
      const off = (alpha[i] * this.width[i]) / 2.0;
      path[i] = [this.points[i][0] + off * this.normal[i][0], this.points[i][1] + off * this.normal[i][1]];
    }

    const kappa = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let prev, next;
      if (this.closed) {
        prev = path[(i - 1 + n) % n];
        next = path[(i + 1) % n];
      } else {
        prev = path[Math.max(i - 1, 0)];
        next = path[Math.min(i + 1, n - 1)];
      }
      const d1x = next[0] - path[i][0], d1y = next[1] - path[i][1];
      const d2x = path[i][0] - prev[0], d2y = path[i][1] - prev[1];
      let ds1 = Math.hypot(d1x, d1y) || 1e-9;
      let ds2 = Math.hypot(d2x, d2y) || 1e-9;
      const cross = d2x * d1y - d2y * d1x;
      kappa[i] = (2 * cross) / (ds1 * ds2 * (ds1 + ds2));
    }
    return { kappa, path };
  }

  pathLength(alpha) {
    const { path } = this.curvatureOfOffset(alpha);
    const n = this.n;
    const seg = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const next = this.closed ? path[(i + 1) % n] : path[Math.min(i + 1, n - 1)];
      seg[i] = Math.hypot(next[0] - path[i][0], next[1] - path[i][1]);
    }
    return seg;
  }
}
