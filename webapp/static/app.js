const trackSelect = document.getElementById("trackSelect");
const carSelect = document.getElementById("carSelect");
const spacingInput = document.getElementById("spacing");
const maxiterInput = document.getElementById("maxiter");
const spacingOut = document.getElementById("spacingOut");
const maxiterOut = document.getElementById("maxiterOut");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");
const trackMetaEl = document.getElementById("trackMeta");
const trackCanvas = document.getElementById("trackCanvas");
const speedCanvas = document.getElementById("speedCanvas");
const deltaCanvas = document.getElementById("deltaCanvas");
const ggCanvas = document.getElementById("ggCanvas");

let tracks = {};
let cars = {};

const G = 9.81;
const SPEED_LO = [67, 97, 238]; // #4361ee
const SPEED_HI = [255, 183, 3]; // #ffb703

function lerpColor(a, b, t) {
  return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)}, ${Math.round(a[1] + (b[1] - a[1]) * t)}, ${Math.round(a[2] + (b[2] - a[2]) * t)})`;
}

async function loadOptions() {
  const [trackRes, carRes] = await Promise.all([
    fetch("/api/tracks").then((r) => r.json()),
    fetch("/api/cars").then((r) => r.json()),
  ]);
  tracks = trackRes;
  cars = carRes;

  for (const key of Object.keys(tracks)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = tracks[key].name || key;
    trackSelect.appendChild(opt);
  }
  for (const key of Object.keys(cars)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = cars[key].name || key;
    carSelect.appendChild(opt);
  }
}

function setStatus(msg, isError) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", !!isError);
}

function setLoading(loading) {
  runBtn.disabled = loading;
  runBtn.querySelector(".btn-spinner").hidden = !loading;
  runBtn.querySelector(".btn-label").textContent = loading ? "Optimizing…" : "Run simulation";
}

function fmtTime(t) {
  return `${t.toFixed(3)} s`;
}

function fmtKmh(v) {
  return `${Math.round(v)} km/h`;
}

function drawTrack(result) {
  const ctx = trackCanvas.getContext("2d");
  const w = trackCanvas.width;
  const h = trackCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const allPts = [...result.left_edge, ...result.right_edge];
  const xs = allPts.map((p) => p[0]);
  const ys = allPts.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const pad = 30;
  const scale = Math.min((w - 2 * pad) / (maxX - minX || 1), (h - 2 * pad) / (maxY - minY || 1));

  const tx = (x) => pad + (x - minX) * scale;
  const ty = (y) => h - pad - (y - minY) * scale; // flip Y for screen coords

  function drawPoly(points, style, close) {
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = tx(p[0]), y = ty(p[1]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    if (close) ctx.closePath();
    ctx.strokeStyle = style;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  drawPoly(result.left_edge, "#3a4150", true);
  drawPoly(result.right_edge, "#3a4150", true);

  const speeds = result.speed_mps;
  const maxSpeed = Math.max(...speeds);
  const minSpeed = Math.min(...speeds);

  document.getElementById("legendMin").textContent = fmtKmh(minSpeed * 3.6);
  document.getElementById("legendMax").textContent = fmtKmh(maxSpeed * 3.6);

  for (let i = 0; i < result.path.length; i++) {
    const p = result.path[i];
    const next = result.path[(i + 1) % result.path.length];
    const t = (speeds[i] - minSpeed) / (maxSpeed - minSpeed || 1);
    ctx.beginPath();
    ctx.moveTo(tx(p[0]), ty(p[1]));
    ctx.lineTo(tx(next[0]), ty(next[1]));
    ctx.strokeStyle = lerpColor(SPEED_LO, SPEED_HI, t);
    ctx.lineWidth = 3;
    ctx.stroke();
  }

  // start/finish marker
  const s = result.path[0];
  ctx.beginPath();
  ctx.arc(tx(s[0]), ty(s[1]), 5, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
}

function drawAxes(ctx, w, h, pad) {
  ctx.strokeStyle = "#2a303a";
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.stroke();
}

function drawSpeedProfile(result) {
  const ctx = speedCanvas.getContext("2d");
  const w = speedCanvas.width;
  const h = speedCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const dist = result.distance_m;
  const speedsKmh = result.speed_mps.map((v) => v * 3.6);
  const baseSpeedsKmh = result.baseline_speed_mps.map((v) => v * 3.6);
  const maxD = Math.max(...dist);
  const maxV = Math.max(...speedsKmh, ...baseSpeedsKmh);
  const pad = 26;

  const tx = (d) => pad + (d / maxD) * (w - 2 * pad);
  const ty = (v) => h - pad - (v / maxV) * (h - 2 * pad);

  drawAxes(ctx, w, h, pad);

  function plotLine(distArr, valsKmh, color, width) {
    ctx.beginPath();
    distArr.forEach((d, i) => {
      const x = tx(d), y = ty(valsKmh[i]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
  }

  plotLine(result.baseline_distance_m, baseSpeedsKmh, "#555f6e", 1.5);
  plotLine(dist, speedsKmh, "#3ddc84", 2);

  ctx.fillStyle = "#9aa4b2";
  ctx.font = "11px sans-serif";
  ctx.fillText("optimized", pad + 4, pad + 12);
  ctx.fillStyle = "#3ddc84";
  ctx.fillRect(pad - 4, pad + 5, 8, 3);
  ctx.fillStyle = "#9aa4b2";
  ctx.fillText("centerline", pad + 4, pad + 28);
  ctx.fillStyle = "#555f6e";
  ctx.fillRect(pad - 4, pad + 21, 8, 3);
}

function drawDeltaChart(result) {
  const ctx = deltaCanvas.getContext("2d");
  const w = deltaCanvas.width;
  const h = deltaCanvas.height;
  ctx.clearRect(0, 0, w, h);

  // Cumulative time saved at matching distance: baseline cumulative time
  // minus optimized cumulative time, resampled onto the optimized line's
  // distance grid (both cover the same track length).
  const dist = result.distance_m;
  const dt = result.dt_s;
  const baseDist = result.baseline_distance_m;
  const baseDt = result.baseline_dt_s;

  const baseCumTime = [];
  let acc = 0;
  for (const d of baseDt) { acc += d; baseCumTime.push(acc); }

  const optCumTime = [];
  acc = 0;
  for (const d of dt) { acc += d; optCumTime.push(acc); }

  function interpAt(xs, ys, x) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0, hi = xs.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xs[mid] <= x) lo = mid; else hi = mid;
    }
    const t = (x - xs[lo]) / (xs[hi] - xs[lo]);
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }

  const delta = dist.map((d, i) => interpAt(baseDist, baseCumTime, d) - optCumTime[i]);

  const maxD = Math.max(...dist);
  const maxDelta = Math.max(...delta, 0.01);
  const minDelta = Math.min(...delta, 0);
  const pad = 22;

  const tx = (d) => pad + (d / maxD) * (w - 2 * pad);
  const ty = (v) => h - pad - ((v - minDelta) / (maxDelta - minDelta || 1)) * (h - 2 * pad);

  drawAxes(ctx, w, h, pad);

  // zero line
  ctx.strokeStyle = "#3a4150";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(pad, ty(0));
  ctx.lineTo(w - pad, ty(0));
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.beginPath();
  dist.forEach((d, i) => {
    const x = tx(d), y = ty(delta[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#3ddc84";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = "#9aa4b2";
  ctx.font = "11px sans-serif";
  ctx.fillText(`+${maxDelta.toFixed(2)}s`, pad + 4, pad + 10);
}

function drawGGDiagram(result) {
  const ctx = ggCanvas.getContext("2d");
  const w = ggCanvas.width;
  const h = ggCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const speed = result.speed_mps;
  const kappa = result.curvature;
  const ds = result.distance_m.map((d, i, arr) => (i === 0 ? d : d - arr[i - 1]));

  // lateral accel from v^2 * kappa; longitudinal accel from dv/dt along path
  const aLat = speed.map((v, i) => (v * v * kappa[i]) / G);
  const aLong = [];
  for (let i = 0; i < speed.length; i++) {
    const j = (i + 1) % speed.length;
    const dt = ds[j] / Math.max(speed[i], 0.1);
    aLong.push((speed[j] - speed[i]) / Math.max(dt, 1e-3) / G);
  }

  const maxAbs = Math.max(1, ...aLat.map(Math.abs), ...aLong.map(Math.abs));
  const pad = 24;
  const cx = w / 2, cy = h / 2;
  const scale = (Math.min(w, h) / 2 - pad) / maxAbs;

  ctx.strokeStyle = "#2a303a";
  ctx.beginPath();
  ctx.moveTo(pad, cy); ctx.lineTo(w - pad, cy);
  ctx.moveTo(cx, pad); ctx.lineTo(cx, h - pad);
  ctx.stroke();

  // reference circle at 1g, 2g...
  for (let g = 1; g <= Math.ceil(maxAbs); g++) {
    ctx.beginPath();
    ctx.arc(cx, cy, g * scale, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(154,164,178,0.15)";
    ctx.stroke();
  }

  for (let i = 0; i < aLat.length; i++) {
    const x = cx + aLat[i] * scale;
    const y = cy - aLong[i] * scale;
    ctx.beginPath();
    ctx.arc(x, y, 1.6, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(61,220,132,0.55)";
    ctx.fill();
  }

  ctx.fillStyle = "#9aa4b2";
  ctx.font = "11px sans-serif";
  ctx.fillText("brake", cx - 16, pad + 10);
  ctx.fillText("accel", cx - 16, h - pad - 4);
  ctx.fillText("lat", w - pad - 20, cy - 6);
}

async function runSimulation() {
  const trackKey = trackSelect.value;
  const carKey = carSelect.value;
  if (!trackKey || !carKey) return;

  setLoading(true);
  setStatus("Building GG-diagram and optimizing line…");

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track: tracks[trackKey],
        car: cars[carKey],
        spacing: parseFloat(spacingInput.value),
        maxiter: parseInt(maxiterInput.value, 10),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const result = await res.json();

    trackMetaEl.hidden = false;
    document.getElementById("metaTrackName").textContent = result.track_name;
    document.getElementById("metaLength").textContent = `${(result.track_length_m / 1000).toFixed(2)} km`;
    document.getElementById("metaCarName").textContent = result.car_name;

    document.getElementById("baselineTime").textContent = fmtTime(result.baseline_lap_time);
    document.getElementById("optimizedTime").textContent = fmtTime(result.optimized_lap_time);
    document.getElementById("gainTime").textContent = `-${result.improvement_s.toFixed(3)} s`;

    const speedsKmh = result.speed_mps.map((v) => v * 3.6);
    document.getElementById("topSpeed").textContent = fmtKmh(Math.max(...speedsKmh));
    document.getElementById("avgSpeed").textContent = fmtKmh(
      speedsKmh.reduce((a, b) => a + b, 0) / speedsKmh.length
    );
    document.getElementById("minSpeed").textContent = fmtKmh(Math.min(...speedsKmh));

    resultsEl.hidden = false;
    drawTrack(result);
    drawSpeedProfile(result);
    drawDeltaChart(result);
    drawGGDiagram(result);
    setStatus(`Done — ${result.history.length} optimizer iterations.`);
  } catch (e) {
    setStatus(`Error: ${e.message}`, true);
  } finally {
    setLoading(false);
  }
}

spacingInput.addEventListener("input", () => { spacingOut.textContent = parseFloat(spacingInput.value).toFixed(1); });
maxiterInput.addEventListener("input", () => { maxiterOut.textContent = maxiterInput.value; });
runBtn.addEventListener("click", runSimulation);

loadOptions().then(() => {
  setStatus("Ready.");
  if (trackSelect.value && carSelect.value) {
    runSimulation();
  }
});
