const trackSelect = document.getElementById("trackSelect");
const carSelect = document.getElementById("carSelect");
const runBtn = document.getElementById("runBtn");
const statusEl = document.getElementById("status");
const trackCanvas = document.getElementById("trackCanvas");
const speedCanvas = document.getElementById("speedCanvas");

let tracks = {};
let cars = {};

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

function setStatus(msg) {
  statusEl.textContent = msg;
}

function fmtTime(t) {
  return `${t.toFixed(3)} s`;
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

  for (let i = 0; i < result.path.length; i++) {
    const p = result.path[i];
    const next = result.path[(i + 1) % result.path.length];
    const t = (speeds[i] - minSpeed) / (maxSpeed - minSpeed || 1);
    const r = Math.round(255 * (1 - t));
    const g = Math.round(180 + 75 * t);
    ctx.beginPath();
    ctx.moveTo(tx(p[0]), ty(p[1]));
    ctx.lineTo(tx(next[0]), ty(next[1]));
    ctx.strokeStyle = `rgb(${r}, ${g}, 90)`;
    ctx.lineWidth = 3;
    ctx.stroke();
  }
}

function drawSpeedProfile(result) {
  const ctx = speedCanvas.getContext("2d");
  const w = speedCanvas.width;
  const h = speedCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const dist = result.distance_m;
  const speedsKmh = result.speed_mps.map((v) => v * 3.6);
  const maxD = Math.max(...dist);
  const maxV = Math.max(...speedsKmh);
  const pad = 30;

  const tx = (d) => pad + (d / maxD) * (w - 2 * pad);
  const ty = (v) => h - pad - (v / maxV) * (h - 2 * pad);

  ctx.strokeStyle = "#2a303a";
  ctx.beginPath();
  ctx.moveTo(pad, h - pad);
  ctx.lineTo(w - pad, h - pad);
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, h - pad);
  ctx.stroke();

  ctx.beginPath();
  dist.forEach((d, i) => {
    const x = tx(d), y = ty(speedsKmh[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = "#3ddc84";
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = "#9aa4b2";
  ctx.font = "12px sans-serif";
  ctx.fillText("Speed (km/h) vs distance (m)", pad, 16);
}

async function runSimulation() {
  const trackKey = trackSelect.value;
  const carKey = carSelect.value;
  if (!trackKey || !carKey) return;

  runBtn.disabled = true;
  setStatus("Building GG-diagram and optimizing line…");

  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        track: tracks[trackKey],
        car: cars[carKey],
        spacing: parseFloat(document.getElementById("spacing").value),
        maxiter: parseInt(document.getElementById("maxiter").value, 10),
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const result = await res.json();
    document.getElementById("baselineTime").textContent = fmtTime(result.baseline_lap_time);
    document.getElementById("optimizedTime").textContent = fmtTime(result.optimized_lap_time);
    document.getElementById("gainTime").textContent = `-${result.improvement_s.toFixed(3)} s`;

    drawTrack(result);
    drawSpeedProfile(result);
    setStatus(`Done — ${result.history.length} optimizer iterations.`);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", runSimulation);
loadOptions().then(() => setStatus("Ready."));
