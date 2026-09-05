// Web Worker: runs the whole simulation + optimization locally in the
// browser (off the main thread, so the UI stays responsive), using only the
// physics files in this folder — no server round-trip for any computation.
importScripts("spline.js", "track.js", "vehicle_dynamics.js", "lap_sim.js", "optimizer.js");

self.onmessage = (e) => {
  const { track, car, spacing, maxiter, nCtrl } = e.data;

  try {
    const preparedCar = prepareCar(car);
    const trackR = resampleTrack(track, spacing);
    if (trackR.n < 8) {
      self.postMessage({ type: "error", message: "Track too short/coarse after resampling." });
      return;
    }

    const gg = buildGGLookup(preparedCar);
    const alpha0 = new Float64Array(trackR.n);
    const baseline = simulateLap(trackR, alpha0, gg);

    const { final, history } = optimizeLine(trackR, gg, {
      maxiter,
      nCtrl,
      onProgress: (iter, lapTime) => {
        self.postMessage({ type: "progress", iter, lapTime, total: maxiter });
        return true;
      },
    });

    const n = trackR.n;
    const leftEdge = new Array(n), rightEdge = new Array(n);
    for (let i = 0; i < n; i++) {
      const halfW = trackR.width[i] / 2;
      leftEdge[i] = [trackR.points[i][0] + halfW * trackR.normal[i][0], trackR.points[i][1] + halfW * trackR.normal[i][1]];
      rightEdge[i] = [trackR.points[i][0] - halfW * trackR.normal[i][0], trackR.points[i][1] - halfW * trackR.normal[i][1]];
    }

    let distAcc = 0;
    const distanceM = Array.from(final.ds).map((d) => (distAcc += d));
    let baseDistAcc = 0;
    const baselineDistanceM = Array.from(baseline.ds).map((d) => (baseDistAcc += d));

    self.postMessage({
      type: "done",
      result: {
        track_name: track.name,
        car_name: car.name,
        track_length_m: distanceM[distanceM.length - 1],
        baseline_lap_time: baseline.lap_time,
        optimized_lap_time: final.lap_time,
        improvement_s: baseline.lap_time - final.lap_time,
        path: final.path,
        speed_mps: Array.from(final.speed),
        curvature: Array.from(final.kappa),
        distance_m: distanceM,
        dt_s: Array.from(final.dt),
        baseline_speed_mps: Array.from(baseline.speed),
        baseline_distance_m: baselineDistanceM,
        baseline_dt_s: Array.from(baseline.dt),
        left_edge: leftEdge,
        right_edge: rightEdge,
        history,
      },
    });
  } catch (err) {
    self.postMessage({ type: "error", message: err.message || String(err) });
  }
};
