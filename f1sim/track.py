"""Track loading and geometry: centerline, resampling, curvature, normals."""
import json
import numpy as np


class Track:
    def __init__(self, name, points, width, closed=True):
        """points: (N,2) raw centerline waypoints in meters.
        width: scalar or (N,) array of track width in meters.
        closed: whether the track loops back to the start."""
        self.name = name
        self.closed = closed
        self.raw_points = np.asarray(points, dtype=float)
        n = len(self.raw_points)
        self.raw_width = np.full(n, width, dtype=float) if np.isscalar(width) else np.asarray(width, dtype=float)

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name", "unnamed"),
            points=data["points"],
            width=data.get("width", 12.0),
            closed=data.get("closed", True),
        )

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def resample(self, spacing=2.0):
        """Resample centerline to ~uniform arc-length spacing. Returns a
        RacingLine-ready set of arrays: points, width, tangent, normal, s."""
        pts = self.raw_points
        w = self.raw_width
        if self.closed:
            pts = np.vstack([pts, pts[0]])
            w = np.append(w, w[0])

        seg = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        total_len = cum[-1]

        n_samples = max(int(total_len / spacing), 10)
        s_new = np.linspace(0, total_len, n_samples, endpoint=not self.closed)

        x = np.interp(s_new, cum, pts[:, 0])
        y = np.interp(s_new, cum, pts[:, 1])
        w_new = np.interp(s_new, cum, w)

        return ResampledTrack(np.column_stack([x, y]), w_new, self.closed, s_new)


class ResampledTrack:
    """Uniformly-sampled closed/open centerline with derived geometry."""

    def __init__(self, points, width, closed, s):
        self.points = points          # (N,2)
        self.width = width            # (N,)
        self.closed = closed
        self.s = s                    # cumulative arc length (N,)
        self.n = len(points)
        self.ds = np.gradient(s) if not closed else self._closed_ds()
        self._compute_frames()

    def _closed_ds(self):
        s = self.s
        ds = np.diff(s, append=s[0] + (s[-1] - s[-2]) + (s[1] - s[0]))
        return ds

    def _compute_frames(self):
        pts = self.points
        if self.closed:
            nxt = np.roll(pts, -1, axis=0)
            prv = np.roll(pts, 1, axis=0)
        else:
            nxt = np.vstack([pts[1:], pts[-1:]])
            prv = np.vstack([pts[:1], pts[:-1]])

        tangent = nxt - prv
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        norm[norm == 0] = 1e-9
        tangent = tangent / norm
        # left-hand normal (90 deg CCW)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

        self.tangent = tangent
        self.normal = normal

    def curvature_of_offset(self, alpha):
        """Curvature of the path offset by alpha (fraction of half-width,
        -1..1, +1 = full offset toward the left normal) at each sample.
        Computed via finite differences on the offset path itself."""
        offset = (alpha * self.width / 2.0)[:, None] * self.normal
        path = self.points + offset

        if self.closed:
            p_prev = np.roll(path, 1, axis=0)
            p_next = np.roll(path, -1, axis=0)
        else:
            p_prev = np.vstack([path[:1], path[:-1]])
            p_next = np.vstack([path[1:], path[-1:]])

        d1 = p_next - path
        d2 = path - p_prev
        ds1 = np.linalg.norm(d1, axis=1)
        ds2 = np.linalg.norm(d2, axis=1)
        ds1[ds1 == 0] = 1e-9
        ds2[ds2 == 0] = 1e-9

        cross = d2[:, 0] * d1[:, 1] - d2[:, 1] * d1[:, 0]
        kappa = 2 * cross / (ds1 * ds2 * (ds1 + ds2))
        return kappa, path

    def path_length(self, alpha):
        _, path = self.curvature_of_offset(alpha)
        if self.closed:
            nxt = np.roll(path, -1, axis=0)
        else:
            nxt = np.vstack([path[1:], path[-1:]])
        seg = np.linalg.norm(nxt - path, axis=1)
        return seg
