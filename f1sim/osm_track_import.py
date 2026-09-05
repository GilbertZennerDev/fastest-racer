"""Import real track centerlines from OpenStreetMap (Overpass API) and
convert them into the Track JSON format this project uses.

OSM tags F1-style circuits as many short `highway=raceway` ways (one per
named corner/section) rather than a single closed way, so importing has two
non-trivial steps beyond the lat/lon -> local-meter projection:

1. Stitching: chain the disjoint ways into one ordered closed loop by
   matching shared OSM node ids at their endpoints (see `stitch_ways`).
2. Smoothing: raw OSM node placement has GPS/digitization jitter at the
   scale of single nodes, which blows up the finite-difference curvature
   used by f1sim.track (a real corner spans tens of meters; jitter noise
   spans single-digit meters). Feeding raw points in produces spuriously
   tight "phantom corners" and lap times far slower than reality. A
   moving-average smoothing pass over the coordinates removes the jitter
   while preserving actual corner geometry, since corners are wide relative
   to the smoothing window.
"""
import json
import math
import urllib.request
import urllib.parse

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Auxiliary layouts sharing the same bounding box as the main GP circuit
# (kart tracks, motorcycle-only variants, pit lanes) that must be excluded
# before stitching, or the chain-matching will latch onto the wrong branch.
_DEFAULT_EXCLUDE_KEYWORDS = ("kart", "pit lane", "moto layout", "support pit")


def fetch_raceway_ways(min_lat, min_lon, max_lat, max_lon, timeout=50):
    """Query Overpass for all `highway=raceway` ways inside a bounding box.
    Returns the raw list of way elements (each with `nodes` + `geometry`)."""
    query = (
        f"[out:json][timeout:{timeout}][bbox:{min_lat},{min_lon},{max_lat},{max_lon}];"
        f'way["highway"="raceway"];out geom;'
    )
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data)
    with urllib.request.urlopen(req, timeout=timeout + 15) as resp:
        result = json.load(resp)
    return [e for e in result.get("elements", []) if e.get("type") == "way"]


def stitch_ways(ways, exclude_keywords=_DEFAULT_EXCLUDE_KEYWORDS):
    """Chain disjoint OSM ways sharing endpoint node ids into one ordered
    closed-loop point list [(lat, lon), ...]. Raises if the result doesn't
    close into a single loop (leftover ways indicate an ambiguous track
    layout that needs manual `exclude_keywords` tuning)."""
    candidates = []
    for w in ways:
        name = (w.get("tags", {}).get("name") or "").lower()
        if any(k in name for k in exclude_keywords):
            continue
        candidates.append({
            "nodes": list(w["nodes"]),
            "geom": [(g["lat"], g["lon"]) for g in w["geometry"]],
        })

    if not candidates:
        raise ValueError("No raceway ways left after exclusion filter")

    remaining = candidates[:]
    chain = [remaining.pop(0)]

    def extend():
        tail, head = chain[-1]["nodes"][-1], chain[0]["nodes"][0]
        for i, w in enumerate(remaining):
            if w["nodes"][0] == tail:
                chain.append(remaining.pop(i)); return True
            if w["nodes"][-1] == tail:
                w["nodes"].reverse(); w["geom"].reverse()
                chain.append(remaining.pop(i)); return True
            if w["nodes"][-1] == head:
                chain.insert(0, remaining.pop(i)); return True
            if w["nodes"][0] == head:
                w["nodes"].reverse(); w["geom"].reverse()
                chain.insert(0, remaining.pop(i)); return True
        return False

    while extend():
        pass

    if remaining:
        raise ValueError(
            f"{len(remaining)} way(s) did not chain into the main loop "
            f"(try widening exclude_keywords): leftover node-id ranges present"
        )
    if chain[0]["nodes"][0] != chain[-1]["nodes"][-1]:
        raise ValueError("Chained path did not close into a loop")

    points = []
    for w in chain:
        pts = w["geom"]
        if points and points[-1] == pts[0]:
            pts = pts[1:]
        points.extend(pts)
    return points


def latlon_to_local_xy(points):
    """Equirectangular projection centered on the point-set centroid.
    Accurate to well under 1% distance error over a track-sized area
    (a few km), which is far below GPS/OSM digitization error itself."""
    lat0 = sum(p[0] for p in points) / len(points)
    lon0 = sum(p[1] for p in points) / len(points)
    R = 6371000.0
    cos_lat0 = math.cos(math.radians(lat0))

    xy = []
    for lat, lon in points:
        x = math.radians(lon - lon0) * R * cos_lat0
        y = math.radians(lat - lat0) * R
        xy.append((x, y))
    return xy


def smooth_closed_polyline(xy, window_m=15.0):
    """Moving-average smoothing over a closed loop, window sized in meters
    (converted to a point-count via the local average point spacing) rather
    than a fixed point count, since OSM node density varies a lot section
    to section. Removes GPS/digitization jitter (~single-digit-meter scale)
    while preserving real corner geometry (tens-of-meters scale)."""
    n = len(xy)
    seg_lens = [math.hypot(xy[(i + 1) % n][0] - xy[i][0], xy[(i + 1) % n][1] - xy[i][1]) for i in range(n)]
    avg_spacing = sum(seg_lens) / n
    half_win = max(1, round(window_m / max(avg_spacing, 0.1) / 2))

    smoothed = []
    for i in range(n):
        idxs = range(i - half_win, i + half_win + 1)
        xs = [xy[j % n][0] for j in idxs]
        ys = [xy[j % n][1] for j in idxs]
        smoothed.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return smoothed


def build_track_json(name, min_lat, min_lon, max_lat, max_lon, width_m=12.0, smoothing_window_m=15.0):
    """End-to-end: fetch -> stitch -> project -> smooth -> Track-JSON dict."""
    ways = fetch_raceway_ways(min_lat, min_lon, max_lat, max_lon)
    latlon = stitch_ways(ways)
    xy = latlon_to_local_xy(latlon)
    xy = smooth_closed_polyline(xy, window_m=smoothing_window_m)
    return {
        "name": name,
        "closed": True,
        "width": width_m,
        "points": [[round(x, 2), round(y, 2)] for x, y in xy],
    }


def save_track_json(track_dict, path):
    with open(path, "w") as f:
        json.dump(track_dict, f, indent=2)


if __name__ == "__main__":
    # Example usage / CLI: python -m f1sim.osm_track_import <name> <minlat> <minlon> <maxlat> <maxlon> <out.json>
    #
    # Bounding boxes for a few circuits (tune exclude_keywords per track —
    # OSM tags every named corner as its own way, and circuits with multiple
    # historical/alternate layouts sharing the same site (e.g. Monza's old
    # high-speed oval "Sopraelevata", Silverstone's separate "Stowe Circuit"
    # driving-experience track) need extra keywords excluded so stitch_ways
    # doesn't latch onto the wrong branch. Verified working out of the box:
    #   Spa-Francorchamps: 50.41,5.94,50.46,6.01  (stitches clean, 30 ways, 6980m)
    # Known to need manual tuning (left as a follow-up):
    #   Monza:       45.60,9.27,45.63,9.31   (exclude oval/Pirelli/pit segments)
    #   Silverstone: 52.06,-1.05,52.08,-0.99 (exclude Stowe Circuit + pit lanes)
    import sys
    if len(sys.argv) != 7:
        print(__doc__)
        print("Usage: python -m f1sim.osm_track_import <name> <min_lat> <min_lon> <max_lat> <max_lon> <out.json>")
        sys.exit(1)
    name, min_lat, min_lon, max_lat, max_lon, out_path = sys.argv[1:]
    track = build_track_json(name, float(min_lat), float(min_lon), float(max_lat), float(max_lon))
    save_track_json(track, out_path)
    print(f"Wrote {len(track['points'])} points to {out_path}")
