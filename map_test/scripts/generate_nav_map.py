#!/usr/bin/env python3
"""
Generate a 2D navigation map by "building a track" along the trajectory.

Pipeline:
  1. Load global RGB PCD (ground=green) + trajectory poses
  2. Project green points to 2D grid
  3. Distance transform on green mask → clearance at each cell
  4. Sample clearance at trajectory points → per-point half-width
  5. Smooth and cap widths → clean "track" / "runway" corridor
  6. Delete all 3D points inside the track corridor
  7. Flatten remaining → 2D occupancy grid (PGM + YAML)

Usage:
  python3 generate_nav_map.py \
    --global-map-pcd map/global_map_ground_green.pcd \
    --pose-file pose.json \
    --output-dir map/ \
    --resolution 0.05 \
    --max-half-width 1.5 \
    --visualize
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a track-style corridor along trajectory → 2D nav map."
    )
    p.add_argument("--global-map-pcd", type=Path, required=True,
                   help="Input global RGB PCD (ground=green).")
    p.add_argument("--pose-file", type=Path, required=True,
                   help="Trajectory pose file (JSON or text).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory for PGM + YAML + optional files.")
    p.add_argument("--map-name", type=str, default="nav_map",
                   help="Base name for output files (default: nav_map).")
    p.add_argument("--resolution", type=float, default=0.05,
                   help="Grid resolution m/pixel (default: 0.05).")
    p.add_argument("--padding", type=float, default=2.0,
                   help="Extra padding around bounds in meters (default: 2.0).")

    g = p.add_argument_group("Ground color")
    g.add_argument("--ground-rgb", type=lambda s: int(s, 0), default=0x00FF00,
                   help="RGB value for ground (default: 0x00FF00 = green).")
    g.add_argument("--color-tolerance", type=int, default=30,
                   help="Per-channel tolerance for ground color (default: 30).")

    t = p.add_argument_group("Track corridor")
    t.add_argument("--min-half-width", type=float, default=0.3,
                   help="Minimum corridor half-width in meters (default: 0.3).")
    t.add_argument("--max-half-width", type=float, default=1.5,
                   help="Maximum corridor half-width in meters (default: 1.5).")
    t.add_argument("--safety-margin", type=float, default=0.3,
                   help="Shrink corridor from green boundary to protect walls (default: 0.3m).")
    t.add_argument("--smoothing-window", type=int, default=30,
                   help="Smooth half-widths over N trajectory points (default: 30).")
    t.add_argument("--min-ground-points", type=int, default=1,
                   help="Min green pts per cell to be traversable (default: 1).")
    t.add_argument("--close-kernel", type=int, default=5,
                   help="Morphological close kernel to fill gaps in green (default: 5, 0=off).")

    o = p.add_argument_group("Output options")
    o.add_argument("--obstacle-inflate", type=float, default=0.0,
                   help="Inflate obstacles by this radius before output (default: 0.0m).")
    o.add_argument("--output-corridor-pcd", type=Path, default=None,
                   help="Save remaining obstacle 3D PCD after corridor deletion.")
    o.add_argument("--visualize", action="store_true",
                   help="Save color visualization PNG.")
    return p.parse_args()


# ─── PCD Reader ───────────────────────────────────────────────────────────────

class BinaryPCD:
    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_offset = self._read_header()
        if self.header.get("DATA", [""])[0].lower() != "binary":
            raise ValueError(f"{path}: only DATA binary supported")
        self.fields = self.header["FIELDS"]
        self.sizes = [int(v) for v in self.header["SIZE"]]
        self.types = self.header["TYPE"]
        self.counts = [int(v) for v in self.header.get("COUNT", ["1"] * len(self.fields))]
        self.points = int(self.header.get("POINTS", [
            str(int(self.header.get("WIDTH", ["0"])[0]) * int(self.header.get("HEIGHT", ["1"])[0]))
        ])[0])
        self.dtype = self._build_dtype()

    def _read_header(self) -> Tuple[Dict[str, List[str]], int]:
        header: Dict[str, List[str]] = {}
        with self.path.open("rb") as fp:
            while True:
                line = fp.readline()
                if not line:
                    raise ValueError(f"{self.path}: EOF in header")
                text = line.decode("utf-8", errors="ignore").strip()
                if not text or text.startswith("#"):
                    continue
                tokens = text.split()
                key = tokens[0].upper()
                header[key] = tokens[1:]
                if key == "DATA":
                    break
            return header, fp.tell()

    def _build_dtype(self) -> np.dtype:
        tables = {
            "F": {4: np.float32, 8: np.float64},
            "U": {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64},
            "I": {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64},
        }
        fields = []
        for name, sz, tc, cnt in zip(self.fields, self.sizes, self.types, self.counts):
            sz = int(sz)
            cnt = int(cnt)
            if tc not in tables or sz not in tables[tc]:
                raise ValueError(f"{self.path}: unsupported TYPE {tc} SIZE {sz}")
            dt = np.dtype(tables[tc][sz])
            if sz > 1:
                dt = dt.newbyteorder("<")
            fields.append((name, dt) if cnt == 1 else (name, dt, (cnt,)))
        return np.dtype(fields)

    def read_all(self) -> np.ndarray:
        with self.path.open("rb") as fp:
            fp.seek(self.data_offset)
            return np.fromfile(fp, dtype=self.dtype, count=self.points)


# ─── Pose Loader ──────────────────────────────────────────────────────────────

def _pose_xy(vals: Sequence[float]) -> Tuple[float, float]:
    if len(vals) == 7:
        return float(vals[0]), float(vals[1])
    if len(vals) in (12, 16):
        return float(vals[3]), float(vals[7])
    raise ValueError(f"Unsupported pose length {len(vals)}")


def load_pose_xy(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    xy: List[Tuple[float, float]] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "poses" in payload:
            payload = payload["poses"]
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    if "x" in item and "y" in item:
                        xy.append((float(item["x"]), float(item["y"])))
                    elif "tx" in item and "ty" in item:
                        xy.append((float(item["tx"]), float(item["ty"])))
                    elif "T" in item:
                        xy.append(_pose_xy([float(v) for v in item["T"]]))
                elif isinstance(item, list):
                    xy.append(_pose_xy([float(v) for v in item]))
            if xy:
                return np.asarray(xy, dtype=np.float64)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        xy.append(_pose_xy([float(v) for v in line.replace(",", " ").split()]))
    if not xy:
        raise ValueError(f"No valid poses in {path}")
    return np.asarray(xy, dtype=np.float64)


# ─── Core Helpers ─────────────────────────────────────────────────────────────

def extract_xyz_rgb(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = points["x"].astype(np.float32, copy=False)
    y = points["y"].astype(np.float32, copy=False)
    z = points["z"].astype(np.float32, copy=False)
    names = set(points.dtype.names or ())
    if "rgb" in names:
        rf = points["rgb"]
        rgb = rf.view(np.uint32).copy() if rf.dtype.kind == "f" else rf.astype(np.uint32)
    elif "rgba" in names:
        rf = points["rgba"]
        rgb = (rf.view(np.uint32).copy() if rf.dtype.kind == "f" else rf.astype(np.uint32)) & np.uint32(0x00FFFFFF)
    else:
        raise ValueError("PCD missing rgb/rgba field")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return np.stack((x[valid], y[valid], z[valid]), axis=1), rgb[valid]


def classify_ground(rgb_u32: np.ndarray, ground_rgb: int, tol: int) -> np.ndarray:
    gr = np.uint32(ground_rgb)
    tr, tg, tb = int((gr >> 16) & 0xFF), int((gr >> 8) & 0xFF), int(gr & 0xFF)
    r = ((rgb_u32 >> 16) & 0xFF).astype(np.int16)
    g = ((rgb_u32 >> 8) & 0xFF).astype(np.int16)
    b = (rgb_u32 & 0xFF).astype(np.int16)
    return (np.abs(r - tr) <= tol) & (np.abs(g - tg) <= tol) & (np.abs(b - tb) <= tol)


def rasterize(xy: np.ndarray, mask: np.ndarray,
              min_x: float, min_y: float, res: float, W: int, H: int) -> np.ndarray:
    xs, ys = xy[mask, 0], xy[mask, 1]
    gx = np.floor((xs - min_x) / res).astype(np.int32)
    gy = np.floor((ys - min_y) / res).astype(np.int32)
    ok = (gx >= 0) & (gy >= 0) & (gx < W) & (gy < H)
    idx = gy[ok] * W + gx[ok]
    return np.bincount(idx, minlength=W * H).reshape(H, W).astype(np.uint32)


def smooth_1d(arr: np.ndarray, window: int) -> np.ndarray:
    """Uniform moving-average smoothing."""
    if window <= 1 or len(arr) < 2:
        return arr.copy()
    w = min(window, len(arr))
    kernel = np.ones(w, dtype=np.float64) / w
    # Pad edges to avoid boundary artifacts
    pad = w // 2
    padded = np.concatenate([np.full(pad, arr[0]), arr, np.full(pad, arr[-1])])
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad:pad + len(arr)]


# ─── Track Corridor Builder ──────────────────────────────────────────────────

def compute_trajectory_directions(pose_xy: np.ndarray) -> np.ndarray:
    """Compute smoothed unit direction + left-normal at each trajectory point."""
    n = len(pose_xy)
    dirs = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        if i == 0:
            d = pose_xy[min(1, n - 1)] - pose_xy[0]
        elif i == n - 1:
            d = pose_xy[-1] - pose_xy[max(0, n - 2)]
        else:
            d = pose_xy[i + 1] - pose_xy[i - 1]
        norm = np.linalg.norm(d)
        dirs[i] = d / norm if norm > 1e-6 else [1.0, 0.0]
    # Smooth directions
    dirs[:, 0] = smooth_1d(dirs[:, 0], 15)
    dirs[:, 1] = smooth_1d(dirs[:, 1], 15)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    dirs /= norms
    return dirs


def ray_cast_green(green_mask: np.ndarray,
                   cx: float, cy: float,
                   dx: float, dy: float,
                   max_steps: int) -> int:
    """Walk from (cx,cy) in direction (dx,dy) on green mask.
    Return number of steps while still on green cells."""
    H, W = green_mask.shape
    ln = math.sqrt(dx * dx + dy * dy)
    if ln < 1e-9:
        return 0
    dx, dy = dx / ln, dy / ln
    for s in range(1, max_steps + 1):
        gx = int(round(cx + dx * s))
        gy = int(round(cy + dy * s))
        if gx < 0 or gx >= W or gy < 0 or gy >= H:
            return s  # out of bounds = edge
        if not green_mask[gy, gx]:
            return s  # hit non-green = boundary
    return max_steps


def compute_asymmetric_clearance(
    green_mask: np.ndarray,
    pose_xy: np.ndarray,
    directions: np.ndarray,
    min_x: float, min_y: float, res: float,
    max_width_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cast perpendicular rays left & right from each trajectory point through
    the green mask.  Returns (left_widths, right_widths) in meters.
    """
    H, W = green_mask.shape
    max_steps = int(max_width_m / res) + 2

    px = (pose_xy[:, 0] - min_x) / res
    py = (pose_xy[:, 1] - min_y) / res

    # Left normal: rotate direction 90° CCW
    left_nx = -directions[:, 1]
    left_ny =  directions[:, 0]

    left_w = np.zeros(len(pose_xy), dtype=np.float64)
    right_w = np.zeros(len(pose_xy), dtype=np.float64)

    for i in range(len(pose_xy)):
        cx_f, cy_f = float(px[i]), float(py[i])
        lnx, lny = float(left_nx[i]), float(left_ny[i])

        steps_l = ray_cast_green(green_mask, cx_f, cy_f,  lnx,  lny, max_steps)
        steps_r = ray_cast_green(green_mask, cx_f, cy_f, -lnx, -lny, max_steps)

        left_w[i] = steps_l * res
        right_w[i] = steps_r * res

    return left_w, right_w


def build_track_corridor_asym(
    pose_xy: np.ndarray,
    directions: np.ndarray,
    left_widths: np.ndarray,
    right_widths: np.ndarray,
    min_x: float, min_y: float, res: float,
    W: int, H: int,
) -> np.ndarray:
    """
    Build corridor mask with independent left/right widths per trajectory point.
    Returns bool mask (H, W).
    """
    px = ((pose_xy[:, 0] - min_x) / res).astype(np.float32)
    py = ((pose_xy[:, 1] - min_y) / res).astype(np.float32)
    # Left normal in grid coords
    lnx = (-directions[:, 1]).astype(np.float32)
    lny = ( directions[:, 0]).astype(np.float32)
    lw = (left_widths / res).astype(np.float32)
    rw = (right_widths / res).astype(np.float32)

    mask = np.zeros((H, W), dtype=np.uint8)

    for i in range(len(px) - 1):
        p0x, p0y = float(px[i]), float(py[i])
        p1x, p1y = float(px[i + 1]), float(py[i + 1])
        n0x, n0y = float(lnx[i]), float(lny[i])
        n1x, n1y = float(lnx[i + 1]), float(lny[i + 1])
        lw0, lw1 = float(lw[i]), float(lw[i + 1])
        rw0, rw1 = float(rw[i]), float(rw[i + 1])

        # Quad: left_start, left_end, right_end, right_start
        poly = np.array([
            [p0x + n0x * lw0, p0y + n0y * lw0],   # left start
            [p1x + n1x * lw1, p1y + n1y * lw1],   # left end
            [p1x - n1x * rw1, p1y - n1y * rw1],   # right end
            [p0x - n0x * rw0, p0y - n0y * rw0],   # right start
        ], dtype=np.float32)
        cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 255)

        # End-cap at segment start (average of L/R)
        avg_w = max(1.0, 0.5 * (lw0 + rw0))
        cv2.circle(mask, (int(round(p0x)), int(round(p0y))),
                   int(round(avg_w)), 255, -1)

    # End-cap for last point
    i = len(px) - 1
    avg_w = max(1.0, 0.5 * (float(lw[i]) + float(rw[i])))
    cv2.circle(mask, (int(round(float(px[i]))), int(round(float(py[i])))),
               int(round(avg_w)), 255, -1)

    return mask > 0


# ─── Output Writers ───────────────────────────────────────────────────────────

def write_pcd_xyz_rgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = xyz.shape[0]
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    out = np.empty(n, dtype=dtype)
    out["x"], out["y"], out["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out["rgb"] = rgb.astype(np.uint32, copy=False).view(np.float32)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\n"
        "TYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\nDATA binary\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(header.encode("ascii"))
        fp.write(np.ascontiguousarray(out).tobytes())


def write_pgm(path: Path, grid: np.ndarray) -> None:
    cv2.imwrite(str(path), np.flipud(grid))


def write_yaml(path: Path, pgm_name: str, res: float, ox: float, oy: float) -> None:
    path.write_text(
        f"image: {pgm_name}\nresolution: {res}\n"
        f"origin: [{ox}, {oy}, 0.0]\nnegate: 0\n"
        f"occupied_thresh: 0.65\nfree_thresh: 0.196\n",
        encoding="utf-8",
    )


def save_visualization(
    path: Path, corridor: np.ndarray, obstacle: np.ndarray,
    pose_xy: np.ndarray, min_x: float, min_y: float, res: float,
) -> None:
    H, W = corridor.shape
    vis = np.full((H, W, 3), 128, dtype=np.uint8)
    vis[corridor] = [200, 255, 200]    # light green = free
    vis[obstacle] = [40, 40, 40]       # dark = obstacle
    px = np.floor((pose_xy[:, 0] - min_x) / res).astype(np.int32)
    py = np.floor((pose_xy[:, 1] - min_y) / res).astype(np.int32)
    for i in range(len(px) - 1):
        if 0 <= px[i] < W and 0 <= py[i] < H and 0 <= px[i+1] < W and 0 <= py[i+1] < H:
            cv2.line(vis, (px[i], py[i]), (px[i+1], py[i+1]), (0, 0, 255), 1)
    cv2.imwrite(str(path), np.flipud(vis))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    print("Step 1: Loading data...")
    pcd = BinaryPCD(args.global_map_pcd)
    xyz, rgb = extract_xyz_rgb(pcd.read_all())
    pose_xy = load_pose_xy(args.pose_file)
    print(f"  Points: {len(xyz)}, Poses: {len(pose_xy)}")

    is_ground = classify_ground(rgb, args.ground_rgb, args.color_tolerance)
    print(f"  Green: {np.sum(is_ground)}, Non-green: {np.sum(~is_ground)}")

    # ── Grid setup ───────────────────────────────────────────────────────
    xy = xyz[:, :2]
    ax = np.concatenate([xy[:, 0], pose_xy[:, 0].astype(np.float32)])
    ay = np.concatenate([xy[:, 1], pose_xy[:, 1].astype(np.float32)])
    min_x = float(np.min(ax)) - args.padding
    max_x = float(np.max(ax)) + args.padding
    min_y = float(np.min(ay)) - args.padding
    max_y = float(np.max(ay)) + args.padding
    W = int(math.ceil((max_x - min_x) / args.resolution))
    H = int(math.ceil((max_y - min_y) / args.resolution))
    print(f"\nStep 2: Grid {W}x{H}, res={args.resolution}m")

    # ── Rasterize green → distance transform ─────────────────────────────
    print("\nStep 3: Building green mask & distance transform...")
    green_counts = rasterize(xy, is_ground, min_x, min_y, args.resolution, W, H)
    green_mask = green_counts >= args.min_ground_points

    # Fill small gaps in ground with morphological close
    if args.close_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (args.close_kernel, args.close_kernel))
        gu8 = green_mask.astype(np.uint8) * 255
        gu8 = cv2.morphologyEx(gu8, cv2.MORPH_CLOSE, k, iterations=1)
        green_mask = gu8 > 0

    print(f"  Green cells: {np.sum(green_mask)}")

    # ── Per-pose clearance: perpendicular ray-cast (left/right independent) ─
    print("\nStep 4: Computing asymmetric clearance (L/R ray-cast)...")
    directions = compute_trajectory_directions(pose_xy)
    left_raw, right_raw = compute_asymmetric_clearance(
        green_mask, pose_xy, directions,
        min_x, min_y, args.resolution, args.max_half_width,
    )
    print(f"  Left  raw — min: {left_raw.min():.2f}m, "
          f"max: {left_raw.max():.2f}m, mean: {left_raw.mean():.2f}m")
    print(f"  Right raw — min: {right_raw.min():.2f}m, "
          f"max: {right_raw.max():.2f}m, mean: {right_raw.mean():.2f}m")

    # Apply safety margin, smooth, and clamp (independently for L/R)
    left_w = smooth_1d(left_raw - args.safety_margin, args.smoothing_window)
    right_w = smooth_1d(right_raw - args.safety_margin, args.smoothing_window)
    left_w = np.clip(left_w, args.min_half_width, args.max_half_width)
    right_w = np.clip(right_w, args.min_half_width, args.max_half_width)
    print(f"  Final left  — min: {left_w.min():.2f}m, max: {left_w.max():.2f}m")
    print(f"  Final right — min: {right_w.min():.2f}m, max: {right_w.max():.2f}m")

    # Save per-pose widths to CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.map_name}_widths.csv"
    with csv_path.open("w") as f:
        f.write("pose_idx,x,y,left_raw,right_raw,left_final,right_final\n")
        for i in range(len(pose_xy)):
            f.write(f"{i},{pose_xy[i,0]:.4f},{pose_xy[i,1]:.4f},"
                    f"{left_raw[i]:.4f},{right_raw[i]:.4f},"
                    f"{left_w[i]:.4f},{right_w[i]:.4f}\n")
    print(f"  Widths CSV: {csv_path}")

    # ── Build track corridor (asymmetric L/R) ─────────────────────────────
    print("\nStep 5: Building asymmetric track corridor...")
    corridor = build_track_corridor_asym(
        pose_xy, directions, left_w, right_w,
        min_x, min_y, args.resolution, W, H,
    )
    print(f"  Corridor cells: {np.sum(corridor)}")

    # ── Delete corridor from 3D, flatten remaining ───────────────────────
    print("\nStep 6: Deleting corridor points, flattening remaining...")
    gx = np.floor((xy[:, 0] - min_x) / args.resolution).astype(np.int32)
    gy = np.floor((xy[:, 1] - min_y) / args.resolution).astype(np.int32)
    in_bounds = (gx >= 0) & (gy >= 0) & (gx < W) & (gy < H)

    in_corridor = np.zeros(len(xyz), dtype=np.bool_)
    in_corridor[in_bounds] = corridor[gy[in_bounds], gx[in_bounds]]

    remaining_mask = ~in_corridor
    print(f"  Deleted (corridor): {np.sum(in_corridor)}")
    print(f"  Remaining (obstacle): {np.sum(remaining_mask)}")

    # Remaining points → obstacle cells
    obs_counts = rasterize(xy, remaining_mask, min_x, min_y, args.resolution, W, H)
    obstacle_cells = obs_counts >= 1

    # Inflate obstacles if requested
    if args.obstacle_inflate > 0:
        ipx = max(1, int(round(args.obstacle_inflate / args.resolution)))
        ki = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ipx+1, 2*ipx+1))
        obstacle_cells = cv2.dilate(obstacle_cells.astype(np.uint8) * 255, ki) > 0
        corridor = corridor & (~obstacle_cells)
        print(f"  Obstacles inflated by {args.obstacle_inflate}m")

    # ── Occupancy grid ───────────────────────────────────────────────────
    print("\nStep 7: Generating occupancy grid...")
    grid = np.full((H, W), 205, dtype=np.uint8)   # unknown
    grid[obstacle_cells] = 0                        # occupied
    grid[corridor] = 254                            # free (corridor overrides)

    # ── Write outputs ────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pgm_name = f"{args.map_name}.pgm"
    pgm_path = args.output_dir / pgm_name
    yaml_path = args.output_dir / f"{args.map_name}.yaml"

    write_pgm(pgm_path, grid)
    write_yaml(yaml_path, pgm_name, args.resolution, min_x, min_y)
    print(f"\n  PGM:  {pgm_path}")
    print(f"  YAML: {yaml_path}")

    if args.output_corridor_pcd is not None:
        write_pcd_xyz_rgb(args.output_corridor_pcd, xyz[remaining_mask], rgb[remaining_mask])
        print(f"  Obstacle PCD: {args.output_corridor_pcd}")

    if args.visualize:
        vis_path = args.output_dir / f"{args.map_name}_vis.png"
        save_visualization(vis_path, corridor, obstacle_cells, pose_xy, min_x, min_y, args.resolution)
        print(f"  Visualization: {vis_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
