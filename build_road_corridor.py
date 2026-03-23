#!/usr/bin/env python3
"""
Build road corridor with AUTO-DETERMINED width from the bird's-eye view.

Method:
  1. Project PCD to 2D bird's-eye view
  2. Build point coverage grid → Gaussian-smoothed density map
  3. Perpendicular profile analysis along trajectory → auto-determine road half-width
  4. Dilate trajectory by auto half-width → smooth road mask
  5. Map 2D road mask back to 3D → remove points → save PCD

Usage:
    python3 build_road_corridor.py
"""

from __future__ import annotations
import argparse, math
from pathlib import Path
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# PCD I/O
# ---------------------------------------------------------------------------
class BinaryPCD:
    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_offset = self._read_header()
        if self.header["DATA"][0].lower() != "binary":
            raise ValueError("only binary PCD supported")
        self.fields = self.header["FIELDS"]
        self.sizes = [int(v) for v in self.header["SIZE"]]
        self.types = self.header["TYPE"]
        self.counts = [int(v) for v in self.header.get("COUNT", ["1"]*len(self.fields))]
        self.points = int(self.header["POINTS"][0])
        self.dtype = self._build_dtype()

    def _read_header(self):
        header = {}
        with self.path.open("rb") as fp:
            while True:
                line = fp.readline()
                if not line: raise ValueError("EOF in header")
                text = line.decode("utf-8", errors="ignore").strip()
                if not text or text.startswith("#"): continue
                tokens = text.split()
                key = tokens[0].upper()
                header[key] = tokens[1:]
                if key == "DATA": break
            return header, fp.tell()

    def _scalar_dtype(self, tc, sz):
        t = {"F": {4: np.float32, 8: np.float64},
             "U": {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64},
             "I": {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}}[tc][sz]
        dt = np.dtype(t)
        return dt.newbyteorder("<") if sz > 1 else dt

    def _build_dtype(self):
        fields = []
        for name, sz, tc, cnt in zip(self.fields, self.sizes, self.types, self.counts):
            dt = self._scalar_dtype(tc, int(sz))
            fields.append((name, dt) if int(cnt)==1 else (name, dt, (int(cnt),)))
        return np.dtype(fields)

    def read_all(self):
        with self.path.open("rb") as fp:
            fp.seek(self.data_offset)
            return np.fromfile(fp, dtype=self.dtype, count=self.points)

def write_pcd_header(fp, n, fs, ss, ts, cs):
    fp.write(f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
             f"FIELDS {fs}\nSIZE {ss}\nTYPE {ts}\nCOUNT {cs}\n"
             f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
             f"POINTS {n}\nDATA binary\n".encode("ascii"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_poses(path):
    xy = []
    for line in path.read_text().strip().splitlines():
        vals = [float(v) for v in line.strip().split()]
        if len(vals) >= 2: xy.append((vals[0], vals[1]))
    return np.array(xy, dtype=np.float64)

def decode_rgb(rgb_float):
    ri = rgb_float.view(np.uint32)
    return ((ri>>16)&0xFF).astype(np.uint8), ((ri>>8)&0xFF).astype(np.uint8), (ri&0xFF).astype(np.uint8)

def w2g(x, y, x0, y0, res):
    return np.floor((x-x0)/res).astype(np.int32), np.floor((y-y0)/res).astype(np.int32)

def in_grid(gx, gy, w, h):
    return (gx>=0)&(gy>=0)&(gx<w)&(gy<h)

# ---------------------------------------------------------------------------
# Perpendicular profile road width detection
# ---------------------------------------------------------------------------
def compute_road_widths_profile(pose_xy, coverage_smooth, x_min, y_min, res,
                                gw, gh, max_search_m, obstacle_threshold,
                                skip_near_m=0.5, min_hit_cells=3,
                                heading_window=10):
    """
    For each trajectory point, walk perpendicular to the heading direction.
    Detect road edge where non-ground (white) point density EXCEEDS
    obstacle_threshold for min_hit_cells consecutive cells.
    Returns (hw_left, hw_right) arrays in metres – one per side.
    """
    n = len(pose_xy)
    hw_left  = np.full(n, max_search_m)
    hw_right = np.full(n, max_search_m)
    max_px = int(max_search_m / res)
    skip_px = max(1, int(skip_near_m / res))  # skip near-trajectory ghost points

    # --- compute smoothed headings ---
    headings = np.zeros(n)
    for i in range(n):
        i0 = max(0, i - heading_window)
        i1 = min(n - 1, i + heading_window)
        dx = pose_xy[i1, 0] - pose_xy[i0, 0]
        dy = pose_xy[i1, 1] - pose_xy[i0, 1]
        if math.hypot(dx, dy) > 1e-6:
            headings[i] = math.atan2(dy, dx)
        elif i > 0:
            headings[i] = headings[i - 1]

    # --- per-point perpendicular scan ---
    for i in range(n):
        perp = headings[i] + math.pi / 2
        cos_p, sin_p = math.cos(perp), math.sin(perp)
        cx = int((pose_xy[i, 0] - x_min) / res)
        cy = int((pose_xy[i, 1] - y_min) / res)
        if not (0 <= cx < gw and 0 <= cy < gh):
            continue

        for side_idx, sign in enumerate((1, -1)):  # left / right
            hits = 0
            detected = max_search_m
            for d in range(skip_px, max_px):
                sx = cx + int(sign * d * cos_p)
                sy = cy + int(sign * d * sin_p)
                if 0 <= sx < gw and 0 <= sy < gh:
                    if coverage_smooth[sy, sx] > obstacle_threshold:
                        hits += 1
                        if hits >= min_hit_cells:
                            detected = (d - min_hit_cells + 1) * res
                            break
                    else:
                        hits = 0
                else:
                    detected = d * res
                    break
            if side_idx == 0:
                hw_left[i] = detected
            else:
                hw_right[i] = detected

    return hw_left, hw_right


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Auto road corridor from bird's-eye view")
    p.add_argument("--global-map-pcd", type=Path, default=Path("map/global_map_ground_green.pcd"))
    p.add_argument("--pose-file", type=Path, default=Path("pose.json"))
    p.add_argument("--output", type=Path, default=Path("map/global_map_road_cleared.pcd"))
    p.add_argument("--preview", type=Path, default=Path("map/road_preview.png"))
    p.add_argument("--resolution", type=float, default=0.05)
    p.add_argument("--max-half-width", type=float, default=2.5)
    p.add_argument("--min-half-width", type=float, default=0.5)
    p.add_argument("--width-percentile", type=float, default=60.0,
                   help="Percentile of detected half-widths (75=favor wider measurements)")
    p.add_argument("--padding", type=float, default=1.0)
    p.add_argument("--density-sigma", type=float, default=0.5,
                   help="Gaussian sigma (metres) for coverage smoothing")
    p.add_argument("--obstacle-threshold", type=float, default=75.0,
                   help="Road edge where non-ground point density exceeds this value")
    p.add_argument("--skip-near", type=float, default=0.5,
                   help="Ignore non-ground points within this distance (m) of trajectory (e.g. human ghost)")
    return p.parse_args()

def main():
    args = parse_args()
    pcd_path = Path.cwd()/args.global_map_pcd if not args.global_map_pcd.is_absolute() else args.global_map_pcd
    pose_path = Path.cwd()/args.pose_file if not args.pose_file.is_absolute() else args.pose_file
    out_path = Path.cwd()/args.output if not args.output.is_absolute() else args.output
    preview_path = Path.cwd()/args.preview if not args.preview.is_absolute() else args.preview
    res = args.resolution

    print("="*60, flush=True)
    print("  Auto Road Corridor (Perpendicular Profile + Dilation)", flush=True)
    print("="*60, flush=True)

    # --- 1. Load ---
    print(f"\n[1/7] Loading data", flush=True)
    pose_xy = load_poses(pose_path)
    print(f"  Poses: {len(pose_xy)}", flush=True)
    reader = BinaryPCD(pcd_path)
    has_rgb = "rgb" in reader.fields
    data = reader.read_all()
    all_x, all_y, all_z = data["x"].astype(np.float32), data["y"].astype(np.float32), data["z"].astype(np.float32)
    valid = np.isfinite(all_x) & np.isfinite(all_y) & np.isfinite(all_z)
    print(f"  PCD: {reader.points:,} ({np.sum(valid):,} valid)", flush=True)

    if has_rgb:
        rgb_f = data["rgb"].astype(np.float32)
        r, g, b = decode_rgb(rgb_f)
        green = ((r==0)&(g==255)&(b==0)) & valid
    else:
        green = valid.copy()
    print(f"  Green: {np.sum(green):,}, Non-green: {np.sum(valid&~green):,}", flush=True)

    # --- 2. Bird's-eye view ---
    print(f"\n[2/7] Bird's-eye view (res={res}m)", flush=True)
    x_min = min(float(all_x[valid].min()), float(pose_xy[:,0].min())) - args.padding
    x_max = max(float(all_x[valid].max()), float(pose_xy[:,0].max())) + args.padding
    y_min = min(float(all_y[valid].min()), float(pose_xy[:,1].min())) - args.padding
    y_max = max(float(all_y[valid].max()), float(pose_xy[:,1].max())) + args.padding
    gw = int(math.ceil((x_max - x_min) / res))
    gh = int(math.ceil((y_max - y_min) / res))
    print(f"  Grid: {gw} x {gh}", flush=True)

    # --- 3. Non-green coverage grid (road surface indicator) ---
    print(f"\n[3/7] Building non-green coverage grid", flush=True)
    non_green = valid & ~green
    coverage_grid = np.zeros((gh, gw), dtype=np.uint8)
    gx_cov, gy_cov = w2g(all_x[non_green], all_y[non_green], x_min, y_min, res)
    ok_cov = in_grid(gx_cov, gy_cov, gw, gh)
    coverage_grid[gy_cov[ok_cov], gx_cov[ok_cov]] = 255

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    coverage_closed = cv2.morphologyEx(coverage_grid, cv2.MORPH_CLOSE, k3, iterations=1)

    sigma_px = max(1.0, args.density_sigma / res)
    ksize = int(sigma_px * 6) | 1          # must be odd
    coverage_smooth = cv2.GaussianBlur(
        coverage_closed.astype(np.float32), (ksize, ksize), sigmaX=sigma_px)
    print(f"  Non-green cells: {np.sum(coverage_grid > 0):,} / Grid total: {gw*gh:,}", flush=True)
    print(f"  Gaussian sigma: {args.density_sigma}m ({sigma_px:.0f}px)", flush=True)

    # Diagnostic: coverage at a few trajectory points
    mid = len(pose_xy) // 2
    for idx in [0, mid, len(pose_xy)-1]:
        cx = int((pose_xy[idx, 0] - x_min) / res)
        cy = int((pose_xy[idx, 1] - y_min) / res)
        if 0 <= cx < gw and 0 <= cy < gh:
            print(f"  [diag] pose[{idx}] coverage_smooth={coverage_smooth[cy,cx]:.1f}", flush=True)

    # --- 4. Perpendicular profile → auto width ---
    print(f"\n[4/7] Perpendicular profile → auto width (obstacle detection)", flush=True)
    search_limit = args.max_half_width + 1.0
    print(f"  Skip near-trajectory: {args.skip_near}m", flush=True)
    hw_left, hw_right = compute_road_widths_profile(
        pose_xy, coverage_smooth, x_min, y_min, res, gw, gh,
        max_search_m=search_limit, skip_near_m=args.skip_near,
        obstacle_threshold=args.obstacle_threshold)

    # Pool all valid left/right measurements independently
    all_hw = np.concatenate([hw_left, hw_right])
    valid_hw = all_hw[all_hw < search_limit - 0.01]
    if len(valid_hw) > 0:
        auto_hw = float(np.percentile(valid_hw, args.width_percentile))
        print(f"  All measurements: {len(valid_hw)} valid / {len(all_hw)} total", flush=True)
        print(f"  Distribution: min={valid_hw.min():.2f}m, p25={np.percentile(valid_hw,25):.2f}m, "
              f"median={np.median(valid_hw):.2f}m, p75={np.percentile(valid_hw,75):.2f}m, "
              f"max={valid_hw.max():.2f}m", flush=True)
        print(f"  Using p{args.width_percentile:.0f} = {auto_hw:.2f}m", flush=True)
    else:
        auto_hw = args.min_half_width
        print(f"  ⚠ No road edges detected, using min-half-width", flush=True)

    auto_hw = max(args.min_half_width, min(args.max_half_width, auto_hw))
    safety_margin = 0.20
    auto_hw_safe = max(args.min_half_width, auto_hw - safety_margin)

    print(f"  ★ Auto half-width (raw): {auto_hw:.2f}m", flush=True)
    print(f"  ★ Safety margin: -{safety_margin}m", flush=True)
    print(f"  ★ Final half-width: {auto_hw_safe:.2f}m (total: {auto_hw_safe*2:.2f}m)", flush=True)
    auto_hw = auto_hw_safe

    px = np.floor((pose_xy[:,0] - x_min) / res).astype(np.int32)
    py = np.floor((pose_xy[:,1] - y_min) / res).astype(np.int32)

    # --- 5. Dilate trajectory ---
    print(f"\n[5/7] Building road mask", flush=True)
    pts_traj = np.stack([px, py], axis=1).astype(np.int32)
    traj_mask = np.zeros((gh, gw), dtype=np.uint8)
    cv2.polylines(traj_mask, [pts_traj], isClosed=False, color=255, thickness=1)

    dilate_px = int(round(auto_hw / res))
    k_road = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_px+1, 2*dilate_px+1))
    road_mask = cv2.dilate(traj_mask, k_road, iterations=1)
    print(f"  Road area: {np.sum(road_mask > 0) * res * res:.1f} m²", flush=True)

    # --- 6. Preview (Z-buffer: highest point wins) ---
    print(f"\n[6/7] Preview (Z-buffer)", flush=True)
    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    img[:] = (30, 40, 50)

    # Grid coords for all valid points
    gx_all_v, gy_all_v = w2g(all_x[valid], all_y[valid], x_min, y_min, res)
    ok_v = in_grid(gx_all_v, gy_all_v, gw, gh)

    z_v = all_z[valid][ok_v]
    gx_v = gx_all_v[ok_v]
    gy_v = gy_all_v[ok_v]
    is_green_v = green[valid][ok_v]

    # Sort by Z ascending → paint lowest first, highest last (highest wins)
    order = np.argsort(z_v)
    gx_v, gy_v, is_green_v = gx_v[order], gy_v[order], is_green_v[order]

    # Green → (0,255,0), non-green → (200,200,200)
    colors = np.where(is_green_v[:, None], [0, 255, 0], [200, 200, 200]).astype(np.uint8)
    img[gy_v, gx_v] = colors
    cv2.polylines(img, [pts_traj], isClosed=False, color=(255, 100, 0), thickness=2)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color=(0, 0, 255), thickness=3)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), img)
    print(f"  Saved: {preview_path}", flush=True)

    # --- 7. Cut & save ---
    print(f"\n[7/7] Cutting & saving", flush=True)
    gx_all, gy_all = w2g(all_x, all_y, x_min, y_min, res)
    inside = in_grid(gx_all, gy_all, gw, gh)
    in_road = np.zeros(len(all_x), dtype=bool)
    in_road[inside] = road_mask[gy_all[inside], gx_all[inside]] > 0

    keep = valid & ~in_road
    removed = int(np.sum(valid & in_road))
    kept = int(np.sum(keep))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if has_rgb:
        fs, ss, ts, cs = "x y z rgb", "4 4 4 4", "F F F F", "1 1 1 1"
        od = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("rgb","<f4")])
        out = np.empty(kept, dtype=od)
        out["x"], out["y"], out["z"], out["rgb"] = all_x[keep], all_y[keep], all_z[keep], rgb_f[keep]
    else:
        fs, ss, ts, cs = "x y z", "4 4 4", "F F F", "1 1 1"
        od = np.dtype([("x","<f4"),("y","<f4"),("z","<f4")])
        out = np.empty(kept, dtype=od)
        out["x"], out["y"], out["z"] = all_x[keep], all_y[keep], all_z[keep]

    with out_path.open("wb") as fp:
        write_pcd_header(fp, kept, fs, ss, ts, cs)
        fp.write(out.tobytes())

    print(f"\n{'='*60}", flush=True)
    print(f"  Done!", flush=True)
    print(f"  Auto half-width: {auto_hw:.2f}m (total: {auto_hw*2:.2f}m)", flush=True)
    print(f"  Original:  {reader.points:,}", flush=True)
    print(f"  Removed:   {removed:,}", flush=True)
    print(f"  Remaining: {kept:,}", flush=True)
    print(f"  Output:    {out_path}", flush=True)
    print(f"  Preview:   {preview_path}", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == "__main__":
    main()
