#!/usr/bin/env python3
"""
Traversable corridor extraction from PCD map + LiDAR trajectory.

Pipeline:
  1. Load PCD (ground=green, obstacle=white) and trajectory (pose.json)
  2. Project all points to 2D BEV grid, count green/white per cell
  3. Sweep half-widths from min to max, compute incremental white ratio (ΔW/ΔT)
  4. Auto-detect cut width where ΔW/ΔT first exceeds threshold
  5. Apply safety margin, remove corridor points from PCD
  6. Save: cleaned PCD, BEV image, corridor preview, width sweep CSV

Usage:
    python3 generate_traversable_area.py
    python3 generate_traversable_area.py --pcd /path/to/map.pcd --pose-file /path/to/pose.json
    python3 generate_traversable_area.py --threshold 0.08 --safety-margin 0.2
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# PCD I/O
# ═══════════════════════════════════════════════════════════════════════════
class BinaryPCD:
    """Reader for binary PCD files."""

    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_offset = self._read_header()
        if self.header["DATA"][0].lower() != "binary":
            raise ValueError("only binary PCD supported")
        self.fields = self.header["FIELDS"]
        self.sizes = [int(v) for v in self.header["SIZE"]]
        self.types = self.header["TYPE"]
        self.counts = [int(v) for v in self.header.get("COUNT", ["1"] * len(self.fields))]
        self.points = int(self.header["POINTS"][0])
        self.dtype = self._build_dtype()

    def _read_header(self):
        header = {}
        with self.path.open("rb") as fp:
            while True:
                line = fp.readline()
                if not line:
                    raise ValueError("EOF in header")
                text = line.decode("utf-8", errors="ignore").strip()
                if not text or text.startswith("#"):
                    continue
                tokens = text.split()
                key = tokens[0].upper()
                header[key] = tokens[1:]
                if key == "DATA":
                    break
            return header, fp.tell()

    def _scalar_dtype(self, tc, sz):
        t = {
            "F": {4: np.float32, 8: np.float64},
            "U": {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64},
            "I": {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64},
        }[tc][sz]
        dt = np.dtype(t)
        return dt.newbyteorder("<") if sz > 1 else dt

    def _build_dtype(self):
        fields = []
        for name, sz, tc, cnt in zip(self.fields, self.sizes, self.types, self.counts):
            dt = self._scalar_dtype(tc, int(sz))
            fields.append((name, dt) if int(cnt) == 1 else (name, dt, (int(cnt),)))
        return np.dtype(fields)

    def read_all(self):
        with self.path.open("rb") as fp:
            fp.seek(self.data_offset)
            return np.fromfile(fp, dtype=self.dtype, count=self.points)


def write_pcd_binary(path, n, fields_str, sizes_str, types_str, counts_str, raw_bytes):
    """Write a binary PCD file."""
    with open(path, "wb") as fp:
        fp.write(f"# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
                 f"FIELDS {fields_str}\nSIZE {sizes_str}\nTYPE {types_str}\nCOUNT {counts_str}\n"
                 f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                 f"POINTS {n}\nDATA binary\n".encode("ascii"))
        fp.write(raw_bytes)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def decode_rgb(rgb_float):
    """Decode packed float32 RGB → (R, G, B) uint8 arrays."""
    ri = rgb_float.view(np.uint32)
    return ((ri >> 16) & 0xFF).astype(np.uint8), \
           ((ri >> 8) & 0xFF).astype(np.uint8), \
           (ri & 0xFF).astype(np.uint8)


def load_poses(path):
    """Load trajectory XY from pose.json (each line: x y z qw qx qy qz)."""
    xy = []
    for line in path.read_text().strip().splitlines():
        vals = [float(v) for v in line.strip().split()]
        if len(vals) >= 2:
            xy.append((vals[0], vals[1]))
    return np.array(xy, dtype=np.float64)


def w2g(x, y, x0, y0, res):
    """World coordinates → grid indices."""
    return np.floor((x - x0) / res).astype(np.int32), \
           np.floor((y - y0) / res).astype(np.int32)


def in_grid(gx, gy, w, h):
    """Check if grid indices are within bounds."""
    return (gx >= 0) & (gy >= 0) & (gx < w) & (gy < h)


def build_corridor_mask(pose_xy, x_min, y_min, res, gw, gh, half_width):
    """Build corridor mask: draw trajectory polyline then dilate by half_width."""
    px = np.floor((pose_xy[:, 0] - x_min) / res).astype(np.int32)
    py = np.floor((pose_xy[:, 1] - y_min) / res).astype(np.int32)
    pts_traj = np.stack([px, py], axis=1).astype(np.int32)

    traj_mask = np.zeros((gh, gw), dtype=np.uint8)
    cv2.polylines(traj_mask, [pts_traj], isClosed=False, color=255, thickness=1)

    dilate_px = int(round(half_width / res))
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * dilate_px + 1, 2 * dilate_px + 1))
        corridor = cv2.dilate(traj_mask, k, iterations=1)
    else:
        corridor = traj_mask

    return corridor > 0


def render_bev(green_count, white_count, gh, gw):
    """Render BEV image: green-dominant cells → green, white-dominant → white."""
    total_count = green_count + white_count
    non_empty = total_count > 0
    green_dom = (green_count >= white_count) & non_empty
    white_dom = (white_count > green_count) & non_empty

    log_total = np.log1p(total_count.astype(np.float32))
    log_max = max(1.0, float(log_total.max()))
    brightness = np.clip(80 + 175 * log_total / log_max, 0, 255).astype(np.uint8)

    img = np.zeros((gh, gw, 3), dtype=np.uint8)
    img[green_dom, 1] = brightness[green_dom]
    img[white_dom, 0] = brightness[white_dom]
    img[white_dom, 1] = brightness[white_dom]
    img[white_dom, 2] = brightness[white_dom]
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate traversable area from PCD map + trajectory")
    # Input
    p.add_argument("--pcd", type=Path,
                   default=Path("/media/ros/DATA/ZMG/321果园测试数据集/test/global_map_ground_green.pcd"))
    p.add_argument("--pose-file", type=Path,
                   default=Path("/media/ros/DATA/ZMG/321果园测试数据集/test/pose.json"))
    # Output
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: same as PCD)")
    # Grid
    p.add_argument("--resolution", type=float, default=0.1,
                   help="BEV grid resolution in metres (default: 0.1)")
    p.add_argument("--padding", type=float, default=1.0)
    # Sweep
    p.add_argument("--min-half-width", type=float, default=0.8,
                   help="Minimum sweep half-width (default: 0.8m)")
    p.add_argument("--max-half-width", type=float, default=2.5,
                   help="Maximum sweep half-width (default: 2.5m)")
    p.add_argument("--step", type=float, default=0.1,
                   help="Sweep step size (default: 0.1m)")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="ΔW/ΔT threshold (default: 0.10 = 10%%)")
    p.add_argument("--safety-margin", type=float, default=0.1,
                   help="Safety margin to subtract from detected width (default: 0.1m)")
    return p.parse_args()


def main():
    args = parse_args()
    res = args.resolution
    output_dir = Path(args.output_dir) if args.output_dir else args.pcd.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output paths
    bev_path = output_dir / "bev_projection.png"
    preview_path = output_dir / "corridor_cut_preview.png"
    csv_path = output_dir / "corridor_width_sweep.csv"
    out_pcd = output_dir / "global_map_corridor_removed.pcd"

    print("=" * 60)
    print("  Generate Traversable Area")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────
    print(f"\n[1/7] Loading data")
    reader = BinaryPCD(args.pcd)
    data = reader.read_all()
    all_x = data["x"].astype(np.float32)
    all_y = data["y"].astype(np.float32)
    all_z = data["z"].astype(np.float32)
    valid = np.isfinite(all_x) & np.isfinite(all_y) & np.isfinite(all_z)

    has_rgb = "rgb" in reader.fields
    if not has_rgb:
        raise ValueError("PCD must have 'rgb' field (green=ground, white=obstacle)")

    rgb_f = data["rgb"].astype(np.float32)
    r, g, b = decode_rgb(rgb_f)
    is_green = ((r == 0) & (g == 255) & (b == 0)) & valid
    is_white = valid & ~is_green

    pose_xy = load_poses(args.pose_file)
    # Compute trajectory length
    diffs = np.diff(pose_xy, axis=0)
    seg_lengths = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    traj_length = float(np.sum(seg_lengths))
    print(f"  PCD: {reader.points:,} points "
          f"(green: {np.sum(is_green):,}, white: {np.sum(is_white):,})")
    print(f"  Trajectory: {len(pose_xy)} poses, length: {traj_length:.1f}m")

    # ── Step 2: Build BEV grid ────────────────────────────────────────
    print(f"\n[2/7] Building BEV grid (res={res}m)")
    xv, yv = all_x[valid], all_y[valid]
    x_min = min(float(xv.min()), float(pose_xy[:, 0].min())) - args.padding
    x_max = max(float(xv.max()), float(pose_xy[:, 0].max())) + args.padding
    y_min = min(float(yv.min()), float(pose_xy[:, 1].min())) - args.padding
    y_max = max(float(yv.max()), float(pose_xy[:, 1].max())) + args.padding
    gw = int(math.ceil((x_max - x_min) / res))
    gh = int(math.ceil((y_max - y_min) / res))
    print(f"  Extent: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]")
    print(f"  Grid: {gw} x {gh} = {gw * gh:,} cells")

    # ── Step 3: Count green/white per cell ────────────────────────────
    print(f"\n[3/7] Counting points per cell")
    green_count = np.zeros((gh, gw), dtype=np.int32)
    white_count = np.zeros((gh, gw), dtype=np.int32)

    gx_g, gy_g = w2g(all_x[is_green], all_y[is_green], x_min, y_min, res)
    ok_g = in_grid(gx_g, gy_g, gw, gh)
    np.add.at(green_count, (gy_g[ok_g], gx_g[ok_g]), 1)

    gx_w, gy_w = w2g(all_x[is_white], all_y[is_white], x_min, y_min, res)
    ok_w = in_grid(gx_w, gy_w, gw, gh)
    np.add.at(white_count, (gy_w[ok_w], gx_w[ok_w]), 1)

    total_count = green_count + white_count
    cell_is_green = green_count > white_count
    cell_is_white = white_count > green_count
    cell_non_empty = total_count > 0

    n_nonempty = int(np.sum(cell_non_empty))
    n_green_dom = int(np.sum(cell_is_green & cell_non_empty))
    n_white_dom = int(np.sum(cell_is_white & cell_non_empty))
    print(f"  Non-empty: {n_nonempty:,} cells")
    print(f"  Green-dominant: {n_green_dom:,}, White-dominant: {n_white_dom:,}")

    # ── Step 4: Save BEV image ────────────────────────────────────────
    print(f"\n[4/7] Saving BEV projection")
    bev_img = render_bev(green_count, white_count, gh, gw)
    cv2.imwrite(str(bev_path), bev_img)
    print(f"  → {bev_path}")

    # ── Step 5: Sweep half-widths ─────────────────────────────────────
    print(f"\n[5/7] Sweeping half-widths: "
          f"{args.min_half_width}m → {args.max_half_width}m (step {args.step}m)")
    print(f"       Threshold: ΔW/ΔT > {args.threshold:.0%}")

    half_widths = np.arange(args.min_half_width,
                            args.max_half_width + 0.001, args.step)

    print(f"\n  {'HW(m)':>7s} | {'Total':>8s} | {'Green':>8s} | {'White':>8s} | "
          f"{'Empty':>8s} | {'G/(G+W)':>7s} | {'ΔW/ΔT':>7s} | Status")
    print("  " + "-" * 85)

    prev_total = prev_white = 0
    cut_hw = args.max_half_width
    found = False
    results = []

    for hw in half_widths:
        mask = build_corridor_mask(pose_xy, x_min, y_min, res, gw, gh, hw)
        n_total = int(np.sum(mask))
        n_green = int(np.sum(mask & cell_is_green))
        n_white = int(np.sum(mask & cell_is_white))
        n_empty = int(np.sum(mask & ~cell_non_empty))
        g_ratio = n_green / (n_green + n_white) if (n_green + n_white) > 0 else 0

        dt = n_total - prev_total
        dw = n_white - prev_white
        dw_dt = dw / dt if dt > 0 else 0

        status = ""
        if prev_total > 0 and dw_dt > args.threshold and not found:
            cut_hw = hw
            found = True
            status = f"★ CUT"

        print(f"  {hw:7.1f} | {n_total:8,} | {n_green:8,} | {n_white:8,} | "
              f"{n_empty:8,} | {g_ratio:6.1%} | {dw_dt:6.1%} | {status}")

        results.append((hw, n_total, n_green, n_white, n_empty, g_ratio, dw_dt))
        prev_total = n_total
        prev_white = n_white

    if not found:
        print(f"\n  ⚠ Threshold never exceeded, using max: {args.max_half_width}m")

    # Apply safety margin
    cut_hw_safe = max(args.min_half_width, cut_hw - args.safety_margin)
    print(f"\n  Detected: {cut_hw:.1f}m → Safety margin: -{args.safety_margin}m "
          f"→ Final: {cut_hw_safe:.1f}m (total width: {cut_hw_safe * 2:.1f}m)")
    cut_hw = cut_hw_safe

    # Save CSV
    with open(csv_path, "w") as f:
        f.write("half_width_m,total_cells,green_cells,white_cells,"
                "empty_cells,green_ratio,delta_white_ratio\n")
        for row in results:
            f.write(f"{row[0]:.1f},{row[1]},{row[2]},{row[3]},"
                    f"{row[4]},{row[5]:.4f},{row[6]:.4f}\n")
    print(f"  CSV → {csv_path}")

    # ── Step 6: Cut corridor points ───────────────────────────────────
    print(f"\n[6/7] Cutting corridor (half-width={cut_hw:.1f}m)")
    corridor_mask = build_corridor_mask(pose_xy, x_min, y_min, res, gw, gh, cut_hw)

    gx_all, gy_all = w2g(all_x, all_y, x_min, y_min, res)
    inside_grid = in_grid(gx_all, gy_all, gw, gh)
    in_corridor = np.zeros(len(all_x), dtype=bool)
    in_corridor[inside_grid] = corridor_mask[gy_all[inside_grid], gx_all[inside_grid]]

    keep = valid & ~in_corridor
    removed = int(np.sum(valid & in_corridor))
    kept = int(np.sum(keep))
    print(f"  Original:  {int(np.sum(valid)):,}")
    print(f"  Removed:   {removed:,}")
    print(f"  Remaining: {kept:,}")

    # Save PCD
    fs, ss, ts, cs = "x y z rgb", "4 4 4 4", "F F F F", "1 1 1 1"
    od = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    out = np.empty(kept, dtype=od)
    out["x"] = all_x[keep]
    out["y"] = all_y[keep]
    out["z"] = all_z[keep]
    out["rgb"] = rgb_f[keep]
    write_pcd_binary(str(out_pcd), kept, fs, ss, ts, cs, out.tobytes())
    print(f"  PCD → {out_pcd}")

    # ── Step 7: Preview image ─────────────────────────────────────────
    print(f"\n[7/7] Generating preview")
    preview_img = render_bev(green_count, white_count, gh, gw)

    # Corridor boundary (red)
    contours, _ = cv2.findContours(
        corridor_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview_img, contours, -1, color=(0, 0, 255), thickness=2)

    # Trajectory (blue)
    px = np.floor((pose_xy[:, 0] - x_min) / res).astype(np.int32)
    py = np.floor((pose_xy[:, 1] - y_min) / res).astype(np.int32)
    pts_traj = np.stack([px, py], axis=1).astype(np.int32)
    cv2.polylines(preview_img, [pts_traj], isClosed=False,
                  color=(255, 100, 0), thickness=2)

    cv2.imwrite(str(preview_path), preview_img)
    print(f"  Preview → {preview_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Done!")
    print(f"  Resolution:      {res}m")
    print(f"  Cut half-width:  {cut_hw:.1f}m (total: {cut_hw * 2:.1f}m)")
    print(f"  Points removed:  {removed:,}")
    print(f"  Points kept:     {kept:,}")
    print(f"  Output PCD:      {out_pcd}")
    print(f"  BEV image:       {bev_path}")
    print(f"  Corridor preview:{preview_path}")
    print(f"  Width sweep CSV: {csv_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
