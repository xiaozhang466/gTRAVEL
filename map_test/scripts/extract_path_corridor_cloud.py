#!/usr/bin/env python3
"""
Extract a 3D corridor cloud from a stitched global map using trajectory poses.

By default this script removes points inside a fixed half-width trajectory corridor.
It can also auto-estimate a half-width from segmented ground/nonground maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep/remove 3D trajectory corridor from global PCD map."
    )
    parser.add_argument(
        "--global-map-pcd",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_test/map/global_map.pcd"),
        help="Input global map PCD (DATA binary).",
    )
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_test/pose.json"),
        help="Pose file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_test/map/global_map_without_corridor_0p8m.pcd"),
        help="Output point cloud path.",
    )
    parser.add_argument(
        "--corridor-half-width",
        type=float,
        default=0.8,
        help="Half width of trajectory corridor in meters (used when --auto-half-width is disabled).",
    )
    parser.add_argument(
        "--dynamic-profile",
        type=Path,
        default=None,
        help="Optional dynamic width profile file (CSV/TXT: seq,left_width,right_width).",
    )
    parser.add_argument(
        "--auto-half-width",
        action="store_true",
        help="Estimate half-width from ground/nonground segmented maps.",
    )
    parser.add_argument(
        "--ground-map-pcd",
        type=Path,
        default=None,
        help="Ground global map PCD for auto half-width estimation.",
    )
    parser.add_argument(
        "--nonground-map-pcd",
        type=Path,
        default=None,
        help="Nonground global map PCD for auto half-width estimation.",
    )
    parser.add_argument(
        "--min-half-width",
        type=float,
        default=0.5,
        help="Minimum allowed estimated half-width (meters).",
    )
    parser.add_argument(
        "--max-half-width",
        type=float,
        default=2.0,
        help="Maximum allowed estimated half-width (meters).",
    )
    parser.add_argument(
        "--auto-margin",
        type=float,
        default=0.15,
        help="Safety margin subtracted from estimated obstacle distance (meters).",
    )
    parser.add_argument(
        "--auto-percentile",
        type=float,
        default=60.0,
        help="Percentile of pose-wise clearances used for robust width estimation.",
    )
    parser.add_argument(
        "--ground-min-points",
        type=int,
        default=2,
        help="Minimum ground points per cell to mark ground support.",
    )
    parser.add_argument(
        "--nonground-min-points",
        type=int,
        default=1,
        help="Minimum nonground points per cell to mark obstacle.",
    )
    parser.add_argument(
        "--auto-sample-radius-cells",
        type=int,
        default=2,
        help="Sample max clearance in local pose neighborhood with this radius (in cells).",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.05,
        help="XY mask raster resolution in meters.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Extra padding for XY bounds in meters.",
    )
    parser.add_argument(
        "--chunk-points",
        type=int,
        default=1_000_000,
        help="Points per chunk when streaming global map.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=-1e9,
        help="Optional Z lower bound for kept points.",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=1e9,
        help="Optional Z upper bound for kept points.",
    )
    parser.add_argument(
        "--mode",
        choices=("remove_corridor", "keep_corridor"),
        default="remove_corridor",
        help="remove_corridor: delete points in corridor; keep_corridor: only keep points in corridor.",
    )
    return parser.parse_args()


def pose_xy_from_values(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 7:
        return float(values[0]), float(values[1])
    if len(values) == 12:
        return float(values[3]), float(values[7])
    if len(values) == 16:
        return float(values[3]), float(values[7])
    raise ValueError(f"Unsupported pose format length: {len(values)}")


def load_pose_xy(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty pose file: {path}")

    xy: List[Tuple[float, float]] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "poses" in payload:
            payload = payload["poses"]
        if not isinstance(payload, list):
            raise ValueError("JSON pose payload must be list.")

        for item in payload:
            if isinstance(item, dict):
                if "x" in item and "y" in item:
                    xy.append((float(item["x"]), float(item["y"])))
                elif "tx" in item and "ty" in item:
                    xy.append((float(item["tx"]), float(item["ty"])))
                elif "T" in item and isinstance(item["T"], list):
                    xy.append(pose_xy_from_values([float(v) for v in item["T"]]))
                else:
                    raise ValueError(f"Unsupported JSON pose object: {item}")
            elif isinstance(item, list):
                xy.append(pose_xy_from_values([float(v) for v in item]))
            else:
                raise ValueError(f"Unsupported JSON pose element: {type(item)}")

        if not xy:
            raise ValueError(f"No valid pose entries in {path}")
        return np.asarray(xy, dtype=np.float64)
    except json.JSONDecodeError:
        pass

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        try:
            xy.append(pose_xy_from_values(vals))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc

    if not xy:
        raise ValueError(f"No valid pose entries in {path}")
    return np.asarray(xy, dtype=np.float64)


class BinaryPCD:
    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_offset = self._read_header()
        data_mode = self.header["DATA"][0].lower()
        if data_mode != "binary":
            raise ValueError(f"{path}: only DATA binary is supported, got {data_mode}")
        self.fields = self.header["FIELDS"]
        self.sizes = [int(v) for v in self.header["SIZE"]]
        self.types = self.header["TYPE"]
        self.counts = [int(v) for v in self.header.get("COUNT", ["1"] * len(self.fields))]
        if len(self.fields) != len(self.sizes) or len(self.fields) != len(self.types):
            raise ValueError(f"{path}: malformed PCD field metadata")
        self.points = int(self.header["POINTS"][0])
        self.dtype = self._build_dtype()
        for axis in ("x", "y", "z"):
            if axis not in self.dtype.names:
                raise ValueError(f"{path}: PCD must contain {axis} field")
        self.has_intensity = "intensity" in self.dtype.names

    def _read_header(self) -> Tuple[Dict[str, List[str]], int]:
        header: Dict[str, List[str]] = {}
        with self.path.open("rb") as fp:
            while True:
                line = fp.readline()
                if not line:
                    raise ValueError(f"{self.path}: EOF while parsing header")
                text = line.decode("utf-8", errors="ignore").strip()
                if not text or text.startswith("#"):
                    continue
                tokens = text.split()
                key = tokens[0].upper()
                header[key] = tokens[1:]
                if key == "DATA":
                    break
            offset = fp.tell()
        return header, offset

    def _scalar_dtype(self, type_code: str, size: int) -> np.dtype:
        if type_code == "F":
            table = {4: np.float32, 8: np.float64}
        elif type_code == "U":
            table = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
        elif type_code == "I":
            table = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
        else:
            raise ValueError(f"{self.path}: unsupported TYPE {type_code}")
        if size not in table:
            raise ValueError(f"{self.path}: unsupported SIZE {size} for TYPE {type_code}")
        dt = np.dtype(table[size])
        if size > 1:
            dt = dt.newbyteorder("<")
        return dt

    def _build_dtype(self) -> np.dtype:
        fields = []
        for name, size, type_code, count in zip(self.fields, self.sizes, self.types, self.counts):
            dt = self._scalar_dtype(type_code, size)
            if count == 1:
                fields.append((name, dt))
            else:
                fields.append((name, dt, (count,)))
        return np.dtype(fields)

    def iter_chunks(self, chunk_points: int) -> Iterator[np.ndarray]:
        with self.path.open("rb") as fp:
            fp.seek(self.data_offset)
            remain = self.points
            while remain > 0:
                n = min(chunk_points, remain)
                chunk = np.fromfile(fp, dtype=self.dtype, count=n)
                if len(chunk) == 0:
                    break
                remain -= len(chunk)
                yield chunk


def world_to_grid(
    x: np.ndarray, y: np.ndarray, min_x: float, min_y: float, res: float
) -> Tuple[np.ndarray, np.ndarray]:
    gx = np.floor((x - min_x) / res).astype(np.int32)
    gy = np.floor((y - min_y) / res).astype(np.int32)
    return gx, gy


def in_grid(gx: np.ndarray, gy: np.ndarray, width: int, height: int) -> np.ndarray:
    return (gx >= 0) & (gy >= 0) & (gx < width) & (gy < height)


def update_bounds_from_reader(reader: BinaryPCD, bounds: List[float], chunk_points: int) -> None:
    for chunk in reader.iter_chunks(chunk_points):
        x = chunk["x"].astype(np.float32, copy=False)
        y = chunk["y"].astype(np.float32, copy=False)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        xv, yv = x[valid], y[valid]
        bounds[0] = min(bounds[0], float(np.min(xv)))
        bounds[1] = max(bounds[1], float(np.max(xv)))
        bounds[2] = min(bounds[2], float(np.min(yv)))
        bounds[3] = max(bounds[3], float(np.max(yv)))


def rasterize_counts(
    reader: BinaryPCD,
    min_x: float,
    min_y: float,
    width: int,
    height: int,
    resolution: float,
    chunk_points: int,
) -> np.ndarray:
    counts = np.zeros((height, width), dtype=np.uint32)
    area = width * height
    for chunk in reader.iter_chunks(chunk_points):
        x = chunk["x"].astype(np.float32, copy=False)
        y = chunk["y"].astype(np.float32, copy=False)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        gx, gy = world_to_grid(x, y, min_x, min_y, resolution)
        inside = in_grid(gx, gy, width, height)
        if not np.any(inside):
            continue
        idx = gy[inside] * width + gx[inside]
        bc = np.bincount(idx, minlength=area).reshape(height, width).astype(np.uint32)
        counts += bc
    return counts


def estimate_half_width_from_segmented(
    pose_xy: np.ndarray,
    ground_reader: BinaryPCD,
    nonground_reader: BinaryPCD,
    min_x: float,
    min_y: float,
    width: int,
    height: int,
    resolution: float,
    chunk_points: int,
    ground_min_points: int,
    nonground_min_points: int,
    percentile: float,
    margin: float,
    sample_radius_cells: int,
    min_half_width: float,
    max_half_width: float,
) -> float:
    ground_counts = rasterize_counts(
        ground_reader, min_x, min_y, width, height, resolution, chunk_points
    )
    nonground_counts = rasterize_counts(
        nonground_reader, min_x, min_y, width, height, resolution, chunk_points
    )

    ground_support = ground_counts >= max(1, ground_min_points)
    obstacles = nonground_counts >= max(1, nonground_min_points)

    # Fill sparse holes of sampled map support and slightly inflate obstacle marks.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    g_u8 = ground_support.astype(np.uint8) * 255
    g_u8 = cv2.dilate(g_u8, k3, iterations=1)
    g_u8 = cv2.morphologyEx(g_u8, cv2.MORPH_CLOSE, k3, iterations=1)
    ground_support = g_u8 > 0

    o_u8 = obstacles.astype(np.uint8) * 255
    o_u8 = cv2.dilate(o_u8, k3, iterations=1)
    obstacles = o_u8 > 0

    free_like = ground_support & (~obstacles)
    free_u8 = (free_like.astype(np.uint8) * 255)

    # Distance to nearest non-free cell (obstacle or unsupported area).
    dist_pix = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5)
    dist_m = dist_pix * resolution

    px, py = world_to_grid(
        pose_xy[:, 0].astype(np.float32), pose_xy[:, 1].astype(np.float32), min_x, min_y, resolution
    )
    pose_valid = in_grid(px, py, width, height)
    px, py = px[pose_valid], py[pose_valid]
    if len(px) == 0:
        raise ValueError("No valid pose samples in grid for auto half-width estimation.")

    if sample_radius_cells < 0:
        sample_radius_cells = 0
    if sample_radius_cells == 0:
        clearances = dist_m[py, px]
    else:
        clearances = np.empty(len(px), dtype=np.float32)
        for i, (gx, gy) in enumerate(zip(px, py)):
            x0 = max(0, int(gx) - sample_radius_cells)
            x1 = min(width, int(gx) + sample_radius_cells + 1)
            y0 = max(0, int(gy) - sample_radius_cells)
            y1 = min(height, int(gy) + sample_radius_cells + 1)
            clearances[i] = float(np.max(dist_m[y0:y1, x0:x1]))
    clearances = clearances[np.isfinite(clearances)]
    clearances = clearances[clearances > 0.0]
    if len(clearances) < 10:
        raise ValueError("Too few valid clearance samples for auto half-width estimation.")

    p = float(np.clip(percentile, 1.0, 99.0))
    raw = float(np.percentile(clearances, p))
    estimated = raw - float(margin)
    return float(np.clip(estimated, min_half_width, max_half_width))


def build_corridor_mask(
    pose_xy: np.ndarray,
    min_x: float,
    min_y: float,
    width: int,
    height: int,
    resolution: float,
    half_width: float,
) -> np.ndarray:
    px, py = world_to_grid(
        pose_xy[:, 0].astype(np.float32), pose_xy[:, 1].astype(np.float32), min_x, min_y, resolution
    )
    pose_valid = in_grid(px, py, width, height)
    px, py = px[pose_valid], py[pose_valid]
    if len(px) < 2:
        raise ValueError("Too few valid pose points in map bounds.")

    base = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(px) - 1):
        cv2.line(base, (int(px[i]), int(py[i])), (int(px[i + 1]), int(py[i + 1])), color=255, thickness=1)

    radius = max(1, int(round(half_width / resolution)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = cv2.dilate(base, kernel, iterations=1)
    return mask > 0


def load_dynamic_profile(path: Path) -> Dict[int, Tuple[float, float]]:
    if not path.exists():
        raise FileNotFoundError(path)

    profile: Dict[int, Tuple[float, float]] = {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty dynamic profile file: {path}")

    # First try CSV with headers.
    try:
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is not None:
                fields = {f.strip().lower() for f in reader.fieldnames if f is not None}
                if {"seq", "left_width", "right_width"}.issubset(fields):
                    for row in reader:
                        seq = int(float(row["seq"]))
                        lw = float(row["left_width"])
                        rw = float(row["right_width"])
                        profile[seq] = (lw, rw)
                    if profile:
                        return profile
    except Exception:
        pass

    # Fallback: plain text rows "seq left right" or "seq,left,right"
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) < 3:
            raise ValueError(f"{path}:{line_no}: expected at least 3 columns (seq left right)")
        try:
            seq = int(float(parts[0]))
            lw = float(parts[1])
            rw = float(parts[2])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: failed to parse seq/left/right") from exc
        profile[seq] = (lw, rw)

    if not profile:
        raise ValueError(f"No valid dynamic profile rows in {path}")
    return profile


def build_dynamic_corridor_mask(
    pose_xy: np.ndarray,
    min_x: float,
    min_y: float,
    width: int,
    height: int,
    resolution: float,
    profile: Dict[int, Tuple[float, float]],
    fallback_half_width: float,
) -> np.ndarray:
    px, py = world_to_grid(
        pose_xy[:, 0].astype(np.float32), pose_xy[:, 1].astype(np.float32), min_x, min_y, resolution
    )
    pose_valid = in_grid(px, py, width, height)
    valid_idx = np.where(pose_valid)[0]
    px, py = px[pose_valid], py[pose_valid]
    if len(px) < 2:
        raise ValueError("Too few valid pose points in map bounds.")

    base = np.zeros((height, width), dtype=np.uint8)
    for i in range(len(px) - 1):
        idx0 = int(valid_idx[i])
        idx1 = int(valid_idx[i + 1])
        p0 = np.array([float(px[i]), float(py[i])], dtype=np.float32)
        p1 = np.array([float(px[i + 1]), float(py[i + 1])], dtype=np.float32)
        d = p1 - p0
        seg_len = float(np.linalg.norm(d))
        if seg_len < 1e-3:
            continue
        n = np.array([-d[1], d[0]], dtype=np.float32) / seg_len  # left normal

        lw0_m, rw0_m = profile.get(idx0, (fallback_half_width, fallback_half_width))
        lw1_m, rw1_m = profile.get(idx1, (fallback_half_width, fallback_half_width))
        lw0 = max(1.0, float(lw0_m) / resolution)
        rw0 = max(1.0, float(rw0_m) / resolution)
        lw1 = max(1.0, float(lw1_m) / resolution)
        rw1 = max(1.0, float(rw1_m) / resolution)

        poly = np.array(
            [
                p0 + n * lw0,
                p1 + n * lw1,
                p1 - n * rw1,
                p0 - n * rw0,
            ],
            dtype=np.float32,
        )
        poly_i = np.round(poly).astype(np.int32)
        cv2.fillConvexPoly(base, poly_i, color=255)

        # Asymmetric end-caps: avoid forcing left/right to max(lw, rw).
        cap0_r = max(1.0, 0.5 * (lw0 + rw0))
        cap1_r = max(1.0, 0.5 * (lw1 + rw1))
        cap0_c = p0 + n * (0.5 * (lw0 - rw0))
        cap1_c = p1 + n * (0.5 * (lw1 - rw1))
        cv2.circle(
            base,
            (int(round(cap0_c[0])), int(round(cap0_c[1]))),
            int(round(cap0_r)),
            color=255,
            thickness=-1,
        )
        cv2.circle(
            base,
            (int(round(cap1_c[0])), int(round(cap1_c[1]))),
            int(round(cap1_r)),
            color=255,
            thickness=-1,
        )

    return base > 0


def write_pcd_header(fp, point_count: int, with_intensity: bool) -> None:
    if with_intensity:
        fields = "x y z intensity"
        size = "4 4 4 4"
        ftype = "F F F F"
        count = "1 1 1 1"
    else:
        fields = "x y z"
        size = "4 4 4"
        ftype = "F F F"
        count = "1 1 1"

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {size}\n"
        f"TYPE {ftype}\n"
        f"COUNT {count}\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    )
    fp.write(header.encode("ascii"))


def main() -> None:
    args = parse_args()

    if args.corridor_half_width <= 0:
        raise ValueError("--corridor-half-width must be > 0")
    if args.min_half_width <= 0 or args.max_half_width <= 0:
        raise ValueError("--min-half-width and --max-half-width must be > 0")
    if args.min_half_width > args.max_half_width:
        raise ValueError("--min-half-width cannot be greater than --max-half-width")
    if args.resolution <= 0:
        raise ValueError("--resolution must be > 0")
    if args.chunk_points <= 0:
        raise ValueError("--chunk-points must be > 0")
    if args.z_min > args.z_max:
        raise ValueError("--z-min cannot be greater than --z-max")
    if not args.global_map_pcd.exists():
        raise FileNotFoundError(args.global_map_pcd)
    if not args.pose_file.exists():
        raise FileNotFoundError(args.pose_file)
    if args.dynamic_profile is not None and not args.dynamic_profile.exists():
        raise FileNotFoundError(args.dynamic_profile)
    if args.auto_half_width:
        if args.ground_map_pcd is None or args.nonground_map_pcd is None:
            raise ValueError("--auto-half-width requires --ground-map-pcd and --nonground-map-pcd")
        if not args.ground_map_pcd.exists():
            raise FileNotFoundError(args.ground_map_pcd)
        if not args.nonground_map_pcd.exists():
            raise FileNotFoundError(args.nonground_map_pcd)

    pose_xy = load_pose_xy(args.pose_file)
    print(f"Loaded poses: {len(pose_xy)}")

    reader = BinaryPCD(args.global_map_pcd)
    ground_reader = BinaryPCD(args.ground_map_pcd) if args.auto_half_width else None
    nonground_reader = BinaryPCD(args.nonground_map_pcd) if args.auto_half_width else None
    bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]  # min_x, max_x, min_y, max_y
    update_bounds_from_reader(reader, bounds, args.chunk_points)
    if ground_reader is not None:
        update_bounds_from_reader(ground_reader, bounds, args.chunk_points)
    if nonground_reader is not None:
        update_bounds_from_reader(nonground_reader, bounds, args.chunk_points)
    bounds[0] = min(bounds[0], float(np.min(pose_xy[:, 0])))
    bounds[1] = max(bounds[1], float(np.max(pose_xy[:, 0])))
    bounds[2] = min(bounds[2], float(np.min(pose_xy[:, 1])))
    bounds[3] = max(bounds[3], float(np.max(pose_xy[:, 1])))
    if not np.isfinite(bounds).all():
        raise ValueError("Failed to determine map bounds.")

    min_x = bounds[0] - args.padding
    max_x = bounds[1] + args.padding
    min_y = bounds[2] - args.padding
    max_y = bounds[3] + args.padding
    grid_w = int(math.ceil((max_x - min_x) / args.resolution))
    grid_h = int(math.ceil((max_y - min_y) / args.resolution))
    if grid_w <= 1 or grid_h <= 1:
        raise ValueError("Invalid grid size from bounds.")

    dynamic_profile = None
    if args.dynamic_profile is not None:
        dynamic_profile = load_dynamic_profile(args.dynamic_profile)
        print(f"Loaded dynamic profile rows: {len(dynamic_profile)}")

    if dynamic_profile is not None:
        corridor = build_dynamic_corridor_mask(
            pose_xy=pose_xy,
            min_x=min_x,
            min_y=min_y,
            width=grid_w,
            height=grid_h,
            resolution=args.resolution,
            profile=dynamic_profile,
            fallback_half_width=args.corridor_half_width,
        )
        print(
            "Using dynamic profile corridor "
            f"(fallback half-width for missing seq: {args.corridor_half_width:.3f} m)"
        )
    elif args.auto_half_width:
        half_width = estimate_half_width_from_segmented(
            pose_xy=pose_xy,
            ground_reader=ground_reader,
            nonground_reader=nonground_reader,
            min_x=min_x,
            min_y=min_y,
            width=grid_w,
            height=grid_h,
            resolution=args.resolution,
            chunk_points=args.chunk_points,
            ground_min_points=args.ground_min_points,
            nonground_min_points=args.nonground_min_points,
            percentile=args.auto_percentile,
            margin=args.auto_margin,
            sample_radius_cells=args.auto_sample_radius_cells,
            min_half_width=args.min_half_width,
            max_half_width=args.max_half_width,
        )
        print(
            f"Auto half-width estimated: {half_width:.3f} m "
            f"(clamped to [{args.min_half_width:.3f}, {args.max_half_width:.3f}] m)"
        )
        corridor = build_corridor_mask(
            pose_xy=pose_xy,
            min_x=min_x,
            min_y=min_y,
            width=grid_w,
            height=grid_h,
            resolution=args.resolution,
            half_width=half_width,
        )
    else:
        half_width = args.corridor_half_width
        print(f"Using fixed half-width: {half_width:.3f} m")
        corridor = build_corridor_mask(
            pose_xy=pose_xy,
            min_x=min_x,
            min_y=min_y,
            width=grid_w,
            height=grid_h,
            resolution=args.resolution,
            half_width=half_width,
        )
    print(f"Grid: {grid_w} x {grid_h}, resolution={args.resolution}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="corridor_cloud_", suffix=".bin", dir=str(args.output.parent))
    os.close(tmp_fd)

    with_intensity = reader.has_intensity
    out_count = 0
    frame = 0
    try:
        with open(tmp_path, "wb") as tmp_fp:
            for chunk in reader.iter_chunks(args.chunk_points):
                frame += 1
                x = chunk["x"].astype(np.float32, copy=False)
                y = chunk["y"].astype(np.float32, copy=False)
                z = chunk["z"].astype(np.float32, copy=False)
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                x, y, z = x[valid], y[valid], z[valid]

                intensity = None
                if with_intensity:
                    intensity = chunk["intensity"].astype(np.float32, copy=False)[valid]

                gx, gy = world_to_grid(x, y, min_x, min_y, args.resolution)
                inside = in_grid(gx, gy, grid_w, grid_h)
                if not np.any(inside):
                    continue
                gx, gy = gx[inside], gy[inside]
                x, y, z = x[inside], y[inside], z[inside]
                if with_intensity:
                    intensity = intensity[inside]

                in_corridor = corridor[gy, gx]
                if args.mode == "remove_corridor":
                    keep = ~in_corridor
                else:
                    keep = in_corridor
                if args.z_min > -1e8:
                    keep &= z >= args.z_min
                if args.z_max < 1e8:
                    keep &= z <= args.z_max
                if not np.any(keep):
                    continue

                x, y, z = x[keep], y[keep], z[keep]
                if with_intensity:
                    intensity = intensity[keep]

                if with_intensity:
                    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
                    data = np.empty(len(x), dtype=dtype)
                    data["x"], data["y"], data["z"], data["intensity"] = x, y, z, intensity
                else:
                    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                    data = np.empty(len(x), dtype=dtype)
                    data["x"], data["y"], data["z"] = x, y, z

                tmp_fp.write(data.tobytes())
                out_count += len(x)
                if frame % 10 == 0:
                    print(f"Processed chunks: {frame}, kept points: {out_count}")

        with args.output.open("wb") as out_fp:
            write_pcd_header(out_fp, out_count, with_intensity)
            with open(tmp_path, "rb") as in_fp:
                while True:
                    buf = in_fp.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    out_fp.write(buf)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    print(f"Mode: {args.mode}")
    print(f"Saved: {args.output}")
    print(f"Kept points: {out_count}")


if __name__ == "__main__":
    main()
