#!/usr/bin/env python3
"""
Build a global RGB point-cloud map by stitching per-frame colored PCD files with poses.

Default input:
  - PCDs:  /home/ros/gTRAVEL_ws/src/map_315/pcd_ground_green
  - Poses: /home/ros/gTRAVEL_ws/src/map_315/pose.json

Default output:
  - /home/ros/gTRAVEL_ws/src/map_315/map/global_map_ground_green.pcd
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch colored per-frame PCDs into a global RGB map.")
    parser.add_argument(
        "--pcd-dir",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_315/pcd_ground_green"),
        help="Directory containing per-frame colored .pcd files.",
    )
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_315/pose.json"),
        help="Pose file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_315/map/global_map_ground_green.pcd"),
        help="Output global RGB map path (.pcd).",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="Start frame index in sorted pcd list.")
    parser.add_argument("--end-idx", type=int, default=-1, help="End frame index (inclusive). -1 means last.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Maximum number of frames to process.")
    parser.add_argument(
        "--pose-index-mode",
        choices=("filename", "order"),
        default="filename",
        help="Pose index source: numeric file stem or sorted frame order.",
    )
    parser.add_argument(
        "--pose-index-offset",
        type=int,
        default=0,
        help="Additive offset applied to computed pose index (e.g., +2 means pose_idx=base_idx+2).",
    )
    parser.add_argument("--invert-pose", action="store_true", help="Use inverse of each pose matrix.")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsample size in meters.")
    return parser.parse_args()


def numeric_stem(path: Path) -> int:
    stem = path.stem
    return int(stem) if stem.isdigit() else -1


def collect_pcd_files(pcd_dir: Path) -> List[Path]:
    files = [p for p in pcd_dir.glob("*.pcd") if p.is_file()]
    files.sort(key=lambda p: (numeric_stem(p) < 0, numeric_stem(p) if numeric_stem(p) >= 0 else p.name))
    return files


class BinaryPCD:
    def __init__(self, path: Path):
        self.path = path
        self.header, self.data_offset = self._read_header()
        data_mode = self.header.get("DATA", [""])[0].lower()
        if data_mode != "binary":
            raise ValueError(f"{path}: only DATA binary is supported, got {data_mode}")

        self.fields = self.header["FIELDS"]
        self.sizes = [int(v) for v in self.header["SIZE"]]
        self.types = self.header["TYPE"]
        self.counts = [int(v) for v in self.header.get("COUNT", ["1"] * len(self.fields))]
        if not (len(self.fields) == len(self.sizes) == len(self.types) == len(self.counts)):
            raise ValueError(f"{path}: malformed PCD field metadata")

        if "POINTS" in self.header:
            self.points = int(self.header["POINTS"][0])
        else:
            width = int(self.header.get("WIDTH", ["0"])[0])
            height = int(self.header.get("HEIGHT", ["1"])[0])
            self.points = width * height

        self.dtype = self._build_dtype()
        for axis in ("x", "y", "z"):
            if axis not in self.dtype.names:
                raise ValueError(f"{path}: missing required field '{axis}'")

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

    def read_points(self) -> np.ndarray:
        with self.path.open("rb") as fp:
            fp.seek(self.data_offset)
            arr = np.fromfile(fp, dtype=self.dtype, count=self.points)
        if len(arr) != self.points:
            raise ValueError(f"{self.path}: truncated data. expected {self.points}, got {len(arr)}")
        return arr


def read_pcd_xyz_rgb(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pcd = BinaryPCD(path)
    points = pcd.read_points()

    xyz = np.empty((points.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = points["x"].astype(np.float32, copy=False)
    xyz[:, 1] = points["y"].astype(np.float32, copy=False)
    xyz[:, 2] = points["z"].astype(np.float32, copy=False)

    names = set(points.dtype.names or ())
    if "rgb" in names:
        rgb_field = points["rgb"]
        if rgb_field.dtype.kind == "f" and rgb_field.dtype.itemsize == 4:
            rgb = rgb_field.view(np.uint32).copy()
        else:
            rgb = rgb_field.astype(np.uint32, copy=True)
    elif "rgba" in names:
        rgba = points["rgba"]
        rgba_u32 = rgba.view(np.uint32).copy() if (rgba.dtype.kind == "f" and rgba.dtype.itemsize == 4) else rgba.astype(np.uint32, copy=True)
        rgb = rgba_u32 & np.uint32(0x00FFFFFF)
    else:
        raise ValueError(f"{path}: missing rgb/rgba field")

    valid = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)

    return xyz[valid], rgb[valid]


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_from_values(values: Sequence[float]) -> np.ndarray:
    if len(values) == 7:
        tx, ty, tz = values[:3]
        qw, qx, qy, qz = values[3:]
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        return T
    if len(values) == 12:
        T = np.eye(4, dtype=np.float64)
        T[:3, :] = np.asarray(values, dtype=np.float64).reshape(3, 4)
        return T
    if len(values) == 16:
        return np.asarray(values, dtype=np.float64).reshape(4, 4)
    raise ValueError(f"Unsupported pose format with {len(values)} values")


def load_poses(path: Path) -> List[np.ndarray]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty pose file: {path}")

    poses: List[np.ndarray] = []
    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "poses" in payload:
            payload = payload["poses"]
        if not isinstance(payload, list):
            raise ValueError("JSON pose payload must be list.")

        for item in payload:
            if isinstance(item, dict):
                if all(k in item for k in ("x", "y", "z", "qw", "qx", "qy", "qz")):
                    vals = [item["x"], item["y"], item["z"], item["qw"], item["qx"], item["qy"], item["qz"]]
                elif all(k in item for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")):
                    vals = [item["tx"], item["ty"], item["tz"], item["qw"], item["qx"], item["qy"], item["qz"]]
                elif "T" in item and isinstance(item["T"], list):
                    vals = item["T"]
                else:
                    raise ValueError(f"Unsupported JSON pose object: {item}")
                poses.append(pose_from_values([float(v) for v in vals]))
            elif isinstance(item, list):
                poses.append(pose_from_values([float(v) for v in item]))
            else:
                raise ValueError(f"Unsupported JSON pose element: {type(item)}")

        if poses:
            return poses
    except json.JSONDecodeError:
        pass

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        try:
            poses.append(pose_from_values(vals))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc

    if not poses:
        raise ValueError(f"No valid pose rows in {path}")
    return poses


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    return (xyz @ R.T + t).astype(np.float32, copy=False)


def voxel_downsample_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0.0 or xyz.shape[0] == 0:
        return xyz, rgb

    keys = np.floor(xyz / float(voxel_size)).astype(np.int64)
    _, first_idx = np.unique(keys, axis=0, return_index=True)
    first_idx.sort()
    return xyz[first_idx], rgb[first_idx]


def write_pcd_xyz_rgb(path: Path, xyz: np.ndarray, rgb_u32: np.ndarray) -> None:
    if xyz.shape[0] != rgb_u32.shape[0]:
        raise ValueError("xyz and rgb size mismatch")

    n = xyz.shape[0]
    out_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    out = np.empty(n, dtype=out_dtype)
    out["x"] = xyz[:, 0]
    out["y"] = xyz[:, 1]
    out["z"] = xyz[:, 2]
    out["rgb"] = rgb_u32.astype(np.uint32, copy=False).view(np.float32)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(header.encode("ascii"))
        fp.write(np.ascontiguousarray(out).tobytes())


def main() -> None:
    args = parse_args()

    if not args.pcd_dir.exists():
        raise FileNotFoundError(args.pcd_dir)
    if not args.pose_file.exists():
        raise FileNotFoundError(args.pose_file)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    if args.start_idx < 0:
        raise ValueError("--start-idx must be >= 0")

    files = collect_pcd_files(args.pcd_dir)
    if not files:
        raise RuntimeError(f"No PCD files found in {args.pcd_dir}")

    end_idx = len(files) - 1 if args.end_idx < 0 else min(args.end_idx, len(files) - 1)
    if args.start_idx > end_idx:
        raise ValueError("start/end range is empty")

    selected = files[args.start_idx : end_idx + 1 : args.frame_stride]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]

    poses = load_poses(args.pose_file)

    xyz_chunks: List[np.ndarray] = []
    rgb_chunks: List[np.ndarray] = []

    used = 0
    skipped = 0
    for order_idx, pcd_path in enumerate(selected):
        base_idx = order_idx if args.pose_index_mode == "order" else numeric_stem(pcd_path)
        pose_idx = base_idx + int(args.pose_index_offset)
        if pose_idx < 0 or pose_idx >= len(poses):
            print(
                f"[WARN] skip {pcd_path.name}: invalid pose index {pose_idx} "
                f"(base={base_idx}, offset={args.pose_index_offset})"
            )
            skipped += 1
            continue

        xyz, rgb = read_pcd_xyz_rgb(pcd_path)
        if xyz.shape[0] == 0:
            skipped += 1
            continue

        T = np.linalg.inv(poses[pose_idx]) if args.invert_pose else poses[pose_idx]
        xyz_w = transform_points(xyz, T)

        xyz_chunks.append(xyz_w)
        rgb_chunks.append(rgb)
        used += 1

        if used % 100 == 0:
            print(f"Processed {used} frames...")

    if not xyz_chunks:
        raise RuntimeError("No valid frames were stitched")

    xyz_all = np.concatenate(xyz_chunks, axis=0)
    rgb_all = np.concatenate(rgb_chunks, axis=0)

    xyz_all, rgb_all = voxel_downsample_xyz_rgb(xyz_all, rgb_all, args.voxel_size)
    write_pcd_xyz_rgb(args.output, xyz_all, rgb_all)

    print("=== Done ===")
    print(f"Frames used: {used}")
    print(f"Frames skipped: {skipped}")
    print(f"Points written: {xyz_all.shape[0]}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
