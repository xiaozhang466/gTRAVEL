#!/usr/bin/env python3
"""
Build a global point-cloud map by stitching per-frame LiDAR PCD files with poses.

Default input:
  - PCDs:  /home/ros/gTRAVEL_ws/src/map_test/pcd
  - Poses: /home/ros/gTRAVEL_ws/src/map_test/pose.json

Default output:
  - /home/ros/gTRAVEL_ws/src/map_test/map/global_map.pcd
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch LiDAR frames into a global PCD map.")
    parser.add_argument(
        "--pcd-dir",
        type=Path,
        default=Path("/home/ros/gTRAVEL_ws/src/map_test/pcd"),
        help="Directory containing per-frame .pcd files.",
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
        default=Path("/home/ros/gTRAVEL_ws/src/map_test/map/global_map.pcd"),
        help="Output global map path (.pcd).",
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
    parser.add_argument("--invert-pose", action="store_true", help="Use inverse of each pose matrix.")
    parser.add_argument(
        "--quat-order",
        choices=("wxyz", "xyzw"),
        default="wxyz",
        help="Quaternion field order in 7-value poses: wxyz means x y z qw qx qy qz; xyzw means x y z qx qy qz qw.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Per-frame voxel downsample size in meters.")
    parser.add_argument("--no-intensity", action="store_true", help="Do not write intensity in output map.")
    return parser.parse_args()


def numeric_stem(path: Path) -> int:
    stem = path.stem
    return int(stem) if stem.isdigit() else -1


def collect_pcd_files(pcd_dir: Path) -> List[Path]:
    files = [p for p in pcd_dir.glob("*.pcd") if p.is_file()]
    files.sort(key=lambda p: (numeric_stem(p) < 0, numeric_stem(p) if numeric_stem(p) >= 0 else p.name))
    return files


def _pcd_scalar_dtype(type_code: str, size: int) -> np.dtype:
    if type_code == "F":
        table = {4: np.float32, 8: np.float64}
    elif type_code == "U":
        table = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
    elif type_code == "I":
        table = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}
    else:
        raise ValueError(f"Unsupported PCD TYPE: {type_code}")

    if size not in table:
        raise ValueError(f"Unsupported PCD SIZE {size} for TYPE {type_code}")

    base = np.dtype(table[size])
    if size > 1:
        base = base.newbyteorder("<")
    return base


def _parse_pcd_header(fp) -> Dict[str, List[str]]:
    header: Dict[str, List[str]] = {}
    while True:
        line = fp.readline()
        if not line:
            raise ValueError("Unexpected EOF while parsing PCD header")
        text = line.decode("utf-8", errors="ignore").strip()
        if not text or text.startswith("#"):
            continue
        tokens = text.split()
        key = tokens[0].upper()
        header[key] = tokens[1:]
        if key == "DATA":
            break
    return header


def read_pcd_xyz_intensity(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with path.open("rb") as fp:
        header = _parse_pcd_header(fp)

        data_mode = header.get("DATA", [""])[0].lower()
        if data_mode != "binary":
            raise ValueError(f"{path}: only DATA binary is supported, got {data_mode}")

        fields = header["FIELDS"]
        sizes = [int(v) for v in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]

        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"{path}: malformed PCD field metadata")

        if "POINTS" in header:
            point_count = int(header["POINTS"][0])
        else:
            width = int(header.get("WIDTH", ["0"])[0])
            height = int(header.get("HEIGHT", ["1"])[0])
            point_count = width * height

        dtype_fields = []
        for name, size, type_code, count in zip(fields, sizes, types, counts):
            scalar_dtype = _pcd_scalar_dtype(type_code, size)
            if count == 1:
                dtype_fields.append((name, scalar_dtype))
            else:
                dtype_fields.append((name, scalar_dtype, (count,)))
        pcd_dtype = np.dtype(dtype_fields)

        raw = fp.read(point_count * pcd_dtype.itemsize)
        if len(raw) != point_count * pcd_dtype.itemsize:
            raise ValueError(f"{path}: data size mismatch, expected {point_count * pcd_dtype.itemsize} bytes")

    data = np.frombuffer(raw, dtype=pcd_dtype, count=point_count)
    for axis in ("x", "y", "z"):
        if axis not in data.dtype.names:
            raise ValueError(f"{path}: missing required field '{axis}'")

    xyz = np.empty((point_count, 3), dtype=np.float32)
    xyz[:, 0] = data["x"].astype(np.float32, copy=False)
    xyz[:, 1] = data["y"].astype(np.float32, copy=False)
    xyz[:, 2] = data["z"].astype(np.float32, copy=False)

    intensity = None
    if "intensity" in data.dtype.names:
        intensity = np.asarray(data["intensity"], dtype=np.float32).reshape(-1)

    return xyz, intensity


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


def pose_from_values(values: Sequence[float], quat_order: str = "wxyz") -> np.ndarray:
    if len(values) == 7:
        tx, ty, tz = values[:3]
        if quat_order == "wxyz":
            qw, qx, qy, qz = values[3:]
        else:
            qx, qy, qz, qw = values[3:]
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


def _parse_pose_json(payload, quat_order: str = "wxyz") -> List[np.ndarray]:
    if isinstance(payload, dict):
        if "poses" in payload:
            payload = payload["poses"]
        else:
            raise ValueError("JSON pose file must contain a top-level list or 'poses' key")

    if not isinstance(payload, list):
        raise ValueError("JSON pose file must be a list")

    poses: List[np.ndarray] = []
    for item in payload:
        if isinstance(item, dict):
            if all(k in item for k in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")):
                vals = [item["tx"], item["ty"], item["tz"], item["qx"], item["qy"], item["qz"], item["qw"]]
                poses.append(pose_from_values([float(v) for v in vals], quat_order="xyzw"))
                continue
            elif all(k in item for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")):
                vals = [item["tx"], item["ty"], item["tz"], item["qw"], item["qx"], item["qy"], item["qz"]]
                poses.append(pose_from_values([float(v) for v in vals], quat_order="wxyz"))
                continue
            elif all(k in item for k in ("x", "y", "z", "qx", "qy", "qz", "qw")):
                vals = [item["x"], item["y"], item["z"], item["qx"], item["qy"], item["qz"], item["qw"]]
                poses.append(pose_from_values([float(v) for v in vals], quat_order="xyzw"))
                continue
            elif all(k in item for k in ("x", "y", "z", "qw", "qx", "qy", "qz")):
                vals = [item["x"], item["y"], item["z"], item["qw"], item["qx"], item["qy"], item["qz"]]
                poses.append(pose_from_values([float(v) for v in vals], quat_order="wxyz"))
                continue
            elif "T" in item and isinstance(item["T"], list):
                vals = item["T"]
            else:
                raise ValueError(f"Unsupported JSON pose object: {item}")
            poses.append(pose_from_values([float(v) for v in vals], quat_order=quat_order))
        elif isinstance(item, list):
            poses.append(pose_from_values([float(v) for v in item], quat_order=quat_order))
        else:
            raise ValueError(f"Unsupported JSON pose element type: {type(item)}")
    return poses


def load_poses(path: Path, quat_order: str = "wxyz") -> List[np.ndarray]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty pose file: {path}")

    try:
        payload = json.loads(text)
        return _parse_pose_json(payload, quat_order=quat_order)
    except json.JSONDecodeError:
        pass

    poses: List[np.ndarray] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        values = [float(v) for v in line.replace(",", " ").split()]
        try:
            poses.append(pose_from_values(values, quat_order=quat_order))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} -> {exc}") from exc
    return poses


def voxel_downsample(points: np.ndarray, intensity: Optional[np.ndarray], voxel_size: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size <= 0.0 or len(points) == 0:
        return points, intensity
    voxel = np.floor(points / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(voxel, axis=0, return_index=True)
    keep_idx.sort()
    sampled_points = points[keep_idx]
    sampled_intensity = intensity[keep_idx] if intensity is not None else None
    return sampled_points, sampled_intensity


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

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be >= 1")

    pcd_files = collect_pcd_files(args.pcd_dir)
    if not pcd_files:
        raise FileNotFoundError(f"No .pcd files found in {args.pcd_dir}")

    poses = load_poses(args.pose_file, quat_order=args.quat_order)
    if not poses:
        raise ValueError(f"No poses loaded from {args.pose_file}")

    start_idx = args.start_idx
    end_idx = len(pcd_files) - 1 if args.end_idx < 0 else min(args.end_idx, len(pcd_files) - 1)
    if start_idx < 0 or start_idx >= len(pcd_files):
        raise IndexError(f"start_idx out of range: {start_idx}, total files: {len(pcd_files)}")
    if end_idx < start_idx:
        raise IndexError(f"end_idx ({end_idx}) is smaller than start_idx ({start_idx})")

    selected: List[Tuple[int, Path]] = [(i, pcd_files[i]) for i in range(start_idx, end_idx + 1, args.frame_stride)]
    if args.max_frames > 0:
        selected = selected[: args.max_frames]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_bin = tempfile.mkstemp(prefix="global_map_", suffix=".bin", dir=str(args.output.parent))
    os.close(fd)

    with_intensity = not args.no_intensity
    out_dtype = (
        np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
        if with_intensity
        else np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    )

    total_points = 0
    try:
        with open(temp_bin, "wb") as tmp_fp:
            for frame_i, (order_idx, pcd_path) in enumerate(selected):
                file_id = numeric_stem(pcd_path)
                if args.pose_index_mode == "filename" and 0 <= file_id < len(poses):
                    pose_idx = file_id
                else:
                    pose_idx = order_idx
                    if pose_idx >= len(poses):
                        raise IndexError(
                            f"Pose index out of range for {pcd_path.name}: pose_idx={pose_idx}, pose_count={len(poses)}"
                        )

                T = poses[pose_idx]
                if args.invert_pose:
                    T = np.linalg.inv(T)

                points, intensity = read_pcd_xyz_intensity(pcd_path)
                if args.voxel_size > 0.0:
                    points, intensity = voxel_downsample(points, intensity, args.voxel_size)

                R = T[:3, :3].astype(np.float32)
                t = T[:3, 3].astype(np.float32)
                points_global = points @ R.T + t

                chunk = np.empty(points_global.shape[0], dtype=out_dtype)
                chunk["x"] = points_global[:, 0]
                chunk["y"] = points_global[:, 1]
                chunk["z"] = points_global[:, 2]
                if with_intensity:
                    if intensity is None:
                        chunk["intensity"] = 0.0
                    else:
                        chunk["intensity"] = intensity

                tmp_fp.write(chunk.tobytes())
                total_points += points_global.shape[0]

                if (frame_i + 1) % 50 == 0 or frame_i + 1 == len(selected):
                    print(
                        f"[{frame_i + 1:4d}/{len(selected)}] {pcd_path.name} "
                        f"pose={pose_idx} accumulated_points={total_points}"
                    )

        with open(args.output, "wb") as out_fp:
            write_pcd_header(out_fp, total_points, with_intensity)
            with open(temp_bin, "rb") as tmp_fp:
                shutil.copyfileobj(tmp_fp, out_fp, length=16 * 1024 * 1024)

    finally:
        if os.path.exists(temp_bin):
            os.remove(temp_bin)

    print(f"Saved global map: {args.output}")
    print(f"Total frames: {len(selected)}, total points: {total_points}")


if __name__ == "__main__":
    main()
