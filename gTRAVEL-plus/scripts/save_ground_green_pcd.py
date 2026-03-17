#!/usr/bin/env python3
"""
Export per-frame PCD files with ground points colored green.

This node listens to ground-segmentation output (`/gtravelp/ground_pc` by default),
reads the corresponding source PCD from `pcd_dir` by frame seq, then writes a new
PCD to `output_dir` with ground points colored green.

- Ground points: RGB = (0, 255, 0)
- Non-ground points: original fields are kept unchanged
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2, PointField


ROS_TO_NP = {
    PointField.INT8: "i1",
    PointField.UINT8: "u1",
    PointField.INT16: "i2",
    PointField.UINT16: "u2",
    PointField.INT32: "i4",
    PointField.UINT32: "u4",
    PointField.FLOAT32: "f4",
    PointField.FLOAT64: "f8",
}


def numeric_stem(path: Path) -> int:
    stem = path.stem
    return int(stem) if stem.isdigit() else -1


def collect_pcd_files(pcd_dir: Path) -> List[Path]:
    files = [p for p in pcd_dir.glob("*.pcd") if p.is_file()]
    files.sort(key=lambda p: (numeric_stem(p) < 0, numeric_stem(p) if numeric_stem(p) >= 0 else p.name))
    return files


def make_np_view(msg: PointCloud2, wanted: Sequence[str]) -> np.ndarray:
    fields = {f.name: f for f in msg.fields}
    names, formats, offsets = [], [], []
    endian = ">" if msg.is_bigendian else "<"

    for name in wanted:
        if name not in fields:
            continue
        f = fields[name]
        if f.datatype not in ROS_TO_NP:
            continue
        base = ROS_TO_NP[f.datatype]
        fmt = "|" + base if base[1] == "1" else endian + base
        names.append(name)
        formats.append(fmt)
        offsets.append(f.offset)

    if not {"x", "y", "z"}.issubset(set(names)):
        raise ValueError("PointCloud2 missing x/y/z fields.")

    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step})
    n_points = msg.width * msg.height
    return np.ndarray(shape=(n_points,), dtype=dtype, buffer=msg.data)


def cloud_xyzi(msg: PointCloud2) -> np.ndarray:
    arr = make_np_view(msg, ("x", "y", "z", "intensity"))
    x = arr["x"].astype(np.float32, copy=False)
    y = arr["y"].astype(np.float32, copy=False)
    z = arr["z"].astype(np.float32, copy=False)

    if "intensity" in arr.dtype.names:
        intensity = arr["intensity"].astype(np.float32, copy=False)
    else:
        intensity = np.zeros_like(x, dtype=np.float32)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(intensity)
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float32)

    out = np.empty((int(np.sum(valid)), 4), dtype=np.float32)
    out[:, 0] = x[valid]
    out[:, 1] = y[valid]
    out[:, 2] = z[valid]
    out[:, 3] = intensity[valid]
    return out


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    arr = make_np_view(msg, ("x", "y", "z"))
    x = arr["x"].astype(np.float32, copy=False)
    y = arr["y"].astype(np.float32, copy=False)
    z = arr["z"].astype(np.float32, copy=False)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    out = np.empty((int(np.sum(valid)), 3), dtype=np.float32)
    out[:, 0] = x[valid]
    out[:, 1] = y[valid]
    out[:, 2] = z[valid]
    return out


def compose_colored_cloud(ground_xyz: np.ndarray, nonground_xyz: np.ndarray, ground_rgb_u32: int, nonground_rgb_u32: int) -> np.ndarray:
    n_ground = int(ground_xyz.shape[0])
    n_nonground = int(nonground_xyz.shape[0])
    n_total = n_ground + n_nonground

    out_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")])
    out = np.empty(n_total, dtype=out_dtype)

    if n_ground > 0:
        out["x"][:n_ground] = ground_xyz[:, 0]
        out["y"][:n_ground] = ground_xyz[:, 1]
        out["z"][:n_ground] = ground_xyz[:, 2]
    if n_nonground > 0:
        out["x"][n_ground:] = nonground_xyz[:, 0]
        out["y"][n_ground:] = nonground_xyz[:, 1]
        out["z"][n_ground:] = nonground_xyz[:, 2]

    rgb_u32 = np.empty(n_total, dtype=np.uint32)
    if n_ground > 0:
        rgb_u32[:n_ground] = np.uint32(ground_rgb_u32)
    if n_nonground > 0:
        rgb_u32[n_ground:] = np.uint32(nonground_rgb_u32)
    out["rgb"] = rgb_u32.view(np.float32)
    return out


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
        return arr.copy()

    def viewpoint(self) -> str:
        vp = self.header.get("VIEWPOINT", ["0", "0", "0", "1", "0", "0", "0"])
        return " ".join(vp)


def _dtype_to_pcd_meta(dtype: np.dtype) -> Tuple[List[str], List[str], List[str], List[str]]:
    names: List[str] = []
    sizes: List[str] = []
    types: List[str] = []
    counts: List[str] = []

    if dtype.names is None:
        raise ValueError("Expected a structured dtype for PCD output")

    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        if field_dtype.subdtype is not None:
            base, shape = field_dtype.subdtype
            count = int(np.prod(shape))
        else:
            base = field_dtype
            count = 1

        if base.kind == "f":
            type_code = "F"
        elif base.kind == "u":
            type_code = "U"
        elif base.kind == "i":
            type_code = "I"
        else:
            raise ValueError(f"Unsupported dtype kind for field '{name}': {base.kind}")

        names.append(name)
        sizes.append(str(base.itemsize))
        types.append(type_code)
        counts.append(str(count))

    return names, sizes, types, counts


def write_binary_pcd(path: Path, points: np.ndarray, viewpoint: str) -> None:
    names, sizes, types, counts = _dtype_to_pcd_meta(points.dtype)
    point_count = int(points.shape[0])

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {' '.join(names)}\n"
        f"SIZE {' '.join(sizes)}\n"
        f"TYPE {' '.join(types)}\n"
        f"COUNT {' '.join(counts)}\n"
        f"WIDTH {point_count}\n"
        "HEIGHT 1\n"
        f"VIEWPOINT {viewpoint}\n"
        f"POINTS {point_count}\n"
        "DATA binary\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(header.encode("ascii"))
        fp.write(np.ascontiguousarray(points).tobytes())


def pack_rgb_u32(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)


def field_to_u32(field: np.ndarray) -> np.ndarray:
    dt = field.dtype
    if dt.kind == "f" and dt.itemsize == 4:
        return field.view(np.uint32).copy()
    if dt.kind in ("u", "i") and dt.itemsize <= 4:
        return field.astype(np.uint32, copy=True)
    raise ValueError(f"Unsupported color field dtype: {dt}")


def u32_to_field(values: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    if target_dtype.kind == "f" and target_dtype.itemsize == 4:
        return values.view(np.float32)
    return values.astype(target_dtype, copy=False)


def build_ground_mask(src_points: np.ndarray, ground_xyzi: np.ndarray, xyz_quant: float = 1e-4) -> np.ndarray:
    n = int(src_points.shape[0])
    mask = np.zeros(n, dtype=np.bool_)
    if n == 0 or ground_xyzi.size == 0:
        return mask

    names = set(src_points.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        return mask

    q = float(max(1e-9, xyz_quant))
    src_xyz = np.empty((n, 3), dtype=np.float32)
    src_xyz[:, 0] = src_points["x"].astype(np.float32, copy=False)
    src_xyz[:, 1] = src_points["y"].astype(np.float32, copy=False)
    src_xyz[:, 2] = src_points["z"].astype(np.float32, copy=False)
    g_xyz = np.ascontiguousarray(ground_xyzi[:, :3].astype(np.float32, copy=False))

    src_q = np.round(src_xyz / q).astype(np.int64)
    g_q = np.round(g_xyz / q).astype(np.int64)

    counter = Counter(map(tuple, g_q.tolist()))
    for idx, key in enumerate(map(tuple, src_q.tolist())):
        remain = counter.get(key, 0)
        if remain <= 0:
            continue
        mask[idx] = True
        if remain == 1:
            del counter[key]
        else:
            counter[key] = remain - 1

    return mask


def colorize_points(
    points: np.ndarray,
    ground_mask: np.ndarray,
    green_rgb_u32: int,
    nonground_rgb_u32: int = 0x000000,
) -> np.ndarray:
    names = set(points.dtype.names or ())
    n = int(points.shape[0])

    if "rgb" in names:
        out = points.copy()
        rgb_u32 = np.full(n, np.uint32(nonground_rgb_u32), dtype=np.uint32)
        rgb_u32[ground_mask] = np.uint32(green_rgb_u32)
        out["rgb"] = u32_to_field(rgb_u32, out.dtype.fields["rgb"][0])
        return out

    if "rgba" in names:
        out = points.copy()
        rgba_u32 = np.full(n, np.uint32(0xFF000000 | nonground_rgb_u32), dtype=np.uint32)
        rgba_u32[ground_mask] = np.uint32(0xFF000000 | green_rgb_u32)
        out["rgba"] = u32_to_field(rgba_u32, out.dtype.fields["rgba"][0])
        return out

    rgb_u32 = np.full(n, np.uint32(nonground_rgb_u32), dtype=np.uint32)
    rgb_u32[ground_mask] = np.uint32(green_rgb_u32)

    out_dtype = np.dtype(points.dtype.descr + [("rgb", "<f4")])
    out = np.empty(points.shape[0], dtype=out_dtype)
    for name in points.dtype.names or ():
        out[name] = points[name]
    out["rgb"] = rgb_u32.view(np.float32)
    return out


def drop_field(points: np.ndarray, field_name: str) -> np.ndarray:
    names = points.dtype.names or ()
    if field_name not in names:
        return points

    new_dtype = np.dtype([(name, points.dtype.fields[name][0]) for name in names if name != field_name])
    out = np.empty(points.shape[0], dtype=new_dtype)
    for name in out.dtype.names or ():
        out[name] = points[name]
    return out


class GroundGreenPCDExporter:
    def __init__(self) -> None:
        self.pcd_dir = Path(rospy.get_param("~pcd_dir", "/home/ros/gTRAVEL_ws/src/map_315/pcd"))
        self.output_dir = Path(rospy.get_param("~output_dir", "/home/ros/gTRAVEL_ws/src/map_315/pcd_ground_green"))
        self.ground_topic = rospy.get_param("~ground_topic", "/gtravelp/ground_pc")
        self.nonground_topic = rospy.get_param("~nonground_topic", "/gtravelp/nonground_pc")
        self.start_idx = int(rospy.get_param("~start_idx", 0))

        self.expected_frames = int(rospy.get_param("~expected_frames", -1))
        self.auto_stop = bool(rospy.get_param("~auto_stop", True))
        self.overwrite = bool(rospy.get_param("~overwrite", True))

        green_r = int(rospy.get_param("~green_r", 0))
        green_g = int(rospy.get_param("~green_g", 255))
        green_b = int(rospy.get_param("~green_b", 0))
        self.green_rgb_u32 = ((green_r & 0xFF) << 16) | ((green_g & 0xFF) << 8) | (green_b & 0xFF)
        nonground_r = int(rospy.get_param("~nonground_r", 255))
        nonground_g = int(rospy.get_param("~nonground_g", 255))
        nonground_b = int(rospy.get_param("~nonground_b", 255))
        self.nonground_rgb_u32 = ((nonground_r & 0xFF) << 16) | ((nonground_g & 0xFF) << 8) | (nonground_b & 0xFF)

        files = collect_pcd_files(self.pcd_dir) if self.pcd_dir.exists() else []
        all_seqs = [numeric_stem(p) for p in files]
        all_seqs = sorted([s for s in all_seqs if s >= self.start_idx])

        if self.expected_frames > 0 and all_seqs:
            self.target_seqs = all_seqs[: min(self.expected_frames, len(all_seqs))]
        else:
            self.target_seqs = all_seqs
        self.target_seq_set = set(self.target_seqs) if self.target_seqs else None

        self.processed: set[int] = set()
        self.pending_ground: Dict[int, np.ndarray] = {}
        self.pending_nonground: Dict[int, np.ndarray] = {}
        self.first_seen_seq: int | None = None

        rospy.loginfo("Ground-green exporter")
        rospy.loginfo("pcd_dir: %s", str(self.pcd_dir))
        rospy.loginfo("output_dir: %s", str(self.output_dir))
        rospy.loginfo("ground_topic: %s", self.ground_topic)
        rospy.loginfo("nonground_topic: %s", self.nonground_topic)
        if self.target_seq_set is not None:
            rospy.loginfo("target frames: %d", len(self.target_seq_set))
        else:
            rospy.loginfo("target frames: unknown (no pcd index list)")

        self.sub_ground = rospy.Subscriber(self.ground_topic, PointCloud2, self._cb_ground, queue_size=200)
        self.sub_nonground = rospy.Subscriber(self.nonground_topic, PointCloud2, self._cb_nonground, queue_size=200)

    def _cb_ground(self, msg: PointCloud2) -> None:
        seq = int(msg.header.seq)
        if seq < self.start_idx:
            return
        self._on_first_seen(seq)
        if self.target_seq_set is not None and seq not in self.target_seq_set:
            return
        if seq in self.processed:
            return
        self.pending_ground[seq] = cloud_xyz(msg)
        self._consume_seq(seq)

    def _cb_nonground(self, msg: PointCloud2) -> None:
        seq = int(msg.header.seq)
        if seq < self.start_idx:
            return
        self._on_first_seen(seq)
        if self.target_seq_set is not None and seq not in self.target_seq_set:
            return
        if seq in self.processed:
            return
        self.pending_nonground[seq] = cloud_xyz(msg)
        self._consume_seq(seq)

    def _on_first_seen(self, seq: int) -> None:
        if self.first_seen_seq is not None:
            return
        self.first_seen_seq = seq
        if self.target_seq_set is None:
            return
        self.target_seq_set = {s for s in self.target_seq_set if s >= seq}
        rospy.loginfo("First seen seq=%d, adjusted target frames=%d", seq, len(self.target_seq_set))

    def _consume_seq(self, seq: int) -> None:
        if seq in self.processed:
            return
        ground_xyz = self.pending_ground.get(seq)
        nonground_xyz = self.pending_nonground.get(seq)
        if ground_xyz is None or nonground_xyz is None:
            return

        out_path = self.output_dir / f"{seq}.pcd"
        if out_path.exists() and not self.overwrite:
            self.processed.add(seq)
            self.pending_ground.pop(seq, None)
            self.pending_nonground.pop(seq, None)
            rospy.loginfo("Skip existing: %s", str(out_path))
            self._maybe_shutdown()
            return
        try:
            out_points = compose_colored_cloud(
                ground_xyz,
                nonground_xyz,
                self.green_rgb_u32,
                self.nonground_rgb_u32,
            )
            write_binary_pcd(out_path, out_points, "0 0 0 1 0 0 0")

            self.processed.add(seq)
            self.pending_ground.pop(seq, None)
            self.pending_nonground.pop(seq, None)
            rospy.loginfo(
                "Saved seq=%d -> %s (points=%d, ground=%d)",
                seq,
                str(out_path),
                int(out_points.shape[0]),
                int(ground_xyz.shape[0]),
            )
            self._maybe_shutdown()

        except Exception as exc:
            self.processed.add(seq)
            self.pending_ground.pop(seq, None)
            self.pending_nonground.pop(seq, None)
            rospy.logerr("Failed seq=%d: %s", seq, str(exc))
            self._maybe_shutdown()

    def _maybe_shutdown(self) -> None:
        done = len(self.processed)
        total = len(self.target_seq_set) if self.target_seq_set is not None else self.expected_frames
        if total > 0 and done > 0 and done % 20 == 0:
            rospy.loginfo("Progress: %d/%d", done, total)

        if self.auto_stop and total > 0 and done >= total:
            rospy.loginfo("All target frames handled (%d). Shutdown.", total)
            rospy.signal_shutdown("ground green export done")


def main() -> None:
    rospy.init_node("save_ground_green_pcd")
    GroundGreenPCDExporter()
    rospy.spin()


if __name__ == "__main__":
    main()
