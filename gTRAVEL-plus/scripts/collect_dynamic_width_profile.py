#!/usr/bin/env python3
"""
Collect per-frame asymmetric corridor widths using local history projection.

Input topics (default):
  - /gtravelp/ground_pc
  - /gtravelp/nonground_pc

Output:
  - CSV profile with columns: seq,left_width,right_width,...
"""

from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

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


@dataclass
class HistoryFrame:
    seq: int
    pose_xy: np.ndarray
    yaw: float
    ground_world: np.ndarray
    nonground_world: np.ndarray


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


def cloud_xyz(msg: PointCloud2) -> np.ndarray:
    arr = make_np_view(msg, ("x", "y", "z"))
    x = arr["x"].astype(np.float32, copy=False)
    y = arr["y"].astype(np.float32, copy=False)
    z = arr["z"].astype(np.float32, copy=False)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    return np.stack((x[valid], y[valid], z[valid]), axis=1)


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
        dtype=np.float32,
    )


def pose_matrix_from_values(values: Sequence[float]) -> np.ndarray:
    if len(values) == 7:
        tx, ty, tz = float(values[0]), float(values[1]), float(values[2])
        qw, qx, qy, qz = float(values[3]), float(values[4]), float(values[5]), float(values[6])
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = quaternion_to_rotation_matrix(qx, qy, qz, qw)
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        return T

    if len(values) == 12:
        T = np.eye(4, dtype=np.float32)
        T[:3, :] = np.asarray(values, dtype=np.float32).reshape(3, 4)
        return T

    if len(values) == 16:
        return np.asarray(values, dtype=np.float32).reshape(4, 4)

    raise ValueError(f"Unsupported pose row length: {len(values)}")


def _parse_pose_json(payload: object) -> List[np.ndarray]:
    if isinstance(payload, dict):
        if "poses" in payload:
            payload = payload["poses"]
        else:
            raise ValueError("JSON pose payload must be a list or contain 'poses'")

    if not isinstance(payload, list):
        raise ValueError("JSON pose payload must be a list")

    poses: List[np.ndarray] = []
    for item in payload:
        if isinstance(item, dict):
            if all(k in item for k in ("x", "y", "z", "qw", "qx", "qy", "qz")):
                vals = [item["x"], item["y"], item["z"], item["qw"], item["qx"], item["qy"], item["qz"]]
                poses.append(pose_matrix_from_values([float(v) for v in vals]))
                continue
            if all(k in item for k in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")):
                vals = [item["tx"], item["ty"], item["tz"], item["qw"], item["qx"], item["qy"], item["qz"]]
                poses.append(pose_matrix_from_values([float(v) for v in vals]))
                continue
            if "T" in item and isinstance(item["T"], list):
                poses.append(pose_matrix_from_values([float(v) for v in item["T"]]))
                continue
            raise ValueError(f"Unsupported JSON pose object: {item}")

        if isinstance(item, list):
            poses.append(pose_matrix_from_values([float(v) for v in item]))
            continue

        raise ValueError(f"Unsupported JSON pose entry: {type(item)}")

    return poses


def load_pose_transforms(path: Path) -> List[np.ndarray]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty pose file: {path}")

    try:
        payload = json.loads(text)
        poses = _parse_pose_json(payload)
        if poses:
            return poses
    except json.JSONDecodeError:
        pass

    poses: List[np.ndarray] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        try:
            poses.append(pose_matrix_from_values(vals))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc

    if not poses:
        raise ValueError(f"No valid pose rows in {path}")
    return poses


def pose_xy_yaw(T: np.ndarray) -> Tuple[np.ndarray, float]:
    xy = np.array([float(T[0, 3]), float(T[1, 3])], dtype=np.float32)
    yaw = math.atan2(float(T[1, 0]), float(T[0, 0]))
    return xy, yaw


def angle_diff_rad(a: float, b: float) -> float:
    d = a - b
    return (d + math.pi) % (2.0 * math.pi) - math.pi


def transform_points(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    return (xyz @ R.T + t).astype(np.float32, copy=False)


class WidthCollector:
    def __init__(self):
        self.pose_file = Path(rospy.get_param("~pose_file", "/home/ros/gTRAVEL_ws/src/map_test/pose.json"))
        self.output_csv = Path(
            rospy.get_param(
                "~output_csv",
                "/home/ros/gTRAVEL_ws/src/map_test/map/dynamic_width_profile.csv",
            )
        )
        self.ground_topic = rospy.get_param("~ground_topic", "/gtravelp/ground_pc")
        self.nonground_topic = rospy.get_param("~nonground_topic", "/gtravelp/nonground_pc")
        self.auto_stop = bool(rospy.get_param("~auto_stop", True))

        legacy_x_min = float(rospy.get_param("~x_min", -0.5))
        legacy_x_max = float(rospy.get_param("~x_max", 2.0))
        self.x_slice_min = float(rospy.get_param("~x_slice_min", legacy_x_min))
        self.x_slice_max = float(rospy.get_param("~x_slice_max", legacy_x_max))
        self.z_min = float(rospy.get_param("~z_min", -1.0))
        self.z_max = float(rospy.get_param("~z_max", 2.5))
        self.min_lateral = float(rospy.get_param("~min_lateral", 0.2))
        self.safety_margin = float(rospy.get_param("~safety_margin", 0.15))
        self.edge_margin = float(rospy.get_param("~edge_margin", 0.10))

        self.min_half_width = float(rospy.get_param("~min_half_width", 0.5))
        self.max_half_width = float(rospy.get_param("~max_half_width", 2.0))
        self.default_half_width = float(rospy.get_param("~default_half_width", 1.0))
        self.smoothing_alpha = float(rospy.get_param("~smoothing_alpha", 0.4))

        self.ground_percentile = float(rospy.get_param("~ground_percentile", 90.0))
        self.obs_percentile = float(rospy.get_param("~obs_percentile", 5.0))
        self.min_ground_pts = int(rospy.get_param("~min_ground_pts", 20))
        self.min_obs_pts = int(rospy.get_param("~min_obs_pts", 10))

        self.max_history_frames = int(rospy.get_param("~max_history_frames", 50))
        self.max_history_distance = float(rospy.get_param("~max_history_distance", 10.0))
        self.max_history_yaw_diff_deg = float(rospy.get_param("~max_history_yaw_diff_deg", 60.0))
        self.max_history_yaw_diff_rad = math.radians(max(0.0, self.max_history_yaw_diff_deg))
        self.buffer_point_radius = float(rospy.get_param("~buffer_point_radius", 25.0))

        pose_list = load_pose_transforms(self.pose_file)
        self.pose_T: List[np.ndarray] = [T.astype(np.float32, copy=False) for T in pose_list]
        self.pose_T_inv: List[np.ndarray] = [np.linalg.inv(T).astype(np.float32) for T in self.pose_T]
        pose_xy_yaw_list = [pose_xy_yaw(T) for T in self.pose_T]
        self.pose_xy = [v[0] for v in pose_xy_yaw_list]
        self.pose_yaw = [v[1] for v in pose_xy_yaw_list]

        self.expected_frames = int(rospy.get_param("~expected_frames", -1))
        if self.expected_frames <= 0:
            self.expected_frames = len(self.pose_T)
        if self.expected_frames > len(self.pose_T):
            rospy.logwarn(
                "expected_frames (%d) > pose count (%d), clamp to pose count",
                self.expected_frames,
                len(self.pose_T),
            )
            self.expected_frames = len(self.pose_T)

        self.pending_ground: Dict[int, np.ndarray] = {}
        self.pending_nonground: Dict[int, np.ndarray] = {}
        self.widths: Dict[int, Tuple[float, float]] = {}
        self.ground_stats: Dict[int, Dict[str, Optional[float]]] = {}
        self.obs_stats: Dict[int, Dict[str, Optional[float]]] = {}
        self.history_used: Dict[int, int] = {}
        self._history: Deque[HistoryFrame] = deque()
        self._processed_seq: set[int] = set()

        self._prev_left: Optional[float] = None
        self._prev_right: Optional[float] = None
        self._shutdown_once = False

        rospy.loginfo("Collect dynamic width profile (local submap projection)")
        rospy.loginfo("Expected frames: %d", self.expected_frames)
        rospy.loginfo("x slice [%.2f, %.2f], z range [%.2f, %.2f]", self.x_slice_min, self.x_slice_max, self.z_min, self.z_max)
        rospy.loginfo(
            "history frames=%d distance<=%.2fm yaw<=%.1fdeg radius<=%.2fm",
            self.max_history_frames,
            self.max_history_distance,
            self.max_history_yaw_diff_deg,
            self.buffer_point_radius,
        )
        rospy.loginfo("half-width clamp [%.2f, %.2f], default %.2f", self.min_half_width, self.max_half_width, self.default_half_width)

        self.sub_ground = rospy.Subscriber(self.ground_topic, PointCloud2, self._cb_ground, queue_size=200)
        self.sub_nonground = rospy.Subscriber(self.nonground_topic, PointCloud2, self._cb_nonground, queue_size=200)
        rospy.on_shutdown(self._on_shutdown)

    def _prefilter_sensor_points(self, xyz: np.ndarray) -> np.ndarray:
        if xyz.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        out = xyz
        if self.buffer_point_radius > 0.0:
            r2 = out[:, 0] * out[:, 0] + out[:, 1] * out[:, 1]
            out = out[r2 <= self.buffer_point_radius * self.buffer_point_radius]
        return out

    def _extract_ground_stats(self, xyz: np.ndarray) -> Dict[str, Optional[float]]:
        if xyz.size == 0:
            return {"left_ground": None, "right_ground": None, "n_ground": 0}

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        m = (x >= self.x_slice_min) & (x <= self.x_slice_max) & (z >= self.z_min) & (z <= self.z_max)
        y = y[m]

        out: Dict[str, Optional[float]] = {
            "left_ground": None,
            "right_ground": None,
            "n_ground": int(len(y)),
        }
        if len(y) < self.min_ground_pts:
            return out

        yl = y[y > self.min_lateral]
        yr = -y[y < -self.min_lateral]
        if len(yl) >= 5:
            out["left_ground"] = float(np.percentile(yl, np.clip(self.ground_percentile, 50.0, 99.0)))
        if len(yr) >= 5:
            out["right_ground"] = float(np.percentile(yr, np.clip(self.ground_percentile, 50.0, 99.0)))
        return out

    def _extract_obs_stats(self, xyz: np.ndarray) -> Dict[str, Optional[float]]:
        if xyz.size == 0:
            return {"left_obs": None, "right_obs": None, "n_obs": 0}

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        m = (x >= self.x_slice_min) & (x <= self.x_slice_max) & (z >= self.z_min) & (z <= self.z_max)
        y = y[m]

        out: Dict[str, Optional[float]] = {
            "left_obs": None,
            "right_obs": None,
            "n_obs": int(len(y)),
        }
        if len(y) < self.min_obs_pts:
            return out

        yl = y[y > self.min_lateral]
        yr = -y[y < -self.min_lateral]
        obs_q = float(np.clip(self.obs_percentile, 1.0, 40.0))
        if len(yl) >= 5:
            out["left_obs"] = float(np.percentile(yl, obs_q))
        if len(yr) >= 5:
            out["right_obs"] = float(np.percentile(yr, obs_q))
        return out

    def _select_history(self, seq: int) -> List[HistoryFrame]:
        cur_xy = self.pose_xy[seq]
        cur_yaw = self.pose_yaw[seq]

        selected: List[HistoryFrame] = []
        for frame in self._history:
            if seq - frame.seq > self.max_history_frames:
                continue
            d = float(np.linalg.norm(frame.pose_xy - cur_xy))
            if d > self.max_history_distance:
                continue
            if abs(angle_diff_rad(frame.yaw, cur_yaw)) > self.max_history_yaw_diff_rad:
                continue
            selected.append(frame)

        if not selected and self._history:
            selected.append(self._history[-1])
        return selected

    def _estimate_width(
        self, seq: int
    ) -> Tuple[Tuple[float, float], Dict[str, Optional[float]], Dict[str, Optional[float]], int]:
        selected = self._select_history(seq)
        T_cur_inv = self.pose_T_inv[seq]

        g_chunks = [transform_points(f.ground_world, T_cur_inv) for f in selected if f.ground_world.size > 0]
        n_chunks = [transform_points(f.nonground_world, T_cur_inv) for f in selected if f.nonground_world.size > 0]

        ground_local = np.concatenate(g_chunks, axis=0) if g_chunks else np.empty((0, 3), dtype=np.float32)
        nonground_local = np.concatenate(n_chunks, axis=0) if n_chunks else np.empty((0, 3), dtype=np.float32)

        g = self._extract_ground_stats(ground_local)
        o = self._extract_obs_stats(nonground_local)

        left_candidates: List[float] = []
        right_candidates: List[float] = []

        lg = g.get("left_ground")
        rg = g.get("right_ground")
        lo = o.get("left_obs")
        ro = o.get("right_obs")

        if lg is not None:
            left_candidates.append(float(lg) - self.edge_margin)
        if rg is not None:
            right_candidates.append(float(rg) - self.edge_margin)
        if lo is not None:
            left_candidates.append(float(lo) - self.safety_margin)
        if ro is not None:
            right_candidates.append(float(ro) - self.safety_margin)

        left = min(left_candidates) if left_candidates else self.default_half_width
        right = min(right_candidates) if right_candidates else self.default_half_width

        left = float(np.clip(left, self.min_half_width, self.max_half_width))
        right = float(np.clip(right, self.min_half_width, self.max_half_width))

        if self._prev_left is not None and self._prev_right is not None:
            a = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
            left = a * left + (1.0 - a) * self._prev_left
            right = a * right + (1.0 - a) * self._prev_right

        hist_used = len(selected)
        return (left, right), g, o, hist_used

    def _consume_seq(self, seq: int) -> None:
        if seq in self._processed_seq:
            return
        g_sensor = self.pending_ground.get(seq)
        n_sensor = self.pending_nonground.get(seq)
        if g_sensor is None or n_sensor is None:
            return
        if seq < 0 or seq >= self.expected_frames:
            return

        g_sensor = self._prefilter_sensor_points(g_sensor)
        n_sensor = self._prefilter_sensor_points(n_sensor)

        T_w_cur = self.pose_T[seq]
        g_world = transform_points(g_sensor, T_w_cur)
        n_world = transform_points(n_sensor, T_w_cur)

        frame = HistoryFrame(
            seq=seq,
            pose_xy=self.pose_xy[seq],
            yaw=self.pose_yaw[seq],
            ground_world=g_world,
            nonground_world=n_world,
        )
        self._history.append(frame)
        while len(self._history) > max(1, self.max_history_frames):
            self._history.popleft()

        (left, right), g_stats, o_stats, hist_used = self._estimate_width(seq)
        self.widths[seq] = (left, right)
        self.ground_stats[seq] = g_stats
        self.obs_stats[seq] = o_stats
        self.history_used[seq] = hist_used
        self._prev_left = left
        self._prev_right = right
        self._processed_seq.add(seq)

        self.pending_ground.pop(seq, None)
        self.pending_nonground.pop(seq, None)

        keep_after = seq - max(2 * self.max_history_frames, 200)
        stale_g = [k for k in self.pending_ground.keys() if k < keep_after]
        stale_n = [k for k in self.pending_nonground.keys() if k < keep_after]
        for k in stale_g:
            self.pending_ground.pop(k, None)
        for k in stale_n:
            self.pending_nonground.pop(k, None)

        rospy.loginfo_throttle(
            1.0,
            "profile frames=%d/%d latest seq=%d left=%.3f right=%.3f history=%d",
            len(self.widths),
            self.expected_frames,
            seq,
            left,
            right,
            hist_used,
        )

        if self.auto_stop and len(self.widths) >= self.expected_frames:
            rospy.loginfo("Collected all expected frames. Auto shutdown.")
            rospy.signal_shutdown("dynamic width profile collected")

    def _cb_ground(self, msg: PointCloud2) -> None:
        seq = int(msg.header.seq)
        if seq < 0 or seq >= self.expected_frames:
            return
        self.pending_ground[seq] = cloud_xyz(msg)
        self._consume_seq(seq)

    def _cb_nonground(self, msg: PointCloud2) -> None:
        seq = int(msg.header.seq)
        if seq < 0 or seq >= self.expected_frames:
            return
        self.pending_nonground[seq] = cloud_xyz(msg)
        self._consume_seq(seq)

    def _on_shutdown(self) -> None:
        if self._shutdown_once:
            return
        self._shutdown_once = True
        self._write_csv()

    def _write_csv(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        prev_l = self.default_half_width
        prev_r = self.default_half_width
        rows = []
        for seq in range(self.expected_frames):
            if seq in self.widths:
                l, r = self.widths[seq]
            else:
                l, r = prev_l, prev_r

            g = self.ground_stats.get(seq, {})
            o = self.obs_stats.get(seq, {})
            rows.append(
                {
                    "seq": seq,
                    "left_width": f"{l:.6f}",
                    "right_width": f"{r:.6f}",
                    "history_used": int(self.history_used.get(seq, 0)),
                    "left_ground": "" if g.get("left_ground") is None else f"{float(g['left_ground']):.6f}",
                    "right_ground": "" if g.get("right_ground") is None else f"{float(g['right_ground']):.6f}",
                    "left_obs": "" if o.get("left_obs") is None else f"{float(o['left_obs']):.6f}",
                    "right_obs": "" if o.get("right_obs") is None else f"{float(o['right_obs']):.6f}",
                    "n_ground": int(g.get("n_ground", 0)),
                    "n_obs": int(o.get("n_obs", 0)),
                }
            )
            prev_l, prev_r = l, r

        with self.output_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    "seq",
                    "left_width",
                    "right_width",
                    "history_used",
                    "left_ground",
                    "right_ground",
                    "left_obs",
                    "right_obs",
                    "n_ground",
                    "n_obs",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        rospy.loginfo("Saved dynamic width profile: %s (%d rows)", str(self.output_csv), len(rows))


def main() -> None:
    rospy.init_node("collect_dynamic_width_profile")
    WidthCollector()
    rospy.spin()


if __name__ == "__main__":
    main()
