#!/usr/bin/env python3
"""
Collect per-frame asymmetric corridor widths from segmented ROS topics.

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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    return np.stack((x[valid], y[valid], z[valid]), axis=1)


def _pose_from_values(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 7:
        return float(values[0]), float(values[1])
    if len(values) == 12:
        return float(values[3]), float(values[7])
    if len(values) == 16:
        return float(values[3]), float(values[7])
    raise ValueError(f"Unsupported pose row length: {len(values)}")


def load_pose_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty pose file: {path}")

    try:
        payload = json.loads(text)
        if isinstance(payload, dict) and "poses" in payload:
            payload = payload["poses"]
        if not isinstance(payload, list):
            raise ValueError("JSON pose payload must be a list")
        count = 0
        for item in payload:
            if isinstance(item, dict):
                if ("x" in item and "y" in item) or ("tx" in item and "ty" in item):
                    count += 1
                elif "T" in item and isinstance(item["T"], list):
                    _pose_from_values([float(v) for v in item["T"]])
                    count += 1
                else:
                    raise ValueError(f"Unsupported JSON pose object: {item}")
            elif isinstance(item, list):
                _pose_from_values([float(v) for v in item])
                count += 1
            else:
                raise ValueError(f"Unsupported JSON pose entry: {type(item)}")
        return count
    except json.JSONDecodeError:
        pass

    count = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        _pose_from_values(vals)
        count += 1
    return count


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

        self.x_min = float(rospy.get_param("~x_min", 0.5))
        self.x_max = float(rospy.get_param("~x_max", 8.0))
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
        self.min_ground_pts = int(rospy.get_param("~min_ground_pts", 20))
        self.min_obs_pts = int(rospy.get_param("~min_obs_pts", 5))

        self.expected_frames = int(rospy.get_param("~expected_frames", -1))
        if self.expected_frames <= 0:
            self.expected_frames = load_pose_count(self.pose_file)

        self.ground_stats: Dict[int, Dict[str, Optional[float]]] = {}
        self.obs_stats: Dict[int, Dict[str, Optional[float]]] = {}
        self.widths: Dict[int, Tuple[float, float]] = {}
        self._prev_left: Optional[float] = None
        self._prev_right: Optional[float] = None
        self._shutdown_once = False

        rospy.loginfo("Collect dynamic width profile")
        rospy.loginfo("Expected frames: %d", self.expected_frames)
        rospy.loginfo("x range [%.2f, %.2f], z range [%.2f, %.2f]", self.x_min, self.x_max, self.z_min, self.z_max)
        rospy.loginfo("half-width clamp [%.2f, %.2f], default %.2f", self.min_half_width, self.max_half_width, self.default_half_width)

        self.sub_ground = rospy.Subscriber(self.ground_topic, PointCloud2, self._cb_ground, queue_size=200)
        self.sub_nonground = rospy.Subscriber(self.nonground_topic, PointCloud2, self._cb_nonground, queue_size=200)
        rospy.on_shutdown(self._on_shutdown)

    def _extract_ground_stats(self, xyz: np.ndarray) -> Dict[str, Optional[float]]:
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        m = (x >= self.x_min) & (x <= self.x_max) & (z >= self.z_min) & (z <= self.z_max)
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
            out["left_ground"] = float(np.percentile(yl, self.ground_percentile))
        if len(yr) >= 5:
            out["right_ground"] = float(np.percentile(yr, self.ground_percentile))
        return out

    def _extract_obs_stats(self, xyz: np.ndarray) -> Dict[str, Optional[float]]:
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        m = (x >= self.x_min) & (x <= self.x_max) & (z >= self.z_min) & (z <= self.z_max)
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
        if len(yl) > 0:
            out["left_obs"] = float(np.min(yl))
        if len(yr) > 0:
            out["right_obs"] = float(np.min(yr))
        return out

    def _fuse_widths(self, seq: int) -> None:
        if seq in self.widths:
            return
        g = self.ground_stats.get(seq)
        o = self.obs_stats.get(seq)
        if g is None or o is None:
            return

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

        self._prev_left = left
        self._prev_right = right
        self.widths[seq] = (left, right)

        rospy.loginfo_throttle(
            1.0,
            "profile frames=%d/%d latest seq=%d left=%.3f right=%.3f",
            len(self.widths),
            self.expected_frames,
            seq,
            left,
            right,
        )

        if self.auto_stop and len(self.widths) >= self.expected_frames:
            rospy.loginfo("Collected all expected frames. Auto shutdown.")
            rospy.signal_shutdown("dynamic width profile collected")

    def _cb_ground(self, msg: PointCloud2) -> None:
        seq = int(msg.header.seq)
        if seq < 0 or seq >= self.expected_frames:
            return
        xyz = cloud_xyz(msg)
        self.ground_stats[seq] = self._extract_ground_stats(xyz)
        self._fuse_widths(seq)

    def _cb_nonground(self, msg: PointCloud2) -> None:
        seq = int(msg.header.seq)
        if seq < 0 or seq >= self.expected_frames:
            return
        xyz = cloud_xyz(msg)
        self.obs_stats[seq] = self._extract_obs_stats(xyz)
        self._fuse_widths(seq)

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
                source = "measured"
            else:
                l, r = prev_l, prev_r
                source = "filled_prev"

            g = self.ground_stats.get(seq, {})
            o = self.obs_stats.get(seq, {})
            rows.append(
                {
                    "seq": seq,
                    "left_width": f"{l:.6f}",
                    "right_width": f"{r:.6f}",
                    "source": source,
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
                    "source",
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

