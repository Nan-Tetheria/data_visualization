#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make sure repo root is on sys.path if you import this file from scripts/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.actuations_to_joints import ActuationsToJointsModel  # noqa: E402


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d)
    orig_shape = rot6d.shape[:-1]
    assert rot6d.shape[-1] == 6, f"Expected last dim=6, got {rot6d.shape}"

    x = rot6d.reshape(-1, 2, 3)
    row1 = x[:, 0, :]
    row2 = x[:, 1, :]

    row1 = row1 / (np.linalg.norm(row1, axis=1, keepdims=True) + 1e-12)
    row2 = row2 - (np.sum(row1 * row2, axis=1, keepdims=True)) * row1
    row2 = row2 / (np.linalg.norm(row2, axis=1, keepdims=True) + 1e-12)
    row3 = np.cross(row1, row2)

    R = np.stack([row1, row2, row3], axis=1)   # rows
    R = np.transpose(R, (0, 2, 1))             # column-major
    return R.reshape(*orig_shape, 3, 3)


def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # URDF: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    a = axis / n
    x, y, z = a
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ], dtype=np.float64)


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def origin_to_T(xyz: Tuple[float, float, float], rpy: Tuple[float, float, float]) -> np.ndarray:
    R = rpy_to_R(rpy[0], rpy[1], rpy[2])
    return make_T(R, np.array(xyz, dtype=np.float64))


@dataclass
class Joint:
    name: str
    jtype: str
    parent: str
    child: str
    origin_xyz: Tuple[float, float, float]
    origin_rpy: Tuple[float, float, float]
    axis: np.ndarray
    limit: Optional[Tuple[float, float]]


def _parse_floats(s: Optional[str], n: int, default: float = 0.0) -> Tuple[float, ...]:
    if s is None:
        return tuple([default] * n)
    parts = s.strip().split()
    if len(parts) != n:
        raise ValueError(f"Expected {n} floats, got '{s}'")
    return tuple(float(x) for x in parts)


def load_urdf_joints(urdf_path: str) -> Tuple[List[Joint], List[str]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links: List[str] = []
    for link in root.findall("link"):
        name = link.get("name")
        if name:
            links.append(name)

    joints: List[Joint] = []
    for j in root.findall("joint"):
        name = j.get("name")
        jtype = j.get("type")
        if not name or not jtype:
            continue

        parent_el = j.find("parent")
        child_el = j.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link")
        child = child_el.get("link")
        if not parent or not child:
            continue

        origin_el = j.find("origin")
        if origin_el is not None:
            xyz = _parse_floats(origin_el.get("xyz"), 3, 0.0)
            rpy = _parse_floats(origin_el.get("rpy"), 3, 0.0)
        else:
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)

        axis_el = j.find("axis")
        if axis_el is not None and axis_el.get("xyz") is not None:
            axis = np.array(_parse_floats(axis_el.get("xyz"), 3, 0.0), dtype=np.float64)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        limit_el = j.find("limit")
        limit = None
        if limit_el is not None and (limit_el.get("lower") is not None or limit_el.get("upper") is not None):
            lower = float(limit_el.get("lower", "0.0"))
            upper = float(limit_el.get("upper", "0.0"))
            limit = (lower, upper)

        joints.append(Joint(
            name=name,
            jtype=jtype,
            parent=parent,
            child=child,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
            limit=limit
        ))

    return joints, links


def find_root_link(joints: List[Joint], links: List[str]) -> str:
    children = set(j.child for j in joints)
    candidates = [l for l in links if l not in children]
    return candidates[0] if candidates else (joints[0].parent if joints else links[0])


FINGER_ORDER = ["index", "middle", "ring", "pinky"]


def model_joints16_to_urdf_q(j16_rad: List[float]) -> Dict[str, float]:
    if len(j16_rad) != 16:
        raise ValueError(f"Expected 16 joints, got {len(j16_rad)}")

    q: Dict[str, float] = {}
    q["right_thumb_cmc_abd"] = float(j16_rad[0])
    q["right_thumb_cmc_flex"] = float(j16_rad[1])
    q["right_thumb_mcp"] = float(j16_rad[2])
    q["right_thumb_ip"] = float(j16_rad[3])

    base = 4
    for fi, fname in enumerate(FINGER_ORDER):
        mcp_flex, pip, dip = j16_rad[base + 3 * fi: base + 3 * fi + 3]
        q[f"right_{fname}_mcp_flex"] = float(mcp_flex)
        q[f"right_{fname}_pip"] = float(pip)
        q[f"right_{fname}_dip"] = float(dip)

    return q


class HandSkeletonFromUrdf:
    """
    One-time load URDF + one-time init ActuationsToJointsModel.
    Per call: (wrist xyz + wrist rot6d + act7) -> joint points + skeleton segments (world).
    """

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.joints, self.links = load_urdf_joints(urdf_path)
        self.root_link = find_root_link(self.joints, self.links)

        self.by_parent: Dict[str, List[Joint]] = {}
        for j in self.joints:
            self.by_parent.setdefault(j.parent, []).append(j)

        self.model = ActuationsToJointsModel()

        # Cache joint order (URDF order) for consistent point output
        self.joint_names_order = [j.name for j in self.joints]

    def _compute_fk(self, q_urdf_rad: Dict[str, float], base_T_world: np.ndarray):
        link_T_world: Dict[str, np.ndarray] = {self.root_link: base_T_world.copy()}
        joint_T_world: Dict[str, np.ndarray] = {}

        stack = [self.root_link]
        while stack:
            parent_link = stack.pop()
            if parent_link not in link_T_world:
                continue
            T_world_parent = link_T_world[parent_link]

            for j in self.by_parent.get(parent_link, []):
                T_parent_joint = origin_to_T(j.origin_xyz, j.origin_rpy)
                T_world_joint = T_world_parent @ T_parent_joint
                joint_T_world[j.name] = T_world_joint

                if j.jtype in ("revolute", "continuous"):
                    angle = float(q_urdf_rad.get(j.name, 0.0))
                    Rm = axis_angle_to_R(j.axis, angle)
                    T_joint_child = make_T(Rm, np.zeros(3))
                elif j.jtype == "prismatic":
                    disp = float(q_urdf_rad.get(j.name, 0.0))
                    a = j.axis.astype(np.float64)
                    n = np.linalg.norm(a)
                    a = a / n if n > 1e-12 else a
                    T_joint_child = make_T(np.eye(3), a * disp)
                else:
                    T_joint_child = np.eye(4, dtype=np.float64)

                link_T_world[j.child] = T_world_joint @ T_joint_child
                stack.append(j.child)

        return link_T_world, joint_T_world

    def compute(
        self,
        wrist_xyz: np.ndarray,     # (3,)
        wrist_rot6d: np.ndarray,   # (6,)
        actuations7: np.ndarray,   # (7,)
        *,
        actuations_to_joints_output_is_degrees: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Returns:
          joint_points_world: (Nj,3) joint origin points in URDF joint order
          segments_world:     (Ns,2,3) segments (parent->joint and joint->child) in world
          q_urdf_rad:         dict of URDF joint angles (rad)
        """

        wrist_xyz = np.asarray(wrist_xyz, dtype=np.float64).reshape(3)
        wrist_rot6d = np.asarray(wrist_rot6d, dtype=np.float64).reshape(6)
        actuations7 = np.asarray(actuations7, dtype=np.float64).reshape(7)

        # Base pose from wrist pose
        Rw = rot6d_to_matrix(wrist_rot6d[None, :])[0]
        base_T = make_T(Rw, wrist_xyz)

        # Actuations -> 16 joints
        j16_out = self.model.hand_joints(actuations7.tolist())  # user-confirmed output is degrees
        if actuations_to_joints_output_is_degrees:
            j16_rad = [float(np.deg2rad(v)) for v in j16_out]
        else:
            j16_rad = [float(v) for v in j16_out]

        q_urdf_rad = model_joints16_to_urdf_q(j16_rad)

        link_T_world, joint_T_world = self._compute_fk(q_urdf_rad, base_T)

        # Joint points in URDF joint order
        pts = []
        for name in self.joint_names_order:
            Tj = joint_T_world.get(name)
            if Tj is None:
                pts.append([np.nan, np.nan, np.nan])
            else:
                pts.append(Tj[:3, 3])
        joint_points_world = np.asarray(pts, dtype=np.float64)

        # Skeleton segments: for each URDF joint, add (parent->joint) and (joint->child)
        segs = []
        for j in self.joints:
            if j.name not in joint_T_world:
                continue
            if j.parent not in link_T_world or j.child not in link_T_world:
                continue
            p_parent = link_T_world[j.parent][:3, 3]
            p_joint = joint_T_world[j.name][:3, 3]
            p_child = link_T_world[j.child][:3, 3]
            segs.append([p_parent, p_joint])
            segs.append([p_joint, p_child])

        segments_world = np.asarray(segs, dtype=np.float64) if len(segs) > 0 else np.zeros((0, 2, 3), dtype=np.float64)
        return joint_points_world, segments_world, q_urdf_rad