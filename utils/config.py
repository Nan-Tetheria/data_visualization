# utils/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class IOConfig:
    val_npz: str
    pred_npz: str
    pred_key: str
    pred_hand_key: str
    save_path: str = ""


@dataclass
class RuntimeConfig:
    fps: float = 20.0


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class CameraDistEquidistant:
    k1: float
    k2: float
    k3: float
    k4: float


@dataclass
class CameraConfig:
    intrinsics: CameraIntrinsics
    distortion_equidistant: CameraDistEquidistant
    T_w_c: List[float]
    euler_zxy_deg: List[float]
    forward_offset_m: float = 0.1


@dataclass
class HandConfig:
    urdf_path: str
    actuations_to_joints_output_is_degrees: bool = True
    horizon: int = 16
    step: int = 4


@dataclass
class FigConfig:
    size: List[float]
    axis_len: float = 0.03


@dataclass
class VizConfig:
    fig: FigConfig

    show_obs_traj_3d: bool = True
    show_gt_proj_2d: bool = True

    show_pred_cam_fwd_traj_3d: bool = False
    pred_pose_step: int = 4

    show_pred_proj_2d: bool = False
    show_pred_line_2d: bool = True
    show_gt_line_2d: bool = True

    show_hand_current_3d: bool = False
    show_hand_future_3d: bool = False
    hand: Optional[HandConfig] = None

    show_hand_current_2d: bool = False
    show_hand_future_2d: bool = False


@dataclass
class AppConfig:
    io: IOConfig
    runtime: RuntimeConfig
    camera: CameraConfig
    viz: VizConfig


def _req(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required config key: {key}")
    return d[key]


def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    io = raw["io"]
    runtime = raw.get("runtime", {})
    camera = raw["camera"]
    viz = raw["viz"]

    intr = camera["intrinsics"]
    dist = camera["distortion_equidistant"]

    hand_cfg = None
    if viz.get("hand") is not None:
        h = viz["hand"]
        hand_cfg = HandConfig(
            urdf_path=_req(h, "urdf_path"),
            actuations_to_joints_output_is_degrees=h.get("actuations_to_joints_output_is_degrees", True),
            horizon=int(h.get("horizon", 16)),
            step=int(h.get("step", 4)),
        )

    cfg = AppConfig(
        io=IOConfig(
            val_npz=_req(io, "val_npz"),
            pred_npz=_req(io, "pred_npz"),
            pred_key=_req(io, "pred_key"),
            pred_hand_key=io.get("pred_hand_key", "action.hand"),
            save_path=io.get("save_path", ""),
        ),
        runtime=RuntimeConfig(fps=float(runtime.get("fps", 20.0))),
        camera=CameraConfig(
            intrinsics=CameraIntrinsics(
                fx=float(intr["fx"]), fy=float(intr["fy"]), cx=float(intr["cx"]), cy=float(intr["cy"])
            ),
            distortion_equidistant=CameraDistEquidistant(
                k1=float(dist["k1"]), k2=float(dist["k2"]), k3=float(dist["k3"]), k4=float(dist["k4"])
            ),
            T_w_c=list(camera["T_w_c"]),
            euler_zxy_deg=list(camera["euler_zxy_deg"]),
            forward_offset_m=float(camera.get("forward_offset_m", 0.1)),
        ),
        viz=VizConfig(
            fig=FigConfig(size=list(viz["fig"]["size"]), axis_len=float(viz["fig"].get("axis_len", 0.03))),
            show_obs_traj_3d=bool(viz.get("show_obs_traj_3d", True)),
            show_gt_proj_2d=bool(viz.get("show_gt_proj_2d", True)),

            show_pred_cam_fwd_traj_3d=bool(viz.get("show_pred_cam_fwd_traj_3d", False)),
            pred_pose_step=int(viz.get("pred_pose_step", 4)),

            show_pred_proj_2d=bool(viz.get("show_pred_proj_2d", False)),
            show_pred_line_2d=bool(viz.get("show_pred_line_2d", True)),
            show_gt_line_2d=bool(viz.get("show_gt_line_2d", True)),

            show_hand_current_3d=bool(viz.get("show_hand_current_3d", False)),
            show_hand_future_3d=bool(viz.get("show_hand_future_3d", False)),
            hand=hand_cfg,

            show_hand_current_2d=bool(viz.get("show_hand_current_2d", False)),
            show_hand_future_2d=bool(viz.get("show_hand_future_2d", False)),
        ),
    )
    return cfg