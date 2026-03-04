# utils/projection.py
import numpy as np
from utils.camera_model import project_points_equidistant_full


def world_to_camera_points(P_w: np.ndarray, tc_w: np.ndarray, Rc_w: np.ndarray) -> np.ndarray:
    # P_cam = Rc_w^T * (P_w - tc_w)
    P_w = np.asarray(P_w, dtype=np.float64).reshape(-1, 3)
    dt = P_w - np.asarray(tc_w, dtype=np.float64).reshape(1, 3)
    return (np.asarray(Rc_w, dtype=np.float64).T @ dt.T).T


def original_to_display_uv(uv0: np.ndarray, w_disp: int, h_disp: int) -> np.ndarray:
    # Display image is rotated 180deg relative to original projection coords.
    uv0 = np.asarray(uv0, dtype=np.float64).reshape(-1, 2)
    uv = np.empty_like(uv0)
    uv[:, 0] = (w_disp - 1) - uv0[:, 0]
    uv[:, 1] = (h_disp - 1) - uv0[:, 1]
    return uv


def project_points_world_to_display(
    P_w: np.ndarray,
    tc_w: np.ndarray,
    Rc_w: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, k3: float, k4: float,
    w_disp: int, h_disp: int,
    w0: int, h0: int,
):
    # Returns (uv_disp, keep_idx in original P_w).
    P_cam = world_to_camera_points(P_w, tc_w, Rc_w)
    uv0 = project_points_equidistant_full(P_cam, fx, fy, cx, cy, k1, k2, k3, k4)

    inb0 = (uv0[:, 0] >= 0) & (uv0[:, 0] < w0) & (uv0[:, 1] >= 0) & (uv0[:, 1] < h0)
    keep = np.where(inb0)[0]
    if keep.size == 0:
        return np.empty((0, 2), dtype=np.float64), keep

    uv0_keep = uv0[keep]
    uv_disp = original_to_display_uv(uv0_keep, w_disp=w_disp, h_disp=h_disp)
    return uv_disp, keep


def project_segments_world_to_display(
    segs_w: np.ndarray,  # (S,2,3)
    tc_w: np.ndarray,
    Rc_w: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, k3: float, k4: float,
    w_disp: int, h_disp: int,
    w0: int, h0: int,
) -> np.ndarray:
    segs_w = np.asarray(segs_w, dtype=np.float64)
    if segs_w.ndim != 3 or segs_w.shape[1:] != (2, 3) or segs_w.shape[0] == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    P0 = segs_w[:, 0, :]
    P1 = segs_w[:, 1, :]

    uv0, keep0 = project_points_world_to_display(
        P0, tc_w, Rc_w, fx, fy, cx, cy, k1, k2, k3, k4, w_disp, h_disp, w0, h0
    )
    uv1, keep1 = project_points_world_to_display(
        P1, tc_w, Rc_w, fx, fy, cx, cy, k1, k2, k3, k4, w_disp, h_disp, w0, h0
    )

    if keep0.size == 0 or keep1.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    keep = np.array(sorted(set(keep0.tolist()).intersection(set(keep1.tolist()))), dtype=np.int64)
    if keep.size == 0:
        return np.empty((0, 2, 2), dtype=np.float64)

    map0 = {idx: j for j, idx in enumerate(keep0.tolist())}
    map1 = {idx: j for j, idx in enumerate(keep1.tolist())}

    out = np.empty((keep.size, 2, 2), dtype=np.float64)
    for k, idx in enumerate(keep):
        out[k, 0, :] = uv0[map0[int(idx)]]
        out[k, 1, :] = uv1[map1[int(idx)]]
    return out