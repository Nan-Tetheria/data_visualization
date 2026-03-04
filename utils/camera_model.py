# utils/camera_model.py
import numpy as np


def rotate_image_180(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img[::-1, ::-1]
    return img[::-1, ::-1, :]


def project_points_equidistant_full(
    P_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    k1: float, k2: float, k3: float, k4: float
) -> np.ndarray:
    # Project 3D camera-frame points (Nx3) to pixels (Nx2) with equidistant fisheye.
    P_cam = np.asarray(P_cam, dtype=np.float64).reshape(-1, 3)
    uv = np.full((P_cam.shape[0], 2), np.nan, dtype=np.float64)

    z = P_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return uv

    P = P_cam[valid]
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    r = np.sqrt(x * x + y * y)

    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)

    scale = np.ones_like(r)
    mask = r > 1e-12
    scale[mask] = theta_d[mask] / r[mask]

    xd = scale * x
    yd = scale * y

    u = fx * xd + cx
    v = fy * yd + cy
    uv[valid] = np.stack([u, v], axis=1)
    return uv