# utils/rotation.py
import numpy as np


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    orig_shape = rot6d.shape[:-1]
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected last dim=6, got {rot6d.shape}")

    x = rot6d.reshape(-1, 2, 3)
    r1 = x[:, 0, :]
    r2 = x[:, 1, :]

    r1 = r1 / (np.linalg.norm(r1, axis=1, keepdims=True) + 1e-12)
    r2 = r2 - (np.sum(r1 * r2, axis=1, keepdims=True)) * r1
    r2 = r2 / (np.linalg.norm(r2, axis=1, keepdims=True) + 1e-12)
    r3 = np.cross(r1, r2)

    R = np.stack([r1, r2, r3], axis=1)   # rows
    R = np.transpose(R, (0, 2, 1))       # columns-as-basis
    return R.reshape(*orig_shape, 3, 3)


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected R (...,3,3), got {R.shape}")
    c0 = R[..., :, 0]
    c1 = R[..., :, 1]
    return np.concatenate([c0, c1], axis=-1)


def rot_x(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa, ca]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa],
                     [0, 1, 0],
                     [-sa, 0, ca]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa, ca, 0],
                     [0, 0, 1]], dtype=np.float64)


def euler_zxy_custom_deg_to_rot(z_deg: float, x_deg: float, y_deg: float) -> np.ndarray:
    # Intrinsic rotation order: Z -> X -> Y
    # Column-vector convention: R = Ry(y) @ Rx(x) @ Rz(z)
    z = np.deg2rad(z_deg)
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)
    return rot_y(y) @ rot_x(x) @ rot_z(z)