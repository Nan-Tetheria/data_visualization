# utils/geometry.py
import numpy as np


def segments_from_points(uv: np.ndarray) -> np.ndarray:
    # uv: (N,2) -> (N-1,2,2)
    if uv is None:
        return np.empty((0, 2, 2), dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    if uv.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack([uv[:-1], uv[1:]], axis=1)