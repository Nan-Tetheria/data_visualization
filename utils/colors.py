# utils/colors.py
import numpy as np


def colors_pred_future_gradient(n: int) -> np.ndarray:
    # green -> red
    if n <= 0:
        return np.empty((0, 4), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    rgba = np.zeros((n, 4), dtype=np.float64)
    rgba[:, 0] = t
    rgba[:, 1] = 1.0 - t
    rgba[:, 3] = 1.0
    return rgba


def colors_gt_future_gradient(n: int) -> np.ndarray:
    # black -> white
    if n <= 0:
        return np.empty((0, 4), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n)
    rgba = np.zeros((n, 4), dtype=np.float64)
    rgba[:, 0] = t
    rgba[:, 1] = t
    rgba[:, 2] = t
    rgba[:, 3] = 1.0
    return rgba