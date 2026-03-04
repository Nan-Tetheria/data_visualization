# utils/plot3d.py
import numpy as np


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)

    r = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - r, x_mid + r])
    ax.set_ylim3d([y_mid - r, y_mid + r])
    ax.set_zlim3d([z_mid - r, z_mid + r])


def draw_triad(ax, p, R, axis_len=0.03, lw=2.0):
    # Fixed RGB: X=red, Y=green, Z=blue. R columns are basis vectors in world.
    p = np.asarray(p).reshape(3)
    ex = R[:, 0] * axis_len
    ey = R[:, 1] * axis_len
    ez = R[:, 2] * axis_len

    lx, = ax.plot([p[0], p[0] + ex[0]], [p[1], p[1] + ex[1]], [p[2], p[2] + ex[2]],
                  color="r", linewidth=lw)
    ly, = ax.plot([p[0], p[0] + ey[0]], [p[1], p[1] + ey[1]], [p[2], p[2] + ey[2]],
                  color="g", linewidth=lw)
    lz, = ax.plot([p[0], p[0] + ez[0]], [p[1], p[1] + ez[1]], [p[2], p[2] + ez[2]],
                  color="b", linewidth=lw)
    return (lx, ly, lz)