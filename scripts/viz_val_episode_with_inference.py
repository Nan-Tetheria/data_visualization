#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


# -------------------------
# Camera intrinsics (for ORIGINAL, un-rotated image coords)
# -------------------------
FX = 243.31371125
FY = 243.36591085
CX = 330.20097652
CY = 185.82838602

# Equidistant fisheye distortion
K1 = 0.05295615
K2 = -0.03939104
K3 = 0.01021067
K4 = -0.00235273

# Camera extrinsics in wrist frame
T_W_C = np.array([-0.03704, -0.05092, 0.02236], dtype=np.float64)
EULER_ZXY_DEG = np.array([-171.825 + 180, -11.079, 12.307], dtype=np.float64)

CAM_FORWARD_OFFSET_M = 0.1
P_C_FORWARD = np.array([0.0, 0.0, CAM_FORWARD_OFFSET_M], dtype=np.float64)  # in camera frame


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


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa, ca]], dtype=np.float64)


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0, sa],
                     [0, 1, 0],
                     [-sa, 0, ca]], dtype=np.float64)


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0],
                     [sa, ca, 0],
                     [0, 0, 1]], dtype=np.float64)


def euler_zxy_custom_deg_to_rot(z_deg, x_deg, y_deg):
    """
    Intrinsic rotation order: Z -> X -> Y
    Column-vector convention: R = Ry(y) @ Rx(x) @ Rz(z)
    """
    z = np.deg2rad(z_deg)
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)
    return rot_y(y) @ rot_x(x) @ rot_z(z)


def rotate_image_180(img):
    """Rotate image by 180 degrees."""
    if img.ndim == 2:
        return img[::-1, ::-1]
    return img[::-1, ::-1, :]


def project_points_equidistant_full(P_cam, fx, fy, cx, cy, k1, k2, k3, k4):
    """Project 3D camera-frame points to pixel coordinates using equidistant fisheye model."""
    uv = np.full((P_cam.shape[0], 2), np.nan, dtype=np.float64)

    z = P_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return uv

    P = P_cam[valid]
    x = P[:, 0] / P[:, 2]
    y = P[:, 1] / P[:, 2]
    r = np.sqrt(x*x + y*y)

    theta = np.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    theta_d = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)

    scale = np.ones_like(r)
    mask = r > 1e-12
    scale[mask] = theta_d[mask] / r[mask]

    xd = scale * x
    yd = scale * y

    u = fx * xd + cx
    v = fy * yd + cy

    uv[valid] = np.stack([u, v], axis=1)
    return uv


def colors_pred_future_gradient(n_pts):
    """Pred future: green -> red (near -> far). alpha=1."""
    if n_pts <= 0:
        return np.empty((0, 4), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_pts)
    rgba = np.zeros((n_pts, 4), dtype=np.float64)
    rgba[:, 0] = t
    rgba[:, 1] = 1.0 - t
    rgba[:, 3] = 1.0
    return rgba


def colors_gt_future_gradient(n_pts):
    """GT future: black -> white (near -> far). alpha=1."""
    if n_pts <= 0:
        return np.empty((0, 4), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_pts)
    rgba = np.zeros((n_pts, 4), dtype=np.float64)
    rgba[:, 0] = t
    rgba[:, 1] = t
    rgba[:, 2] = t
    rgba[:, 3] = 1.0
    return rgba


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
    """
    Fixed RGB: X=red, Y=green, Z=blue.
    R is column-major (columns are basis vectors in world frame).
    """
    p = np.asarray(p).reshape(3)
    ex = R[:, 0] * axis_len
    ey = R[:, 1] * axis_len
    ez = R[:, 2] * axis_len

    lx, = ax.plot([p[0], p[0] + ex[0]], [p[1], p[1] + ex[1]], [p[2], p[2] + ex[2]],
                  color='r', linewidth=lw)
    ly, = ax.plot([p[0], p[0] + ey[0]], [p[1], p[1] + ey[1]], [p[2], p[2] + ey[2]],
                  color='g', linewidth=lw)
    lz, = ax.plot([p[0], p[0] + ez[0]], [p[1], p[1] + ez[1]], [p[2], p[2] + ez[2]],
                  color='b', linewidth=lw)
    return (lx, ly, lz)


def segments_from_points(uv: np.ndarray) -> np.ndarray:
    """
    uv: (N,2)
    return: (N-1,2,2) segments connecting adjacent points in current uv order.
    """
    if uv is None or uv.shape[0] < 2:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.stack([uv[:-1], uv[1:]], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_npz", type=str, default="/home/nan/datasets/deploy_data/val_data_episode_0000.npz")
    ap.add_argument("--pred_npz", type=str, default="/home/nan/datasets/deploy_data/model_outputs.npz")
    ap.add_argument("--pred_key", type=str, default="action.arm")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--axis_len", type=float, default=0.03)
    ap.add_argument("--pred_pose_step", type=int, default=4)
    ap.add_argument("--save", type=str, default="", help="If set, save animation to this path (mp4). Requires ffmpeg.")
    args = ap.parse_args()

    # --- Load val episode ---
    val = np.load(args.val_npz, allow_pickle=True)
    imgs = val["observation.images.camera_wrist_right"]   # (T,3,H,W) float32
    obs_state = val["observation.state"]                  # (T,16)

    if obs_state.ndim != 2 or obs_state.shape[1] < 9:
        raise ValueError(f"observation.state expected (T,>=9), got {obs_state.shape}")
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        raise ValueError(f"images expected (T,3,H,W), got {imgs.shape}")

    T = obs_state.shape[0]
    gt_pose9 = obs_state[:, :9]
    t_w = gt_pose9[:, :3].astype(np.float64)                   # wrist position (world-ish)
    R_w = rot6d_to_matrix(gt_pose9[:, 3:9]).astype(np.float64) # wrist rotation (world-ish)

    # --- Camera extrinsics in wrist ---
    R_w_c = euler_zxy_custom_deg_to_rot(EULER_ZXY_DEG[0], EULER_ZXY_DEG[1], EULER_ZXY_DEG[2])

    # --- Camera center and camera +0.1m (world-ish) from observation wrist pose ---
    t_c  = t_w + (R_w @ T_W_C.reshape(3, 1)).reshape(T, 3)
    R_c  = np.einsum("tij,jk->tik", R_w, R_w_c)
    t_cf = t_c + (R_c @ P_C_FORWARD.reshape(3, 1)).reshape(T, 3)

    # --- Load predictions ---
    pred = np.load(args.pred_npz, allow_pickle=True)
    if args.pred_key not in pred:
        raise KeyError(f"Pred key '{args.pred_key}' not found. Available: {list(pred.keys())}")

    pred_arm = pred[args.pred_key]  # (T,H,9)
    if pred_arm.ndim != 3 or pred_arm.shape[-1] != 9:
        raise ValueError(f"{args.pred_key} expected (T,H,9), got {pred_arm.shape}")

    Tp, H, _ = pred_arm.shape
    if Tp != T:
        raise ValueError(f"T mismatch: val T={T}, pred T={Tp}")

    pred_tw = pred_arm[:, :, :3].astype(np.float64)                 # (T,H,3) world-ish
    pred_Rw = rot6d_to_matrix(pred_arm[:, :, 3:9]).astype(np.float64)  # (T,H,3,3) world-ish

    # -------------------------
    # Figure setup
    # -------------------------
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axim = fig.add_subplot(1, 2, 2)

    # --- Left: trajectories (WORLD frame) ---
    ax3d.set_title("Wrist / Camera Center / Camera+0.1m (world frame)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    # Draw world XYZ triad at origin
    world_triad = draw_triad(ax3d, np.array([0.0, 0.0, 0.0], dtype=np.float64),
                             np.eye(3, dtype=np.float64),
                             axis_len=args.axis_len * 1.5, lw=2.5)

    wrist_line, = ax3d.plot(t_w[:, 0], t_w[:, 1], t_w[:, 2], lw=2, color="lightgray", label="wrist (obs)")
    cam_center_line, = ax3d.plot(t_c[:, 0], t_c[:, 1], t_c[:, 2], lw=2, color="black", label="camera center (obs)")
    cam_fwd_line, = ax3d.plot(t_cf[:, 0], t_cf[:, 1], t_cf[:, 2], lw=2, color="purple", label="camera +0.1m (obs)")
    ax3d.legend(loc="upper left")

    ax3d.scatter([t_w[0, 0]], [t_w[0, 1]], [t_w[0, 2]], s=40)
    _triad0 = draw_triad(ax3d, t_w[0], R_w[0], axis_len=args.axis_len, lw=2.0)

    cur_pt = ax3d.scatter([t_w[0, 0]], [t_w[0, 1]], [t_w[0, 2]], s=60)
    cur_triad = list(draw_triad(ax3d, t_w[0], R_w[0], axis_len=args.axis_len, lw=2.0))

    # Initial predicted camera+0.1 future trajectory (world frame)
    t_c_fut0 = pred_tw[0] + (pred_Rw[0] @ T_W_C.reshape(3, 1)).reshape(H, 3)
    R_c_fut0 = np.einsum("hij,jk->hik", pred_Rw[0], R_w_c)
    t_cf_fut0 = t_c_fut0 + (R_c_fut0 @ P_C_FORWARD.reshape(3, 1)).reshape(H, 3)

    future_line, = ax3d.plot(t_cf_fut0[:, 0], t_cf_fut0[:, 1], t_cf_fut0[:, 2], linewidth=2.0)
    future_start = ax3d.scatter([t_cf_fut0[0, 0]], [t_cf_fut0[0, 1]], [t_cf_fut0[0, 2]], s=25)

    pred_triads = []
    step = max(1, int(args.pred_pose_step))
    pred_axis_len = args.axis_len * 0.7
    pred_lw = 1.5
    for h in range(0, H, step):
        pred_triads.extend(draw_triad(ax3d, t_cf_fut0[h], R_c_fut0[h],
                                      axis_len=pred_axis_len, lw=pred_lw))

    # 3D bounds
    all_pts = np.vstack([t_w, t_c, t_cf, t_cf_fut0.reshape(-1, 3), np.zeros((1, 3), dtype=np.float64)])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2
    span = (maxs - mins).max()
    if span < 1e-6:
        span = 1.0
    half = span * 0.6
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect([1, 1, 1])
    set_axes_equal(ax3d)

    # --- Right: image + projected points + line connections ---
    axim.set_title("2D: projected camera+0.1 future (Pred: green->red, GT: black->white)")
    axim.axis("off")

    img_received0 = np.transpose(imgs[0], (1, 2, 0))
    im_artist = axim.imshow(img_received0, vmin=0.0, vmax=1.0)

    pred_sc = axim.scatter([], [], s=80, linewidths=0.0)     # Pred points
    gt_sc = axim.scatter([], [], s=80, linewidths=0.0)       # GT points

    pred_lc = LineCollection([], linewidths=2.0)
    gt_lc = LineCollection([], linewidths=2.0)
    axim.add_collection(pred_lc)
    axim.add_collection(gt_lc)

    # Principal point cross in DISPLAY coords
    h_disp0, w_disp0 = img_received0.shape[:2]
    cx_disp0 = (w_disp0 - 1) - CX
    cy_disp0 = (h_disp0 - 1) - CY
    cam_cross = axim.scatter([cx_disp0], [cy_disp0], s=200, marker="x", linewidths=2)

    current_frame = {"i": 0}
    paused = {"value": False}

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
            if paused["value"]:
                ani.event_source.stop()
                fig.suptitle(f"PAUSED - frame {current_frame['i']+1}/{T}")
            else:
                ani.event_source.start()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(i):
        nonlocal cur_triad, pred_triads
        current_frame["i"] = i

        # -------------------------
        # Update 3D current wrist pose triad (world frame)
        # -------------------------
        p = t_w[i]
        cur_pt._offsets3d = ([p[0]], [p[1]], [p[2]])

        for line in cur_triad:
            line.remove()
        cur_triad.clear()
        cur_triad.extend(draw_triad(ax3d, p, R_w[i], axis_len=args.axis_len, lw=2.0))

        # -------------------------
        # Predicted future camera+0.1 trajectory (world frame)
        # -------------------------
        t_c_fut = pred_tw[i] + (pred_Rw[i] @ T_W_C.reshape(3, 1)).reshape(H, 3)          # (H,3)
        R_c_fut = np.einsum("hij,jk->hik", pred_Rw[i], R_w_c)                            # (H,3,3)
        t_cf_fut = t_c_fut + (R_c_fut @ P_C_FORWARD.reshape(3, 1)).reshape(H, 3)         # (H,3)

        future_line.set_data(t_cf_fut[:, 0], t_cf_fut[:, 1])
        future_line.set_3d_properties(t_cf_fut[:, 2])
        future_start._offsets3d = ([t_cf_fut[0, 0]], [t_cf_fut[0, 1]], [t_cf_fut[0, 2]])

        for line in pred_triads:
            line.remove()
        pred_triads.clear()
        for h in range(0, H, step):
            pred_triads.extend(draw_triad(ax3d, t_cf_fut[h], R_c_fut[h],
                                          axis_len=pred_axis_len, lw=pred_lw))

        # -------------------------
        # Update 2D image (received) + ORIGINAL via rotate180
        # -------------------------
        img_received = np.transpose(imgs[i], (1, 2, 0))
        im_artist.set_data(img_received)
        h_disp, w_disp = img_received.shape[:2]

        cx_disp = (w_disp - 1) - CX
        cy_disp = (h_disp - 1) - CY
        cam_cross.set_offsets(np.array([[cx_disp, cy_disp]], dtype=np.float64))

        img_orig = rotate_image_180(img_received)
        h0, w0 = img_orig.shape[:2]

        # Current camera pose (world frame) for projection frame
        tc_i = t_c[i]
        Rc_i = R_c[i]

        # -------------------------
        # Project Pred camera+0.1 points into current camera frame
        # -------------------------
        dt_pred = t_cf_fut - tc_i.reshape(1, 3)      # (H,3) in world
        P_cam_pred = (Rc_i.T @ dt_pred.T).T          # (H,3) in current camera frame
        uv_pred = project_points_equidistant_full(P_cam_pred, FX, FY, CX, CY, K1, K2, K3, K4)

        inb0_pred = (uv_pred[:, 0] >= 0) & (uv_pred[:, 0] < w0) & (uv_pred[:, 1] >= 0) & (uv_pred[:, 1] < h0)
        uv0_pred = uv_pred[inb0_pred]

        if uv0_pred.shape[0] > 0:
            uv_disp_pred = np.empty_like(uv0_pred)
            uv_disp_pred[:, 0] = (w_disp - 1) - uv0_pred[:, 0]
            uv_disp_pred[:, 1] = (h_disp - 1) - uv0_pred[:, 1]
            pred_sc.set_offsets(uv_disp_pred)

            rgba_pred = colors_pred_future_gradient(uv_disp_pred.shape[0])
            pred_sc.set_facecolors(rgba_pred)
            pred_sc.set_edgecolors(rgba_pred)

            seg_pred = segments_from_points(uv_disp_pred)
            pred_lc.set_segments(seg_pred)
            pred_lc.set_colors(rgba_pred[:-1] if seg_pred.shape[0] > 0 else [])
        else:
            pred_sc.set_offsets(np.empty((0, 2)))
            pred_lc.set_segments(np.empty((0, 2, 2)))
            pred_lc.set_colors([])

        # -------------------------
        # Project GT/observation camera+0.1 points (i..i+H-1) into current camera frame
        # -------------------------
        j1 = min(T, i + H)  # exclusive
        if i < j1:
            idx = np.arange(i, j1, dtype=np.int64)  # near -> far
            dt_gt = t_cf[idx] - tc_i.reshape(1, 3)  # world
            P_cam_gt = (Rc_i.T @ dt_gt.T).T
            uv_gt = project_points_equidistant_full(P_cam_gt, FX, FY, CX, CY, K1, K2, K3, K4)

            inb0_gt = (uv_gt[:, 0] >= 0) & (uv_gt[:, 0] < w0) & (uv_gt[:, 1] >= 0) & (uv_gt[:, 1] < h0)
            uv0_gt = uv_gt[inb0_gt]

            if uv0_gt.shape[0] > 0:
                uv_disp_gt = np.empty_like(uv0_gt)
                uv_disp_gt[:, 0] = (w_disp - 1) - uv0_gt[:, 0]
                uv_disp_gt[:, 1] = (h_disp - 1) - uv0_gt[:, 1]
                gt_sc.set_offsets(uv_disp_gt)

                rgba_gt = colors_gt_future_gradient(uv_disp_gt.shape[0])
                gt_sc.set_facecolors(rgba_gt)
                gt_sc.set_edgecolors(rgba_gt)

                seg_gt = segments_from_points(uv_disp_gt)
                gt_lc.set_segments(seg_gt)
                gt_lc.set_colors(rgba_gt[:-1] if seg_gt.shape[0] > 0 else [])
            else:
                gt_sc.set_offsets(np.empty((0, 2)))
                gt_lc.set_segments(np.empty((0, 2, 2)))
                gt_lc.set_colors([])
        else:
            gt_sc.set_offsets(np.empty((0, 2)))
            gt_lc.set_segments(np.empty((0, 2, 2)))
            gt_lc.set_colors([])

        fig.suptitle(f"frame {i+1}/{T}")

        return ([wrist_line, cam_center_line, cam_fwd_line,
                 cur_pt, future_line, future_start,
                 im_artist, cam_cross,
                 pred_sc, gt_sc,
                 pred_lc, gt_lc]
                + list(world_triad) + cur_triad + pred_triads)

    interval_ms = 1000.0 / max(args.fps, 1e-6)
    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)

    plt.tight_layout()
    if args.save:
        ani.save(args.save, fps=args.fps)
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()