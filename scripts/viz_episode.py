#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.config import load_config
from utils.rotation import rot6d_to_matrix, matrix_to_rot6d, euler_zxy_custom_deg_to_rot
from utils.camera_model import rotate_image_180, project_points_equidistant_full
from utils.colors import colors_pred_future_gradient, colors_gt_future_gradient
from utils.geometry import segments_from_points
from utils.projection import project_points_world_to_display, project_segments_world_to_display
from utils.plot3d import draw_triad, set_axes_equal

from utils.hand_skeleton import HandSkeletonFromUrdf


def compute_obs_poses(obs_state: np.ndarray):
    # obs_state: (T,>=9), first 9 are wrist pose9: xyz + rot6d
    gt_pose9 = obs_state[:, :9]
    t_w = gt_pose9[:, :3].astype(np.float64)
    R_w = rot6d_to_matrix(gt_pose9[:, 3:9]).astype(np.float64)
    return t_w, R_w


def compute_camera_from_wrist(
    t_w: np.ndarray,
    R_w: np.ndarray,
    T_w_c: np.ndarray,
    R_w_c: np.ndarray,
    P_c_forward: np.ndarray,
):
    # camera center in world
    t_c = t_w + (R_w @ T_w_c.reshape(3, 1)).reshape(t_w.shape[0], 3)
    R_c = np.einsum("tij,jk->tik", R_w, R_w_c)
    t_cf = t_c + (R_c @ P_c_forward.reshape(3, 1)).reshape(t_w.shape[0], 3)
    return t_c, R_c, t_cf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/viz_default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # --- Load val ---
    val = np.load(cfg.io.val_npz, allow_pickle=True)
    imgs = val["observation.images.camera_wrist_right"]   # (T,3,H,W)
    obs_state = val["observation.state"]                  # (T,16)

    if obs_state.ndim != 2 or obs_state.shape[1] < 9:
        raise ValueError(f"observation.state expected (T,>=9), got {obs_state.shape}")
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        raise ValueError(f"images expected (T,3,H,W), got {imgs.shape}")

    T = obs_state.shape[0]
    t_w, R_w = compute_obs_poses(obs_state)

    # --- Camera params ---
    intr = cfg.camera.intrinsics
    dist = cfg.camera.distortion_equidistant
    T_w_c = np.array(cfg.camera.T_w_c, dtype=np.float64)
    eul = cfg.camera.euler_zxy_deg
    R_w_c = euler_zxy_custom_deg_to_rot(eul[0], eul[1], eul[2])

    P_c_forward = np.array([0.0, 0.0, float(cfg.camera.forward_offset_m)], dtype=np.float64)
    t_c, R_c, t_cf = compute_camera_from_wrist(t_w, R_w, T_w_c, R_w_c, P_c_forward)

    # --- Load pred ---
    pred = np.load(cfg.io.pred_npz, allow_pickle=True)
    if cfg.io.pred_key not in pred:
        raise KeyError(f"Pred key '{cfg.io.pred_key}' not found. Available: {list(pred.keys())}")

    pred_arm = pred[cfg.io.pred_key]  # (T,H,9)
    if pred_arm.ndim != 3 or pred_arm.shape[-1] != 9:
        raise ValueError(f"{cfg.io.pred_key} expected (T,H,9), got {pred_arm.shape}")

    Tp, H, _ = pred_arm.shape
    if Tp != T:
        raise ValueError(f"T mismatch: val T={T}, pred T={Tp}")

    pred_tw = pred_arm[:, :, :3].astype(np.float64)
    pred_Rw = rot6d_to_matrix(pred_arm[:, :, 3:9]).astype(np.float64)

    pred_hand = None
    if cfg.io.pred_hand_key in pred:
        pred_hand = pred[cfg.io.pred_hand_key]
        if pred_hand.ndim != 3 or pred_hand.shape != (T, H, 7):
            raise ValueError(f"{cfg.io.pred_hand_key} expected (T,H,7), got {pred_hand.shape}")

    # --- Figure ---
    fig = plt.figure(figsize=tuple(cfg.viz.fig.size))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axim = fig.add_subplot(1, 2, 2)

    axis_len = float(cfg.viz.fig.axis_len)

    # -------------------------
    # 3D baseline: trajectories from obs
    # -------------------------
    ax3d.set_title("3D: wrist / camera / camera+fwd (obs) (+ optional preds)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    world_triad = draw_triad(ax3d, np.zeros(3), np.eye(3), axis_len=axis_len * 1.5, lw=2.5)

    wrist_line = cam_center_line = cam_fwd_line = None
    if cfg.viz.show_obs_traj_3d:
        wrist_line, = ax3d.plot(t_w[:, 0], t_w[:, 1], t_w[:, 2], lw=2, color="lightgray", label="wrist (obs)")
        cam_center_line, = ax3d.plot(t_c[:, 0], t_c[:, 1], t_c[:, 2], lw=2, color="black", label="camera center (obs)")
        cam_fwd_line, = ax3d.plot(t_cf[:, 0], t_cf[:, 1], t_cf[:, 2], lw=2, color="purple", label="camera+fwd (obs)")

    ax3d.legend(loc="upper left")

    cur_pt = ax3d.scatter([t_w[0, 0]], [t_w[0, 1]], [t_w[0, 2]], s=60)
    cur_triad = list(draw_triad(ax3d, t_w[0], R_w[0], axis_len=axis_len, lw=2.0))

    cam_cur_pt = ax3d.scatter([t_c[0, 0]], [t_c[0, 1]], [t_c[0, 2]], s=50)
    cam_cur_triad = list(draw_triad(ax3d, t_c[0], R_c[0], axis_len=axis_len * 0.8, lw=2.0))

    camf_cur_pt = ax3d.scatter([t_cf[0, 0]], [t_cf[0, 1]], [t_cf[0, 2]], s=50)
    camf_cur_triad = list(draw_triad(ax3d, t_cf[0], R_c[0], axis_len=axis_len * 0.8, lw=2.0))



    # --- Fixed "frame-0" wrist pose marker & triad ---
    start_pt = ax3d.scatter([t_w[0, 0]], [t_w[0, 1]], [t_w[0, 2]], s=40)
    start_triad = list(draw_triad(ax3d, t_w[0], R_w[0], axis_len=axis_len, lw=2.0))

    # -------------------------
    # Optional: pred camera+fwd in 3D
    # -------------------------
    future_line = None
    future_start = None
    pred_triads = []

    if cfg.viz.show_pred_cam_fwd_traj_3d:
        # init from frame 0
        t_c_fut0 = pred_tw[0] + (pred_Rw[0] @ T_w_c.reshape(3, 1)).reshape(H, 3)
        R_c_fut0 = np.einsum("hij,jk->hik", pred_Rw[0], R_w_c)
        t_cf_fut0 = t_c_fut0 + (R_c_fut0 @ P_c_forward.reshape(3, 1)).reshape(H, 3)

        future_line, = ax3d.plot(t_cf_fut0[:, 0], t_cf_fut0[:, 1], t_cf_fut0[:, 2], linewidth=2.0)
        future_start = ax3d.scatter([t_cf_fut0[0, 0]], [t_cf_fut0[0, 1]], [t_cf_fut0[0, 2]], s=25)

        step = max(1, int(cfg.viz.pred_pose_step))
        pred_axis_len = axis_len * 0.7
        pred_lw = 1.5
        for h in range(0, H, step):
            pred_triads.extend(draw_triad(ax3d, t_cf_fut0[h], R_c_fut0[h], axis_len=pred_axis_len, lw=pred_lw))

    # -------------------------
    # Optional: Hand skeleton (3D current & future)
    # -------------------------
    hand_fk = None
    hand_skel_lc = None
    pred_hand_skel_lc = None
    hand_joint_sc = None

    if (cfg.viz.show_hand_current_3d or cfg.viz.show_hand_future_3d or
        cfg.viz.show_hand_current_2d or cfg.viz.show_hand_future_2d):
        if cfg.viz.hand is None:
            raise ValueError("viz.hand must be provided when hand visualization is enabled.")
        hand_fk = HandSkeletonFromUrdf(cfg.viz.hand.urdf_path)

    if cfg.viz.show_hand_current_3d:
        hand_skel_lc = Line3DCollection(np.zeros((1, 2, 3)), linewidths=2.0)
        ax3d.add_collection3d(hand_skel_lc)
        hand_joint_sc = ax3d.scatter([], [], [], s=10)

    if cfg.viz.show_hand_future_3d:
        pred_hand_skel_lc = Line3DCollection(np.zeros((1, 2, 3)), linewidths=1.5)
        ax3d.add_collection3d(pred_hand_skel_lc)

    # -------------------------
    # 3D bounds (include preds only if enabled)
    # -------------------------
    pts = [t_w, t_c, t_cf, np.zeros((1, 3), dtype=np.float64)]
    if cfg.viz.show_pred_cam_fwd_traj_3d:
        pts.append(t_cf_fut0.reshape(-1, 3))
    all_pts = np.vstack(pts)

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2
    span = max((maxs - mins).max(), 1e-6)
    half = span * 0.6
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect([1, 1, 1])
    set_axes_equal(ax3d)

    # -------------------------
    # 2D baseline: image + GT projection (black->white)
    # -------------------------
    axim.set_title("2D: image (+ optional proj overlays)")
    axim.axis("off")

    img0 = np.transpose(imgs[0], (1, 2, 0))
    im_artist = axim.imshow(img0, vmin=0.0, vmax=1.0)

    pred_sc = axim.scatter([], [], s=80, linewidths=0.0) if cfg.viz.show_pred_proj_2d else None
    gt_sc = axim.scatter([], [], s=80, linewidths=0.0) if cfg.viz.show_gt_proj_2d else None

    pred_lc = None
    gt_lc = None
    if cfg.viz.show_pred_proj_2d and cfg.viz.show_pred_line_2d:
        pred_lc = LineCollection([], linewidths=2.0)
        axim.add_collection(pred_lc)
    if cfg.viz.show_gt_proj_2d and cfg.viz.show_gt_line_2d:
        gt_lc = LineCollection([], linewidths=2.0)
        axim.add_collection(gt_lc)

    # principal point cross in display coords
    h_disp0, w_disp0 = img0.shape[:2]
    cx_disp0 = (w_disp0 - 1) - intr.cx
    cy_disp0 = (h_disp0 - 1) - intr.cy
    cam_cross = axim.scatter([cx_disp0], [cy_disp0], s=200, marker="x", linewidths=2)

    # optional: hand on image (current)
    hand2d_sc = None
    hand2d_lc = None
    if cfg.viz.show_hand_current_2d:
        hand2d_sc = axim.scatter([], [], s=20, linewidths=0.0)
        hand2d_lc = LineCollection([], linewidths=2.0)
        axim.add_collection(hand2d_lc)

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

    def update(i: int):
        nonlocal cur_triad, pred_triads
        current_frame["i"] = i

        # current camera pose from obs
        tc_i = t_c[i]
        Rc_i = R_c[i]

        # --- update current wrist marker & triad ---
        p = t_w[i]
        cur_pt._offsets3d = ([p[0]], [p[1]], [p[2]])

        for line in cur_triad:
            line.remove()
        cur_triad.clear()
        cur_triad.extend(draw_triad(ax3d, p, R_w[i], axis_len=axis_len, lw=2.0))

        # -------------------------
        # Update current camera center pose (world frame)
        # -------------------------
        pc = t_c[i]
        cam_cur_pt._offsets3d = ([pc[0]], [pc[1]], [pc[2]])

        for line in cam_cur_triad:
            line.remove()
        cam_cur_triad.clear()
        cam_cur_triad.extend(draw_triad(ax3d, pc, R_c[i], axis_len=axis_len * 0.8, lw=2.0))

        # -------------------------
        # Update current camera+0.1m pose (world frame)
        # -------------------------
        pcf = t_cf[i]
        camf_cur_pt._offsets3d = ([pcf[0]], [pcf[1]], [pcf[2]])

        for line in camf_cur_triad:
            line.remove()
        camf_cur_triad.clear()
        camf_cur_triad.extend(draw_triad(ax3d, pcf, R_c[i], axis_len=axis_len * 0.8, lw=2.0))

        # --- optional: current hand skeleton (3D) ---
        joint_pts_w = None
        segs_w = None
        if cfg.viz.show_hand_current_3d or cfg.viz.show_hand_current_2d:
            act7 = obs_state[i, 9:16]
            joint_pts_w, segs_w, _ = hand_fk.compute(
                wrist_xyz=obs_state[i, 0:3],
                wrist_rot6d=obs_state[i, 3:9],
                actuations7=act7,
                actuations_to_joints_output_is_degrees=cfg.viz.hand.actuations_to_joints_output_is_degrees,
            )

        if cfg.viz.show_hand_current_3d and hand_skel_lc is not None:
            hand_skel_lc.set_segments(segs_w if segs_w is not None else np.empty((0, 2, 3)))
            if hand_joint_sc is not None and joint_pts_w is not None:
                valid = np.isfinite(joint_pts_w).all(axis=1)
                pv = joint_pts_w[valid]
                hand_joint_sc._offsets3d = (pv[:, 0], pv[:, 1], pv[:, 2]) if pv.shape[0] else ([], [], [])

        # --- optional: pred camera+fwd in 3D ---
        t_cf_fut = None
        R_c_fut = None
        if cfg.viz.show_pred_cam_fwd_traj_3d:
            t_c_fut = pred_tw[i] + (pred_Rw[i] @ T_w_c.reshape(3, 1)).reshape(H, 3)
            R_c_fut = np.einsum("hij,jk->hik", pred_Rw[i], R_w_c)
            t_cf_fut = t_c_fut + (R_c_fut @ P_c_forward.reshape(3, 1)).reshape(H, 3)

            future_line.set_data(t_cf_fut[:, 0], t_cf_fut[:, 1])
            future_line.set_3d_properties(t_cf_fut[:, 2])
            future_start._offsets3d = ([t_cf_fut[0, 0]], [t_cf_fut[0, 1]], [t_cf_fut[0, 2]])

            for line in pred_triads:
                line.remove()
            pred_triads.clear()

            step = max(1, int(cfg.viz.pred_pose_step))
            pred_axis_len = axis_len * 0.7
            pred_lw = 1.5
            for h in range(0, H, step):
                pred_triads.extend(draw_triad(ax3d, t_cf_fut[h], R_c_fut[h], axis_len=pred_axis_len, lw=pred_lw))

        # --- optional: future hand skeletons (3D) ---
        if cfg.viz.show_hand_future_3d and pred_hand_skel_lc is not None:
            Hh = H
            if Hh <= 0:
                pred_hand_skel_lc.set_segments(np.empty((0, 2, 3), dtype=np.float64))
                pred_hand_skel_lc.set_colors([])
            else:
                rgba_h = colors_pred_future_gradient(Hh)
                all_segs = []
                all_cols = []
                hand_step = max(1, int(cfg.viz.hand.step))

                for hidx in range(0, Hh, hand_step):
                    wrist_xyz_h = pred_tw[i, hidx, :]
                    wrist_R_h = pred_Rw[i, hidx, :, :]
                    wrist_rot6d_h = matrix_to_rot6d(wrist_R_h)

                    if pred_hand is not None:
                        act7_h = pred_hand[i, hidx, :]
                    else:
                        act7_h = obs_state[i, 9:16]

                    _, segs_w_h, _ = hand_fk.compute(
                        wrist_xyz=wrist_xyz_h,
                        wrist_rot6d=wrist_rot6d_h,
                        actuations7=act7_h,
                        actuations_to_joints_output_is_degrees=True,
                    )
                    if segs_w_h is None or np.asarray(segs_w_h).size == 0:
                        continue

                    segs_w_h = np.asarray(segs_w_h, dtype=np.float64)
                    all_segs.append(segs_w_h)
                    all_cols.append(np.tile(rgba_h[hidx], (segs_w_h.shape[0], 1)))

                if len(all_segs) == 0:
                    pred_hand_skel_lc.set_segments(np.empty((0, 2, 3), dtype=np.float64))
                    pred_hand_skel_lc.set_colors([])
                else:
                    pred_hand_skel_lc.set_segments(np.concatenate(all_segs, axis=0))
                    pred_hand_skel_lc.set_colors(np.concatenate(all_cols, axis=0))

        # --- update image ---
        img_received = np.transpose(imgs[i], (1, 2, 0))
        im_artist.set_data(img_received)
        h_disp, w_disp = img_received.shape[:2]

        cam_cross.set_offsets(np.array([[(w_disp - 1) - intr.cx, (h_disp - 1) - intr.cy]], dtype=np.float64))

        img_orig = rotate_image_180(img_received)
        h0, w0 = img_orig.shape[:2]

        # --- optional: current hand on image ---
        if cfg.viz.show_hand_current_2d and hand2d_sc is not None and hand2d_lc is not None:
            if joint_pts_w is None or segs_w is None:
                hand2d_sc.set_offsets(np.empty((0, 2)))
                hand2d_lc.set_segments(np.empty((0, 2, 2)))
            else:
                uv_hand, _ = project_points_world_to_display(
                    joint_pts_w, tc_i, Rc_i,
                    intr.fx, intr.fy, intr.cx, intr.cy,
                    dist.k1, dist.k2, dist.k3, dist.k4,
                    w_disp, h_disp, w0, h0,
                )
                hand2d_sc.set_offsets(uv_hand)

                segs_hand = project_segments_world_to_display(
                    segs_w, tc_i, Rc_i,
                    intr.fx, intr.fy, intr.cx, intr.cy,
                    dist.k1, dist.k2, dist.k3, dist.k4,
                    w_disp, h_disp, w0, h0,
                )
                hand2d_lc.set_segments(segs_hand)

        # --- optional: pred projection on image (green->red) ---
        if cfg.viz.show_pred_proj_2d and pred_sc is not None:
            if t_cf_fut is None:
                pred_sc.set_offsets(np.empty((0, 2)))
                if pred_lc is not None:
                    pred_lc.set_segments(np.empty((0, 2, 2)))
                    pred_lc.set_colors([])
            else:
                dt_pred = t_cf_fut - tc_i.reshape(1, 3)
                P_cam_pred = (Rc_i.T @ dt_pred.T).T
                uv0_pred = project_points_equidistant_full(
                    P_cam_pred, intr.fx, intr.fy, intr.cx, intr.cy, dist.k1, dist.k2, dist.k3, dist.k4
                )
                inb0 = (uv0_pred[:, 0] >= 0) & (uv0_pred[:, 0] < w0) & (uv0_pred[:, 1] >= 0) & (uv0_pred[:, 1] < h0)
                uv0_pred = uv0_pred[inb0]

                if uv0_pred.shape[0] == 0:
                    pred_sc.set_offsets(np.empty((0, 2)))
                    if pred_lc is not None:
                        pred_lc.set_segments(np.empty((0, 2, 2)))
                        pred_lc.set_colors([])
                else:
                    uv_disp_pred = np.empty_like(uv0_pred)
                    uv_disp_pred[:, 0] = (w_disp - 1) - uv0_pred[:, 0]
                    uv_disp_pred[:, 1] = (h_disp - 1) - uv0_pred[:, 1]

                    pred_sc.set_offsets(uv_disp_pred)
                    rgba = colors_pred_future_gradient(uv_disp_pred.shape[0])
                    pred_sc.set_facecolors(rgba)
                    pred_sc.set_edgecolors(rgba)

                    if pred_lc is not None:
                        segs = segments_from_points(uv_disp_pred)
                        pred_lc.set_segments(segs)
                        pred_lc.set_colors(rgba[:-1] if segs.shape[0] else [])

        # --- baseline/optional: GT projection on image (black->white) ---
        if cfg.viz.show_gt_proj_2d and gt_sc is not None:
            j1 = min(T, i + H)
            if i >= j1:
                gt_sc.set_offsets(np.empty((0, 2)))
                if gt_lc is not None:
                    gt_lc.set_segments(np.empty((0, 2, 2)))
                    gt_lc.set_colors([])
            else:
                idx = np.arange(i, j1, dtype=np.int64)
                dt_gt = t_cf[idx] - tc_i.reshape(1, 3)
                P_cam_gt = (Rc_i.T @ dt_gt.T).T
                uv0_gt = project_points_equidistant_full(
                    P_cam_gt, intr.fx, intr.fy, intr.cx, intr.cy, dist.k1, dist.k2, dist.k3, dist.k4
                )
                inb0 = (uv0_gt[:, 0] >= 0) & (uv0_gt[:, 0] < w0) & (uv0_gt[:, 1] >= 0) & (uv0_gt[:, 1] < h0)
                uv0_gt = uv0_gt[inb0]

                if uv0_gt.shape[0] == 0:
                    gt_sc.set_offsets(np.empty((0, 2)))
                    if gt_lc is not None:
                        gt_lc.set_segments(np.empty((0, 2, 2)))
                        gt_lc.set_colors([])
                else:
                    uv_disp_gt = np.empty_like(uv0_gt)
                    uv_disp_gt[:, 0] = (w_disp - 1) - uv0_gt[:, 0]
                    uv_disp_gt[:, 1] = (h_disp - 1) - uv0_gt[:, 1]

                    gt_sc.set_offsets(uv_disp_gt)
                    rgba = colors_gt_future_gradient(uv_disp_gt.shape[0])
                    gt_sc.set_facecolors(rgba)
                    gt_sc.set_edgecolors(rgba)

                    if gt_lc is not None:
                        segs = segments_from_points(uv_disp_gt)
                        gt_lc.set_segments(segs)
                        gt_lc.set_colors(rgba[:-1] if segs.shape[0] else [])

        fig.suptitle(f"frame {i+1}/{T}")

        artists = (
            [im_artist, cam_cross, cur_pt, cam_cur_pt, camf_cur_pt]
            + list(world_triad)
            + cur_triad + cam_cur_triad + camf_cur_triad + pred_triads
        )
        for a in [wrist_line, cam_center_line, cam_fwd_line, future_line, future_start]:
            if a is not None:
                artists.append(a)

        if hand_skel_lc is not None:
            artists.append(hand_skel_lc)
        if hand_joint_sc is not None:
            artists.append(hand_joint_sc)
        if pred_hand_skel_lc is not None:
            artists.append(pred_hand_skel_lc)

        if pred_sc is not None:
            artists.append(pred_sc)
        if gt_sc is not None:
            artists.append(gt_sc)
        if pred_lc is not None:
            artists.append(pred_lc)
        if gt_lc is not None:
            artists.append(gt_lc)

        if hand2d_sc is not None:
            artists.append(hand2d_sc)
        if hand2d_lc is not None:
            artists.append(hand2d_lc)

        return artists

    interval_ms = 1000.0 / max(cfg.runtime.fps, 1e-6)
    ani = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)

    plt.tight_layout()
    if cfg.io.save_path:
        ani.save(cfg.io.save_path, fps=cfg.runtime.fps)
        print(f"Saved: {cfg.io.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()