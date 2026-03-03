#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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
    R = np.transpose(R, (0, 2, 1))             # column-major like your earlier function
    return R.reshape(*orig_shape, 3, 3)


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
    Draw a coordinate triad at position p with rotation R.
    R is column-major (columns are basis vectors in world frame).
    X=Red, Y=Green, Z=Blue (fixed RGB convention).
    """
    p = np.asarray(p).reshape(3)

    ex = R[:, 0] * axis_len
    ey = R[:, 1] * axis_len
    ez = R[:, 2] * axis_len

    lx, = ax.plot(
        [p[0], p[0] + ex[0]],
        [p[1], p[1] + ex[1]],
        [p[2], p[2] + ex[2]],
        color='r',
        linewidth=lw
    )
    ly, = ax.plot(
        [p[0], p[0] + ey[0]],
        [p[1], p[1] + ey[1]],
        [p[2], p[2] + ey[2]],
        color='g',
        linewidth=lw
    )
    lz, = ax.plot(
        [p[0], p[0] + ez[0]],
        [p[1], p[1] + ez[1]],
        [p[2], p[2] + ez[2]],
        color='b',
        linewidth=lw
    )
    return (lx, ly, lz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_npz", type=str, default="/home/nan/datasets/deploy_data/val_data_episode_0000.npz")
    ap.add_argument("--pred_npz", type=str, default="/home/nan/datasets/deploy_data/model_outputs.npz")
    ap.add_argument("--pred_key", type=str, default="action.arm")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--axis_len", type=float, default=0.03)
    ap.add_argument("--pred_pose_step", type=int, default=4, help="Draw predicted pose triad every N steps along horizon.")
    ap.add_argument("--save", type=str, default="", help="If set, save animation to this path (e.g. out.mp4)")
    args = ap.parse_args()

    # --- Load val episode ---
    val = np.load(args.val_npz, allow_pickle=True)
    imgs = val["observation.images.camera_wrist_right"]   # (T,3,H,W)
    obs_state = val["observation.state"]                  # (T,16)

    if obs_state.ndim != 2 or obs_state.shape[1] < 9:
        raise ValueError(f"observation.state expected (T,>=9), got {obs_state.shape}")
    if imgs.ndim != 4 or imgs.shape[1] != 3:
        raise ValueError(f"images expected (T,3,H,W), got {imgs.shape}")

    T = obs_state.shape[0]
    gt_pose9 = obs_state[:, :9]                 # (T,9) xyz+rot6d
    gt_xyz = gt_pose9[:, :3]                    # (T,3)
    gt_R = rot6d_to_matrix(gt_pose9[:, 3:9])    # (T,3,3)

    # --- Load predictions ---
    pred = np.load(args.pred_npz, allow_pickle=True)
    if args.pred_key not in pred:
        raise KeyError(f"Pred key '{args.pred_key}' not found. Available: {list(pred.keys())}")

    pred_arm = pred[args.pred_key]              # (T,H,9)
    if pred_arm.ndim != 3 or pred_arm.shape[-1] != 9:
        raise ValueError(f"{args.pred_key} expected (T,H,9), got {pred_arm.shape}")

    Tp, H, _ = pred_arm.shape
    if Tp != T:
        raise ValueError(f"T mismatch: val T={T}, pred T={Tp}")

    pred_xyz = pred_arm[:, :, :3]               # (T,H,3)
    pred_R = rot6d_to_matrix(pred_arm[:, :, 3:9])  # (T,H,3,3)

    # --- Figure with 2 subplots ---
    fig = plt.figure(figsize=(14, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axim = fig.add_subplot(1, 2, 2)

    # --- Left: static GT trajectory + first pose triad ---
    ax3d.set_title("3D: GT trajectory + current pose + predicted future(16) with pose triads")
    ax3d.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], linewidth=2.0)
    ax3d.scatter([gt_xyz[0, 0]], [gt_xyz[0, 1]], [gt_xyz[0, 2]], s=40)

    _triad0 = draw_triad(ax3d, gt_xyz[0], gt_R[0], axis_len=args.axis_len, lw=2.0)

    # dynamic artists: current pose
    cur_pt = ax3d.scatter([gt_xyz[0, 0]], [gt_xyz[0, 1]], [gt_xyz[0, 2]], s=60)
    cur_triad = list(draw_triad(ax3d, gt_xyz[0], gt_R[0], axis_len=args.axis_len, lw=2.0))

    # dynamic artists: predicted future polyline
    future_line, = ax3d.plot(
        pred_xyz[0, :, 0], pred_xyz[0, :, 1], pred_xyz[0, :, 2],
        linewidth=2.0
    )
    future_start = ax3d.scatter([pred_xyz[0, 0, 0]], [pred_xyz[0, 0, 1]], [pred_xyz[0, 0, 2]], s=25)

    # predicted future pose triads (every N steps)
    pred_triads = []  # list of line artists (3 per triad)
    step = max(1, int(args.pred_pose_step))
    pred_axis_len = args.axis_len * 0.7
    pred_lw = 1.5

    for h in range(0, H, step):
        p = pred_xyz[0, h]
        R = pred_R[0, h]
        pred_triads.extend(draw_triad(ax3d, p, R, axis_len=pred_axis_len, lw=pred_lw))

    # axis limits
    all_pts = np.concatenate([gt_xyz.reshape(-1, 3), pred_xyz.reshape(-1, 3)], axis=0)
    ax3d.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
    ax3d.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
    ax3d.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    set_axes_equal(ax3d)

    # --- Right: image ---
    axim.set_title("Wrist camera")
    img0 = np.transpose(imgs[0], (1, 2, 0))  # CHW -> HWC
    im_artist = axim.imshow(img0, vmin=0.0, vmax=1.0)
    axim.axis("off")
    txt = axim.text(10, 20, "t=0", color="white", fontsize=12)

    def update(t):
        # current pose point
        p = gt_xyz[t]
        cur_pt._offsets3d = ([p[0]], [p[1]], [p[2]])

        # current triad
        for line in cur_triad:
            line.remove()
        cur_triad.clear()
        cur_triad.extend(draw_triad(ax3d, p, gt_R[t], axis_len=args.axis_len, lw=2.0))

        # predicted future polyline
        f = pred_xyz[t]  # (H,3)
        future_line.set_data(f[:, 0], f[:, 1])
        future_line.set_3d_properties(f[:, 2])
        future_start._offsets3d = ([f[0, 0]], [f[0, 1]], [f[0, 2]])

        # predicted future triads
        for line in pred_triads:
            line.remove()
        pred_triads.clear()
        for h in range(0, H, step):
            ph = pred_xyz[t, h]
            Rh = pred_R[t, h]
            pred_triads.extend(draw_triad(ax3d, ph, Rh, axis_len=pred_axis_len, lw=pred_lw))

        # image
        img = np.transpose(imgs[t], (1, 2, 0))
        im_artist.set_data(img)
        txt.set_text(f"t={t}/{T-1}")

        return [cur_pt, future_line, future_start, im_artist, txt] + cur_triad + pred_triads

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