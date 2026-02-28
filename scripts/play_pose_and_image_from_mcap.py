#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image


BAG_PATH = "/home/nan/datasets/0225_data/episode_0/episode_0.mcap"
TOPIC_POSE = "/right_hand_location"
TOPIC_IMG  = "/vx/color/image_raw"

# --- Camera intrinsics (for the ORIGINAL, un-rotated image) ---
FX = 243.31371125
FY = 243.36591085
CX = 330.20097652
CY = 185.82838602

# --- Equidistant (fisheye) distortion coefficients ---
K1 = 0.05295615
K2 = -0.03939104
K3 = 0.01021067
K4 = -0.00235273

HORIZON = 10
DRAW_PAST = False         # False -> 不画历史（蓝点）
POINT_SIZE = 80          # 点大小：建议 60~200（你说看不到就把这个加大）
POINT_EDGE = 0.0         # 点边框线宽：0 或 0.5 都行

# --- Camera extrinsics in wrist frame ---
T_W_C = np.array([-0.03704, -0.05092, 0.02236], dtype=np.float64)  # meters
EULER_XYZ_DEG = np.array([-171.825 + 180, -11.079, 12.307], dtype=np.float64)  # (z,x,y) custom

# --- A point 0.1m in front of camera optical center along camera +Z ---
CAM_FORWARD_OFFSET_M = 0.1
P_C_FORWARD = np.array([0.0, 0.0, CAM_FORWARD_OFFSET_M], dtype=np.float64)  # in camera frame


def quat_to_rot(qw, qx, qy, qz):
    """Quaternion (w,x,y,z) -> 3x3 rotation."""
    n = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-12:
        return np.eye(3)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [    2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qw*qx)],
        [    2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)


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
    Intrinsic Z -> X -> Y (column-vector convention):
      R = Ry(y) @ Rx(x) @ Rz(z)
    """
    z = np.deg2rad(z_deg)
    x = np.deg2rad(x_deg)
    y = np.deg2rad(y_deg)
    return rot_y(y) @ rot_x(x) @ rot_z(z)


def image_msg_to_numpy(msg: Image):
    """Convert ROS2 Image message to NumPy array."""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    step = msg.step
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, step)[:, :w*3].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]  # BGR -> RGB
        return img

    if enc == "mono8":
        return buf.reshape(h, step)[:, :w].reshape(h, w)

    return buf.reshape(h, step)


def rotate_image_180(img):
    """Rotate image by 180 degrees (flip both axes)."""
    if img.ndim == 2:
        return img[::-1, ::-1]
    return img[::-1, ::-1, :]


def read_two_topics_from_bag():
    """Read pose and image topics from MCAP."""
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=BAG_PATH, storage_id="mcap"),
        ConverterOptions("cdr", "cdr")
    )

    poses, imgs = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == TOPIC_POSE:
            poses.append(deserialize_message(data, PoseStamped))
        elif topic == TOPIC_IMG:
            imgs.append(deserialize_message(data, Image))

    return poses, imgs


def project_points_equidistant_full(P_cam, fx, fy, cx, cy, k1, k2, k3, k4):
    """
    Project 3D points to pixels using equidistant fisheye model.
    Returns uv (N,2); invalid points -> NaN.
    """
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


def main():
    rclpy.init()

    poses, imgs = read_two_topics_from_bag()
    n = min(len(poses), len(imgs))
    if n == 0:
        raise RuntimeError("No messages found for pose/image topics. Check BAG_PATH and topic names.")

    poses = poses[:n]
    imgs = imgs[:n]

    # --- Wrist poses in world ---
    t_w = np.zeros((n, 3), dtype=np.float64)
    R_w = np.zeros((n, 3, 3), dtype=np.float64)
    for i in range(n):
        p = poses[i].pose.position
        o = poses[i].pose.orientation
        t_w[i] = [p.x, p.y, p.z]
        R_w[i] = quat_to_rot(o.w, o.x, o.y, o.z)

    # --- Camera extrinsics in wrist ---
    R_w_c = euler_zxy_custom_deg_to_rot(EULER_XYZ_DEG[0], EULER_XYZ_DEG[1], EULER_XYZ_DEG[2])

    # --- Camera poses in world (center) ---
    t_c = t_w + (R_w @ T_W_C.reshape(3, 1)).reshape(n, 3)
    R_c = np.einsum("nij,jk->nik", R_w, R_w_c)

    # --- Camera forward point (+0.1m) in world ---
    t_cf = t_c + (R_c @ P_C_FORWARD.reshape(3, 1)).reshape(n, 3)

    # --- Relative to first wrist frame ---
    t0 = t_w[0]
    R0T = R_w[0].T

    t_w_rel  = (R0T @ (t_w  - t0).T).T
    t_c_rel  = (R0T @ (t_c  - t0).T).T
    t_cf_rel = (R0T @ (t_cf - t0).T).T
    R_c_rel  = np.einsum("ij,njk->nik", R0T, R_c)

    # --- Plot setup ---
    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axim = fig.add_subplot(1, 2, 2)

    ax3d.set_title("Wrist / Camera Center / Camera+0.1m (relative to wrist frame-0)")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")

    wrist_line, = ax3d.plot(t_w_rel[:, 0], t_w_rel[:, 1], t_w_rel[:, 2],
                            color="lightgray", lw=2, label="wrist")
    cam_center_line, = ax3d.plot(t_c_rel[:, 0], t_c_rel[:, 1], t_c_rel[:, 2],
                                 color="black", lw=2, label="camera center")
    cam_fwd_line, = ax3d.plot(t_cf_rel[:, 0], t_cf_rel[:, 1], t_cf_rel[:, 2],
                              color="purple", lw=2, label="camera +0.1m")
    ax3d.legend(loc="upper left")

    # --- Image view ---
    axim.set_title("2D: projected points (no lines)")
    axim.axis("off")

    # IMPORTANT: show RECEIVED image (already rotated)
    img_received0 = image_msg_to_numpy(imgs[0])
    im_artist = axim.imshow(img_received0)

    # 2D points on RECEIVED image coordinates
    future_sc = axim.scatter([], [], s=POINT_SIZE, c="red", linewidths=POINT_EDGE)
    past_sc = None
    if DRAW_PAST:
        past_sc = axim.scatter([], [], s=POINT_SIZE, c="blue", linewidths=POINT_EDGE)

    # principal point in DISPLAY coords (because we display received image)
    h0, w0 = img_received0.shape[:2]
    cx_disp0 = (w0 - 1) - CX
    cy_disp0 = (h0 - 1) - CY
    cam_cross = axim.scatter([cx_disp0], [cy_disp0], s=200, marker="x", linewidths=2)

    # --- 3D bounds ---
    all_pts = np.vstack([t_w_rel, t_c_rel, t_cf_rel])
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

    axis_len = max(span * 0.15, 0.05)

    def draw_origin_axes():
        ax3d.quiver(0, 0, 0, axis_len, 0, 0, color="red", linewidth=2)
        ax3d.quiver(0, 0, 0, 0, axis_len, 0, color="green", linewidth=2)
        ax3d.quiver(0, 0, 0, 0, 0, axis_len, color="blue", linewidth=2)

    draw_origin_axes()
    current_axes = []

    def draw_current_axes(pos, R):
        dirs = R * axis_len
        qx = ax3d.quiver(pos[0], pos[1], pos[2], dirs[0, 0], dirs[1, 0], dirs[2, 0],
                         color="red", linewidth=2)
        qy = ax3d.quiver(pos[0], pos[1], pos[2], dirs[0, 1], dirs[1, 1], dirs[2, 1],
                         color="green", linewidth=2)
        qz = ax3d.quiver(pos[0], pos[1], pos[2], dirs[0, 2], dirs[1, 2], dirs[2, 2],
                         color="blue", linewidth=2)
        return [qx, qy, qz]

    current_frame = {"i": 0}

    def update(i):
        nonlocal current_axes
        current_frame["i"] = i

        # current camera axes at camera center
        for q in current_axes:
            q.remove()
        current_axes = draw_current_axes(t_c_rel[i], R_c_rel[i])

        # show received image (already rotated)
        img_received = image_msg_to_numpy(imgs[i])
        im_artist.set_data(img_received)
        h, w = img_received.shape[:2]

        # update principal point cross in display coords
        cx_disp = (w - 1) - CX
        cy_disp = (h - 1) - CY
        cam_cross.set_offsets(np.array([[cx_disp, cy_disp]], dtype=np.float64))

        j1 = min(n, i + 1 + HORIZON)
        j0 = max(0, i - HORIZON)

        # We compute projection in ORIGINAL (unrotated) coords, so we need img_orig size.
        # We can get it by unrotating received image.
        img_orig = rotate_image_180(img_received)
        h0, w0 = img_orig.shape[:2]

        # FUTURE points: forward-point trajectory expressed in current camera-center frame
        if i < j1:
            idx_f = np.arange(i, j1, dtype=np.int64)
            dt = t_cf_rel[idx_f] - t_c_rel[i]
            P_cam = (R_c_rel[i].T @ dt.T).T
            uv = project_points_equidistant_full(P_cam, FX, FY, CX, CY, K1, K2, K3, K4)

            # in-bounds in ORIGINAL image coords
            inb0 = (uv[:, 0] >= 0) & (uv[:, 0] < w0) & (uv[:, 1] >= 0) & (uv[:, 1] < h0)
            uv0 = uv[inb0]

            # convert ORIGINAL coords -> DISPLAY(received rotated) coords by 180 flip
            if uv0.shape[0] > 0:
                uv_disp = np.empty_like(uv0)
                uv_disp[:, 0] = (w - 1) - uv0[:, 0]
                uv_disp[:, 1] = (h - 1) - uv0[:, 1]
                future_sc.set_offsets(uv_disp)
            else:
                future_sc.set_offsets(np.empty((0, 2)))
        else:
            future_sc.set_offsets(np.empty((0, 2)))

        # PAST points (optional)
        if past_sc is not None and (j0 <= i):
            idx_p = np.arange(j0, i + 1, dtype=np.int64)
            dt = t_cf_rel[idx_p] - t_c_rel[i]
            P_cam = (R_c_rel[i].T @ dt.T).T
            uv = project_points_equidistant_full(P_cam, FX, FY, CX, CY, K1, K2, K3, K4)

            inb0 = (uv[:, 0] >= 0) & (uv[:, 0] < w0) & (uv[:, 1] >= 0) & (uv[:, 1] < h0)
            uv0 = uv[inb0]

            if uv0.shape[0] > 0:
                uv_disp = np.empty_like(uv0)
                uv_disp[:, 0] = (w - 1) - uv0[:, 0]
                uv_disp[:, 1] = (h - 1) - uv0[:, 1]
                past_sc.set_offsets(uv_disp)
            else:
                past_sc.set_offsets(np.empty((0, 2)))
        else:
            if past_sc is not None:
                past_sc.set_offsets(np.empty((0, 2)))

        fig.suptitle(f"frame {i+1}/{n}")

        artists = [
            wrist_line, cam_center_line, cam_fwd_line,
            im_artist, cam_cross,
            future_sc,
            *current_axes
        ]
        if past_sc is not None:
            artists.append(past_sc)
        return artists

    ani = FuncAnimation(fig, update, frames=n, interval=50, blit=False)

    paused = {"value": False}

    def on_key(event):
        if event.key == " ":
            paused["value"] = not paused["value"]
            if paused["value"]:
                ani.event_source.stop()
                fig.suptitle(f"PAUSED - frame {current_frame['i']+1}/{n}")
            else:
                ani.event_source.start()

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.tight_layout()
    plt.show()
    rclpy.shutdown()


if __name__ == "__main__":
    main()