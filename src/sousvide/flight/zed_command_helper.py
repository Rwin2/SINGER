#!/usr/bin/env python3

import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import cv2
import pyzed.sl as sl
from rclpy.publisher import Publisher

from px4_msgs.msg import (
    VehicleCommand,
    OffboardControlMode,
    VehicleOdometry,
    VehicleRatesSetpoint,
    TrajectorySetpoint,
    ActuatorMotors
)


def _wrap_pi(a: float) -> float:
    return (a + np.pi) % (2*np.pi) - np.pi

def _yaw_from_R(R: np.ndarray) -> float:
    # standard ZYX: yaw = atan2(R[1,0], R[0,0])
    return float(np.arctan2(R[1,0], R[0,0]))

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0.0 else v / n

def _ema_scalar(r_prev: float, z: float, alpha: float, lo: float, hi: float) -> float:
    r = alpha * z + (1.0 - alpha) * r_prev
    return float(np.clip(r, lo, hi))

def _pixel_to_cam_ray(K: np.ndarray, uv: Tuple[int, int]) -> np.ndarray:
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    u, v = float(uv[0]), float(uv[1])
    d = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64)
    return _unit(d)

def _clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0.0 or n <= max_norm else v * (max_norm / n)

def _accel_limit(v_prev: np.ndarray, v_des: np.ndarray, a_max: float, dt: float) -> np.ndarray:
    dv = v_des - v_prev
    lim = a_max * max(dt, 1e-3)
    return v_prev + _clamp_norm(dv, lim)



def quat_xyzw_to_R(qx, qy, qz, qw):
    # Normalize
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    # Rotation (body->world) from quaternion (x,y,z,w)
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]
    ], dtype=np.float32)

def hom_from_R_t(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = t.astype(np.float32)
    return T

def pose_c2b(T_c2b, p_cam):
    p_body_h  = T_c2b @ np.array([*p_cam, 1.0], dtype=np.float32)

    return p_body_h

def pose_from_similarity_xyz(similarity: np.ndarray,
                             xyz_m: np.ndarray,
                             top_percent: float = 10.0,
                             min_pixels: int = 200):
    """
    Returns:
      ok: bool
      p_cam: (3,) float [X,Y,Z] in meters, camera frame (Z forward; ZED convention)
      uv_centroid: (u*, v*) int pixel centroid of ROI (for aiming/visualization)
      mask: boolean mask of selected ROI
    """
    if xyz_m is None or xyz_m.ndim != 3 or xyz_m.shape[2] < 3:
        return False, None, None, None

    Hs, Ws = similarity.shape[:2]
    Hx, Wx = xyz_m.shape[:2]
    if (Hs, Ws) != (Hx, Wx):
        xyz_m = cv2.resize(xyz_m, (Ws, Hs), interpolation=cv2.INTER_NEAREST)

    sim = similarity.astype(np.float32, copy=False)
    n = sim.size
    k = max(0, min(n - 1, int((1.0 - top_percent/100.0) * n)))
    thr = np.partition(sim.ravel(), k)[k]
    mask = (sim >= thr)

    if mask.sum() < min_pixels:
        return False, None, None, mask

    roi_xyz = xyz_m[mask, :3]
    finite = np.all(np.isfinite(roi_xyz), axis=1)
    roi_xyz = roi_xyz[finite]
    if roi_xyz.shape[0] < min_pixels:
        return False, None, None, mask

    # Robust 3D point in camera frame
    p_cam = np.median(roi_xyz, axis=0).astype(np.float32)  # [X,Y,Z]

    # Pixel centroid (for aiming)
    ys, xs = np.nonzero(mask)
    if ys.size:
        uv_centroid = (int(np.median(xs)), int(np.median(ys)))
    else:
        uv_centroid = (None, None)

    return True, p_cam, uv_centroid, mask

def pose_b2w(x_ext, p_body_h, *, quat_is_world_to_body: bool = True):
    t_wb = np.array(x_ext[0:3], dtype=np.float32)          # body pos in WORLD
    qx, qy, qz, qw = map(float, x_ext[6:10])

    R = quat_xyzw_to_R(qx, qy, qz, qw)
    R_wb = R.T if quat_is_world_to_body else R            

    T_b2w = hom_from_R_t(R_wb, t_wb)
    p_world_h = T_b2w @ p_body_h
    target_world = p_world_h[:3].astype(np.float32)
    return target_world, t_wb, R_wb

# def pose_b2w(x_ext, p_body_h):
#     t_wb = np.array(x_ext[0:3], dtype=np.float32)            # body pos [x,y,z]
#     q_wb = np.array(x_ext[6:10], dtype=np.float32)           # qx,qy,qz,qw
#     R_wb = quat_xyzw_to_R(*q_wb)
#     T_b2w = hom_from_R_t(R_wb, t_wb)

#     p_world_h = T_b2w @ p_body_h
#     target_world = p_world_h[:3]

#     return target_world, t_wb


def publish_position_setpoint(timestamp: int,
                              altitude_hold: np.ndarray,
                              desired_position: np.ndarray,
                              traj_sp_pub) -> None:
    ts = TrajectorySetpoint(timestamp=int(timestamp))  # PX4 expects int (µs)
    # Hold current Z, command XY
    ts.position = [
        float(desired_position[0]),
        float(desired_position[1]),
        float(altitude_hold),
    ]
    ts.velocity     = [float('nan'), float('nan'), float('nan')]
    ts.acceleration = [float('nan'), float('nan'), float('nan')]
    ts.jerk         = [float('nan'), float('nan'), float('nan')]
    ts.yaw          = float('nan')
    ts.yawspeed     = float('nan')
    traj_sp_pub.publish(ts)

# in zch.py
def publish_velocity_setpoint(timestamp: int, v_world: np.ndarray, traj_sp_pub) -> None:
    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.position     = [float('nan')]*3
    ts.velocity     = [float(v_world[0]), float(v_world[1]), float(v_world[2])]
    ts.acceleration = [float('nan')]*3
    ts.jerk         = [float('nan')]*3
    ts.yaw          = float('nan')
    ts.yawspeed     = float('nan')
    traj_sp_pub.publish(ts)

def publish_velocity_heading_setpoint(timestamp: int,
                                      v_world: np.ndarray,
                                      yaw: float,
                                      traj_sp_pub) -> None:
    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.position     = [float('nan')]*3
    ts.velocity     = [float(v_world[0]), float(v_world[1]), float(v_world[2])]
    ts.acceleration = [float('nan')]*3
    ts.jerk         = [float('nan')]*3
    ts.yaw          = float(yaw)        # align heading toward target
    ts.yawspeed     = float('nan')      # we drive yaw via setpoint + slew below
    traj_sp_pub.publish(ts)


def traj_from_target_pose(
    x_ext: np.ndarray,
    p_cam: np.ndarray,
    T_c2b: np.ndarray,
    *,
    dt: float = 1.0/20.0,
    total_time: float = 6.0,
    v_max: float = 1.0,
    standoff_m: float = 0.0,
    eps: float = 1e-6
) -> tuple[np.ndarray, dict]:
    """
    Build a straight-line, speed-limited world-frame trajectory from the current body pose to the target.
    Returns (trajectory[N,3], debug_info).
    """
    # CAMERA -> BODY (homogeneous)
    p_body_h = pose_c2b(T_c2b, p_cam)          # [x,y,z,1] body(NED)
    p_body   = p_body_h[:3].astype(np.float32)

    # BODY -> WORLD (using mocap pose)
    p_world_tgt, t_wb, R_wb = pose_b2w(x_ext, p_body_h)

    # Standoff along the body-ray to the target
    if standoff_m > 0.0:
        r_body   = float(np.linalg.norm(p_body)) + eps
        dir_body = p_body / r_body
        # Stop short by standoff_m (don’t pass through target)
        step_world = (R_wb @ (max(r_body - standoff_m, 0.0) * dir_body)).astype(np.float32)
        p_world_goal = (t_wb + step_world).astype(np.float32)
    else:
        p_world_goal = p_world_tgt

    # Build time-aware, speed-limited trajectory at dt
    p0   = t_wb.copy()
    dvec = p_world_goal - p0
    dist = float(np.linalg.norm(dvec))
    if dist < eps:
        traj = np.vstack([p0, p_world_goal]).astype(np.float32)
        dbg = {
            "p_cam":      p_cam.astype(float),
            "p_body":     p_body.astype(float),
            "p_world_tgt":p_world_tgt.astype(float),
            "p_world_goal":p_world_goal.astype(float),
            "R_wb":       R_wb.astype(float),
            "dist":       dist,
            "steps":      1,
            "step_cap":   0.0,
        }
        return traj, dbg

    direction = dvec / dist
    step_cap  = max(v_max, eps) * dt
    t_needed  = min(total_time, dist / max(v_max, eps))
    steps     = max(1, int(np.ceil(t_needed / dt)))

    waypoints = [p0]
    pos = p0.copy()
    for _ in range(steps):
        remaining = p_world_goal - pos
        rdist = float(np.linalg.norm(remaining))
        if rdist <= step_cap + eps:
            pos = p_world_goal
            waypoints.append(pos.copy())
            break
        pos = pos + direction * step_cap
        waypoints.append(pos.copy())

    if np.linalg.norm(waypoints[-1] - p_world_goal) > 1e-4:
        waypoints.append(p_world_goal.copy())

    traj = np.asarray(waypoints, dtype=np.float32)
    dbg = {
        "p_cam":        p_cam.astype(float),
        "p_body":       p_body.astype(float),
        "p_world_tgt":  p_world_tgt.astype(float),
        "p_world_goal": p_world_goal.astype(float),
        "R_wb":         R_wb.astype(float),
        "dist":         dist,
        "steps":        len(traj)-1,
        "step_cap":     step_cap,
        "dt":           dt,
        "v_max":        v_max,
        "total_time":   total_time,
        "standoff_m":   float(standoff_m),
    }
    return traj, dbg
# def traj_from_target_pose(
#     x_ext: np.ndarray,
#     p_cam: np.ndarray,
#     T_c2b: np.ndarray,
#     *,
#     dt: float = 1.0/20.0,     # control period (s)
#     total_time: float = 6.0,  # cap on trajectory duration (s)
#     v_max: float = 1.0,       # max speed (m/s)
#     eps: float = 1e-6
# ) -> np.ndarray:
#     """
#     Returns an (N,3) array of world-frame waypoints from current body pose toward the target.
#     Frames: camera(ZED)->body(NED)->world(NED)
#     """
#     # CAMERA -> BODY
#     p_body_h = pose_c2b(T_c2b, p_cam)            # [x,y,z,1] body (NED)

#     # BODY -> WORLD (mocap pose in x_ext)
#     target_world, t_wb = pose_b2w(x_ext, p_body_h)  # both in world (NED)

#     # Straight-line, speed-limited trajectory
#     p0   = t_wb.astype(np.float32)
#     dvec = target_world.astype(np.float32) - p0
#     dist = float(np.linalg.norm(dvec))
#     if dist < eps:
#         return np.vstack([p0, target_world])  # trivial (already there)

#     direction = dvec / dist
#     step_cap  = max(v_max, eps) * dt
#     max_steps = max(1, int(np.ceil(min(total_time, dist/max(v_max, eps)) / dt)))

#     waypoints = [p0]
#     pos = p0.copy()
#     for _ in range(max_steps):
#         remaining = target_world - pos
#         rdist = float(np.linalg.norm(remaining))
#         if rdist <= step_cap + eps:
#             pos = target_world
#             waypoints.append(pos.copy())
#             break
#         pos = pos + direction * step_cap
#         waypoints.append(pos.copy())

#     if np.linalg.norm(waypoints[-1] - target_world) > 1e-4:
#         waypoints.append(target_world.copy())

#     return np.asarray(waypoints, dtype=np.float32)



def get_camera(height: int, width: int, fps: int, use_depth: bool = False) -> sl.Camera:
    """Initialize the ZED camera."""
    print("Initializing ZED camera if available...")

    try:
        camera = sl.Camera()
        
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.VGA  # Automatically set resolution
        init_params.camera_fps = fps  # Set the camera FPS

        if use_depth == True:
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT  # Neural mode
            init_params.coordinate_units = sl.UNIT.METER
            init_params.depth_minimum_distance = 0.3       # 0.3 m
            init_params.depth_maximum_distance = 12.0      # 12 m
            init_params.depth_stabilization = 30           # % (0–100 int, not bool)

            # Optional: you can set confidence thresholds if API version supports it
            runtime_params = sl.RuntimeParameters()
            runtime_params.confidence_threshold = 95
            runtime_params.texture_confidence_threshold = 100

        err = camera.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Camera Open Error: {repr(err)}. Exiting program.")
            exit()

        if use_depth == True:
            camera.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 4)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.HUE, 0)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 4)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 3)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.GAMMA, 5)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, 0)  # Auto adjust disabled
            camera.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 4)
            camera.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 34)

        print("Camera found!!!")
        return camera

    except Exception as e:
        print(f"No camera found: {e}")
        return None

def get_image(Camera: sl.Camera,
              use_depth: bool = False) -> tuple[np.ndarray, int]:
    """Capture an image from the ZED camera."""
    if Camera is None:
        print("Camera is not initialized.")
        return None, None, None, None

    # runtime_parameters = sl.RuntimeParameters()

    if not hasattr(get_image, "_inited"):
        get_image._image = sl.Mat()
        if use_depth:
            get_image._xyz = sl.Mat()
            get_image._depth_viz = sl.Mat()
        get_image._rt = sl.RuntimeParameters()
        # get_image._rt.confidence_threshold = 95
        # get_image._rt.texture_confidence_threshold = 100
        # get_image._rt.remove_saturated_areas = False
        # get_image._rt.sensing_mode = sl.SENSING_MODE.FILL
        get_image._inited = True

    # Grab a frame from the camera
    if Camera.grab(get_image._rt) == sl.ERROR_CODE.SUCCESS:
        # image = sl.Mat()
        Camera.retrieve_image(get_image._image, sl.VIEW.LEFT)
        img_np = get_image._image.get_data()
        if use_depth:
            # depth = sl.Mat()
            # depth_for_display = sl.Mat()
            # Camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
            Camera.retrieve_measure(get_image._xyz, sl.MEASURE.XYZ)
            xyz_np = get_image._xyz.get_data()
            Camera.retrieve_image(get_image._depth_viz, sl.VIEW.DEPTH)
            depth_viz_np = get_image._depth_viz.get_data()

        timestamp = Camera.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # Get timestamp

        # print(f"Image resolution: {image.get_width()} x {image.get_height()} || Image timestamp: {timestamp.get_milliseconds()}")
        if use_depth:
            return img_np, xyz_np, depth_viz_np, timestamp.get_milliseconds()
        else:
            return img_np, None, None, timestamp.get_milliseconds()

    else:
        print("Failed to capture image.")
        return None, None, None, None

def close_camera(Camera: sl.Camera) -> None:
    """Close the ZED camera."""
    if Camera is not None:
        print('camera is closed')
        Camera.close()
    else:
        print("No camera closed")
    
def vo2x(vo:VehicleOdometry,T_12:np.ndarray=None) -> np.ndarray:
    """Convert VehicleOdometry to state vector in world frame."""

    xv = np.array([vo.position[0],vo.position[1],vo.position[2],
                    vo.velocity[0],vo.velocity[1],vo.velocity[2],
                    vo.q[1],vo.q[2],vo.q[3],vo.q[0],
                    vo.angular_velocity[0],vo.angular_velocity[1],vo.angular_velocity[2]])
    
    xv[6:10] += np.array([0.0,0.0,0.0,1e-9])
    xv[6:10] = xv[6:10]/np.linalg.norm(xv[6:10])

    # Transform to world frame
    if T_12 is not None:
        xv = x_transform(xv,T_12)

    return xv

def x_transform(x1:np.ndarray,T_12:np.ndarray) -> np.ndarray:
    """ Transform state vector given transform matrix."""

    # Variables
    R_12,t_12 = T_12[0:3,0:3],T_12[0:3,3]
    x2 = np.zeros(13)
    R_b1 =  R.from_quat(x1[6:10]).as_matrix()

    # Transform
    x2[0:3] = R_12@x1[0:3] + t_12
    x2[3:6] = R_12@x1[3:6]
    x2[6:10] = R.from_matrix(R_12@R_b1).as_quat()
    
    return x2

def am2unf(am:ActuatorMotors) -> np.ndarray:
    """Convert actuator motors to normalized motor force input."""
    return np.array([am.control[0],am.control[1],am.control[2],am.control[3]])

def vrs2uvr(vr:VehicleRatesSetpoint) -> np.ndarray:
    """Convert vehicle rates setpoint to vehicle rates input."""
    return np.array([vr.thrust_body[2],vr.roll,vr.pitch,vr.yaw])

# def publish_position_hold(timestamp: int,
#                              x_est: np.ndarray,
#                              traj_sp_pub) -> None:
#     """
#     Hold current position & heading using TrajectorySetpoint only.
#     - x_est: 13-element state [x,y,z, vx,vy,vz, qx,qy,qz,qw, ...]
#     - traj_sp_pub: Publisher<TrajectorySetpoint>
#     """
#     sp = TrajectorySetpoint(timestamp=timestamp)

#     # set position [x, y, z]
#     sp.position[0] = x_est[0]
#     sp.position[1] = x_est[1]
#     sp.position[2] = x_est[2]

#     # extract yaw from quaternion (qx,qy,qz,qw in x_est[6..9])
#     qx, qy, qz, qw = x_est[6], x_est[7], x_est[8], x_est[9]
#     sp.yaw = np.arctan2(2*(qw*qz + qx*qy),
#                         1 - 2*(qy*qy + qz*qz))

#     # ignore velocity/acceleration/jerk feed-forward
#     sp.velocity = [float('nan')] * 3
#     sp.acceleration = [float('nan')] * 3
#     sp.jerk = [float('nan')] * 3

#     traj_sp_pub.publish(sp)

def publish_velocity_hold(timestamp: int, traj_sp_pub) -> None:
    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.velocity     = [0.0, 0.0, 0.0]     # hold zero velocity
    ts.position     = [float('nan')]*3
    ts.acceleration = [float('nan')]*3
    ts.jerk         = [float('nan')]*3
    ts.yaw          = float('nan')
    ts.yawspeed     = float('nan')
    traj_sp_pub.publish(ts)

def publish_velocity_hold_with_yaw_rate(timestamp: int,
                                        traj_sp_pub,
                                        rates_sp_pub,
                                        yaw_rate: float) -> None:
    # zero-velocity setpoint → holds X/Y/Z
    ts = TrajectorySetpoint(timestamp=timestamp)
    ts.velocity     = [0.0, 0.0, 0.0]
    ts.position     = [float('nan')] * 3
    ts.acceleration = [float('nan')] * 3
    ts.jerk         = [float('nan')] * 3
    ts.yaw          = float('nan')
    ts.yawspeed     = yaw_rate  # set yaw speed to desired rate
    traj_sp_pub.publish(ts)

    # # yaw-rate setpoint → spins at your chosen rate
    # vrs = VehicleRatesSetpoint(
    #     thrust_body = np.array([0.0, 0.0, float('nan')], dtype=np.float32),
    #     roll        = 0.0,
    #     pitch       = 0.0,
    #     yaw         = yaw_rate,
    #     timestamp   = timestamp
    # )
    # rates_sp_pub.publish(vrs)

# def publish_position_hold_with_yaw_rate(timestamp: int,
#                                            x_est: np.ndarray,
#                                            yaw_rate: float,
#                                            traj_sp_pub,
#                                            rates_sp_pub) -> None:
#     """
#     Hold x,y,z and command constant yaw rate.
#     - yaw_rate: radians/sec (+ = CCW looking down)
#     - traj_sp_pub:   Publisher<TrajectorySetpoint>
#     - rates_sp_pub:  Publisher<VehicleRatesSetpoint>
#     """
#     # publish hold-point TrajectorySetpoint
#     sp = TrajectorySetpoint(timestamp=timestamp)
#     sp.position[0] = x_est[0]
#     sp.position[1] = x_est[1]
#     sp.position[2] = x_est[2]
#     sp.yaw = float('nan')
#     sp.velocity = [float('nan')] * 3
#     sp.acceleration = [float('nan')] * 3
#     sp.jerk = [float('nan')] * 3
#     sp.yawspeed = float('nan')
#     traj_sp_pub.publish(sp)

#     # publish yaw-rate command
#     vrs = VehicleRatesSetpoint(
#         thrust_body=[0.0, 0.0, float('nan')].astype(np.float32),
#         roll=0.0,
#         pitch=0.0,
#         yaw=float(yaw_rate),
#         timestamp=timestamp
#     )
#     rates_sp_pub.publish(vrs)

def publish_uvr_command(timestamp:int,uvr:np.ndarray,vrs_publisher:Publisher) -> None:
    """Publish vehicle rates input (as a vehicle rates setpoint message)."""

    # Pack into message
    vrs = VehicleRatesSetpoint(
        thrust_body = np.array([0.0, 0.0, uvr[0]]).astype(np.float32),
        roll = float(uvr[1]),
        pitch = float(uvr[2]),
        yaw = float(uvr[3]),
        timestamp = timestamp
    )

    # Publish the message
    vrs_publisher.publish(vrs)

def engage_offboard_control_mode(timestamp:int,vc_publisher:Publisher) -> None:
    """Switch to offboard mode."""

    vehicle_command = VehicleCommand(
            command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,param2=6.0,param3=0.0,param4=0.0,param5=0.0,
            param6=0.0,param7=0.0,target_system=1,target_component=1,
            source_system=1,source_component=1,from_external=True,
            timestamp=timestamp
        )

    vc_publisher.publish(vehicle_command)

#NOTE: unused - requires global position estimate
def send_hold_mode(timestamp: int, vc_publisher: Publisher) -> None:
    """Switch PX4 into builtin autonomous HOLD (hover) mode."""
    vehicle_command = VehicleCommand(
        command=VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
        param1=1.0,  # MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        param2=3.0,  # PX4_CUSTOM_MAIN_MODE_POSITION
        param3=0.0, param4=0.0, param5=0.0, param6=0.0, param7=0.0,
        target_system=1, target_component=1,
        source_system=1, source_component=1, from_external=True,
        timestamp=timestamp
    )
    vc_publisher.publish(vehicle_command)

def land(timestamp:int,vc_publisher:Publisher) -> None:
    """Switch to land mode."""

    vehicle_command = VehicleCommand(
        command=VehicleCommand.VEHICLE_CMD_NAV_LAND,
        param1=0.0,param2=0.0,param3=0.0,param4=0.0,param5=0.0,
        param6=0.0,param7=0.0,target_system=1,target_component=1,
        source_system=1,source_component=1,from_external=True,
        timestamp=timestamp
    )

    vc_publisher.publish(vehicle_command)

def heartbeat_offboard_control_mode(
    timestamp: int,
    ocm_publisher: Publisher,
    *,
    body_rate: bool = False,
    velocity: bool  = False
) -> None:
    """Offboard heartbeat that can enable either body-rate or velocity control."""
    offboard_control_mode = OffboardControlMode(
        timestamp   = timestamp,
        position    = False,         
        velocity    = velocity,      # control mode for instruments-only
        acceleration= False,
        attitude    = False,
        body_rate   = body_rate,     # default control mode for Sous Vide
        actuator    = False
    )
    ocm_publisher.publish(offboard_control_mode)

# def heartbeat_offboard_control_mode(timestamp:int,ocm_publisher:Publisher) -> None:
#     """Send offboard heartbeat message."""

#     offboard_control_mode = OffboardControlMode(
#         timestamp = timestamp,
#         position = False,velocity = False,acceleration = False,
#         attitude = False,body_rate = True,actuator = False
#     )

#     ocm_publisher.publish(offboard_control_mode)