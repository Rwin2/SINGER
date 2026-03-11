"""
expert_controllers.py
─────────────────────
Two state-conditioned DAgger expert controllers that replace the original
VehicleRateMPC "recovery-to-reference" expert with true goal-seeking experts:

  • PotentialFieldExpert  – attractive/repulsive potential field → geometric
                            controller.  Fast (no planning, no ACADOS).
  • OnlineRRTExpert       – RRT* replanning at segment boundaries → pure-pursuit
                            waypoint follower → geometric controller.  Gives
                            obstacle-aware, globally consistent paths.

Both expose the same interface as VehicleRateMPC:
    control(tcr, xcr, upr, obj, icr, zcr) → (u, None, None, tsol)

Control vector convention (mirrors vrmpc_rrt.json bounds):
    u[0] ∈ [-1.0,  0.0]  normalised specific thrust  (more negative = more thrust)
    u[1] ∈ [-5.0,  5.0]  body roll  rate  wx  (rad/s)
    u[2] ∈ [-5.0,  5.0]  body pitch rate  wy  (rad/s)
    u[3] ∈ [-5.0,  5.0]  body yaw   rate  wz  (rad/s)
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Shared geometric controller mixin
# ──────────────────────────────────────────────────────────────────────────────

class _GeometricMixin:
    """
    Shared yaw + altitude heading controller used by both expert classes.

    Parameters (set as instance attributes by subclasses):
        goal       : (3,) target world-frame position [x, y, z]
        k_yaw      : yaw-rate proportional gain
        k_pitch    : forward-pitch proportional gain
        k_alt      : altitude proportional gain
        k_alt_vel  : altitude velocity damping gain
    """

    def _to_control(
        self,
        pos: np.ndarray,            # (3,) current position
        vel: np.ndarray,            # (3,) current velocity
        quat: np.ndarray,           # (4,) [qx, qy, qz, qw]
        desired_dir: np.ndarray,    # (3,) desired direction (unit or unnormalised)
    ) -> np.ndarray:
        """
        Map a desired 3-D direction to [thrust, wx, wy, wz].
        The direction points from the current position toward where we want to go.
        """
        qx, qy, qz, qw = quat

        # ── current yaw ──────────────────────────────────────────────────────
        yaw = float(np.arctan2(2*(qw*qz + qx*qy),
                               1 - 2*(qy**2 + qz**2)))

        # ── desired yaw: face horizontal component of desired direction ───────
        dx, dy = float(desired_dir[0]), float(desired_dir[1])
        if abs(dx) + abs(dy) > 1e-3:
            yaw_des = float(np.arctan2(dy, dx))
        else:
            yaw_des = yaw  # already at goal XY → hold heading

        e_yaw = float(np.arctan2(np.sin(yaw_des - yaw), np.cos(yaw_des - yaw)))

        # ── yaw rate ─────────────────────────────────────────────────────────
        wz = float(np.clip(self.k_yaw * e_yaw, -5.0, 5.0))

        # ── pitch rate: tilt forward when aligned with goal direction ─────────
        # d_xy: how much horizontal movement is needed
        d_xy = float(np.linalg.norm(desired_dir[:2]))
        # project onto aligned axis (only tilt when roughly facing the goal)
        forward_drive = float(np.cos(e_yaw)) * d_xy
        wy = float(np.clip(self.k_pitch * forward_drive, -5.0, 5.0))

        # ── no roll for heading-based navigation ──────────────────────────────
        wx = 0.0

        # ── altitude: proportional + velocity damping ─────────────────────────
        # z-convention check: in the scene, altitude = -1.0 with bounds z∈[-2,0]
        # → z increases upward; hover at z = -1.0.
        # u[0] convention: -0.5 ≈ hover, more-negative → more thrust (climb).
        alt_err = float(self.goal[2] - pos[2])
        vz      = float(vel[2])
        thrust  = float(np.clip(
            -0.5 - self.k_alt * alt_err - self.k_alt_vel * vz,
            -1.0, 0.0,
        ))

        return np.array([thrust, wx, wy, wz], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Potential Field Expert
# ──────────────────────────────────────────────────────────────────────────────

class PotentialFieldExpert(_GeometricMixin):
    """
    Goal-seeking + obstacle-avoiding expert using attractive/repulsive potential
    fields.  Fully state-conditioned: answers "what should I do from HERE?" at
    every single control step.

    Parameters
    ----------
    goal        : (3,) world-frame target position
    point_cloud : (N, 3) obstacle point cloud (epcds_arr from scene cache)
    k_att       : attraction gain toward goal
    k_rep       : repulsion gain from obstacles
    d0_rep      : repulsion influence radius (m) — beyond this, obstacles ignored
    k_vel       : velocity damping gain in attractive term
    k_yaw       : yaw rate gain
    k_pitch     : forward pitch gain
    k_alt       : altitude proportional gain
    k_alt_vel   : altitude velocity damping gain
    """

    def __init__(
        self,
        goal:        np.ndarray,
        point_cloud: Optional[np.ndarray],
        k_att:     float = 2.5,
        k_rep:     float = 0.8,
        d0_rep:    float = 1.2,
        k_vel:     float = 0.8,
        k_yaw:     float = 3.0,
        k_pitch:   float = 1.5,
        k_alt:     float = 1.0,
        k_alt_vel: float = 0.3,
    ) -> None:
        self.goal      = np.array(goal[:3], dtype=float)
        self.hz        = 20
        self.nzcr      = None
        self.k_att     = k_att
        self.k_rep     = k_rep
        self.d0        = d0_rep
        self.k_vel     = k_vel
        self.k_yaw     = k_yaw
        self.k_pitch   = k_pitch
        self.k_alt     = k_alt
        self.k_alt_vel = k_alt_vel

        # Build KD-tree once for fast nearest-obstacle queries
        self._pcd: Optional[np.ndarray] = None
        self._kd  = None
        if point_cloud is not None and len(point_cloud) > 0:
            from scipy.spatial import cKDTree
            pcd = np.asarray(point_cloud, dtype=float)
            if pcd.ndim == 2 and pcd.shape[0] == 3 and pcd.shape[1] != 3:
                pcd = pcd.T            # normalise to (N, 3)
            self._pcd = pcd
            self._kd  = cKDTree(pcd)

    # ------------------------------------------------------------------
    def control(self, tcr, xcr, upr=None, obj=None, icr=None, zcr=None):
        pos  = xcr[0:3].copy()
        vel  = xcr[3:6].copy()
        quat = xcr[6:10].copy()

        d   = self._potential_dir(pos, vel)
        u   = self._to_control(pos, vel, quat, d)
        return u, None, None, np.array([0.001, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    def _potential_dir(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Return the desired 3-D movement direction (not necessarily unit)."""
        e    = self.goal - pos
        dist = float(np.linalg.norm(e))

        # Attractive: pull toward goal (saturated at 5 m)
        f_att  = self.k_att * e / max(dist, 1e-6) * min(dist, 5.0)
        f_att -= self.k_vel * vel          # velocity damping

        # Repulsive: push away from nearby obstacle points
        f_rep = np.zeros(3)
        if self._kd is not None:
            k_query = min(60, len(self._pcd))
            dists, idxs = self._kd.query(pos, k=k_query)
            dists = np.atleast_1d(dists)
            idxs  = np.atleast_1d(idxs)
            mask  = dists < self.d0
            if mask.any():
                d_near   = dists[mask]
                dirs_raw = pos - self._pcd[idxs[mask]]          # away from obstacles
                norms    = np.linalg.norm(dirs_raw, axis=1, keepdims=True) + 1e-8
                dirs_n   = dirs_raw / norms
                weights  = self.k_rep * (1.0 / d_near - 1.0 / self.d0) / (d_near ** 2)
                f_rep    = np.clip(
                    np.sum(weights[:, None] * dirs_n, axis=0), -6.0, 6.0
                )

        return f_att + f_rep


# ──────────────────────────────────────────────────────────────────────────────
# Online RRT Expert
# ──────────────────────────────────────────────────────────────────────────────

class OnlineRRTExpert(_GeometricMixin):
    """
    Goal-seeking expert that replans an RRT* path at the start of every DAgger
    segment (triggered by replan_interval or on first call), then tracks the
    resulting waypoints with a pure-pursuit waypoint follower feeding into the
    same geometric controller as PotentialFieldExpert.

    This avoids re-creating VehicleRateMPC / ACADOS on every replan (which would
    take 10–30 s per replan) while still providing globally obstacle-aware paths.

    Parameters
    ----------
    goal            : (3,) world-frame target position
    point_cloud     : (N, 3) obstacle point cloud (epcds_arr from scene cache)
    scene_cfg       : scene YAML dict (altitudes, minbound, maxbound, radii, …)
    obj_idx         : object index in scene (selects altitude and radii lists)
    replan_interval : seconds between RRT* replans (default = segment length = 2 s)
    lookahead_dist  : pure-pursuit lookahead distance (m)
    speed           : desired cruise speed (m/s) – affects path timing only
    k_yaw / k_pitch / k_alt / k_alt_vel : geometric controller gains
    """

    def __init__(
        self,
        goal:             np.ndarray,
        point_cloud:      Optional[np.ndarray],
        scene_cfg:        dict,
        obj_idx:          int          = 0,
        replan_interval:  float        = 2.0,
        lookahead_dist:   float        = 1.5,
        speed:            float        = 1.0,
        k_yaw:            float        = 3.0,
        k_pitch:          float        = 1.5,
        k_alt:            float        = 1.0,
        k_alt_vel:        float        = 0.3,
    ) -> None:
        self.goal            = np.array(goal[:3], dtype=float)
        self.hz              = 20
        self.nzcr            = None
        self.k_yaw           = k_yaw
        self.k_pitch         = k_pitch
        self.k_alt           = k_alt
        self.k_alt_vel       = k_alt_vel
        self.replan_interval = replan_interval
        self.lookahead_dist  = lookahead_dist
        self.speed           = speed

        # Point cloud for RRT collision checking
        if point_cloud is not None and len(point_cloud) > 0:
            pcd = np.asarray(point_cloud, dtype=float)
            if pcd.ndim == 2 and pcd.shape[0] == 3 and pcd.shape[1] != 3:
                pcd = pcd.T
            self._pcd_arr = pcd
        else:
            self._pcd_arr = np.zeros((0, 3))

        # Per-object scene parameters
        altitudes  = scene_cfg.get("altitudes", [-1.0])
        radii_list = scene_cfg.get("radii",     [[2.0, 0.4]])
        idx        = min(obj_idx, len(altitudes) - 1)
        self._altitude = float(altitudes[idx])
        r_pair         = radii_list[min(idx, len(radii_list) - 1)]
        self._r_goal   = float(r_pair[0])
        self._r_coll   = float(r_pair[1])
        self._bounds   = [
            (float(scene_cfg.get("minbound", [-10, -10, -2])[0]),
             float(scene_cfg.get("maxbound", [ 10,  10,  0])[0])),
            (float(scene_cfg.get("minbound", [-10, -10, -2])[1]),
             float(scene_cfg.get("maxbound", [ 10,  10,  0])[1])),
        ]
        self._step_size = float(scene_cfg.get("step_size", 1.0))

        # Internal state
        self._waypoints: Optional[np.ndarray] = None   # (M, 3) world-frame 3D path
        self._last_replan_t: float = -replan_interval  # force replan on first call

    # ------------------------------------------------------------------
    def control(self, tcr, xcr, upr=None, obj=None, icr=None, zcr=None):
        pos  = xcr[0:3].copy()
        vel  = xcr[3:6].copy()
        quat = xcr[6:10].copy()

        # Replan if due (every replan_interval seconds = every DAgger segment)
        if (tcr - self._last_replan_t) >= self.replan_interval or self._waypoints is None:
            self._replan(tcr, pos)

        # Pure-pursuit: get lookahead point on current path
        target = self._lookahead_point(pos)

        # Desired direction toward lookahead (+ altitude from goal)
        desired = target - pos          # 3-D, not unit-normalised
        u       = self._to_control(pos, vel, quat, desired)
        return u, None, None, np.array([0.001, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    def _replan(self, tcr: float, pos: np.ndarray) -> None:
        """Run RRT* from current XY position to goal XY, store 3-D waypoints."""
        t0_plan = time.time()
        try:
            from figs.tsampling.rrt_datagen_v10 import RRT
            rrt = RRT(
                env_arr                    = self._pcd_arr,
                env_pts                    = None,      # KDTree built from env_arr
                start                      = pos[:2].copy(),
                obj                        = self.goal[:2].copy(),
                bounds                     = self._bounds,
                altitude                   = float(pos[2]) if abs(pos[2]) > 0.05
                                             else self._altitude,
                dimension                  = 2,
                algorithm                  = "RRT*",
                step_size                  = self._step_size,
                collision_check_radius     = self._r_coll,
                goal_exclusion_radius      = self._r_goal,
                collision_check_resolution = 0.1,
                max_iter                   = 400,
                exact_step                 = False,
                bounded_step               = True,
                prevent_edge_overlap       = True,
            )
            rrt.build_rrt()
            path_2d = self._extract_path_2d(rrt)
            if path_2d is not None and len(path_2d) >= 2:
                alt     = float(pos[2]) if abs(pos[2]) > 0.05 else self._altitude
                wps_3d  = np.array([[p[0], p[1], alt] for p in path_2d])
                # Append the actual 3-D goal as the last waypoint
                wps_3d  = np.vstack([wps_3d, self.goal[None, :]])
                self._waypoints = wps_3d
                print(f"  [OnlineRRT] replanned t={tcr:.1f}s  "
                      f"path={len(wps_3d)} pts  "
                      f"({time.time()-t0_plan:.1f}s)")
            else:
                raise ValueError("empty path from RRT*")
        except Exception as exc:
            print(f"  [OnlineRRT] replan FAILED ({exc})  → straight-line fallback")
            alt = float(pos[2]) if abs(pos[2]) > 0.05 else self._altitude
            self._waypoints = np.array([
                [pos[0], pos[1], alt],
                [self.goal[0], self.goal[1], alt],
                self.goal.copy(),
            ])
        finally:
            self._last_replan_t = tcr

    # ------------------------------------------------------------------
    def _lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        """
        Pure-pursuit lookahead: find the first waypoint that is at least
        lookahead_dist ahead of current position (measured along the path).
        Falls back to the last waypoint if none is found.
        """
        if self._waypoints is None or len(self._waypoints) == 0:
            return self.goal.copy()

        # Find nearest waypoint index
        dists  = np.linalg.norm(self._waypoints - pos, axis=1)
        i_near = int(np.argmin(dists))

        # Walk forward until lookahead_dist is reached
        for i in range(i_near, len(self._waypoints)):
            if np.linalg.norm(self._waypoints[i] - pos) >= self.lookahead_dist:
                return self._waypoints[i].copy()

        # Lookahead overshoots: return final waypoint (the goal)
        return self._waypoints[-1].copy()

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_path_2d(rrt) -> Optional[List[list]]:
        """Return 2-D waypoint list from RRT root to nearest-to-goal leaf."""
        goal_pos = rrt.goal_node.position
        leaves   = [n for n in rrt.nodes if not n.children]
        if not leaves:
            return None
        best = min(leaves, key=lambda n: np.linalg.norm(n.position - goal_pos))
        path, node = [], best
        while node is not None:
            path.append(node.position.tolist())
            node = node.parent
        path.reverse()
        return path
