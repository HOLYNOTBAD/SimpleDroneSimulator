# control/ibvs_so3_controller.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from control.controller_base import ControllerBase
from models.state import ControlCommand, Observation
from utils.math3d import clamp_norm, quat_to_R


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=float)
    return v / n


def _skew(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(3)
    return np.array(
        [[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]],
        dtype=float,
    )


def _vex(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def _rodrigues(axis_unit: np.ndarray, angle: float) -> np.ndarray:
    K = _skew(axis_unit)
    I = np.eye(3, dtype=float)
    return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


@dataclass(slots=True)
class IBVSSO3ControllerParams:
    """
    Planar-Sector LOS interception controller adapted from the ICRA26 paper.

    This implementation deliberately omits the paper's DC-EKF. In simulation,
    Observation.p_r and Observation.v_r are supplied by PerfectObserver and act
    as the target-state estimate that the controller would otherwise receive
    from the estimator.

    Frames:
      - World frame is NED.
      - Body frame is FRD.
      - Camera frame has x right, y down, z forward.

    Output:
      - total thrust magnitude [N]
      - desired body angular rate [rad/s]
    """

    mass: float = 0.5
    g: float = 9.81

    # Paper outer-loop gains c1, c2, c_omega and position gain.
    c1: float = 0.8
    c2: float = 1.4
    c_omega: float = 1.0
    k_p: float = 0.25
    accel_max: float = 8.0
    vertical_pos_gain: float = 0.0
    vertical_vel_gain: float = 0.0
    vertical_control_weight: float = 0.0

    # PS-LOS sector. Horizontal is barrier-limited; vertical is driven to zero.
    alpha_lon_deg: float = 55.0
    alpha_lat_deg: float = 18.0
    safety_margin_deg: float = 2.0

    # Inner attitude tracker and actuator limits.
    k_att: float = 2.8
    omega_max: float = 6.0
    thrust_min: float = 0.0
    thrust_max: float = 40.0

    # Camera mounting must match scripts/ibvs_so3_ctrl_sim.py camera config.
    camera_mount_pitch_deg: float = -30.0
    use_mount_vertical_los_bias: bool = True
    terminal_vertical_fade_start_m: float = 25.0
    terminal_vertical_fade_end_m: float = 12.0
    terminal_pitch_limit_start_m: float = 30.0
    terminal_pitch_limit_end_m: float = 20.0
    terminal_pitch_up_max_rad_s: float = 0.35

    # Reacquisition / numerical safety.
    reacq_k_yaw: float = 2.0
    reacq_k_pitch: float = 1.6
    reacq_forward_bias: float = 0.5
    reacq_min_forward_component: float = 0.12
    reacq_img_limit: float = 3.0
    eps: float = 1e-6

    @property
    def c_h(self) -> float:
        angle = max(1.0, self.alpha_lon_deg - self.safety_margin_deg)
        return float(np.sin(np.deg2rad(angle)))

    @property
    def c_v(self) -> float:
        angle = max(1.0, self.alpha_lat_deg)
        return float(np.sin(np.deg2rad(angle)))


class IBVSSO3Controller(ControllerBase):
    """
    PS-LOS controller:
      - Eq. (13): z_h = n_hd^T n_t, z_v = n_vd^T n_t
      - Eq. (15): outer acceleration/thrust vector with barrier terms
      - Eq. (16): LOS angular-rate command
      - Eq. (17)-(21): SO(3) attitude tracking via body-rate command
    The paper's coordinated-turn moment compensation and final moment PID
    (Eq. 23)-(26) are intentionally omitted because this simulator's
    advanced-controller contract is rate + thrust.
    """

    def __init__(self, p: IBVSSO3ControllerParams):
        self.p = p
        self._Rd = np.eye(3, dtype=float)
        self._force_des_e = np.array([0.0, 0.0, -self.p.mass * self.p.g], dtype=float)
        self._omega_los_b = np.zeros(3, dtype=float)
        self._reacq_turn_sign = 1.0

    def reset(self) -> None:
        self._Rd = np.eye(3, dtype=float)
        self._force_des_e = np.array([0.0, 0.0, -self.p.mass * self.p.g], dtype=float)
        self._omega_los_b = np.zeros(3, dtype=float)
        self._reacq_turn_sign = 1.0

    def _camera_axes_in_body(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Matches sensors.camera.PinholeCamera: v_c = R_c_b0 * R_mount.T * v_b.
        pitch = np.deg2rad(float(self.p.camera_mount_pitch_deg))
        c = float(np.cos(-pitch))
        s = float(np.sin(-pitch))
        r_mount = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
        r_c_b0 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float)
        r_c_b = r_c_b0 @ r_mount.T
        r_b_c = r_c_b.T
        return r_b_c[:, 0], r_b_c[:, 1], r_b_c[:, 2]

    def _hover_thrust(self, R_e_b: np.ndarray) -> float:
        z_up = float(np.dot(R_e_b @ np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0])))
        z_up = float(np.clip(z_up, 0.2, 1.0))
        return float(np.clip(self.p.mass * self.p.g / z_up, self.p.thrust_min, self.p.thrust_max))

    def _terminal_vertical_weight(self, r: float) -> float:
        start = float(max(self.p.terminal_vertical_fade_start_m, self.p.eps))
        end = float(np.clip(self.p.terminal_vertical_fade_end_m, self.p.eps, start))
        if r >= start:
            return 1.0
        if r <= end:
            return 0.0
        s = (r - end) / max(start - end, self.p.eps)
        return float(s * s * (3.0 - 2.0 * s))

    def _terminal_pitch_up_limit(self, r: float) -> float:
        start = float(max(self.p.terminal_pitch_limit_start_m, self.p.eps))
        end = float(np.clip(self.p.terminal_pitch_limit_end_m, self.p.eps, start))
        max_up = float(np.clip(self.p.terminal_pitch_up_max_rad_s, 0.0, self.p.omega_max))
        if r >= start:
            return self.p.omega_max
        if r <= end:
            return max_up
        s = (r - end) / max(start - end, self.p.eps)
        w = float(s * s * (3.0 - 2.0 * s))
        return max_up + (self.p.omega_max - max_up) * w

    def _reacquire(self, obs: Observation, R_e_b: np.ndarray, pr_e: np.ndarray | None) -> ControlCommand:
        thrust = min(self._hover_thrust(R_e_b) + self.p.reacq_forward_bias, self.p.thrust_max)
        if pr_e is None:
            return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=np.zeros(3, dtype=float))

        dir_b = R_e_b.T @ _normalize(-pr_e, self.p.eps)
        if abs(dir_b[1]) > self.p.eps:
            self._reacq_turn_sign = float(np.sign(dir_b[1]))
        x_forward = max(float(dir_b[0]), float(self.p.reacq_min_forward_component))
        x_img = float(dir_b[1] / x_forward)
        y_img = float(dir_b[2] / x_forward)
        if dir_b[0] <= self.p.reacq_min_forward_component and abs(dir_b[1]) <= self.p.eps:
            x_img = self._reacq_turn_sign * float(self.p.reacq_img_limit)
        x_img = float(np.clip(x_img, -self.p.reacq_img_limit, self.p.reacq_img_limit))
        y_img = float(np.clip(y_img, -self.p.reacq_img_limit, self.p.reacq_img_limit))
        omega = np.array([0.0, self.p.reacq_k_pitch * y_img, self.p.reacq_k_yaw * x_img], dtype=float)
        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=clamp_norm(omega, self.p.omega_max))

    def _update_outer_loop(self, obs: Observation, R_e_b: np.ndarray, pr_e: np.ndarray, vr_e: np.ndarray) -> None:
        r = max(float(np.linalg.norm(pr_e)), self.p.eps)
        nt_e = _normalize(-pr_e, self.p.eps)

        cam_x_b, cam_y_b, _cam_z_b = self._camera_axes_in_body()
        n_hd_e = _normalize(R_e_b @ cam_x_b, self.p.eps)
        n_vd_e = _normalize(R_e_b @ cam_y_b, self.p.eps)

        z_h_raw = float(np.dot(n_hd_e, nt_e))
        z_v_raw = float(np.dot(n_vd_e, nt_e))

        ch = max(self.p.c_h, self.p.eps)
        z_h = float(np.clip(z_h_raw, -0.98 * ch, 0.98 * ch))
        kh = float(z_h / max(ch * ch - z_h * z_h, self.p.eps))

        vertical_weight = self._terminal_vertical_weight(r)
        # The mount-angle image-row bias only makes sense when the camera is
        # pitched downward relative to the aircraft. For upward-looking mounts,
        # forcing a biased vertical LOS near intercept creates the pitch-up
        # posture spike seen in the straight-escape runs.
        has_downward_mount_bias = self.p.use_mount_vertical_los_bias and (self.p.camera_mount_pitch_deg > 0.0)
        z_v_sp = (
            vertical_weight * -float(np.sin(np.deg2rad(self.p.camera_mount_pitch_deg)))
            if has_downward_mount_bias
            else 0.0
        )
        z_v = z_v_raw - z_v_sp
        kv = vertical_weight * float(z_v / max(self.p.c_v, self.p.eps))

        P = np.eye(3, dtype=float) - np.outer(nt_e, nt_e)
        z4 = vr_e + self.p.c1 * pr_e
        a_des_e = (
            -self.p.c1 * vr_e
            -self.p.c2 * z4
            -self.p.k_p * pr_e
            -kh * (P @ n_hd_e) / r
            -kv * (P @ n_vd_e) / r
        )
        a_vertical = -self.p.vertical_pos_gain * pr_e[2] - self.p.vertical_vel_gain * vr_e[2]
        w_vertical = float(np.clip(self.p.vertical_control_weight, 0.0, 1.0))
        a_des_e[2] = (1.0 - w_vertical) * a_des_e[2] + w_vertical * a_vertical
        a_des_e = clamp_norm(a_des_e, self.p.accel_max)

        g_e = np.array([0.0, 0.0, self.p.g], dtype=float)
        self._force_des_e = self.p.mass * (a_des_e - g_e)

        self._omega_los_b = self.p.c_omega * (
            R_e_b.T @ (kh * np.cross(nt_e, n_hd_e) + kv * np.cross(nt_e, n_vd_e))
        )

        nf_e = _normalize(R_e_b @ np.array([0.0, 0.0, -1.0], dtype=float), self.p.eps)
        nfd_e = _normalize(self._force_des_e, self.p.eps)
        if np.linalg.norm(nfd_e) < self.p.eps:
            self._Rd = R_e_b
            return

        axis = np.cross(nf_e, nfd_e)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < self.p.eps:
            self._Rd = R_e_b
            return

        angle = float(np.arccos(np.clip(np.dot(nf_e, nfd_e), -1.0, 1.0)))
        self._Rd = _rodrigues(axis / axis_norm, angle) @ R_e_b

    def compute(self, obs: Observation) -> ControlCommand:
        R_e_b = quat_to_R(obs.q_eb)
        if obs.p_norm is not None and abs(float(obs.p_norm[0])) > self.p.eps:
            self._reacq_turn_sign = float(np.sign(float(obs.p_norm[0])))

        pr_e = None if obs.p_r is None else np.asarray(obs.p_r, dtype=float).reshape(3)
        vr_e = None if obs.v_r is None else np.asarray(obs.v_r, dtype=float).reshape(3)
        if pr_e is None or vr_e is None:
            return self._reacquire(obs, R_e_b, pr_e)

        if (not obs.has_target) or obs.p_norm is None:
            return self._reacquire(obs, R_e_b, pr_e)

        self._update_outer_loop(obs, R_e_b, pr_e, vr_e)

        E = self._Rd.T @ R_e_b - R_e_b.T @ self._Rd
        z_att = 0.5 * _vex(E)
        omega_att_b = -self.p.k_att * z_att
        omega_cmd_b = self._omega_los_b + omega_att_b
        omega_cmd_b[1] = min(float(omega_cmd_b[1]), self._terminal_pitch_up_limit(float(np.linalg.norm(pr_e))))
        omega_cmd_b = clamp_norm(omega_cmd_b, self.p.omega_max)

        nf_e = _normalize(R_e_b @ np.array([0.0, 0.0, -1.0], dtype=float), self.p.eps)
        thrust = float(np.dot(nf_e, self._force_des_e))
        thrust = float(np.clip(thrust, self.p.thrust_min, self.p.thrust_max))
        if thrust <= self.p.eps:
            thrust = self._hover_thrust(R_e_b)

        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=omega_cmd_b)


PSLOSController = IBVSSO3Controller
PSLOSControllerParams = IBVSSO3ControllerParams
