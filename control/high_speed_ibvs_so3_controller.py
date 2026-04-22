# control/high_speed_ibvs_so3_controller.py
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
class HighSpeedIBVSSO3ControllerParams:
    """
    TCST 2025 high-speed IBVS-SO3 interception controller.

    This implements the controller side of "High-Speed Interception
    Multicopter Control by Image-Based Visual Servoing" without the DKF
    observer. In this simulator, PerfectObserver supplies p_r and v_r.
    """

    mass: float = 0.5
    g: float = 9.81

    # Paper gains k1, k2 and LOS barrier kb.
    k1: float = 0.8
    k2: float = 1.4
    k_los: float = 1.5
    kb: float = 0.16

    # SO(3) attitude tracking gain added for this rate-command interface.
    k_att: float = 2.8

    # The paper assumes thrust saturation. accel_max is an adapter for this
    # simple quadrotor model so long-range scenarios do not request unrealistic
    # translational acceleration.
    accel_max: float = 8.0
    omega_max: float = 6.0
    thrust_min: float = 0.0
    thrust_max: float = 40.0

    camera_mount_pitch_deg: float = 0.0

    reacq_k_yaw: float = 2.0
    reacq_k_pitch: float = 1.6
    reacq_forward_bias: float = 0.5
    eps: float = 1e-6


class HighSpeedIBVSSO3Controller(ControllerBase):
    """
    Controller mapping:
      - Eq. (13): omega_1 from designed LOS n_td to target LOS n_t
      - Eq. (19): desired acceleration using z2 = v_r + k1 p_r
      - Eq. (21)-(23): desired force direction and thrust
      - Eq. (26),(28): SO(3) attitude-rate command omega_1 + omega_2

    Drag and target acceleration are treated as disturbances, matching the
    no-estimator deployment requested for this simulator.
    """

    def __init__(self, p: HighSpeedIBVSSO3ControllerParams):
        self.p = p
        self._Rd = np.eye(3, dtype=float)
        self._force_des_e = np.array([0.0, 0.0, -self.p.mass * self.p.g], dtype=float)
        self._omega1_b = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self._Rd = np.eye(3, dtype=float)
        self._force_des_e = np.array([0.0, 0.0, -self.p.mass * self.p.g], dtype=float)
        self._omega1_b = np.zeros(3, dtype=float)

    def _camera_axes_in_body(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pitch = np.deg2rad(float(self.p.camera_mount_pitch_deg))
        c = float(np.cos(-pitch))
        s = float(np.sin(-pitch))
        r_mount = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
        r_c_b0 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float)
        r_b_c = (r_c_b0 @ r_mount.T).T
        return r_b_c[:, 0], r_b_c[:, 1], r_b_c[:, 2]

    def _hover_thrust(self, R_e_b: np.ndarray) -> float:
        z_up = float(np.dot(R_e_b @ np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, -1.0])))
        z_up = float(np.clip(z_up, 0.2, 1.0))
        return float(np.clip(self.p.mass * self.p.g / z_up, self.p.thrust_min, self.p.thrust_max))

    def _reacquire(self, obs: Observation, R_e_b: np.ndarray, pr_e: np.ndarray | None) -> ControlCommand:
        thrust = min(self._hover_thrust(R_e_b) + self.p.reacq_forward_bias, self.p.thrust_max)
        if pr_e is None:
            return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=np.zeros(3, dtype=float))

        dir_b = R_e_b.T @ _normalize(-pr_e, self.p.eps)
        if dir_b[0] <= self.p.eps:
            spin = np.sign(dir_b[1]) if abs(dir_b[1]) > self.p.eps else 1.0
            return ControlCommand(
                t=obs.t,
                thrust=thrust,
                omega_cmd_b=np.array([0.0, 0.0, spin * self.p.omega_max], dtype=float),
            )

        x_img = float(dir_b[1] / dir_b[0])
        y_img = float(dir_b[2] / dir_b[0])
        omega = np.array([0.0, self.p.reacq_k_pitch * y_img, self.p.reacq_k_yaw * x_img], dtype=float)
        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=clamp_norm(omega, self.p.omega_max))

    def _update_outer_loop(self, R_e_b: np.ndarray, pr_e: np.ndarray, vr_e: np.ndarray) -> None:
        r = max(float(np.linalg.norm(pr_e)), self.p.eps)
        nt_e = _normalize(-pr_e, self.p.eps)

        _cam_x_b, _cam_y_b, cam_z_b = self._camera_axes_in_body()
        ntd_e = _normalize(R_e_b @ cam_z_b, self.p.eps)

        z1_raw = float(1.0 - np.dot(ntd_e, nt_e))
        kb = max(float(self.p.kb), self.p.eps)
        z1 = float(np.clip(z1_raw, -0.98 * kb, 0.98 * kb))
        barrier = float(z1 / max(kb * kb - z1 * z1, self.p.eps))

        self._omega1_b = self.p.k_los * barrier * (R_e_b.T @ np.cross(ntd_e, nt_e))

        z2 = vr_e + self.p.k1 * pr_e
        P = -np.eye(3, dtype=float) + np.outer(nt_e, nt_e)
        a_des_e = (
            -self.p.k1 * vr_e
            -self.p.k2 * z2
            -pr_e
            + self.p.k_los * barrier * (P @ ntd_e) / r
        )
        a_des_e = clamp_norm(a_des_e, self.p.accel_max)

        g_e = np.array([0.0, 0.0, self.p.g], dtype=float)
        self._force_des_e = self.p.mass * (a_des_e - g_e)

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

        pr_e = None if obs.p_r is None else np.asarray(obs.p_r, dtype=float).reshape(3)
        vr_e = None if obs.v_r is None else np.asarray(obs.v_r, dtype=float).reshape(3)
        if pr_e is None or vr_e is None:
            return self._reacquire(obs, R_e_b, pr_e)

        if (not obs.has_target) or obs.p_norm is None:
            return self._reacquire(obs, R_e_b, pr_e)

        self._update_outer_loop(R_e_b, pr_e, vr_e)

        E = self._Rd.T @ R_e_b - R_e_b.T @ self._Rd
        omega2_b = -self.p.k_att * _vex(E)
        omega_cmd_b = clamp_norm(self._omega1_b + omega2_b, self.p.omega_max)

        nf_e = _normalize(R_e_b @ np.array([0.0, 0.0, -1.0], dtype=float), self.p.eps)
        thrust = float(np.dot(nf_e, self._force_des_e))
        thrust = float(np.clip(thrust, self.p.thrust_min, self.p.thrust_max))
        if thrust <= self.p.eps:
            thrust = self._hover_thrust(R_e_b)

        return ControlCommand(t=obs.t, thrust=thrust, omega_cmd_b=omega_cmd_b)


TCSTIBVSSO3Controller = HighSpeedIBVSSO3Controller
TCSTIBVSSO3ControllerParams = HighSpeedIBVSSO3ControllerParams
