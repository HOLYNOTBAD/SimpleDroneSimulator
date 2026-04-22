# models/target.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from models.state import TargetState


@dataclass(slots=True)
class TargetParams:
    """Simple target model parameters (L1)."""
    accel_e: np.ndarray | None = None  # constant acceleration in NED (optional)
    mode: str = "constant_velocity"
    s_amplitude_m: float = 8.0
    s_frequency_hz: float = 0.12


class TargetPointMass:
    """
    L1 target model:
    - constant_velocity / straight_escape: constant velocity, or constant acceleration if provided
    - s_curve: forward constant velocity with sinusoidal lateral motion
    """

    def __init__(self, params: TargetParams):
        self.p = params
        if self.p.accel_e is not None:
            self.p.accel_e = np.asarray(self.p.accel_e, dtype=float).reshape(3)
        self._p0: np.ndarray | None = None
        self._v0: np.ndarray | None = None
        self._t0: float | None = None

    def step(self, x: TargetState, dt: float) -> TargetState:
        mode = str(self.p.mode).lower()
        if mode in {"s", "s_curve", "s_escape"}:
            if self._p0 is None or self._v0 is None or self._t0 is None:
                self._p0 = np.asarray(x.p_e, dtype=float).reshape(3).copy()
                self._v0 = np.asarray(x.v_e, dtype=float).reshape(3).copy()
                self._t0 = float(x.t)

            t_next = x.t + dt
            tau = float(t_next - self._t0)
            amp = float(self.p.s_amplitude_m)
            omega = 2.0 * np.pi * float(self.p.s_frequency_hz)

            p_next = self._p0 + self._v0 * tau
            v_next = self._v0.copy()
            p_next[1] = self._p0[1] + amp * np.sin(omega * tau)
            v_next[1] = amp * omega * np.cos(omega * tau)
            return TargetState(t=t_next, p_e=p_next, v_e=v_next)

        if self.p.accel_e is None:
            v_next = x.v_e
        else:
            v_next = x.v_e + self.p.accel_e * dt
        p_next = x.p_e + v_next * dt
        return TargetState(t=x.t + dt, p_e=p_next, v_e=v_next)
