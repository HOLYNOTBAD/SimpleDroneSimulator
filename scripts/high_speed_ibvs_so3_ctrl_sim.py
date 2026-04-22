# scripts/high_speed_ibvs_so3_ctrl_sim.py
from __future__ import annotations

from pathlib import Path
from dataclasses import fields
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError as e:
    raise SystemExit("Please install PyYAML: pip install pyyaml") from e

from models.state import ControlCommand, TargetState, UAVState
from models.rigid_body import RigidBody6DoF, RigidBodyParams
from models.motors import Motors
from models.target import TargetParams, TargetPointMass
from sensors.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from sim.scheduler import MultiRateScheduler, RateConfig
from sim.simulator import Simulator, TerminationConfig

from observe.perfect import PerfectObserver
from control.basic_control.basic_controller import BasicController
from control.high_speed_ibvs_so3_controller import (
    HighSpeedIBVSSO3Controller,
    HighSpeedIBVSSO3ControllerParams,
)
from utils.config import resolve_script_config
from utils.log import NPZLogger
from utils.metrics import Metrics, MetricsConfig


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    config_path = resolve_script_config(__file__, args.config)
    cfg = _load_cfg(config_path)

    logger_cfg = cfg.get("logging", {})
    np.random.seed(int(logger_cfg.get("seed", 0)))

    r = cfg["rates"]
    rates = RateConfig(
        physics_hz=float(r["physics_hz"]),
        control_hz=float(r["control_hz"]),
        camera_hz=float(r["camera_hz"]),
        visualization_hz=float(r.get("visualization_hz", 10.0)),
    )
    sch = MultiRateScheduler(rates)

    viz_cfg = cfg.get("visualization", {})
    cam_cfg = cfg["camera"]
    sim_viz = Simulator(
        scheduler=sch,
        enable=bool(viz_cfg.get("enable", True)),
        t_final=float(cfg.get("termination", {}).get("t_final", 20.0)),
        enable_realtime_animation=bool(viz_cfg.get("enable_realtime_animation", False)),
        enable_offline_animation=bool(viz_cfg.get("enable_offline_animation", True)),
        save_cache=bool(viz_cfg.get("save_cache", False)),
        realtime=bool(viz_cfg.get("realtime", False)),
        enable_fov=bool(viz_cfg.get("enable_fov", True)),
        cam_width=int(cam_cfg.get("width", 640)),
        cam_height=int(cam_cfg.get("height", 480)),
        cam_fx=float(cam_cfg.get("fx", 320.0)),
        cam_fy=float(cam_cfg.get("fy", 320.0)),
        cam_cx=float(cam_cfg.get("cx", 320.0)),
        cam_cy=float(cam_cfg.get("cy", 240.0)),
        trail_len=int(viz_cfg.get("trail_len", 1000)),
        auto_axis=bool(viz_cfg.get("auto_axis", False)),
        ned_axes=bool(viz_cfg.get("ned_axes", True)),
        map_center=tuple(viz_cfg.get("map_center", [0.0, 0.0, 0.0])),
        map_size=tuple(viz_cfg.get("map_size", [160.0, 120.0, 80.0])),
        colors=viz_cfg.get("colors", {}),
        uav_visual_scale=float(viz_cfg.get("uav_visual_scale", 1.0)),
        target_marker_size=float(viz_cfg.get("target_marker_size", 6.0)),
        fov_target_marker_size=float(viz_cfg.get("fov_target_marker_size", 7.0)),
        fov_target_marker_size_min=float(viz_cfg.get("fov_target_marker_size_min", 4.0)),
        fov_target_marker_size_max=float(viz_cfg.get("fov_target_marker_size_max", 18.0)),
        fov_target_diameter_m=float(viz_cfg.get("fov_target_diameter_m", 1.0)),
    )

    u0 = cfg["uav0"]
    t0 = cfg["tgt0"]
    uav = UAVState(
        t=float(u0["t"]),
        p_e=np.array(u0["p_e"], dtype=float),
        v_e=np.array(u0["v_e"], dtype=float),
        q_eb=np.array(u0["q_eb"], dtype=float),
        w_b=np.array(u0["w_b"], dtype=float),
    )
    tgt = TargetState(
        t=float(t0["t"]),
        p_e=np.array(t0["p_e"], dtype=float),
        v_e=np.array(t0["v_e"], dtype=float),
    )

    rb_params = RigidBodyParams.from_yaml(cfg["rigid_body"]["params_yaml"])
    uav_model = RigidBody6DoF(rb_params)
    motors = Motors(rb_params)
    basic_ctrl = BasicController(rb_params)

    tgt_cfg = cfg.get("target", {})
    accel = tgt_cfg.get("accel_e", None)
    tgt_model = TargetPointMass(
        TargetParams(
            accel_e=None if accel is None else np.array(accel, dtype=float),
            mode=str(tgt_cfg.get("mode", "constant_velocity")),
            s_amplitude_m=float(tgt_cfg.get("s_amplitude_m", 8.0)),
            s_frequency_hz=float(tgt_cfg.get("s_frequency_hz", 0.12)),
        )
    )

    camera = PinholeCamera(
        CameraIntrinsics(
            fx=float(cam_cfg["fx"]),
            fy=float(cam_cfg["fy"]),
            cx=float(cam_cfg["cx"]),
            cy=float(cam_cfg["cy"]),
            width=int(cam_cfg["width"]),
            height=int(cam_cfg["height"]),
        ),
        CameraExtrinsics(mount_pitch_deg=float(cam_cfg.get("mount_pitch_deg", 0.0))),
    )
    observer = PerfectObserver()

    ctrl_allowed = {f.name for f in fields(HighSpeedIBVSSO3ControllerParams)}
    ctrl_kwargs = {k: cfg["controller"][k] for k in cfg["controller"] if k in ctrl_allowed}
    controller = HighSpeedIBVSSO3Controller(HighSpeedIBVSSO3ControllerParams(**ctrl_kwargs))
    basic_ctrl.reset()
    motors.reset()

    term = cfg["termination"]
    term_cfg = TerminationConfig(t_final=float(term["t_final"]), hit_radius=float(term["hit_radius"]))
    sim_viz.set_t_final(term_cfg.t_final)

    metrics = Metrics(MetricsConfig(hit_radius=term_cfg.hit_radius))
    logger = NPZLogger(run_dir=str(logger_cfg.get("run_dir", "runs"))) if bool(logger_cfg.get("enable", True)) else None
    filename = logger_cfg.get("filename", None)

    last_cmd = ControlCommand(t=uav.t, thrust=float(rb_params.mass * rb_params.g), omega_cmd_b=np.zeros(3))
    last_i_cmd = np.zeros(rb_params.num_rotors, dtype=float)
    last_omega = np.zeros(rb_params.num_rotors, dtype=float)
    last_cam = None
    nan2 = (np.nan, np.nan)
    termination_reason = None

    steps = int(np.ceil(term_cfg.t_final / sch.dt))
    for k in range(steps):
        t_now = uav.t
        if sch.should_camera(k):
            last_cam = camera.measure(uav, tgt, t_meas=t_now)

        obs = observer.make_observation(t_now=t_now, uav=uav, cam=last_cam, tgt=tgt)

        if sch.should_control(k):
            last_cmd = controller.compute(obs)

        force_sp, motor_cmd = basic_ctrl.step_from_command(uav, last_cmd, sch.dt)
        motor_out = motors.step(motor_cmd.motor_current_cmd, sch.dt)

        uav = uav_model.step(uav, force_b=motor_out.force_b, torque_b=motor_out.torque_b, dt=sch.dt)
        tgt = tgt_model.step(tgt, sch.dt)
        last_i_cmd = motor_out.i_cmd
        last_omega = motor_out.omega
        sim_viz.update(step=k, uav=uav, tgt=tgt, cam=last_cam, has_target=obs.has_target)

        dist = metrics.update(t=t_now, uav_p=uav.p_e, tgt_p=tgt.p_e)
        if logger is not None:
            logger.push("t", t_now)
            logger.push("uav_p", uav.p_e)
            logger.push("uav_v", uav.v_e)
            logger.push("uav_q", uav.q_eb)
            logger.push("uav_w", uav.w_b)
            logger.push("tgt_p", tgt.p_e)
            logger.push("tgt_v", tgt.v_e)
            logger.push("dist", dist)
            logger.push("cam_valid", False if last_cam is None else bool(last_cam.valid))
            logger.push("cam_p_norm", nan2 if last_cam is None or last_cam.p_norm is None else last_cam.p_norm)
            logger.push("cam_uv", nan2 if last_cam is None or last_cam.uv_px is None else last_cam.uv_px)
            logger.push("cmd_thrust", last_cmd.thrust)
            logger.push("cmd_omega", last_cmd.omega_cmd_b)
            logger.push("force_sp_thrust", force_sp.thrust_sp)
            logger.push("force_sp_tau", force_sp.tau_sp_b)
            logger.push("motor_cmd", motor_cmd.motor_current_cmd)
            logger.push("motor_i_cmd", last_i_cmd)
            logger.push("motor_omega", last_omega)

        if metrics.hit:
            termination_reason = "hit"
            break

    summary = metrics.summary()
    save_path = None
    if logger is not None:
        meta = {"config": Path(config_path).as_posix(), "seed": int(logger_cfg.get("seed", 0)), "summary": summary, "rates": r}
        save_path = logger.save(meta=meta, filename=None if filename in (None, "null") else str(filename))

    del save_path
    print(f"hit: {summary['hit']}")
    if summary["t_hit"] is None:
        print("t_hit: None")
    else:
        print(f"t_hit: {float(summary['t_hit']):.3f} s")
    sim_viz.close(block=True)


if __name__ == "__main__":
    main()
