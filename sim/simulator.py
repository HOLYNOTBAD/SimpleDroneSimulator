from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import sys
import time
from itertools import cycle

import numpy as np

from models.state import UAVState, TargetState, CameraMeasurement
from sim.scheduler import MultiRateScheduler
from utils.math3d import quat_to_R

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except Exception:  # pragma: no cover
    plt = None
    Poly3DCollection = None


@dataclass(slots=True)
class TerminationConfig:
    t_final: float = 20.0
    hit_radius: float = 0.5  # meters


class Simulator:
    """
    Real-time visualizer driven by scheduler timing.

    Main purpose:
    - Receive current UAV/Target states from the main simulation loop
    - Visualize UAV via vis_uav(...)
    - Visualize target via vis_target(...)
    - Only refresh when scheduler.should_visualization(step) is true
    """

    def __init__(
        self,
        scheduler: MultiRateScheduler,
        enable: bool = True,
        t_final: float | None = None,
        enable_realtime_animation: bool = True,
        enable_offline_animation: bool = False,
        realtime: bool = True,
        enable_fov: bool = True,
        cam_width: int = 640,
        cam_height: int = 480,
        cam_fx: float = 320.0,
        cam_fy: float = 320.0,
        cam_cx: float = 320.0,
        cam_cy: float = 240.0,
        trail_len: int = 1000,
        auto_axis: bool = False,
        ned_axes: bool = True,
        map_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        map_size: tuple[float, float, float] = (160.0, 120.0, 80.0),
        colors: dict | None = None,
        uav_visual_scale: float = 1.0,
        target_marker_size: float = 6.0,
        fov_target_marker_size: float = 7.0,
        fov_target_marker_size_min: float = 4.0,
        fov_target_marker_size_max: float = 100.0,
        fov_target_diameter_m: float = 5.0,
        cache_dir: str | Path = "sim/cache",
        save_cache: bool = False,
    ):
        self.sch = scheduler
        self.realtime = bool(realtime)
        self.enable_realtime_animation = bool(enable_realtime_animation)
        self.enable_offline_animation = bool(enable_offline_animation)
        backend_is_noninteractive = False
        if plt is not None:
            backend = str(plt.get_backend()).lower()
            backend_name = backend.split(".")[-1]
            noninteractive_backends = {
                "agg",
                "cairo",
                "pdf",
                "pgf",
                "ps",
                "svg",
                "template",
                "backend_inline",
            }
            backend_is_noninteractive = (
                backend_name in noninteractive_backends or "backend_inline" in backend
            )
        runtime_available = bool(enable and (plt is not None) and (Poly3DCollection is not None) and (not backend_is_noninteractive))
        self.enable_realtime_animation = bool(runtime_available and self.enable_realtime_animation)
        self.enable_offline_animation = bool(runtime_available and self.enable_offline_animation)
        self.enable = bool(runtime_available and (self.enable_realtime_animation or self.enable_offline_animation))
        self.enable_fov = bool(enable_fov)
        self.t_final = None if t_final is None else max(float(t_final), 1e-9)
        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)
        self.cam_fx = float(cam_fx)
        self.cam_fy = float(cam_fy)
        self.cam_cx = float(cam_cx)
        self.cam_cy = float(cam_cy)
        self.trail_len = max(10, int(trail_len))
        self.auto_axis = bool(auto_axis)
        self.ned_axes = bool(ned_axes)
        self.map_center = np.asarray(map_center, dtype=float).reshape(3)
        self.map_size = np.asarray(map_size, dtype=float).reshape(3)
        self.map_size = np.maximum(self.map_size, 1.0)
        self.colors = self._build_colors(colors or {})
        self.uav_visual_scale = float(uav_visual_scale)
        self.target_marker_size = float(target_marker_size)
        self.fov_target_marker_size = float(fov_target_marker_size)
        self.fov_target_marker_size_min = float(min(fov_target_marker_size_min, fov_target_marker_size_max))
        self.fov_target_marker_size_max = float(max(fov_target_marker_size_min, fov_target_marker_size_max))
        self.fov_target_diameter_m = max(float(fov_target_diameter_m), 1e-6)
        self.cache_dir = Path(cache_dir)
        self.save_cache = bool(save_cache)

        self._u_hist: list[np.ndarray] = []
        self._t_hist: list[np.ndarray] = []
        self._u_hists: dict[str, list[np.ndarray]] = {}
        self._t_hists: dict[str, list[np.ndarray]] = {}
        self._last_vis_wall_t: float | None = None
        self._cache_frames: list[dict[str, object]] = []
        self._cache_mode: str | None = None
        self._cache_file_path: Path | None = None
        self._cache_stream = None
        self._rotor_circle_th = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        self._rotor_circle_cth = np.cos(self._rotor_circle_th)
        self._rotor_circle_sth = np.sin(self._rotor_circle_th)
        self._progress_start_wall_t: float | None = None
        self._last_progress_pct: int = -1
        self._progress_finished: bool = False

        self.fig = None
        self._layout_gs = None
        self.ax = None
        self._u_line = None
        self._t_line = None
        self._t_dot = None
        self._bbox_lines = []
        self._u_arm1 = None
        self._u_arm2 = None
        self._u_forward = None
        self._u_forward_h1 = None
        self._u_forward_h2 = None
        self._u_rotor_disks = []
        self.ax_fov = None
        self._fov_target_dot = None
        self._fov_status_text = None
        self._multi_artists: dict[str, dict[str, object]] = {}
        self._color_cycle = cycle(
            [
                (31 / 255.0, 119 / 255.0, 180 / 255.0),
                (255 / 255.0, 127 / 255.0, 14 / 255.0),
                (44 / 255.0, 160 / 255.0, 44 / 255.0),
                (214 / 255.0, 39 / 255.0, 40 / 255.0),
                (148 / 255.0, 103 / 255.0, 189 / 255.0),
                (140 / 255.0, 86 / 255.0, 75 / 255.0),
                (227 / 255.0, 119 / 255.0, 194 / 255.0),
                (127 / 255.0, 127 / 255.0, 127 / 255.0),
                (188 / 255.0, 189 / 255.0, 34 / 255.0),
                (23 / 255.0, 190 / 255.0, 207 / 255.0),
            ]
        )

        if self.enable and self.enable_offline_animation and self.save_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_file_path = self.cache_dir / f"sim_cache_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
            self._cache_stream = self._cache_file_path.open("wb")

        if self.enable and self.enable_realtime_animation:
            plt.ion()
            self._init_canvas()

    @staticmethod
    def _parse_color(value, default):
        if value is None:
            return default
        if isinstance(value, str):
            return value
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size not in (3, 4):
            raise ValueError(f"color must have 3 or 4 components, got shape {arr.shape}")
        if np.max(arr) > 1.0:
            arr = arr / 255.0
        return tuple(float(x) for x in arr)

    def _build_colors(self, cfg: dict) -> dict:
        defaults = {
            "uav_traj": (158.0 / 255.0, 22.0 / 255.0, 157.0 / 255.0, 0.55),
            "uav_body_left": (22.0 / 255.0, 5.0 / 255.0, 139.0 / 255.0),
            "uav_body_right": (98.0 / 255.0, 0.0, 170.0 / 255.0),
            "uav_arrow": (204.0 / 255.0, 74.0 / 255.0, 116.0 / 255.0),
            "target_traj": (252.0 / 255.0, 180.0 / 255.0, 49.0 / 255.0),
            "target_dot": (235.0 / 255.0, 120.0 / 255.0, 82.0 / 255.0),
            "fov_status_no_target": (181.0 / 255.0, 35.0 / 255.0, 4.0 / 255.0),
            "fov_status_has_target": "green",
        }
        colors = {k: self._parse_color(cfg.get(k), v) for k, v in defaults.items()}
        colors["fov_target"] = colors["target_dot"]
        return colors

    def _init_canvas(self) -> None:
        if self.enable_fov:
            self.fig = plt.figure(figsize=(12, 6), constrained_layout=True)
            self._layout_gs = self.fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0])
            self.ax = self.fig.add_subplot(self._layout_gs[0, 0], projection="3d")
        else:
            self.fig = plt.figure(figsize=(8, 6))
            self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m] (NED)")
        self.ax.set_title("Real-time UAV/Target Visualization (NED)")

        uav_traj_color = self.colors["uav_traj"]
        traj_alpha = 1.0 if len(uav_traj_color) < 4 else uav_traj_color[3]
        traj_rgb = uav_traj_color[:3]
        target_traj_color = self.colors["target_traj"]
        target_traj_alpha = 1.0 if len(target_traj_color) < 4 else target_traj_color[3]
        target_traj_rgb = target_traj_color[:3]
        (self._u_line,) = self.ax.plot([], [], [], lw=0.9, alpha=traj_alpha, color=traj_rgb)
        (self._t_line,) = self.ax.plot([], [], [], lw=1.0, alpha=target_traj_alpha, color=target_traj_rgb)
        (self._t_dot,) = self.ax.plot(
            [],
            [],
            [],
            marker="o",
            ms=self.target_marker_size,
            color=self.colors["target_dot"],
            linestyle="None",
        )
        (self._u_arm1,) = self.ax.plot([], [], [], color=self.colors["uav_body_left"], lw=2.0)
        (self._u_arm2,) = self.ax.plot([], [], [], color=self.colors["uav_body_right"], lw=2.0)
        (self._u_forward,) = self.ax.plot([], [], [], color=self.colors["uav_arrow"], lw=1.4)
        (self._u_forward_h1,) = self.ax.plot([], [], [], color=self.colors["uav_arrow"], lw=1.2)
        (self._u_forward_h2,) = self.ax.plot([], [], [], color=self.colors["uav_arrow"], lw=1.2)
        self._u_rotor_disks = []
        rotor_colors = [
            self.colors["uav_body_right"],
            self.colors["uav_body_left"],
            self.colors["uav_body_left"],
            self.colors["uav_body_right"],
        ]
        for rotor_color in rotor_colors:
            disk = Poly3DCollection(
                [],
                facecolors=rotor_color,
                edgecolors=rotor_color,
                linewidths=0.6,
                alpha=0.95,
            )
            self.ax.add_collection3d(disk)
            self._u_rotor_disks.append(disk)
        self._apply_fixed_bounds()
        self._init_bbox_lines()
        self._update_bbox_lines_from_axes()
        if self.enable_fov:
            self._init_fov_canvas()
        # Ensure window is created immediately so subsequent updates are visible in real time.
        plt.show(block=False)
        plt.pause(0.001)

    def _init_fov_canvas(self) -> None:
        self.ax_fov = self.fig.add_subplot(self._layout_gs[0, 1])
        self.ax_fov.set_title("Camera FOV (First-person)")
        self.ax_fov.set_xlabel("horizontal angle [deg]")
        self.ax_fov.set_ylabel("vertical angle [deg]")

        half_hfov_deg = float(np.degrees(np.arctan((0.5 * self.cam_width) / max(self.cam_fx, 1e-9))))
        half_vfov_deg = float(np.degrees(np.arctan((0.5 * self.cam_height) / max(self.cam_fy, 1e-9))))

        self.ax_fov.set_xlim(-half_hfov_deg, half_hfov_deg)
        self.ax_fov.set_ylim(half_vfov_deg, -half_vfov_deg)
        self.ax_fov.set_aspect("equal", adjustable="box")
        self.ax_fov.margins(x=0.04, y=0.04)
        self.ax_fov.grid(True, color="0.82", linewidth=0.8)

        border_x = [-half_hfov_deg, half_hfov_deg, half_hfov_deg, -half_hfov_deg, -half_hfov_deg]
        border_y = [-half_vfov_deg, -half_vfov_deg, half_vfov_deg, half_vfov_deg, -half_vfov_deg]
        self.ax_fov.plot(border_x, border_y, color="black", lw=2.8, linestyle="-", zorder=3)

        cross_half_w = 0.06 * half_hfov_deg
        cross_half_h = 0.06 * half_vfov_deg
        self.ax_fov.plot(
            [-cross_half_w, cross_half_w],
            [0.0, 0.0],
            color="red",
            lw=0.5,
            linestyle="-",
            zorder=4,
        )
        self.ax_fov.plot(
            [0.0, 0.0],
            [-cross_half_h, cross_half_h],
            color="red",
            lw=0.5,
            linestyle="-",
            zorder=4,
        )

        # Red target marker
        (self._fov_target_dot,) = self.ax_fov.plot(
            [],
            [],
            marker="o",
            markersize=self.fov_target_marker_size,
            color=self.colors["target_dot"],
            linestyle="None",
        )
        self._fov_target_dot.set_visible(False)
        self._fov_status_text = self.ax_fov.text(
            0.02,
            0.98,
            "no target",
            transform=self.ax_fov.transAxes,
            ha="left",
            va="top",
            color=self.colors["fov_status_no_target"],
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.85, boxstyle="round,pad=0.25"),
        )

    def _apply_fixed_bounds(self) -> None:
        hx, hy, hz = 0.5 * self.map_size
        cx, cy, cz = self.map_center
        self.ax.set_xlim(cx - hx, cx + hx)
        self.ax.set_ylim(cy - hy, cy + hy)
        if self.ned_axes:
            self.ax.set_zlim(cz + hz, cz - hz)  # invert for NED "Down positive"
        else:
            self.ax.set_zlim(cz - hz, cz + hz)
        self.ax.set_box_aspect((float(self.map_size[0]), float(self.map_size[1]), float(self.map_size[2])))

    def set_t_final(self, t_final: float | None) -> None:
        self.t_final = None if t_final is None else max(float(t_final), 1e-9)
        self._progress_start_wall_t = None
        self._last_progress_pct = -1
        self._progress_finished = False

    def _print_progress(self, sim_t: float) -> None:
        if self.t_final is None or self._progress_finished:
            return
        if not getattr(sys.stdout, "isatty", lambda: False)():
            return
        if self._progress_start_wall_t is None:
            self._progress_start_wall_t = time.perf_counter()

        sim_t = min(max(float(sim_t), 0.0), self.t_final)
        progress = sim_t / self.t_final
        pct = int(progress * 100.0)
        if pct == self._last_progress_pct:
            return

        bar_width = 30
        filled = min(bar_width, int(progress * bar_width))
        bar = "=" * filled + " " * (bar_width - filled)
        wall_elapsed = time.perf_counter() - self._progress_start_wall_t
        try:
            print(
                f"\r[Simulator] [{bar}] {pct:3d}% | sim {sim_t:.2f}/{self.t_final:.2f}s | wall {wall_elapsed:.2f}s",
                end="",
                flush=True,
            )
        except OSError:
            self._progress_finished = True
            return
        self._last_progress_pct = pct
        if pct >= 100:
            try:
                print(flush=True)
            except OSError:
                pass
            self._progress_finished = True

    def _cache_frame(self, mode: str, frame: dict[str, object]) -> None:
        if not self.enable_offline_animation:
            return
        if self._cache_mode is None:
            self._cache_mode = mode
        elif self._cache_mode != mode:
            raise RuntimeError(f"Simulator cache mode mismatch: expected {self._cache_mode}, got {mode}")
        self._cache_frames.append(frame)
        if self._cache_stream is not None:
            pickle.dump(frame, self._cache_stream)

    def _cleanup_cache_file(self) -> None:
        if self._cache_file_path is None:
            return
        if self.save_cache:
            print(f"[Simulator] cached animation frames: {self._cache_file_path}")
            return
        try:
            if self._cache_file_path.exists():
                self._cache_file_path.unlink()
                print(f"[Simulator] deleted cache: {self._cache_file_path}")
        except OSError as exc:
            print(f"[Simulator] failed to delete cache {self._cache_file_path}: {exc}")

    def _cache_single_frame(
        self,
        step: int,
        uav: UAVState,
        tgt: TargetState,
        cam: CameraMeasurement | None,
        has_target: bool | None,
    ) -> None:
        self._cache_frame(
            "single",
            {
                "step": int(step),
                "uav": {
                    "t": float(uav.t),
                    "p_e": np.asarray(uav.p_e, dtype=float).copy(),
                    "q_eb": np.asarray(uav.q_eb, dtype=float).copy(),
                },
                "tgt": {
                    "t": float(tgt.t),
                    "p_e": np.asarray(tgt.p_e, dtype=float).copy(),
                },
                "cam": None
                if cam is None
                else {
                    "t_meas": float(cam.t_meas),
                    "p_norm": None if cam.p_norm is None else np.asarray(cam.p_norm, dtype=float).copy(),
                    "valid": bool(cam.valid),
                    "range_m": None if cam.range_m is None else float(cam.range_m),
                },
                "has_target": None if has_target is None else bool(has_target),
            },
        )

    def _cache_multi_frame(
        self,
        step: int,
        uavs: dict[str, UAVState],
        tgts: dict[str, TargetState] | None,
        cam: CameraMeasurement | None,
        has_target: bool | None,
    ) -> None:
        frame_uavs = {
            key: {
                "t": float(uav.t),
                "p_e": np.asarray(uav.p_e, dtype=float).copy(),
                "q_eb": np.asarray(uav.q_eb, dtype=float).copy(),
            }
            for key, uav in uavs.items()
        }
        frame_tgts = None
        if tgts is not None:
            frame_tgts = {
                key: {
                    "t": float(tgt.t),
                    "p_e": np.asarray(tgt.p_e, dtype=float).copy(),
                }
                for key, tgt in tgts.items()
            }
        self._cache_frame(
            "multi",
            {
                "step": int(step),
                "uavs": frame_uavs,
                "tgts": frame_tgts,
                "cam": None
                if cam is None
                else {
                    "t_meas": float(cam.t_meas),
                    "p_norm": None if cam.p_norm is None else np.asarray(cam.p_norm, dtype=float).copy(),
                    "valid": bool(cam.valid),
                    "range_m": None if cam.range_m is None else float(cam.range_m),
                },
                "has_target": None if has_target is None else bool(has_target),
            },
        )

    def _fov_marker_size_for_apparent_angle(self, p_norm: np.ndarray | None, range_m: float | None) -> float:
        if p_norm is None or range_m is None or self.ax_fov is None or self.fig is None:
            return self.fov_target_marker_size
        range_safe = max(float(range_m), 1e-6)
        p_norm = np.asarray(p_norm, dtype=float).reshape(2)
        x_norm = float(p_norm[0])
        y_norm = float(p_norm[1])
        depth_m = range_safe / np.sqrt(1.0 + x_norm * x_norm + y_norm * y_norm)
        depth_m = max(float(depth_m), 1e-6)
        radius_norm = 0.5 * self.fov_target_diameter_m / depth_m

        x_left_deg = float(np.degrees(np.arctan(x_norm - radius_norm)))
        x_right_deg = float(np.degrees(np.arctan(x_norm + radius_norm)))
        y_bottom_deg = float(np.degrees(np.arctan(y_norm - radius_norm)))
        y_top_deg = float(np.degrees(np.arctan(y_norm + radius_norm)))
        x_center_deg = float(np.degrees(np.arctan(x_norm)))
        y_center_deg = float(np.degrees(np.arctan(y_norm)))

        p_left = self.ax_fov.transData.transform((x_left_deg, y_center_deg))
        p_right = self.ax_fov.transData.transform((x_right_deg, y_center_deg))
        p_bottom = self.ax_fov.transData.transform((x_center_deg, y_bottom_deg))
        p_top = self.ax_fov.transData.transform((x_center_deg, y_top_deg))
        diameter_px = max(
            abs(float(p_right[0] - p_left[0])),
            abs(float(p_top[1] - p_bottom[1])),
        )
        if diameter_px <= 1e-9:
            return self.fov_target_marker_size
        diameter_pt = diameter_px * 72.0 / float(self.fig.dpi)
        return float(np.clip(diameter_pt, self.fov_target_marker_size_min, self.fov_target_marker_size_max))

    def _init_bbox_lines(self) -> None:
        # 12 edges of a box (wireframe only)
        self._bbox_lines = []
        for _ in range(12):
            (line,) = self.ax.plot([], [], [], color="0.35", lw=1.0, alpha=0.9)
            self._bbox_lines.append(line)

    def _update_bbox_lines_from_axes(self) -> None:
        if not self._bbox_lines:
            return

        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        z0, z1 = self.ax.get_zlim()
        zmin, zmax = min(z0, z1), max(z0, z1)

        p000 = (xmin, ymin, zmin)
        p001 = (xmin, ymin, zmax)
        p010 = (xmin, ymax, zmin)
        p011 = (xmin, ymax, zmax)
        p100 = (xmax, ymin, zmin)
        p101 = (xmax, ymin, zmax)
        p110 = (xmax, ymax, zmin)
        p111 = (xmax, ymax, zmax)

        edges = [
            (p000, p100), (p010, p110), (p001, p101), (p011, p111),  # x-direction
            (p000, p010), (p100, p110), (p001, p011), (p101, p111),  # y-direction
            (p000, p001), (p100, p101), (p010, p011), (p110, p111),  # z-direction
        ]

        for line, (a, b) in zip(self._bbox_lines, edges):
            line.set_data([a[0], b[0]], [a[1], b[1]])
            line.set_3d_properties([a[2], b[2]])

    def _trim_hist(self) -> None:
        if len(self._u_hist) > self.trail_len:
            self._u_hist = self._u_hist[-self.trail_len :]
        if len(self._t_hist) > self.trail_len:
            self._t_hist = self._t_hist[-self.trail_len :]

    def _trim_histories(self) -> None:
        self._trim_hist()
        for key, hist in self._u_hists.items():
            if len(hist) > self.trail_len:
                self._u_hists[key] = hist[-self.trail_len :]
        for key, hist in self._t_hists.items():
            if len(hist) > self.trail_len:
                self._t_hists[key] = hist[-self.trail_len :]

    def _update_bounds(self) -> None:
        pts = []
        if self._u_hist:
            pts.append(np.vstack(self._u_hist))
        if self._t_hist:
            pts.append(np.vstack(self._t_hist))
        for hist in self._u_hists.values():
            if hist:
                pts.append(np.vstack(hist))
        for hist in self._t_hists.values():
            if hist:
                pts.append(np.vstack(hist))
        if not pts:
            return

        all_xyz = np.vstack(pts)
        mins = all_xyz.min(axis=0)
        maxs = all_xyz.max(axis=0)
        spans = np.maximum(maxs - mins, 1e-6)
        pads = 0.05 * spans

        self.ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
        self.ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
        if self.ned_axes:
            self.ax.set_zlim(maxs[2] + pads[2], mins[2] - pads[2])
        else:
            self.ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
        self.ax.set_box_aspect((float(spans[0] + 2.0 * pads[0]), float(spans[1] + 2.0 * pads[1]), float(spans[2] + 2.0 * pads[2])))

    @staticmethod
    def _set_line3d(line, p0: np.ndarray, p1: np.ndarray) -> None:
        line.set_data([float(p0[0]), float(p1[0])], [float(p0[1]), float(p1[1])])
        line.set_3d_properties([float(p0[2]), float(p1[2])])

    def _reset_visual_runtime_state(self) -> None:
        if self.fig is not None and plt is not None:
            plt.close(self.fig)
        self._u_hist = []
        self._t_hist = []
        self._u_hists = {}
        self._t_hists = {}
        self._last_vis_wall_t = None
        self.fig = None
        self._layout_gs = None
        self.ax = None
        self._u_line = None
        self._t_line = None
        self._t_dot = None
        self._bbox_lines = []
        self._u_arm1 = None
        self._u_arm2 = None
        self._u_forward = None
        self._u_forward_h1 = None
        self._u_forward_h2 = None
        self._u_rotor_disks = []
        self.ax_fov = None
        self._fov_target_dot = None
        self._fov_status_text = None
        self._multi_artists = {}

    def _prepare_replay_canvas(self) -> None:
        if not self.enable:
            return
        self._reset_visual_runtime_state()
        plt.ion()
        self._init_canvas()

    def _maybe_wait_for_next_frame(self) -> None:
        if not self.realtime:
            return
        vis_interval = self.sch.dt * self.sch.n_visualization
        now = time.perf_counter()
        if self._last_vis_wall_t is not None:
            wait_s = vis_interval - (now - self._last_vis_wall_t)
            if wait_s > 0.0:
                time.sleep(wait_s)
        self._last_vis_wall_t = time.perf_counter()

    def _render_single_frame(
        self,
        uav: UAVState,
        tgt: TargetState,
        cam: CameraMeasurement | None = None,
        has_target: bool | None = None,
    ) -> None:
        if self.fig is None:
            return
        self._maybe_wait_for_next_frame()
        self.vis_uav(uav)
        self.vis_target(tgt)
        self.vis_fov(cam, has_target=has_target)
        if self.auto_axis:
            self._update_bounds()
        else:
            self._apply_fixed_bounds()
        self._update_bbox_lines_from_axes()
        self.ax.set_title(f"Real-time UAV/Target Visualization | t={uav.t:.2f}s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.enable_fov and (self.ax_fov is not None):
            self.ax_fov.set_title(f"Camera FOV (First-person) | t={uav.t:.2f}s")
        plt.pause(0.001)

    def _render_multi_frame(
        self,
        uavs: dict[str, UAVState],
        tgts: dict[str, TargetState] | None = None,
        cam: CameraMeasurement | None = None,
        has_target: bool | None = None,
    ) -> None:
        if self.fig is None or not uavs:
            return
        self._maybe_wait_for_next_frame()
        for key, uav in uavs.items():
            artists = self._get_multi_artists(key)
            hist = self._u_hists.setdefault(key, [])
            self._vis_uav_on_artists(uav, hist, artists)
            if tgts is not None and key in tgts:
                tgt_hist = self._t_hists.setdefault(key, [])
                self._vis_target_on_artists(tgts[key], tgt_hist, artists)

        self.vis_fov(cam, has_target=has_target)
        self._trim_histories()
        if self.auto_axis:
            self._update_bounds()
        else:
            self._apply_fixed_bounds()
        self._update_bbox_lines_from_axes()

        lead_key = next(iter(uavs.keys()))
        lead_uav = uavs[lead_key]
        self.ax.set_title(f"Real-time Multi-UAV Visualization | count={len(uavs)} | t={lead_uav.t:.2f}s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self.enable_fov and (self.ax_fov is not None):
            self.ax_fov.set_title(f"Camera FOV (First-person) | t={lead_uav.t:.2f}s")
        plt.pause(0.001)

    @staticmethod
    def _frame_to_uav(frame: dict[str, object]) -> UAVState:
        return UAVState(
            t=float(frame["t"]),
            p_e=np.asarray(frame["p_e"], dtype=float),
            v_e=np.zeros(3, dtype=float),
            q_eb=np.asarray(frame["q_eb"], dtype=float),
            w_b=np.zeros(3, dtype=float),
        )

    @staticmethod
    def _frame_to_target(frame: dict[str, object]) -> TargetState:
        return TargetState(
            t=float(frame["t"]),
            p_e=np.asarray(frame["p_e"], dtype=float),
            v_e=np.zeros(3, dtype=float),
        )

    @staticmethod
    def _frame_to_camera(frame: dict[str, object] | None) -> CameraMeasurement | None:
        if frame is None:
            return None
        return CameraMeasurement(
            t_meas=float(frame["t_meas"]),
            p_norm=None if frame["p_norm"] is None else np.asarray(frame["p_norm"], dtype=float),
            uv_px=None,
            range_m=frame["range_m"],
            valid=bool(frame["valid"]),
        )

    def _replay_cached_animation(self, block: bool = True) -> None:
        if not self._cache_frames:
            return
        self._prepare_replay_canvas()
        if self._cache_mode == "single":
            for frame in self._cache_frames:
                uav = self._frame_to_uav(frame["uav"])
                tgt = self._frame_to_target(frame["tgt"])
                cam = self._frame_to_camera(frame.get("cam"))
                self._render_single_frame(uav=uav, tgt=tgt, cam=cam, has_target=frame.get("has_target"))
        elif self._cache_mode == "multi":
            for frame in self._cache_frames:
                uavs = {key: self._frame_to_uav(uav_frame) for key, uav_frame in frame["uavs"].items()}
                tgts = None
                if frame.get("tgts") is not None:
                    tgts = {key: self._frame_to_target(tgt_frame) for key, tgt_frame in frame["tgts"].items()}
                cam = self._frame_to_camera(frame.get("cam"))
                self._render_multi_frame(uavs=uavs, tgts=tgts, cam=cam, has_target=frame.get("has_target"))

        if block:
            plt.ioff()
            plt.show()
        elif self.fig is not None:
            plt.close(self.fig)

    def _make_uav_artists(self, key: str) -> dict[str, object]:
        base_color = next(self._color_cycle)
        light_color = tuple(min(1.0, c * 0.75 + 0.25) for c in base_color)
        dark_color = tuple(max(0.0, c * 0.75) for c in base_color)
        alpha = 0.7
        (u_line,) = self.ax.plot([], [], [], lw=1.0, alpha=alpha, color=base_color)
        (u_arm1,) = self.ax.plot([], [], [], color=dark_color, lw=2.0)
        (u_arm2,) = self.ax.plot([], [], [], color=light_color, lw=2.0)
        (u_forward,) = self.ax.plot([], [], [], color=base_color, lw=1.4)
        (u_forward_h1,) = self.ax.plot([], [], [], color=base_color, lw=1.2)
        (u_forward_h2,) = self.ax.plot([], [], [], color=base_color, lw=1.2)
        rotor_disks = []
        rotor_colors = [light_color, dark_color, dark_color, light_color]
        for rotor_color in rotor_colors:
            disk = Poly3DCollection(
                [],
                facecolors=rotor_color,
                edgecolors=rotor_color,
                linewidths=0.6,
                alpha=0.9,
            )
            self.ax.add_collection3d(disk)
            rotor_disks.append(disk)
        return {
            "u_line": u_line,
            "u_arm1": u_arm1,
            "u_arm2": u_arm2,
            "u_forward": u_forward,
            "u_forward_h1": u_forward_h1,
            "u_forward_h2": u_forward_h2,
            "u_rotor_disks": rotor_disks,
        }

    def _make_target_artists(self, key: str) -> dict[str, object]:
        base_color = next(self._color_cycle)
        (t_line,) = self.ax.plot([], [], [], lw=1.0, alpha=0.7, color=base_color)
        (t_dot,) = self.ax.plot([], [], [], marker="o", ms=self.target_marker_size, color=base_color, linestyle="None")
        return {
            "t_line": t_line,
            "t_dot": t_dot,
        }

    def _get_multi_artists(self, key: str) -> dict[str, object]:
        artists = self._multi_artists.get(key)
        if artists is None:
            artists = {}
            artists.update(self._make_uav_artists(key))
            artists.update(self._make_target_artists(key))
            self._multi_artists[key] = artists
        return artists

    def _vis_uav_on_artists(self, uav: UAVState, hist: list[np.ndarray], artists: dict[str, object]) -> None:
        hist.append(np.asarray(uav.p_e, dtype=float).copy())
        if len(hist) > self.trail_len:
            del hist[:-self.trail_len]

        up = np.vstack(hist)
        artists["u_line"].set_data(up[:, 0], up[:, 1])
        artists["u_line"].set_3d_properties(up[:, 2])

        p = np.asarray(uav.p_e, dtype=float).reshape(3)
        R = quat_to_R(uav.q_eb)

        arm_len = 0.9 * self.uav_visual_scale
        diag = arm_len / np.sqrt(2.0)
        arrow_len = 2.24 * arm_len
        arrow_head = 0.35 * arm_len

        b_a1 = np.array([diag, diag, 0.0], dtype=float)
        b_a2 = np.array([-diag, -diag, 0.0], dtype=float)
        b_b1 = np.array([diag, -diag, 0.0], dtype=float)
        b_b2 = np.array([-diag, diag, 0.0], dtype=float)

        e_a1 = p + R @ b_a1
        e_a2 = p + R @ b_a2
        e_b1 = p + R @ b_b1
        e_b2 = p + R @ b_b2

        self._set_line3d(artists["u_arm1"], e_a1, e_a2)
        self._set_line3d(artists["u_arm2"], e_b1, e_b2)

        rotors = np.vstack([e_a1, e_a2, e_b1, e_b2])
        rotor_r = 0.48 * arm_len
        e_x = R @ np.array([1.0, 0.0, 0.0], dtype=float)
        e_y = R @ np.array([0.0, 1.0, 0.0], dtype=float)
        for disk, c in zip(artists["u_rotor_disks"], rotors):
            verts = [
                c + rotor_r * (cth_i * e_x + sth_i * e_y)
                for cth_i, sth_i in zip(self._rotor_circle_cth, self._rotor_circle_sth)
            ]
            disk.set_verts([verts])

        b_f = np.array([1.0, 0.0, 0.0], dtype=float)
        b_r = np.array([0.0, 1.0, 0.0], dtype=float)
        e_f = R @ b_f
        e_r = R @ b_r

        tip = p + arrow_len * e_f
        self._set_line3d(artists["u_forward"], p, tip)

        h_base = tip - arrow_head * e_f
        h1 = h_base + 0.5 * arrow_head * e_r
        h2 = h_base - 0.5 * arrow_head * e_r
        self._set_line3d(artists["u_forward_h1"], tip, h1)
        self._set_line3d(artists["u_forward_h2"], tip, h2)

    def _vis_target_on_artists(self, tgt: TargetState, hist: list[np.ndarray], artists: dict[str, object]) -> None:
        hist.append(np.asarray(tgt.p_e, dtype=float).copy())
        if len(hist) > self.trail_len:
            del hist[:-self.trail_len]

        tp = np.vstack(hist)
        artists["t_line"].set_data(tp[:, 0], tp[:, 1])
        artists["t_line"].set_3d_properties(tp[:, 2])
        artists["t_dot"].set_data([tp[-1, 0]], [tp[-1, 1]])
        artists["t_dot"].set_3d_properties([tp[-1, 2]])

    def vis_uav(self, uav: UAVState) -> None:
        if not self.enable:
            return
        artists = {
            "u_line": self._u_line,
            "u_arm1": self._u_arm1,
            "u_arm2": self._u_arm2,
            "u_forward": self._u_forward,
            "u_forward_h1": self._u_forward_h1,
            "u_forward_h2": self._u_forward_h2,
            "u_rotor_disks": self._u_rotor_disks,
        }
        self._vis_uav_on_artists(uav, self._u_hist, artists)

    def vis_target(self, tgt: TargetState) -> None:
        if not self.enable:
            return
        artists = {
            "t_line": self._t_line,
            "t_dot": self._t_dot,
        }
        self._vis_target_on_artists(tgt, self._t_hist, artists)

    def update_multi(
        self,
        step: int,
        uavs: dict[str, UAVState],
        tgts: dict[str, TargetState] | None = None,
        cam: CameraMeasurement | None = None,
        has_target: bool | None = None,
    ) -> None:
        if uavs:
            lead_uav = next(iter(uavs.values()))
            self._print_progress(lead_uav.t)
        if not self.enable:
            return
        if not self.sch.should_visualization(step):
            return
        if not uavs:
            return
        self._cache_multi_frame(step=step, uavs=uavs, tgts=tgts, cam=cam, has_target=has_target)
        if not self.enable_realtime_animation:
            return
        self._render_multi_frame(uavs=uavs, tgts=tgts, cam=cam, has_target=has_target)

    def vis_fov(self, cam: CameraMeasurement | None, has_target: bool | None = None) -> None:
        if (not self.enable) or (not self.enable_fov) or (self._fov_target_dot is None):
            return
        if self._fov_status_text is not None:
            status = bool(has_target) if has_target is not None else bool(cam is not None and cam.valid)
            self._fov_status_text.set_text("has target" if status else "no target")
            self._fov_status_text.set_color(self.colors["fov_status_has_target"] if status else self.colors["fov_status_no_target"])
        if cam is None or (cam.p_norm is None):
            self._fov_target_dot.set_visible(False)
            return

        x_deg = float(np.degrees(np.arctan(cam.p_norm[0])))
        y_deg = float(np.degrees(np.arctan(cam.p_norm[1])))
        self._fov_target_dot.set_data([x_deg], [y_deg])
        self._fov_target_dot.set_markersize(self._fov_marker_size_for_apparent_angle(cam.p_norm, cam.range_m))
        self._fov_target_dot.set_visible(True)

    def update(
        self,
        step: int,
        uav: UAVState,
        tgt: TargetState,
        cam: CameraMeasurement | None = None,
        has_target: bool | None = None,
    ) -> None:
        self._print_progress(uav.t)
        if not self.enable:
            return
        if not self.sch.should_visualization(step):
            return
        self._cache_single_frame(step=step, uav=uav, tgt=tgt, cam=cam, has_target=has_target)
        if not self.enable_realtime_animation:
            return
        self._render_single_frame(uav=uav, tgt=tgt, cam=cam, has_target=has_target)

    def close(self, block: bool = True, termination_reason: str | None = None) -> None:
        if self.t_final is not None and not self._progress_finished:
            if self._last_progress_pct >= 0:
                print(flush=True)
            self._progress_finished = True
        if termination_reason is not None:
            print(f"[Simulator] terminated by event: {termination_reason}")
        if not self.enable:
            return
        if self._cache_stream is not None:
            self._cache_stream.close()
            self._cache_stream = None
        try:
            if self.enable_offline_animation and self._cache_frames:
                self._replay_cached_animation(block=block)
                return
        finally:
            self._cleanup_cache_file()
        if not self.enable_realtime_animation:
            return
        if block:
            plt.ioff()
            plt.show()
        elif self.fig is not None:
            plt.close(self.fig)
