import math
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import os
import matplotlib

# Select an interactive backend if possible before importing pyplot
def _configure_backend() -> None:
    if os.environ.get("MPLBACKEND"):
        return
    for backend in ("Qt5Agg", "TkAgg", "GTK3Agg"):
        try:
            matplotlib.use(backend, force=True)
            return
        except Exception:
            continue

_configure_backend()

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from matplotlib.widgets import Button


@dataclass
class DWAParameters:
    predict_time: float = 7.0
    dt: float = 0.1
    min_speed: float = 0.0
    max_speed: float = 0.3
    max_omega: float = 0.5
    vel_acc: float = 0.5
    omega_acc: float = 0.4
    vx_samples: int = 5
    omega_samples: int = 10
    path_distance_bias: float = 2.0
    goal_distance_bias: float = 5
    occdist_scale: float = 200.0
    speed_ref_bias: float = 5
    away_bias: float = 20.0
    ref_velocity: float = 0.3
    robot_radius: float = 0.5
    lookahead_distance: float = 1
    preview_switch_obst_cost_thresh: float = 10.0
    path_point_spacing: float = 0.2
    obstacle_clearance_margin: float = 0.05


@dataclass
class WorldConfig:
    obstacle_center: Tuple[float, float] = (2.0, 0.0)
    obstacle_side: float = 0.2  # square, side length
    inflation_radius: float = 0.7  # additional inflation beyond the square half-size
    cost_decay_scale: float = 0.3  # for cost falloff outside inflated obstacle
    goal_x: float = 5.0
    goal_y: float = 0.0
    x_limits: Tuple[float, float] = (-1.0, 7.0)
    y_limits: Tuple[float, float] = (-3.0, 3.0)


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0


def compute_dynamic_window(current_v: float, current_omega: float, p: DWAParameters) -> Tuple[float, float, float, float]:
    vs_min = p.min_speed
    vs_max = p.max_speed
    vo_min = -p.max_omega
    vo_max = p.max_omega
    vd_v_min = current_v - p.vel_acc * p.dt
    vd_v_max = current_v + p.vel_acc * p.dt
    vd_o_min = current_omega - p.omega_acc * p.dt
    vd_o_max = current_omega + p.omega_acc * p.dt
    v_min = max(vs_min, vd_v_min)
    v_max = min(vs_max, vd_v_max)
    o_min = max(vo_min, vd_o_min)
    o_max = min(vo_max, vd_o_max)
    return v_min, v_max, o_min, o_max


def simulate_trajectory(x: float, y: float, theta: float, v_cmd: float, omega_cmd: float, p: DWAParameters) -> np.ndarray:
    steps = int(p.predict_time / p.dt)
    traj = np.zeros((steps + 1, 3), dtype=float)
    traj[0] = np.array([x, y, theta], dtype=float)
    for k in range(steps):
        x = x + v_cmd * math.cos(theta) * p.dt
        y = y + v_cmd * math.sin(theta) * p.dt
        theta = theta + omega_cmd * p.dt
        traj[k + 1] = np.array([x, y, theta], dtype=float)
    return traj


def signed_distance_to_inflated_square(px: float, py: float, world: WorldConfig) -> float:
    cx, cy = world.obstacle_center
    half_size = world.obstacle_side / 2.0
    inflated_half = half_size + world.inflation_radius
    dx = abs(px - cx) - inflated_half
    dy = abs(py - cy) - inflated_half
    outside_dx = max(dx, 0.0)
    outside_dy = max(dy, 0.0)
    outside_distance = math.hypot(outside_dx, outside_dy)
    inside_distance = min(max(dx, dy), 0.0)
    return outside_distance + inside_distance  # negative if inside


def normalized_cost_from_distance(distance: float, world: WorldConfig) -> float:
    if distance <= 0.0:
        return 1.0
    return min(0.99, math.exp(-distance / world.cost_decay_scale))


def path_cost(traj: np.ndarray) -> float:
    return abs(traj[-1, 1])


def lookahead_cost(traj: np.ndarray, world: WorldConfig) -> float:
    dx = traj[-1, 0] - world.goal_x
    dy = traj[-1, 1] - world.goal_y
    return math.hypot(dx, dy)


def speed_ref_cost(v_cmd: float, p: DWAParameters) -> float:
    return abs(v_cmd - p.ref_velocity)


def obstacle_cost(traj: np.ndarray, world: WorldConfig) -> float:
    cost_sum = 0.0
    for px, py, _ in traj:
        d = signed_distance_to_inflated_square(px, py, world)
        c_norm = normalized_cost_from_distance(d, world)
        cost_sum += c_norm
    return cost_sum / max(1, int(traj.shape[0]))


def away_from_obstacle_cost(traj: np.ndarray, world: WorldConfig) -> float:
    max_exp = 0.0
    for px, py, _ in traj:
        d = signed_distance_to_inflated_square(px, py, world)
        c_norm = normalized_cost_from_distance(d, world)
        exp_cost = math.exp(5.0 * c_norm)
        if exp_cost > max_exp:
            max_exp = exp_cost
    return max_exp


def evaluate_samples(state: RobotState, p: DWAParameters, world: WorldConfig) -> Tuple[List[Dict], Dict]:
    v_min, v_max, o_min, o_max = compute_dynamic_window(state.v, state.omega, p)
    v_den = max(1, p.vx_samples - 1)
    o_den = max(1, p.omega_samples - 1)

    results: List[Dict] = []
    best = {"total": float("inf")}
    max_obs_cost = 0.0
    for i in range(p.vx_samples):
        v_cmd = v_min + (v_max - v_min) * (i / v_den)
        for j in range(p.omega_samples):
            omega_cmd = o_min + (o_max - o_min) * (j / o_den)
            traj = simulate_trajectory(state.x, state.y, state.theta, v_cmd, omega_cmd, p)

            c_path = path_cost(traj)
            c_look = lookahead_cost(traj, world)
            c_speed = speed_ref_cost(v_cmd, p)
            c_obs = obstacle_cost(traj, world)
            c_away = away_from_obstacle_cost(traj, world)

            if math.isinf(c_obs):
                total = float("inf")
            else:
                total = (p.path_distance_bias * c_path
                         + p.goal_distance_bias * c_look
                         + p.occdist_scale * c_obs
                         + p.speed_ref_bias * c_speed
                         + p.away_bias * c_away)

            record = {
                "i": i, "j": j,
                "v": v_cmd, "omega": omega_cmd,
                "traj": traj,
                "path_cost": p.path_distance_bias * c_path,
                "lookahead_cost": p.goal_distance_bias * c_look,
                "speed_ref_cost": p.speed_ref_bias * c_speed,
                "obstacle_cost": p.occdist_scale * c_obs,
                "away_cost": p.away_bias * c_away,
                "total": total,
            }
            results.append(record)
            if total < best.get("total", float("inf")):
                best = record
            
            if c_obs*p.occdist_scale > max_obs_cost:
                max_obs_cost = c_obs*p.occdist_scale

    return results, best, max_obs_cost


class DWAVisualizer:
    def __init__(self, params: DWAParameters, world: WorldConfig, init_state: RobotState):
        self.params = params
        self.world = world
        self.state = init_state
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.18)
        self.step_button_ax = self.fig.add_axes([0.08, 0.05, 0.2, 0.08])
        self.step_button = Button(self.step_button_ax, "Step (0.1s)")
        self.step_button.on_clicked(self.on_step)
        self.play_button_ax = self.fig.add_axes([0.33, 0.05, 0.2, 0.08])
        self.play_button = Button(self.play_button_ax, "Play")
        self.play_button.on_clicked(self.on_toggle_play)
        self.save_button_ax = self.fig.add_axes([0.58, 0.05, 0.3, 0.08])
        self.save_button = Button(self.save_button_ax, "Save MP4")
        self.save_button.on_clicked(self.on_save_mp4)
        self.play_button_ax = self.fig.add_axes([0.35, 0.05, 0.2, 0.08])
        self.play_button = Button(self.play_button_ax, "Play")
        self.play_button.on_clicked(self.on_toggle_play)

        self.info_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                      va="top", ha="left", fontsize=9,
                                      bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.trajectory_lines = []
        self.best_line = None
        self.robot_artist = None
        self.heading_artist = None
        self.obstacle_patch = None
        self.inflated_patch = None
        self.goal_artist = None
        self.path_line = None
        self.lookahead_artist = None
        self.preview_line = None
        self.active_controller = "DWA"
        self.playing = False
        self.timer = self.fig.canvas.new_timer(interval=int(self.params.dt * 1000))
        self.timer.add_callback(self._timer_tick)
        self.playing = False
        self.timer = self.fig.canvas.new_timer(interval=int(self.params.dt * 1000))
        self.timer.add_callback(self.advance_one_step)
        self.look_idx = 0

        self.path = self.generate_straight_path((0.0, 0.0), (self.world.goal_x, self.world.goal_y), self.params.path_point_spacing)

        self.draw_static()
        self.redraw()

    def draw_static(self):
        self.ax.set_xlim(self.world.x_limits)
        self.ax.set_ylim(self.world.y_limits)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.3)

        cx, cy = self.world.obstacle_center
        side = self.world.obstacle_side
        half = side / 2.0

        self.obstacle_patch = Rectangle((cx - half, cy - half), side, side,
                                        facecolor="black", edgecolor="black", alpha=0.6, zorder=2)
        self.ax.add_patch(self.obstacle_patch)

        inflated_half = half + self.world.inflation_radius
        inflated_side = 2.0 * inflated_half
        self.inflated_patch = Rectangle((cx - inflated_half, cy - inflated_half),
                                        inflated_side, inflated_side,
                                        facecolor="none", edgecolor="red", linestyle="--", alpha=0.8, zorder=3)
        self.ax.add_patch(self.inflated_patch)

        self.goal_artist = self.ax.plot([self.world.goal_x], [self.world.goal_y],
                                        marker="*", color="green", markersize=12, zorder=4, label="Goal")[0]
        self.ax.legend(loc="upper right")

        if self.path:
            px = [p[0] for p in self.path]
            py = [p[1] for p in self.path]
            self.path_line, = self.ax.plot(px, py, color="#888", linestyle="-", linewidth=1.5, alpha=0.7, zorder=0)

    def redraw(self):
        for ln in self.trajectory_lines:
            ln.remove()
        self.trajectory_lines.clear()
        if self.best_line is not None:
            self.best_line.remove()
            self.best_line = None
        if self.robot_artist is not None:
            self.robot_artist.remove()
            self.robot_artist = None
        if self.heading_artist is not None:
            self.heading_artist.remove()
            self.heading_artist = None
        if self.lookahead_artist is not None:
            self.lookahead_artist.remove()
            self.lookahead_artist = None
        if self.preview_line is not None:
            self.preview_line.remove()
            self.preview_line = None

        results, best, max_obs_cost = evaluate_samples(self.state, self.params, self.world)

        for rec in results:
            traj = rec["traj"]
            color = "#aaaaaa"
            if math.isinf(rec["obstacle_cost"]):
                color = "#444444"
            ln, = self.ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.0, alpha=0.8, zorder=1)
            self.trajectory_lines.append(ln)

        if best and np.isfinite(best["total"]):
            traj = best["traj"]
            self.best_line, = self.ax.plot(traj[:, 0], traj[:, 1], color="crimson", linewidth=2.5, zorder=5)

        look_pt, preview_v, preview_omega = self.compute_preview_command(self.state, advance=False)
        prev_traj = simulate_trajectory(self.state.x, self.state.y, self.state.theta, preview_v, preview_omega, self.params)
        self.preview_line, = self.ax.plot(prev_traj[:, 0], prev_traj[:, 1], color="#2a9d8f", linestyle=":", linewidth=1.8, zorder=4)
        if look_pt is not None:
            self.lookahead_artist = self.ax.plot([look_pt[0]], [look_pt[1]], marker="o", color="#2a9d8f", markersize=6, zorder=6)[0]

        robot_circle = Circle((self.state.x, self.state.y), radius=self.params.robot_radius,
                              facecolor="royalblue", edgecolor="navy", alpha=0.6, zorder=6)
        self.ax.add_patch(robot_circle)
        self.robot_artist = robot_circle

        arrow_length = max(0.6 * self.params.robot_radius, 0.15)
        hx = self.state.x + arrow_length * math.cos(self.state.theta)
        hy = self.state.y + arrow_length * math.sin(self.state.theta)
        self.heading_artist = self.ax.arrow(self.state.x, self.state.y,
                                            hx - self.state.x, hy - self.state.y,
                                            head_width=0.02, head_length=0.02, fc="navy", ec="navy", zorder=7)

        self.info_text.set_text(self.format_info_text(best))
        self.fig.canvas.draw_idle()

    def format_info_text(self, best: Dict) -> str:
        if not best or not np.isfinite(best.get("total", float("inf"))):
            return f"Controller: {self.active_controller}\nBest: none (all collided)\n"
        return (
            f"Controller: {self.active_controller}\n"
            f"Best v={best['v']:.3f} m/s, ω={best['omega']:.3f} rad/s\n"
            f"path={best['path_cost']:.3f}, look={best['lookahead_cost']:.3f}, "
            f"obs={best['obstacle_cost']:.3f}, speed={best['speed_ref_cost']:.3f}, away={best['away_cost']:.3f}\n"
            f"total={best['total']:.3f}"
        )

    def print_cost_table(self, results: List[Dict]):
        rows = []
        header = ("i", "j", "v", "omega", "path", "look", "obs", "speed", "away", "total")
        rows.append("{:>2} {:>2} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9}".format(*header))
        for rec in results:
            rows.append("{i:2d} {j:2d} {v:7.3f} {omega:7.3f} {path_cost:7.3f} {lookahead_cost:7.3f} "
                        "{obstacle_cost:7} {speed_ref_cost:7.3f} {away_cost:7.3f} {total:9}".format(
                i=rec["i"], j=rec["j"], v=rec["v"], omega=rec["omega"],
                path_cost=rec["path_cost"], lookahead_cost=rec["lookahead_cost"],
                obstacle_cost="inf" if math.isinf(rec["obstacle_cost"]) else f"{rec['obstacle_cost']:.3f}",
                speed_ref_cost=rec["speed_ref_cost"], away_cost=rec["away_cost"],
                total="inf" if math.isinf(rec["total"]) else f"{rec['total']:.3f}",
            ))
        print("\n".join(rows))
        sys.stdout.flush()

    def on_step(self, _event):
        self.advance_one_step()

    def _timer_tick(self):
        if self.playing:
            self.advance_one_step()

    def on_toggle_play(self, _event):
        self.playing = not self.playing
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def advance_one_step(self):
        results, best, max_obs_cost = evaluate_samples(self.state, self.params, self.world)
        self.print_cost_table(results)

        look_pt, pv_v, pv_omega = self.compute_preview_command(self.state, advance=True)
        pv_traj = simulate_trajectory(self.state.x, self.state.y, self.state.theta, pv_v, pv_omega, self.params)
        pv_obs = max_obs_cost

        use_preview = (not math.isinf(pv_obs)) and (pv_obs <= self.params.preview_switch_obst_cost_thresh)
        print(f"Use preview: {use_preview}")
        sys.stdout.flush()
        if use_preview:
            v_cmd, omega_cmd = pv_v, pv_omega
            self.active_controller = "PREVIEW"
        else:
            if best and np.isfinite(best.get("total", float("inf"))):
                v_cmd = best["v"]
                omega_cmd = best["omega"]
            else:
                v_cmd = 0.0
                omega_cmd = 0.0
            self.active_controller = "DWA"

        print(f"Controller: {self.active_controller}")
        sys.stdout.flush()

        self.state.x = self.state.x + v_cmd * math.cos(self.state.theta) * self.params.dt
        self.state.y = self.state.y + v_cmd * math.sin(self.state.theta) * self.params.dt
        self.state.theta = self.state.theta + omega_cmd * self.params.dt
        self.state.v = v_cmd
        self.state.omega = omega_cmd

        self.redraw()

    def on_save_mp4(self, _event):
        was_playing = self.playing
        if was_playing:
            self.on_toggle_play(None)
        state_snapshot = RobotState(self.state.x, self.state.y, self.state.theta, self.state.v, self.state.omega)
        look_idx_snapshot = getattr(self, 'look_idx', 0)

        frames = int(20.0 / self.params.dt)
        fps = max(1, int(round(1.0 / self.params.dt)))
        out_path = os.path.abspath("dwa_sim.mp4")

        def _update(_frame):
            self.advance_one_step()
            return []

        anim = animation.FuncAnimation(self.fig, _update, frames=frames, blit=False)
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800, metadata={"title": "DWA Simulation", "artist": "visualiser"})
            anim.save(out_path, writer=writer, dpi=150)
            print(f"Saved MP4 to {out_path}")
        except Exception as e:
            print("Failed to save MP4. Ensure ffmpeg is installed (sudo apt-get install -y ffmpeg). Error:", e)
        finally:
            self.state = state_snapshot
            self.look_idx = look_idx_snapshot
            self.redraw()
            if was_playing:
                self.on_toggle_play(None)

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)

    def generate_straight_path(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], spacing: float) -> List[Tuple[float, float]]:
        x0, y0 = start_xy
        x1, y1 = goal_xy
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length == 0.0:
            return [(x0, y0)]
        num = max(1, int(length / spacing))
        pts = []
        for i in range(num + 1):
            t = i / num
            pts.append((x0 + t * dx, y0 + t * dy))
        return pts

    def compute_preview_command(self, state: RobotState, advance: bool = False) -> Tuple[Tuple[float, float] or None, float, float]:
        if not hasattr(self, 'path') or not self.path:
            return None, 0.0, 0.0
        x = state.x
        y = state.y
        theta = state.theta
        ld = max(1e-3, self.params.lookahead_distance)
        start_idx = getattr(self, 'look_idx', 0)
        found_idx = None
        skipped_inside = 0
        skipped_near = 0
        for i in range(start_idx, len(self.path)):
            if math.hypot(self.path[i][0] - x, self.path[i][1] - y) < ld:
                continue
            sd = signed_distance_to_inflated_square(self.path[i][0], self.path[i][1], self.world)
            if sd <= 0.0:
                skipped_inside += 1
                continue
            if sd <= self.params.obstacle_clearance_margin:
                skipped_near += 1
                continue
            found_idx = i
            break

        if found_idx is None:
            target_idx = len(self.path) - 1
            if advance:
                self.look_idx = target_idx
            look_pt = self.path[target_idx]
            print("Lookahead fallback: no valid point ahead; preview holding 0 velocity (use DWA if needed)")
            sys.stdout.flush()
            return look_pt, 0.0, 0.0

        if advance:
            if found_idx > start_idx:
                print(f"Lookahead advanced from {start_idx} to {found_idx} (skipped inside={skipped_inside}, near={skipped_near})")
                sys.stdout.flush()
            self.look_idx = found_idx
        target_idx = getattr(self, 'look_idx', found_idx)
        look_pt = self.path[min(target_idx, len(self.path) - 1)]
        alpha = math.atan2(look_pt[1] - y, look_pt[0] - x) - theta
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        v_cmd = self.params.ref_velocity
        omega_cmd = 2.0 * v_cmd * math.sin(alpha) / ld
        omega_cmd = max(-self.params.max_omega, min(self.params.max_omega, omega_cmd))
        return look_pt, v_cmd, omega_cmd


def main():
    params = DWAParameters()
    world = WorldConfig()
    state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)
    viz = DWAVisualizer(params, world, state)
    print("Press the 'Step (0.1s)' button to advance the robot at 10 Hz per click. Press 'q' to quit.")
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        print("Detected non-interactive backend ({}).".format(backend))
        print("Set MPLBACKEND=TkAgg (or Qt5Agg/GTK3Agg) and ensure a GUI is available. Example:")
        print("  MPLBACKEND=TkAgg python3 detector/src/scripts/dwa_visualiser.py")
        out_path = os.path.abspath("dwa_visualiser.png")
        viz.fig.savefig(out_path, dpi=150)
        print("Saved a static figure to {}".format(out_path))
        return
    plt.show()


if __name__ == "__main__":
    main()