import math
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

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
class Waypoint:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


@dataclass
class DWAResult:
    best_v: float = 0.0
    best_omega: float = 0.0
    obs_cost: float = 0.0


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0


@dataclass
class DWAParameters:
    # DWA parameters
    predict_time: float = 3.0
    dt: float = 0.1
    min_speed: float = 0.0
    max_speed: float = 0.3
    max_omega: float = 0.5
    vel_acc: float = 0.5
    omega_acc: float = 0.4
    vx_samples: int = 5
    omega_samples: int = 10
    
    # Cost function weights - BALANCED
    path_distance_bias: float = 10.0
    heading_bias: float = 0.0
    goal_distance_bias: float = 20
    occdist_scale: float = 50.0
    speed_ref_bias: float = 3.0
    away_bias: float = 20
    
    # Preview controller parameters
    ref_velocity: float = 0.3
    robot_radius: float = 0.5
    lookahead_distance: float = 1.0
    preview_switch_obst_cost_thresh: float = 10.0
    path_point_spacing: float = 0.2
    obstacle_clearance_margin: float = 0.05
    
    # Preview control matrices parameters
    Q_params: List[float] = field(default_factory=lambda: [5.0, 6.0, 5.0])
    R_param: float = 1.0
    preview_steps: int = 5
    preview_loop_thresh: float = 1e-5


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
    path_type: str = 'straight'  # 'straight' or 'snake'
    path_amplitude: float = 4.0
    path_wavelength: float = 6.0
    path_length: float = 10.0
    straight_path_distance: float = 5.0


class DWAController:
    def __init__(self, params: DWAParameters, world: WorldConfig, path: List[Waypoint]):
        self.params = params
        self.world = world
        self.current_path = path
        self.traj_list = []

    def compute_dynamic_window(self, current_v: float, current_omega: float) -> Tuple[float, float, float, float]:
        vs_min = self.params.min_speed
        vs_max = self.params.max_speed
        vo_min = -self.params.max_omega
        vo_max = self.params.max_omega
        
        vd_v_min = current_v - self.params.vel_acc * self.params.dt
        vd_v_max = current_v + self.params.vel_acc * self.params.dt
        vd_o_min = current_omega - self.params.omega_acc * self.params.dt
        vd_o_max = current_omega + self.params.omega_acc * self.params.dt
        
        v_min = max(vs_min, vd_v_min)
        v_max = min(vs_max, vd_v_max)
        o_min = max(vo_min, vd_o_min)
        o_max = min(vo_max, vd_o_max)
        
        return v_min, v_max, o_min, o_max

    def simulate_trajectory(self, x: float, y: float, theta: float, v_cmd: float, omega_cmd: float) -> np.ndarray:
        steps = int(self.params.predict_time / self.params.dt)
        traj = np.zeros((steps + 1, 3), dtype=float)
        traj[0] = np.array([x, y, theta], dtype=float)
        
        for k in range(steps):
            x = x + v_cmd * math.cos(theta) * self.params.dt
            y = y + v_cmd * math.sin(theta) * self.params.dt
            theta = theta + omega_cmd * self.params.dt
            traj[k + 1] = np.array([x, y, theta], dtype=float)
            
        return traj

    def signed_distance_to_inflated_square(self, px: float, py: float) -> float:
        cx, cy = self.world.obstacle_center
        half_size = self.world.obstacle_side / 2.0
        inflated_half = half_size + self.world.inflation_radius
        
        dx = abs(px - cx) - inflated_half
        dy = abs(py - cy) - inflated_half
        
        outside_dx = max(dx, 0.0)
        outside_dy = max(dy, 0.0)
        outside_distance = math.hypot(outside_dx, outside_dy)
        inside_distance = min(max(dx, dy), 0.0)
        
        return outside_distance + inside_distance

    def normalized_cost_from_distance(self, distance: float) -> float:
        if distance <= 0.0:
            return 1.0
        return min(0.99, math.exp(-distance / self.world.cost_decay_scale))

    def _crosstrack_error(self, x_r: float, y_r: float, x_ref: float, y_ref: float, theta_ref: float) -> float:
        return (y_ref - y_r) * math.cos(theta_ref) - (x_ref - x_r) * math.sin(theta_ref)

    def calc_crosstrack_cost(self, target_idx: int) -> float:
        if not self.traj_list.any() or not self.current_path:
            return 0.0

        final_x, final_y, _ = self.traj_list[-1]

        if target_idx >= len(self.current_path):
            return 0.0

        ref_wp = self.current_path[target_idx]
        cte = self._crosstrack_error(final_x, final_y, ref_wp.x, ref_wp.y, ref_wp.theta)
        return abs(cte)

    def calc_lookahead_heading_cost(self, target_idx: int) -> float:
        if not self.traj_list.any() or not self.current_path:
            return 0.0

        final_x, final_y, final_theta = self.traj_list[-1]
        
        if target_idx >= len(self.current_path):
            return 0.0
            
        lookahead_pt = self.current_path[target_idx]
        # angle_to_target = math.atan2(lookahead_pt.y - final_y, lookahead_pt.x - final_x)
        # error = angle_to_target - final_theta
        error = lookahead_pt.theta - final_theta
        
        # Normalize error
        while error > math.pi: 
            error -= 2 * math.pi
        while error < -math.pi: 
            error += 2 * math.pi
        
        return abs(error)

    def calc_lookahead_cost(self, target_idx: int) -> float:
        if not self.traj_list.any() or not self.current_path:
            return 0.0
            
        final_x, final_y, _ = self.traj_list[-1]
        
        if target_idx >= len(self.current_path):
            return 0.0

        lookahead_pt = self.current_path[target_idx]
        return math.hypot(final_x - lookahead_pt.x, final_y - lookahead_pt.y)

    def calc_speed_ref_cost(self, v: float) -> float:
        return abs(v - self.params.ref_velocity)

    def calc_obstacle_cost(self) -> float:
        if not self.traj_list.any():
            return 0.0

        cost_sum = 0.0
        for px, py, _ in self.traj_list:
            d = self.signed_distance_to_inflated_square(px, py)
            c_norm = self.normalized_cost_from_distance(d)
            cost_sum += c_norm
            
        return cost_sum / max(1, int(self.traj_list.shape[0]))

    def calc_away_from_obstacle_cost(self) -> float:
        if not self.traj_list.any():
            return 0.0

        max_exp_cost = 0.0
        for px, py, _ in self.traj_list:
            d = self.signed_distance_to_inflated_square(px, py)
            c_norm = self.normalized_cost_from_distance(d)
            exp_cost = math.exp(4.0 * c_norm)
            if exp_cost > max_exp_cost:
                max_exp_cost = exp_cost
                
        return max_exp_cost

    def dwa_main_control(self, state: RobotState, target_idx: int) -> Tuple[DWAResult, np.ndarray]:
        v_min, v_max, omega_min, omega_max = self.compute_dynamic_window(state.v, state.omega)
        
        min_cost = float('inf')
        best_v, best_omega = state.v, state.omega
        best_traj_away_cost = 0.0
        best_trajectory = None
        
        v_range = np.linspace(v_min, v_max, self.params.vx_samples)
        omega_range = np.linspace(omega_min, omega_max, self.params.omega_samples)

        for v_sample in v_range:
            for omega_sample in omega_range:
                self.traj_list = self.simulate_trajectory(state.x, state.y, state.theta, v_sample, omega_sample)
                
                crosstrack_cost = self.calc_crosstrack_cost(target_idx)
                heading_cost = self.calc_lookahead_heading_cost(target_idx)
                lookahead_cost = self.calc_lookahead_cost(target_idx)
                speed_ref_cost = self.calc_speed_ref_cost(v_sample)
                obs_cost = self.calc_obstacle_cost()
                away_cost = self.calc_away_from_obstacle_cost()

                total_cost = (self.params.path_distance_bias * crosstrack_cost +
                             self.params.heading_bias * heading_cost +
                             self.params.goal_distance_bias * lookahead_cost +
                             self.params.occdist_scale * obs_cost +
                             self.params.speed_ref_bias * speed_ref_cost +
                             self.params.away_bias * away_cost)
                
                # Only consider non-colliding trajectories
                collision = False
                for px, py, _ in self.traj_list:
                    if self.signed_distance_to_inflated_square(px, py) <= 0:
                        collision = True
                        break
                
                if not collision and total_cost < min_cost:
                    min_cost = total_cost
                    best_v = v_sample
                    best_omega = omega_sample
                    best_traj_away_cost = away_cost * self.params.away_bias
                    best_trajectory = self.traj_list.copy()
        
        # Fallback if no valid trajectory found
        if best_trajectory is None:
            print("WARNING: No valid DWA trajectory found, using safe fallback")
            best_v = 0.1
            best_omega = 0.0
            best_trajectory = self.simulate_trajectory(state.x, state.y, state.theta, best_v, best_omega)
            best_traj_away_cost = 100.0  # High cost to discourage using this
        
        result = DWAResult()
        result.best_v = best_v
        result.best_omega = best_omega
        result.obs_cost = best_traj_away_cost
        
        return result, best_trajectory


class PreviewController:
    def __init__(self, params: DWAParameters, world: WorldConfig):
        self.params = params
        self.world = world
        self.current_state = RobotState()
        self.targetid = 0
        self.path_generated = False
        self.initial_alignment = False
        self.active_controller = "PREVIEW"
        
        self.current_path: List[Waypoint] = []
        self.path_curvatures: List[float] = []
        self.best_dwa_trajectory: Optional[np.ndarray] = None
        
        self.vel_acc_bound = self.params.vel_acc * self.params.dt
        self.omega_acc_bound = self.params.omega_acc * self.params.dt
        
        self.dwa_controller: Optional[DWAController] = None
        
        self.generate_path()
        self.dwa_controller = DWAController(self.params, self.world, self.current_path)
        self.calculate_all_curvatures()
    
    def signed_distance_to_inflated_square(self, px: float, py: float) -> float:
        """Calculate signed distance to inflated obstacle (negative if inside)"""
        cx, cy = self.world.obstacle_center
        half_size = self.world.obstacle_side / 2.0
        inflated_half = half_size + self.world.inflation_radius
        
        dx = abs(px - cx) - inflated_half
        dy = abs(py - cy) - inflated_half
        
        outside_dx = max(dx, 0.0)
        outside_dy = max(dy, 0.0)
        outside_distance = math.hypot(outside_dx, outside_dy)
        inside_distance = min(max(dx, dy), 0.0)
        
        return outside_distance + inside_distance

    def generate_path(self):
        self.current_path = []
        
        if self.world.path_type == 'snake':
            self._generate_snake_path()
        else:
            self._generate_straight_path()
            
        self.path_generated = True
        print(f"Generated {self.world.path_type} path with {len(self.current_path)} points.")

    def _generate_snake_path(self):
        start_x, start_y = self.current_state.x, self.current_state.y
        num_points = int(math.ceil(self.world.path_length / self.params.path_point_spacing)) + 1
        
        for i in range(num_points):
            x = start_x + (self.world.path_length * i) / (num_points - 1)
            y = start_y + self.world.path_amplitude * math.sin(2.0 * math.pi * (x - start_x) / self.world.path_wavelength)
            dx = 1.0
            dy = self.world.path_amplitude * (2.0 * math.pi / self.world.path_wavelength) * math.cos(2.0 * math.pi * (x - start_x) / self.world.path_wavelength)
            theta = math.atan2(dy, dx)
            self.current_path.append(Waypoint(x, y, theta))

    def _generate_straight_path(self):
        start_x, start_y, start_theta = self.current_state.x, self.current_state.y, self.current_state.theta
        num_points = int(math.ceil(self.world.straight_path_distance / self.params.path_point_spacing)) + 1
        
        for i in range(num_points):
            dist = i * self.params.path_point_spacing
            x = start_x + dist * math.cos(start_theta)
            y = start_y + dist * math.sin(start_theta)
            self.current_path.append(Waypoint(x, y, start_theta))

    def calculate_all_curvatures(self):
        self.path_curvatures = [0.0] * len(self.current_path)
        for i in range(1, len(self.current_path) - 1):
            p_prev = self.current_path[i-1]
            p_curr = self.current_path[i]
            p_next = self.current_path[i+1]
            
            dx1, dy1 = p_curr.x - p_prev.x, p_curr.y - p_prev.y
            dx2, dy2 = p_next.x - p_curr.x, p_next.y - p_curr.y
            
            angle1 = math.atan2(dy1, dx1)
            angle2 = math.atan2(dy2, dx2)
            dtheta = angle2 - angle1
            
            while dtheta > math.pi: 
                dtheta -= 2 * math.pi
            while dtheta < -math.pi: 
                dtheta += 2 * math.pi
                
            dist = math.hypot(dx1, dy1)
            if dist > 1e-6:
                self.path_curvatures[i] = dtheta / dist

    def calc_gains(self):
        # A, B, D matrices
        A = np.array([[0, 1, 0], 
                      [0, 0, self.current_state.v], 
                      [0, 0, 0]])
        B = np.array([[0], [0], [1]])
        D = np.array([[0], [-self.current_state.v**2], [-self.current_state.v]])

        # Discretize matrices
        Ad = np.eye(3) + A * self.params.dt
        Bd = B * self.params.dt
        Dd = D * self.params.dt

        # Q, R matrices
        Q = np.diag(self.params.Q_params) * self.params.dt
        R = np.array([[self.params.R_param / self.params.dt]])

        # Solve Discrete Algebraic Riccati Equation (DARE)
        P = Q
        for _ in range(100):
            P_next = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad) + Q
            if np.linalg.norm(P_next - P) < self.params.preview_loop_thresh:
                break
            P = P_next
        
        # Calculate gains Kb (feedback) and Kf (feedforward)
        inv_term = np.linalg.inv(R + Bd.T @ P @ Bd)
        self.Kb = inv_term @ (Bd.T @ P @ Ad)
        
        lambda0 = Ad.T @ np.linalg.inv(np.eye(3) + P @ Bd @ inv_term @ Bd.T)

        Pc = np.zeros((3, self.params.preview_steps + 1))
        for i in range(self.params.preview_steps + 1):
            Pc[:, i] = (np.linalg.matrix_power(lambda0, i) @ P @ Dd).flatten()

        Lmatrix = np.eye(self.params.preview_steps + 1, k=1)
        
        Kf_term = Pc @ Lmatrix
        Kf_term[:, 0] += (P @ Dd).flatten()
        self.Kf = inv_term @ Bd.T @ Kf_term

    def compute_control(self, cross_track_error: float, heading_error: float) -> float:
        self.x_state = np.array([cross_track_error, 
                                self.current_state.v * math.sin(heading_error), 
                                heading_error])
        
        preview_curv = np.zeros(self.params.preview_steps + 1)
        for i in range(self.params.preview_steps + 1):
            preview_idx = self.targetid + i
            if preview_idx < len(self.path_curvatures):
                preview_curv[i] = self.path_curvatures[preview_idx]
        
        self.calc_gains()
        u_fb = -self.Kb @ self.x_state
        u_ff = -self.Kf @ preview_curv
        
        return u_fb[0] + u_ff[0]

    def bound_vel(self, ref_vel: float):
        if abs(ref_vel - self.current_state.v) < self.vel_acc_bound:
            self.current_state.v = ref_vel
        else:
            self.current_state.v += np.sign(ref_vel - self.current_state.v) * self.vel_acc_bound
        self.current_state.v = np.clip(self.current_state.v, 0.0, self.params.max_speed)

    def bound_omega(self, ref_omega: float):
        if abs(ref_omega - self.current_state.omega) < self.omega_acc_bound:
            self.current_state.omega = ref_omega
        else:
            self.current_state.omega += np.sign(ref_omega - self.current_state.omega) * self.omega_acc_bound
        self.current_state.omega = np.clip(self.current_state.omega, -self.params.max_omega, self.params.max_omega)

    def normalized_cost_from_distance(self, distance: float) -> float:
        """Helper method for obstacle cost calculation"""
        if distance <= 0.0:
            return 1.0
        return min(0.99, math.exp(-distance / self.world.cost_decay_scale))

    def run_control(self) -> bool:
        if not self.path_generated or self.dwa_controller is None:
            return False

        # Update lookahead point (targetid) with obstacle avoidance
        start_idx = self.targetid
        found_valid_lookahead = False
        skipped_inside = 0
        
        for i in range(start_idx, len(self.current_path)):
            path_point = self.current_path[i]
            
            # Check distance to obstacle
            sd = self.signed_distance_to_inflated_square(path_point.x, path_point.y)
            
            # Skip points inside obstacles
            if sd <= 0.0:
                skipped_inside += 1
                continue
            
            # Check if this point is beyond lookahead distance
            dist_to_robot = math.hypot(path_point.x - self.current_state.x, 
                                      path_point.y - self.current_state.y)
            if dist_to_robot < self.params.lookahead_distance:
                continue
                
            # Found valid lookahead point
            self.targetid = i
            found_valid_lookahead = True
            
            if i > start_idx:
                print(f"Lookahead advanced from {start_idx} to {i} (skipped inside={skipped_inside})")
            break
        
        # Fallback: if no valid lookahead found, use the last point
        if not found_valid_lookahead:
            self.targetid = len(self.current_path) - 1
            print(f"Lookahead fallback: using last point {self.targetid}")
            
        target_pt = self.current_path[self.targetid]

        # Calculate errors for Preview Controller
        cross_track_error = ((target_pt.y - self.current_state.y) * math.cos(target_pt.theta) - 
                            (target_pt.x - self.current_state.x) * math.sin(target_pt.theta))
                             
        angle_to_target = math.atan2(target_pt.y - self.current_state.y, target_pt.x - self.current_state.x)
        lookahead_heading_error = self.current_state.theta - angle_to_target
        
        while lookahead_heading_error > math.pi: 
            lookahead_heading_error -= 2 * math.pi
        while lookahead_heading_error < -math.pi: 
            lookahead_heading_error += 2 * math.pi
        
        # Initial alignment
        if not self.initial_alignment:
            max_lookahead_heading_error = 0.2
            if abs(lookahead_heading_error) < max_lookahead_heading_error:
                self.initial_alignment = True
            else:
                print("Initial alignment running")
                self.bound_vel(0.0)
                kp_adjust_cte = 2.0
                self.bound_omega(-kp_adjust_cte * lookahead_heading_error)
                return False

        # Evaluate DWA to get costs and best command
        dwa_result, best_trajectory = self.dwa_controller.dwa_main_control(self.current_state, self.targetid)
        self.best_dwa_trajectory = best_trajectory

        # SIMPLE SWITCHING LOGIC (like original simulation)
        # Calculate preview control command for comparison
        preview_v = self.params.ref_velocity
        preview_omega = self.compute_control(cross_track_error, lookahead_heading_error)
        
        # Simulate preview trajectory to check obstacle cost
        preview_traj = self.dwa_controller.simulate_trajectory(
            self.current_state.x, self.current_state.y, self.current_state.theta, 
            preview_v, preview_omega
        )
        
        # Calculate obstacle cost for preview trajectory
        preview_obs_cost = 0.0
        for px, py, _ in preview_traj:
            d = self.signed_distance_to_inflated_square(px, py)
            c_norm = self.normalized_cost_from_distance(d)
            preview_obs_cost += c_norm
        preview_obs_cost = preview_obs_cost / len(preview_traj) * self.params.occdist_scale
        
        print(f"Preview obs cost: {preview_obs_cost:.2f}, DWA obs cost: {dwa_result.obs_cost:.2f}")
        
        # Simple switching: Use preview if its obstacle cost is low enough
        use_preview = preview_obs_cost <= self.params.preview_switch_obst_cost_thresh
        
        if use_preview:
            v_ref = preview_v
            omega_ref = preview_omega
            self.active_controller = "PREVIEW"
            print(f"Using PREVIEW controller (obs cost: {preview_obs_cost:.2f})")
        else:
            v_ref = dwa_result.best_v
            omega_ref = dwa_result.best_omega
            self.active_controller = "DWA"
            print(f"Using DWA controller (obs cost: {preview_obs_cost:.2f} > {self.params.preview_switch_obst_cost_thresh:.2f})")

        # Reduce speed near goal
        goal_pt = self.current_path[-1]
        goal_distance = math.hypot(self.current_state.x - goal_pt.x, self.current_state.y - goal_pt.y)
        goal_distance_threshold = 0.2
        if goal_distance < 1.0:
            goal_reduce_factor = 0.5
            v_ref *= goal_reduce_factor * goal_distance
        
        # Apply acceleration limits
        self.bound_vel(v_ref)
        self.bound_omega(omega_ref)

        # Update robot state using the selected commands
        self.current_state.x += self.current_state.v * math.cos(self.current_state.theta) * self.params.dt
        self.current_state.y += self.current_state.v * math.sin(self.current_state.theta) * self.params.dt
        self.current_state.theta += self.current_state.omega * self.params.dt

        # Check for goal completion
        if goal_distance < goal_distance_threshold:
            print("Goal reached!")
            return True
            
        return False


class DWAVisualizer:
    def __init__(self, params: DWAParameters, world: WorldConfig, init_state: RobotState):
        self.params = params
        self.world = world
        self.state = init_state
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.18)
        
        # Setup buttons
        self.step_button_ax = self.fig.add_axes([0.08, 0.05, 0.2, 0.08])
        self.step_button = Button(self.step_button_ax, "Step (0.1s)")
        self.step_button.on_clicked(self.on_step)
        
        self.play_button_ax = self.fig.add_axes([0.33, 0.05, 0.2, 0.08])
        self.play_button = Button(self.play_button_ax, "Play")
        self.play_button.on_clicked(self.on_toggle_play)
        
        self.save_button_ax = self.fig.add_axes([0.58, 0.05, 0.3, 0.08])
        self.save_button = Button(self.save_button_ax, "Save MP4")
        self.save_button.on_clicked(self.on_save_mp4)

        self.info_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                     va="top", ha="left", fontsize=9,
                                     bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        
        # Visualization elements
        self.trajectory_lines = []
        self.best_line = None
        self.robot_artist = None
        self.heading_artist = None
        self.obstacle_patch = None
        self.inflated_patch = None
        self.goal_artist = None
        self.path_line = None
        self.lookahead_artist = None
        self.best_trajectory_line = None
        self.preview_trajectory_line = None
        
        self.playing = False
        self.timer = self.fig.canvas.new_timer(interval=int(self.params.dt * 1000))
        self.timer.add_callback(self._timer_tick)
        
        # Initialize preview controller
        self.preview_controller = PreviewController(params, world)
        
        self.draw_static()
        self.redraw()

    def draw_static(self):
        self.ax.set_xlim(self.world.x_limits)
        self.ax.set_ylim(self.world.y_limits)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.3)

        # Draw obstacle
        cx, cy = self.world.obstacle_center
        side = self.world.obstacle_side
        half = side / 2.0

        self.obstacle_patch = Rectangle((cx - half, cy - half), side, side,
                                       facecolor="black", edgecolor="black", alpha=0.6, zorder=2)
        self.ax.add_patch(self.obstacle_patch)

        # Draw inflated obstacle
        inflated_half = half + self.world.inflation_radius
        inflated_side = 2.0 * inflated_half
        self.inflated_patch = Rectangle((cx - inflated_half, cy - inflated_half),
                                       inflated_side, inflated_side,
                                       facecolor="none", edgecolor="red", linestyle="--", alpha=0.8, zorder=3)
        self.ax.add_patch(self.inflated_patch)

        # Draw goal
        self.goal_artist = self.ax.plot([self.world.goal_x], [self.world.goal_y],
                                       marker="*", color="green", markersize=12, zorder=4, label="Goal")[0]

        # Draw path
        if self.preview_controller.current_path:
            px = [p.x for p in self.preview_controller.current_path]
            py = [p.y for p in self.preview_controller.current_path]
            self.path_line, = self.ax.plot(px, py, color="#888", linestyle="-", linewidth=1.5, alpha=0.7, zorder=0)
            
        self.ax.legend(loc="upper right")

    def redraw(self):
        # Clear previous visualization elements
        for ln in self.trajectory_lines:
            ln.remove()
        self.trajectory_lines.clear()
        
        if self.best_trajectory_line is not None:
            self.best_trajectory_line.remove()
            self.best_trajectory_line = None
            
        if self.preview_trajectory_line is not None:
            self.preview_trajectory_line.remove()
            self.preview_trajectory_line = None
            
        if self.robot_artist is not None:
            self.robot_artist.remove()
            self.robot_artist = None
            
        if self.heading_artist is not None:
            self.heading_artist.remove()
            self.heading_artist = None
            
        if self.lookahead_artist is not None:
            self.lookahead_artist.remove()
            self.lookahead_artist = None

        # Update state from preview controller
        self.state = self.preview_controller.current_state

        # Always show DWA trajectories in background when DWA is available
        if self.preview_controller.dwa_controller:
            state = self.preview_controller.current_state
            dwa = self.preview_controller.dwa_controller
            
            v_min, v_max, omega_min, omega_max = dwa.compute_dynamic_window(state.v, state.omega)
            v_range = np.linspace(v_min, v_max, self.params.vx_samples)
            omega_range = np.linspace(omega_min, omega_max, self.params.omega_samples)

            for v_sample in v_range:
                for omega_sample in omega_range:
                    traj = dwa.simulate_trajectory(state.x, state.y, state.theta, v_sample, omega_sample)
                    
                    # Check for collision
                    collision = False
                    for px, py, _ in traj:
                        if dwa.signed_distance_to_inflated_square(px, py) <= 0:
                            collision = True
                            break
                    
                    color = "#444444" if collision else "#aaaaaa"
                    ln, = self.ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.0, alpha=0.5, zorder=1)
                    self.trajectory_lines.append(ln)

        # Show best DWA trajectory in red
        if (hasattr(self.preview_controller, 'best_dwa_trajectory') and 
            self.preview_controller.best_dwa_trajectory is not None and
            self.preview_controller.active_controller == "DWA"):
            
            self.best_trajectory_line, = self.ax.plot(
                self.preview_controller.best_dwa_trajectory[:, 0], 
                self.preview_controller.best_dwa_trajectory[:, 1], 
                color="crimson", linewidth=2.5, zorder=5, label="Best DWA Trajectory"
            )

        # Always calculate and show preview trajectory in green
        if self.preview_controller.active_controller == "PREVIEW":
            state = self.preview_controller.current_state
            target_pt = self.preview_controller.current_path[self.preview_controller.targetid]
            
            cross_track_error = ((target_pt.y - state.y) * math.cos(target_pt.theta) - 
                                (target_pt.x - state.x) * math.sin(target_pt.theta))
                                 
            angle_to_target = math.atan2(target_pt.y - state.y, target_pt.x - state.x)
            lookahead_heading_error = state.theta - angle_to_target
            
            while lookahead_heading_error > math.pi: 
                lookahead_heading_error -= 2 * math.pi
            while lookahead_heading_error < -math.pi: 
                lookahead_heading_error += 2 * math.pi
            
            omega_ref = self.preview_controller.compute_control(cross_track_error, lookahead_heading_error)
            
            preview_traj = self.preview_controller.dwa_controller.simulate_trajectory(
                state.x, state.y, state.theta, self.params.ref_velocity, omega_ref
            )
            
            self.preview_trajectory_line, = self.ax.plot(
                preview_traj[:, 0], preview_traj[:, 1], 
                color="#2a9d8f", linestyle=":", linewidth=2.0, zorder=4, label="Preview Trajectory"
            )

        # Draw lookahead point
        if self.preview_controller.targetid < len(self.preview_controller.current_path):
            look_pt = self.preview_controller.current_path[self.preview_controller.targetid]
            self.lookahead_artist = self.ax.plot([look_pt.x], [look_pt.y], 
                                                marker="o", color="#2a9d8f", markersize=6, zorder=6)[0]

        # Draw robot
        robot_circle = Circle((self.state.x, self.state.y), radius=self.params.robot_radius,
                             facecolor="royalblue", edgecolor="navy", alpha=0.6, zorder=6)
        self.ax.add_patch(robot_circle)
        self.robot_artist = robot_circle

        # Draw heading arrow
        arrow_length = max(0.6 * self.params.robot_radius, 0.15)
        hx = self.state.x + arrow_length * math.cos(self.state.theta)
        hy = self.state.y + arrow_length * math.sin(self.state.theta)
        self.heading_artist = self.ax.arrow(self.state.x, self.state.y,
                                           hx - self.state.x, hy - self.state.y,
                                           head_width=0.02, head_length=0.02, 
                                           fc="navy", ec="navy", zorder=7)

        # Update info text
        self.info_text.set_text(self.format_info_text())
        self.fig.canvas.draw_idle()

    def format_info_text(self) -> str:
        controller = self.preview_controller
        trajectory_info = ""
        
        if controller.active_controller == "DWA" and self.best_trajectory_line is not None:
            trajectory_info = "\nShowing: Best DWA trajectory (red)"
        elif controller.active_controller == "PREVIEW" and self.preview_trajectory_line is not None:
            trajectory_info = "\nShowing: Preview trajectory (green)"
        
        return (
            f"Controller: {controller.active_controller}\n"
            f"Position: ({controller.current_state.x:.2f}, {controller.current_state.y:.2f})\n"
            f"Heading: {math.degrees(controller.current_state.theta):.1f}°\n"
            f"Velocity: {controller.current_state.v:.2f} m/s\n"
            f"Omega: {math.degrees(controller.current_state.omega):.1f}°/s\n"
            f"Lookahead ID: {controller.targetid}/{len(controller.current_path)}"
            f"{trajectory_info}"
        )

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
        goal_reached = self.preview_controller.run_control()
        self.redraw()
        
        if goal_reached:
            self.playing = False
            self.play_button.label.set_text("Play")
            self.timer.stop()

    def on_save_mp4(self, _event):
        was_playing = self.playing
        if was_playing:
            self.on_toggle_play(None)
            
        # Save current state for restoration
        state_snapshot = RobotState(self.state.x, self.state.y, self.state.theta, 
                                   self.state.v, self.state.omega)
        
        frames = int(20.0 / self.params.dt)
        fps = max(1, int(round(1.0 / self.params.dt)))
        out_path = os.path.abspath("dwa_sim.mp4")

        def _update(_frame):
            self.advance_one_step()
            return []

        anim = animation.FuncAnimation(self.fig, _update, frames=frames, blit=False)
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800, 
                                           metadata={"title": "DWA Simulation", "artist": "visualiser"})
            anim.save(out_path, writer=writer, dpi=150)
            print(f"Saved MP4 to {out_path}")
        except Exception as e:
            print("Failed to save MP4. Ensure ffmpeg is installed (sudo apt-get install -y ffmpeg). Error:", e)
        finally:
            # Restore state
            self.state = state_snapshot
            self.preview_controller.current_state = state_snapshot
            self.redraw()
            if was_playing:
                self.on_toggle_play(None)

    def on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)


def main():
    params = DWAParameters()
    world = WorldConfig()
    state = RobotState(x=0.0, y=0.0, theta=0.0, v=0.0, omega=0.0)
    
    viz = DWAVisualizer(params, world, state)
    
    print("Press the 'Step (0.1s)' button to advance the robot at 10 Hz per click.")
    print("Press 'Play' to run continuously. Press 'q' to quit.")
    
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or "inline" in backend:
        print("Detected non-interactive backend ({}).".format(backend))
        print("Set MPLBACKEND=TkAgg (or Qt5Agg/GTK3Agg) and ensure a GUI is available.")
        out_path = os.path.abspath("dwa_visualiser.png")
        viz.fig.savefig(out_path, dpi=150)
        print("Saved a static figure to {}".format(out_path))
        return
        
    plt.show()


if __name__ == "__main__":
    main()