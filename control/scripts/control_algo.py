#!/usr/bin/env python3

import rospy
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# Import ROS message types
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Float64, Bool
from visualization_msgs.msg import MarkerArray, Marker

# === Data Structures (like C++ structs) ===

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

# === DWA Controller Class ===

class DWAController:
    def __init__(self, path: List[Waypoint]):
        self.current_path = path
        
        # Load DWA parameters from ROS parameter server
        self.predict_time = rospy.get_param('~dwa_controller/predict_time', 2.0)
        self.path_distance_bias = rospy.get_param('~dwa_controller/path_distance_bias', 0.0)
        self.heading_bias = rospy.get_param('~dwa_controller/heading_bias', 25.0)
        self.goal_distance_bias = rospy.get_param('~dwa_controller/goal_distance_bias', 0.5)
        self.occdist_scale = rospy.get_param('~dwa_controller/occdist_scale', 10.0)
        self.speed_ref_bias = rospy.get_param('~dwa_controller/speed_ref_bias', 0.005)
        self.away_bias = rospy.get_param('~dwa_controller/away_bias', 20.0)
        self.vx_samples = rospy.get_param('~dwa_controller/vx_samples', 3)
        self.omega_samples = rospy.get_param('~dwa_controller/omega_samples', 5)
        self.dt_dwa = rospy.get_param('preview_controller/dt_dwa', 0.1)
        self.ref_velocity = rospy.get_param('preview_controller/linear_velocity', 0.3)
        
        # Shared parameters with PreviewController
        self.vel_acc = rospy.get_param('preview_controller/vel_acc', 0.5)
        self.omega_acc = rospy.get_param('preview_controller/omega_acc', 0.4)
        self.min_speed = rospy.get_param('preview_controller/min_speed', 0.0)
        self.max_speed = rospy.get_param('preview_controller/max_speed', 0.3)
        self.max_omega = rospy.get_param('preview_controller/max_omega', 0.5)
        
        self.costmap_received = False
        self.occ_grid = None
        self.traj_list = []

        self.occ_sub = rospy.Subscriber("/local_costmap", OccupancyGrid, self.costmap_callback)
        self.traj_pub = rospy.Publisher("dwa_trajectories", MarkerArray, queue_size=1)
        
    def costmap_callback(self, msg: OccupancyGrid):
        self.occ_grid = msg
        self.costmap_received = True

    def world_to_costmap(self, x: float, y: float, robot_x: float, robot_y: float) -> Tuple[int, int]:
        if not self.costmap_received:
            return -1, -1
        
        rel_x = x - robot_x
        rel_y = y - robot_y
        
        origin_x = self.occ_grid.info.origin.position.x
        origin_y = self.occ_grid.info.origin.position.y
        resolution = self.occ_grid.info.resolution
        
        mx = int((rel_x - origin_x) / resolution)
        my = int((rel_y - origin_y) / resolution)
        
        return mx, my
        
    def get_costmap_cost(self, mx: int, my: int) -> int:
        width = self.occ_grid.info.width
        height = self.occ_grid.info.height
        if 0 <= mx < width and 0 <= my < height:
            idx = my * width + mx
            return self.occ_grid.data[idx]
        return 100 # Lethal cost for out of bounds

    def calc_dynamic_window(self, v: float, omega: float) -> Tuple[float, float, float, float]:
        vs = (self.min_speed, self.max_speed, -self.max_omega, self.max_omega)
        vd = (v - self.vel_acc * self.dt_dwa, v + self.vel_acc * self.dt_dwa,
              omega - self.omega_acc * self.dt_dwa, omega + self.omega_acc * self.dt_dwa)
              
        v_min = max(vs[0], vd[0])
        v_max = min(vs[1], vd[1])
        omega_min = max(vs[2], vd[2])
        omega_max = min(vs[3], vd[3])
        
        return v_min, v_max, omega_min, omega_max

    def calc_trajectory(self, x: float, y: float, theta: float, v: float, omega: float) -> np.ndarray:
        traj = [(x, y, theta)]
        steps = int(self.predict_time / self.dt_dwa)
        for _ in range(steps):
            x += v * math.cos(theta) * self.dt_dwa
            y += v * math.sin(theta) * self.dt_dwa
            theta += omega * self.dt_dwa
            traj.append((x, y, theta))
        return np.array(traj)

    def calc_lookahead_heading_cost(self, target_idx: int) -> float:
        if not self.traj_list.any() or not self.current_path:
            return 0.0

        final_x, final_y, final_theta = self.traj_list[-1]
        
        if target_idx >= len(self.current_path):
            return 0.0
            
        lookahead_pt = self.current_path[target_idx]
        
        angle_to_target = math.atan2(lookahead_pt.y - final_y, lookahead_pt.x - final_x)
        error = angle_to_target - final_theta
        
        # Normalize error
        while error > math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi
        
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
        return abs(v - self.ref_velocity)

    def calc_obstacle_cost(self, robot_x: float, robot_y: float) -> float:
        if not self.traj_list.any() or not self.costmap_received:
            return 0.0
        
        cost_sum = 0.0
        for pt in self.traj_list:
            mx, my = self.world_to_costmap(pt[0], pt[1], robot_x, robot_y)
            cost_sum += self.get_costmap_cost(mx, my) / 100.0 # Normalize cost 0-1
        
        return cost_sum / len(self.traj_list)

    def dwa_main_control(self, state: RobotState, target_idx: int) -> DWAResult:
        dw = self.calc_dynamic_window(state.v, state.omega)
        v_min, v_max, omega_min, omega_max = dw
        
        min_cost = float('inf')
        best_v, best_omega = state.v, state.omega
        max_obstacle_cost = 0.0

        v_range = np.linspace(v_min, v_max, self.vx_samples)
        omega_range = np.linspace(omega_min, omega_max, self.omega_samples)

        for v_sample in v_range:
            for omega_sample in omega_range:
                self.traj_list = self.calc_trajectory(state.x, state.y, state.theta, v_sample, omega_sample)
                
                heading_cost = self.calc_lookahead_heading_cost(target_idx)
                lookahead_cost = self.calc_lookahead_cost(target_idx)
                speed_ref_cost = self.calc_speed_ref_cost(v_sample)
                obs_cost = self.calc_obstacle_cost(state.x, state.y)
                
                total_cost = (self.heading_bias * heading_cost +
                              self.goal_distance_bias * lookahead_cost +
                              self.occdist_scale * obs_cost +
                              self.speed_ref_bias * speed_ref_cost)
                
                if obs_cost > max_obstacle_cost:
                    max_obstacle_cost = obs_cost

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_v = v_sample
                    best_omega = omega_sample
        
        result = DWAResult()
        result.best_v = best_v
        result.best_omega = best_omega
        result.obs_cost = self.occdist_scale * max_obstacle_cost
        
        rospy.loginfo_throttle(1.0, f"DWA max obstacle scaled cost = {result.obs_cost:.2f}")
        return result

# === Preview Controller Class ===

class PreviewController:
    def __init__(self):
        # Initialize state variables
        self.current_state = RobotState()
        self.targetid = 0
        self.initial_pose_received = False
        self.path_generated = False
        self.initial_alignment = False
        self.start_moving = False
        self.active_controller = "PREVIEW" # Default controller

        # Load parameters
        self.load_params()

        # Setup ROS publishers and subscribers
        self.setup_ros_communications()

        self.vel_acc_bound = self.vel_acc * self.dt
        self.omega_acc_bound = self.omega_acc * self.dt
        self.dwa_controller = None # Will be initialized after path is generated

    def load_params(self):
        # Path parameters
        self.path_type = rospy.get_param('preview_controller/path_type', 'snake')
        self.path_amplitude = rospy.get_param('preview_controller/amplitude', 4.0)
        self.path_wavelength = rospy.get_param('preview_controller/wavelength', 6.0)
        self.path_length = rospy.get_param('preview_controller/length', 10.0)
        self.path_point_spacing = rospy.get_param('preview_controller/point_spacing', 0.3)
        self.straight_path_distance = rospy.get_param('preview_controller/straight_path_distance', 5.0)
        
        # Controller parameters
        self.linear_velocity = rospy.get_param('preview_controller/linear_velocity', 0.3)
        self.dt = rospy.get_param('preview_controller/preview_dt', 0.1)
        self.max_vel = rospy.get_param('preview_controller/max_vel', 0.3)
        self.max_omega = rospy.get_param('preview_controller/max_omega', 0.6)
        self.vel_acc = rospy.get_param('preview_controller/vel_acc', 0.5)
        self.omega_acc = rospy.get_param('preview_controller/omega_acc', 0.4)
        self.lookahead_distance = rospy.get_param('preview_controller/lookahead_distance', 0.5)
        self.max_lookahead_heading_error = rospy.get_param('preview_controller/max_lookahead_heading_error', 0.2)
        self.kp_adjust_cte = rospy.get_param('preview_controller/kp_adjust_cte', 2.0)
        self.goal_distance_threshold = rospy.get_param('preview_controller/goal_distance_threshold', 0.2)
        self.goal_reduce_factor = rospy.get_param('preview_controller/goal_reduce_factor', 0.5)
        
        # Hysteresis thresholds
        self.dwa_activation_cost_thresh = rospy.get_param('preview_controller/dwa_activation_cost_thresh', 10.0)
        self.preview_reactivation_cost_thresh = rospy.get_param('preview_controller/preview_reactivation_cost_thresh', 5.0)

        # Preview control matrices parameters
        self.Q_params = rospy.get_param('preview_controller/Q_params', [5.0, 6.0, 5.0])
        self.R_param = rospy.get_param('preview_controller/R', 1.0)
        self.preview_steps = rospy.get_param('preview_controller/preview_steps', 5) # Added from constructor default
        self.preview_loop_thresh = rospy.get_param('preview_controller/preview_loop_thresh', 1e-5)

    def setup_ros_communications(self):
        self.robot_vel_pub = rospy.Publisher("/atrv/cmd_vel", Twist, queue_size=10)
        self.path_pub = rospy.Publisher("planned_path", Path, queue_size=10)
        self.robot_pose_sub = rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)
        # ... Add other debug publishers if needed

    def robot_pose_callback(self, msg: PoseStamped):
        self.current_state.x = msg.pose.position.x
        self.current_state.y = msg.pose.position.y
        self.current_state.theta = msg.pose.orientation.z # Assuming Z stores theta
        
        if not self.initial_pose_received:
            self.initial_pose_received = True
            rospy.loginfo(f"Initial robot pose received: x={self.current_state.x:.2f}, y={self.current_state.y:.2f}")
            if self.path_type == 'snake':
                self.generate_snake_path()
            else:
                self.generate_straight_path()
            
            self.dwa_controller = DWAController(self.current_path)
            self.path_generated = True
            self.calculate_all_curvatures()
        
        if self.path_generated:
            self.publish_path()

    def generate_snake_path(self):
        self.current_path = []
        start_x, start_y = self.current_state.x, self.current_state.y
        num_points = int(math.ceil(self.path_length / self.path_point_spacing)) + 1
        
        for i in range(num_points):
            x = start_x + (self.path_length * i) / (num_points - 1)
            y = start_y + self.path_amplitude * math.sin(2.0 * math.pi * (x - start_x) / self.path_wavelength)
            dx = 1.0
            dy = self.path_amplitude * (2.0 * math.pi / self.path_wavelength) * math.cos(2.0 * math.pi * (x - start_x) / self.path_wavelength)
            theta = math.atan2(dy, dx)
            self.current_path.append(Waypoint(x, y, theta))
        rospy.loginfo(f"Generated snake path with {len(self.current_path)} points.")

    def generate_straight_path(self):
        self.current_path = []
        start_x, start_y, start_theta = self.current_state.x, self.current_state.y, self.current_state.theta
        num_points = int(math.ceil(self.straight_path_distance / self.path_point_spacing)) + 1
        
        for i in range(num_points):
            dist = i * self.path_point_spacing
            x = start_x + dist * math.cos(start_theta)
            y = start_y + dist * math.sin(start_theta)
            self.current_path.append(Waypoint(x, y, start_theta))
        rospy.loginfo(f"Generated straight path with {len(self.current_path)} points.")
        
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
            while dtheta > math.pi: dtheta -= 2 * math.pi
            while dtheta < -math.pi: dtheta += 2 * math.pi
            dist = math.hypot(dx1, dy1)
            if dist > 1e-6:
                self.path_curvatures[i] = dtheta / dist

    def calc_gains(self):
        # A, B, D matrices
        A = np.array([[0, 1, 0], [0, 0, self.current_state.v], [0, 0, 0]])
        B = np.array([[0], [0], [1]])
        D = np.array([[0], [-self.current_state.v**2], [-self.current_state.v]])

        # Discretize matrices
        Ad = np.eye(3) + A * self.dt
        Bd = B * self.dt
        Dd = D * self.dt

        # Q, R matrices
        Q = np.diag(self.Q_params) * self.dt
        R = np.array([[self.R_param / self.dt]])

        # Solve Discrete Algebraic Riccati Equation (DARE)
        P = Q
        for _ in range(100):
            P_next = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad) + Q
            if np.linalg.norm(P_next - P) < self.preview_loop_thresh:
                break
            P = P_next
        
        # Calculate gains Kb (feedback) and Kf (feedforward)
        inv_term = np.linalg.inv(R + Bd.T @ P @ Bd)
        self.Kb = inv_term @ (Bd.T @ P @ Ad)
        
        lambda0 = Ad.T @ np.linalg.inv(np.eye(3) + P @ Bd @ inv_term @ Bd.T)

        Pc = np.zeros((3, self.preview_steps + 1))
        for i in range(self.preview_steps + 1):
            Pc[:, i] = (np.linalg.matrix_power(lambda0, i) @ P @ Dd).flatten()

        Lmatrix = np.eye(self.preview_steps + 1, k=1)
        
        Kf_term = Pc @ Lmatrix
        Kf_term[:, 0] += (P @ Dd).flatten()
        self.Kf = inv_term @ Bd.T @ Kf_term

    def compute_control(self, cross_track_error: float, heading_error: float) -> float:
        self.x_state = np.array([cross_track_error, self.current_state.v * math.sin(heading_error), heading_error])
        
        preview_curv = np.zeros(self.preview_steps + 1)
        for i in range(self.preview_steps + 1):
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
        self.current_state.v = np.clip(self.current_state.v, 0.0, self.max_vel)

    def bound_omega(self, ref_omega: float):
        if abs(ref_omega - self.current_state.omega) < self.omega_acc_bound:
            self.current_state.omega = ref_omega
        else:
            self.current_state.omega += np.sign(ref_omega - self.current_state.omega) * self.omega_acc_bound
        self.current_state.omega = np.clip(self.current_state.omega, -self.max_omega, self.max_omega)

    def run_control(self):
        if not self.path_generated or self.dwa_controller is None:
            return False

        # Update lookahead point (targetid)
        while (self.targetid + 1 < len(self.current_path) and 
               math.hypot(self.current_state.x - self.current_path[self.targetid].x, 
                          self.current_state.y - self.current_path[self.targetid].y) < self.lookahead_distance):
            self.targetid += 1
            
        target_pt = self.current_path[self.targetid]

        # Calculate errors for Preview Controller
        cross_track_error = ((target_pt.y - self.current_state.y) * math.cos(target_pt.theta) - 
                             (target_pt.x - self.current_state.x) * math.sin(target_pt.theta))
                             
        angle_to_target = math.atan2(target_pt.y - self.current_state.y, target_pt.x - self.current_state.x)
        lookahead_heading_error = self.current_state.theta - angle_to_target
        while lookahead_heading_error > math.pi: lookahead_heading_error -= 2 * math.pi
        while lookahead_heading_error < -math.pi: lookahead_heading_error += 2 * math.pi
        
        # Initial alignment
        if not self.initial_alignment:
            if abs(lookahead_heading_error) < self.max_lookahead_heading_error:
                self.initial_alignment = True
            else:
                self.bound_vel(0.0)
                self.bound_omega(-self.kp_adjust_cte * lookahead_heading_error)
                self.publish_cmd_vel()
                return False

        # Evaluate DWA to get costs and best command
        dwa_result = self.dwa_controller.dwa_main_control(self.current_state, self.targetid)

        # Hysteresis Logic
        if self.active_controller == "PREVIEW" and dwa_result.obs_cost > self.dwa_activation_cost_thresh:
            self.active_controller = "DWA"
            rospy.logwarn(f"SWITCH: PREVIEW -> DWA (cost {dwa_result.obs_cost:.2f} > {self.dwa_activation_cost_thresh:.2f})")
        elif self.active_controller == "DWA" and dwa_result.obs_cost < self.preview_reactivation_cost_thresh:
            self.active_controller = "PREVIEW"
            rospy.loginfo(f"SWITCH: DWA -> PREVIEW (cost {dwa_result.obs_cost:.2f} < {self.preview_reactivation_cost_thresh:.2f})")

        # Set velocity based on controller
        if self.active_controller == "DWA":
            v_ref = dwa_result.best_v
            omega_ref = dwa_result.best_omega
            rospy.loginfo_throttle(0.5, f"Controller: DWA | v={v_ref:.2f}, w={omega_ref:.2f}")
        else: # PREVIEW
            v_ref = self.linear_velocity
            omega_ref = self.compute_control(cross_track_error, lookahead_heading_error)
            rospy.loginfo_throttle(0.5, f"Controller: PREVIEW | v={v_ref:.2f}, w={omega_ref:.2f}")

        # Reduce speed near goal
        goal_pt = self.current_path[-1]
        goal_distance = math.hypot(self.current_state.x - goal_pt.x, self.current_state.y - goal_pt.y)
        if goal_distance < 1.0:
            v_ref *= self.goal_reduce_factor * goal_distance
        
        # Apply acceleration limits and publish
        self.bound_vel(v_ref)
        self.bound_omega(omega_ref)
        self.publish_cmd_vel()

        # Check for goal completion
        if goal_distance < self.goal_distance_threshold:
            self.stop_robot()
            rospy.loginfo("Goal reached!")
            return True
        return False

    def publish_cmd_vel(self):
        cmd = Twist()
        cmd.linear.x = self.current_state.v
        cmd.angular.z = self.current_state.omega
        self.robot_vel_pub.publish(cmd)
        
    def stop_robot(self):
        self.current_state.v = 0.0
        self.current_state.omega = 0.0
        self.publish_cmd_vel()

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"
        for i in range(self.targetid, len(self.current_path)):
            wp = self.current_path[i]
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp.x
            pose.pose.position.y = wp.y
            # Simplified orientation for 2D
            pose.pose.orientation.z = math.sin(wp.theta / 2.0)
            pose.pose.orientation.w = math.cos(wp.theta / 2.0)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def spin(self):
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            if self.path_generated:
                goal_reached = self.run_control()
                if goal_reached:
                    break
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('final_control_node')
        controller = PreviewController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass