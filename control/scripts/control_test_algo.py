#!/usr/bin/env python3

import rospy
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple

# Import ROS message types
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import Float64, Bool
from visualization_msgs.msg import MarkerArray, Marker
import tf2_ros
import tf2_geometry_msgs

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
    lookahead_x: float = 0.0
    lookahead_y: float = 0.0
    lookahead_theta: float = 0.0

@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0

# === DWA Controller Class ===

class DWAController:
    def __init__(self, path: List[Waypoint], target_idx_ref, max_points_ref):
        self.current_path = path
        self.target_idx_ref = target_idx_ref  # Reference to PreviewController's targetid
        self.max_path_points_ref = max_points_ref  # Reference to path size
        
        # Load DWA parameters from ROS parameter server
        self.predict_time = rospy.get_param('dwa_controller/predict_time', 2.0)
        self.path_distance_bias = rospy.get_param('dwa_controller/path_distance_bias', 20.0)
        self.lookahead_heading_bias = rospy.get_param('dwa_controller/lookahead_heading_bias', 20.0)
        self.goal_distance_bias = rospy.get_param('dwa_controller/goal_distance_bias', 0.5)
        self.occdist_scale = rospy.get_param('dwa_controller/occdist_scale', 200.0)
        self.speed_ref_bias = rospy.get_param('dwa_controller/speed_ref_bias', 10.0)
        self.away_bias = rospy.get_param('dwa_controller/away_bias', 20.0)
        self.lookahead_distance = rospy.get_param('dwa_controller/lookahead_distance', 0.5)
        self.lookahead_obstacle_cost_thresh = rospy.get_param('dwa_controller/lookahead_obstacle_cost_thresh', 50.0)
        self.vx_samples = rospy.get_param('dwa_controller/vx_samples', 3)
        self.omega_samples = rospy.get_param('dwa_controller/omega_samples', 5)
        self.dt_dwa = rospy.get_param('preview_controller/dt_dwa', 0.1)
        self.ref_velocity = rospy.get_param('preview_controller/linear_velocity', 0.3)
        
        # Shared parameters with PreviewController
        self.vel_acc = rospy.get_param('preview_controller/vel_acc', 0.5)
        self.omega_acc = rospy.get_param('preview_controller/omega_acc', 0.4)
        self.min_speed = rospy.get_param('preview_controller/min_speed', 0.0)
        self.max_speed = rospy.get_param('preview_controller/max_speed', 0.3)
        self.max_omega = rospy.get_param('preview_controller/max_omega', 0.5)
        self.robot_radius = rospy.get_param('preview_controller/robot_radius', 0.5)
        
        self.costmap_received = False
        self.occ_grid = None
        self.traj_list = []
        
        # Temp lookahead variables
        self.temp_lookahead_x = 0.0
        self.temp_lookahead_y = 0.0
        self.temp_lookahead_theta = 0.0

        self.occ_sub = rospy.Subscriber("/local_costmap", OccupancyGrid, self.costmap_callback)
        self.traj_pub = rospy.Publisher("dwa_trajectories", MarkerArray, queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.world_frame = "odom"

    def costmap_callback(self, msg: OccupancyGrid):
        self.occ_grid = msg
        self.costmap_received = True

    def chkside(self, x1: float, y1: float, path_theta: float, robot_x: float, robot_y: float) -> bool:
        """Checks if robot is in front of point (same as C++ implementation)"""
        ineq = 0.0
        t = False
        
        if abs(math.tan(path_theta)) < 1e-6:
            # Near vertical line
            ineq = robot_x - x1
            if ineq > 0:
                t = False
                if -math.pi/2 < path_theta < math.pi/2:
                    t = True
            else:
                t = True
                if -math.pi/2 < path_theta < math.pi/2:
                    t = False
        else:
            m = -1.0 / math.tan(path_theta)
            ineq = robot_y - (m * robot_x) - y1 + (m * x1)
        
        if ineq > 0:
            t = True
            if path_theta < 0:
                t = False
        else:
            t = False
            if path_theta < 0:
                t = True
        
        return t

    def query_cost_at_world(self, wx: float, wy: float, robot_x: float, robot_y: float) -> float:
        """Query cost at world coordinates (returns 0-100 scale)"""
        if not self.costmap_received:
            return 100.0
        
        rel_x = wx - robot_x
        rel_y = wy - robot_y
        
        origin_x = self.occ_grid.info.origin.position.x
        origin_y = self.occ_grid.info.origin.position.y
        resolution = self.occ_grid.info.resolution
        
        mx = int((rel_x - origin_x) / resolution)
        my = int((rel_y - origin_y) / resolution)
        
        if mx < 0 or my < 0 or mx >= self.occ_grid.info.width or my >= self.occ_grid.info.height:
            return 100.0
        
        idx = my * self.occ_grid.info.width + mx
        if idx < 0 or idx >= len(self.occ_grid.data):
            return 100.0
        
        return float(self.occ_grid.data[idx])

    def point_in_costmap_frame_to_map_indices(self, x: float, y: float) -> Tuple[int, int]:
        if not self.costmap_received:
            return -1, -1

        origin_x = self.occ_grid.info.origin.position.x
        origin_y = self.occ_grid.info.origin.position.y
        resolution = self.occ_grid.info.resolution
        
        mx = int((x - origin_x) / resolution)
        my = int((y - origin_y) / resolution)

        return mx, my

    def world_to_costmap(self, wx: float, wy: float, robot_x: float, robot_y: float, robot_yaw: float) -> Tuple[int, int, bool]:
        """Convert world coordinates to costmap indices"""
        if not self.costmap_received:
            return -1, -1, False
        
        dx = wx - robot_x
        dy = wy - robot_y
        
        # Transform to base_link
        rel_x = dx * math.cos(robot_yaw) + dy * math.sin(robot_yaw)
        rel_y = -dx * math.sin(robot_yaw) + dy * math.cos(robot_yaw)
        
        origin_x = self.occ_grid.info.origin.position.x
        origin_y = self.occ_grid.info.origin.position.y
        resolution = self.occ_grid.info.resolution
        
        mx1 = int((rel_x - origin_x) / resolution)
        my1 = int((rel_y - origin_y) / resolution)

        try:
            # Get the single transform from world frame to the costmap's frame
            transform = self.tf_buffer.lookup_transform(
                self.occ_grid.header.frame_id, # Target frame (e.g., 'base_link')
                self.world_frame,              # Source frame (e.g., 'odom')
                rospy.Time(0),                 # Get the latest available transform
                rospy.Duration(0.1)            # Timeout
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(1.0, f"TF transform error in DWA: {e}")
            return float('inf') # Return a high cost if transform fails

        p_world = PointStamped()
        p_world.header.frame_id = self.world_frame
        p_world.point.x = wx
        p_world.point.y = wy
        p_costmap_frame = tf2_geometry_msgs.do_transform_point(p_world, transform)
        
        mx, my = self.point_in_costmap_frame_to_map_indices(p_costmap_frame.point.x, p_costmap_frame.point.y)

        # print(f"MX MY 1: {mx1,my1} || MX MY 2: {mx,my} ")
        
        valid = (0 <= mx < self.occ_grid.info.width and 0 <= my < self.occ_grid.info.height)
        return mx, my, valid

    def get_costmap_cost(self, mx: int, my: int) -> int:
        """Get cost at map coordinates"""
        idx = my * self.occ_grid.info.width + mx
        if 0 <= idx < len(self.occ_grid.data):
            return int(self.occ_grid.data[idx])
        return 100

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
            # Normalize theta
            while theta > math.pi:
                theta -= 2 * math.pi
            while theta < -math.pi:
                theta += 2 * math.pi
            traj.append((x, y, theta))
        return np.array(traj)

    def calc_lookahead_heading_cost(self) -> float:
        """Calculate lookahead heading cost and update temp_lookahead variables"""
        if len(self.traj_list) == 0 or not self.current_path or self.target_idx_ref is None:
            return 0.0
        
        final_x, final_y, final_theta = self.traj_list[-1]
        
        current_target = self.target_idx_ref[0]
        if current_target >= len(self.current_path):
            current_target = len(self.current_path) - 1
        
        temp_look_ahead_idx = current_target
        
        # Advance if behind robot
        while (temp_look_ahead_idx + 1 < len(self.current_path) and 
               self.chkside(self.current_path[temp_look_ahead_idx].x,
                           self.current_path[temp_look_ahead_idx].y,
                           self.current_path[temp_look_ahead_idx].theta,
                           final_x, final_y)):
            temp_look_ahead_idx += 1
        
        # Advance based on lookahead distance
        while (temp_look_ahead_idx + 1 < len(self.current_path) and
               math.hypot(self.current_path[temp_look_ahead_idx].x - final_x,
                         self.current_path[temp_look_ahead_idx].y - final_y) < self.lookahead_distance):
            temp_look_ahead_idx += 1
        
        # Advance if lookahead point is in obstacle
        while (temp_look_ahead_idx + 1 < len(self.current_path) and
               self.query_cost_at_world(self.current_path[temp_look_ahead_idx].x,
                                       self.current_path[temp_look_ahead_idx].y,
                                       final_x, final_y) > self.lookahead_obstacle_cost_thresh):
            temp_look_ahead_idx += 1
        
        # Store lookahead point
        self.temp_lookahead_x = self.current_path[temp_look_ahead_idx].x
        self.temp_lookahead_y = self.current_path[temp_look_ahead_idx].y
        self.temp_lookahead_theta = self.current_path[temp_look_ahead_idx].theta
        
        # Calculate heading error
        angle_to_target = math.atan2(self.temp_lookahead_y - final_y, 
                                     self.temp_lookahead_x - final_x)
        error = angle_to_target - final_theta
        
        # Normalize error
        while error > math.pi:
            error -= 2 * math.pi
        while error < -math.pi:
            error += 2 * math.pi
        
        return abs(error)

    def cross_track_error(self, x_r: float, y_r: float, x_ref: float, y_ref: float, theta_ref: float) -> float:
        return (y_ref - y_r) * math.cos(theta_ref) - (x_ref - x_r) * math.sin(theta_ref)

    def calc_path_cost(self) -> float:
        """Calculate cross-track error using temp lookahead point"""
        if len(self.traj_list) == 0:
            return 0.0
        
        traj_x, traj_y, _ = self.traj_list[-1]
        return abs(self.cross_track_error(traj_x, traj_y, self.temp_lookahead_x, 
                                         self.temp_lookahead_y, self.temp_lookahead_theta))

    def calc_speed_ref_cost(self, v: float) -> float:
        return abs(v - self.ref_velocity)

    def calc_obstacle_cost(self) -> float:
        """Calculate average obstacle cost along trajectory"""
        if len(self.traj_list) == 0 or not self.costmap_received:
            return 0.0
        
        cost_sum = 0.0
        for pt in self.traj_list:
            mx, my, valid = self.world_to_costmap(pt[0], pt[1], 
                                                   self.traj_list[0][0], 
                                                   self.traj_list[0][1], 
                                                   self.traj_list[0][2])
            if not valid:
                continue
            cost_sum += self.get_costmap_cost(mx, my) / 100.0
        
        avg_cost = cost_sum / max(1, len(self.traj_list))
        return avg_cost

    def calc_away_from_obstacle_cost(self) -> float:
        """Calculate exponential penalty for obstacles"""
        if len(self.traj_list) == 0 or not self.costmap_received:
            return 0.0
        
        total_exp_cost = 0.0
        count = 0
        
        for pt in self.traj_list:
            mx, my, valid = self.world_to_costmap(pt[0], pt[1],
                                                   self.traj_list[0][0],
                                                   self.traj_list[0][1],
                                                   self.traj_list[0][2])
            if not valid:
                continue
            
            c = self.get_costmap_cost(mx, my) / 100.0
            exp_cost = math.exp(c)
            total_exp_cost += exp_cost
            count += 1
        
        return total_exp_cost / max(1, count)

    def dwa_main_control(self, state: RobotState) -> DWAResult:
        dw = self.calc_dynamic_window(state.v, state.omega)
        v_min, v_max, omega_min, omega_max = dw
        
        min_cost = float('inf')
        best_v, best_omega = state.v, state.omega
        max_obstacle_cost = 0.0
        best_lookahead_x = 0.0
        best_lookahead_y = 0.0
        best_lookahead_theta = 0.0
        
        # Visualization setup
        traj_markers = MarkerArray()
        marker_id = 0
        
        v_range = np.linspace(v_min, v_max, self.vx_samples) if self.vx_samples > 1 else [v_min]
        omega_range = np.linspace(omega_min, omega_max, self.omega_samples) if self.omega_samples > 1 else [omega_min]

        iprint = 0

        for v_sample in v_range:
            for omega_sample in omega_range:
                self.traj_list = self.calc_trajectory(state.x, state.y, state.theta, v_sample, omega_sample)
                
                # Calculate costs (lookahead_heading first to populate temp variables)
                lookahead_heading_cost = self.calc_lookahead_heading_cost()
                path_cost = self.calc_path_cost()
                speed_ref_cost = self.calc_speed_ref_cost(v_sample)
                obs_cost = self.calc_obstacle_cost()
                away_cost = self.calc_away_from_obstacle_cost()
                
                total_cost = (self.path_distance_bias * path_cost +
                             self.lookahead_heading_bias * lookahead_heading_cost +
                             self.occdist_scale * obs_cost +
                             self.speed_ref_bias * speed_ref_cost +
                             self.away_bias * away_cost)
                
                # Visualization marker
                traj_marker = Marker()
                traj_marker.header.frame_id = "odom"
                traj_marker.header.stamp = rospy.Time.now()
                traj_marker.ns = "dwa_paths"
                traj_marker.id = marker_id
                marker_id += 1
                traj_marker.type = Marker.LINE_STRIP
                traj_marker.action = Marker.ADD
                traj_marker.pose.orientation.w = 1.0
                traj_marker.scale.x = 0.02
                
                if math.isinf(total_cost):
                    traj_marker.color.r, traj_marker.color.g, traj_marker.color.b, traj_marker.color.a = 1.0, 0.0, 0.0, 1.0
                else:
                    normalized_cost = min(1.0, total_cost / 300.0)
                    traj_marker.color.r = normalized_cost
                    traj_marker.color.g = 1.0 - normalized_cost
                    traj_marker.color.b = 0.0
                    traj_marker.color.a = 0.6
                
                for pt in self.traj_list:
                    p = PointStamped()
                    p.point.x = pt[0]
                    p.point.y = pt[1]
                    p.point.z = 0
                    traj_marker.points.append(p.point)
                
                traj_markers.markers.append(traj_marker)
                
                if obs_cost > max_obstacle_cost:
                    max_obstacle_cost = obs_cost
                
                if total_cost < min_cost:
                    min_cost = total_cost
                    best_v = v_sample
                    best_omega = omega_sample
                    best_lookahead_x = self.temp_lookahead_x
                    best_lookahead_y = self.temp_lookahead_y
                    best_lookahead_theta = self.temp_lookahead_theta

                    iprint += 1
                    if iprint == 1:
                        print(f"T {total_cost:.2f} || "
                            f"CTE {self.path_distance_bias * path_cost:.2f}  | "
                            f"H {self.lookahead_heading_bias * lookahead_heading_cost:.2f}  | "
                            f"OC {self.occdist_scale * obs_cost:.2f}  | "
                            f"SR {self.speed_ref_bias * speed_ref_cost:.2f}  | "
                            f"AB {self.away_bias * away_cost:.2f}")
                        iprint = 0
        
        self.traj_pub.publish(traj_markers)
        
        result = DWAResult()
        result.best_v = best_v
        result.best_omega = best_omega
        result.obs_cost = self.occdist_scale * max_obstacle_cost
        result.lookahead_x = best_lookahead_x
        result.lookahead_y = best_lookahead_y
        result.lookahead_theta = best_lookahead_theta
        
        rospy.loginfo(f"DWA max obstacle avg cost (0-1) = {max_obstacle_cost:.2f}")
        return result

# === Preview Controller Class ===

class PreviewController:
    def __init__(self):
        # Initialize state variables
        self.current_state = RobotState()
        self.targetid = [0]  # Use list to pass by reference to DWA
        self.max_path_points = [0]  # Use list to pass by reference to DWA
        self.initial_pose_received = False
        self.path_generated = False
        self.initial_alignment = False
        self.start_moving = False
        self.use_start_stop = True
        self.active_controller = "DWA"  # Default controller (matches C++)

        # Load parameters
        self.load_params()

        # Setup ROS publishers and subscribers
        self.setup_ros_communications()

        self.vel_acc_bound = self.vel_acc * self.dt
        self.omega_acc_bound = self.omega_acc * self.dt
        self.dwa_controller = None
        self.current_path = []
        self.path_curvatures = []
        
        # Preview control variables
        self.lookahead_heading_error = 0.0

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
        self.robot_radius = rospy.get_param('preview_controller/robot_radius', 0.5)
        self.max_cte = rospy.get_param('preview_controller/max_cte', 1.5)
        
        # Hysteresis thresholds
        self.dwa_activation_cost_thresh = rospy.get_param('preview_controller/dwa_activation_cost_thresh', 10.0)
        self.preview_reactivation_cost_thresh = rospy.get_param('preview_controller/preview_reactivation_cost_thresh', 5.0)

        # Preview control matrices parameters
        self.Q_params = rospy.get_param('preview_controller/Q_params', [5.0, 6.0, 5.0])
        self.R_param = rospy.get_param('preview_controller/R', 1.0)
        self.preview_steps = rospy.get_param('preview_controller/preview_steps', 5)
        self.preview_loop_thresh = rospy.get_param('preview_controller/preview_loop_thresh', 1e-5)

    def setup_ros_communications(self):
        self.robot_vel_pub = rospy.Publisher("/atrv/cmd_vel", Twist, queue_size=10)
        self.path_pub = rospy.Publisher("planned_path", Path, queue_size=10)
        self.lookahead_point_pub = rospy.Publisher("lookahead_point", PoseStamped, queue_size=10)
        
        # Debug publishers (matching C++)
        self.cross_track_error_pub = rospy.Publisher("debug/cross_track_error", Float64, queue_size=10)
        self.heading_error_pub = rospy.Publisher("debug/heading_error", Float64, queue_size=10)
        self.lookahead_heading_error_pub = rospy.Publisher("debug/lookahead_heading_error", Float64, queue_size=10)
        self.current_v_pub = rospy.Publisher("debug/current_v", Float64, queue_size=10)
        self.current_omega_pub = rospy.Publisher("debug/current_omega", Float64, queue_size=10)
        self.path_curvature_pub = rospy.Publisher("debug/path_curvature", Float64, queue_size=10)
        
        self.robot_pose_sub = rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_callback)
        self.start_moving_sub = rospy.Subscriber("/start_moving", Bool, self.start_moving_callback)
        self.stop_moving_sub = rospy.Subscriber("/stop_moving", Bool, self.stop_moving_callback)

    def start_moving_callback(self, msg: Bool):
        if msg.data:
            self.start_moving = True

    def stop_moving_callback(self, msg: Bool):
        if msg.data:
            self.start_moving = False

    def robot_pose_callback(self, msg: PoseStamped):
        self.current_state.x = msg.pose.position.x
        self.current_state.y = msg.pose.position.y
        self.current_state.theta = msg.pose.orientation.z
        
        if not self.initial_pose_received:
            self.initial_pose_received = True
            rospy.loginfo(f"Initial robot pose received: x={self.current_state.x:.2f}, y={self.current_state.y:.2f}")
            if self.path_type == 'snake':
                self.generate_snake_path()
            else:
                self.generate_straight_path()
            
            self.max_path_points[0] = len(self.current_path)
            self.dwa_controller = DWAController(self.current_path, self.targetid, self.max_path_points)
            self.path_generated = True
            self.calculate_all_curvatures()
        
        if self.path_generated:
            self.publish_path()

    def chkside(self, path_theta: float) -> bool:
        """Check if robot is in front of current target point"""
        if self.targetid[0] + 1 >= self.max_path_points[0]:
            return False
        
        x1 = self.current_path[self.targetid[0]].x
        y1 = self.current_path[self.targetid[0]].y
        robot_x = self.current_state.x
        robot_y = self.current_state.y
        
        ineq = 0.0
        t = False
        
        if abs(math.tan(path_theta)) < 1e-6:
            ineq = robot_x - x1
            if ineq > 0:
                t = False
                if -math.pi/2 < path_theta < math.pi/2:
                    t = True
            else:
                t = True
                if -math.pi/2 < path_theta < math.pi/2:
                    t = False
        else:
            m = -1.0 / math.tan(path_theta)
            ineq = robot_y - (m * robot_x) - y1 + (m * x1)
        
        if ineq > 0:
            t = True
            if path_theta < 0:
                t = False
        else:
            t = False
            if path_theta < 0:
                t = True
        
        return t

    def generate_snake_path(self):
        self.current_path = []
        start_x, start_y, start_theta = self.current_state.x, self.current_state.y, self.current_state.theta
        num_points = int(math.ceil(self.path_length / self.path_point_spacing)) + 1
        
        for i in range(num_points):
            x = start_x + (self.path_length * i) / (num_points - 1)
            y = start_y + self.path_amplitude * math.sin(2.0 * math.pi * (x - start_x) / self.path_wavelength)
            dx = 1.0
            dy = self.path_amplitude * (2.0 * math.pi / self.path_wavelength) * math.cos(2.0 * math.pi * (x - start_x) / self.path_wavelength)
            theta = math.atan2(dy, dx)
            while theta > math.pi:
                theta -= 2 * math.pi
            while theta < -math.pi:
                theta += 2 * math.pi
            self.current_path.append(Waypoint(x, y, theta))
        rospy.loginfo(f"Generated snake path with {len(self.current_path)} points.")

    def generate_straight_path(self):
        self.current_path = []
        start_x, start_y, start_theta = self.current_state.x, self.current_state.y, self.current_state.theta
        num_points = int(math.ceil(self.straight_path_distance / self.path_point_spacing)) + 1
        
        for i in range(num_points):
            dist = i * self.path_point_spacing
            if i == num_points - 1:
                dist = self.straight_path_distance
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
            while dtheta > math.pi:
                dtheta -= 2 * math.pi
            while dtheta < -math.pi:
                dtheta += 2 * math.pi
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

        # Solve DARE
        P = Q
        for _ in range(100):
            P_next = Ad.T @ P @ Ad - (Ad.T @ P @ Bd) @ np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad) + Q
            if np.linalg.norm(P_next - P) < self.preview_loop_thresh:
                break
            P = P_next
        
        # Calculate gains
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
            preview_idx = self.targetid[0] + i
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

    def cross_track_error(self, x_r: float, y_r: float, x_ref: float, y_ref: float, theta_ref: float) -> float:
        return (y_ref - y_r) * math.cos(theta_ref) - (x_ref - x_r) * math.sin(theta_ref)

    def lookahead_heading_error_calc(self, x_ref: float, y_ref: float, theta_ref: float):
        """Calculate lookahead heading error (like C++)"""
        self.lookahead_heading_error = self.current_state.theta - math.atan2(
            y_ref - self.current_state.y, x_ref - self.current_state.x)
        while self.lookahead_heading_error > math.pi:
            self.lookahead_heading_error -= 2 * math.pi
        while self.lookahead_heading_error < -math.pi:
            self.lookahead_heading_error += 2 * math.pi

    def run_control(self):
        if not self.path_generated or self.dwa_controller is None:
            return False

        rospy.loginfo("Running run control")
        
        # Advance target until point is in front of robot (chkside)
        while (self.targetid[0] + 1 < self.max_path_points[0] and 
               self.chkside(self.current_path[self.targetid[0]].theta)):
            self.targetid[0] += 1
        
        # Advance target based on lookahead distance
        while (self.targetid[0] + 1 < self.max_path_points[0] and 
               math.hypot(self.current_state.x - self.current_path[self.targetid[0]].x,
                         self.current_state.y - self.current_path[self.targetid[0]].y) < self.lookahead_distance):
            self.targetid[0] += 1
        
        # Advance lookahead if point lies in obstacle cells (cost >= 50)
        if self.dwa_controller.costmap_received:
            while self.targetid[0] + 1 < self.max_path_points[0]:
                lx = self.current_path[self.targetid[0]].x
                ly = self.current_path[self.targetid[0]].y
                c = self.dwa_controller.query_cost_at_world(lx, ly, self.current_state.x, self.current_state.y)
                if c >= 50.0:
                    rospy.logwarn_throttle(1.0, f"Lookahead at idx {self.targetid[0]} in obstacle (cost={c:.1f}). Advancing.")
                    self.targetid[0] += 1
                    continue
                break
        
        target_pt = self.current_path[self.targetid[0]]

        # Calculate cross-track error
        cross_track_error = self.cross_track_error(self.current_state.x, self.current_state.y,
                                                    target_pt.x, target_pt.y, target_pt.theta)
        
        # Publish cross-track error
        self.cross_track_error_pub.publish(Float64(data=cross_track_error))
        
        # Calculate lookahead heading error
        self.lookahead_heading_error_calc(target_pt.x, target_pt.y, target_pt.theta)
        
        # Publish lookahead heading error
        self.lookahead_heading_error_pub.publish(Float64(data=self.lookahead_heading_error))
        
        # Initial alignment
        if not self.initial_alignment:
            if abs(self.lookahead_heading_error) < self.max_lookahead_heading_error:
                self.initial_alignment = True
            else:
                self.bound_vel(0.0)
                self.bound_omega(-self.kp_adjust_cte * self.lookahead_heading_error)
                rospy.loginfo(f"Adjusting Lookahead Heading Error: {self.lookahead_heading_error:.3f}")
                self.publish_cmd_vel()
                return False
        
        # Calculate goal distance and reduce velocity if close
        goal_pt = self.current_path[-1]
        goal_distance = math.hypot(self.current_state.x - goal_pt.x, self.current_state.y - goal_pt.y)
        
        if goal_distance < 1.0:
            self.bound_vel(goal_distance * self.linear_velocity * self.goal_reduce_factor)
        else:
            self.bound_vel(self.linear_velocity)

        # Evaluate DWA
        dwa_result = self.dwa_controller.dwa_main_control(self.current_state)
        rospy.loginfo(f"DWA max obstacle cost = {dwa_result.obs_cost:.2f}")

        # Hysteresis switching
        use_preview = False
        if self.active_controller == "PREVIEW":
            if dwa_result.obs_cost > self.dwa_activation_cost_thresh:
                use_preview = False
                rospy.logwarn(f"SWITCH: PREVIEW -> DWA (cost {dwa_result.obs_cost:.2f} > {self.dwa_activation_cost_thresh:.2f})")
            else:
                use_preview = True
        else:  # DWA
            if dwa_result.obs_cost <= self.preview_reactivation_cost_thresh:
                use_preview = True
                rospy.logwarn(f"SWITCH: DWA -> PREVIEW (cost {dwa_result.obs_cost:.2f} <= {self.preview_reactivation_cost_thresh:.2f})")
            else:
                use_preview = False
        
        if not self.dwa_controller.costmap_received:
            use_preview = False

        if not use_preview:
            # Use DWA
            self.current_state.v = dwa_result.best_v
            self.current_state.omega = dwa_result.best_omega
            self.active_controller = "DWA"
            rospy.loginfo(f"Controller: DWA | v={self.current_state.v:.3f}, omega={self.current_state.omega:.3f}")
        else:
            # Use Preview
            self.bound_vel(self.linear_velocity)
            heading_error = self.lookahead_heading_error
            
            # Publish heading error
            self.heading_error_pub.publish(Float64(data=heading_error))
            
            omega_ref = self.compute_control(cross_track_error, heading_error)
            self.bound_omega(omega_ref)
            
            # Publish path curvature
            if self.targetid[0] < len(self.path_curvatures):
                path_curvature = self.path_curvatures[self.targetid[0]]
            else:
                path_curvature = 0.0
            self.path_curvature_pub.publish(Float64(data=path_curvature))
            
            self.active_controller = "PREVIEW"
            rospy.loginfo(f"Controller: PREVIEW | v={self.current_state.v:.3f}, omega={self.current_state.omega:.3f}")
        
        # Publish current velocity and omega
        self.current_v_pub.publish(Float64(data=self.current_state.v))
        self.current_omega_pub.publish(Float64(data=self.current_state.omega))
        
        # Publish cmd_vel
        self.publish_cmd_vel()
        
        # Publish lookahead point based on active controller
        look_pose = PoseStamped()
        look_pose.header.stamp = rospy.Time.now()
        look_pose.header.frame_id = "odom"
        
        if self.active_controller == "DWA":
            # Use DWA's lookahead point
            look_pose.pose.position.x = dwa_result.lookahead_x
            look_pose.pose.position.y = dwa_result.lookahead_y
            look_pose.pose.position.z = 0.0
            look_pose.pose.orientation.z = dwa_result.lookahead_theta
        else:
            # Use Preview's lookahead point
            look_pose.pose.position.x = target_pt.x
            look_pose.pose.position.y = target_pt.y
            look_pose.pose.position.z = 0.0
            look_pose.pose.orientation.z = target_pt.theta
        
        self.lookahead_point_pub.publish(look_pose)

        # Check for goal completion
        if goal_distance < self.goal_distance_threshold:
            self.stop_robot()
            rospy.loginfo("Goal reached!")
            return True
        return False

    def publish_cmd_vel(self):
        cmd = Twist()
        cmd.linear.x = self.current_state.v
        cmd.angular.z = np.clip(self.current_state.omega, -self.max_omega, self.max_omega)
        self.robot_vel_pub.publish(cmd)
        
    def stop_robot(self):
        self.current_state.v = 0.0
        self.current_state.omega = 0.0
        self.publish_cmd_vel()

    def publish_path(self):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "odom"
        for i in range(self.targetid[0], len(self.current_path)):
            wp = self.current_path[i]
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp.x
            pose.pose.position.y = wp.y
            pose.pose.orientation.z = math.sin(wp.theta / 2.0)
            pose.pose.orientation.w = math.cos(wp.theta / 2.0)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def spin(self):
        rate = rospy.Rate(1.0 / self.dt)
        while not rospy.is_shutdown():
            if self.start_moving or not self.use_start_stop:
                if self.path_generated:
                    goal_reached = self.run_control()
                    if goal_reached:
                        break
            rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('control_test_algo_node')
        controller = PreviewController()
        controller.spin()
    except rospy.ROSInterruptException:
        pass 