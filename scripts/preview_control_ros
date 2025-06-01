#!/usr/bin/env python3
import rospy
import numpy as np
import csv
import os
from datetime import datetime
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
from scipy.interpolate import CubicSpline, interp1d
from scipy.linalg import solve_discrete_are

class PreviewController:
    def __init__(self, v=1.0, dt=0.1, preview_steps=20):
        self.v = v
        self.dt = dt
        self.preview_steps = preview_steps
        self.kd = 2
        self.etheta_threshold = 0.2
        self.A = np.array([[0, 1, 0], [0, 0, v], [0, 0, 0]])
        self.B = np.array([[0], [0], [1]])
        self.D = np.array([[0], [-v**2], [-v]])
        self.Q = np.diag([10, 0.01, 0.1])  
        self.R = np.array([[0.01]])
        self.calc_gains()
        self.prev_ey = 0
        self.prev_etheta = 0
        self.max_domega = 0.2
        self.prev_omega = 0

    def calc_gains(self):
        A_d = np.eye(3) + self.A * self.dt
        B_d = self.B * self.dt
        D_d = self.D * self.dt
        Q_d = self.Q * self.dt
        R_d = self.R / self.dt

        P = solve_discrete_are(A_d, B_d, Q_d, R_d)
        lambda0 = A_d.T @ np.linalg.inv((np.eye(3) + P @ B_d @ np.linalg.inv(R_d) @ B_d.T))
        self.Kb = np.linalg.inv(R_d + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
        self.Pc = np.zeros((3, self.preview_steps+1))
        for i in range(self.preview_steps + 1):
            Pc_column = (np.linalg.matrix_power(lambda0, i) @ P @ D_d)
            self.Pc[:, i] = Pc_column.flatten() 
        
        top = np.hstack([np.zeros((self.preview_steps, 1)), np.eye(self.preview_steps)])
        bottom = np.zeros((1, self.preview_steps+1))
        self.Lmatrix = np.vstack([top, bottom])
        Kf_term = P @ D_d + self.Pc @ self.Lmatrix
        self.Kf = np.linalg.inv(R_d + B_d.T @ P @ B_d) @ B_d.T @ Kf_term

    def compute_control(self, x_r, y_r, theta_r, path_x, path_y, path_curv):
        x_ref = path_x
        y_ref = path_y
        theta_ref = np.arctan2(y_ref - y_r, x_ref - x_r)

        ey = np.sin(theta_r) * (x_r - x_ref) - np.cos(theta_r) * (y_r - y_ref)
        etheta = theta_r - theta_ref
        eydot = self.v * etheta

        self.prev_ey = ey
        self.prev_etheta = etheta
        x_state = np.array([ey, eydot, etheta])
        
        preview_curv = path_curv[0:0 + self.preview_steps + 1]
        if len(preview_curv) < self.preview_steps + 1:
            preview_curv = np.pad(preview_curv, (0, self.preview_steps + 1 - len(preview_curv)), 'edge')

        u_fb = -self.Kb @ x_state
        u_ff = -self.Kf @ preview_curv
        omega = u_fb + u_ff

        return omega.item(), x_state, u_fb.item(), ey

def calculate_curvature(x, y):
    curvatures = np.zeros(len(x))
    for i in range(1, len(x)-1):
        dx1 = x[i] - x[i-1]
        dy1 = y[i] - y[i-1]
        dx2 = x[i+1] - x[i]
        dy2 = y[i+1] - y[i]

        angle1 = np.arctan2(dy1, dx1)
        angle2 = np.arctan2(dy2, dx2)
        dtheta = angle2 - angle1
        dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

        dist = np.hypot(dx1, dy1)
        if dist > 1e-6:
            curvatures[i] = dtheta / dist

    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    return curvatures

class NavigationSystem:
    def __init__(self):
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.cmd_vel = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)
        self.look_pub = rospy.Publisher('/lookahead_point', PoseStamped, queue_size=10)

        self.lookahead_distance = 1.0      
        self.v_max = 0.7            
        self.v_min = 0.4            
        self.goal_distance_threshold = 0.2
        self.slow_down_distance = 0.5 
        
        self.current_goal = None
        self.current_pose = None
        self.current_path = None
        self.current_headings = None
        self.current_curvatures = None
        self.control_rate = rospy.Rate(20)  
        self.poserec = False
        self.goalrec = False
        self.gen = False
        self.finished = False
        self.reached = False
        self.fp = True
        
        self.controller = PreviewController(v=self.v_max, dt=0.05, preview_steps=5)
        
        rospy.loginfo("Pluto Navigation System initialized with Preview Control")

    def find_closest_point(self, path, current_pos):
        robot_x, robot_y = current_pos
        theta_robot = self.current_pose.pose.orientation.z
        
        ahead_points = []
        for i, point in enumerate(path):
            dx = point[0] - robot_x
            dy = point[1] - robot_y            
            
            if (dx * np.cos(theta_robot) + (dy * np.sin(theta_robot))) > -0.1:  
                ahead_points.append((i, point))
        
        if not ahead_points:
            return 0  
        
        closest_idx, _ = min(ahead_points, 
                                      key=lambda x: np.sqrt((x[1][0]-robot_x)**2 + (x[1][1]-robot_y)**2))
        return closest_idx

    def prune_passed_points(self, path, closest_idx):
        return path[closest_idx:]

    def find_lookahead_point(self, path, current_pos, closest_idx):
        lookahead_dist = self.lookahead_distance
        
        for i in range(closest_idx, len(path)):
            dist = np.sqrt((path[i][0] - current_pos[0])**2 + (path[i][1] - current_pos[1])**2)
            if dist >= lookahead_dist:
                return path[i], i
        
        return path[-1], len(path) - 1
        
    def publish_look_pose(self, x, y):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        self.look_pub.publish(pose)

    def stop_robot(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = 0
        self.cmd_vel.publish(cmd_vel)

    def distancecalc(self, x1, x2):
        dist = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        if dist < 1.2:
            return 1
        else:
            return 0

    def prepare_path_data(self, path):
        x_path = np.array([p[0] for p in path])
        y_path = np.array([p[1] for p in path])
        
        dx = np.gradient(x_path)
        dy = np.gradient(y_path)
        headings = np.arctan2(dy, dx)
        
        curvatures = calculate_curvature(x_path, y_path)
        
        return x_path, y_path, headings, curvatures

    def run_control(self, is_last_goal=False):
        x_robot = self.current_pose.pose.position.x
        y_robot = self.current_pose.pose.position.y
        current_pos = (x_robot, y_robot)

        theta_robot = self.current_pose.pose.orientation.z
        
        while(self.distancecalc(self.current_path[self.targetid],current_pos) and self.targetid + 1< len(self.current_path)):
            self.targetid+=1
        self.current_path = self.current_path[self.targetid:]
        self.current_curvatures = self.current_curvatures[self.targetid:]
        self.targetid = 0
        self.closest_idx = self.find_closest_point(self.current_path, current_pos)
        
        x_ref = self.current_path[0][0]
        y_ref = self.current_path[0][1]


        x_goal, y_goal = self.current_path[-1][0], self.current_path[-1][1]
        goal_distance = np.sqrt((x_goal - x_robot) ** 2 + (y_goal - y_robot) ** 2)
        
        if self.fp:
            if np.abs(ey) < 0.5:  
                self.fp = False
                self.v = self.v_max
            else:
                self.v = 0
        else:
            slow_down_dist = self.slow_down_distance
            if goal_distance < slow_down_dist and is_last_goal:
                self.v = self.v_min + (self.v_max - self.v_min) * (goal_distance / slow_down_dist)
            else:
                self.v = self.v_max

        
        omega, _, _, ey = self.controller.compute_control(
            x_robot, y_robot, theta_robot,
            x_ref,y_ref,
            self.current_curvatures
        )
        
        cmd_vel = Twist()
        cmd_vel.linear.x = self.v
        cmd_vel.angular.z = np.clip(omega, -1.2, 1.2)
        self.cmd_vel.publish(cmd_vel)
        
        lookahead_point, _ = self.find_lookahead_point(self.current_path, current_pos, self.closest_idx)
        self.publish_look_pose(lookahead_point[0], lookahead_point[1])
        
        if goal_distance < self.goal_distance_threshold:
            self.stop_robot()
            rospy.loginfo("Goal reached!")
            self.reached = True
            return True
            
        return False

if __name__ == '__main__':
    try:
        rospy.init_node('navigation_system')
        nav_system = NavigationSystem()      
        nav_system.run_control()
    except rospy.ROSInterruptException:
        pass