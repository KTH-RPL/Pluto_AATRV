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
from scipy.ndimage import gaussian_filter1d

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
        self.Q = np.diag([5, 6, 5])  
        self.R = np.array([[1]])
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

    def compute_control(self, x_r, y_r, theta_r, path_x, path_y, path_curv,vel):
        self.v = vel
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
        rospy.loginfo(f"curve {path_curv}")
        if len(preview_curv) < self.preview_steps + 1:
            preview_curv = np.pad(preview_curv, (0, self.preview_steps + 1 - len(preview_curv)), 'edge')

        u_fb = -self.Kb @ x_state
        u_ff = -self.Kf @ preview_curv
        omega = u_fb + u_ff

        return omega.item(), x_state, u_fb.item(), ey



class NavigationSystem:
    def __init__(self):
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.cmd_vel = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)
        self.look_pub = rospy.Publisher('/lookahead_point', PoseStamped, queue_size=10)

        self.lookahead_distance = 1.0      
        self.v_max = 0.3            
        self.v_min = 0.1       
        self.max_omega = 0.7     
        self.goal_distance_threshold = 0.2
        self.slow_down_distance = 1.2 
        
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
        self.fp = False
        self.targetid = 0

        self.controller = PreviewController(v=self.v_max, dt=0.05, preview_steps=3)
        
        rospy.loginfo("Pluto Navigation System initialized with Preview Control")
        
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
        if dist < 0.5:
            return 1
        else:
            return 0

    def chkside(self,cp,pose,orientation):
        y1 = self.current_path[cp][1]
        x1 = self.current_path[cp][0]
        y2 = self.current_path[cp+1][1]
        x2 = self.current_path[cp+1][0]
        m = -(x2 - x1)/(y1 - y2)
        ineq = pose[1] - y1 + (pose[0]/m) - (x1/m)
        if ineq > 0:
            if orientation < 0:
                return 0
            else:
                return 1
        else:
            if orientation < 0:
                return 1
            else:
                return 0

    def run_control(self, is_last_goal=False):
        x_robot = self.current_pose.pose.position.x
        y_robot = self.current_pose.pose.position.y
        current_pos = (x_robot, y_robot)

        theta_robot = self.current_pose.pose.orientation.z
        
        while( self.targetid + 1< len(self.current_path) and self.distancecalc(self.current_path[self.targetid],current_pos)):
            self.targetid+=1
        while(self.targetid + 1< len(self.current_path) and self.chkside(self.targetid,current_pos,self.current_path[self.targetid][2])):
            self.targetid+=1

        rospy.loginfo(f"target id {(self.current_path)}")
        self.current_path = self.current_path[self.targetid:][:]
        self.current_curvatures = self.current_curvatures[self.targetid:]
        self.targetid = 0
        
        x_ref = self.current_path[0][0]
        y_ref = self.current_path[0][1]


        x_goal, y_goal = self.current_path[-1][0], self.current_path[-1][0]
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
            self.current_curvatures,
            self.v
        ) 
        
        cmd_vel = Twist()
        cmd_vel.linear.x = self.v
        if omega < -self.max_omega:
            omega = -self.max_omega
        elif omega > self.max_omega:
            omega = self.max_omega
        cmd_vel.angular.z = omega

        self.publish_look_pose(self.current_path[0][0], self.current_path[0][1])
        self.cmd_vel.publish(cmd_vel)

        if goal_distance < self.goal_distance_threshold:
            self.stop_robot()
            rospy.loginfo("Goal reached!")
            self.reached = True
            return True
            
        return False
    


    def calculate_curvature(self,x, y):
        curvatures = np.zeros(len(x))
        for i in range(1, len(x)-1):
            x1,y1 = x[i],y[i]
            x2,y2 = x[i-1],y[i-1]
            x3,y3 = x[i+1],y[i+1]

            A = 2 * (x2 - x1)
            B = 2 * (y2 - y1)
            C = x2**2 + y2**2 - x1**2 - y1**2
            D = 2 * (x3 - x1)
            E = 2 * (y3 - y1)
            F = x3**2 + y3**2 - x1**2 - y1**2

            denominator = A * E - B * D
            if denominator < 1e-4:
                curvatures[i] = 0
            else:
                
                h = (C * E - B * F) / denominator
                k = (A * F - C * D) / denominator

                r = np.sqrt((x1 - h)**2 + (y1 - k)**2)

                if r > 1e-6:
                    curvature_magnitude = 1 / r
                    dx1, dy1 = x2 - x1, y2 - y1  
                    dx2, dy2 = x3 - x1, y3 - y1  
                    cross_product = dx1 * dy2 - dy1 * dx2
                    sign = 1 if cross_product > 0 else -1  
                    curvatures[i] = sign * curvature_magnitude
                else:
                    curvatures[i] = 0

        curvatures[0] = curvatures[1]
        curvatures[-1] = curvatures[-2]
        # curvatures = gaussian_filter1d(curvatures, sigma=3)
        return curvatures



    # def calculate_curvature(self,x, y):
    #     curvatures = np.zeros(len(x))
    #     for i in range(1, len(x)-1):
    #         dx1 = x[i] - x[i-1]
    #         dy1 = y[i] - y[i-1]
    #         dx2 = x[i+1] - x[i]
    #         dy2 = y[i+1] - y[i]

    #         angle1 = np.arctan2(dy1, dx1)
    #         angle2 = np.arctan2(dy2, dx2)
    #         dtheta = angle2 - angle1
    #         dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))

    #         dist = np.hypot(dx1, dy1)
    #         if dist > 1e-6:
    #             curvatures[i] = dtheta / dist

    #     curvatures[0] = curvatures[1]
    #     curvatures[-1] = curvatures[-2]
    #     return curvatures

if __name__ == '__main__':
    try:
        rospy.init_node('navigation_system')
        nav_system = NavigationSystem()      
        nav_system.run_control()
    except rospy.ROSInterruptException:
        pass