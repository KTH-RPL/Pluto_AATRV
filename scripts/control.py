#!/usr/bin/env python3
import rospy
import numpy as np
import csv
import os
from datetime import datetime
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
from local_planner import execute_planning

class NavigationSystem:
    def __init__(self):
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback)
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.robot_pose_callback)
        
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.cmd_vel = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)
        
        # Initialize data recording
        self.recording_file = None
        self.csv_writer = None
        self.setup_data_recording()
        
        # Control parameters
        self.lookahead_distance = 1.5
        self.k_angular = 1.5         
        self.v_max = 0.4             
        self.v_min = 0.1            
        self.goal_distance_threshold = 0.2
        self.slow_down_distance = 1.0 
        self.min_lookahead = 1.2      
        self.max_lookahead = 1.5    
        self.pathgen = False
        self.current_goal = None
        self.current_pose = None
        self.current_path = None
        self.current_headings = None
        self.closest_idx = 0           
        self.control_rate = rospy.Rate(20)  
        self.poserec = False
        self.goalrec = False
        self.gen = False
        self.finished = False
        self.reached = False
        self.fp = True
        rospy.loginfo("Pluto Navigation System initialized")

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
        
        closest_idx, closest_point = min(ahead_points, 
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


    def chkside(self,cp,pose):
        y1 = self.current_path[cp][1]
        x1 = self.current_path[cp][0]
        y2 = self.current_path[cp+1][1]
        x2 = self.current_path[cp+1][0]
        m = -(x2 - x1)/(y1 - y2)
        ineq = pose[1] - y1 + (pose[0]/m) - (x1/m)
        if ineq > 0:
            return 1
        else:
            return 0


    def stop_robot(self):
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = 0
        self.cmd_vel.publish(cmd_vel)



    def run_control(self, is_last_goal=False): 

        x_robot = self.current_pose.pose.position.x
        y_robot = self.current_pose.pose.position.y
        current_pos = (x_robot, y_robot)
        theta_robot = self.current_pose.pose.orientation.z

        self.closest_idx = self.find_closest_point(self.current_path, current_pos)
        closest_point = self.current_path[self.closest_idx]

        if self.closest_idx + 1 < len(self.current_path):
            side = self.chkside(self.closest_idx, current_pos)
            remid = self.closest_idx + side
        else:
            remid = self.closest_idx

        remaining_path = self.prune_passed_points(self.current_path, remid)

        x_goal, y_goal = self.current_path[-1][0], self.current_path[-1][1]
        goal_distance = np.sqrt((x_goal - x_robot) ** 2 + (y_goal - y_robot) ** 2)

        self.record_data(self.current_pose, closest_point, self.closest_idx, goal_distance)

        # if goal_distance < self.goal_distance_threshold:
        #     cmd_vel = Twist()
        #     cmd_vel.linear.x = 0
        #     cmd_vel.angular.z = 0
        #     self.cmd_vel.publish(cmd_vel)
        #     rospy.loginfo("Goal reached!")
        #     self.reached = True
        #     return True  

        lookahead_point, lookahead_idx = self.find_lookahead_point(remaining_path, current_pos, 0)
        actual_lookahead_idx = self.closest_idx + lookahead_idx
        heading_ref = self.current_headings[actual_lookahead_idx]

        heading_error = heading_ref - theta_robot

        if self.fp:
            if np.abs(heading_error) < np.pi / 2:
                self.fp = False
                v = self.v_max
            else:
                v = 0
        else:
            slow_down_dist = self.slow_down_distance
            if goal_distance < slow_down_dist and is_last_goal == True:
                v = self.v_min + (self.v_max - self.v_min) * (goal_distance / slow_down_dist)
            else:
                v = self.v_max

        omega = self.k_angular * heading_error
        omega = np.clip(omega, -1.2, 1.2)

        cmd_vel = Twist()
        cmd_vel.linear.x = v
        cmd_vel.angular.z = omega
        self.cmd_vel.publish(cmd_vel)

        # self.control_rate.sleep()

        # return False  


if __name__ == '__main__':
    try:
        nav_system = NavigationSystem()      
        nav_system.run_control()
    except rospy.ROSInterruptException:
        pass
