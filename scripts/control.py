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
        rospy.init_node('pluto_navigation_system', anonymous=False, disable_signals=True)        
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
        self.k_angular = 3.0           
        self.v_max = 0.4             
        self.v_min = 0.1            
        self.goal_distance_threshold = 0.2
        self.slow_down_distance = 1.0 
        self.min_lookahead = 1.2      
        self.max_lookahead = 1.5    
        
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
        

    def plan_path(self):
        if self.current_pose is None or self.current_goal is None:
            rospy.logwarn("Cannot plan path - missing pose or goal")
            return
            
        current_position = (self.current_pose.pose.position.x, 
                          self.current_pose.pose.position.y)
        
        rospy.loginfo("Planning new path...")
        path_points, _, _, _ = execute_planning(current_position, self.current_goal)
        
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for point in path_points:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.current_path = path_points
        rospy.loginfo("Path published with {} waypoints".format(len(path_msg.poses)))
  
    def generate_circular_path(self):
        if self.current_pose is None:
            return
            
        x0 = self.current_pose.pose.position.x
        y0 = self.current_pose.pose.position.y
        theta0 = self.current_pose.pose.orientation.z
        self.circle_radius
        circumference = 2 * np.pi * self.circle_radius
        num_points = int(circumference / self.point_spacing)
        
        path_points = []
        headings = []
        
        for i in range(num_points + 1):
            angle = (i / num_points) * self.circle_angle
            
            x = x0 + self.circle_radius * np.cos(theta0 + angle + np.pi/2)
            y = y0 + self.circle_radius * np.sin(theta0 + angle + np.pi/2)
            
            heading = theta0 + angle + np.pi  
            
            heading = (heading + np.pi) % (2 * np.pi) - np.pi
            
            path_points.append([x, y])
            headings.append(heading)
        
        self.current_path = np.array(path_points)
        self.current_headings = np.array(headings)
        self.closest_idx = 0
        
        self.publish_path(path_points,headings)
        rospy.loginfo(f"Generated circular path with {len(path_points)} points and is {(path_points)}")

    def generate_offset_path(self):
        if self.current_pose is None:
            return
            
        x0 = self.current_pose.pose.position.x
        y0 = self.current_pose.pose.position.y
        theta0 = self.current_pose.pose.orientation.z
        
        offset_angle = np.radians(30)
        
        path_points = []
        headings = []
        heading = theta0 + offset_angle
        for i in range(1, 6):
            distance = 1.2 * i
            x = x0 + distance * np.cos(heading)
            y = y0 + distance * np.sin(heading)
            
            heading = heading + offset_angle  
            
            heading = (heading + np.pi) % (2 * np.pi) - np.pi
            
            path_points.append([x, y])
            headings.append(heading)
        
        self.current_path = np.array(path_points)
        self.current_headings = np.array(headings)
        self.closest_idx = 0
        
        self.publish_path(path_points, headings)
        rospy.loginfo(f"Generated offset path with {len(path_points)} points and {path_points}")

    def publish_path(self, path_points, headings):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for point, head in zip(path_points, headings):
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.z = head
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

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

    def run_control(self):
        try:
            while not rospy.is_shutdown():        
                x_robot = self.current_pose.pose.position.x
                y_robot = self.current_pose.pose.position.y
                current_pos = (x_robot, y_robot)
                theta_robot = self.current_pose.pose.orientation.z
                
                
                self.closest_idx = self.find_closest_point(self.current_path, current_pos)
                closest_point = self.current_path[self.closest_idx]
                if self.closest_idx + 1 < len(self.current_path):
                    side = self.chkside(self.closest_idx,current_pos)
                    remid = self.closest_idx + side
                else:
                    remid = self.closest_idx
                remaining_path = self.prune_passed_points(self.current_path, remid)
                
                
                x_goal, y_goal = self.current_path[-1][0], self.current_path[-1][1]
                goal_distance = np.sqrt((x_goal - x_robot)**2 + (y_goal - y_robot)**2)
                
                self.record_data(self.current_pose, closest_point, self.closest_idx, goal_distance)
                
                if goal_distance < self.goal_distance_threshold:
                    cmd_vel = Twist()  
                    cmd_vel.linear.x = 0
                    cmd_vel.angular.z = 0
                    self.cmd_vel.publish(cmd_vel)
                    rospy.loginfo("Goal reached!")
                    self.reached = True
                    break
                
                lookahead_point, lookahead_idx = self.find_lookahead_point(
                    remaining_path, current_pos, 0)
                
                actual_lookahead_idx = self.closest_idx + lookahead_idx
                heading_ref = self.current_headings[actual_lookahead_idx]                    

                heading_error = heading_ref - theta_robot
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                
                print("Error",heading_error)                       
                if self.fp == True:                            
                    if np.abs(heading_error) < np.pi/2:
                        self.fp = False
                        v = self.v_max
                    else:
                        v = 0
                elif goal_distance < self.slow_down_distance:
                    v = self.v_min + (self.v_max - self.v_min) * (goal_distance / self.slow_down_distance)
                else:
                    v = self.v_max
                omega = self.k_angular * heading_error
                max_omega = 1.2
                omega = np.clip(omega, -max_omega, max_omega)
                print("omega",omega)
                cmd_vel = Twist()
                cmd_vel.linear.x = v
                cmd_vel.angular.z = omega
                self.cmd_vel.publish(cmd_vel)
            

            self.control_rate.sleep()
                
        finally:
            if self.recording_file is not None:
                self.recording_file.close()
                rospy.loginfo("Data recording file closed")

if __name__ == '__main__':
    try:
        nav_system = NavigationSystem()      
        nav_system.run_control()
    except rospy.ROSInterruptException:
        pass
