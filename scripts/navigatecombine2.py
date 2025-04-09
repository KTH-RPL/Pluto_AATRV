#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
# from tf.transformations import euler_from_quaternion
from local_planner import execute_planning

class NavigationSystem:
    def __init__(self):
        rospy.init_node('pluto_navigation_system', anonymous=False, disable_signals=True)        
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback)
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.robot_pose_callback)
        
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)
        

        self.circle_radius = 2.0  
        self.point_spacing = 0.2  
        self.circle_angle = 2 * np.pi  

        self.lookahead_distance = 0.75  
        self.k_angular = 1.2           
        self.v_max = 0.3              
        self.v_min = 1,0              
        self.goal_distance_threshold = 0.3
        self.slow_down_distance = 1.5  
        self.min_lookahead = 0.5       
        self.max_lookahead = 2.0       
        
        self.current_goal = None
        self.current_pose = None
        self.current_path = None
        self.current_headings = None
        self.closest_idx = 0           
        self.control_rate = rospy.Rate(35)  
        
        rospy.loginfo("Pluto Navigation System initialized")

    def goal_callback(self, msg):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"New goal received: {self.current_goal}")
        self.current_path = None  
        self.closest_idx = 0      

    def robot_pose_callback(self, msg):
        self.current_pose = msg
        # print("CUrrent ",self.current_path)
        if self.current_path is None:
            self.generate_circular_path()

    def get_yaw_from_pose(self, pose):
        yaw = pose.pose.orientation.z
        return yaw
    def generate_circular_path(self):
        if self.current_pose is None:
            return
            
        x0 = self.current_pose.pose.position.x
        y0 = self.current_pose.pose.position.y
        theta0 = self.get_yaw_from_pose(self.current_pose)
        
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

    def publish_path(self, path_points,headings):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for point,head in zip(path_points,headings):
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = head
            
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
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
        self.current_headings = None  
        rospy.loginfo(f"Path published with {len(path_msg.poses)} waypoints")

    def find_closest_point(self, path, current_pos):
        distances = [np.sqrt((p[0]-current_pos[0])**2 + (p[1]-current_pos[1])**2) for p in path]
        return np.argmin(distances)

    def find_lookahead_point(self, path, current_pos, closest_idx):
        lookahead_dist = self.lookahead_distance
        
        for i in range(closest_idx, len(path)):
            dist = np.sqrt((path[i][0] - current_pos[0])**2 + (path[i][1] - current_pos[1])**2)
            if dist >= lookahead_dist:
                return path[i], i
        
        return path[-1], len(path) - 1

    def prune_passed_points(self, path, closest_idx):
        return path[closest_idx:]

    def run_control(self):
        while not rospy.is_shutdown():
            if self.current_path is not None and self.current_pose is not None:
                x_robot = self.current_pose.pose.position.x
                y_robot = self.current_pose.pose.position.y
                current_pos = (x_robot, y_robot)
                theta_robot = self.get_yaw_from_pose(self.current_pose)
                print("Current pose ",x_robot,y_robot)
                self.closest_idx = self.find_closest_point(self.current_path, current_pos)
                
                remaining_path = self.prune_passed_points(self.current_path, self.closest_idx)
                
                x_goal, y_goal = self.current_path[-1][0], self.current_path[-1][1]
                goal_distance = np.sqrt((x_goal - x_robot)**2 + (y_goal - y_robot)**2)
                
                if goal_distance < self.goal_distance_threshold:
                    cmd_vel = Twist()  
                    self.cmd_vel_pub.publish(cmd_vel)
                    rospy.loginfo("Goal reached!")
                    self.current_path = None
                    self.control_rate.sleep()
                    continue
                
                lookahead_point, lookahead_idx = self.find_lookahead_point(remaining_path, current_pos, 0)
                
                heading_ref = self.current_headings[lookahead_idx]
                
                if goal_distance < self.slow_down_distance:
                    v = self.v_min + (self.v_max - self.v_min) * (goal_distance / self.slow_down_distance)
                else:
                    v = self.v_max
                
                heading_error = heading_ref - theta_robot
                heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi  
                
                omega = self.k_angular * heading_error
                
                max_omega = 0.8 + 1.5 * v
                omega = np.clip(omega, -max_omega, max_omega)
                
                cmd_vel = Twist()
                cmd_vel.linear.x = v
                cmd_vel.angular.z = omega
                self.cmd_vel_pub.publish(cmd_vel)
                
                rospy.loginfo(f"V: {v:.2f}, Omega: {omega:.2f}, Heading error: {np.degrees(heading_error):.1f}°")

            self.control_rate.sleep()

if __name__ == '__main__':
    try:
        nav_system = NavigationSystem()      
        nav_system.run_control()
    except rospy.ROSInterruptException:
        pass