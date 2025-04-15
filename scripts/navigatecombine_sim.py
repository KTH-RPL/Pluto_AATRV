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
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback) #goal in map frame
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.robot_pose_callback) #pose in map frame
        self.pose_offset_sub = rospy.Subscriber('/pose_offset', PoseStamped, self.pose_offset_callback)

        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.global_path_pub = rospy.Publisher('/global_planned_path', Path, queue_size=10) #path in map frame
        self.local_robot_pose_pub = rospy.Publisher('/local_robot_pose', Path, queue_size=10) #path in map frame
        self.local_goal_pose_pub = rospy.Publisher('/local_goal_pose', Path, queue_size=10) #path in map frame
        self.cmd_vel_pub = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)

        
        # Initialize data recording
        self.recording_file = None
        self.csv_writer = None
        self.setup_data_recording()
        
        # Control parameters
        self.lookahead_distance = 1.0
        self.k_angular = 2.0           
        self.v_max = 0.4             
        self.v_min = 0.1            
        self.goal_distance_threshold = 0.2
        self.slow_down_distance = 1.0 
        self.min_lookahead = 0.5       
        self.max_lookahead = 1.0    
        
        self.current_goal = None
        self.current_pose = None
        self.unrotated_current_pose = None
        self.current_path = None
        self.current_headings = None
        self.closest_idx = 0           
        self.control_rate = rospy.Rate(20)  
        self.poserec = False
        self.goalrec = False
        self.gen = False
        self.finished = False
        self.reached = False

        self.start_x = None
        self.start_y = None
        self.yaw_offset = None

        rospy.loginfo("Pluto Navigation System initialized")



    def setup_data_recording(self):
        # Create a directory for logs if it doesn't exist
        log_dir = os.path.join(os.path.expanduser('~'), 'robot_navigation_logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(log_dir, f'nav_data_{timestamp}.csv')
        
        self.recording_file = open(filename, 'w')
        self.csv_writer = csv.writer(self.recording_file)
        # Write header
        self.csv_writer.writerow([
            'timestamp', 
            'robot_x', 
            'robot_y', 
            'robot_heading', 
            'closest_path_x', 
            'closest_path_y',
            'closest_path_heading',
            'closest_idx',
            'goal_distance',
            'heading_ref',
            'heading_err',
            'v',
            'omega',
            'start_x',
            'start_y',
            'yaw_offset',
            'remaining_path'
        ])
        rospy.loginfo(f"Data recording started at {filename}")

    def record_data(self, robot_pose, closest_point, closest_idx, goal_distance, heading_ref, heading_error, v, omega, remaining_path):
        # Data to Record:
        # 1/ Current pose (x, y, theta)
        # 2/ Closest point (idx, x, y, theta)
        # 3/ Goal distance
        # 4/ Heading Ref & Heading Error
        # 5/ Commands (v, omega)
        # 6/ Initial offsets
        # 7/ Remaining Paths
        
        if self.csv_writer is None:
            return
            
        timestamp = rospy.Time.now().to_sec()
        robot_x = robot_pose.pose.position.x
        robot_y = robot_pose.pose.position.y
        robot_heading =robot_pose.pose.orientation.z
        
        if closest_point is not None and self.current_headings is not None:
            closest_x, closest_y = closest_point
            closest_heading = self.current_headings[closest_idx]
        else:
            closest_x, closest_y, closest_heading = None, None, None
        
        rospy.loginfo(f"t: {timestamp} | x: {robot_x:.2f} | y: {robot_y:.2f} | θ: {robot_heading:.2f} | \
                        goal: id{closest_idx} [{closest_x:.2f}, {closest_y:.2f}, {closest_heading:.2f}] - dist: {goal_distance:.2f} | \
                        θ_ref: {heading_ref:.2f} | θ_err: {heading_error:.2f} | v: {v:.2f} | ω: {omega:.2f}")
        self.csv_writer.writerow([
            timestamp,
            robot_x,
            robot_y,
            robot_heading,
            closest_x,
            closest_y,
            closest_heading,
            closest_idx,
            goal_distance,
            heading_ref,
            heading_error,
            v,
            omega,
            self.start_x,
            self.start_y,
            self.yaw_offset,
            remaining_path
        ])
        self.recording_file.flush()  # Ensure data is written to disk

    def goal_callback(self, msg):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"New goal received: {self.current_goal}")
        self.goalrec = True
        self.closest_idx = 0   

        x_rot, y_rot, yaw_rot = self.rotate_pose_to_local_frame(msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.z)

        new_pose = PoseStamped()
        new_pose.pose.position.x = x_rot
        new_pose.pose.position.y = y_rot
        new_pose.pose.orientation.z = yaw_rot
        new_pose.header.stamp = rospy.Time.now()
        new_pose.header.frame_id = "robot"

        self.local_goal_pose_pub.publish(new_pose)

        if self.current_path is None and not self.gen:
            self.plan_path()
            self.gen = True   

    def robot_pose_callback(self, msg):
        # Rotate to local frame
        x_rot, y_rot, yaw_rot = self.rotate_pose_to_local_frame(msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.z)

        new_pose = PoseStamped()
        new_pose.pose.position.x = x_rot
        new_pose.pose.position.y = y_rot
        new_pose.pose.orientation.z = yaw_rot
        new_pose.header.stamp = rospy.Time.now()
        new_pose.header.frame_id = "robot"
        
        self.unrotated_current_pose = msg
        self.current_pose = new_pose
        self.poserec = True

        self.local_robot_pose_pub.publish(new_pose)
    
    def rotate_pose_to_local_frame(self, x, y, yaw):
        # Rotate coordinates by -yaw_offset
        if self.yaw_offset == None:
            rospy.loginfo("Yaw Offset is None")
            return x, y, yaw
        x_rot = x * np.cos(-self.yaw_offset) - y * np.sin(-self.yaw_offset)
        y_rot = x * np.sin(-self.yaw_offset) + y * np.cos(-self.yaw_offset)
        
        # Adjust yaw (heading)
        yaw_rot = yaw - self.yaw_offset
        
        return x_rot, y_rot, yaw_rot

    def pose_offset_callback(self, msg):
        if msg.pose.position.x is None or msg.pose.position.y is None or msg.pose.orientation.z is None:
            rospy.logwarn("Received invalid pose_offset message with None values!")
            
        self.start_x = msg.pose.position.x
        self.start_y = msg.pose.position.y
        self.yaw_offset = msg.pose.orientation.z
        # rospy.loginfo(f"Received message for pose offset {self.start_x}")
        

    def plan_path(self):
        if self.current_pose is None or self.current_goal is None:
            rospy.logwarn("Cannot plan path - missing pose or goal")
            return
            
        # current_position = (self.current_pose.pose.position.x, 
        #                   self.current_pose.pose.position.y)

        current_position = (self.unrotated_current_pose.pose.position.x, 
                          self.unrotated_current_pose.pose.position.y)
        
        rospy.loginfo("Planning new path...")
        initial_path, _, _, _ = execute_planning(current_position, self.current_goal)

        path_points, headings = self.rotate_path_to_local_frame(initial_path)

        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for point in path_points:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)


        gl_path_msg = Path()
        gl_path_msg.header.stamp = rospy.Time.now()
        gl_path_msg.header.frame_id = "map"
        
        for point in initial_path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            gl_path_msg.poses.append(pose)
        self.global_path_pub.publish(gl_path_msg)
        
        self.current_path = path_points
        self.current_headings = headings

    
        rospy.loginfo("Path published with {} waypoints".format(len(path_msg.poses)))
  

    def rotate_path_to_local_frame(self, path):
        path_points = [(x,y) for x,y,_ in path]
        headings = [h for _,_,h in path]
        rotated_points = []
        rotated_headings = []
        
        for (x, y), heading in zip(path_points, headings):
            # Rotate coordinates by -yaw_offset
            x_rot = x * np.cos(-self.yaw_offset) - y * np.sin(-self.yaw_offset)
            y_rot = x * np.sin(-self.yaw_offset) + y * np.cos(-self.yaw_offset)
            rotated_points.append([x_rot, y_rot])

            # Also rotate heading
            heading_rot = heading - self.yaw_offset
            rotated_headings.append(heading_rot)
        
        return rotated_points, rotated_headings

    def generate_offset_path(self):
        if self.current_pose is None:
            return
            
        x0 = self.current_pose.pose.position.x
        y0 = self.current_pose.pose.position.y
        theta0 = self.current_pose.pose.orientation.z
        
        offset_angle = theta0 + np.radians(0)
        
        path_points = []
        headings = []
        
        for i in range(1, 5):
            distance = 0.5 * i
            x = x0 + distance * np.cos(offset_angle)
            y = y0 + distance * np.sin(offset_angle)
            
            path_points.append([x, y])
            headings.append(offset_angle)
        
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

    def run_control(self):
        try:
            while not rospy.is_shutdown():
                if self.start_x == -999 or self.start_x == None:
                    rospy.loginfo_throttle(5, f"[CONTROLLER] WAITING FOR POSE OFFSET")
                    continue
                else:
                    rospy.loginfo_once(f"[CONTROLLER] POSE OFFSET RETRIEVED")

                if self.poserec and self.goalrec and not self.reached:
                    rospy.loginfo_throttle(5, f"[CONTROLLER] CONTROLLER PROCESSING")
                    if self.gen is True and self.current_path is not None and self.current_pose is not None:
                        x_robot = self.current_pose.pose.position.x
                        y_robot = self.current_pose.pose.position.y
                        current_pos = (x_robot, y_robot)
                        theta_robot = self.current_pose.pose.orientation.z
                        
                        
                        self.closest_idx = self.find_closest_point(self.current_path, current_pos)
                        closest_point = self.current_path[self.closest_idx]
                        
                        remaining_path = self.prune_passed_points(self.current_path, self.closest_idx)
                        
                        
                        x_goal, y_goal = self.current_path[-1][0], self.current_path[-1][1]
                        goal_distance = np.sqrt((x_goal - x_robot)**2 + (y_goal - y_robot)**2)
                                                
                        if goal_distance < self.goal_distance_threshold:
                            cmd_vel = Twist()  
                            cmd_vel.linear.x = 0
                            cmd_vel.angular.z = 0
                            self.cmd_vel_pub.publish(cmd_vel)
                            rospy.loginfo("Goal reached!")
                            self.reached = True
                            continue
                        
                        lookahead_point, lookahead_idx = self.find_lookahead_point(
                            remaining_path, current_pos, 0)
                        
                        actual_lookahead_idx = self.closest_idx + lookahead_idx
                        heading_ref = self.current_headings[actual_lookahead_idx]
                        
                        if goal_distance < self.slow_down_distance:
                            v = self.v_min + (self.v_max - self.v_min) * (goal_distance / self.slow_down_distance)
                        else:
                            v = self.v_max
                        
                        heading_error = heading_ref - theta_robot
                        heading_error = (heading_error + np.pi) % (2 * np.pi) - np.pi
                        
                        omega = self.k_angular * heading_error
                        max_omega = 1.2
                        omega = np.clip(omega, -max_omega, max_omega)
                        
                        cmd_vel = Twist()
                        cmd_vel.linear.x = v
                        cmd_vel.angular.z = omega
                        self.cmd_vel_pub.publish(cmd_vel)
                        
                        # Data to Record:
                        # 1/ Current pose (x, y, theta)
                        # 2/ Closest point (idx, x, y, theta)
                        # 3/ Goal distance
                        # 4/ Heading Ref & Heading Error
                        # 5/ Commands (v, omega)
                        self.record_data(self.current_pose, closest_point, self.closest_idx, goal_distance, heading_ref, heading_error, v, omega, remaining_path)

                        
                    else:
                        cmd_vel = Twist()  
                        cmd_vel.linear.x = 0
                        cmd_vel.angular.z = 0
                        self.cmd_vel_pub.publish(cmd_vel)

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