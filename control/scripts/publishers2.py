#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from vectornav.msg import Ins
from tf.transformations import euler_from_quaternion

class RobotPosePublisher:
    def __init__(self):
        """
        Initializes the node, subscribers, and publisher.
        This node correctly performs dead reckoning and aligns to a global path.
        """
        rospy.init_node('robot_pose_publisher')

        # --- Subscribers and Publisher ---
        self.odom_sub = rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)
        self.orientation_sub = rospy.Subscriber('/vectornav/INS', Ins, self.orientation_callback)
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)

        # --- State Variables ---
        # Current integrated pose in the local "odom" frame
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_yaw = 0.0

        # Store the last sensor readings to calculate deltas (changes)
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        self.last_ins_yaw = 0.0
        
        # This static transform takes points from our local 'odom' frame to the global 'map' frame
        self.map_transform = None

        # State flags
        self.odom_initialized = False
        self.imu_initialized = False
        self.is_globally_aligned = False

        rospy.loginfo("Robot Pose Publisher initialized. Waiting for sensor data...")

    def path_callback(self, msg):
        """
        When the first global path is received, this calculates the static transform
        (translation and rotation) required to align the local 'odom' frame with the
        global 'map' frame.
        """
        if self.is_globally_aligned or not msg.poses or not (self.odom_initialized and self.imu_initialized):
            return

        # Get the target starting pose from the global path in the 'map' frame
        path_start_pose = msg.poses[0]
        target_map_x = path_start_pose.pose.position.x
        target_map_y = path_start_pose.pose.position.y
        q = path_start_pose.pose.orientation
        (_, _, target_map_yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Calculate the rotational offset
        rotation_offset = target_map_yaw - self.pose_yaw
        
        # Calculate the translational offset
        cos_rot = np.cos(rotation_offset)
        sin_rot = np.sin(rotation_offset)
        translation_x = target_map_x - (self.pose_x * cos_rot - self.pose_y * sin_rot)
        translation_y = target_map_y - (self.pose_x * sin_rot + self.pose_y * cos_rot)

        # Store the complete transform
        self.map_transform = {
            'x': translation_x,
            'y': translation_y,
            'yaw': rotation_offset
        }

        self.is_globally_aligned = True
        rospy.loginfo("Global path received! Aligned to map frame.")

    def odom_callback(self, msg):
        current_odom_x = msg.pose.pose.position.x
        current_odom_y = msg.pose.pose.position.y

        if not self.odom_initialized:
            self.last_odom_x = current_odom_x
            self.last_odom_y = current_odom_y
            self.odom_initialized = True
            return

        # Calculate displacement in the robot's own moving frame
        delta_x_robot = current_odom_x - self.last_odom_x
        delta_y_robot = current_odom_y - self.last_odom_y

        # Rotate displacement into the world frame (using the robot's current yaw)
        delta_x_world = delta_x_robot * np.cos(self.pose_yaw) - delta_y_robot * np.sin(self.pose_yaw)
        delta_y_world = delta_x_robot * np.sin(self.pose_yaw) + delta_y_robot * np.cos(self.pose_yaw)

        # Integrate to update the pose
        self.pose_x += delta_x_world
        self.pose_y += delta_y_world

        # Update last known values for the next iteration
        self.last_odom_x = current_odom_x
        self.last_odom_y = current_odom_y
        
        self.publish_robot_pose()

    def orientation_callback(self, msg):
        # Convert yaw from degrees [0, 360) to radians [-pi, pi]
        current_yaw_rad = np.deg2rad(msg.yaw)
        current_yaw_rad = np.arctan2(np.sin(current_yaw_rad), np.cos(current_yaw_rad))

        if not self.imu_initialized:
            self.last_ins_yaw = current_yaw_rad
            self.imu_initialized = True
            return
        
        # Calculate change in yaw and update the pose
        # This simple subtraction works because we update frequently
        delta_yaw = current_yaw_rad - self.last_ins_yaw
        self.pose_yaw += delta_yaw
        
        # Normalize the final yaw
        self.pose_yaw = -np.arctan2(np.sin(self.pose_yaw), np.cos(self.pose_yaw))
        
        self.last_ins_yaw = current_yaw_rad
        
    def publish_robot_pose(self):
        # Only publish if we have initialized data
        if not (self.odom_initialized and self.imu_initialized):
            return

        final_x, final_y, final_yaw = self.pose_x, self.pose_y, self.pose_yaw
        frame_id = "odom"

        # If aligned, transform the local pose to the global map frame
        if self.is_globally_aligned:
            cos_rot = np.cos(self.map_transform['yaw'])
            sin_rot = np.sin(self.map_transform['yaw'])
            
            final_x = self.pose_x * cos_rot - self.pose_y * sin_rot + self.map_transform['x']
            final_y = self.pose_x * sin_rot + self.pose_y * cos_rot + self.map_transform['y']
            final_yaw = self.pose_yaw + self.map_transform['yaw']
            frame_id = "map"
        
        # Normalize the final yaw one last time
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))
        
        # --- Create and Publish Message ---
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position.x = final_x
        pose.pose.position.y = final_y
        
        # Convert final yaw to quaternion for the message
        pose.pose.orientation.z = final_yaw
        
        self.robot_pos_pub.publish(pose)

if __name__ == '__main__':
    try:
        robot_pose_publisher = RobotPosePublisher()
        rospy.spin()  # Event-based, no need for a while loop
    except rospy.ROSInterruptException:
        rospy.loginfo("Robot Pose Publisher node shut down.")

