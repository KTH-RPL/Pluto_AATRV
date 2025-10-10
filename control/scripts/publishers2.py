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
        This node correctly fuses odometry and IMU data by rotating the total
        odometry displacement by the initial IMU yaw to create a clean local frame.
        It then aligns this local frame to the global map frame upon receiving a path.
        """
        rospy.init_node('robot_pose_publisher')

        # --- Subscribers and Publisher ---
        self.odom_sub = rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)
        self.orientation_sub = rospy.Subscriber('/vectornav/INS', Ins, self.orientation_callback)
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)

        # --- State Variables ---
        self.raw_odom_x = None
        self.raw_odom_y = None
        self.raw_ins_yaw_rad = None
        
        # Store the initial sensor readings to define the origin of the local frame
        self.initial_odom_x = 0.0
        self.initial_odom_y = 0.0
        self.initial_ins_yaw = 0.0

        # This static transform takes points from our local 'odom' frame to the global 'map' frame
        self.odom_to_map_trans_x = 0.0
        self.odom_to_map_trans_y = 0.0
        self.odom_to_map_rot_yaw = 0.0

        # State flags
        self.local_pose_initialized = False
        self.is_globally_aligned = False

        rospy.loginfo("Robot Pose Publisher initialized. Waiting for sensor data...")

    def path_callback(self, msg):
        """
        When the first global path is received, this calculates the static transform
        (translation and rotation) between the robot's current local 'odom' frame
        and the global 'map' frame.
        """
        if self.is_globally_aligned or not msg.poses or not self.local_pose_initialized:
            return

        # Get the target starting pose from the global path in the 'map' frame
        path_start_pose = msg.poses[0]
        target_map_x = path_start_pose.pose.position.x
        target_map_y = path_start_pose.pose.position.y
        q = path_start_pose.pose.orientation
        (_, _, target_map_yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Get the robot's current pose in the 'odom' frame
        current_odom_x, current_odom_y, current_odom_yaw = self.get_current_local_pose()

        # Calculate the rotational part of the transform
        self.odom_to_map_rot_yaw = target_map_yaw - current_odom_yaw
        
        # Calculate the translational part of the transform
        # We must rotate the current odom position by the new rotation to align it properly
        cos_theta = np.cos(self.odom_to_map_rot_yaw)
        sin_theta = np.sin(self.odom_to_map_rot_yaw)
        self.odom_to_map_trans_x = target_map_x - (current_odom_x * cos_theta - current_odom_y * sin_theta)
        self.odom_to_map_trans_y = target_map_y - (current_odom_x * sin_theta + current_odom_y * cos_theta)

        self.is_globally_aligned = True
        rospy.loginfo("Global path received! Aligned to map frame.")

    def odom_callback(self, msg):
        self.raw_odom_x = msg.pose.pose.position.x
        self.raw_odom_y = msg.pose.pose.position.y

    def orientation_callback(self, msg):
        yaw_rad = np.deg2rad(msg.yaw)
        # --- FIX --- Negate the yaw value here to correct for a flipped IMU convention.
        # This ensures all subsequent calculations use the standard ROS sign convention
        # (counter-clockwise positive).
        self.raw_ins_yaw_rad = -np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad))

    def get_current_local_pose(self):
        """Calculates and returns the robot's pose in the local odom frame."""
        # Calculate displacement in the raw odom frame
        delta_x_raw_odom = self.raw_odom_x - self.initial_odom_x
        delta_y_raw_odom = self.raw_odom_y - self.initial_odom_y

        # Rotate the displacement vector backwards by the initial yaw.
        # This aligns the movement to our local frame where the robot started at yaw=0.
        cos_initial = np.cos(-self.initial_ins_yaw)
        sin_initial = np.sin(-self.initial_ins_yaw)
        
        local_x = delta_x_raw_odom * cos_initial - delta_y_raw_odom * sin_initial
        local_y = delta_x_raw_odom * sin_initial + delta_y_raw_odom * cos_initial

        # The local yaw is just the change from the initial yaw
        local_yaw = self.raw_ins_yaw_rad - self.initial_ins_yaw
        
        return local_x, local_y, local_yaw

    def process_and_publish_pose(self):
        if self.raw_odom_x is None or self.raw_ins_yaw_rad is None:
            rospy.loginfo_throttle(5, "Waiting for initial odometry and INS messages...")
            return

        if not self.local_pose_initialized:
            # Initialize the local frame by capturing the first sensor readings.
            self.initial_odom_x = self.raw_odom_x
            self.initial_odom_y = self.raw_odom_y
            self.initial_ins_yaw = self.raw_ins_yaw_rad
            self.local_pose_initialized = True
            rospy.loginfo("Initial local pose captured. Publishing in 'odom' frame.")
            return

        # --- Pose Calculation ---
        # 1. Get the robot's current pose in its local odom frame
        local_x, local_y, local_yaw = self.get_current_local_pose()

        final_x, final_y, final_yaw = local_x, local_y, local_yaw
        frame_id = "odom"

        # 2. If aligned, transform the local pose to the global map frame
        if self.is_globally_aligned:
            cos_theta = np.cos(self.odom_to_map_rot_yaw)
            sin_theta = np.sin(self.odom_to_map_rot_yaw)
            
            final_x = local_x * cos_theta - local_y * sin_theta + self.odom_to_map_trans_x
            final_y = local_x * sin_theta + local_y * cos_theta + self.odom_to_map_trans_y
            final_yaw = local_yaw + self.odom_to_map_rot_yaw
            frame_id = "map"

        # Normalize the final yaw angle
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))
        
        # --- Publishing ---
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = frame_id
        pose.pose.position.x = final_x
        pose.pose.position.y = final_y
        
        pose.pose.orientation.z = np.sin(final_yaw / 2.0)
        pose.pose.orientation.w = np.cos(final_yaw / 2.0)
        
        self.robot_pos_pub.publish(pose)

if __name__ == '__main__':
    try:
        robot_pose_publisher = RobotPosePublisher()
        rate = rospy.Rate(50)  # 50 Hz
        while not rospy.is_shutdown():
            robot_pose_publisher.process_and_publish_pose()
            rate.sleep()
    except rospy.ROSInterruptException:
        rospy.loginfo("Robot Pose Publisher node shut down.")

