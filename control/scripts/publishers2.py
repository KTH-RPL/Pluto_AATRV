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
        This node operates in two phases:
        1. LOCAL PHASE: On startup, it fuses odometry and IMU to publish a pose
           relative to its starting point in an 'odom' frame.
        2. GLOBAL PHASE: When a global path is received, it re-calculates its
           offsets to align with the path's starting pose and begins publishing
           the pose in a 'map' frame.
        """
        rospy.init_node('robot_pose_publisher')

        # --- Subscribers and Publisher ---
        self.odom_sub = rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)
        self.orientation_sub = rospy.Subscriber('/vectornav/INS', Ins, self.orientation_callback)
        # --- MODIFIED --- Subscribing to /planned_path for the global plan
        self.path_sub = rospy.Subscriber('/planned_path', Path, self.path_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)

        # --- State Variables ---
        self.raw_odom_x = None
        self.raw_odom_y = None
        self.raw_ins_yaw_rad = None

        # Offsets are first used for local frame, then recalculated for global alignment.
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.yaw_offset = 0.0

        # State flags
        self.local_pose_initialized = False
        self.is_globally_aligned = False

        rospy.loginfo("Robot Pose Publisher initialized. Waiting for sensor data...")

    def path_callback(self, msg):
        """
        When a global path is received, this callback triggers the re-alignment
        from the local odom frame to the global map frame. It runs only once.
        """
        # --- Run this re-alignment logic only once ---
        if self.is_globally_aligned or not msg.poses:
            return

        # --- Ensure we have sensor data before trying to align ---
        if self.raw_odom_x is None or self.raw_ins_yaw_rad is None:
            rospy.logwarn_throttle(5, "Received global path, but waiting for sensor data to align.")
            return

        # Get the target starting pose from the global path
        path_start_pose = msg.poses[0]
        target_start_x = path_start_pose.pose.position.x
        target_start_y = path_start_pose.pose.position.y

        # Convert the target orientation from quaternion to yaw (in radians)
        q = path_start_pose.pose.orientation
        orientation_list = [q.x, q.y, q.z, q.w]
        (_, _, target_start_yaw) = euler_from_quaternion(orientation_list)

        # Calculate the NEW offsets required to align the raw sensor readings with the global frame
        self.x_offset = self.raw_odom_x - target_start_x
        self.y_offset = self.raw_odom_y - target_start_y
        self.yaw_offset = self.raw_ins_yaw_rad - target_start_yaw

        self.is_globally_aligned = True
        rospy.loginfo("Global path received! Re-aligning to map frame. New offsets calculated.")

    def odom_callback(self, msg):
        """Stores the latest raw X and Y position from the odometry topic."""
        self.raw_odom_x = msg.pose.pose.position.x
        self.raw_odom_y = msg.pose.pose.position.y

    def orientation_callback(self, msg):
        """
        Stores the latest raw yaw from the INS/IMU topic, converted to radians.
        """
        yaw_rad = np.deg2rad(msg.yaw)
        if yaw_rad > np.pi:
            yaw_rad -= 2 * np.pi
        self.raw_ins_yaw_rad = yaw_rad

    def process_and_publish_pose(self):
        """
        Core logic function. Establishes a local frame, then publishes poses.
        The meaning of the pose (local vs. global) changes when the path is received.
        """
        # Wait for the first messages from both odom and INS
        if self.raw_odom_x is None or self.raw_ins_yaw_rad is None:
            rospy.loginfo_throttle(5, "Waiting for initial odometry and INS messages...")
            return

        # --- LOCAL INITIALIZATION (runs only once at the very start) ---
        if not self.local_pose_initialized:
            # Set initial offsets to establish a local frame starting at (0,0,0)
            self.x_offset = self.raw_odom_x
            self.y_offset = self.raw_odom_y
            self.yaw_offset = self.raw_ins_yaw_rad
            self.local_pose_initialized = True
            rospy.loginfo("Initial local pose captured. Publishing relative to start in 'odom' frame.")

        # Calculate the current pose by applying the current set of offsets
        # (These offsets will be local at first, then global after path_callback runs)
        current_x = self.raw_odom_x - self.x_offset
        current_y = self.raw_odom_y - self.y_offset
        current_yaw = self.raw_ins_yaw_rad - self.yaw_offset

        # Normalize the final yaw angle
        if current_yaw > np.pi:
            current_yaw -= 2 * np.pi
        elif current_yaw < -np.pi:
            current_yaw += 2 * np.pi

        # Create and publish the PoseStamped message
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        # --- Frame ID is now dynamic ---
        pose.header.frame_id = "map" if self.is_globally_aligned else "odom"
        pose.pose.position.x = current_x
        pose.pose.position.y = current_y

        # Convert final yaw to quaternion for the message
        pose.pose.orientation.z = np.sin(current_yaw / 2.0)
        pose.pose.orientation.w = np.cos(current_yaw / 2.0)

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

