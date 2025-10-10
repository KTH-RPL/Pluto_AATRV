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
        This node correctly fuses odometry and IMU data by rotating odometry
        displacements by the IMU's yaw before integrating them.
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

        # --- NEW --- Variables for tracking the integrated pose
        self.current_pose_x = 0.0
        self.current_pose_y = 0.0
        self.current_pose_yaw = 0.0
        self.last_odom_x = 0.0
        self.last_odom_y = 0.0
        
        # This transform stores the offset between the odom frame and the map frame
        self.map_to_odom_x_offset = 0.0
        self.map_to_odom_y_offset = 0.0
        self.map_to_odom_yaw_offset = 0.0

        # State flags
        self.local_pose_initialized = False
        self.is_globally_aligned = False

        rospy.loginfo("Robot Pose Publisher initialized. Waiting for sensor data...")

    def path_callback(self, msg):
        """
        When the first global path is received, this calculates the static transform
        (offset) between the robot's current local 'odom' frame and the global 'map' frame.
        """
        if self.is_globally_aligned or not msg.poses or not self.local_pose_initialized:
            return

        # Get the target starting pose from the global path
        path_start_pose = msg.poses[0]
        target_start_x = path_start_pose.pose.position.x
        target_start_y = path_start_pose.pose.position.y
        q = path_start_pose.pose.orientation
        (_, _, target_start_yaw) = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Calculate the offsets. This is the transform from the map frame to our odom frame.
        self.map_to_odom_x_offset = self.current_pose_x - target_start_x
        self.map_to_odom_y_offset = self.current_pose_y - target_start_y
        self.map_to_odom_yaw_offset = self.current_pose_yaw - target_start_yaw

        self.is_globally_aligned = True
        rospy.loginfo("Global path received! Aligned to map frame.")

    def odom_callback(self, msg):
        self.raw_odom_x = msg.pose.pose.position.x
        self.raw_odom_y = msg.pose.pose.position.y

    def orientation_callback(self, msg):
        yaw_rad = np.deg2rad(msg.yaw)
        self.raw_ins_yaw_rad = np.arctan2(np.sin(yaw_rad), np.cos(yaw_rad))

    def process_and_publish_pose(self):
        if self.raw_odom_x is None or self.raw_ins_yaw_rad is None:
            rospy.loginfo_throttle(5, "Waiting for initial odometry and INS messages...")
            return

        if not self.local_pose_initialized:
            # Initialize the local frame.
            self.last_odom_x = self.raw_odom_x
            self.last_odom_y = self.raw_odom_y
            self.yaw_offset = self.raw_ins_yaw_rad # The initial IMU reading is our reference "zero"
            self.local_pose_initialized = True
            rospy.loginfo("Initial local pose captured. Publishing in 'odom' frame.")
            return

        # --- Core Integration Logic ---
        # 1. Calculate displacement in the robot's frame (since last update)
        delta_x_robot = self.raw_odom_x - self.last_odom_x
        # Odom from this robot might only report forward motion in X
        delta_y_robot = self.raw_odom_y - self.last_odom_y 

        # 2. Get the current yaw relative to the starting orientation
        local_yaw = self.raw_ins_yaw_rad - self.yaw_offset

        # 3. Rotate the displacement into the local odom frame
        delta_x_odom = delta_x_robot * np.cos(local_yaw) - delta_y_robot * np.sin(local_yaw)
        delta_y_odom = delta_x_robot * np.sin(local_yaw) + delta_y_robot * np.cos(local_yaw)

        # 4. Integrate the displacement to update the pose in the odom frame
        self.current_pose_x += delta_x_odom
        self.current_pose_y += delta_y_odom
        self.current_pose_yaw = local_yaw

        # 5. Update the last known odometry for the next iteration
        self.last_odom_x = self.raw_odom_x
        self.last_odom_y = self.raw_odom_y

        # --- Frame Alignment and Publishing ---
        final_x, final_y, final_yaw = self.current_pose_x, self.current_pose_y, self.current_pose_yaw
        frame_id = "odom"

        if self.is_globally_aligned:
            # If aligned, transform the pose from the odom frame to the map frame
            final_x -= self.map_to_odom_x_offset
            final_y -= self.map_to_odom_y_offset
            final_yaw -= self.map_to_odom_yaw_offset
            frame_id = "map"

        # Normalize the final yaw angle
        final_yaw = np.arctan2(np.sin(final_yaw), np.cos(final_yaw))
        
        # Create and publish the PoseStamped message
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

