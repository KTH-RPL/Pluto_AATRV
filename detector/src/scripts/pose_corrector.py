#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class PoseCorrector:
    """
    A ROS node that subscribes to a PoseStamped topic, corrects its orientation
    to be purely 2D (zero roll and pitch), and republishes it.
    """
    def __init__(self):
        # Initialize the node
        rospy.init_node('pose_corrector_node', anonymous=True)

        # Create a publisher for the corrected pose
        self.corrected_pose_pub = rospy.Publisher('/corrected_robot_pose', PoseStamped, queue_size=10)

        # Create a subscriber to the original robot pose topic
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.pose_callback)

        rospy.loginfo("Pose Corrector node started. Subscribing to /robot_pose...")

    def pose_callback(self, msg):
        """
        Callback function to process the incoming PoseStamped message.
        """
        # Create a new PoseStamped message for the output
        corrected_pose = PoseStamped()

        # 1. Copy the header and position directly from the incoming message.
        # It's good practice to update the timestamp to the time of correction.
        corrected_header = Header()
        corrected_header.frame_id = "base_footprint"
        corrected_header.seq = msg.header.seq
        corrected_header.stamp = msg.header.stamp  # Use the original timestamp

        corrected_pose.header = corrected_header
        corrected_pose.pose.position = msg.pose.position

        corrected_q = quaternion_from_euler(0.0, 0.0, msg.pose.orientation.z)

        # 5. Assign the new, corrected quaternion to our message
        corrected_pose.pose.orientation.x = corrected_q[0]
        corrected_pose.pose.orientation.y = corrected_q[1]
        corrected_pose.pose.orientation.z = corrected_q[2]
        corrected_pose.pose.orientation.w = corrected_q[3]

        # 6. Publish the corrected pose
        self.corrected_pose_pub.publish(corrected_pose)

    def run(self):
        """
        Keeps the node alive.
        """
        rospy.spin()

if __name__ == '__main__':
    try:
        corrector = PoseCorrector()
        corrector.run()
    except rospy.ROSInterruptException:
        pass