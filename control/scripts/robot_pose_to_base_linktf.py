#!/usr/bin/env python
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry, PoseStamped
import numpy as np

class OdomToBaseLinkPublisher:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber("/robot_pose", PoseStamped, self.odom_callback)

    def odom_callback(self, msg):
        # Extract position and orientation
        pos = msg.pose.position
        yaw = msg.pose.orientation.z

        pose = PoseStamped()
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.z = math.sin(yaw/2)
        pose.pose.orientation.w = math.cos(yaw/2) 



        # Normalize quaternion to avoid TF errors
        quat = np.array([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            quat = np.array([0, 0, 0, 1])
        else:
            quat = quat / norm

        # Publish transform
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"

        t.transform.translation.x = pos.x
        t.transform.translation.y = pos.y
        t.transform.translation.z = pos.z
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('rb_pose_to_baselink_tf_publisher')
    OdomToBaseLinkPublisher()
    rospy.loginfo("robot_pose->base_link publisher running...")
    rospy.spin()
