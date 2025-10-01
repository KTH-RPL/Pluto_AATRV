#!/usr/bin/env python
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np

class OdomToBaseLinkPublisher:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        self.sub = rospy.Subscriber("/atrv/odom", Odometry, self.odom_callback)

    def odom_callback(self, msg):
        # Extract position and orientation
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # Normalize quaternion to avoid TF errors
        quat = np.array([ori.x, ori.y, ori.z, ori.w])
        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            quat = np.array([0, 0, 0, 1])
        else:
            quat = quat / norm

        # Publish transform
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"

        t.transform.translation.x = pos.x
        t.transform.translation.y = pos.y
        t.transform.translation.z = pos.z
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('odom_to_baselink_tf_publisher')
    OdomToBaseLinkPublisher()
    rospy.loginfo("odom->base_link publisher running...")
    rospy.spin()
