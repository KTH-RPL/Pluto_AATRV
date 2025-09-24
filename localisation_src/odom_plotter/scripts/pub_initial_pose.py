#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler
import numpy as np
import math

def publish_initial_pose():
    rospy.init_node('initialpose_publisher')

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)

    # Rotation matrix
    #R = np.array([[0.9510566,  0.3090167,  0.0],
    #              [-0.3090167, 0.9510566,  0.0],
    #              [0.0,        0.0,        1.0]])

    # Homogeneous transformation matrix
    #T = np.eye(4)
    #T[:3, :3] = R
    #T[0, 3] = 126.409
    #T[1, 3] = 53.545
    #T[2, 3] = 42.0

    # Convert rotation matrix to quaternion
    q = quaternion_from_euler(0.0, 0.0, math.radians(177.7))

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = 'map'

    msg.pose.pose.position.x = 147.096863#126.409
    msg.pose.pose.position.y = 79.561790#53.545
    msg.pose.pose.position.z = 42.016739#42.0

    msg.pose.pose.orientation.x = q[0]
    msg.pose.pose.orientation.y = q[1]
    msg.pose.pose.orientation.z = q[2]
    msg.pose.pose.orientation.w = q[3]

    # Optional: set a default covariance
    msg.pose.covariance = [0.25, 0, 0, 0, 0, 0,
                           0, 0.25, 0, 0, 0, 0,
                           0, 0, 0.25, 0, 0, 0,
                           0, 0, 0, 0.0685389, 0, 0,
                           0, 0, 0, 0, 0.0685389, 0,
                           0, 0, 0, 0, 0, 0.0685389]

    rospy.sleep(1)  # wait for publisher to register
    pub.publish(msg)
    print("Initial pose published!")

if __name__ == "__main__":
    publish_initial_pose()
