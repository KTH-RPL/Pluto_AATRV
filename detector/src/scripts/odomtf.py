#!/usr/bin/env python3

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
import geometry_msgs.msg

def odom_callback(msg):
    # Create a TransformBroadcaster
    br = tf2_ros.TransformBroadcaster()
    
    # Create a TransformStamped message
    t = geometry_msgs.msg.TransformStamped()

    t.header.stamp = rospy.Time.now() # Use current time
    t.header.frame_id = msg.header.frame_id # Should be "odom"
    t.child_frame_id = msg.child_frame_id  # Should be "base_link"

    # Copy the pose from the Odometry message
    t.transform.translation = msg.pose.pose.position
    t.transform.rotation = msg.pose.pose.orientation

    # Send the transform
    br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('odom_tf_broadcaster')
    
    # Subscribe to the /odom topic
    rospy.Subscriber('/atrv/odom', Odometry, odom_callback)
    
    rospy.loginfo("Odometry to TF broadcaster started.")
    rospy.spin()