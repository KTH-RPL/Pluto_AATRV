#!/usr/bin/env python3
import rospy
import tf
from nav_msgs.msg import Odometry

class OdomTFPublisher:
    def __init__(self):
        rospy.init_node('odom_tf_publisher')

        # Subscribe to your odometry topic
        rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)

        # TF broadcaster
        self.br = tf.TransformBroadcaster()

        rospy.loginfo("Odom TF publisher started...")
        rospy.spin()

    def odom_callback(self, msg):
        # Extract position and orientation
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # Publish odom -> base_link transform
        self.br.sendTransform(
            (pos.x, pos.y, pos.z),
            (ori.x, ori.y, ori.z, ori.w),
            msg.header.stamp,
            msg.child_frame_id,   # usually "base_link"
            msg.header.frame_id   # usually "odom"
        )

if __name__ == "__main__":
    try:
        OdomTFPublisher()
    except rospy.ROSInterruptException:
        pass
