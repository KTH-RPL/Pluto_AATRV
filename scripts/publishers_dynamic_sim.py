#!/usr/bin/env python

from geometry_msgs.msg import Twist, Quaternion, PoseStamped
import tf
import math
import rospy

class RobotPosePublisher:

    def __init__(self, offset=[-333520.64199354, -6582420.00142414, 0.0]):
        self.offset = offset
        self.robot_x = 140.0
        self.robot_y = 85.0
        self.robot_yaw = -3.07815109641803

        self.last_time = rospy.Time.now()

        # Publishers
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)
        self.robot_pos_offset_pub = rospy.Publisher('/pose_offset', PoseStamped, queue_size=10)

        # Subscriber to velocity commands
        self.cmd_sub = rospy.Subscriber('/atrv/cmd_vel', Twist, self.cmd_vel_callback)
        self.current_cmd = Twist()  # Default 0 velocity

        rospy.loginfo("[INIT] Static Publisher Initialized")

    def cmd_vel_callback(self, msg):
        self.current_cmd = msg

    def update_pose(self):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # Integrate position
        v = self.current_cmd.linear.x
        omega = self.current_cmd.angular.z

        dx = v * math.cos(self.robot_yaw) * dt
        dy = v * math.sin(self.robot_yaw) * dt
        dtheta = omega * dt

        self.robot_x += dx
        self.robot_y += dy
        self.robot_yaw += dtheta

    def publish_robot_pose(self):
        self.update_pose()

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"

        pose.pose.position.x = self.robot_x
        pose.pose.position.y = self.robot_y

        quat = tf.transformations.quaternion_from_euler(0, 0, self.robot_yaw)
        pose.pose.orientation = Quaternion(*quat)

        self.robot_pos_pub.publish(pose)

        # Offset pose (optional - static)
        pose_offset = PoseStamped()
        pose_offset.header.stamp = rospy.Time.now()
        pose_offset.header.frame_id = "map"
        pose_offset.pose.position.x = 142.0
        pose_offset.pose.position.y = 87.0
        pose_offset.pose.orientation.z = -3.09746141116609

        self.robot_pos_offset_pub.publish(pose_offset)

        rospy.loginfo_throttle(5, "[PUBLISH] Sim pose x={:.2f}, y={:.2f}, yaw={:.2f}".format(
            self.robot_x, self.robot_y, self.robot_yaw))

if __name__ == '__main__':
    rospy.init_node('robot_pose_publisher')
    rospy.loginfo("[START] robot_pose_publisher node started.")
    offset = [-333520.64199354, -6582420.00142414, 0.0]
    robot_pose_publisher = RobotPosePublisher(offset)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        robot_pose_publisher.publish_robot_pose()
        rate.sleep()
