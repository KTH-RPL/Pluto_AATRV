#!/usr/bin/env python3

'''
To publish:
1/ Static Robot Offset
2/ Static Robot Position 
'''

import rospy
import utm
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from vectornav.msg import Ins


class RobotPosePublisher:

    def __init__(self, offset=[-333520.64199354, -6582420.00142414, 0.0]):
        self.offset = offset
        # self.gnss = rospy.Subscriber('/reach/fix', NavSatFix, self.gnss_callback)
        # self.orientation = rospy.Subscriber('/vectornav/INS', Ins, self.orientation_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)
        self.robot_x = 140.0
        self.robot_y = 85.0
        self.robot_yaw = -3.07815109641803

        self.robot_pos_offset_pub = rospy.Publisher('/pose_offset', PoseStamped, queue_size=10)

        rospy.loginfo("[INIT] Static Publisher Initialized")


    def publish_robot_pose(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.robot_x
        pose.pose.position.y = self.robot_y
        pose.pose.orientation.z = self.robot_yaw
        self.robot_pos_pub.publish(pose)

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = 142.0
        pose.pose.position.y = 87.0
        pose.pose.orientation.z = -3.09746141116609

        self.robot_pos_offset_pub.publish(pose)

        rospy.loginfo_throttle(5, "[PUBLISH] Publishing pose: x={:.2f}, y={:.2f}, yaw={:.2f}".format(
            self.robot_x, self.robot_y, self.robot_yaw))


def convert_gnss_to_utm(lat, lon):
    utm_coords = utm.from_latlon(lat, lon)
    x = utm_coords[0]
    y = utm_coords[1]
    return x, y


class PathPublisher:

    def __init__(self):
        self.path = rospy.Publisher('/path', Path, queue_size=10)
        rospy.loginfo("[INIT] PathPublisher initialized.")

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.z = point[2]
            path_msg.poses.append(pose)
        self.path.publish(path_msg)
        rospy.loginfo("[PUBLISH] Path published with {} points.".format(len(path)))


if __name__ == '__main__':
    rospy.init_node('robot_pose_publisher')
    rospy.loginfo("[START] robot_pose_publisher node started.")
    offset = [-333520.64199354, -6582420.00142414, 0.0]
    robot_pose_publisher = RobotPosePublisher(offset)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        robot_pose_publisher.publish_robot_pose()
        rate.sleep()
