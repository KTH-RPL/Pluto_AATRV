#!/usr/bin/env python

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
        self.gnss = rospy.Subscriber('/reach/fix', NavSatFix, self.gnss_callback)
        self.orientation = rospy.Subscriber('/vectornav/INS', Ins, self.orientation_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.first_yaw = 0.0
        self.offsetang = 0 #53

    def gnss_callback(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        x, y = convert_gnss_to_utm(lat, lon)
        self.robot_x = x + self.offset[0]
        self.robot_y = y + self.offset[1]
    
    def orientation_callback(self, msg):
        yaw = msg.yaw
        if yaw < 0:
            yaw = 360 + yaw
        self.robot_yaw = (-self.offsetang + yaw)
        self.robot_yaw = 90 - self.robot_yaw  
        if yaw < -180:
            yaw = 360+yaw
        
        # print(self.robot_yaw)
        # print("Hello")
        
        self.robot_yaw = self.robot_yaw * np.pi / 180
        # print(self.robot_yaw)
        # self.robot_yaw = np.pi/2 - self.robot_yaw  
        if self.robot_yaw > np.pi:
            self.robot_yaw -= 2 * np.pi
        elif self.robot_yaw < -np.pi:
            self.robot_yaw += 2 * np.pi
        
        # print(self.robot_yaw)
    
    def publish_robot_pose(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.robot_x
        pose.pose.position.y = self.robot_y
        pose.pose.orientation.z = self.robot_yaw
        self.robot_pos_pub.publish(pose)
        # rospy.loginfo(pose)

def convert_gnss_to_utm(lat, lon):
    utm_coords = utm.from_latlon(lat, lon)
    x = utm_coords[0]
    y = utm_coords[1]
    return x, y


class PathPublisher:

    def __init__(self):
        self.path = rospy.Publisher('/path', Path, queue_size=10)

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


if __name__ == '__main__':
    rospy.init_node('robot_pose_publisher')
    offset = [-333520.64199354, -6582420.00142414, 0.0]
    robot_pose_publisher = RobotPosePublisher(offset)
    # path_publisher = PathPublisher()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        robot_pose_publisher.publish_robot_pose()
        rate.sleep()