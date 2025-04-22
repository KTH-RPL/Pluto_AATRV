#!/usr/bin/env python3

import rospy
import utm
import numpy as np
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from vectornav.msg import Ins
import utm

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
        self.offsetang = 200 # 53
        self
        rospy.loginfo("[INIT] RobotPosePublisher initialized with offset: [{:.2f}, {:.2f}, {:.2f}]".format(
            self.offset[0], self.offset[1], self.offset[2]))

    def gnss_callback(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        x, y = self.convert_gnss_to_utm(lat, lon)
        rospy.loginfo("[GNSS] GNSS fix received: lat={}, lon={}".format(lat, lon))
        rospy.loginfo("[GNSS] UTM coordinates before offset: x={:.2f}, y={:.2f}".format(x, y))

        self.robot_x = x + self.offset[0]
        self.robot_y = y + self.offset[1]

        rospy.loginfo("[GNSS] UTM coordinates after offset: x={:.2f}, y={:.2f}".format(self.robot_x, self.robot_y))

    def convert_gnss_to_utm(self,lat, lon):
        utm_coords = utm.from_latlon(lat, lon)
        x = utm_coords[0]
        y = utm_coords[1]
        return x, y
    
    def firstyaw_cb(self,msg):
        self.first_yaw = msg.pose.orientation.z


    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
    
    def orientation_callback(self, msg):
        yaw = msg.yaw

        if yaw < 0:
            yaw = 360 + yaw
        # self.robot_yaw = (self.first_yaw - yaw)  
        self.robot_yaw = (self.offsetang + yaw) 
        self.robot_yaw = self.robot_yaw * np.pi / 180
        
        
        # self.robot_yaw = np.pi/2 - self.robot_yaw  
        if self.robot_yaw > np.pi:
            self.robot_yaw -= 2 * np.pi
        elif self.robot_yaw < -np.pi:
            self.robot_yaw += 2 * np.pi
        self.robot_yaw =- self.robot_yaw
        print("First_yaw ,",self.first_yaw," MSG ",yaw,"  robot yaw ",self.robot_yaw)

            
    def publish_robot_pose(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.robot_x
        pose.pose.position.y = self.robot_y
        pose.pose.orientation.z = self.robot_yaw
        self.robot_pos_pub.publish(pose)

if __name__ == '__main__':
    rospy.init_node('robot_pose_publisher1')
    robot_pose_publisher = RobotPosePublisher()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        robot_pose_publisher.publish_robot_pose()
        rate.sleep()