#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from vectornav.msg import Ins

class RobotPosePublisher:
    def __init__(self):
        self.odom_sub = rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)
        self.orientation_sub = rospy.Subscriber('/vectornav/INS', Ins, self.orientation_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)
        self.firstyaw_sub = rospy.Subscriber('/firstins',PoseStamped, self.firstyaw_cb)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.first_yaw = 0.0
        self.offset = 0.0

    def firstyaw_cb(self,msg):
        self.first_yaw = msg.pose.orientation.z


    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
    
    def orientation_callback(self, msg):
        yaw = msg.yaw
        if yaw < 0:
            yaw = 360 + yaw
        self.robot_yaw = (self.first_yaw - yaw)  
        print("First_yaw ,",self.first_yaw," MSG ",yaw,"  robot yaw ",self.robot_yaw)
        self.robot_yaw = self.robot_yaw * np.pi / 180
 
        # self.robot_yaw = np.pi/2 - self.robot_yaw  
        if self.robot_yaw > np.pi:
            self.robot_yaw -= 2 * np.pi
        elif self.robot_yaw < -np.pi:
            self.robot_yaw += 2 * np.pi
    
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