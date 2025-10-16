#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from vectornav.msg import Ins
from tf.transformations import euler_from_quaternion

# TOGGLE FOR PUBLISHERS
publish_pose = False         # Publish Robot Pose (Based on Odom for x y and IMU for yaw)
publish_waypoint = True     # Publish Waypoint Goals for Global Planner
publish_ndt_pose_yaw = True         # Subscribe to NDT Pose and publish to Robot Pose (converting quarternion yaw into just z yaw)


# ------------------------------------------------------------------------
# Waypoint Publisher for Global Planner
class WaypointPublisher:
    def __init__(self):
        # Create a publisher for the "/waypoints" topic with message type Path
        self.path_publisher = rospy.Publisher('/waypoints', Path, queue_size=10, latch=True)

        # The list of waypoint coordinates (x, y)
        self.waypoints_coords = [
            (120, 65),
            (119.5, 56),
            (140, 30),
            (190, 10),
            (170, -50),
            (145, -60),
            (121, -105),
            (100, -136),
            (25, -110)
        ]

        # Create the Path message
        self.path_msg = self.create_path_message()

    def create_path_message(self):
        """
        Creates a nav_msgs/Path message from the list of waypoint coordinates.
        """
        path = Path()
        # Set the frame ID for the path. This is important for visualization in RViz.
        path.header.frame_id = "map"
        path.header.stamp = rospy.Time.now()

        for coord in self.waypoints_coords:
            pose = PoseStamped()
            # Set the timestamp for each pose
            pose.header.stamp = rospy.Time.now()
            # Set the frame ID for each pose
            pose.header.frame_id = "map"

            # Set the position (x, y, z). We assume z=0 for 2D navigation.
            pose.pose.position.x = coord[0]
            pose.pose.position.y = coord[1]
            pose.pose.position.z = 0.0

            # Set the orientation. A quaternion of (0,0,0,1) means no rotation.
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 0.0

            path.poses.append(pose)

        return path

    def publish_waypoint(self):
        if not self.path_msg:
            self.path_msg = self.create_path_message()
        self.path_publisher.publish(self.path_msg)

# ------------------------------------------------------------------------        
# Publisher for Robot Pose (based on Odom for x y and IMU for yaw)
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
        self.fp = True

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
        self.robot_yaw = yaw 
        self.robot_yaw = self.robot_yaw * np.pi / 180
        
        
        # self.robot_yaw = np.pi/2 - self.robot_yaw  
        if self.robot_yaw > np.pi:
            self.robot_yaw -= 2 * np.pi
        elif self.robot_yaw < -np.pi:
            self.robot_yaw += 2 * np.pi

        if self.fp == True:
            self.first_yaw = self.robot_yaw
            self.robot_yaw-=self.robot_yaw
            self.fp =False
        else:
            self.robot_yaw -= self.first_yaw
            self.robot_yaw = -self.robot_yaw
        
        if self.robot_yaw > np.pi:
            self.robot_yaw -= 2 * np.pi
        elif self.robot_yaw < -np.pi:
            self.robot_yaw += 2 * np.pi
        # print("First_yaw ,",self.first_yaw," MSG ",yaw,"  robot yaw ",self.robot_yaw)

            
    def publish_robot_pose(self):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = self.robot_x
        pose.pose.position.y = self.robot_y
        pose.pose.orientation.z = self.robot_yaw
        self.robot_pos_pub.publish(pose)

# ------------------------------------------------------------------------        
# Publisher for NDT Robot Pose (Only changing the yaw)
class RobotNDTPosePublisher:
    def __init__(self):
        self.ndt_sub = rospy.Subscriber('/ndt_pose', PoseStamped, self.ndt_callback)
        self.robot_pos_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)

    def ndt_callback(self,msg):
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = msg.pose.position.x
        pose.pose.position.y = msg.pose.position.y

        orientation_q = msg.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        pose.pose.orientation.x = 0
        pose.pose.orientation.y = 0
        pose.pose.orientation.w = 0
        pose.pose.orientation.z = yaw
        self.robot_pos_pub.publish(pose)


# ------------------------------------------------------------------------
# MAIN CALLER
if __name__ == '__main__':
    # Publisher for Robot Pose (based on Odom for x y and IMU for yaw)
    # if publish_pose:
    #     rospy.init_node('robot_pose_publisher1')
    #     robot_pose_publisher = RobotPosePublisher()

    # if publish_waypoint:
    #     rospy.init_node('waypoint_publisher_node', anonymous=True)
    #     waypoint_publisher = WaypointPublisher()
    #     waypoint_publisher.publish_waypoint() # Publish it once?

    if publish_ndt_pose_yaw:
        rospy.init_node('robot_ndt_pose_publisher1')
        robot_ndt_pose_publisher = RobotNDTPosePublisher()
        waypoint_publisher = WaypointPublisher()
        waypoint_publisher.publish_waypoint() # Publish it once?


    # Periodical Publish
    #rate = rospy.Rate(10)
    rospy.spin()
    #while not rospy.is_shutdown():
        # Publisher for Robot Pose (based on Odom for x y and IMU for yaw)
        # if publish_pose:
        #    robot_ndt_pose_publisher.publish_robot_pose()
    

        #rate.sleep()