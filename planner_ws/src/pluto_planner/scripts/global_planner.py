#!/usr/bin/env python3
import rospy
import actionlib
from sensor_msgs.msg import NavSatFix
from pluto_planner.msg import RRTPlannerAction, RRTPlannerGoal
import utm
import math

class GlobalPlanner:
    def __init__(self):
        rospy.init_node("global_planner_node")

        # Define the goals (in meters) relative to geo_offset
        self.goals = [(2, 2), (4, 4), (6, 6), (9, 9)]
        self.current_goal_index = 0
        self.goal_tolerance = 0.5  # Distance to consider goal reached

        self.robot_x = None
        self.robot_y = None
        self.goal_active = False

        # Action client to communicate with the local planner
        self.client = actionlib.SimpleActionClient("get_next_goal", RRTPlannerAction)
        rospy.loginfo("Waiting for local planner action server...")
        self.client.wait_for_server()

        # GNSS Subscriber (simulated position)
        rospy.Subscriber("reach/fix", NavSatFix, self.gnss_callback)

        # Publisher for start point, goal points, and markers
        self.marker_pub = rospy.Publisher("/start_point", Marker, queue_size=10)
        self.goal_pub = rospy.Publisher("/active_goal", Marker, queue_size=10)

        # Main loop: Check if the robot is near the goal
        rospy.Timer(rospy.Duration(1), self.check_goal_status)
        rospy.loginfo("Global Planner Node started!")

    def gnss_callback(self, msg):
        geo_offset = [-333520.64199354, -6582420.00142414, 0.0]
        utm_x, utm_y, _, _ = utm.from_latlon(msg.latitude, msg.longitude)
        self.robot_x = utm_x + geo_offset[0]
        self.robot_y = utm_y + geo_offset[1]
        
        # Publish the start point marker when the first GNSS message is received
        if self.robot_x is not None and self.robot_y is not None:
            self.publish_start_marker(self.robot_x, self.robot_y)

    def check_goal_status(self, event):
        if not self.goal_active and self.current_goal_index < len(self.goals):
            self.send_goal()
        elif self.goal_active:
            goal_x, goal_y = self.goals[self.current_goal_index]
            if self.robot_x and self.robot_y:
                distance = math.sqrt((self.robot_x - goal_x)**2 + (self.robot_y - goal_y)**2)
                if distance < self.goal_tolerance:
                    rospy.loginfo(f"Goal {self.current_goal_index + 1} reached.")
                    self.current_goal_index += 1
                    self.goal_active = False  # Allow sending the next goal

    def send_goal(self):
        if self.current_goal_index < len(self.goals):
            goal_x, goal_y = self.goals[self.current_goal_index]
            goal = RRTPlannerGoal()
            goal.goal_pos.x = goal_x
            goal.goal_pos.y = goal_y

            rospy.loginfo(f"Sending goal {self.current_goal_index + 1}: ({goal_x}, {goal_y})")
            self.client.send_goal(goal)
            self.goal_active = True
            
            # Publish the goal point to RViz
            self.publish_goal_marker(goal_x, goal_y)
        else:
            rospy.loginfo("All goals reached! Stopping robot.")
            rospy.signal_shutdown("Mission complete")
            
    def publish_start_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "start_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.g = 1.0  # Green for start point
        marker_pub.publish(marker)
        
    def publish_goal_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "goal_point"
        marker.id = self.current_goal_index  # Different ID for each goal
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 1.0  # Red for goal point
        self.goal_pub.publish(marker)

if __name__ == "__main__":
    try:
        GlobalPlanner()
    except rospy.ROSInterruptException:
        rospy.loginfo("Global Planner Node terminated.")

