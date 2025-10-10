#!/usr/bin/env python3

import rospy
import actionlib
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Path

from robot_controller.msg import PlanGlobalPathAction, PlanGlobalPathGoal

from utils import gmap_utility
import matplotlib.pyplot as plt


path_pub_ = rospy.Publisher("global_path", Path, queue_size=1, latch=True)

def feedback_callback(feedback):
    rospy.loginfo("Received feedback: Path segment with %d poses has been planned.", len(feedback.current_segment.poses))
    try: 
        plt.close('all')
    except:
        pass
    # gmap_utility.polygon_map.visualize(feedback.current_segment.poses)

def done_callback(status, result):
    if status == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("Action finished successfully!")
        rospy.loginfo("Final global path contains %d poses.", len(result.global_plan.poses))

         # Publish the received path to the 'global_path' topic
        if result.global_plan.poses:
            # The 'result.global_plan' is already a nav_msgs/Path message, so we can publish it directly.
            path_pub_.publish(result.global_plan)
            rospy.loginfo("Published the final path to 'global_path'.")
        else:
            rospy.logwarn("Received an empty final path. Not publishing.")
            
        try: 
            plt.close('all')
        except:
            pass
        # gmap_utility.polygon_map.visualize(result.global_plan.poses)
    elif status == actionlib.GoalStatus.PREEMPTED:
        rospy.logwarn("Action was preempted by a new goal.")
    else:
        rospy.logerr("Action failed with status code: %s", actionlib.GoalStatus.to_string(status))

def send_test_goal():
    # 1. Initialize the action client
    client = actionlib.SimpleActionClient('plan_global_path', PlanGlobalPathAction)

    rospy.loginfo("Waiting for the 'plan_global_path' action server...")
    client.wait_for_server()
    rospy.loginfo("Action server found")

    # 2. Define the waypoints to send
    # This is a list of simple (x, y) coordinates
    waypoints_coords = [
        (120,65), 
        (119.5, 56), 
        (140, 30), 
        (190,10), 
        (170,-50), 
        (145,-60), 
        (121,-105), 
        (100,-136), 
        (25,-110)
    ]

    # 3. Create the goal message
    goal = PlanGlobalPathGoal()
    goal.waypoints = PoseArray()
    goal.waypoints.header.frame_id = "map"
    goal.waypoints.header.stamp = rospy.Time.now()

    for x, y in waypoints_coords:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.orientation.w = 1.0
        goal.waypoints.poses.append(pose)

    rospy.loginfo("Sending goal with %d waypoints.", len(goal.waypoints.poses))

    # 4. Send the goal to the server
    client.send_goal(goal, done_cb=done_callback, feedback_cb=feedback_callback)


if __name__ == '__main__':
    try:
        rospy.init_node('global_path_action_client')
        send_test_goal()
        # Keep the node alive to receive callbacks
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Client node interrupted and shut down.")
