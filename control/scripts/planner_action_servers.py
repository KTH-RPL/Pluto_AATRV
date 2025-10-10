#!/usr/bin/env python3

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path
import tf.transformations as tft

from utils.global_planner import execute_global_planning
from robot_controller.msg import PlanGlobalPathAction, PlanGlobalPathFeedback, PlanGlobalPathResult

# Initial robot pose, will be updated from POSE_TOPIC
CURRENT_ROBOT_POSE = None
POSE_TOPIC = '/robot_pose'

# Callback function to update the robot's current pose
def pose_callback(data):
    global CURRENT_ROBOT_POSE
    CURRENT_ROBOT_POSE = data

class GlobalPlannerActionServer:
    def __init__(self):
        self._action_name = 'plan_global_path'
        rospy.loginfo(f"Initializing '{self._action_name}' action server...")

        # Initialize subscriber for the robot's current pose
        self.pose_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, pose_callback)
        rospy.loginfo(f"Subscribed to robot pose on topic {POSE_TOPIC}")

        # Wait until the first pose is received
        rospy.loginfo("Waiting for the first robot pose to be received...")
        while CURRENT_ROBOT_POSE is None and not rospy.is_shutdown():
            try:
                rospy.sleep(0.5)
            except rospy.ROSInterruptException:
                rospy.loginfo("Shutdown requested while waiting for pose.")
                return
        rospy.loginfo("Robot pose received. Action server is ready.")

        # Initialize Action Server
        self._as = actionlib.SimpleActionServer(
            self._action_name,
            PlanGlobalPathAction,
            execute_cb=self.execute_callback,
            auto_start=False
        )
        self._as.start()
        rospy.loginfo(f"Action server {self._action_name} started.")

    # Callback when received a new goal
    def execute_callback(self, goal):
        rospy.loginfo(f"Received a new goal with {len(goal.waypoints.poses)} waypoints.")

        if not goal.waypoints.poses:
            rospy.logwarn("Goal contains no waypoints. Aborting action.")
            self._as.set_aborted(text="Received an empty goal array.")
            return
        
        # Find the nearest waypoint to the current robot pose
        robot_pos = CURRENT_ROBOT_POSE.pose.position

        # TO REMOVE: SIMULATE INITIAL ROBOT POSE
        robot_pos.x, robot_pos.y = (138, 82)

        waypoints = goal.waypoints.poses
        min_dist_sq = float('inf')
        start_index = 0

        for i, waypoint in enumerate(waypoints):
            waypoint_pos = waypoint.position
            dist_sq = (robot_pos.x - waypoint_pos.x)**2 + (robot_pos.y - waypoint_pos.y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                start_index = i
        
        rospy.loginfo(f"Nearest waypoint is at index {start_index}. Omitting {start_index} point(s) from the start of the list.")

        # Create a new list of waypoints to process, starting from the nearest one.
        waypoints_to_process = waypoints[start_index:]

        # Initialize variables for the planning loop
        full_path = Path()
        full_path.header.frame_id = "map"
        current_start_pose = CURRENT_ROBOT_POSE

        # Loop through each waypoint in the goal
        for i, waypoint in enumerate(waypoints_to_process):
            rospy.loginfo(f"Planning for waypoint {i + 1}/{len(waypoints_to_process)}...")

            # 1. Check for Preemption: Has the client cancelled this goal?
            if self._as.is_preempt_requested():
                rospy.loginfo("Goal has been preempted by the client.")
                self._as.set_preempted()
                return

            # 2. Define Start and Goal for the planner function
            start_tuple = (current_start_pose.pose.position.x, current_start_pose.pose.position.y)
            goal_tuple = (waypoint.position.x, waypoint.position.y)

            # 3. Execute the Global Planner
            try:
                rospy.loginfo(f"Running RRT from {start_tuple} to {goal_tuple}")
                path_points, _, _, _ = execute_global_planning(start_tuple, goal_tuple, sim_plan=False)

                if not path_points:
                    rospy.logerr(f"Global planner failed to find a path to waypoint {i + 1}. Aborting.")
                    self._as.set_aborted(text=f"Failed to plan to waypoint {i + 1}")
                    return

                segment_path = self.points_to_path_msg(path_points)
                rospy.loginfo(f"Successfully generated path segment for waypoint {i + 1}.")

            except Exception as e:
                rospy.logerr(f"An error occurred during global planning: {e}")
                self._as.set_aborted(text=f"Exception during planning: {e}")
                return

            # 4. Publish Feedback
            feedback = PlanGlobalPathFeedback()
            feedback.current_segment = segment_path
            self._as.publish_feedback(feedback)

            # 5. Append segment to the full path and update the start for the next iteration
            if full_path.poses:
                full_path.poses.extend(segment_path.poses[1:]) # for goal 2 until last, no need to append start position to the full path
            else:
                full_path.poses.extend(segment_path.poses)
            if segment_path.poses:
                current_start_pose = segment_path.poses[-1]

        # 6. Once all waypoints are processed, set the final result
        result = PlanGlobalPathResult()
        full_path.header.stamp = rospy.Time.now()
        result.global_plan = full_path
        self._as.set_succeeded(result)
        rospy.loginfo("Successfully planned path through all waypoints.")

    def points_to_path_msg(self, points):
        # Converts a list of (x, y) points to a nav_msgs/Path message.
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()

        for p in points:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = p[0]
            pose_stamped.pose.position.y = p[1]
            pose_stamped.pose.orientation.w = 1.0  # Default orientation
            path_msg.poses.append(pose_stamped)

        return path_msg

if __name__ == '__main__':
    try:
        rospy.init_node('global_planner_action_server')
        server = GlobalPlannerActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
