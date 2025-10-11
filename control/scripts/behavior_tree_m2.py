#!/usr/bin/env python3

import sys
import py_trees as pt
import py_trees_ros as ptr
import rospy
import actionlib
from rsequence import RSequence
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from nav_msgs.msg import Path
from std_msgs.msg import Float32

from robot_controller.msg import PlanGlobalPathAction, PlanGlobalPathGoal
from robot_controller.srv import RunControl

import gmap_utility
import py_trees.display as display
import matplotlib.pyplot as plt
import json
import os


class M2BehaviourTree(ptr.trees.BehaviourTree):
    def __init__(self):
        rospy.loginfo("Initialising behaviour tree")

        c_goal = goal_robot_condition()

        # Create the check_localization behavior
        check_loc = check_localization(c_goal)
        
        b0 = pt.composites.Selector(
            name="Global Planner Fallback",
            children=[global_path_exist(c_goal), global_path_client(c_goal)]
        )

        # Now Pluto sequence checks localization before path planning & control
        s1 = pt.composites.Sequence(
            name="Pluto Sequence", 
            children=[check_loc, b0, control_planner(c_goal)]
        )

        b1 = pt.composites.Selector(
            name="Goal Reached Fallback",
            children=[goal_reached(c_goal), s1]
        )

        tree = RSequence(name="Main sequence", children=[b1])
        super(M2BehaviourTree, self).__init__(tree)

        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown():
            self.tick_tock(0.1)  # 10Hz


class goal_robot_condition():
    def __init__(self):
        self.current_goal = None
        self.robot_pose = None
        self.goals = None  
        self.get_global_path = None

        # Track goal index for multi-goal execution
        self.goal_index = 0
        self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=1)

        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_cb)
        rospy.Subscriber("/waypoints", Path, self.waypoints_callback)
        rospy.Subscriber("/transform_probability", Float32, self.transform_probability_cb)

        # Save pose on shutdown
        rospy.on_shutdown(self.save_last_pose_on_shutdown)

        # for localization condition 
        self.transform_probability = None
        self.pose_history_file = os.path.expanduser("~/robot_pose_history.json")
        os.makedirs(os.path.dirname(self.pose_history_file), exist_ok=True)

        self.valid_poses = []   # Store history of valid poses with scores
        self.last_good_pose = None  # To store the last pose with good score

    def transform_probability_cb(self, msg):
        self.transform_probability = msg.data   
        
    def robot_pose_cb(self, msg):
        self.robot_pose = msg

        if self.transform_probability is not None:
            pose_entry = {
                "timestamp": rospy.Time.now().to_sec(),
                "fitness_score": self.transform_probability,
                "pose": {
                    "position": {
                        "x": msg.pose.position.x,
                        "y": msg.pose.position.y,
                        "z": msg.pose.position.z
                    },
                    "orientation": {
                        "x": msg.pose.orientation.x,
                        "y": msg.pose.orientation.y,
                        "z": msg.pose.orientation.z,
                        "w": msg.pose.orientation.w
                    }
                }
            }
            
            self.valid_poses.append(pose_entry)
            
            # Update last good pose if score is good
            if self.transform_probability <= 4.2:
                self.last_good_pose = pose_entry
                
            # Keep only last 1000 poses
            if len(self.valid_poses) > 1000:
                self.valid_poses = self.valid_poses[-1000:]
                
            self.save_pose_history()
    
    def save_pose_history(self):
        try:
            with open(self.pose_history_file, 'w') as f:
                json.dump({"valid_poses": self.valid_poses}, f, indent=4)
        except Exception as e:
            rospy.logwarn(f"Failed to save pose history: {e}")

    def get_last_good_pose(self):
        """Get the most recent pose that had a good score (<= 4.2)."""
        if self.last_good_pose:
            return self.last_good_pose
        
        for pose in reversed(self.valid_poses):
            if pose["fitness_score"] <= 4.2:
                self.last_good_pose = pose
                return pose
        return None

    def save_last_pose_on_shutdown(self):
        rospy.loginfo("Saving last robot pose on shutdown.")
        self.save_pose_history()

    def waypoints_callback(self, msg):
        if not self.goals:   
            pose_array = PoseArray()
            pose_array.header.frame_id = msg.header.frame_id
            pose_array.header.stamp = rospy.Time.now()

            for ps in msg.poses:
                pose = Pose()
                pose.position = ps.pose.position
                pose.orientation = ps.pose.orientation
                pose_array.poses.append(pose)
            self.goals = pose_array
            rospy.loginfo(f"[Behaviour Tree] Received {len(self.goals.poses)} goals.")


class goal_reached(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(goal_reached, self).__init__("Goal_reached")
        self.c_goal = c_goal
        self.finished = False

    def update(self):
        if self.finished:
            return pt.common.Status.SUCCESS
        
        if not self.c_goal.goals or not self.c_goal.robot_pose:
            rospy.logerr("Either goal is empty or not getting robot_pose")
            return pt.common.Status.RUNNING

        final_goal = self.c_goal.goals.poses[-1].position
        current_pose = self.c_goal.robot_pose.pose.position
        distance = ((current_pose.x - final_goal.x) ** 2 + 
                    (current_pose.y - final_goal.y) ** 2) ** 0.5

        if distance < 0.6:
            rospy.loginfo("[goal_reached] Final goal reached.")
            self.finished = True
            return pt.common.Status.SUCCESS

        return pt.common.Status.FAILURE


class check_localization(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(check_localization, self).__init__("CheckLocalization")
        self.c_goal = c_goal
        self.max_score = 4.2
        self.pose_pub = rospy.Publisher('/initial_pose', PoseStamped, queue_size=1)

    def update(self):
        if self.c_goal.transform_probability is None:
            rospy.logwarn("[CheckLocalization] No localization score received")
            return pt.common.Status.RUNNING

        if self.c_goal.transform_probability > self.max_score:
            rospy.logwarn(f"[CheckLocalization] Poor localization score: {self.c_goal.transform_probability}")
            
            last_good = self.c_goal.get_last_good_pose()
            if last_good:
                corrected_pose = PoseStamped()
                corrected_pose.header.frame_id = "map"
                corrected_pose.header.stamp = rospy.Time.now()
                corrected_pose.pose.position.x = last_good["pose"]["position"]["x"]
                corrected_pose.pose.position.y = last_good["pose"]["position"]["y"]
                corrected_pose.pose.position.z = last_good["pose"]["position"]["z"]
                corrected_pose.pose.orientation.x = last_good["pose"]["orientation"]["x"]
                corrected_pose.pose.orientation.y = last_good["pose"]["orientation"]["y"]
                corrected_pose.pose.orientation.z = last_good["pose"]["orientation"]["z"]
                corrected_pose.pose.orientation.w = last_good["pose"]["orientation"]["w"]
                
                self.pose_pub.publish(corrected_pose)
                rospy.loginfo("[CheckLocalization] Published last good pose with score: %f", 
                              last_good["fitness_score"])
            
            return pt.common.Status.FAILURE

        return pt.common.Status.SUCCESS


class global_path_exist(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(global_path_exist, self).__init__("global_path_exists")
        self.c_goal = c_goal

    def update(self):
        if self.c_goal.get_global_path:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE


class global_path_client(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(global_path_client, self).__init__("SendGoals")
        self.client = actionlib.SimpleActionClient("plan_global_path", PlanGlobalPathAction)
        self.sent = False        
        self.c_goal = c_goal
        self.path_pub_ = rospy.Publisher("global_path", Path, queue_size=1, latch=True)

    def initialise(self):
        self.sent = False
        self.c_goal.get_global_path = None
        rospy.loginfo("[Behaviour Tree] Waiting for the 'plan_global_path' action server...")
        self.client.wait_for_server()
        rospy.loginfo("[Behaviour Tree] Action server found")
        
    def feedback_callback(self, feedback):
        try:
            rospy.loginfo(
                "Received feedback: Path segment with %d poses has been planned.",
                len(feedback.current_segment.poses)
            )
            # try: 
            #     plt.close('all')
            # except:
            #     pass
            # gmap_utility.polygon_map.visualize(feedback.current_segment.poses)

            # !!TO EVALUATE!!
            # Publish the intermediate path to the 'global_path' topic
            ## This would reduce init time of the robot (not waiting for the whole global plan)
            ## But need to do some changes in final_control_algo.cpp to able receiving segments
            # if feedback.global_plan.poses:
            #     self.path_pub_.publish(feedback.global_plan)
            #     rospy.loginfo("Published the intermediate path to 'global_path'.")
            # else:
            #     rospy.logwarn("Received an empty intermediate path. Not publishing.")

            except Exception as e:
                rospy.logwarn("Feedback callback failed: %s", str(e))

    def done_callback(self, status, result):
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Action finished successfully!")
            rospy.loginfo("Final global path contains %d poses.", len(result.global_plan.poses))

            # Publish the received path to the 'global_path' topic
            if result.global_plan.poses:
                # The 'result.global_plan' is already a nav_msgs/Path message, so we can publish it directly.
                self.path_pub_.publish(result.global_plan)
                rospy.loginfo("Published the final path to 'global_path'.")
            else:
                rospy.logwarn("Received an empty final path. Not publishing.")

            # try:
            #     plt.close('all')
            #     gmap_utility.polygon_map.visualize(result.global_plan.poses)
            # except Exception as e:
            #     rospy.logwarn("Visualization failed in done_callback: %s", str(e))
        elif status == actionlib.GoalStatus.PREEMPTED:
            rospy.logwarn("Action was preempted by a new goal.")
        else:
            rospy.logerr("Action failed with status code: %s", actionlib.GoalStatus.to_string(status))

    def update(self):
        if self.c_goal.get_global_path:
            return pt.common.Status.SUCCESS

        if not self.sent:
            pluto_goal = PlanGlobalPathGoal()
            pluto_goal.goal = self.c_goal.goals
            self.client.send_goal(pluto_goal, 
                                  done_cb=self.done_callback, 
                                  feedback_cb=self.feedback_callback)
            self.sent = True

            goal_pose = PoseStamped()
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose = self.c_goal.goals.poses[self.c_goal.goal_index]
            self.c_goal.goal_pose_pub.publish(goal_pose)
            rospy.loginfo(f"[MultiGoalClient] Sent goal {self.c_goal.goal_index + 1}/{len(self.c_goal.goals.poses)}")

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            result = self.client.get_result()
            if result:
                self.c_goal.get_global_path = result.global_plan.poses
            self.sent = False
            return pt.common.Status.SUCCESS

        elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
            rospy.logwarn("[MultiGoalClient] Goal execution failed.")
            return pt.common.Status.FAILURE

        return pt.common.Status.RUNNING


class control_planner(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(control_planner, self).__init__("Control")
        self.c_goal = c_goal
        rospy.wait_for_service('run_control')
        self.run_control_srv = rospy.ServiceProxy('run_control', RunControl)

    def update(self):
        if not self.c_goal.get_global_path:
            rospy.logwarn("[ControlPlanner] No global path, control not running")
            return pt.common.Status.FAILURE

        try:
            resp = self.run_control_srv(False)  # False = not last goal
            if resp.status == 0:   # RUNNING
                return pt.common.Status.RUNNING
            elif resp.status == 1: # SUCCESS
                return pt.common.Status.SUCCESS
            else:                  # FAILURE
                return pt.common.Status.FAILURE
        except rospy.ServiceException as e:
            rospy.logerr(f"[ControlPlanner] Service call failed: {e}")
            return pt.common.Status.FAILURE


if __name__ == "__main__":
    print(sys.executable)
    rospy.init_node("Multi_Goal_behaviour_tree_controller")
    tree = M2BehaviourTree()
