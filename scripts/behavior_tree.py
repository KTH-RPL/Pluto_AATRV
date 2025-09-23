#!/usr/bin/env python3

import sys
import py_trees as pt
import py_trees_ros as ptr
import rospy
import actionlib
from rsequence import RSequence
# from control import NavigationSystem
from geometry_msgs.msg import PoseArray, PoseStamped, Pose
from nav_msgs.msg import Path

from robot_controller.msg import PlanGlobalPathAction, PlanGlobalPathGoal ## Need to change
import gmap_utility
import py_trees.display as display
import matplotlib.pyplot as plt
import json
import os

class M1BehaviourTree(ptr.trees.BehaviourTree):
    def __init__(self):
        rospy.loginfo("Initialising behaviour tree")

        c_goal = goal_robot_condition()
        # global_path_client  ### Action Server is already subscibing to the robot_pose, will it be good if i subscribe??
    
        b0 = pt.composites.Selector(
            name="Global Planner Fallback",
            children=[global_path_exist(c_goal),global_path_client(c_goal) ]
        )
        # b1 = pt.composites.Selector(name="Home fallback",children=[TaskCompletedCondition(c_goal),GoHomeClient(c_goal)])
        s1 = pt.composites.Sequence(name = "Pluto Sequence", children = [b0, control_planner(c_goal)])

        b1 = pt.composites.Selector(name="Goal Reached Fallback",children=[goal_reached(c_goal),s1])

        tree = RSequence(name="Main sequence", children=[b1])
        super(M1BehaviourTree, self).__init__(tree)

        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown():
            self.tick_tock(0.1) # 10Hz
            # rospy.loginfo("\n" + display.unicode_tree(self.root, show_status=True))


class goal_robot_condition():
    def __init__(self):
        self.current_goal = None
        self.robot_pose = None
        self.goals = None  
        self.get_global_path = None
        # self.task_completed = False
        self.pose_file = os.path.expanduser("~/robot_last_pose.json")
        os.makedirs(os.path.dirname(self.pose_file), exist_ok=True)
        # self.global_path_pub = rospy.Publisher('/global_path', Path, queue_size=10)
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_cb)
        rospy.Subscriber("/waypoints", Path,self.waypoints_callback)
        # Save pose every 5 seconds
        rospy.Timer(rospy.Duration(5.0), self.save_pose_timer_cb)
        # Save pose on shutdown
        rospy.on_shutdown(self.save_last_pose_on_shutdown)
        
    def robot_pose_cb(self, msg):
        self.robot_pose = msg
        # if self.home_pose is None:
        #     self.home_pose = msg.pose
        #     rospy.loginfo(f"[Home Pose Saved] x: {self.home_pose.position.x}, y: {self.home_pose.position.y}")
    
    def save_pose_timer_cb(self, event):
        if self.robot_pose:
            self.save_pose_to_file(self.robot_pose.pose)

    def save_last_pose_on_shutdown(self):
        """Save the last known pose when ROS shuts down."""
        if self.robot_pose:
            rospy.loginfo("Saving last robot pose on shutdown.")
            self.save_pose_to_file(self.robot_pose.pose)

    def save_pose_to_file(self, pose):
        pose_dict = {
            "position": {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z
            },
            "orientation": {
                "x": pose.orientation.x,
                "y": pose.orientation.y,
                "z": pose.orientation.z,
                "w": pose.orientation.w
            }
        }

        try:
            with open(self.pose_file, 'w') as f:
                json.dump(pose_dict, f, indent=4)
        except Exception as e:
            rospy.logwarn(f"Failed to save pose: {e}")
    def waypoints_callback(self, msg):
        if not self.goals:   ### Need to check like if goal changes then it wont run again
            # Convert Path (PoseStamped list) into PoseArray
            pose_array = PoseArray()
            pose_array.header.frame_id = msg.header.frame_id
            pose_array.header.stamp = rospy.Time.now()

            for ps in msg.poses:
                pose = Pose()
                pose.position = ps.pose.position
                pose.orientation = ps.pose.orientation
                pose_array.poses.append(pose)
            self.goals = pose_array
            rospy.loginfo(f"[Behaviour Tree] Received {len(self.goals)} goals.")

class goal_reached(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(goal_reached, self).__init__("Goal_reached")
        self.c_goal = c_goal
        self.finished = False
        

    def update(self):
        if self.finished == True:
            return pt.common.Status.SUCCESS
        
        if not self.c_goal.goals or not self.c_goal.robot_pose:
            rospy.logerr("Either goal is empty or not getting robot_pose")
            return pt.common.Status.RUNNING

        final_goal = self.c_goal.goals.poses[-1].position
        current_pose = self.c_goal.robot_pose.pose.position
        distance = ((current_pose.x - final_goal.x) ** 2 + (current_pose.y - final_goal.y) ** 2) ** 0.5

        if distance < 0.6:
            rospy.loginfo("[goal_reached] Final goal reached.")
            self.finished = True
            return pt.common.Status.SUCCESS

        return pt.common.Status.FAILURE

## May be in this i should add proper condition of wheter path correctly added
class global_path_exist(pt.behaviour.Behaviour):
    def __init__(self,c_goal):
        super(global_path_exist,self).__init__("global_path_exists")
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

    def initialise(self):
        self.sent = False
        self.c_goal.get_global_path = None
        rospy.loginfo("[Behaviour Tree] Waiting for the 'plan_global_path' action server...")
        self.client.wait_for_server()
        rospy.loginfo("[Behaviour Tree] Action server found")
        
    def feedback_callback(self,feedback):
        try:
            rospy.loginfo(
                "Received feedback: Path segment with %d poses has been planned.",
                len(feedback.current_segment.poses)
            )
            plt.close('all')
            gmap_utility.polygon_map.visualize(feedback.current_segment.poses)

        except Exception as e:
            rospy.logwarn("Feedback callback failed: %s", str(e))

    def done_callback(self,status, result):
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Action finished successfully!")
            rospy.loginfo("Final global path contains %d poses.", len(result.global_plan.poses))
            try:
                plt.close('all')
                gmap_utility.polygon_map.visualize(result.global_plan.poses)
            except Exception as e:
                rospy.logwarn("Visualization failed in done_callback: %s", str(e))
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
            self.client.send_goal(pluto_goal, done_cb = self.done_callback, feedback_cb=self.feedback_callback)
            self.sent = True
            goal_pose = PoseStamped()
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose = self.c_goal.goals[self.c_goal.goal_index]
            self.c_goal.goal_pose_pub.publish(goal_pose)
            rospy.loginfo(f"[MultiGoalClient] Sent goal {self.c_goal.goal_index + 1}/{len(self.c_goal.goals)}")

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            result = self.client.get_result()
            if result:   # always good to check it's not None
                self.c_goal.get_global_path = result.global_plan.poses
            self.sent = False

        elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
            rospy.logwarn("[MultiGoalClient] Goal execution failed.")
            # self.sent = False   assuming initialize will be called when it return failure 
            return pt.common.Status.FAILURE

        return pt.common.Status.RUNNING

class control_planner(pt.behaviour.Behaviour):
    def __init__(self,c_goal):
        super(control_planner,self).__init__("Control")
        self.c_goal = c_goal
    def update(self):
        if self.c_goal.get_global_path:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE




# class MultiGoalClient(pt.behaviour.Behaviour):
#     def __init__(self, c_goal):
#         super(MultiGoalClient, self).__init__("SendGoals")
#         self.client = actionlib.SimpleActionClient("pluto_goal", PlutoGoalAction)
#         self.sent = False
#         self.feedback_msg = None
#         self.c_goal = c_goal

#     def initialise(self):
#         self.sent = False
#         self.feedback_msg = None
#         rospy.loginfo("[MultiGoalClient] Waiting for action server...")
#         self.client.wait_for_server()

#     def feedback_cb(self, feedback):
#         self.feedback_msg = feedback.status
#         rospy.loginfo(self.feedback_msg)
        

#     def update(self):
#         if self.c_goal.goal_index >= len(self.c_goal.goals):
#             return pt.common.Status.SUCCESS

#         if not self.sent:
#             pluto_goal = PlutoGoalGoal()
#             pluto_goal.goal = self.c_goal.goals[self.c_goal.goal_index]
#             self.client.send_goal(pluto_goal, feedback_cb=self.feedback_cb)
#             self.sent = True
#             goal_pose = PoseStamped()
#             goal_pose.header.stamp = rospy.Time.now()
#             goal_pose.pose = self.c_goal.goals[self.c_goal.goal_index]
#             self.c_goal.goal_pose_pub.publish(goal_pose)
#             rospy.loginfo(f"[MultiGoalClient] Sent goal {self.c_goal.goal_index + 1}/{len(self.c_goal.goals)}")

#         if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
#             self.c_goal.goal_index += 1
#             self.sent = False

#         elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
#             rospy.logwarn("[MultiGoalClient] Goal execution failed.")
#             # self.sent = False   assuming initialize will be called when it return failure 
#             return pt.common.Status.FAILURE

#         return pt.common.Status.RUNNING

# class TaskCompletedCondition(pt.behaviour.Behaviour):
#     def __init__(self, c_goal):
#         super(TaskCompletedCondition, self).__init__("TaskCompleted?")
#         self.c_goal = c_goal

#     def update(self):
#         if self.c_goal.task_completed:
#             rospy.loginfo("[TaskCompletedCondition] All tasks done. Preventing re-execution.")
#             return pt.common.Status.SUCCESS
#         return pt.common.Status.FAILURE

# class GoHomeClient(pt.behaviour.Behaviour):
#     def __init__(self, c_goal):
#         super(GoHomeClient, self).__init__("GoHomeClient")
#         self.client = actionlib.SimpleActionClient("pluto_goal", PlutoGoalAction)
#         self.navsystem = NavigationSystem()
#         self.sent = False
#         self.c_goal = c_goal

#     def initialise(self):
#         self.sent = False
#         rospy.loginfo("[GoHomeClient] Waiting for action server...")
#         self.client.wait_for_server()

#     def update(self):
#         if not self.sent:
#             pluto_goal = PlutoGoalGoal()
#             pluto_goal.goal = self.c_goal.home_pose
#             self.client.send_goal(pluto_goal)
#             self.sent = True
#             goal_pose = PoseStamped()
#             goal_pose.header.stamp = rospy.Time.now()
#             goal_pose.pose = self.c_goal.home_pose
#             self.c_goal.goal_pose_pub.publish(goal_pose)
#             rospy.loginfo("[GoHomeClient] Sent robot to home position.")

#         if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
#             # sent command to stop the robot finally 
#             self.navsystem.stop_robot()
#             self.c_goal.task_completed = True
#             rospy.loginfo("[GoHomeClient] Reached home position.")
#             return pt.common.Status.SUCCESS
#         elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
#             rospy.logwarn("[GoHomeClient] Failed to return home.")
#             return pt.common.Status.FAILURE

#         return pt.common.Status.RUNNING


if __name__ == "__main__":
    print(sys.executable)
    rospy.init_node("Multi_Goal_behaviour_tree_controller")
    tree = M1BehaviourTree()
