#!/usr/bin/env python3

import sys
import py_trees as pt
import py_trees_ros as ptr
import rospy
import actionlib
from rsequence import RSequence
from control import NavigationSystem
from geometry_msgs.msg import PoseArray, PoseStamped
from milestone1.msg import PlutoGoalAction, PlutoGoalGoal  # need to change
import py_trees.display as display

class M1BehaviourTree(ptr.trees.BehaviourTree):
    def __init__(self):
        rospy.loginfo("Initialising behaviour tree")

        c_goal = goal_robot_condition()
        multi_goal_client = MultiGoalClient(c_goal)

        b0 = pt.composites.Selector(
            name="Goal fallback",
            children=[goal_reached(c_goal), multi_goal_client]
        )

        s1 = pt.composites.Sequence(name = "Home Sequence", children = [b0,GoHomeClient(c_goal)])

        tree = RSequence(name="Main sequence", children=[b0, s1])
        super(M1BehaviourTree, self).__init__(tree)

        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown():
            self.tick_tock(1)
            # rospy.loginfo("\n" + display.unicode_tree(self.root, show_status=True))


class goal_robot_condition():
    def __init__(self):
        self.current_goal = None
        self.goal_index = 0
        self.goals = []
        self.feedback_status = 0
        self.home_pose = None
        rospy.Subscriber("/robot_pose", PoseStamped, self.home_pose_cb)
        rospy.Subscriber("/goal_pose", PoseArray, self.goals_callback)

    def home_pose_cb(self, msg):
        if self.home_pose is None:
            self.home_pose = msg.pose
            rospy.loginfo(f"[Home Pose Saved] x: {self.home_pose.position.x}, y: {self.home_pose.position.y}")
    def goals_callback(self, msg):
        if not self.goals:
            self.goals = msg.poses
            rospy.loginfo(f"[MultiGoalClient] Received {len(self.goals)} goals.")


class goal_reached(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(goal_reached, self).__init__("Goal_reached")
        self.robot_pose = None
        self.c_goal = c_goal
        
        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg

    def update(self):
        if not self.c_goal.goals or not self.robot_pose:
            rospy.logerr("Either goal is empty or not getting robot_pose")
            return pt.common.Status.RUNNING

        final_goal = self.c_goal.goals[-1].position
        current_pose = self.robot_pose.pose.position
        distance = ((current_pose.x - final_goal.x) ** 2 + (current_pose.y - final_goal.y) ** 2) ** 0.5

        if distance < 1.0 and self.c_goal.goal_index == (len(self.c_goal.goals) - 1) :
            rospy.loginfo("[goal_reached] Final goal reached.")
            return pt.common.Status.SUCCESS

        return pt.common.Status.FAILURE


class MultiGoalClient(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(MultiGoalClient, self).__init__("SendGoals")
        self.client = actionlib.SimpleActionClient("pluto_goal", PlutoGoalAction)
        self.sent = False
        self.feedback_msg = None
        self.c_goal = c_goal

    def initialise(self):
        self.sent = False
        self.feedback_msg = None
        rospy.loginfo("[MultiGoalClient] Waiting for action server...")
        self.client.wait_for_server()

    def feedback_cb(self, feedback):
        self.feedback_msg = feedback.status
        rospy.loginfo(self.feedback_msg)
        

    def update(self):
        if self.c_goal.goal_index >= len(self.c_goal.goals):
            return pt.common.Status.SUCCESS

        if not self.sent:
            pluto_goal = PlutoGoalGoal()
            pluto_goal.goal = self.c_goal.goals[self.c_goal.goal_index]
            self.client.send_goal(pluto_goal, feedback_cb=self.feedback_cb)
            self.sent = True
            rospy.loginfo(f"[MultiGoalClient] Sent goal {self.c_goal.goal_index + 1}/{len(self.c_goal.goals)}")

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            self.c_goal.goal_index += 1
            self.sent = False

        elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
            rospy.logwarn("[MultiGoalClient] Goal execution failed.")
            # self.sent = False   assuming initialize will be called when it return failure 
            return pt.common.Status.FAILURE

        return pt.common.Status.RUNNING


class GoHomeClient(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(GoHomeClient, self).__init__("GoHomeClient")
        self.client = actionlib.SimpleActionClient("pluto_goal", PlutoGoalAction)
        self.navsystem = NavigationSystem()
        self.sent = False
        self.c_goal = c_goal

    def initialise(self):
        self.sent = False
        rospy.loginfo("[GoHomeClient] Waiting for action server...")
        self.client.wait_for_server()

    def update(self):
        if not self.sent:
            pluto_goal = PlutoGoalGoal()
            pluto_goal.goal = self.c_goal.home_pose
            self.client.send_goal(pluto_goal)
            self.sent = True
            rospy.loginfo("[GoHomeClient] Sent robot to home position.")

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            self.navsystem.stop_robot()
            rospy.loginfo("[GoHomeClient] Reached home position.")

            # sent command to stop the robot finally 

            return pt.common.Status.SUCCESS
        elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
            rospy.logwarn("[GoHomeClient] Failed to return home.")
            return pt.common.Status.FAILURE

        return pt.common.Status.RUNNING


if __name__ == "__main__":
    print(sys.executable)
    rospy.init_node("Multi_Goal_behaviour_tree_controller")
    tree = M1BehaviourTree()
