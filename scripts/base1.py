#!/usr/bin/env python3

import sys
import time
import py_trees as pt
import py_trees_ros as ptr
import rospy
from rsequence import RSequence
from geometry_msgs.msg import Pose, PoseArray
from local_planner import execute_planning
from control import NavigationSystem
import py_trees.display as display


class M1BehaviourTree(ptr.trees.BehaviourTree):
    def __init__(self):
        rospy.loginfo("Initialising behaviour tree")
        self.nav_system = NavigationSystem()

        self.goal_manager = GoalManager()

        s0 = planning_control(self.nav_system, self.goal_manager)
        b0 = pt.composites.Selector(
            name="Goal fallback",
            children=[goal_reached(self.goal_manager), s0]
        )

        tree = RSequence(name="Main sequence", children=[b0])
        super(M1BehaviourTree, self).__init__(tree)

        self.setup(timeout=10000)
        while not rospy.is_shutdown():
            self.tick_tock(1)
            rospy.loginfo("\n" + display.unicode_tree(self.root, show_status=True))


class GoalManager:
    def __init__(self):
        self.goals = []
        self.current_goal = None
        rospy.Subscriber("/goal_pose", PoseArray, self.goals_callback)

    def goals_callback(self, msg):
        rospy.loginfo("Received new goal list.")
        self.goals = list(msg.poses)
        if self.goals:
            self.current_goal = self.goals.pop(0)

    def next_goal(self):
        if self.goals:
            self.current_goal = self.goals.pop(0)
        else:
            self.current_goal = None


class goal_reached(pt.behaviour.Behaviour):
    def __init__(self, goal_manager):
        super(goal_reached, self).__init__("Goal_reached")
        self.goal_manager = goal_manager
        self.robot_pose = None
        self.goal_pose = None

        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def update(self):
        if self.goal_manager.current_goal is None:
            rospy.loginfo("[goal_reached] No current goal to check.")
            return pt.common.Status.FAILURE

        if self.robot_pose:
            goal_pose = self.goal_manager.current_goal.position
            distance = ((self.robot_pose.x - goal_pose.x) ** 2 +
                        (self.robot_pose.y - goal_pose.y) ** 2) ** 0.5

            if distance < 0.2:
                rospy.loginfo("[goal_reached] Goal reached!")
                self.goal_manager.next_goal()
                return pt.common.Status.SUCCESS
            else:
                return pt.common.Status.FAILURE
        else:
            rospy.logwarn("[goal_reached] Waiting for robot pose.")
            return pt.common.Status.RUNNING


class planning_control(pt.behaviour.Behaviour):
    def __init__(self, nav_system, goal_manager):
        super(planning_control, self).__init__("planning_control")
        self.nav_system = nav_system
        self.goal_manager = goal_manager
        self.robot_pose = None
        self.path_generated = False

        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def update(self):
      current_goal = self.goal_manager.current_goal
      if current_goal is None:
          rospy.loginfo("[planning_control] No goal to plan for.")
          return pt.common.Status.FAILURE

      if not self.path_generated and self.robot_pose:
          goal_pose = current_goal.position
          self.nav_system.current_goal = goal_pose
          self.nav_system.current_pose = self.robot_pose
          self.nav_system.current_path, _, _, _ = execute_planning(goal_pose, self.robot_pose)
          rospy.loginfo("[planning_control] Path generated.")
          self.path_generated = True
          rospy.sleep(1)

      elif self.path_generated:
          is_last_goal = (len(self.goal_manager.goals) == 0)
          goal_reached = self.nav_system.run_control(is_last_goal=is_last_goal)

          if goal_reached:
              self.path_generated = False
              rospy.loginfo("[planning_control] Goal reached and control stopped.")
              return pt.common.Status.SUCCESS
          else:
              return pt.common.Status.RUNNING

      else:
          rospy.loginfo("[planning_control] Waiting for path generation.")
          return pt.common.Status.RUNNING
if __name__ == "__main__":
    print(sys.executable)
    rospy.init_node("behaviour_tree_controller")
    tree = M1BehaviourTree()
