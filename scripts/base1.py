#!/usr/bin/env python3

import sys
import time
import py_trees as pt
import py_trees_ros as ptr
import rospy
from rsequence import RSequence
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
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
            children=[goal_reached(self.nav_system,self.goal_manager), s0]
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
        self.nextgoal = True
        rospy.Subscriber("/goal_pose", PoseArray, self.goals_callback)

    def goals_callback(self, msg):
        # rospy.loginfo("Received new goal list.")
        if not self.goals:
            self.goals = list(msg.poses)
            # rospy.loginfo(self.goals)
            if self.goals:
                self.current_goal = self.goals.pop(0)

    def advance_goal(self):
        if self.goals:
            self.current_goal = self.goals.pop(0)
        else:
            self.current_goal = None
        
        self.next_goal = True


class goal_reached(pt.behaviour.Behaviour):
    def __init__(self, navsystem, goal_manager):
        super(goal_reached, self).__init__("Goal_reached")
        self.goal_manager = goal_manager
        self.navsystem = navsystem
        self.robot_pose = None
        self.goal_pose = None
        self.fp = False

        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg

    def update(self):
        if self.robot_pose and self.goal_manager.current_goal:
            if self.fp ==  False:
                rospy.loginfo("[goal_reached] Got the goal and robot pose")
                self.fp = True
            # rospy.loginfo("[goal_reached] Checking Goal Condition")
            self.goal_pose = self.goal_manager.current_goal
            
            distance = ((self.robot_pose.pose.position.x - self.goal_pose.position.x) ** 2 +
                    (self.robot_pose.pose.position.y - self.goal_pose.position.y) ** 2) ** 0.5

            if distance < 0.5:
                if (len(self.goal_manager.goals) == 0):
                    rospy.loginfo("[goal_reached] Final Goal reached!")
                    self.navsystem.stop_robot()                    
                    return pt.common.Status.SUCCESS
                else:
                    rospy.loginfo("[goal_reached] Intermediate Goal reached!")
                    rospy.loginfo(self.goal_pose)

                    self.goal_manager.advance_goal()
                    return pt.common.Status.FAILURE
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

        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg

    def update(self):
      current_goal = self.goal_manager.current_goal
      if self.goal_manager.nextgoal == True:
          goal_pose = (current_goal.position.x,current_goal.position.y)
          self.nav_system.current_goal = goal_pose
          self.nav_system.current_pose = self.robot_pose
          self.nav_system.targetid = 0
          current_position = (self.robot_pose.pose.position.x, 
                          self.robot_pose.pose.position.y)
          self.nav_system.current_path, _, _, _ = execute_planning(current_position,goal_pose)

          rospy.loginfo("[planning_control] Path generated.")
          self.path_generated = True
          self.goal_manager.nextgoal = False
          rospy.sleep(1)
      is_last_goal = (len(self.goal_manager.goals) == 0)
      self.nav_system.run_control(is_last_goal=is_last_goal)
      return pt.common.Status.RUNNING

if __name__ == "__main__":
    print(sys.executable)
    rospy.init_node("behaviour_tree_controller")
    tree = M1BehaviourTree()
