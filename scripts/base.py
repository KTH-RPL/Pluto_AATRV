#!/usr/bin/env python3

import sys
import time
import py_trees as pt
import py_trees_ros as ptr
import rospy
from rsequence import RSequence
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path
from local_planner import execute_planning

from publishers1 import RobotPosePublisher
import py_trees.display as display
from control import NavigationSystem

class M1BehaviourTree(ptr.trees.BehaviourTree):

    def __init__(self):

        rospy.loginfo("Initialising behaviour tree")
        
        # s0 = pt.composites.Sequence(name="pluto_move",children=[planning_control()])
        s0 = planning_control()
        b0 = pt.composites.Selector(
            name="Goal fallback", 
            children=[goal_reached(),s0]
        )  


        tree = RSequence(name="Main sequence", children=[b0]) 
        super(M1BehaviourTree, self).__init__(tree)

        # execute the behaviour tree
        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown(): 
            self.tick_tock(1)
            rospy.loginfo("\n" + display.unicode_tree(self.root, show_status=True))

class goal_reached(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising goal_reached behaviour.")
        super(goal_reached, self).__init__("Goal_reached")

        self.goal_reached = False
        self.robot_pose = None
        self.goal_pose = None
        self.fp = False
        

        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)
        rospy.Subscriber("/goal_pose", PoseStamped, self.goal_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg

    def goal_callback(self, msg):
        self.goal_pose = msg

    def update(self):

        if self.goal_pose and self.robot_pose:
            if self.fp ==  False:
                rospy.loginfo("[goal_reached] Got the goal and robot pose")
                self.fp = True
            rospy.loginfo("[goal_reached] Checking Goal Condition")
            distance = ((self.robot_pose.pose.position.x - self.goal_pose.pose.position.x) ** 2 +
                    (self.robot_pose.pose.position.y - self.goal_pose.pose.position.y) ** 2) ** 0.5

            if distance < 1:  # threshold
                self.goal_reached = True
        else:
            rospy.logerr("[goal_reached] Didn't get the goal and robot pose")
            return pt.common.Status.RUNNING

        if self.goal_reached:
            rospy.loginfo("[goal_reached] Goal Reached")
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class planning_control(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising generate_path behaviour.")
        super(planning_control, self).__init__("generate_path")

        self.path = None
        self.path_generated = False
        self.goal_pose = None
        self.robot_pose = None

        rospy.Subscriber("/goal_pose", PoseStamped, self.goal_callback)
        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)
        self.nav_system = NavigationSystem()


    def goal_callback(self, msg):    
        self.goal_pose = msg               

    def robotPose_callback(self, msg):
        self.robot_pose = msg

    def update(self):   

        if not self.path_generated:
            self.nav_system.current_goal = self.goal_pose
            self.nav_system.current_pose = self.robot_pose
            # self.path = nav_system.generate_offset_path()
            goal = (self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
            robot = (self.robot_pose.pose.position.x, self.robot_pose.pose.position.y)
            self.path, _, _, _ = execute_planning(robot, goal)
            rospy.loginfo("[generate_path] Path generated successfully.")
            self.path_generated = True   
            self.nav_system.current_path = self.path   
            rospy.sleep(1)

        elif self.path_generated:
            rospy.loginfo("[planning_control] Control ")
            self.nav_system.run_control()
            return pt.common.Status.RUNNING
        else:
            return pt.common.Status.FAILURE
        
# class controls(pt.behaviour.Behaviour):
#     def __init__(self):
#         rospy.loginfo("Initialising controls behaviour.")
#         super(controls, self).__init__("controls")

#     def update(self):
#         ## publish control commands
#         pf = PathFollower()
#         try:
#             pf.run()
#             return pt.common.Status.RUNNING
#         except Exception as e:
#             rospy.logerr("Error in control branch {}".format(e))
#             return pt.common.Status.FAILURE

if __name__ == "__main__":
    rospy.init_node("btree")
    print(sys.executable)
    tree = M1BehaviourTree()
