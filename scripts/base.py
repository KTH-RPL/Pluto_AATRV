#!/usr/bin/env python3

import sys
import time
import py_trees as pt
import py_trees_ros as ptr
import rospy
from rsequence import RSequence
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from local_planner import execute_planning

from control import PathFollower
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
        

        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)
        rospy.Subscriber("/goal/pose", Pose, self.goal_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def goal_callback(self, msg):
        self.goal_pose = msg.position

    def update(self):

        if self.goal_pose and self.robot_pose:
            rospy.loginfo("[goal_reached] Got the goal and robot pose")
           
            distance = ((self.robot_pose.x - self.goal_pose.x) ** 2 +
                    (self.robot_pose.y - self.goal_pose.y) ** 2) ** 0.5

            if distance < 0.2:  # threshold
                self.goal_reached = True
        else:
            rospy.logerr("[goal_reached] Didn't get the goal and robot pose")
            return pt.common.Status.RUNNING

        if self.goal_reached:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class planning_control(pt.behaviour.Behaviour):
    def __init__(self,nav_system):
        rospy.loginfo("Initialising generate_path behaviour.")
        super(planning_control, self).__init__("generate_path")

        self.path = None
        self.path_generated = False
        self.goal_pose = None
        self.robot_pose = None

        rospy.Subscriber("/goal/pose", Pose, self.goal_callback)
        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)


    def goal_callback(self, msg):    
        self.goal_pose = msg.position    
           

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def update(self):   
        if not self.path_generated:
            nav_system = NavigationSystem()
            nav_system.current_goal = self.goal_pose
            nav_system.current_pose = self.robot_pose
            # self.path = nav_system.generate_offset_path()
            self.path, _, _, _ = execute_planning(self.goal_pose, self.robot_pose)
            rospy.loginfo("[generate_path] Path generated successfully.")
            self.path_generated = True   
            nav_system.current_path = self.path   
            rospy.sleep(10)

        elif self.path_generated:
            nav_system.run_control()
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
    print(sys.executable)
    tree = M1BehaviourTree()
