#!/usr/bin/env python3

import time
import py_trees as pt, py_trees_ros as ptr, rospy
from btree.rsequence import RSequence
from std_srvs.srv import Empty, SetBool, SetBoolRequest 
from geometry_msgs.msg import Pose,PoseWithCovarianceStamped

class BehaviourTree(ptr.trees.BehaviourTree):

    def __init__(self):

        rospy.loginfo("Initialising behaviour tree")
        p1 = pt.composites.Selector(name="Move to goal",children=[checkPath,generate_path()])
        s1 = pt.composites.Sequence(name = "path_planner",children = [getGoalPosition,p1])
        s2 = controls()
        s0 = pt.composites.Sequence(name="pluto_move",children=[s1,s2])

        b0 = pt.composites.Selector(
            name="Goal fallback", 
            children=[goal_reached,s0 ]
        )  


        tree = RSequence(name="Main sequence", children=[b0]) 
        super(BehaviourTree, self).__init__(tree)

        # execute the behaviour tree
        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown(): self.tick_tock(1)


class goal_reached(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising goal_reached behaviour.")
        super().__init__("Goal_reached")

        self.goal_reached = False
        self.robot_pose = None
        self.goal_pose = None

        rospy.Subscriber("/placeholder1", Pose, self.robotPose_callback)
        rospy.Subscriber("/placeholder2", Pose, self.goal_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg.pose

    def goal_callback(self, msg):
        self.goal_pose = msg.pose

    def update(self):
        
        if self.robot_pose and self.goal_pose:
            if (self.robot_pose.x == self.goal_pose.x) and (self.robot_pose.y == self.goal_pose.y):
                rospy.loginfo("Goal Reached!")
                self.goal_reached = True
        
        if self.goal_reached:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
