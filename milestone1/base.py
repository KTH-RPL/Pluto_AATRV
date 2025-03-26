#!/usr/bin/env python3

import time
import py_trees as pt, py_trees_ros as ptr, rospy
from btree.rsequence import RSequence
from std_srvs.srv import Empty, SetBool, SetBoolRequest 
from geometry_msgs.msg import PoseWithCovarianceStamped

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
