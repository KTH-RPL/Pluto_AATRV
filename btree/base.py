#!/usr/bin/env python3

import time
import py_trees as pt, py_trees_ros as ptr, rospy
from rsequence import RSequence
from std_srvs.srv import Empty, SetBool, SetBoolRequest 
from geometry_msgs.msg import PoseWithCovarianceStamped

class BehaviourTree(ptr.trees.BehaviourTree):

    def __init__(self):

        rospy.loginfo("Initialising behaviour tree")

        b0 = pt.composites.Selector(
            name="Rotate fallback", 
            children=[counter(2, "Rotated?"), go("Rotate!", 0, 0)]
        )  

        # s0 =pt.composites.Sequence(name="localization sequence",children=[b0,localize(0.05,20),clearCostmap()])
        b1 = pt.composites.Selector(name="localization fallback",children=[localize(0.01),go("Rotate",0,1)])
        s0 =pt.composites.Sequence(name="localization sequence",children=[b1,b0,clearCostmap()])
    
        # lower head
        
        # b2 = movehead("down")
        # b1 = pick()
        # go to door until at door

   
          # for C part 31

        # tuck the arm
        # b1 = tuckarm()
        

        # go to table
        # b2 = pt.composites.Selector(
        # 	name="Go to table fallback",
        # 	children=[counter(5, "At table?"), go("Go to table!", 0, -1)]
        # )

        # move to table
        # b3 = pt.composites.Selector(
        #     name="Go to table2 fallback",
        #     children=[counter(8, "At table2?"), go("Go to table2!", 2, 0)]
        # )


        # b4 = place()
        # become the tree
        # tree = RSequence(name="Main sequence", children=[b0, b1, b2, b3, b4])
        # tree = RSequence(name="Main sequence", children=[b4, b1]) 
        tree = RSequence(name="Main sequence", children=[s0]) 
        super(BehaviourTree, self).__init__(tree)

        # execute the behaviour tree
        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown(): self.tick_tock(1)	