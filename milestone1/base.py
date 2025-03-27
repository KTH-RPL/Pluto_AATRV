#!/usr/bin/env python3

import time
import py_trees as pt, py_trees_ros as ptr, rospy
from btree.rsequence import RSequence
from std_srvs.srv import Empty, SetBool, SetBoolRequest 
from geometry_msgs.msg import Pose,PoseWithCovarianceStamped
from nav_msgs.msg import Path
from local_planner import execute_planning
from control import PathFollower
from publishers import PathPublisher

class BehaviourTree(ptr.trees.BehaviourTree):

    def __init__(self):

        rospy.loginfo("Initialising behaviour tree")
        p1 = pt.composites.Selector(name="Move to goal",children=[path_already_exist,generate_path])
        s1 = pt.composites.Sequence(name = "path_planner",children = [getGoalPosition,p1])
    
        s0 = pt.composites.Sequence(name="pluto_move",children=[s1,controls])

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

        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)
        rospy.Subscriber("/goal", Pose, self.goal_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def goal_callback(self, msg):
        self.goal_pose = msg.position

    def update(self):

        if self.goal_pose and self.robot_pose:
            distance = ((self.robot_pose.x - self.goal_pose.x) ** 2 +
                    (self.robot_pose.y - self.goal_pose.y) ** 2) ** 0.5

            if distance < 0.1:  # threshold
                self.goal_reached = True
        
        if self.goal_reached:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class getGoalPosition(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising getGoalPosition behaviour.")
        super().__init__("getGoalPosition")

        self.goal_pose = None

        rospy.Subscriber("/goal", Pose, self.goal_callback)

    def goal_callback(self, msg):
        self.goal_pose = msg.position

    def update(self):
        if self.goal_pose:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class path_already_exist(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising path_already_exist behaviour.")
        super().__init__("path_already_exist")

        self.path = None

        rospy.Subscriber("/path", Path, self.path_callback) 

    def path_callback(self, msg):
        self.path = msg.poses

    def update(self):     
        if self.path:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class generate_path(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising generate_path behaviour.")
        super().__init__("generate_path")

        self.path = None

        rospy.Subscriber("/goal", Pose, self.goal_callback)
        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)

    def goal_callback(self, msg):    
        self.goal_pose = msg.position       

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def update(self):   
        if self.goal_pose and self.robot_pose:
            self.path, _, _, _ = execute_planning(self.goal_pose, self.robot_pose)

            ## publish path
            pp = PathPublisher()
            pp.publish_path(self.path)

            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class controls(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising controls behaviour.")
        super().__init__("controls")

    def update(self):
        ## publish control commands
        pf = PathFollower()
        pf.run()
        return pt.common.Status.RUNNING


        