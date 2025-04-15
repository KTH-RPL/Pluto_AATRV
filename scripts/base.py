#!/usr/bin/env miniconda3/envs/milestone1/bin/python

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
from publishers import PathPublisher
import py_trees.display as display

class M1BehaviourTree(ptr.trees.BehaviourTree):

    def __init__(self):

        rospy.loginfo("Initialising behaviour tree")

    
        s0 = pt.composites.Sequence(name="pluto_move",children=[generate_path(),controls()])

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
        self.goal_published = None

        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)
        rospy.Subscriber("/goal", Pose, self.goal_callback)

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def goal_callback(self, msg):
        self.goal_pose = msg.position

    def update(self):

        if self.goal_pose:
            self.goal_published = self.goal_pose
        elif self.goal_published:
            rospy.loginfo("[goal_reached] Calculating Goal and Robot distance")
            distance = ((self.robot_pose.x - self.goal_published.x) ** 2 +
                    (self.robot_pose.y - self.goal_published.y) ** 2) ** 0.5

            if distance < 1:  # threshold
                self.goal_reached = True
        
        if self.goal_reached:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class generate_path(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising generate_path behaviour.")
        super(generate_path, self).__init__("generate_path")

        self.path = None
        self.path_generated = False
        self.goal_pose = None
        self.robot_pose = None

        rospy.Subscriber("/goal", Pose, self.goal_callback)
        rospy.Subscriber("/robot_pose", Pose, self.robotPose_callback)

    def goal_callback(self, msg):    
        self.goal_pose = msg.position       

    def robotPose_callback(self, msg):
        self.robot_pose = msg.position

    def update(self):   
        if not self.path_generated and self.goal_pose and self.robot_pose:

            self.path, _, _, _ = execute_planning(self.goal_pose, self.robot_pose)
            rospy.loginfo("[generate_path] Path generated successfully.")
            self.path_generated = True


            ## publish path
            rate = rospy.Rate(10)
            pp = PathPublisher()
            pp.publish_path(self.path)
            rate.sleep()

            return pt.common.Status.SUCCESS
        
        elif self.path_generated:
            rospy.loginfo("[generate_path] Path already exist")
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
        
class controls(pt.behaviour.Behaviour):
    def __init__(self):
        rospy.loginfo("Initialising controls behaviour.")
        super(controls, self).__init__("controls")

    def update(self):
        ## publish control commands
        pf = PathFollower()
        try:
            pf.run()
            return pt.common.Status.RUNNING
        except Exception as e:
            rospy.logerr("Error in control branch {}".format(e))
            return pt.common.Status.FAILURE

if __name__ == "__main__":
    print(sys.executable)
    tree = M1BehaviourTree()