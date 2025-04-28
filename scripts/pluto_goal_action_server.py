#!/usr/bin/env python3

import sys

arguments = sys.argv
sys.argv = sys.argv[:1]

import rospy
import actionlib
from  milestone1.msg import PlutoGoalAction, PlutoGoalResult, PlutoGoalFeedback, PlutoGoalGoal
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from local_planner import execute_planning
from control import NavigationSystem

class PlutoGoalActionServer:
    def __init__(self, sim_obs):
        self.server = actionlib.SimpleActionServer("pluto_goal", PlutoGoalAction, self.execute_cb, False)
        self.server.start()
        self.robot_pose = None
        self.nav_system = NavigationSystem()
        self.sim_obs = sim_obs
        self.path_pub = rospy.Publisher('/planned_path', PoseStamped, queue_size=10)
        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)
        

    def robotPose_callback(self, msg):
        self.robot_pose = msg
    
    
    def execute_cb(self, goal:PlutoGoalGoal):
        feedback = PlutoGoalFeedback()
        result = PlutoGoalResult()


        rospy.loginfo("[PlutoGoal Server] Navigating to recived Goal")

        feedback.status = "Planning path..."
        self.server.publish_feedback(feedback)

        # here we can also processed the goal whether goal is valid or not, and accordingly cancel the goal 
        current_position = (self.robot_pose.pose.position.x, 
            self.robot_pose.pose.position.y)
        # rospy.sleep(3)   # if planning takes time
        try:
            self.nav_system.targetid = 0
            self.nav_system.fp = True

            self.nav_system.current_path, path_success, _, _ = execute_planning(current_position,(goal.goal.position.x, goal.goal.position.y), add_sim_obstacles=self.sim_obs)
            
            if not path_success:
                result.success = False
                result.message = "Planning failed for goal"
                self.server.set_aborted(result)
                return
            else:
                # Publish the path
                path_msg = Path()
                path_msg.header.stamp = rospy.Time.now()
                path_msg.header.frame_id = "map"
                
                for point in self.nav_system.current_path:
                    pose = PoseStamped()
                    pose.pose.position.x = point[0]
                    pose.pose.position.y = point[1]
                    path_msg.poses.append(pose)
                
                self.path_pub.publish(path_msg)

        except Exception as e:
            result.success = False
            result.message = f"Planning failed with error: {str(e)}"
            self.server.set_aborted(result)
            return  # Only return here if there's an exception

        feedback.status = "Following path..."
        self.server.publish_feedback(feedback)

        try:
            dist = ((self.robot_pose.pose.position.x - goal.goal.position.x )**2 + (self.robot_pose.pose.position.y - goal.goal.position.y )**2)**0.5
            while dist > 0.6:    
                self.nav_system.current_pose = self.robot_pose
                self.nav_system.run_control()  ## it can also return something whether we are heading correctly towards goal or should be abort current goal
                dist = ((self.robot_pose.pose.position.x - goal.goal.position.x )**2 + (self.robot_pose.pose.position.y - goal.goal.position.y )**2)**0.5
        except Exception as e:
            result.success = False
            result.message = f"Control failed at goal: {str(e)}"
            self.server.set_aborted(result)
            return

            # feedback.current_goal_index = i
            # self.server.publish_feedback(feedback)

        result.success = True
        result.message = "Visited the goal"
        self.server.set_succeeded(result)

if __name__ == "__main__":
    rospy.init_node('pluto_goal_server')
    
    sim_obs = False
    if '--sim_obs' in arguments:
        sim_obs = True
        rospy.loginfo("Simulation obstacles is enabled.")
    else:
        rospy.loginfo("Simulation obstacles is disabled.")

    
    PlutoGoalActionServer(sim_obs)
    rospy.spin()
