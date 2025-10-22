#!/usr/bin/env python3

import sys
import py_trees as pt
import py_trees_ros as ptr
import rospy
import actionlib
from rsequence import RSequence
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32, Bool

from robot_controller.msg import PlanGlobalPathAction, PlanGlobalPathGoal
from robot_controller.srv import RunControl
from sensor_msgs.msg import PointCloud2
# import gmap_utility
import py_trees.display as display
import matplotlib.pyplot as plt
import json
import os
import math


class M2BehaviourTree(ptr.trees.BehaviourTree):
    def __init__(self):
        rospy.loginfo("Initialising behaviour tree")

        c_goal = goal_robot_condition()

        b0 = pt.composites.Selector(
            name="Global Planner Fallback",
            children=[global_path_exist(c_goal), global_path_client(c_goal)]
        )

        s1 = RSequence(
            name="Planner and Control Sequence", 
            children=[b0, control_planner(c_goal)]
        )

        b1 = pt.composites.Selector(
            name="Goal Reached Fallback",
            children=[goal_reached(c_goal), s1]
        )

        tree = RSequence(name="Main sequence", children=[check_ouster_points(c_goal,timeout=1.0 ), check_localization(c_goal), b1])
        super(M2BehaviourTree, self).__init__(tree)

        rospy.sleep(5)
        self.setup(timeout=10000)
        while not rospy.is_shutdown():
            rospy.loginfo("Ticking the tree")
            self.tick_tock(0.1)  # 10Hz


class goal_robot_condition():
    def __init__(self):
        self.current_goal = None
        self.robot_pose = None
        self.goals = None  
        self.get_global_path = None
        self.has_published = False
        # Track goal index for multi-goal execution
        self.goal_index = 0
        self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

        self.last_ouster_time = rospy.Time.now()
        self.is_converged = None

        rospy.Subscriber("/is_converged", Bool, self.is_converged_cb)
        rospy.Subscriber("/ouster/points", PointCloud2, self.ouster_cb)
        rospy.Subscriber("/ndt_pose", PoseStamped, self.ndt_pose_cb)
        rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_cb)
        rospy.Subscriber("/waypoints", Path, self.waypoints_callback)
        # rospy.Subscriber("/transform_probability", Float32, self.transform_probability_cb)

        # Save pose on shutdown
        rospy.on_shutdown(self.save_last_pose_on_shutdown)

        # for localization condition 
        # self.transform_probability = None
        self.pose_history_file = os.path.expanduser("~/robot_pose_history.json")
        os.makedirs(os.path.dirname(self.pose_history_file), exist_ok=True)

        self.valid_poses = []   # Store history of valid poses with scores
        self.last_good_pose = None  # To store the last pose with good score
        # self.localization_recovery_start_time = None  # Track recovery start time
        # self.max_recovery_time = 5.0  # Maximum 30 seconds for recovery

    # def transform_probability_cb(self, msg):
    #     self.transform_probability = msg.data   

    def is_converged_cb(self,msg):
        self.is_converged = msg.data

    def ouster_cb(self, msg):
        self.last_ouster_time = rospy.Time.now()

    def ndt_pose_cb(self, msg):
        # self.robot_pose = msg

        # if self.transform_probability is not None:
        pose_entry = {
            "timestamp": rospy.Time.now().to_sec(),
            # "fitness_score": self.transform_probability,
            "pose": {
                "position": {
                    "x": msg.pose.position.x,
                    "y": msg.pose.position.y,
                    "z": msg.pose.position.z
                },
                "orientation": {
                    "x": msg.pose.orientation.x,
                    "y": msg.pose.orientation.y,
                    "z": msg.pose.orientation.z,
                    "w": msg.pose.orientation.w
                }
            }
        }
        
        self.valid_poses.append(pose_entry)
        self.last_good_pose = pose_entry    
            # Update last good pose if score is good
            # if self.transform_probability <= 0.5:
            #     self.last_good_pose = pose_entry
            #     # Reset recovery timer when we get good localization
            #     self.localization_recovery_start_time = None

        # Keep only last 1000 poses
        if len(self.valid_poses) > 1000:
            self.valid_poses = self.valid_poses[-1000:]
            
        self.save_pose_history()
    
    def robot_pose_cb(self, msg):
        self.robot_pose = msg
    
    def save_pose_history(self):
        try:
            with open(self.pose_history_file, 'w') as f:
                json.dump({"valid_poses": self.valid_poses}, f, indent=4)
        except Exception as e:
            rospy.logwarn(f"Failed to save pose history: {e}")

    def get_last_good_pose(self):
        """Get the most recent pose that had a good score (<= 4.2)."""
        if self.last_good_pose:
            return self.last_good_pose
        
        for pose in reversed(self.valid_poses):
            if pose["fitness_score"] <= 0.5:
                self.last_good_pose = pose
                return pose
        return None
    
    # def quaternion_to_yaw(self,orientation):
    #     """Convert quaternion to yaw angle (in radians)"""
    #     x = orientation.x
    #     y = orientation.y
    #     z = orientation.z
    #     w = orientation.w
        
    #     # Convert quaternion to yaw
    #     yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    #     return yaw
    
    # def start_localization_recovery(self):
    #     """Start tracking recovery time"""
    #     if self.localization_recovery_start_time is None:
    #         self.localization_recovery_start_time = rospy.Time.now().to_sec()
    #         rospy.loginfo("[Localization] Starting recovery timer")

    # def is_recovery_timeout(self):
    #     """Check if recovery timeout has been reached"""
    #     if self.localization_recovery_start_time is None:
    #         return False
    #     elapsed = rospy.Time.now().to_sec() - self.localization_recovery_start_time
    #     return elapsed > self.max_recovery_time
    
    def save_last_pose_on_shutdown(self):
        rospy.loginfo("Saving last robot pose on shutdown.")
        self.save_pose_history()

    def waypoints_callback(self, msg):
        if not self.goals:   
            pose_array = PoseArray()
            pose_array.header.frame_id = msg.header.frame_id
            pose_array.header.stamp = rospy.Time.now()

            for ps in msg.poses:
                pose = Pose()
                pose.position = ps.pose.position
                pose.orientation = ps.pose.orientation
                pose_array.poses.append(pose)
            self.goals = pose_array
            rospy.loginfo(f"[Behaviour Tree] Received {len(self.goals.poses)} goals.")


class check_localization(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(check_localization, self).__init__("CheckLocalization")
        self.c_goal = c_goal
        self.initial_pose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=10)
    
    def update(self):
        if not self.c_goal.is_converged:
            last_valid_pos = self.c_goal.get_last_good_pose()
            if not self.c_goal.has_published and last_valid_pos:
                try:
                    pose_msg = PoseWithCovarianceStamped()
                    pose_msg.header.frame_id = "map"
                    pose_msg.header.stamp = rospy.Time.now()

                    pose_msg.pose.pose.position.x = last_valid_pos["pose"]["position"]["x"]
                    pose_msg.pose.pose.position.y = last_valid_pos["pose"]["position"]["y"]
                    pose_msg.pose.pose.position.z = last_valid_pos["pose"]["position"]["z"]

                    pose_msg.pose.pose.orientation.x = last_valid_pos["pose"]["orientation"]["x"]
                    pose_msg.pose.pose.orientation.y = last_valid_pos["pose"]["orientation"]["y"]
                    pose_msg.pose.pose.orientation.z = last_valid_pos["pose"]["orientation"]["z"]
                    pose_msg.pose.pose.orientation.w = last_valid_pos["pose"]["orientation"]["w"]

                    # Set covariance matrix as 0.25 * identity
                    pose_msg.pose.covariance = [0.25 if i % 7 == 0 else 0.0 for i in range(36)]  # faster without numpy

                    self.initial_pose_pub.publish(pose_msg)
                    self.c_goal.has_published = True
                    rospy.loginfo("[CheckLocalization] Published last valid pose to /initialpose for relocalization.")
                except Exception as e:
                    rospy.logwarn(f"[CheckLocalization] Failed to publish last valid pose: {e}")
            else:
                rospy.logwarn("[CheckLocalization] No last valid pose found in history to publish.")

            return pt.common.Status.FAILURE

        else:
            # Everything OK
            self.c_goal.has_published = False
            rospy.loginfo("[CheckLocalization] Localization is converging")
            return pt.common.Status.SUCCESS


class check_ouster_points(pt.behaviour.Behaviour):
    def __init__(self,c_goal,  timeout = 2.0):
        super(check_ouster_points, self).__init__("CheckOusterPoints")
        self.c_goal = c_goal
        self.timeout = timeout

    def update(self):
        time_since_last = (rospy.Time.now() - self.c_goal.last_ouster_time).to_sec()
        if time_since_last > self.timeout:
            rospy.logwarn(f"[CheckOusterPoints] No /ouster/points message for {time_since_last:.1f}s — returning FAILURE.")
            return pt.common.Status.FAILURE
        else:
            rospy.loginfo("[CheckOusterPoints] /ouster/points messages active.")
            return pt.common.Status.SUCCESS


class goal_reached(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(goal_reached, self).__init__("Goal_reached")
        self.c_goal = c_goal
        self.finished = False

    def update(self):
        if self.finished:
            rospy.loginfo("[goal_reached] Final goal reached.")
            return pt.common.Status.SUCCESS
        
        if not self.c_goal.goals or not self.c_goal.robot_pose:
            rospy.logerr(f"Either goal is empty or not getting robot_pose:{self.c_goal.goals, self.c_goal.robot_pose}")
            return pt.common.Status.RUNNING

        final_goal = self.c_goal.goals.poses[-1].position
        current_pose = self.c_goal.robot_pose.pose.position
        distance = ((current_pose.x - final_goal.x) ** 2 + 
                    (current_pose.y - final_goal.y) ** 2) ** 0.5

        if distance < 0.6:
            rospy.loginfo("[goal_reached] Final goal reached.")
            self.finished = True
            return pt.common.Status.SUCCESS
        rospy.loginfo("[goal_reached] Final goal not reached yet.")
        return pt.common.Status.FAILURE


class global_path_exist(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(global_path_exist, self).__init__("global_path_exists")
        self.c_goal = c_goal

    def update(self):
        if self.c_goal.get_global_path:
            rospy.loginfo("Global path already exists ")
            return pt.common.Status.SUCCESS
        else:
            rospy.loginfo("Global path does not exist -> going to global path client")
            return pt.common.Status.FAILURE


class global_path_client(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(global_path_client, self).__init__("SendGoals")
        self.client = actionlib.SimpleActionClient("plan_global_path", PlanGlobalPathAction)
        self.sent = False        
        self.c_goal = c_goal
        self.path_pub_ = rospy.Publisher("/global_path", Path, queue_size=1, latch=True)

    def initialise(self):
        self.sent = False
        self.c_goal.get_global_path = None
        rospy.loginfo("[global_path_client] Waiting for the 'plan_global_path' action server...")
        self.client.wait_for_server()
        rospy.loginfo("[global_path_client] Action server found")
        
    def feedback_callback(self, feedback):
        try:
            rospy.loginfo(
                "[global_path_client] Received feedback: Path segment with %d poses has been planned.",
                len(feedback.current_segment.poses)
            )
            # try: 
            #     plt.close('all')
            # except:
            #     pass
            # gmap_utility.polygon_map.visualize(feedback.current_segment.poses)

            # !!TO EVALUATE!!
            # Publish the intermediate path to the 'global_path' topic
            ## This would reduce init time of the robot (not waiting for the whole global plan)
            ## But need to do some changes in final_control_algo.cpp to able receiving segments
            # if feedback.global_plan.poses:
            #     self.path_pub_.publish(feedback.global_plan)
            #     rospy.loginfo("Published the intermediate path to 'global_path'.")
            # else:
            #     rospy.logwarn("Received an empty intermediate path. Not publishing.")

        except Exception as e:
            rospy.logwarn("Feedback callback failed: %s", str(e))

    def done_callback(self, status, result):
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[global_path_client] Action finished successfully!")
            rospy.loginfo("[global_path_client] Final global path contains %d poses.", len(result.global_plan.poses))

            # Publish the received path to the 'global_path' topic
            if result.global_plan.poses:
                # The 'result.global_plan' is already a nav_msgs/Path message, so we can publish it directly.
                self.path_pub_.publish(result.global_plan)
                rospy.loginfo(" [global_path_client] Published the final path to 'global_path'.")
            else:
                rospy.logwarn("[global_path_client] Received an empty final path. Not publishing.")

            # try:
            #     plt.close('all')
            #     gmap_utility.polygon_map.visualize(result.global_plan.poses)
            # except Exception as e:
            #     rospy.logwarn("Visualization failed in done_callback: %s", str(e))
        elif status == actionlib.GoalStatus.PREEMPTED:
            rospy.logwarn("[global_path_client] Action was preempted by a new goal.")
        else:
            rospy.logerr("[global_path_client] Action failed with status code: %s", actionlib.GoalStatus.to_string(status))

    def update(self):
        if self.c_goal.get_global_path:
            rospy.loginfo("[global_path_client] Got the global Path")
            return pt.common.Status.SUCCESS

        if not self.sent:
            pluto_goal = PlanGlobalPathGoal()
            pluto_goal.waypoints = self.c_goal.goals
            self.client.send_goal(pluto_goal, 
                                  done_cb=self.done_callback, 
                                  feedback_cb=self.feedback_callback)
            self.sent = True

            goal_pose = PoseStamped()
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose = self.c_goal.goals.poses[self.c_goal.goal_index]
            self.c_goal.goal_pose_pub.publish(goal_pose)
            rospy.loginfo(f"[global_path_client] Sent goal {self.c_goal.goal_index + 1}/{len(self.c_goal.goals.poses)}")

        if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            result = self.client.get_result()
            if result:
                self.c_goal.get_global_path = result.global_plan.poses
            self.sent = False
            rospy.loginfo("[global_path_client] planner returns success")
            return pt.common.Status.SUCCESS

        elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
            rospy.logwarn("[global_path_client] Goal execution failed.")
            return pt.common.Status.FAILURE

        return pt.common.Status.RUNNING


class control_planner(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(control_planner, self).__init__("ControlPlanner")
        self.c_goal = c_goal
           
    def initialise(self):
        """Called when the behavior first becomes RUNNING"""
        try:
            rospy.loginfo("Waiting for RunControl Service")
            rospy.wait_for_service('run_control', timeout=5.0)
            self.run_control_srv = rospy.ServiceProxy('run_control', RunControl)
            
            rospy.loginfo("RunControl service connected successfully")
        except rospy.ROSException:
            rospy.logerr("RunControl service not available within timeout")

    def update(self):    
        try:
            resp = self.run_control_srv(False)  # False = not last goal
            rospy.loginfo("Received the response from service")
            
            if resp.status == 0:   # RUNNING
                rospy.loginfo("Controller service is running")
                return pt.common.Status.RUNNING
            elif resp.status == 1: # SUCCESS
                rospy.loginfo("Controller service returns success")
                return pt.common.Status.SUCCESS
            else:                  # FAILURE
                rospy.logwarn("Controller service returns Failure")
                return pt.common.Status.FAILURE
                
        except rospy.ServiceException as e:
            rospy.logerr(f"[ControlPlanner] Service call failed: {e}")
            # Mark client as not ready for re-initialization
            self.control_client_ready = False
            return pt.common.Status.FAILURE
        


###  Below commented code will also returned the robot back to its home position where it started

# class goal_robot_condition():
#     def __init__(self):
#         self.current_goal = None
#         self.robot_pose = None
#         self.goals = None  
#         self.get_global_path = None
#         self.has_published = False
        
#         # Mission state tracking
#         self.mission_phase = "GOAL"  # GOAL or HOME
#         self.start_pose = None  # Store starting position
#         self.original_goals = None  # Store original waypoints
#         self.original_global_path = None  # Store the original global path
        
#         # Track goal index for multi-goal execution
#         self.goal_index = 0
#         self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)

#         self.last_ouster_time = rospy.Time.now()
#         self.is_converged = None

#         rospy.Subscriber("/is_converged", Bool, self.is_converged_cb)
#         rospy.Subscriber("/ouster/points", PointCloud2, self.ouster_cb)
#         rospy.Subscriber("/ndt_pose", PoseStamped, self.ndt_pose_cb)
#         rospy.Subscriber("/robot_pose", PoseStamped, self.robot_pose_cb)
#         rospy.Subscriber("/waypoints", Path, self.waypoints_callback)

#         # Save pose on shutdown
#         rospy.on_shutdown(self.save_last_pose_on_shutdown)

#         self.pose_history_file = os.path.expanduser("~/robot_pose_history.json")
#         os.makedirs(os.path.dirname(self.pose_history_file), exist_ok=True)

#         self.valid_poses = []   # Store history of valid poses with scores
#         self.last_good_pose = None  # To store the last pose with good score

#     def ouster_cb(self, msg):
#         self.last_ouster_time = rospy.Time.now()
    
#     def is_converged_cb(self,msg):
#         self.is_converged = msg.data

#     def ndt_pose_cb(self, msg):
#         pose_entry = {
#             "timestamp": rospy.Time.now().to_sec(),
#             "pose": {
#                 "position": {
#                     "x": msg.pose.position.x,
#                     "y": msg.pose.position.y,
#                     "z": msg.pose.position.z
#                 },
#                 "orientation": {
#                     "x": msg.pose.orientation.x,
#                     "y": msg.pose.orientation.y,
#                     "z": msg.pose.orientation.z,
#                     "w": msg.pose.orientation.w
#                 }
#             }
#         }
        
#         self.valid_poses.append(pose_entry)
#         self.last_good_pose = pose_entry    

#         # Keep only last 1000 poses
#         if len(self.valid_poses) > 1000:
#             self.valid_poses = self.valid_poses[-1000:]
            
#         self.save_pose_history()
    
#     def robot_pose_cb(self, msg):
#         self.robot_pose = msg
#         self.last_pose_time = rospy.Time.now()
        
#         # Save start pose when we first get robot pose and don't have one yet
#         if self.start_pose is None and msg.pose.position.x != 0 and msg.pose.position.y != 0:
#             self.start_pose = msg
#             rospy.loginfo(f"[Mission] Start position saved: ({self.start_pose.pose.position.x:.2f}, {self.start_pose.pose.position.y:.2f})")
            
    
#     def save_pose_history(self):
#         try:
#             with open(self.pose_history_file, 'w') as f:
#                 json.dump({"valid_poses": self.valid_poses}, f, indent=4)
#         except Exception as e:
#             rospy.logwarn(f"Failed to save pose history: {e}")

#     def get_last_good_pose(self):
#         """Get the most recent pose."""
#         if self.last_good_pose:
#             return self.last_good_pose
        
#         for pose in reversed(self.valid_poses):
#             return pose
#         return None
    
#     def prepare_return_mission(self):
#         """Prepare return path by reversing the global path"""
#         if self.original_global_path is None or len(self.original_global_path) == 0:
#             rospy.logwarn("[Mission] No global path available to reverse!")
#             return False
            
#         # Reverse the global path for return trip
#         return_path = list(reversed(self.original_global_path))
#         self.get_global_path = return_path
#         self.mission_phase = "HOME"
#         self.goal_index = 0
        
#         rospy.loginfo(f"[Mission] Return path prepared by reversing global path. {len(return_path)} waypoints.")
#         return True
    
#     def save_global_path(self, path_poses):
#         """Store the original global path for later reversal"""
#         self.original_global_path = path_poses
#         rospy.loginfo(f"[Mission] Original global path saved with {len(path_poses)} poses")
    
#     def save_last_pose_on_shutdown(self):
#         rospy.loginfo("Saving last robot pose on shutdown.")
#         self.save_pose_history()

#     def waypoints_callback(self, msg):
#         if not self.goals:   
#             pose_array = PoseArray()
#             pose_array.header.frame_id = msg.header.frame_id
#             pose_array.header.stamp = rospy.Time.now()

#             for ps in msg.poses:
#                 pose = Pose()
#                 pose.position = ps.pose.position
#                 pose.orientation = ps.pose.orientation
#                 pose_array.poses.append(pose)
            
#             self.goals = pose_array
#             self.original_goals = pose_array  # Store original for potential reset
#             rospy.loginfo(f"[Behaviour Tree] Received {len(self.goals.poses)} goals for GOAL mission.")


# class global_path_client(pt.behaviour.Behaviour):
#     def __init__(self, c_goal):
#         super(global_path_client, self).__init__("SendGoals")
#         self.client = actionlib.SimpleActionClient("plan_global_path", PlanGlobalPathAction)
#         self.sent = False        
#         self.c_goal = c_goal
#         self.path_pub_ = rospy.Publisher("/global_path", Path, queue_size=1, latch=True)

#     def initialise(self):
#         self.sent = False
#         self.c_goal.get_global_path = None
#         rospy.loginfo("[global_path_client] Waiting for the 'plan_global_path' action server...")
#         self.client.wait_for_server()
#         rospy.loginfo("[global_path_client] Action server found")
        
#     def feedback_callback(self, feedback):
#         try:
#             rospy.loginfo(
#                 "Received feedback: Path segment with %d poses has been planned.",
#                 len(feedback.current_segment.poses)
#             )
#         except Exception as e:
#             rospy.logwarn(" [global_path_client] Feedback callback failed: %s", str(e))

#     def done_callback(self, status, result):
#         if status == actionlib.GoalStatus.SUCCEEDED:
#             rospy.loginfo(f"[global_path_client] [{self.c_goal.mission_phase}] Action finished successfully!")
#             rospy.loginfo("[global_path_client] Final global path contains %d poses.", len(result.global_plan.poses))

#             # Store the original global path
#             if result.global_plan.poses:
#                 self.c_goal.save_global_path(result.global_plan.poses)
#                 self.c_goal.get_global_path = result.global_plan.poses
#                 self.path_pub_.publish(result.global_plan)
#                 rospy.loginfo(f"[global_path_client] Published the {self.c_goal.mission_phase} path to 'global_path'.")
#             else:
#                 rospy.logwarn("[global_path_client] Received an empty final path. Not publishing.")

#         elif status == actionlib.GoalStatus.PREEMPTED:
#             rospy.logwarn("[global_path_client] Action was preempted by a new goal.")
#         else:
#             rospy.logerr("[global_path_client] Action failed with status code: %s", actionlib.GoalStatus.to_string(status))

#     def update(self):
#         # For HOME phase, we don't need to call the planner - just use reversed path
#         if self.c_goal.mission_phase == "HOME" and self.c_goal.get_global_path:
#             # Just publish the reversed path that's already stored
#             return_path = Path()
#             return_path.header.frame_id = "map"
#             return_path.header.stamp = rospy.Time.now()
#             return_path.poses = [pose for pose in self.c_goal.get_global_path]  # Already reversed
            
#             self.path_pub_.publish(return_path)
#             rospy.loginfo(f"[global_path_client] [HOME] Published reversed path with {len(return_path.poses)} poses")
#             return pt.common.Status.SUCCESS

#         # For GOAL phase, use the normal planner
#         if self.c_goal.get_global_path:
#             return pt.common.Status.SUCCESS

#         if not self.sent:
#             pluto_goal = PlanGlobalPathGoal()
#             pluto_goal.waypoints = self.c_goal.goals
#             self.client.send_goal(pluto_goal, 
#                                   done_cb=self.done_callback, 
#                                   feedback_cb=self.feedback_callback)
#             self.sent = True

#             goal_pose = PoseStamped()
#             goal_pose.header.stamp = rospy.Time.now()
#             goal_pose.pose = self.c_goal.goals.poses[self.c_goal.goal_index]
#             self.c_goal.goal_pose_pub.publish(goal_pose)
            
#             rospy.loginfo(f"[GOAL] Sent GOAL goal: ({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})")

#         if self.client.get_state() == actionlib.GoalStatus.SUCCEEDED:
#             result = self.client.get_result()
#             if result:
#                 self.c_goal.get_global_path = result.global_plan.poses
#             self.sent = False
#             return pt.common.Status.SUCCESS

#         elif self.client.get_state() in [actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.REJECTED]:
#             rospy.logwarn(f"[GOAL] Goal execution failed.")
#             return pt.common.Status.FAILURE

#         return pt.common.Status.RUNNING


# class goal_reached(pt.behaviour.Behaviour):
#     def __init__(self, c_goal):
#         super(goal_reached, self).__init__("Goal_reached")
#         self.c_goal = c_goal
#         self.finished = False

#     def update(self):
#         if self.finished:
#             rospy.loginfo("[goal_reached] Complete mission finished - both GOAL and return.")
#             return pt.common.Status.SUCCESS
        
#         if not self.c_goal.goals or not self.c_goal.robot_pose:
#             rospy.logerr(f"Either goal is empty or not getting robot_pose:{self.c_goal.goals, self.c_goal.robot_pose}")
#             return pt.common.Status.RUNNING

#         # Get current target based on mission phase
#         if self.c_goal.mission_phase == "GOAL":
#             target_goal = self.c_goal.goals.poses[-1]  # Final goal of GOAL
#             target_description = "GOAL"
#         else:  # HOME phase
#             if self.c_goal.start_pose:
#                 target_goal = self.c_goal.start_pose.pose  # Start position
#             else:
#                 rospy.logerr("[goal_reached] No start pose available for return!")
#                 return pt.common.Status.FAILURE
#             target_description = "return"

#         current_pose = self.c_goal.robot_pose.pose.position
#         distance = ((current_pose.x - target_goal.position.x) ** 2 + 
#                     (current_pose.y - target_goal.position.y) ** 2) ** 0.5

#         if distance < 0.6:
#             if self.c_goal.mission_phase == "GOAL":
#                 rospy.loginfo("[goal_reached] GOAL goal reached! Preparing return mission...")
#                 # Switch to return mission using reversed path
#                 if self.c_goal.prepare_return_mission():
#                     return pt.common.Status.FAILURE  # Continue mission
#                 else:
#                     rospy.logerr("[goal_reached] Failed to prepare return mission!")
#                     return pt.common.Status.FAILURE
#             else:  # HOME phase
#                 rospy.loginfo("[goal_reached] HOME goal reached! Mission complete!")
#                 self.finished = True
#                 return pt.common.Status.SUCCESS
                
#         rospy.loginfo_throttle(5, f"[goal_reached] {self.c_goal.mission_phase} {target_description} goal not reached yet. Distance: {distance:.2f}m")
#         return pt.common.Status.FAILURE


if __name__ == "__main__":
    print(sys.executable)
    rospy.init_node("Multi_Goal_behaviour_tree_controller")
    tree = M2BehaviourTree()
