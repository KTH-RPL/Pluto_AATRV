import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
from nav_msgs.msg import Path
from utils.global_planner import execute_global_planning
from utils.local_planner import execute_planning as execute_local_planning
import gmap_utility
import os
import sys

# Define a variable for the start position
# In a real-world scenario, this would come from a localization node
CURRENT_ROBOT_POSE = PoseStamped()
POSE_TOPIC = '/robot_pose'

def pose_callback(data):
    global CURRENT_ROBOT_POSE
    CURRENT_ROBOT_POSE = data

class GlobalPlannerServer:
    def __init__(self):
        # Initialize a publisher for the global plan
        self.global_plan_pub = rospy.Publisher('/global_plan', Path, queue_size=10)
        # Initialize a subscriber for goal poses
        self.goal_sub = rospy.Subscriber('/global_goals_array', PoseArray, self.goal_callback)
        # Subscriber for the robot's current pose
        self.pose_sub = rospy.Subscriber(POSE_TOPIC, PoseStamped, pose_callback)
        rospy.loginfo("Global Planner Server is ready.")

    def goal_callback(self, data):
        rospy.loginfo("Received a new goal array.")
        
        # Take the last pose in the array as the final goal
        if not data.poses:
            rospy.logwarn("Received an empty goal array. Not planning.")
            return

        # Get the start and end positions
        start = (CURRENT_ROBOT_POSE.pose.position.x, CURRENT_ROBOT_POSE.pose.position.y, CURRENT_ROBOT_POSE.pose.orientation.z)
        final_goal_pose = data.poses[-1]
        goal = (final_goal_pose.position.x, final_goal_pose.position.y, 0.0) # Assuming theta is not in PoseArray for simplicity

        # Run the global planner to get the path
        # Using the execute_global_planning function
        try:
            rospy.loginfo("Running RRT global planning...")
            path_points, tree, nodes, raw_path = execute_global_planning(start, goal, sim_plan=False)
            
            if not path_points:
                rospy.logerr("Global planner failed to find a path.")
                return

            rospy.loginfo("Global plan generated successfully. Publishing...")

            # Convert the list of waypoints to a nav_msgs/Path message
            path_msg = self.points_to_path_msg(path_points)
            self.global_plan_pub.publish(path_msg)
            rospy.loginfo("Global plan published.")
            
        except Exception as e:
            rospy.logerr(f"An error occurred during global planning: {e}")

    def points_to_path_msg(self, points):
        path_msg = Path()
        path_msg.header.frame_id = "map"  # Assuming the map frame is "map"
        path_msg.header.stamp = rospy.Time.now()

        for p in points:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = p[0]
            pose_stamped.pose.position.y = p[1]
            pose_stamped.pose.orientation.w = 1.0  # Placeholder, assuming 2D
            path_msg.poses.append(pose_stamped)

        return path_msg

class LocalPlannerNode:
    def __init__(self):
        # Initialize a publisher for the local plan
        self.local_plan_pub = rospy.Publisher('/goals_array', PoseArray, queue_size=10)
        # Initialize a subscriber for the global plan
        self.global_plan_sub = rospy.Subscriber('/global_plan', Path, self.global_plan_callback)
        rospy.loginfo("Local Planner Node is ready.")

    def global_plan_callback(self, path_msg):
        rospy.loginfo(f"Received global path with {len(path_msg.poses)} points.")

        # Local planning logic: convert path_msg to a list of tuples
        path_list = [(p.pose.position.x, p.pose.position.y) for p in path_msg.poses]
        
        # Take the next few waypoints as the local path
        local_path_size = 5  # Example: Look at the next 5 waypoints
        local_path_points_raw = path_list[:local_path_size]

        if not local_path_points_raw:
            rospy.logwarn("Smoothed path is too short or empty. Not publishing local path.")
            return

        # Use the local planner function to smooth the path
        # In a real scenario, this would handle dynamic obstacles
        final_path, _, _, _ = execute_local_planning(local_path_points_raw[0], local_path_points_raw[-1])

        # Convert the points to a PoseArray message
        local_path_array = PoseArray()
        local_path_array.header = path_msg.header
        
        for p in final_path:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = p[0]
            pose_stamped.pose.position.y = p[1]
            # Use the calculated heading for the orientation
            pose_stamped.pose.orientation.z = p[2]
            local_path_array.poses.append(pose_stamped.pose)

        rospy.loginfo(f"Publishing local path with {len(local_path_array.poses)} points.")
        self.local_plan_pub.publish(local_path_array)

if __name__ == '__main__':
    try:
        # Check if gmap_utility exists before proceeding
        if 'gmap_utility' not in sys.modules:
            rospy.logerr("Error: The 'gmap_utility' module was not found. Please ensure it is in the same directory.")
            sys.exit(1)

        # Initialize the ROS node
        rospy.init_node('navigation_system', anonymous=True)

        # Create the global planner server
        global_server = GlobalPlannerServer()

        # Create the local planner node
        local_planner = LocalPlannerNode()

        # Spin ROS to keep the node alive
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
