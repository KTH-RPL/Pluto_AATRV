import rospy
import random
import math
import numpy as np
import utm
from sensor_msgs.msg import NavSatFix
import actionlib
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray
import pluto_planner.msg
from sensor import Sensor
from rrt import RRT
import tf2_ros
from math import pi, atan2
from tf.transformations import quaternion_from_euler
import pluto_planner.msg 

# Global Vars
gnss_topic = "reach/fix"
geo_offset = [-333520.64199354, -6582420.00142414, 0.0]
trusted_gnss = False
grid_map = None

tf_buffer = None
listener = None
planner_as = None

robot_frame_id = ""
max_nodes = 1000
extension_range = 1.0
radius = 0.105

sensor = None
l = 0.25

tree_pub = None
best_branch_pub = None

# Euclidean distance function
def euclidean_dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# TO BE REPLACED WITH RAJESH's function
# Check if a node is valid
def is_node_valid(node, map_param):
    xmin, ymin, xmax, ymax = map_param
    x, y, theta = node

    # Check for outside boundaries
    if x < xmin or x > xmax or y < ymin or y > ymax:
        return False

    # Check for collision with obstacles
    # obstacle_margin = 0.2
    # for obs in obstacles:
    #     obs_x, obs_y, obs_r = obs
    #     dist = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
    #     if dist <= obs_r + obstacle_margin:
    #         return False

    return True

# Generate neighbors for the Dubins car
def generate_neighbors(node, map_param, step_size=0.5, num_samples=5):
    neighbors = []
    x, y, theta = node

    for _ in range(num_samples):  # Generate multiple candidate points
        angle = random.uniform(0, 2 * math.pi)  # Random direction
        xn = x + step_size * math.cos(angle)
        yn = y + step_size * math.sin(angle)
        thetan = random.uniform(-math.pi, math.pi)  # Random heading

        new_node = (xn, yn, thetan)

        if is_node_valid(new_node, map_param):
            neighbors.append(new_node)  # No steering angle needed

    return neighbors



def rrt_planner(start, goal, map_param, step_size=0.5, num_samples=5, max_iter=1000000, goal_threshold=0.5, prob_goal_bias=0.05, feedback_cb=None):
    tree = {start: None}
    nodes = [start]
    explored = []

    xlb, ylb, xub, yub = map_param

    iq = 0

    for _ in range(max_iter):
        # 1. Select node to explore
        ## Sample random point as direction to grow
        if random.random() < prob_goal_bias:
            x_rand = goal
        else:
            x_rand = (
                random.uniform(xlb, xub),
                random.uniform(ylb, yub),
                random.uniform(-math.pi, math.pi)
            )

        ## Find nearest node in the tree
        attempts = 0
        max_attempts = 10 

        while x_nearest in explored and attempts < max_attempts:
            x_rand = (
                random.uniform(xlb, xub),
                random.uniform(ylb, yub),
                random.uniform(-math.pi, math.pi)
            )
            x_nearest = min(nodes, key=lambda node: euclidean_dist(node[:2], x_rand[:2]))
            attempts += 1

        if attempts == max_attempts:
            rospy.logwarn("Could not find a new node to explore after multiple attempts!")
            continue  # Skip this iteration

        # print(x_nearest)

        # Generate random neighbors based on x_nearest point
        neighbors = generate_neighbors(x_nearest, map_param, step_size=step_size, num_samples=num_samples)
        # print(neighbors)


        for x_new in neighbors:
            # Add new node to the tree if valid
            if is_node_valid(x_new, map_param):
                tree[x_new] = (x_nearest)
                nodes.append(x_new)

                # Publish feedback
                if feedback_cb:
                    path = reconstruct_path(tree, x_new)
                    feedback_cb(path, euclidean_dist(x_new[:2], goal[:2]))

                # Step 5: Goal proximity check
                if euclidean_dist(x_new[:2], goal[:2]) < goal_threshold:
                    #draw_map(car, nodes, start, goal, x_new)
                    return reconstruct_path(tree, x_new)

            # Plotting

            iq+=1

    return None

# Convert path to ROS Path message
def path_to_ros_path(path):
    ros_path = Path()
    ros_path.header.frame_id = "map"

    for node in path:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.pose.position.x = node[0]
        pose.pose.position.y = node[1]
        pose.pose.position.z = 0
        ros_path.poses.append(pose)

    return ros_path

# Reconstruct the path from the tree
def reconstruct_path(tree, current_node):
    total_path = [current_node]
    # print(tree)
    while current_node in tree and (tree[current_node] != None):
        current_node = tree[current_node]
        total_path.append(current_node)

    total_path.reverse()
    return total_path


# Update Robot Loc based on GNSS data
def gnss_feedback(msg):
    try:
        global robot_x, robot_y, geo_offset, trusted_gnss

        if msg.position_covariance_type == 0:
            rospy.logwarn("GNSS covariance unknown, data might be unreliable")
            trusted_gnss = False
        else:
            trusted_gnss = True

        covariance_matrix = np.array(msg.position_covariance).reshape((3, 3))
        cov_x, cov_y = covariance_matrix[0, 0], covariance_matrix[1, 1]
        if cov_x > 5 or cov_y > 5: 
            rospy.logwarn("High GNSS covariance")
            trusted_gnss = False
        else:
            trusted_gnss = True

        lat, lon = msg.latitude, msg.longitude
        utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)

        robot_x = utm_x - geo_offset[0]
        robot_y = utm_y - geo_offset[1]

        rospy.loginfo("Updated robot position: (%.2f, %.2f)", robot_x, robot_y)
    
    except Exception as e:
        rospy.logerr(f"Error in gnss_feedback: {e}")


path_pub = rospy.Publisher("rrt_path", Path, queue_size=10)

def execute_planner(goal):
    rospy.loginfo("Received goal: (%.2f, %.2f)", goal.goal_pos.x, goal.goal_pos.y)
    global robot_x, robot_y

    if robot_x is None or robot_y is None:
        rospy.logwarn("Robot position is not available yet!")
        planner_as.set_aborted()
        return
    
    map_param = (0,0,10,10) #NEED UPDATE based on bounding box
    goal_pos = (goal.goal_pos.x, goal.goal_pos.y, 0.0)
    start = (robot_x, robot_y, 0.0)

    def feedback_cb(path, gain):
        feedback = pluto_planner.msg.RRTPlannerFeedback()
        feedback.gain = gain
        feedback.path = path_to_ros_path(path)
        planner_as.publish_feedback(feedback)

    path = rrt_planner(start, goal_pos, map_param, feedback_cb=feedback_cb)
    if path:
        result = pluto_planner.msg.RRTPlannerResult()
        result.gain = euclidean_dist(path[-1][:2], goal_pos[:2])
        result.path = path_to_ros_path(path)
        path_pub.publish(result.path)  # Publish the path
        planner_as.set_succeeded(result)
        rospy.loginfo("Path found, sending result.")
    else:
        rospy.logwarn("No path found!")
        planner_as.set_aborted()


if __name__ == "__main__":
    rospy.init_node("pluto_planner")

    # Subscribers
    rospy.Subscriber(gnss_topic, NavSatFix, gnss_feedback)

    # Create TF buffer
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    # Action Servers
    planner_as = actionlib.SimpleActionServer("get_next_goal", pluto_planner.msg.RRTPlannerAction,
                                                 execute_cb=execute_planner, auto_start=False)
    planner_as.start()
    rospy.spin()
