import rospy
import random
import math
import numpy as np
import utm
from sensor_msgs.msg import NavSatFix
import actionlib
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

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

# Robot and Sensor Params
robot_frame_id = ""
max_nodes = 1000
extension_range = 1.0
radius = 0.105

tf_buffer = None
listener = None
planner_as = None

sensor = None
l = 0.25

tree_pub = None
best_branch_pub = None
path_pub = None

global_obstacles = None
map_param = (0, 0, 10, 10)  # TO BE UPDATED


# For Visualization

obstacle_pub = None
tree_pub = None
path_pub = None

def visualize_obstacles(obstacles):
    marker_array = MarkerArray()
    
    for i, (obs_x, obs_y, obs_r) in enumerate(obstacles):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = obs_x
        marker.pose.position.y = obs_y
        marker.pose.position.z = 0.0  # Set to 0 for 2D
        marker.scale.x = obs_r * 2  # Diameter for cylinder
        marker.scale.y = obs_r * 2  # Diameter for cylinder
        marker.scale.z = 0.1  # Small height
        marker.color.a = 0.6  # Transparency
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue
        
        marker_array.markers.append(marker)
    
    obstacle_pub.publish(marker_array)

def visualize_tree(tree, nodes):
    marker_array = MarkerArray()
    
    for i, node in enumerate(nodes):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = node[0]
        marker.pose.position.y = node[1]
        marker.pose.position.z = 0.0  # Set to 0 for 2D
        marker.scale.x = 0.1  # Sphere size
        marker.scale.y = 0.1  # Sphere size
        marker.scale.z = 0.1  # Sphere size
        marker.color.a = 1.0  # Fully opaque
        marker.color.g = 1.0  # Green
        
        marker_array.markers.append(marker)
    
    tree_pub.publish(marker_array)

def visualize_path(path):
    path_marker = Marker()
    path_marker.header.frame_id = "map"
    path_marker.header.stamp = rospy.Time.now()
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.scale.x = 0.05  # Line thickness
    path_marker.color.a = 1.0  # Fully opaque
    path_marker.color.r = 0.0  # Red
    path_marker.color.g = 1.0  # Green
    path_marker.color.b = 0.0  # Blue
    
    for node in path:
        point = Point()
        point.x = node[0]
        point.y = node[1]
        point.z = 0.0  # Set to 0 for 2D
        path_marker.points.append(point)
    
    path_pub.publish(path_marker)
    
#----------------------

# Load Obstacles for Simulation
def load_obstacles_from_file(self, file_path):
        """
        Load obstacles from a text file. Each line in the file contains:
        x, y, radius for an obstacle.
        """
        obstacles = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        x, y, radius = float(parts[0]), float(parts[1]), float(parts[2])
                        obstacles.append((x, y, radius))
        except FileNotFoundError:
            rospy.logerr(f"Obstacle file not found: {file_path}")
        except Exception as e:
            rospy.logerr(f"Error loading obstacles from file: {e}")
        
        return obstacles

# Euclidean distance function
def euclidean_dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# TO BE REPLACED WITH RAJESH's function
# Check if a node is valid
def is_node_valid(node, map_param, obstacles=global_obstacles):
    """
    Check if a node is valid (within map boundaries and not colliding with obstacles).
    
    Args:
        node (tuple): The node (x, y, theta) to check.
        map_param (tuple): The map boundaries (xmin, ymin, xmax, ymax).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.
    
    Returns:
        bool: True if the node is valid, False otherwise.
    """
    xmin, ymin, xmax, ymax = map_param
    x, y, theta = node

    # Check for outside boundaries
    if x < xmin or x > xmax or y < ymin or y > ymax:
        return False

    # If obstacles are provided, check for collision
    if obstacles is not None:
        for obs in obstacles:
            obs_x, obs_y, obs_r = obs  # Obstacle is a circle (x, y, radius)
            dist = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
            if dist <= obs_r:
                return False

    return True

# Generate neighbors for the Dubins car
def generate_neighbors(node, map_param, step_size=0.5, num_samples=5, obstacles=global_obstacles):
    neighbors = []
    x, y, theta = node

    for _ in range(num_samples):  # Generate multiple candidate points
        angle = random.uniform(0, 2 * math.pi)  # Random direction
        xn = x + step_size * math.cos(angle)
        yn = y + step_size * math.sin(angle)
        thetan = random.uniform(-math.pi, math.pi)  # Random heading

        new_node = (xn, yn, thetan)

        if is_node_valid(new_node, map_param, obstacles=obstacles):
            neighbors.append(new_node)  # No steering angle needed

    return neighbors



def rrt_planner(start, goal, map_param, step_size=0.5, goal_threshold=0.2, num_samples=5, max_iter=1000000, prob_goal_bias=0.05, feedback_cb=None, obstacles=global_obstacles):
    global global_obstacles
    if global_obstacles != obstacles:
        global_obstacles = obstacles
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

        x_nearest = min(nodes, key=lambda node: euclidean_dist(node[:2], x_rand[:2]))
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
        neighbors = generate_neighbors(x_nearest, map_param, step_size=step_size, num_samples=num_samples, obstacles=obstacles)
        # print(neighbors)


        for x_new in neighbors:
            # Add new node to the tree if valid
            if is_node_valid(x_new, map_param, obstacles=obstacles):
                tree[x_new] = (x_nearest)
                nodes.append(x_new)

                # Publish feedback
                if feedback_cb:
                    path = reconstruct_path(tree, x_new)
                    feedback_cb(path, euclidean_dist(x_new[:2], goal[:2]))

                # Step 5: Goal proximity check
                if euclidean_dist(x_new[:2], goal[:2]) < goal_threshold:
                    #draw_map(car, nodes, start, goal, x_new)
                    return reconstruct_path(tree, x_new), tree, nodes

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

        robot_x = utm_x + geo_offset[0]
        robot_y = utm_y + geo_offset[1]

        rospy.loginfo("Updated robot position: (%.2f, %.2f)", robot_x, robot_y)
    
    except Exception as e:
        rospy.logerr(f"Error in gnss_feedback: {e}")


# Path Smoothing

def is_line_of_sight(a, b, map_param, obstacles=global_obstacles):
    """
    Checks if the line segment connecting points a and b is free of obstacles.
    
    Args:
        a (tuple): The start point (x, y, theta).
        b (tuple): The end point (x, y, theta).
        map_param (tuple): The map boundaries (xmin, ymin, xmax, ymax).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.
    
    Returns:
        bool: True if there is a clear line of sight, False otherwise.
    """
    x1, y1, _ = a
    x2, y2, _ = b
    xmin, ymin, xmax, ymax = map_param

    # Check if the line is within map boundaries
    num_points = 50
    for i in range(num_points + 1):
        t = i / num_points
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False
        # If obstacles are provided, check for collisions
        if obstacles is not None:
            for obs in obstacles:
                obs_x, obs_y, obs_r = obs
                dist = math.sqrt((x - obs_x) ** 2 + (y - obs_y) ** 2)
                if dist <= obs_r:
                    return False
    return True


def smooth_path(path, map_param, obstacles=global_obstacles):
    """
    Smooth the given path by removing unnecessary nodes, using line of sight checks.
    
    Args:
        path (list): List of waypoints in the path.
        map_param (tuple): The map boundaries (xmin, ymin, xmax, ymax).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.
    
    Returns:
        list: The smoothed path.
    """
    smoothed_path = [path[0]]  # Start with the first waypoint
    i = 0

    while i < len(path) - 1:
        j = len(path) - 1  # Start checking from the end of the path
        while j > i + 1:
            if is_line_of_sight(path[i], path[j], map_param, obstacles):
                break
            j -= 1
        smoothed_path.append(path[j])
        i = j

    return smoothed_path



def execute_planner(goal):
    rospy.loginfo("Received goal: (%.2f, %.2f)", goal.goal_pos.x, goal.goal_pos.y)
    global robot_x, robot_y, obstacles

    if robot_x is None or robot_y is None:
        rospy.logwarn("Robot position is not available yet!")
        planner_as.set_aborted()
        return
    
    goal_pos = (goal.goal_pos.x, goal.goal_pos.y, 0.0)
    start = (robot_x, robot_y, 0.0)

    def feedback_cb(path, gain):
        feedback = pluto_planner.msg.RRTPlannerFeedback()
        feedback.gain = gain
        feedback.path = path_to_ros_path(path)
        planner_as.publish_feedback(feedback)

    path, _, _ = rrt_planner(start, goal_pos, map_param, feedback_cb=feedback_cb)
    if path:
        smoothed_path = smooth_path(path, map_param, obstacles=global_obstacles)
        result = pluto_planner.msg.RRTPlannerResult()
        result.gain = euclidean_dist(smoothed_path[-1][:2], goal_pos[:2])
        result.path = path_to_ros_path(smoothed_path)
        path_pub.publish(result.path)  # Publish the path
        planner_as.set_succeeded(result)
        rospy.loginfo("Path found, sending result.")
        
        # Visualize the obstacles, tree, and path in RViz
        visualize_obstacles(global_obstacles)
        visualize_tree(nodes, global_obstacles)
        visualize_path(smoothed_path)
    else:
        rospy.logwarn("No path found!")
        planner_as.set_aborted()


if __name__ == "__main__":
    rospy.init_node("pluto_planner")
    
    global_obstacles = load_obstacles_from_file("./obstacles.txt")


    # Subscribers
    rospy.Subscriber(gnss_topic, NavSatFix, gnss_feedback)
    
    # Publishers
    path_pub = rospy.Publisher("/local_planner/path", Path, queue_size=10)
    obstacle_pub = rospy.Publisher("/local_planner/visualized_obstacles", MarkerArray, queue_size=10)
    tree_pub = rospy.Publisher("/local_planner/visualized_tree", MarkerArray, queue_size=10)

    # Create TF buffer
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)

    # Action Servers
    planner_as = actionlib.SimpleActionServer("get_next_goal", pluto_planner.msg.RRTPlannerAction,
                                                 execute_cb=execute_planner, auto_start=False)
    planner_as.start()
    rospy.spin()
