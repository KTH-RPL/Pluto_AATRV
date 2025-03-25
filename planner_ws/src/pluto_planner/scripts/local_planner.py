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

from scipy.interpolate import CubicSpline
from geometry_msgs.msg import Point

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

max_dist_per_point = 1.5

# For Visualization

obstacle_pub = None
tree_pub = None
path_pub = None

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points: a -> b -> c.
    
    Args:
        a, b, c (tuple): Points (x, y) representing path waypoints.

    Returns:
        float: Angle in radians between the segments a-b and b-c.
    """
    vec_ab = (b[0] - a[0], b[1] - a[1])
    vec_bc = (c[0] - b[0], c[1] - b[1])

    dot_product = vec_ab[0] * vec_bc[0] + vec_ab[1] * vec_bc[1]
    mag_ab = math.sqrt(vec_ab[0] ** 2 + vec_ab[1] ** 2)
    mag_bc = math.sqrt(vec_bc[0] ** 2 + vec_bc[1] ** 2)

    if mag_ab == 0 or mag_bc == 0:  # Avoid division by zero
        return 0.0

    # Compute angle between vectors in radians, clamping the value for acos
    cos_angle = dot_product / (mag_ab * mag_bc)
    
    # Clamp the value to avoid math domain errors due to floating-point precision
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Compute the angle
    angle = math.acos(cos_angle)
    return angle


import scipy.interpolate as si

def generate_spline_path(path, num_points=50):
    """
    Generates a smooth path using quadratic B-spline interpolation through three waypoints.
    
    Args:
        path (list): List of three waypoints [(x1, y1), (x2, y2), (x3, y3)].
        num_points (int): Total number of points to generate along the spline.
    
    Returns:
        list: Smoothed path as a list of (x, y) tuples.
    """
  
    # Extract x and y coordinates from path
    x = [node[0] for node in path]
    y = [node[1] for node in path]

    # Define a uniform parameterization (parameter t)
    t = np.linspace(0, 1, len(path))

    # Fit a quadratic B-spline curve (degree k=2)
    spl_x = si.BSpline(t, x, k=3)  # Quadratic B-spline for x coordinates
    spl_y = si.BSpline(t, y, k=3)  # Quadratic B-spline for y coordinates

    # Generate points along the spline (smooth points)
    t_smooth = np.linspace(0, 1, num_points)  # More points for smoothness
    x_smooth = spl_x(t_smooth)
    y_smooth = spl_y(t_smooth)

    # Create smoothed path with interpolated points
    smoothed_path = [(x_smooth[i], y_smooth[i], 0.0) for i in range(num_points)]
    return smoothed_path

def apply_spline_at_sharp_turns(path, angle_threshold=math.pi / 6, num_points=15):
    """
    Apply cubic spline smoothing only at sharp turns detected along the path.
    
    Args:
        path (list): List of waypoints [(x, y), (x, y), ...].
        angle_threshold (float): Angle in radians to classify a turn as sharp.
        num_points (int): Number of interpolated points in the smoothed segments.

    Returns:
        list: A partially smoothed path.
    """
    if len(path) < 3:
        rospy.logwarn("Not enough points for sharp turn detection, returning original path.")
        return path

    smoothed_path = [path[0]]  # Start with the first point

    i = 1
    while i < len(path) - 1:
        angle = calculate_angle(path[i - 1], path[i], path[i + 1])
        segment = [path[i - 1], path[i], path[i + 1]]

        # Apply smoothing if the turn angle exceeds the threshold (sharp turn)
        if math.pi/30 < abs(angle) < angle_threshold or abs(angle) > math.pi /4.7:
            # Identify a window of points around the sharp turn for smoothing
            start_idx = max(0, i - 1)
            end_idx = min(len(path) - 1, i + 2)
            segment = get_equidistant_points_on_curve(path[start_idx:end_idx + 1], 20)
            print(f"({angle}) SHARP TURN: {segment}")

            # Apply spline smoothing to the sharp turn segment
            smoothed_segment = generate_spline_path(segment, num_points)
            smoothed_path.extend(smoothed_segment[1:-1])  # Avoid duplicate endpoints

            i = end_idx  # Skip over the smoothed segment
        else:
            print(f"({angle}) ANGLE: {segment}")
            smoothed_path.append(path[i])  # Retain original point if no sharp turn
            i += 1

    smoothed_path.append(path[-1])  # Add the final point
    return smoothed_path

def get_equidistant_points_on_curve(path, num_points):
    """
    Get equidistant points along the B-spline curve.
    
    Args:
        path (list): Smoothed B-spline path.
        num_points (int): Number of equidistant points to sample from the B-spline.
    
    Returns:
        list: Equidistant points along the smooth B-spline path.
    """
    # Calculate the cumulative distances along the spline
    distances = [0.0]
    for i in range(1, len(path)):
        dist = np.linalg.norm(np.array(path[i][:2]) - np.array(path[i-1][:2]))
        distances.append(distances[-1] + dist)

    # Total path length
    total_length = distances[-1]

    # Equidistant points
    equidistant_points = []
    for i in range(num_points):
        target_dist = total_length * (i / (num_points - 1))
        # Find the closest point in the smoothed path
        for j in range(1, len(distances)):
            if distances[j] >= target_dist:
                t = (target_dist - distances[j-1]) / (distances[j] - distances[j-1])
                x = path[j-1][0] + t * (path[j][0] - path[j-1][0])
                y = path[j-1][1] + t * (path[j][1] - path[j-1][1])
                equidistant_points.append((x, y, 0.0))
                break

    return equidistant_points

def equidistant_points_dynamic(path, max_distance_between_points):
    """
    Convert the path into equidistant points based on a maximum distance between points.
    
    Args:
        path (list): List of waypoints [(x, y), (x, y), ...].
        max_distance_between_points (float): The maximum distance between consecutive points.
    
    Returns:
        list: List of equidistant points along the path, with original points unchanged.
    """
    equidistant_path = []  # Initialize the list for equidistant points

    # Always add the first point of the path as the starting point
    equidistant_path.append(path[0])

    # Iterate through the path segments and interpolate points based on max distance
    for i in range(1, len(path)):
        p0 = path[i - 1]
        p1 = path[i]

        # Calculate the Euclidean distance between the two points
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        segment_distance = math.sqrt(dx ** 2 + dy ** 2)

        # Add original points to the equidistant path if the distance is large enough
        if segment_distance > max_distance_between_points:
            # Calculate how many points are needed for this segment
            num_points_in_segment = int(segment_distance // max_distance_between_points)

            # Interpolate points along this segment
            for j in range(1, num_points_in_segment + 1):
                ratio = j / (num_points_in_segment + 1)
                new_point = ((1 - ratio) * p0[0] + ratio * p1[0], (1 - ratio) * p0[1] + ratio * p1[1])
                equidistant_path.append(new_point)

        # Always add the original endpoint of the segment
        equidistant_path.append(p1)

    return equidistant_path


def ensure_no_backtracking(path):
    """
    Ensure that the path points do not backtrack by checking the direction of consecutive points.

    Args:
        path (list): List of waypoints [(x, y), (x, y), ...].

    Returns:
        list: List of waypoints with no backtracking.
    """
    if len(path) < 2:
        return path

    cleaned_path = [path[0]]  # Start with the first point

    for i in range(1, len(path)):
        # Calculate direction between consecutive points
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]

        if i > 1:
            # Compare with the previous direction to ensure we don't backtrack
            dx2 = path[i][0] - path[i - 2][0]
            dy2 = path[i][1] - path[i - 2][1]

            # Dot product to determine if we are moving in the same direction
            dot_product = dx1 * dx2 + dy1 * dy2
            if dot_product < 0:  # If negative, it indicates backtracking
                continue  # Skip the current point to avoid backtracking

        cleaned_path.append(path[i])

    return cleaned_path

def smooth_path_with_spline(path, map_param, obstacles=global_obstacles):
    """
    Smooth the given path by first simplifying it and then applying cubic spline smoothing.

    Args:
        path (list): List of waypoints in the path.
        map_param (tuple): The map boundaries (xmin, ymin, xmax, ymax).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.

    Returns:
        list: The smoothed path.
    """
    # Step 1: Simplify path by removing unnecessary nodes using line-of-sight
    simplified_path = smooth_path(path, map_param, obstacles)
    
    # Step 2: Convert simplified path into equidistance points
    equidistant_path = equidistant_points_dynamic(simplified_path, max_dist_per_point)
    
    # Step 3: Change edges into smooth spline
    spline_path = apply_spline_at_sharp_turns(equidistant_path)
    
    # Step 4: Ensure that there is no backtracking (i.e., points should be in a consistent direction)
    smoothed_path = ensure_no_backtracking(spline_path)

    return smoothed_path

def visualize_spline_path(spline_path):
    spline_marker = Marker()
    spline_marker.header.frame_id = "map"
    spline_marker.header.stamp = rospy.Time.now()
    spline_marker.type = Marker.LINE_STRIP
    spline_marker.action = Marker.ADD
    spline_marker.scale.x = 0.05  # Line thickness
    spline_marker.color.a = 1.0  # Fully opaque
    spline_marker.color.r = 0.0  # Red
    spline_marker.color.g = 0.5  # Green
    spline_marker.color.b = 1.0  # Blue
    
    for node in spline_path:
        point = Point()
        point.x = node[0]
        point.y = node[1]
        point.z = 0.0  # Set to 0 for 2D
        spline_marker.points.append(point)
    
    path_pub.publish(spline_marker)

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
        # smoothed_path = smooth_path(path, map_param, obstacles=global_obstacles)
        smoothed_path = smooth_path_with_spline(path, map_param, obstacles=global_obstacles)
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
