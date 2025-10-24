"""
Global Planner Function
Author: Prasetyo Wibowo L S <pwlsa@kth.se>

Assumptions:
1. Robot is defined as a simple shape (circle)
2. Robot position defined by the center of the circle
"""

import gmap_utility
import numpy as np
import math
# import rospy
import scipy.interpolate as si
import random
import os
from scipy.spatial import KDTree



import argparse
parser = argparse.ArgumentParser(description="Local Planner for Pluto")
parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose mode")
args, unknown = parser.parse_known_args()
verbose_g = False
if args.verbose:
    verbose_g = True

# --------------- Global Variables ---------------------
max_dist_per_point = 0.3
robot_radius = 1.2
sim_obs_gl = None


# --------------- Generate RRT Path --------------------
# Euclidean distance function
def euclidean_dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# TO BE REPLACED WITH RAJESH's function
# Check if a node is valid
def is_node_valid(node, map_param=None, obstacles=None):
    if map_param is None:
        xmin, ymin, xmax, ymax = -math.inf, -math.inf, math.inf, math.inf
    else:
        xmin, ymin, xmax, ymax = map_param
        
    if len(node)==3:
        x, y, _ = node
    elif len(node)==2:
        x,y = node

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
def generate_neighbors(node, step_size=0.5, num_samples=5, check_node_valid=is_node_valid, verbose=False):
    neighbors = []
    x, y = node[:2]

    for _ in range(num_samples):  # Generate multiple candidate points
        angle = random.uniform(-math.pi, math.pi)  # Random direction
        xn = x + step_size * math.cos(angle)
        yn = y + step_size * math.sin(angle)
        thetan = random.uniform(-math.pi, math.pi)  # Random heading

        new_node = (xn, yn, thetan)

        if check_node_valid(new_node):
        # if is_node_valid(new_node, map_param, obstacles=obstacles):
            neighbors.append(new_node)  # No steering angle needed
        # else:
            # print(f"{new_node} is not valid")

    return neighbors

def rrt_planner(start, goal, map_param, check_node_valid=is_node_valid, step_size=3.0, goal_threshold=0.5, num_samples=10, max_iter=1000000, prob_goal_bias=0.3, verbose=False):
    """
    RRT planner optimized with a k-d tree for fast nearest-neighbor searches.
    """
    tree = {start: None}
    nodes = [start]
    
    # --- KDTree Optimization ---
    # We maintain a separate list of 2D coordinates to build the k-d tree,
    # as it's much faster than working with the full (x, y, theta) tuples.
    node_coords = [start[:2]]

    # --- DYNAMIC STEP SIZE LOGIC ---
    # Define the min/max bounds for our step size.
    MIN_STEP_SIZE = 0.4  # The smallest step the planner can take.
    MAX_STEP_SIZE = 5.0  # The largest step the planner can take.
    
    # Calculate the straight-line distance to the goal.
    distance_to_goal = euclidean_dist(start[:2], goal[:2])
    
    # Scale the step size based on the distance. Here, we'll make the
    # ideal step size about 1/5th of the total distance.
    # You can tune this scaling factor (0.2) to change aggressiveness.
    dynamic_step_size = distance_to_goal * 0.2
    
    # Clamp the calculated step size between the min and max bounds.
    # This prevents it from being too small for tiny distances or too
    # large for very long distances.
    step_size = max(MIN_STEP_SIZE, min(dynamic_step_size, MAX_STEP_SIZE))
    
    xlb, ylb, xub, yub = map_param

    for i in range(max_iter):
        if verbose:
            print("Iteration {}".format(i))
        
        # 1. Sample a random point
        if random.random() < prob_goal_bias:
            x_rand = goal
        else:
            x_rand = (
                random.uniform(xlb, xub),
                random.uniform(ylb, yub),
                random.uniform(-math.pi, math.pi)
            )
        
        # --- KDTree Optimization ---
        # 2. Find the nearest node using the k-d tree (O(log N) search)
        # This replaces the slow linear search: `min(nodes, key=...)`
        kdtree = KDTree(node_coords)
        _, idx = kdtree.query(x_rand[:2])
        x_nearest = nodes[idx]
        
        if verbose:
            print("> Exploration node selected {}".format(x_nearest))
            
        # 3. Generate random neighbors from the nearest node
        neighbors = generate_neighbors(x_nearest, step_size=step_size, num_samples=num_samples, check_node_valid=check_node_valid, verbose=verbose)

        if verbose:
            print("> Neighbors generated")

        # 4. Process new potential nodes
        for x_new in neighbors:
            # Add the new node to the tree if it's valid
            # (The original code had a duplicate validity check here, which is fine)
            if check_node_valid(x_new):
                tree[x_new] = x_nearest
                nodes.append(x_new)
                
                # --- KDTree Optimization ---
                # Keep the coordinate list in sync with the main node list
                node_coords.append(x_new[:2])
                
                if verbose:
                    print("Append node {}".format(x_new))

                # 5. Goal proximity check
                if euclidean_dist(x_new[:2], goal[:2]) < goal_threshold:
                    if verbose:
                        print("Goal reached on iteration {}".format(i))
                    # Return the full path, the tree, and all nodes
                    return reconstruct_path(tree, x_new), tree, nodes

    print("RRT planner failed to find a path after max iterations.")
    # Return None for all expected values if no path is found
    return None, None, None
    
def rrt_planner_old(start, goal, map_param, check_node_valid=is_node_valid, step_size=3.0, goal_threshold=0.5, num_samples=10, max_iter=1000000, prob_goal_bias=0.3, verbose=False):
    tree = {start: None}
    nodes = [start]
    explored = []

    xlb, ylb, xub, yub = map_param

    iq = 0

    for i in range(max_iter):
        if verbose:
            print("Iteration {}".format(i))
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
            
        if verbose:
            print("> Direction selected")

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
            print("Could not find a new node to explore after multiple attempts!")
            continue  # Skip this iteration
        
        if verbose:
            print("> Exploration node selected {}".format(x_nearest))
            

        # print(x_nearest)

        # Generate random neighbors based on x_nearest point
        neighbors = generate_neighbors(x_nearest, step_size=step_size, num_samples=num_samples, check_node_valid=check_node_valid, verbose=verbose)
        # print(neighbors)

        if verbose:
            print("> Neighbors generated")

        for x_new in neighbors:
            # Add new node to the tree if valid
            if check_node_valid(x_new):
            # if check_node_valid(x_new, map_param, obstacles=obstacles):
                tree[x_new] = (x_nearest)
                nodes.append(x_new)
                if verbose:
                    print("Append node {}".format(x_new))

                # Step 5: Goal proximity check
                if euclidean_dist(x_new[:2], goal[:2]) < goal_threshold:
                    #draw_map(car, nodes, start, goal, x_new)
                    if verbose:
                        print("Goal reached on iteration {}".format(i))
                    return reconstruct_path(tree, x_new), tree, nodes

            iq+=1
        # explored.append(x_nearest) ##

    return None

# Reconstruct the path from the tree
def reconstruct_path(tree, current_node):
    total_path = [current_node]
    # print(tree)
    while current_node in tree and (tree[current_node] != None):
        current_node = tree[current_node]
        total_path.append(current_node)

    total_path.reverse()
    return total_path


# --------------- Path Smoothing --------------------

def is_line_of_sight(a, b, check_node_valid=is_node_valid):
    """
    Checks if the line segment connecting points a and b is free of obstacle
    
    Args:
        a (tuple): The start point (x, y, theta).
        b (tuple): The end point (x, y, theta).
        map_param (tuple): The map boundaries (xmin, ymin, xmax, ymax).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.
    
    Returns:
        bool: True if there is a clear line of sight, False otherwise.
    """
    x1, y1 = a[:2]
    x2, y2 = b[:2]

    # Check if the line is within map boundaries
    num_points = 50
    for i in range(num_points + 1):
        t = i / num_points
        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2
        # if x < xmin or x > xmax or y < ymin or y > ymax:
        #     return False
        
        # Check the validity of the interpolated point
        if not check_node_valid((x, y)):
            return False
        
    return True


def simplify_path(path, map_param, check_node_valid=is_node_valid):
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
            if is_line_of_sight(path[i], path[j], check_node_valid=check_node_valid):
                break
            j -= 1
        smoothed_path.append(path[j])
        i = j

    return smoothed_path

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

def apply_spline_at_sharp_turns(path, angle_threshold=math.pi / 6, num_points=15, check_node_valid=is_node_valid):
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
        print("Not enough points for sharp turn detection, returning original path.")
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
            # print(f"({angle}) SHARP TURN: {segment}")

            # Apply spline smoothing to the sharp turn segment
            smoothed_segment = generate_spline_path(segment, num_points)

             # Check if the smoothed segment is valid
            is_valid_segment = all(check_node_valid(point) for point in smoothed_segment) if check_node_valid else True

            if is_valid_segment:
                # Add smoothed points, avoiding duplicate endpoints
                smoothed_path.extend(smoothed_segment[1:-1])
                i = end_idx  # Skip over the smoothed segment
            else:
                smoothed_path.append(path[i])  # Retain original point if smoothed path not valid
                i += 1
        else:
            # print(f"({angle}) ANGLE: {segment}")
            smoothed_path.append(path[i])  # Retain original point if no sharp turn
            i += 1

    smoothed_path.append(path[-1])  # Add the final point
    return smoothed_path

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

def generate_trajectory_heading(path):
    """
    To generate trajectory heading based on the path given 

    Args:
        path (list): List of waypoints in the path [(x, y)]

    Returns:
        list: List of points and heading in the format of [(x, y, theta)]. Theta defined based on t
    """

    final_path = []

    for i in range(len(path)):
        x, y = path[i][:2]
    
        # For last node, heading = previous node, else calculate the heading using atan2
        if i == len(path)-1:
            theta = final_path[i-1][-1]
        else:
            x2, y2 = path[i+1][:2]
            theta =  math.atan2((y2-y), (x2-x))

        final_path.append((x,y,theta))

    return final_path




def smooth_path(path, map_param, check_node_valid=is_node_valid, verbose=False):
    """
    Smooth the given path by first simplifying it and then applying equidistant conversion. 

    Args:
        path (list): List of waypoints in the path.
        map_param (tuple): The map boundaries (xmin, ymin, xmax, ymax).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.

    Returns:
        list: The smoothed path.
    """
    # Step 1: Simplify path by removing unnecessary nodes using line-of-sight
    simplified_path = simplify_path(path, map_param, check_node_valid=check_node_valid)
    print("1. Simplify path done")
    
    # Step 2: Convert simplified path into equidistance points
    equidistant_path = equidistant_points_dynamic(simplified_path, max_dist_per_point)
    print("2. Equidistant path done")
    
    # # Step 3: Change edges into smooth spline
    # spline_path = apply_spline_at_sharp_turns(equidistant_path, check_node_valid=check_node_valid)
    # print("3. Spline path done")
    
    # # Step 4: Ensure that there is no backtracking (i.e., points should be in a consistent direction)
    # smoothed_path = ensure_no_backtracking(spline_path)
    # print("4. Backtrack check done")

    # # Step 5: Generate heading of the trajectory points
    # finalized_path = generate_trajectory_heading(smoothed_path)
    # print("5. Final path done")

    finalized_path = equidistant_path

    return finalized_path


def execute_global_planning(start, goal, sim_plan=False, sim_obstacles=None, verbose=verbose_g, add_sim_obstacles=False):
    global robot_radius, sim_obs_gl
    
    if sim_plan:
        def check_node_valid(node):
            return is_node_valid(node, obstacles = sim_obstacles)
        
    elif add_sim_obstacles:
        if sim_obs_gl is None:
            sim_obs_gl = []
            num_obs = 2
            while len(sim_obs_gl) < num_obs:
                t = random.uniform(0.3, 0.8)
                base_x = start[0] + t * (goal[0] - start[0])
                base_y = start[1] + t * (goal[1] - start[1])

                obs_x = base_x
                obs_y = base_y

                if len(sim_obs_gl) > 0:
                    obs_x = base_x + random.uniform(-2.0, 2.0)
                    obs_y = base_y + random.uniform(-2.0, 2.0)

                obstacle_radius = random.uniform(0.2, 0.5)

                # Check if start or goal is inside obstacle
                dist_to_start = ((obs_x - start[0])**2 + (obs_y - start[1])**2)**0.5
                dist_to_goal = ((obs_x - goal[0])**2 + (obs_y - goal[1])**2)**0.5

                if dist_to_start <= obstacle_radius + robot_radius or dist_to_goal <= obstacle_radius + robot_radius:
                    # Bad obstacle, retry
                    print("Bad obstacle, retrying obstacle generation")
                    continue

                sim_obs_gl.append((obs_x, obs_y, obstacle_radius))

        print("ADDING SIMULATED OBSTACLES", sim_obs_gl)
            
        def check_node_valid(node):
            return is_node_valid(node, obstacles = sim_obs_gl)       

    else:
        # folder = "../milestone2_gmap_dataset"
        # folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), folder)
        # boundary_file = os.path.join(folder, "final_boundary_comb.ply")
        # obstacle_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("obst")]
        polygon_map = gmap_utility.polygon_map

        def check_node_valid(node, robot_radius=robot_radius):
            node = node[:2]
            is_safe, message = polygon_map.is_valid_robot_pos(node, radius=robot_radius)
            return is_safe

    map_param = [start[0], start[1], goal[0], goal[1]]
    print(map_param)

    # Run the RRT planner
    print("Running RRT Planner")
    path, tree, nodes = rrt_planner(start, goal, map_param, check_node_valid=check_node_valid, verbose=verbose)  # Pass obstacles to the planner
    print("0. RRT path done")

    # Smooth the path using the smoothing function from local_planner
    smoothed_path = smooth_path(path, map_param, check_node_valid=check_node_valid, verbose=verbose)  # Pass obstacles to smooth_path
    
    print("Generated path: {}".format(smoothed_path))
    return smoothed_path, tree, nodes, path

# -------- SAMPLE USAGE --------------------
# from local_planner import execute_planning
# path, _, _, _ = execute_planning(start, goal) 