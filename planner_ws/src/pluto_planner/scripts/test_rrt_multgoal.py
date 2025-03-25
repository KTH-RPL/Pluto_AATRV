import random
import math
import matplotlib.pyplot as plt
from local_planner import rrt_planner, euclidean_dist, smooth_path, smooth_path_with_spline  # Importing planner and smoothing function

# Define sample map boundaries
map_param = (0, 0, 15, 15)  # (xmin, ymin, xmax, ymax)

# Euclidean distance function to check collision between points
def euclidean_dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# Check if the point collides with any obstacle
def is_collision(point, obstacles, radius=0.5):
    x, y = point
    if len(obstacles) > 0:
        for (ox, oy, oradius) in obstacles:
            if euclidean_dist((x, y), (ox, oy)) < (oradius + radius):  # Check if within radius of an obstacle
                return True
    return False

# Generate a random valid position that doesn't overlap with obstacles
def generate_random_position(map_param, obstacles, radius=0.5):
    xmin, ymin, xmax, ymax = map_param
    while True:
        x = random.uniform(xmin + radius, xmax - radius)  # Ensure point is within the map and not on edges
        y = random.uniform(ymin + radius, ymax - radius)
        if not is_collision((x, y), obstacles, radius):
            return (x, y, 0.0)

# Generate a random obstacle that does not overlap with others or the start/goal
def generate_random_obstacle(map_param, obstacles, radius_min=0.2, radius_max=1.0):
    radius = random.uniform(radius_min, radius_max)
    while True:
        x, y, _ = generate_random_position(map_param, obstacles, radius)
        if not is_collision((x, y), obstacles, radius):
            return (x, y, radius)

# Randomize start, goal, and obstacles
def randomize_start_goal_obstacles(map_param, num_obstacles=10, num_goals=3):
    

    # Random obstacles
    obstacles = []
    for _ in range(num_obstacles):
        obstacles.append(generate_random_obstacle(map_param, obstacles))
        
    # Random start point
    start = generate_random_position(map_param, obstacles)
    
    # Random goals
    goals = []
    for _ in range(num_goals):
        goal = generate_random_position(map_param, [(start[0], start[1], 0)] + obstacles)
        goals.append(goal)
    

    
    return start, goals, obstacles

# Visualization Utility Function
def plot_obstacles(obstacles):
    """
    Plot the obstacles on the map.
    
    Args:
        obstacles (list): List of obstacles as (x, y, radius).
    """
    for obs in obstacles:
        obs_x, obs_y, obs_r = obs
        circle = plt.Circle((obs_x, obs_y), obs_r, color='purple', alpha=0.5)
        plt.gca().add_artist(circle)

def draw_rrt_tree_and_paths(tree, nodes, path, smoothed_path, start, goals, obstacles=None):
    """
    Visualize the RRT tree, explored nodes, final path, and smoothed path.
    
    Args:
        tree (dict): Dictionary of nodes and their parents.
        nodes (list): List of all explored nodes.
        path (list): Final unsmoothed path from start to goal.
        smoothed_path (list): Final smoothed path.
        start (tuple): Starting position (x, y).
        goals (list): List of goal positions (x, y).
        obstacles (list, optional): List of obstacles as (x, y, radius). Defaults to None.
    """

    plt.figure(figsize=(10, 10))

    # Plot obstacles
    if obstacles:
        plot_obstacles(obstacles)

    # # Plot branches (edges between nodes in the tree)
    # for node, parent in tree.items():
    #     if parent is not None:
    #         x1, y1, _ = node
    #         x2, y2, _ = parent
    #         plt.plot([x1, x2], [y1, y2], color="blue", linestyle="dotted", alpha=0.4)

    # # Plot explored nodes
    # x_nodes, y_nodes = zip(*[(n[0], n[1]) for n in nodes])
    # plt.scatter(x_nodes, y_nodes, color="gray", s=10, label="Explored Nodes")

    # Plot the final unsmoothed path
    if path:
        x_path, y_path = zip(*[(p[0], p[1]) for p in path])
        plt.plot(x_path, y_path, color="red", linewidth=2, label="Original Path")

    # Plot the smoothed path
    if smoothed_path:
        x_smooth, y_smooth = zip(*[(p[0], p[1]) for p in smoothed_path])
        plt.plot(x_smooth, y_smooth, color="green", linewidth=2, linestyle="--", label="Smoothed Path")
        plt.scatter(x_smooth, y_smooth, color="red", s=20, label="Path Points")  # 's' sets dot size

    # Highlight the start point
    plt.scatter(*start[:2], color="green", s=100, label="Start", edgecolors="black")

    # Plot the goals with different colors
    goal_colors = ['magenta', 'orange', 'blue', 'cyan', 'yellow']  # Extend if there are more goals
    for i, goal in enumerate(goals):
        plt.scatter(*goal[:2], color=goal_colors[i % len(goal_colors)], s=100, label=f"Goal {i+1}", edgecolors="black")

    plt.title("RRT Path Planning: Original vs. Smoothed Path with Obstacles")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.grid(True)
    plt.legend(loc="best")
    plt.axis("equal")
    plt.show()


# Randomize start, goal (multiple goals), and obstacles
start, goals, obstacles = randomize_start_goal_obstacles(map_param, num_obstacles=30, num_goals=3)

# Override start and goal
start = (map_param[0]+1, map_param[1]+1, 0.0)
goals = [(map_param[0]+1, map_param[3]-1, 0.0), (map_param[2]-1, map_param[3]-1, 0.0), (map_param[2]-1, map_param[1]+1, 0.0)]


# Initialize path variables
total_path = []
smoothed_total_path = []

# Iterate through the goal points and plan the path sequentially
current_start = start
for goal in goals:
    print(f"\nNavigating to goal: {goal}")
    
    # Run the RRT planner for each goal
    path, tree, nodes = rrt_planner(current_start, goal, map_param, obstacles=obstacles)  # Pass obstacles to the planner
    
    # Test results
    if path:
        print(f"✅ Path found with {len(path)} waypoints.")
        print("Path:", path)
        total_path.extend(path)
        
        # Smooth the path using the smoothing function from local_planner
        smoothed_path = smooth_path_with_spline(path, map_param, obstacles=obstacles)  # Pass obstacles to smooth_path
        print(f"✅ Smoothed path with {len(smoothed_path)} waypoints.")
        print("Smoothed Path:", smoothed_path)
        smoothed_total_path.extend(smoothed_path)
        
        # Update the start point to the last point of the current goal
        current_start = path[-1]
    else:
        print("❌ No path found!")

# Visualize the RRT exploration, original path, and smoothed path with obstacles
draw_rrt_tree_and_paths(tree, nodes, total_path, smoothed_total_path, start, goals, obstacles=obstacles)

