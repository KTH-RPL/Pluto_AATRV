import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from global_planner import execute_global_planning, rrt_planner, is_node_valid, is_line_of_sight
import math
import gmap_utility

def main():
    """
    Main function to run the RRT planner and visualize the results.
    """
    # Define start and goal points (x, y, theta)
    start = (1.0, 1.0, 0.0)
    goal = (18.0, 18.0, 0.0)

    # Use real obstacles from the polygon_map in global_planner.py
    sim_obstacles = gmap_utility.polygon_map.obstacles
    
    # You can also use the obstacle generation function from the global_planner.py
    # smoothed_path, tree, nodes, raw_path = execute_global_planning(start, goal, sim_plan=False, add_sim_obstacles=True)

    # Execute the global planning function from the imported script
    print("Executing global planning...")
    smoothed_path, tree, nodes, raw_path = execute_global_planning(start, goal, sim_plan=False)
    print("Planning complete.")

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', 'box')
    ax.set_title('RRT Path Planning Visualization')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    # Add obstacles to the plot
    for obstacle in sim_obstacles:
        # Obstacle is a polygon, so we plot the vertices
        poly = patches.Polygon(obstacle.exterior.xy, closed=True, edgecolor='black', facecolor='gray', alpha=0.7)
        ax.add_patch(poly)

    # Plot the RRT tree
    # The 'tree' dictionary maps a node to its parent
    print("Plotting RRT tree...")
    for node, parent in tree.items():
        if parent is not None:
            # Draw a line from the parent to the child
            ax.plot([parent[0], node[0]], [parent[1], node[1]], 'go-', markersize=2, linewidth=0.5, alpha=0.5)

    # Plot the original RRT path
    print("Plotting RRT path...")
    if raw_path:
        raw_path_x = [p[0] for p in raw_path]
        raw_path_y = [p[1] for p in raw_path]
        ax.plot(raw_path_x, raw_path_y, 'b--', linewidth=2, label='RRT Raw Path')

    # Plot the smoothed path
    print("Plotting smoothed path...")
    if smoothed_path:
        smoothed_path_x = [p[0] for p in smoothed_path]
        smoothed_path_y = [p[1] for p in smoothed_path]
        ax.plot(smoothed_path_x, smoothed_path_y, 'r-', linewidth=3, label='Smoothed Path')
        
    # Plot start and goal points
    ax.plot(start[0], start[1], 'ro', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'bo', markersize=10, label='Goal')

    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()
