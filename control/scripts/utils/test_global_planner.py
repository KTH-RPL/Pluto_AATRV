import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from global_planner_test import execute_global_planning, rrt_planner, is_node_valid, is_line_of_sight
import math
import gmap_utility

def main():
    """
    Main function to run the RRT planner and visualize the results.
    """
    # Define start and goal points (x, y, theta)
    start = (138, 82)
    # goals = [(25,-110)]
    goals = [
            (120,65),
            (119.5, 50), 
            (120, 37), 
            (140, 30), 
            (190,10), 
            (170,-50), 
            (145,-60), 
            (121,-105), 
            (100,-136), 
            (25,-110)
        ]
    robot_radius = 1.2

    # Use real obstacles from the polygon_map in global_planner.py
    sim_obstacles = gmap_utility.polygon_map.obstacles
    
    # You can also use the obstacle generation function from the global_planner.py
    # smoothed_path, tree, nodes, raw_path = execute_global_planning(start, goal, sim_plan=False, add_sim_obstacles=True)

    # Execute the global planning function from the imported script
    print("Executing global planning...")
    total_path = [start]
    for goal in goals:
        smoothed_path, tree, nodes, raw_path = execute_global_planning(start, goal, sim_plan=False)
        start = goal
        total_path.extend(smoothed_path[1:])
        print("Planning complete.")

    gmap_utility.polygon_map.visualize(trajectory=total_path, radius=robot_radius)


if __name__ == "__main__":
    main()
