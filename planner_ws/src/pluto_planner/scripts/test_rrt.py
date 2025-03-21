import random
import math
from pluto_planner import rrt_planner, euclidean_dist

# Define test map boundaries
map_param = (0, 0, 10, 10)  # (xmin, ymin, xmax, ymax)

# Define start and goal positions
start = (1, 1, 0.0)
goal = (8, 8, 0.0)

# Run the planner
path = rrt_planner(start, goal, map_param)

# Test results
if path:
    print(f"✅ Path found with {len(path)} waypoints.")
    print("Path:", path)
else:
    print("❌ No path found!")