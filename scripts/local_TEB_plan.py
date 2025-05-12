"""
Local Planner Function with TEB (Time Elastic Band) Method
Author: Prasetyo Wibowo L S <pwlsa@kth.se>

Assumptions:
1. Robot is defined as a simple shape (circle)
2. Robot position defined by the center of the circle

Input:
1. Current Position of the Robot
2. Goal Position
3. Occupancy Map

Output:
1. List of points to follow by the controller
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === INPUTS ===
start = (1.0, 1.0)
goal = (9.0, 9.0)

def is_free(x, y):
    """Example collision checker with a circular obstacle"""
    obs_x, obs_y, radius = 5.0, 5.0, 1.0
    return np.sqrt((x - obs_x)**2 + (y - obs_y)**2) > radius

# === STEP 1: Generate Initial Path ===
def generate_initial_path(start, goal, num_points=10):
    path = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = (1 - t) * start[0] + t * goal[0]
        y = (1 - t) * start[1] + t * goal[1]
        path.append([x, y])
    return np.array(path)

# === STEP 2: Define Cost Function ===
def teb_cost(x_flat, N, is_free, lambda_obs=10.0):
    """
    x_flat: Flattened array of [x0, y0, x1, y1, ..., xN-1, yN-1]
    """
    x = x_flat.reshape(N, 2)
    cost = 0.0

    # 1. Smoothness (penalize velocity and acceleration)
    for i in range(1, N-1):
        prev = x[i-1]
        curr = x[i]
        next_ = x[i+1]
        v1 = curr - prev
        v2 = next_ - curr
        acc = v2 - v1
        cost += np.linalg.norm(acc)**2

    # 2. Obstacle avoidance
    for pt in x:
        if not is_free(pt[0], pt[1]):
            cost += lambda_obs

    # 3. Time-optimal (shorter paths preferred)
    for i in range(N-1):
        cost += np.linalg.norm(x[i+1] - x[i])

    return cost

# === STEP 3: Optimize the path ===
initial_path = generate_initial_path(start, goal, num_points=15)
x0 = initial_path.flatten()

bounds = []
for pt in initial_path:
    bounds.extend([(0.0, 10.0), (0.0, 10.0)])  # Assuming map is 10x10

result = minimize(teb_cost, x0, args=(len(initial_path), is_free), bounds=bounds)

# === STEP 4: Return trajectory ===
final_path = result.x.reshape(-1, 2)

# === STEP 5: Visualize ===
def visualize(path, start, goal):
    plt.figure(figsize=(6, 6))
    plt.plot(path[:, 0], path[:, 1], 'b-o', label='TEB Trajectory')
    plt.plot(start[0], start[1], 'go', label='Start')
    plt.plot(goal[0], goal[1], 'ro', label='Goal')

    # Draw obstacle
    obs = plt.Circle((5, 5), 1.0, color='gray', alpha=0.5)
    plt.gca().add_patch(obs)

    plt.grid()
    plt.legend()
    plt.axis('equal')
    plt.title("Time Elastic Band Local Planner")
    plt.show()

visualize(final_path, start, goal)

# === OUTPUT ===
print("Trajectory to follow:")
for pt in final_path:
    print(f"{pt[0]:.2f}, {pt[1]:.2f}")
