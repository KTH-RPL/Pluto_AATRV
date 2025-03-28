import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import alphashape
from shapely.geometry import Point, Polygon
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon, Point

class PolygonMap:
    def __init__(self, boundary_ply, obstacle_ply_list):
        self.boundary = self.load_polygon(boundary_ply)
        self.obstacles = [self.load_polygon(ply) for ply in obstacle_ply_list]

    def load_polygon(self, ply_file):
        pcd = o3d.io.read_point_cloud(ply_file)
        vertices = np.asarray(pcd.points)[:, :2]  
        return Polygon(vertices)

    def is_valid_robot_pos(self, center, radius):
        # robot_polygon = self.get_robot_polygon(center, length, width)
        # robot_shape = Polygon(robot_polygon)

        robot_shape = Point(center).buffer(radius)

        if not robot_shape.within(self.boundary):
            return False, "Outside boundary"

        for obs in self.obstacles:
            if robot_shape.intersects(obs):
                return False, "Collision with obstacle"

        return True, "Safe position"

    # def get_robot_polygon(self, center, length, width):
    #     cx, cy = center
    #     half_length = length / 2
    #     half_width = width / 2

    #     return [
    #         (cx - half_length, cy - half_width),
    #         (cx + half_length, cy - half_width),
    #         (cx + half_length, cy + half_width),
    #         (cx - half_length, cy + half_width)
    #     ]

    def is_trajectory_safe(self, trajectory, radius):
        for point in trajectory:
            valid, message = self.is_valid_robot_pos(point, radius)
            if not valid:
                return False, f"Unsafe at {point}: {message}"
        return True, "Trajectory is safe"

    def visualize(self, trajectory=None, radius=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot boundary
        bx, by = self.boundary.exterior.xy
        ax.plot(bx, by, 'b-', linewidth=2, label="Boundary")
        ax.fill(bx, by, 'cyan', alpha=0.2)

        # Plot obstacles
        for i,obs in enumerate(self.obstacles):
            ox, oy = obs.exterior.xy
            ax.plot(ox, oy, 'r-', linewidth=2)
            ax.fill(ox, oy, 'red', alpha=0.4)

        # Plot trajectory
        if trajectory:
            tx, ty = zip(*trajectory)
            ax.plot(tx, ty, 'm--', linewidth=2, label="Trajectory")

        # Plot robot at start and end
        if trajectory and radius:
            for center, color, label in zip([trajectory[0], trajectory[-1]], ['green', 'orange'], ["Start", "End"]):
                circle = plt.Circle(center, radius, color=color, alpha=0.5, label=f"Robot - {label}")
                ax.add_patch(circle)

        ax.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Polygon Map with Obstacles and Trajectory")
        plt.grid(True)
        plt.show()

import os
"""for testing"""

folder = "./boundaries"
boundary_file = os.path.join(folder, "milestone1_vertices.ply")
obstacle_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("obst")]

polygon_map = PolygonMap(boundary_file, obstacle_files)

from local_planner import execute_planning, robot_radius


start = (120,97)
goal = (140,65)

path, _, _, _ = execute_planning(start, goal) 
path_clean = [(p[0],p[1]) for p in path]

# Visualize the map with trajectory
polygon_map.visualize(trajectory=path_clean, radius=robot_radius)
