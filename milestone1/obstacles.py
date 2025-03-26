import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon

# import os


class PolygonMap:
    def __init__(self, boundary_ply, obstacle_ply_list):
        self.boundary = self.load_polygon(boundary_ply)
        self.obstacles = [self.load_polygon(ply) for ply in obstacle_ply_list]

    def load_polygon(self, ply_file):
        pcd = o3d.io.read_point_cloud(ply_file)
        vertices = np.asarray(pcd.points)[:, :2]  
        return Polygon(vertices)

    def is_valid_robot_pos(self, center, length, width):
        robot_polygon = self.get_robot_polygon(center, length, width)
        robot_shape = Polygon(robot_polygon)

        if not robot_shape.within(self.boundary):
            return False, "Outside boundary"

        for obs in self.obstacles:
            if robot_shape.intersects(obs):
                return False, "Collision with obstacle"

        return True, "Safe position"

    def get_robot_polygon(self, center, length, width):
        cx, cy = center
        half_length = length / 2
        half_width = width / 2

        return [
            (cx - half_length, cy - half_width),
            (cx + half_length, cy - half_width),
            (cx + half_length, cy + half_width),
            (cx - half_length, cy + half_width)
        ]

    def is_trajectory_safe(self, trajectory, length, width):
        for point in trajectory:
            valid, message = self.is_valid_robot_pos(point, length, width)
            if not valid:
                return False, f"Unsafe at {point}: {message}"
        return True, "Trajectory is safe"

    def visualize(self, trajectory=None, length=None, width=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot boundary
        bx, by = self.boundary.exterior.xy
        ax.plot(bx, by, 'b-', linewidth=2, label="Boundary")
        ax.fill(bx, by, 'cyan', alpha=0.2)

        # Plot obstacles
        for i, obs in enumerate(self.obstacles):
            ox, oy = obs.exterior.xy
            ax.plot(ox, oy, 'r-', linewidth=2)
            ax.fill(ox, oy, 'red', alpha=0.4)

        # Plot trajectory
        if trajectory:
            tx, ty = zip(*trajectory)
            ax.plot(tx, ty, 'm--', linewidth=2, label="Trajectory")

        # Plot robot at start and end
        if trajectory and length and width:
            start_robot = self.get_robot_polygon(trajectory[0], length, width)
            end_robot = self.get_robot_polygon(trajectory[-1], length, width)

            for robot_polygon, color, label in zip([start_robot, end_robot], ['green', 'orange'], ["Start", "End"]):
                rx, ry = zip(*robot_polygon)
                ax.plot(rx + (rx[0],), ry + (ry[0],), color, linewidth=2, label=f"Robot - {label}")
                ax.fill(rx, ry, color, alpha=0.5)

        ax.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Polygon Map with Obstacles and Trajectory")
        plt.grid(True)
        plt.show()


"""for testing"""

# folder = "milestone1"
# boundary_file = os.path.join(folder, "milestone1_vertices.ply")
# obstacle_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith("obst")]

# trajectory = [(130, 85), (128, 88), (126, 90), (125, 93), (120, 97)]
# car_length = 1
# car_width = 1
# polygon_map = PolygonMap(boundary_file, obstacle_files)

# # Check if trajectory is safe
# is_safe, message = polygon_map.is_trajectory_safe(trajectory, length=car_length, width=car_width)
# print(is_safe)
# print(message)

# # Visualize the map with trajectory
# polygon_map.visualize(trajectory=trajectory, length=car_length, width=car_width)