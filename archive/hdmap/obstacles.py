import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon


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
            return False, "Robot is outside the boundary"

        for obs in self.obstacles:
            if robot_shape.intersects(obs):
                return False, "Robot collides with an obstacle"

        return True, "Robot is in free space"

    def get_robot_polygon(self, center, length, width):
        cx, cy = center
        half_length = length / 2
        half_width = width / 2

        return [
            (cx - half_length, cy - half_width),
            (cx + half_length, cy - half_width),
            (cx + half_length, cy + half_width),
            (cx - half_length, cy + half_width),
        ]

    def visualize(self, center=None, length=None, width=None):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot boundary
        bx, by = self.boundary.exterior.xy
        ax.plot(bx, by, "b-", linewidth=2, label="Boundary")
        ax.fill(bx, by, "cyan", alpha=0.2)

        # Plot obstacles
        for i, obs in enumerate(self.obstacles):
            ox, oy = obs.exterior.xy
            ax.plot(ox, oy, "r-", linewidth=2)
            ax.fill(ox, oy, "red", alpha=0.4)

        # Plot robot polygon
        if center and length and width:
            robot_polygon = self.get_robot_polygon(center, length, width)
            rx, ry = zip(*robot_polygon)
            ax.plot(rx + (rx[0],), ry + (ry[0],), "g-", linewidth=2, label="Robot")
            ax.fill(rx, ry, "green", alpha=0.5)

        ax.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Polygon Map with Obstacles and Robot")
        plt.grid(True)
        plt.show()
