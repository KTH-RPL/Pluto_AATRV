#!/usr/bin/env python3

# ROS1 Imports
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point, Pose, Quaternion

# Standard Python Imports
import numpy as np
import cv2

class CostmapGeneratorNode:
    def __init__(self):
        rospy.init_node('costmap_generator_node', anonymous=True)

        # --- Get parameters from ROS Param Server ---
        # Topics
        obstacle_topic = rospy.get_param('~obstacle_topic', '/detected_obstacles')
        costmap_topic = rospy.get_param('~costmap_topic', '/local_costmap')

        # Costmap properties
        self.map_size_meters = rospy.get_param('~map_size', 20.0)    # 20m x 20m
        self.resolution = rospy.get_param('~resolution', 0.1)       # 0.1m per cell
        self.robot_radius = rospy.get_param('~robot_radius', 0.7)   # Robot radius for inflation
        self.map_frame = rospy.get_param('~map_frame', 'base_link') # Costmap is in the robot's frame

        # --- Pre-calculate map properties ---
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        
        # The origin of the map in the base_link frame.
        # Since the map is centered on the robot, the origin (bottom-left corner)
        # is at (-map_size/2, -map_size/2).
        self.origin_x = -self.map_size_meters / 2.0
        self.origin_y = -self.map_size_meters / 2.0
        
        # --- Create the inflation kernel once ---
        # This is more efficient than creating it in every callback.
        inflation_radius_cells = int(self.robot_radius / self.resolution)
        # Using an elliptical kernel is a good approximation for a circle
        self.inflation_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * inflation_radius_cells + 1, 2 * inflation_radius_cells + 1)
        )

        # --- ROS Subscriber and Publisher ---
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, self.obstacles_callback, queue_size=1)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)

        rospy.loginfo("Costmap Generator Node initialized.")
        rospy.loginfo(f"Listening for obstacles on: {obstacle_topic}")
        rospy.loginfo(f"Publishing costmap on: {costmap_topic}")

    def world_to_map(self, wx, wy):
        """Converts world coordinates (in meters, in map_frame) to map coordinates (in cells)."""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def obstacles_callback(self, marker_array_msg):
        """Processes obstacle markers and generates the costmap."""
        try:
            # 1. Create a fresh, empty grid for this timestep
            # OccupancyGrid values: 0=free, 100=occupied
            costmap_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)

            # 2. Rasterize (draw) each obstacle polygon onto the grid
            for marker in marker_array_msg.markers:
                if marker.action != marker.ADD or marker.type != marker.LINE_STRIP or len(marker.points) < 3:
                    continue

                # Convert polygon points from world coordinates to map cell coordinates
                polygon_points_map = []
                for world_point in marker.points:
                    mx, my = self.world_to_map(world_point.x, world_point.y)
                    # Ensure points are within map bounds
                    if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
                        polygon_points_map.append([mx, my])
                
                if len(polygon_points_map) < 3:
                    continue

                # Use OpenCV to fill the polygon with the 'occupied' value (100)
                cv2.fillPoly(costmap_grid, [np.array(polygon_points_map, dtype=np.int32)], 100)

            # 3. Inflate the obstacles by the robot's radius
            # cv2.dilate grows the white areas (obstacles).
            inflated_grid = cv2.dilate(costmap_grid, self.inflation_kernel)

            # 4. Publish the OccupancyGrid message
            self.publish_costmap(inflated_grid)

        except Exception as e:
            rospy.logerr(f"Error in costmap generation: {e}")

    def publish_costmap(self, grid_data):
        """Converts the NumPy grid to an OccupancyGrid message and publishes it."""
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame

        # --- Fill in the metadata (info) ---
        msg.info.resolution = self.resolution
        msg.info.width = self.width_cells
        msg.info.height = self.height_cells
        
        # Set the origin of the map
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0 # No rotation

        # --- Fill in the map data ---
        # The data is a 1D array in row-major order.
        # OccupancyGrid data is int8, so we cast our uint8 grid.
        msg.data = grid_data.flatten().astype(np.int8).tolist()

        self.costmap_pub.publish(msg)


def main():
    try:
        CostmapGeneratorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()