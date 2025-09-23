#!/usr/bin/env python3

# ROS1 Imports
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Pose

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
        self.map_size_meters = rospy.get_param('~map_size', 20.0)
        self.resolution = rospy.get_param('~resolution', 0.1)
        self.map_frame = rospy.get_param('~map_frame', 'base_link')
        
        # --- NEW: Gradient Inflation Parameters ---
        # The radius at which cost is at its maximum (lethal cost)
        self.inflation_radius = rospy.get_param('~inflation_radius', 0.7)
        # The radius at which the cost gradient ends (cost becomes zero)
        self.gradient_end_radius = rospy.get_param('~gradient_end_radius', 1.2)
        # Max cost value in the OccupancyGrid (0-100)
        self.max_cost = rospy.get_param('~max_cost', 100)

        # --- Pre-calculate map and gradient properties ---
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        self.origin_x = -self.map_size_meters / 2.0
        self.origin_y = -self.map_size_meters / 2.0
        
        # Calculate the width of the gradient in meters
        self.gradient_width = self.gradient_end_radius - self.inflation_radius
        if self.gradient_width <= 0:
            rospy.logwarn("Gradient end radius must be greater than inflation radius. Disabling gradient.")
            self.gradient_width = 0 # Disable gradient if params are invalid

        # --- ROS Subscriber and Publisher ---
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, self.obstacles_callback, queue_size=1)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)

        rospy.loginfo("Costmap Generator Node with Gradient Inflation initialized.")
        rospy.loginfo(f"Inflation Radius (max cost): {self.inflation_radius}m")
        rospy.loginfo(f"Gradient End Radius (zero cost): {self.gradient_end_radius}m")

    def world_to_map(self, wx, wy):
        """Converts world coordinates (in meters, in map_frame) to map coordinates (in cells)."""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def obstacles_callback(self, marker_array_msg):
        """Processes obstacle markers and generates the gradient costmap."""
        try:
            # 1. Create a grid to mark the raw obstacle locations
            obstacle_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)

            # 2. Rasterize each obstacle polygon onto the grid
            for marker in marker_array_msg.markers:
                if marker.action != marker.ADD or marker.type != marker.LINE_STRIP or len(marker.points) < 3:
                    continue

                polygon_points_map = []
                for world_point in marker.points:
                    mx, my = self.world_to_map(world_point.x, world_point.y)
                    if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
                        polygon_points_map.append([mx, my])
                
                if len(polygon_points_map) < 3:
                    continue
                
                # Fill the polygon area. We'll mark it with 255 to distinguish from free space (0).
                cv2.fillPoly(obstacle_grid, [np.array(polygon_points_map, dtype=np.int32)], 255)

            # 3. Use Distance Transform to calculate the distance from every cell to the nearest obstacle
            # We want distances from non-obstacle points to obstacle points.
            # `distanceTransform` computes distance from a zero pixel to the nearest non-zero pixel.
            # So, we invert our grid: obstacles become 0, free space becomes 255.
            dist_transform_input = cv2.bitwise_not(obstacle_grid)
            
            # Calculate Euclidean distance for each pixel to the nearest zero (our obstacles)
            # The result is a float32 array where each value is the distance in pixels.
            dist_pixels = cv2.distanceTransform(dist_transform_input, cv2.DIST_L2, 5)

            # Convert pixel distances to meter distances
            dist_meters = dist_pixels * self.resolution
            
            # 4. Create the final costmap with the gradient
            # Initialize with zeros (free space)
            costmap_grid = np.zeros_like(dist_meters, dtype=np.uint8)
            
            # Apply cost function based on distance
            # Zone 1: Lethal cost (within inflation_radius)
            lethal_mask = dist_meters <= self.inflation_radius
            costmap_grid[lethal_mask] = self.max_cost
            
            # Zone 2: Gradient cost (between inflation_radius and gradient_end_radius)
            if self.gradient_width > 0:
                gradient_mask = (dist_meters > self.inflation_radius) & (dist_meters < self.gradient_end_radius)
                
                # Get distances only in the gradient zone
                gradient_distances = dist_meters[gradient_mask]
                
                # Linearly map distance to cost:
                # Cost = Max_Cost * (1 - (current_dist - inflation_radius) / gradient_width)
                slope = -self.max_cost / self.gradient_width
                intercept = self.max_cost * (self.gradient_end_radius / self.gradient_width)
                gradient_costs = slope * gradient_distances + intercept

                costmap_grid[gradient_mask] = gradient_costs.astype(np.uint8)

            # Zone 3 (outside gradient_end_radius) is already 0.

            # 5. Publish the resulting OccupancyGrid
            self.publish_costmap(costmap_grid)

        except Exception as e:
            rospy.logerr(f"Error in costmap generation: {e}")

    def publish_costmap(self, grid_data):
        """Converts the NumPy grid to an OccupancyGrid message and publishes it."""
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.map_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.width_cells
        msg.info.height = self.height_cells
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
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