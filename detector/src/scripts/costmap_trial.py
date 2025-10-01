#!/usr/bin/env python3

# ROS1 Imports
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Pose, PointStamped

# <<< NEW: Import TF2 for coordinate transformations
import tf2_ros
import tf2_geometry_msgs # For transforming geometry_msgs

# Standard Python Imports
import numpy as np
import cv2
from collections import deque

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
        
        # Gradient Inflation Parameters
        self.inflation_radius = rospy.get_param('~inflation_radius', 0.4)
        self.gradient_end_radius = rospy.get_param('~gradient_end_radius', 1.2)
        self.max_cost = rospy.get_param('~max_cost', 100)

        # <<< NEW: Obstacle Memory Parameter
        self.memory_duration = rospy.Duration(rospy.get_param('~memory_duration', 5.0)) # seconds

        # <<< NEW: TF2 listener setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # <<< NEW: Data structure to store obstacles with timestamps
        # We will store (timestamp, marker) tuples
        self.persisted_obstacles = []

        # --- Pre-calculate map properties ---
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        self.origin_x = -self.map_size_meters / 2.0
        self.origin_y = -self.map_size_meters / 2.0
        
        self.gradient_width = self.gradient_end_radius - self.inflation_radius
        if self.gradient_width <= 0:
            rospy.logwarn("Gradient end radius must be greater than inflation radius. Disabling gradient.")
            self.gradient_width = 0

        # --- ROS Subscriber and Publisher ---
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, self.obstacles_callback, queue_size=1)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)

        rospy.loginfo("Costmap Generator Node with Memory and Gradient Inflation initialized.")
        rospy.loginfo(f"Obstacle memory duration: {self.memory_duration.to_sec()}s")

    def world_to_map(self, wx, wy):
        """Converts world coordinates (in meters, in map_frame) to map coordinates (in cells)."""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def obstacles_callback(self, marker_array_msg):
        """Processes new and remembered obstacles to generate the costmap."""
        now = rospy.Time.now()

        # --- Step 1: Transform new obstacles from base_link to odom frame and add to memory buffer ---
        for marker in marker_array_msg.markers:
            try:
                # Transform obstacle from base_link (at detection time) to odom (fixed world frame)
                # This "freezes" the obstacle in world coordinates
                transform_to_odom = self.tf_buffer.lookup_transform(
                    'odom',  # target: fixed world frame
                    marker.header.frame_id,  # source: base_link (where obstacle was detected)
                    marker.header.stamp,  # at the time it was detected
                    rospy.Duration(0.1)
                )
                
                # Create marker in odom frame
                marker_in_odom = Marker()
                marker_in_odom.header.frame_id = 'odom'
                marker_in_odom.header.stamp = marker.header.stamp
                marker_in_odom.action = marker.action
                marker_in_odom.type = marker.type
                marker_in_odom.points = []
                
                # Transform each point to odom
                for point in marker.points:
                    p_stamped = PointStamped()
                    p_stamped.header = marker.header
                    p_stamped.point = point
                    transformed_p = tf2_geometry_msgs.do_transform_point(p_stamped, transform_to_odom)
                    marker_in_odom.points.append(transformed_p.point)
                
                # Store in odom frame
                self.persisted_obstacles.append((now, marker_in_odom))
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(2.0, f"Could not transform new obstacle to odom: {e}")
                continue
        
        # --- Step 2: Filter out old obstacles from the memory buffer ---
        if self.memory_duration.to_sec() > 0:
            self.persisted_obstacles = [
                (ts, m) for ts, m in self.persisted_obstacles 
                if (now - ts) < self.memory_duration
            ]
        else: # If memory is zero, only use the current ones (already transformed to odom above)
            pass


        # --- Step 3: Transform all persisted obstacles (now in odom) into the current map_frame (base_link) ---
        obstacles_in_current_frame = []
        for timestamp, marker in self.persisted_obstacles:
            try:
                # Transform from odom (fixed) to current base_link position
                # Use rospy.Time(0) to get the latest transform (current robot position)
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,  # 'base_link' at current time
                    'odom',  # source frame (fixed world frame where obstacles are stored)
                    rospy.Time(0),  # Use latest transform (current time)
                    rospy.Duration(0.1)
                )
                
                # Create a new marker to hold the transformed points
                transformed_marker = Marker()
                transformed_marker.header.frame_id = self.map_frame
                transformed_marker.points = []
                transformed_marker.action = marker.action
                transformed_marker.type = marker.type
                
                # Transform each point from odom to current base_link
                for point in marker.points:
                    p_stamped = PointStamped()
                    p_stamped.header.frame_id = 'odom'
                    p_stamped.header.stamp = rospy.Time(0)
                    p_stamped.point = point
                    
                    transformed_point_stamped = tf2_geometry_msgs.do_transform_point(p_stamped, transform)
                    transformed_marker.points.append(transformed_point_stamped.point)

                obstacles_in_current_frame.append(transformed_marker)
            
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(2.0, f"Could not transform obstacle from odom to {self.map_frame}: {e}")
                continue

        # --- Step 4: Generate costmap from the combined (new + remembered) obstacles ---
        try:
            obstacle_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)

            # Rasterize each transformed obstacle polygon
            for marker in obstacles_in_current_frame:
                if marker.action != marker.ADD or marker.type != marker.LINE_STRIP or len(marker.points) < 3:
                    continue

                polygon_points_map = []
                for world_point in marker.points:
                    mx, my = self.world_to_map(world_point.x, world_point.y)
                    if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
                        polygon_points_map.append([mx, my])
                
                if len(polygon_points_map) < 3:
                    continue
                
                cv2.fillPoly(obstacle_grid, [np.array(polygon_points_map, dtype=np.int32)], 255)

            # --- The rest of the logic is unchanged ---

            dist_transform_input = cv2.bitwise_not(obstacle_grid)
            dist_pixels = cv2.distanceTransform(dist_transform_input, cv2.DIST_L2, 5)
            dist_meters = dist_pixels * self.resolution
            
            costmap_grid = np.zeros_like(dist_meters, dtype=np.uint8)
            
            lethal_mask = dist_meters <= self.inflation_radius
            costmap_grid[lethal_mask] = self.max_cost
            
            if self.gradient_width > 0:
                gradient_mask = (dist_meters > self.inflation_radius) & (dist_meters < self.gradient_end_radius)
                gradient_distances = dist_meters[gradient_mask]
                slope = -self.max_cost / self.gradient_width
                intercept = self.max_cost * (self.gradient_end_radius / self.gradient_width)
                gradient_costs = slope * gradient_distances + intercept
                costmap_grid[gradient_mask] = gradient_costs.astype(np.uint8)

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