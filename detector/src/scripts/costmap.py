#!/usr/bin/env python3

# ROS1 Imports
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
import tf2_ros

# Standard Python Imports
import numpy as np
import cv2

class GlobalCostmapGenerator:
    def __init__(self):
        rospy.init_node('global_costmap_generator', anonymous=True)

        # --- Get parameters from ROS Param Server ---
        obstacle_topic = rospy.get_param('~obstacle_topic', '/detected_obstacles')
        costmap_topic = rospy.get_param('~costmap_topic', '/global_costmap')
        robot_pose_topic = rospy.get_param('~robot_pose_topic', '/robot_pose')
        
        # Costmap properties
        self.map_size_meters = rospy.get_param('~map_size', 20.0)  # Local window size around robot
        self.resolution = rospy.get_param('~resolution', 0.1)
        self.global_frame = rospy.get_param('~global_frame', 'odom')
        
        # Gradient Inflation Parameters
        self.inflation_radius = rospy.get_param('~inflation_radius', 0.4)
        self.gradient_end_radius = rospy.get_param('~gradient_end_radius', 1.2)
        self.max_cost = rospy.get_param('~max_cost', 100)
        
        # Obstacle Memory Parameter
        self.memory_duration = rospy.Duration(rospy.get_param('~memory_duration', 2.0))
        
        # Robot pose - initialize to something reasonable
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_pose_received = False
        
        # Store obstacles with timestamps in global frame
        self.persisted_obstacles = []  # List of (timestamp, marker) tuples
        
        # Pre-calculate map properties
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        
        self.gradient_width = self.gradient_end_radius - self.inflation_radius
        if self.gradient_width <= 0:
            rospy.logwarn("Gradient end radius must be greater than inflation radius. Disabling gradient.")
            self.gradient_width = 0
        
        # Store the current costmap for querying
        self.current_costmap = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        
        # ROS Subscribers and Publishers
        self.robot_pose_sub = rospy.Subscriber(robot_pose_topic, PoseStamped, 
                                               self.robot_pose_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, 
                                            self.obstacles_callback, queue_size=1)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)
        
        # Timer to publish costmap periodically even without new obstacles
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        rospy.loginfo("Global Costmap Generator initialized.")
        rospy.loginfo(f"Global frame: '{self.global_frame}'")
        rospy.loginfo(f"Local window size: {self.map_size_meters}m x {self.map_size_meters}m")
        rospy.loginfo(f"Resolution: {self.resolution}m/cell ({self.width_cells}x{self.height_cells} cells)")
        rospy.loginfo(f"Obstacle memory duration: {self.memory_duration.to_sec()}s")
        rospy.loginfo(f"Inflation radius: {self.inflation_radius}m, Gradient end: {self.gradient_end_radius}m")
    
    def robot_pose_callback(self, msg):
        """Update the robot's current position in the global frame."""
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        if not self.robot_pose_received:
            self.robot_pose_received = True
            rospy.loginfo(f"Robot pose received: ({self.robot_x:.2f}, {self.robot_y:.2f})")
    
    def world_to_map(self, wx, wy):
        """
        Converts world coordinates (in meters, in global frame) to map coordinates (in cells).
        The map origin is centered on the current robot position.
        """
        # Origin is at robot position minus half the map size
        origin_x = self.robot_x - self.map_size_meters / 2.0
        origin_y = self.robot_y - self.map_size_meters / 2.0
        
        mx = int((wx - origin_x) / self.resolution)
        my = int((wy - origin_y) / self.resolution)
        return mx, my
    
    def map_to_world(self, mx, my):
        """
        Converts map coordinates (in cells) to world coordinates (in meters, in global frame).
        """
        origin_x = self.robot_x - self.map_size_meters / 2.0
        origin_y = self.robot_y - self.map_size_meters / 2.0
        
        wx = mx * self.resolution + origin_x
        wy = my * self.resolution + origin_y
        return wx, wy
    
    def obstacles_callback(self, marker_array_msg):
        """Process new obstacles and generate the costmap."""
        now = rospy.Time.now()
        
        rospy.logdebug(f"Received MarkerArray with {len(marker_array_msg.markers)} markers")
        
        # Add new obstacles to the persisted list
        new_obstacles_added = 0
        if marker_array_msg.markers:
            for marker in marker_array_msg.markers:
                # Skip DELETE markers
                if marker.action == Marker.DELETE or marker.action == Marker.DELETEALL:
                    rospy.logdebug("Skipping DELETE marker")
                    continue
                
                # Log the marker details
                rospy.logdebug(f"Marker {marker.id}: type={marker.type}, frame={marker.header.frame_id}, "
                              f"pos=({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f}), "
                              f"scale=({marker.scale.x:.2f}, {marker.scale.y:.2f})")
                
                # Store the marker (already in global frame)
                self.persisted_obstacles.append((now, marker))
                new_obstacles_added += 1
        
        if new_obstacles_added > 0:
            rospy.loginfo_throttle(1.0, f"Added {new_obstacles_added} new obstacles. Total: {len(self.persisted_obstacles)}")
        
        # Remove old obstacles based on memory duration
        before_cleanup = len(self.persisted_obstacles)
        if self.memory_duration.to_sec() > 0:
            self.persisted_obstacles = [
                (ts, m) for ts, m in self.persisted_obstacles
                if (now - ts) < self.memory_duration
            ]
        after_cleanup = len(self.persisted_obstacles)
        
        if before_cleanup != after_cleanup:
            rospy.logdebug(f"Cleaned up {before_cleanup - after_cleanup} old obstacles")
    
    def timer_callback(self, event):
        """Periodically generate and publish the costmap."""
        if not self.robot_pose_received:
            rospy.logwarn_throttle(2.0, "Waiting for robot pose...")
            return
        
        self.generate_costmap()
    
    def generate_costmap(self):
        """Generate the costmap from all persisted obstacles."""
        try:
            # Initialize empty obstacle grid
            obstacle_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
            
            obstacles_drawn = 0
            obstacles_outside = 0
            
            # Draw all obstacles that fall within the current map window
            for timestamp, marker in self.persisted_obstacles:
                if marker.type == Marker.CUBE:
                    # For CUBE markers, extract the rectangular footprint
                    center_x = marker.pose.position.x
                    center_y = marker.pose.position.y
                    half_width = marker.scale.x / 2.0
                    half_height = marker.scale.y / 2.0
                    
                    # Create rectangle corners in world frame
                    corners_world = [
                        (center_x - half_width, center_y - half_height),
                        (center_x + half_width, center_y - half_height),
                        (center_x + half_width, center_y + half_height),
                        (center_x - half_width, center_y + half_height)
                    ]
                    
                    # Convert to map coordinates
                    corners_map = []
                    for wx, wy in corners_world:
                        mx, my = self.world_to_map(wx, wy)
                        # Add some tolerance for edges
                        if -10 <= mx < self.width_cells + 10 and -10 <= my < self.height_cells + 10:
                            # Clip to valid range
                            mx = max(0, min(mx, self.width_cells - 1))
                            my = max(0, min(my, self.height_cells - 1))
                            corners_map.append([mx, my])
                    
                    if len(corners_map) >= 3:
                        cv2.fillPoly(obstacle_grid, [np.array(corners_map, dtype=np.int32)], 255)
                        obstacles_drawn += 1
                        rospy.logdebug(f"Drew CUBE obstacle at ({center_x:.2f}, {center_y:.2f})")
                    else:
                        obstacles_outside += 1
                
                elif marker.type == Marker.CYLINDER:
                    # For CYLINDER markers, create a circular footprint
                    center_x = marker.pose.position.x
                    center_y = marker.pose.position.y
                    radius = marker.scale.x / 2.0
                    
                    cx, cy = self.world_to_map(center_x, center_y)
                    radius_cells = int(radius / self.resolution)
                    
                    # Check if center is reasonably close to map
                    if -radius_cells <= cx < self.width_cells + radius_cells and \
                       -radius_cells <= cy < self.height_cells + radius_cells:
                        # Clip coordinates to valid range
                        cx = max(0, min(cx, self.width_cells - 1))
                        cy = max(0, min(cy, self.height_cells - 1))
                        cv2.circle(obstacle_grid, (cx, cy), radius_cells, 255, -1)
                        obstacles_drawn += 1
                        rospy.logdebug(f"Drew CYLINDER obstacle at ({center_x:.2f}, {center_y:.2f})")
                    else:
                        obstacles_outside += 1
            
            if obstacles_drawn > 0:
                rospy.loginfo_throttle(2.0, f"Drew {obstacles_drawn} obstacles, {obstacles_outside} outside map window")
            
            # Apply distance transform and gradient inflation
            dist_transform_input = cv2.bitwise_not(obstacle_grid)
            dist_pixels = cv2.distanceTransform(dist_transform_input, cv2.DIST_L2, 5)
            dist_meters = dist_pixels * self.resolution
            
            costmap_grid = np.zeros_like(dist_meters, dtype=np.uint8)
            
            # Lethal zone
            lethal_mask = dist_meters <= self.inflation_radius
            costmap_grid[lethal_mask] = self.max_cost
            lethal_count = np.sum(lethal_mask)
            
            # Gradient zone
            gradient_count = 0
            if self.gradient_width > 0:
                gradient_mask = (dist_meters > self.inflation_radius) & (dist_meters < self.gradient_end_radius)
                gradient_distances = dist_meters[gradient_mask]
                slope = -self.max_cost / self.gradient_width
                intercept = self.max_cost * (self.gradient_end_radius / self.gradient_width)
                gradient_costs = slope * gradient_distances + intercept
                costmap_grid[gradient_mask] = gradient_costs.astype(np.uint8)
                gradient_count = np.sum(gradient_mask)
            
            if obstacles_drawn > 0:
                rospy.loginfo_throttle(2.0, f"Costmap stats: {lethal_count} lethal cells, {gradient_count} gradient cells")
            
            # Store current costmap for querying
            self.current_costmap = costmap_grid
            
            self.publish_costmap(costmap_grid)
        
        except Exception as e:
            rospy.logerr(f"Error in costmap generation: {e}")
            import traceback
            traceback.print_exc()
    
    def publish_costmap(self, grid_data):
        """Convert the NumPy grid to an OccupancyGrid message and publish it."""
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.global_frame
        msg.info.resolution = self.resolution
        msg.info.width = self.width_cells
        msg.info.height = self.height_cells
        msg.info.origin.position.x = self.robot_x - self.map_size_meters / 2.0
        msg.info.origin.position.y = self.robot_y - self.map_size_meters / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid_data.flatten().astype(np.int8).tolist()
        self.costmap_pub.publish(msg)
    
    def get_cost_at_world_point(self, world_x, world_y):
        """
        Query the cost at a given world point (x, y in global frame).
        
        Args:
            world_x: X coordinate in global frame (meters)
            world_y: Y coordinate in global frame (meters)
        
        Returns:
            int: Cost value [0-100], or -1 if point is outside the current map window
        """
        mx, my = self.world_to_map(world_x, world_y)
        
        # Check if point is within current map bounds
        if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
            return int(self.current_costmap[my, mx])
        else:
            return -1  # Point is outside the map window

def main():
    try:
        node = GlobalCostmapGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()