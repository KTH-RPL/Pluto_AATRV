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
import time

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# ==============================================================================
# Numba-optimized helper functions
# ==============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _apply_lethal_and_gradient(dist_meters, costmap_grid, inflation_radius, 
                                   gradient_end_radius, max_cost, gradient_width):
        """Fast lethal and gradient computation."""
        lethal_count = 0
        gradient_count = 0
        
        for i in range(dist_meters.shape[0]):
            for j in range(dist_meters.shape[1]):
                d = dist_meters[i, j]
                
                if d <= inflation_radius:
                    costmap_grid[i, j] = max_cost
                    lethal_count += 1
                elif gradient_width > 0 and d < gradient_end_radius:
                    slope = -max_cost / gradient_width
                    intercept = max_cost * (gradient_end_radius / gradient_width)
                    cost = int(slope * d + intercept)
                    costmap_grid[i, j] = max(0, min(cost, max_cost))
                    gradient_count += 1
        
        return lethal_count, gradient_count
    
    @jit(nopython=True)
    def _world_to_map_batch(wx_array, wy_array, robot_x, robot_y, map_size, resolution):
        """Fast batch conversion of world to map coordinates."""
        origin_x = robot_x - map_size / 2.0
        origin_y = robot_y - map_size / 2.0
        
        mx_array = np.zeros_like(wx_array, dtype=np.int32)
        my_array = np.zeros_like(wy_array, dtype=np.int32)
        
        for i in range(len(wx_array)):
            mx_array[i] = int((wx_array[i] - origin_x) / resolution)
            my_array[i] = int((wy_array[i] - origin_y) / resolution)
        
        return mx_array, my_array

else:
    def _apply_lethal_and_gradient(dist_meters, costmap_grid, inflation_radius, 
                                   gradient_end_radius, max_cost, gradient_width):
        """Fallback lethal and gradient computation."""
        lethal_mask = dist_meters <= inflation_radius
        costmap_grid[lethal_mask] = max_cost
        lethal_count = np.sum(lethal_mask)
        
        gradient_count = 0
        if gradient_width > 0:
            gradient_mask = (dist_meters > inflation_radius) & (dist_meters < gradient_end_radius)
            gradient_distances = dist_meters[gradient_mask]
            slope = -max_cost / gradient_width
            intercept = max_cost * (gradient_end_radius / gradient_width)
            gradient_costs = (slope * gradient_distances + intercept).astype(np.uint8)
            costmap_grid[gradient_mask] = gradient_costs
            gradient_count = np.sum(gradient_mask)
        
        return lethal_count, gradient_count
    
    def _world_to_map_batch(wx_array, wy_array, robot_x, robot_y, map_size, resolution):
        """Fallback batch conversion."""
        origin_x = robot_x - map_size / 2.0
        origin_y = robot_y - map_size / 2.0
        
        mx_array = ((wx_array - origin_x) / resolution).astype(np.int32)
        my_array = ((wy_array - origin_y) / resolution).astype(np.int32)
        
        return mx_array, my_array

class GlobalCostmapGenerator:
    def __init__(self):
        rospy.init_node('global_costmap_generator', anonymous=True)

        # --- Get parameters from ROS Param Server ---
        obstacle_topic = rospy.get_param('~obstacle_topic', '/detected_obstacles')
        costmap_topic = rospy.get_param('~costmap_topic', '/global_costmap')
        robot_pose_topic = rospy.get_param('~robot_pose_topic', '/robot_pose')
        
        # Costmap properties
        self.map_size_meters = rospy.get_param('~map_size', 20.0)
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
        self.persisted_obstacles = []
        self.new_obstacles_received = False
        
        # Pre-calculate map properties
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        
        self.gradient_width = self.gradient_end_radius - self.inflation_radius
        if self.gradient_width <= 0:
            rospy.logwarn("Gradient end radius must be greater than inflation radius. Disabling gradient.")
            self.gradient_width = 0
        
        # Store the current costmap for querying
        self.current_costmap = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        
        # Performance optimization: only regenerate on new obstacles
        self.last_published_time = 0.0
        self.min_publish_interval = rospy.get_param('~min_publish_interval', 0.05)  # Fallback publish interval
        
        # ROS Subscribers and Publishers
        self.robot_pose_sub = rospy.Subscriber(robot_pose_topic, PoseStamped, 
                                               self.robot_pose_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, 
                                            self.obstacles_callback, queue_size=1)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)
        
        # Lightweight timer for occasional publishes (fallback)
        self.publish_timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)
        
        rospy.loginfo("Global Costmap Generator initialized.")
        rospy.loginfo(f"Numba optimization: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
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
        """Converts world coordinates to map coordinates."""
        origin_x = self.robot_x - self.map_size_meters / 2.0
        origin_y = self.robot_y - self.map_size_meters / 2.0
        
        mx = int((wx - origin_x) / self.resolution)
        my = int((wy - origin_y) / self.resolution)
        return mx, my
    
    def map_to_world(self, mx, my):
        """Converts map coordinates to world coordinates."""
        origin_x = self.robot_x - self.map_size_meters / 2.0
        origin_y = self.robot_y - self.map_size_meters / 2.0
        
        wx = mx * self.resolution + origin_x
        wy = my * self.resolution + origin_y
        return wx, wy
    
    def obstacles_callback(self, marker_array_msg):
        """Process new obstacles and trigger costmap generation."""
        now = rospy.Time.now()
        
        rospy.logdebug(f"Received MarkerArray with {len(marker_array_msg.markers)} markers")
        
        # Add new obstacles to the persisted list
        new_obstacles_added = 0
        if marker_array_msg.markers:
            for marker in marker_array_msg.markers:
                if marker.action == Marker.DELETE or marker.action == Marker.DELETEALL:
                    rospy.logdebug("Skipping DELETE marker")
                    continue
                
                self.persisted_obstacles.append((now, marker))
                new_obstacles_added += 1
        
        if new_obstacles_added > 0:
            rospy.loginfo_throttle(1.0, f"Added {new_obstacles_added} new obstacles. Total: {len(self.persisted_obstacles)}")
            self.new_obstacles_received = True
        
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
        
        # Generate costmap immediately on new obstacles
        self.generate_costmap()
    
    def timer_callback(self, event):
        """Periodic fallback publish (low frequency)."""
        if not self.robot_pose_received:
            return
        
        current_time = time.time()
        if current_time - self.last_published_time > self.min_publish_interval:
            self.publish_costmap(self.current_costmap)
    
    def generate_costmap(self):
        """Generate the costmap from all persisted obstacles with optimizations."""
        try:
            if not self.robot_pose_received:
                return
            
            obstacle_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
            
            obstacles_drawn = 0
            obstacles_outside = 0
            
            # Draw obstacles using vectorized operations where possible
            for timestamp, marker in self.persisted_obstacles:
                if marker.type == Marker.CUBE:
                    # For CUBE markers: extract and draw rectangular footprint
                    center_x = marker.pose.position.x
                    center_y = marker.pose.position.y
                    half_width = marker.scale.x / 2.0
                    half_height = marker.scale.y / 2.0
                    
                    # Batch convert all corners at once (more cache-friendly)
                    corners_world_x = np.array([
                        center_x - half_width, center_x + half_width,
                        center_x + half_width, center_x - half_width
                    ], dtype=np.float32)
                    corners_world_y = np.array([
                        center_y - half_height, center_y - half_height,
                        center_y + half_height, center_y + half_height
                    ], dtype=np.float32)
                    
                    if NUMBA_AVAILABLE:
                        mx_arr, my_arr = _world_to_map_batch(
                            corners_world_x, corners_world_y,
                            self.robot_x, self.robot_y,
                            self.map_size_meters, self.resolution
                        )
                    else:
                        mx_arr = ((corners_world_x - (self.robot_x - self.map_size_meters / 2.0)) / self.resolution).astype(np.int32)
                        my_arr = ((corners_world_y - (self.robot_y - self.map_size_meters / 2.0)) / self.resolution).astype(np.int32)
                    
                    # Check bounds and build corners list
                    valid_corners = []
                    for mx, my in zip(mx_arr, my_arr):
                        if -10 <= mx < self.width_cells + 10 and -10 <= my < self.height_cells + 10:
                            mx = max(0, min(mx, self.width_cells - 1))
                            my = max(0, min(my, self.height_cells - 1))
                            valid_corners.append([mx, my])
                    
                    if len(valid_corners) >= 3:
                        cv2.fillPoly(obstacle_grid, [np.array(valid_corners, dtype=np.int32)], 255)
                        obstacles_drawn += 1
                    else:
                        obstacles_outside += 1
                
                elif marker.type == Marker.CYLINDER:
                    # For CYLINDER markers: draw circular footprint
                    center_x = marker.pose.position.x
                    center_y = marker.pose.position.y
                    radius = marker.scale.x / 2.0
                    
                    cx, cy = self.world_to_map(center_x, center_y)
                    radius_cells = int(radius / self.resolution)
                    
                    if -radius_cells <= cx < self.width_cells + radius_cells and \
                       -radius_cells <= cy < self.height_cells + radius_cells:
                        cx = max(0, min(cx, self.width_cells - 1))
                        cy = max(0, min(cy, self.height_cells - 1))
                        cv2.circle(obstacle_grid, (cx, cy), radius_cells, 255, -1)
                        obstacles_drawn += 1
                    else:
                        obstacles_outside += 1
            
            if obstacles_drawn > 0:
                rospy.loginfo_throttle(2.0, f"Drew {obstacles_drawn} obstacles, {obstacles_outside} outside map window")
            
            # Distance transform (already fast in OpenCV)
            dist_transform_input = cv2.bitwise_not(obstacle_grid)
            dist_pixels = cv2.distanceTransform(dist_transform_input, cv2.DIST_L2, 5)
            dist_meters = dist_pixels * self.resolution
            
            # Initialize costmap
            costmap_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
            
            # Apply lethal and gradient using Numba (if available) for fast computation
            lethal_count, gradient_count = _apply_lethal_and_gradient(
                dist_meters, costmap_grid, self.inflation_radius,
                self.gradient_end_radius, self.max_cost, self.gradient_width
            )
            
            if obstacles_drawn > 0:
                rospy.loginfo_throttle(2.0, f"Costmap stats: {lethal_count} lethal cells, {gradient_count} gradient cells")
            
            # Store current costmap
            self.current_costmap = costmap_grid
            
            # Publish immediately
            self.publish_costmap(costmap_grid)
        
        except Exception as e:
            rospy.logerr(f"Error in costmap generation: {e}")
            import traceback
            traceback.print_exc()
    
    def publish_costmap(self, grid_data):
        """Convert the NumPy grid to an OccupancyGrid message and publish it."""
        self.last_published_time = time.time()
        
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
        
        # Optimized: flatten once, convert once
        msg.data = grid_data.flatten().astype(np.int8).tolist()
        self.costmap_pub.publish(msg)
    
    def get_cost_at_world_point(self, world_x, world_y):
        """Query the cost at a given world point."""
        mx, my = self.world_to_map(world_x, world_y)
        
        if 0 <= mx < self.width_cells and 0 <= my < self.height_cells:
            return int(self.current_costmap[my, mx])
        else:
            return -1

def main():
    try:
        node = GlobalCostmapGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()