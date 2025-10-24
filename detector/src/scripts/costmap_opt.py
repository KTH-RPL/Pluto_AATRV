#!/usr/bin/env python3

# ROS1 Imports
import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped

# Standard Python Imports
import numpy as np
import cv2
from numba import jit, prange
from collections import deque

@jit(nopython=True, cache=True, fastmath=True)
def apply_gradient_inflation_numba(dist_meters, inflation_radius, gradient_end_radius, 
                                   max_cost, gradient_width):
    """
    Numba-optimized gradient inflation calculation.
    Processes the entire distance transform in parallel.
    """
    height, width = dist_meters.shape
    costmap = np.zeros((height, width), dtype=np.uint8)
    
    slope = -max_cost / gradient_width if gradient_width > 0 else 0
    intercept = max_cost * (gradient_end_radius / gradient_width) if gradient_width > 0 else 0
    
    for i in prange(height):
        for j in prange(width):
            dist = dist_meters[i, j]
            
            if dist <= inflation_radius:
                costmap[i, j] = max_cost
            elif gradient_width > 0 and dist < gradient_end_radius:
                cost = slope * dist + intercept
                costmap[i, j] = max(0, min(int(cost), max_cost))
    
    return costmap

@jit(nopython=True, cache=True)
def world_to_map_batch(world_coords, origin_x, origin_y, resolution, width_cells, height_cells):
    """
    Batch convert world coordinates to map coordinates.
    Returns mask of valid points and their map coordinates.
    """
    n_points = len(world_coords)
    map_coords = np.zeros((n_points, 2), dtype=np.int32)
    valid_mask = np.zeros(n_points, dtype=np.bool_)
    
    for i in range(n_points):
        wx, wy = world_coords[i]
        mx = int((wx - origin_x) / resolution)
        my = int((wy - origin_y) / resolution)
        
        # Check bounds with tolerance
        if -10 <= mx < width_cells + 10 and -10 <= my < height_cells + 10:
            # Clip to valid range
            mx = max(0, min(mx, width_cells - 1))
            my = max(0, min(my, height_cells - 1))
            map_coords[i, 0] = mx
            map_coords[i, 1] = my
            valid_mask[i] = True
    
    return map_coords, valid_mask


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
        
        # Rate limiting parameter (Hz)
        self.update_rate = rospy.get_param('~update_rate', 10.0)  # Default 10 Hz
        
        # Robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_pose_received = False
        
        # Use deque for efficient memory management (O(1) popleft)
        max_obstacles = int(self.memory_duration.to_sec() * 100)  # Estimate capacity
        self.persisted_obstacles = deque(maxlen=max_obstacles)
        
        # Pre-calculate map properties
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        
        self.gradient_width = self.gradient_end_radius - self.inflation_radius
        if self.gradient_width <= 0:
            rospy.logwarn("Gradient end radius must be greater than inflation radius. Disabling gradient.")
            self.gradient_width = 0
        
        # Pre-allocate arrays for reuse (avoid repeated allocations)
        self.current_costmap = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        self.obstacle_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        self.dist_transform_input = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        
        # Flag to track if new obstacles arrived
        self.new_obstacles_flag = False
        self.last_update_time = rospy.Time.now()
        
        # ROS Subscribers and Publishers
        self.robot_pose_sub = rospy.Subscriber(robot_pose_topic, PoseStamped, 
                                               self.robot_pose_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, 
                                            self.obstacles_callback, queue_size=10)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)
        
        # Timer based on update_rate parameter
        timer_period = 1.0 / self.update_rate
        self.publish_timer = rospy.Timer(rospy.Duration(timer_period), self.timer_callback)
        
        rospy.loginfo("Optimized Global Costmap Generator initialized.")
        rospy.loginfo(f"Update rate: {self.update_rate} Hz")
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
    
    def obstacles_callback(self, marker_array_msg):
        """Process new obstacles with minimal overhead."""
        now = rospy.Time.now()
        
        # Add new obstacles (skip DELETE markers)
        new_count = 0
        for marker in marker_array_msg.markers:
            if marker.action not in (Marker.DELETE, Marker.DELETEALL):
                self.persisted_obstacles.append((now, marker))
                new_count += 1
        
        if new_count > 0:
            self.new_obstacles_flag = True
            rospy.logdebug(f"Added {new_count} obstacles. Total: {len(self.persisted_obstacles)}")
        
        # Remove old obstacles (only if memory duration is set)
        if self.memory_duration.to_sec() > 0:
            # Remove from left while old
            while self.persisted_obstacles and (now - self.persisted_obstacles[0][0]) >= self.memory_duration:
                self.persisted_obstacles.popleft()
    
    def timer_callback(self, event):
        """Rate-limited costmap generation."""
        if not self.robot_pose_received:
            rospy.logwarn_throttle(2.0, "Waiting for robot pose...")
            return
        
        # Check if enough time has passed since last update
        now = rospy.Time.now()
        time_since_update = (now - self.last_update_time).to_sec()
        min_period = 1.0 / self.update_rate
        
        if time_since_update < min_period and not self.new_obstacles_flag:
            return  # Skip update to maintain rate limit
        
        self.generate_costmap()
        self.last_update_time = now
        self.new_obstacles_flag = False
    
    def generate_costmap(self):
        """Optimized costmap generation using pre-allocated arrays and Numba."""
        try:
            # Reset obstacle grid (reuse array)
            self.obstacle_grid.fill(0)
            
            origin_x = self.robot_x - self.map_size_meters / 2.0
            origin_y = self.robot_y - self.map_size_meters / 2.0
            
            obstacles_drawn = 0
            
            # Draw all obstacles
            for timestamp, marker in self.persisted_obstacles:
                if marker.type == Marker.CUBE:
                    center_x = marker.pose.position.x
                    center_y = marker.pose.position.y
                    half_width = marker.scale.x / 2.0
                    half_height = marker.scale.y / 2.0
                    
                    # Create corners as numpy array for batch processing
                    corners_world = np.array([
                        [center_x - half_width, center_y - half_height],
                        [center_x + half_width, center_y - half_height],
                        [center_x + half_width, center_y + half_height],
                        [center_x - half_width, center_y + half_height]
                    ], dtype=np.float32)
                    
                    # Batch convert to map coordinates
                    corners_map, valid_mask = world_to_map_batch(
                        corners_world, origin_x, origin_y, self.resolution,
                        self.width_cells, self.height_cells
                    )
                    
                    if np.sum(valid_mask) >= 3:
                        valid_corners = corners_map[valid_mask]
                        cv2.fillPoly(self.obstacle_grid, [valid_corners], 255)
                        obstacles_drawn += 1
                
                elif marker.type == Marker.CYLINDER:
                    center_x = marker.pose.position.x
                    center_y = marker.pose.position.y
                    radius = marker.scale.x / 2.0
                    
                    cx, cy = self.world_to_map(center_x, center_y)
                    radius_cells = int(radius / self.resolution)
                    
                    if -radius_cells <= cx < self.width_cells + radius_cells and \
                       -radius_cells <= cy < self.height_cells + radius_cells:
                        cx = max(0, min(cx, self.width_cells - 1))
                        cy = max(0, min(cy, self.height_cells - 1))
                        cv2.circle(self.obstacle_grid, (cx, cy), radius_cells, 255, -1)
                        obstacles_drawn += 1
            
            # Distance transform (reuse input array)
            np.bitwise_not(self.obstacle_grid, out=self.dist_transform_input)
            dist_pixels = cv2.distanceTransform(self.dist_transform_input, cv2.DIST_L2, 5)
            dist_meters = dist_pixels * self.resolution
            
            # Apply gradient inflation using Numba (parallelized)
            self.current_costmap = apply_gradient_inflation_numba(
                dist_meters, self.inflation_radius, self.gradient_end_radius,
                self.max_cost, self.gradient_width
            )
            
            # Publish
            self.publish_costmap(self.current_costmap)
            
            if obstacles_drawn > 0:
                rospy.logdebug(f"Generated costmap with {obstacles_drawn} obstacles")
        
        except Exception as e:
            rospy.logerr(f"Error in costmap generation: {e}")
    
    def publish_costmap(self, grid_data):
        """Publish OccupancyGrid message."""
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
        
        # Direct conversion without intermediate list (more efficient)
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