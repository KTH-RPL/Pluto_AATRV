#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
import tf2_ros

import numpy as np
import cv2

class GlobalCostmapGenerator:
    def __init__(self):
        rospy.init_node('global_costmap_generator', anonymous=True)

        obstacle_topic = rospy.get_param('~obstacle_topic', '/detected_obstacles')
        costmap_topic = rospy.get_param('~costmap_topic', '/global_costmap')
        robot_pose_topic = rospy.get_param('~robot_pose_topic', '/robot_pose')
        
        self.map_size_meters = rospy.get_param('~map_size', 20.0)
        self.resolution = rospy.get_param('~resolution', 0.1)
        
        self.inflation_radius = rospy.get_param('~inflation_radius', 0.4)
        self.gradient_end_radius = rospy.get_param('~gradient_end_radius', 1.2)
        self.max_cost = rospy.get_param('~max_cost', 100)
        
        self.memory_duration = rospy.Duration(rospy.get_param('~memory_duration', 2.0))
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_pose_received = False
        
        self.persisted_obstacles = []
        
        self.width_cells = int(self.map_size_meters / self.resolution)
        self.height_cells = int(self.map_size_meters / self.resolution)
        
        self.gradient_width = self.gradient_end_radius - self.inflation_radius
        if self.gradient_width <= 0:
            rospy.logwarn("Gradient end radius must be greater than inflation radius. Disabling gradient.")
            self.gradient_width = 0
        
        self.current_costmap = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
        
        self.robot_pose_sub = rospy.Subscriber(robot_pose_topic, PoseStamped, 
                                               self.robot_pose_callback, queue_size=1)
        self.obstacle_sub = rospy.Subscriber(obstacle_topic, MarkerArray, 
                                            self.obstacles_callback, queue_size=1)
        self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)
        
        self.publish_timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)
        
        rospy.loginfo("Global Costmap Generator initialized (base_link frame).")
        rospy.loginfo(f"Local window size: {self.map_size_meters}m x {self.map_size_meters}m")
        rospy.loginfo(f"Resolution: {self.resolution}m/cell ({self.width_cells}x{self.height_cells} cells)")
        rospy.loginfo(f"Obstacle memory duration: {self.memory_duration.to_sec()}s")
        rospy.loginfo(f"Inflation radius: {self.inflation_radius}m, Gradient end: {self.gradient_end_radius}m")
    
    def robot_pose_callback(self, msg):
        self.robot_x = msg.pose.position.x
        self.robot_y = msg.pose.position.y
        self.robot_theta = msg.pose.orientation.z
        if not self.robot_pose_received:
            self.robot_pose_received = True
            rospy.loginfo(f"Robot pose received: ({self.robot_x:.2f}, {self.robot_y:.2f}, {self.robot_theta:.2f})")
    
    def world_to_baselink(self, wx, wy):
        dx = wx - self.robot_x
        dy = wy - self.robot_y
        
        cos_theta = np.cos(self.robot_theta)
        sin_theta = np.sin(self.robot_theta)
        
        bx = dx * cos_theta + dy * sin_theta
        by = -dx * sin_theta + dy * cos_theta
        
        return bx, by
    
    def baselink_to_map(self, bx, by):
        origin_x = -self.map_size_meters / 2.0
        origin_y = -self.map_size_meters / 2.0
        
        mx = int((bx - origin_x) / self.resolution)
        my = int((by - origin_y) / self.resolution)
        return mx, my
    
    def map_to_baselink(self, mx, my):
        origin_x = -self.map_size_meters / 2.0
        origin_y = -self.map_size_meters / 2.0
        
        bx = mx * self.resolution + origin_x
        by = my * self.resolution + origin_y
        return bx, by
    
    def obstacles_callback(self, marker_array_msg):
        now = rospy.Time.now()
        
        rospy.logdebug(f"Received MarkerArray with {len(marker_array_msg.markers)} markers")
        
        new_obstacles_added = 0
        if marker_array_msg.markers:
            for marker in marker_array_msg.markers:
                if marker.action == Marker.DELETE or marker.action == Marker.DELETEALL:
                    rospy.logdebug("Skipping DELETE marker")
                    continue
                
                rospy.logdebug(f"Marker {marker.id}: type={marker.type}, frame={marker.header.frame_id}, "
                              f"pos=({marker.pose.position.x:.2f}, {marker.pose.position.y:.2f}), "
                              f"scale=({marker.scale.x:.2f}, {marker.scale.y:.2f})")
                
                self.persisted_obstacles.append((now, marker))
                new_obstacles_added += 1
        
        if new_obstacles_added > 0:
            rospy.loginfo_throttle(1.0, f"Added {new_obstacles_added} new obstacles. Total: {len(self.persisted_obstacles)}")
        
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
        if not self.robot_pose_received:
            rospy.logwarn_throttle(2.0, "Waiting for robot pose...")
            return
        
        self.generate_costmap()
    
    def generate_costmap(self):
        try:
            obstacle_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.uint8)
            
            obstacles_drawn = 0
            obstacles_outside = 0
            
            for timestamp, marker in self.persisted_obstacles:
                center_x_world = marker.pose.position.x
                center_y_world = marker.pose.position.y
                
                bx, by = self.world_to_baselink(center_x_world, center_y_world)
                
                if marker.type == Marker.CUBE:
                    half_width = marker.scale.x / 2.0
                    half_height = marker.scale.y / 2.0
                    
                    corners_baselink = [
                        (bx - half_width, by - half_height),
                        (bx + half_width, by - half_height),
                        (bx + half_width, by + half_height),
                        (bx - half_width, by + half_height)
                    ]
                    
                    corners_map = []
                    for bx_corner, by_corner in corners_baselink:
                        mx, my = self.baselink_to_map(bx_corner, by_corner)
                        if -10 <= mx < self.width_cells + 10 and -10 <= my < self.height_cells + 10:
                            mx = max(0, min(mx, self.width_cells - 1))
                            my = max(0, min(my, self.height_cells - 1))
                            corners_map.append([mx, my])
                    
                    if len(corners_map) >= 3:
                        cv2.fillPoly(obstacle_grid, [np.array(corners_map, dtype=np.int32)], 255)
                        obstacles_drawn += 1
                        rospy.logdebug(f"Drew CUBE obstacle at baselink ({bx:.2f}, {by:.2f})")
                    else:
                        obstacles_outside += 1
                
                elif marker.type == Marker.CYLINDER:
                    radius = marker.scale.x / 2.0
                    
                    cx, cy = self.baselink_to_map(bx, by)
                    radius_cells = int(radius / self.resolution)
                    
                    if -radius_cells <= cx < self.width_cells + radius_cells and \
                       -radius_cells <= cy < self.height_cells + radius_cells:
                        cx = max(0, min(cx, self.width_cells - 1))
                        cy = max(0, min(cy, self.height_cells - 1))
                        cv2.circle(obstacle_grid, (cx, cy), radius_cells, 255, -1)
                        obstacles_drawn += 1
                        rospy.logdebug(f"Drew CYLINDER obstacle at baselink ({bx:.2f}, {by:.2f})")
                    else:
                        obstacles_outside += 1
            
            if obstacles_drawn > 0:
                rospy.loginfo_throttle(2.0, f"Drew {obstacles_drawn} obstacles, {obstacles_outside} outside map window")
            
            dist_transform_input = cv2.bitwise_not(obstacle_grid)
            dist_pixels = cv2.distanceTransform(dist_transform_input, cv2.DIST_L2, 5)
            dist_meters = dist_pixels * self.resolution
            
            costmap_grid = np.zeros_like(dist_meters, dtype=np.uint8)
            
            lethal_mask = dist_meters <= self.inflation_radius
            costmap_grid[lethal_mask] = self.max_cost
            lethal_count = np.sum(lethal_mask)
            
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
            
            self.current_costmap = costmap_grid
            
            self.publish_costmap(costmap_grid)
        
        except Exception as e:
            rospy.logerr(f"Error in costmap generation: {e}")
            import traceback
            traceback.print_exc()
    
    def publish_costmap(self, grid_data):
        msg = OccupancyGrid()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.info.resolution = self.resolution
        msg.info.width = self.width_cells
        msg.info.height = self.height_cells
        msg.info.origin.position.x = -self.map_size_meters / 2.0
        msg.info.origin.position.y = -self.map_size_meters / 2.0
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid_data.flatten().astype(np.int8).tolist()
        self.costmap_pub.publish(msg)
    
    def get_cost_at_baselink_point(self, bx, by):
        mx, my = self.baselink_to_map(bx, by)
        
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

