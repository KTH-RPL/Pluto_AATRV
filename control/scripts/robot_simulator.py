#!/usr/bin/env python3

import rospy
import tf2_ros
import math
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import tf2_geometry_msgs

class RobotSimulator:
    def __init__(self):
        rospy.init_node('robot_simulator', anonymous=False)
        
        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0
        
        # Get parameters
        self.rate_hz = rospy.get_param('~rate', 50.0)
        self.robot_radius = rospy.get_param('~robot_radius', 0.5)
        self.initial_x = rospy.get_param('~initial_x', 0.0)
        self.initial_y = rospy.get_param('~initial_y', 0.0)
        self.initial_theta = rospy.get_param('~initial_theta', 0.0)
        
        # Initialize position
        self.x = self.initial_x
        self.y = self.initial_y
        self.theta = self.initial_theta
        
        # Publishers
        self.robot_pose_pub = rospy.Publisher('/robot_pose', PoseStamped, queue_size=10)
        self.robot_marker_pub = rospy.Publisher('/robot_marker', Marker, queue_size=10)
        self.obstacle_pub = rospy.Publisher('/detected_obstacles', MarkerArray, queue_size=10)
        
        # Subscribers
        self.cmd_vel_sub = rospy.Subscriber('/atrv/cmd_vel', Twist, self.cmd_vel_callback)
        
        # TF broadcaster and buffer for transformations
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Create static obstacles
        self.create_obstacles()
        
        rospy.loginfo("Robot Simulator initialized at (%.2f, %.2f, %.2f)", 
                     self.x, self.y, self.theta)
        
    def create_obstacles(self):
        """Create static obstacles along the path"""
        self.obstacles = []
        
        # Small obstacle directly on path to test avoidance
        obs_small = {
            'center': [3.5, 0.5],  # Changed to list so it's mutable
            'size': 0.5,
            'type': 'square',
            'velocity': (-0.2, 0.0)  # Moving at 0.5 m/s in x, 0.3 m/s in y
        }
        
        # # Obstacle 1: Square obstacle at (3, 2)
        # obs1 = {
        #     'center': (3.0, 2.0),
        #     'size': 0.8,
        #     'type': 'square'
        # }
        
        # # Obstacle 2: Square obstacle at (6, -1.5)
        # obs2 = {
        #     'center': (6.0, -1.5),
        #     'size': 0.8,
        #     'type': 'square'
        # }
        
        # # Obstacle 3: Square obstacle at (8, 1)
        # obs3 = {
        #     'center': (8.0, 1.0),
        #     'size': 0.8,
        #     'type': 'square'
        # }
        
        self.obstacles = [obs_small]
        rospy.loginfo("Created %d dynamic obstacles", len(self.obstacles))
    
    def cmd_vel_callback(self, msg):
        """Update robot velocity from control commands"""
        self.v = msg.linear.x
        self.omega = msg.angular.z
    
    def update_robot_state(self, dt):
        """Update robot position based on velocity commands"""
        # Simple unicycle model
        if abs(self.omega) < 1e-6:
            # Straight line motion
            self.x += self.v * math.cos(self.theta) * dt
            self.y += self.v * math.sin(self.theta) * dt
        else:
            # Arc motion
            radius = self.v / self.omega
            dtheta = self.omega * dt
            self.x += radius * (math.sin(self.theta + dtheta) - math.sin(self.theta))
            self.y += radius * (-math.cos(self.theta + dtheta) + math.cos(self.theta))
            self.theta += dtheta
        
        # Normalize theta to [-pi, pi]
        while self.theta > math.pi:
            self.theta -= 2 * math.pi
        while self.theta < -math.pi:
            self.theta += 2 * math.pi
    
    def update_obstacles(self, dt):
        """Update obstacle positions based on their velocities"""
        for obs in self.obstacles:
            vx, vy = obs['velocity']
            obs['center'][0] += vx * dt
            obs['center'][1] += vy * dt
    
    def publish_robot_pose(self):
        """Publish robot pose as PoseStamped"""
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        pose_msg.pose.position.z = 0.0
        
        # Store theta in orientation.z (simplified 2D representation)
        pose_msg.pose.orientation.z = self.theta
        pose_msg.pose.orientation.w = 1.0
        
        self.robot_pose_pub.publish(pose_msg)
    
    def publish_tf(self):
        """Publish TF transform from odom to base_link"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        
        # Convert theta to quaternion
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        
        self.tf_broadcaster.sendTransform(t)
    
    def publish_robot_marker(self):
        """Publish robot visualization marker"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.x
        marker.pose.position.y = self.y
        marker.pose.position.z = 0.1
        
        # Orientation
        qz = math.sin(self.theta / 2.0)
        qw = math.cos(self.theta / 2.0)
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        
        # Size
        marker.scale.x = self.robot_radius * 2
        marker.scale.y = self.robot_radius * 2
        marker.scale.z = 0.2
        
        # Color (blue)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        marker.lifetime = rospy.Duration(0.2)
        
        self.robot_marker_pub.publish(marker)
        
        # Publish direction indicator (arrow)
        arrow = Marker()
        arrow.header = marker.header
        arrow.ns = "robot_direction"
        arrow.id = 1
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        
        # Arrow from robot center pointing forward
        start = Point()
        start.x = self.x
        start.y = self.y
        start.z = 0.15
        
        end = Point()
        end.x = self.x + self.robot_radius * 1.5 * math.cos(self.theta)
        end.y = self.y + self.robot_radius * 1.5 * math.sin(self.theta)
        end.z = 0.15
        
        arrow.points = [start, end]
        
        arrow.scale.x = 0.1  # shaft diameter
        arrow.scale.y = 0.15  # head diameter
        arrow.scale.z = 0.0
        
        arrow.color.r = 1.0
        arrow.color.g = 0.0
        arrow.color.b = 0.0
        arrow.color.a = 1.0
        
        arrow.lifetime = rospy.Duration(0.2)
        
        self.robot_marker_pub.publish(arrow)
    
    def publish_obstacles(self):
        """Publish obstacle markers for costmap generation
        
        Obstacles are stored as dynamic positions in the odom frame (updated with velocities),
        but published in base_link frame (after transformation).
        This mimics real obstacle detection which happens in the robot's frame.
        """
        marker_array = MarkerArray()
        current_time = rospy.Time.now()
        
        # Try to get the transform from odom to base_link
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link',  # target frame
                'odom',       # source frame
                rospy.Time(0),  # latest available
                rospy.Duration(2.0)  # timeout
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, f"Could not transform obstacles from odom to base_link: {e}")
            return
        
        for i, obs in enumerate(self.obstacles):
            marker = Marker()
            marker.header.frame_id = "base_link"  # Publish in base_link frame
            marker.header.stamp = current_time
            marker.ns = "obstacles"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Create square polygon in odom frame
            cx, cy = obs['center']
            size = obs['size']
            half_size = size / 2.0
            
            # Square corners in odom frame
            corners_odom = [
                (cx - half_size, cy - half_size),
                (cx + half_size, cy - half_size),
                (cx + half_size, cy + half_size),
                (cx - half_size, cy + half_size),
                (cx - half_size, cy - half_size)  # Close the loop
            ]
            
            # Transform each corner from odom to base_link
            for corner_x, corner_y in corners_odom:
                # Create a PointStamped in odom frame
                point_odom = tf2_geometry_msgs.PointStamped()
                point_odom.header.frame_id = "odom"
                point_odom.header.stamp = current_time
                point_odom.point.x = corner_x
                point_odom.point.y = corner_y
                point_odom.point.z = 0.0
                
                # Transform to base_link
                point_base_link = tf2_geometry_msgs.do_transform_point(point_odom, transform)
                
                # Add to marker
                marker.points.append(point_base_link.point)
            
            marker.scale.x = 0.05  # line width
            
            # Color (red)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.lifetime = rospy.Duration(0)  # Never expire
            
            marker_array.markers.append(marker)
        
        self.obstacle_pub.publish(marker_array)
    
    def run(self):
        """Main simulation loop"""
        rate = rospy.Rate(self.rate_hz)
        dt = 1.0 / self.rate_hz
        
        # Wait a bit for TF buffer to populate
        rospy.sleep(0.5)
        
        # Publish obstacles once at startup and then periodically
        obstacle_pub_counter = 0
        
        rospy.loginfo("Robot simulator running at %.1f Hz", self.rate_hz)
        
        while not rospy.is_shutdown():
            # Update robot state
            self.update_robot_state(dt)
            
            # Update dynamic obstacles
            self.update_obstacles(dt)
            
            # Publish everything
            self.publish_robot_pose()
            self.publish_tf()
            self.publish_robot_marker()
            
            # Publish obstacles every iteration (50Hz)
            self.publish_obstacles()
            
            rate.sleep()

if __name__ == '__main__':
    try:
        simulator = RobotSimulator()
        simulator.run()
    except rospy.ROSInterruptException:
        pass 