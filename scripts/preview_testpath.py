#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from preview_control_ros import NavigationSystem  # Replace with the actual filename

class SnakePathGenerator:
    def __init__(self):
        self.nav_system = NavigationSystem()
        self.path_pub = rospy.Publisher('/snake_path', Path, queue_size=10)
        self.path = None
        path,orientation = self.generate_snake_path()
        self.nav_system.x_path = np.array([p[0] for p in path])
        self.nav_system.y_path = np.array([p[1] for p in path])
        self.nav_system.curvatures = self.nav_system.calculate_curvatures(self.nav_system.x_path,self.nav_system.y_path)
        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)
        

    def robotPose_callback(self, msg):
        self.nav_system.current_pose = msg
        
    def generate_snake_path(self, amplitude=2.0, wavelength=5.0, length=10.0, point_spacing=0.5):
        """
        Generate a snake path with specified point spacing and orientations
        
        Parameters:
        - amplitude: Width of the snake pattern (meters)
        - wavelength: Length of one full oscillation (meters)
        - length: Total length of the path (meters)
        - point_spacing: Distance between consecutive points along the curve (meters)
        
        Returns:
        - path: List of (x, y) tuples
        - orientations: List of orientation angles (radians)
        """
        num_points = int(np.ceil(length / point_spacing)) + 1
        x = np.linspace(0, length, num_points)
        
        y = amplitude * np.sin(2 * np.pi * x / wavelength)
        
        path = []
        orientations = []
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        for i in range(len(x)):
            path.append((x[i], y[i]))
            orientation = np.arctan2(dy[i], dx[i])
            orientation = (orientation + np.pi) % (2 * np.pi) - np.pi 
            orientations.append(orientation)
        return path, orientations
    
    def run(self):
        rate = rospy.Rate(20)  
        
        while not rospy.is_shutdown():                
            self.nav_system.run_control(is_last_goal=True)            
            if self.nav_system.reached:
                rospy.loginfo("Finished following snake path!")
                break                
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('snake_path_follower')
    try:
        path_follower = SnakePathGenerator()
        path_follower.run()
    except rospy.ROSInterruptException:
        pass