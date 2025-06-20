#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from preview_control_ros import NavigationSystem  

class SnakePathGenerator:
    def __init__(self):
        self.nav_system = NavigationSystem()
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.path = None
        path = self.generate_snake_path()
        self.publish_path(path)
        self.nav_system.current_path = np.array(path).T
        self.nav_system.current_curvatures = self.nav_system.calculate_curvature(
            self.nav_system.current_path[0, :],  
            self.nav_system.current_path[1, :]   
        )
        rospy.Subscriber("/robot_pose", PoseStamped, self.robotPose_callback)
        

    def robotPose_callback(self, msg):
        self.nav_system.current_pose = msg


    def publish_path(self, path_points):
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"  
        
        for point in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.z = 1.0
            path_msg.poses.append(pose)
            
        self.path_pub.publish(path_msg)
        self.path = path_msg  

    def generate_snake_path(self, amplitude=3.0, wavelength=7.0, length=10.0, point_spacing=1.0):        
        num_points = int(np.ceil(length / point_spacing)) + 1
        y = np.linspace(0, length, num_points)        
        x = amplitude * np.sin(2 * np.pi * y / wavelength)        
        path = []        
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        for i in range(len(x)):            
            orientation = np.arctan2(dy[i], dx[i])
            orientation = (orientation + np.pi) % (2 * np.pi) - np.pi 
            path.append([x[i], y[i],orientation])
        return path
    
    def run(self):
        rate = rospy.Rate(20)  
        while not rospy.is_shutdown():
            if self.nav_system.current_pose is not None:                
                self.nav_system.run_control(is_last_goal=True)            
            if self.nav_system.reached:
                rospy.loginfo("Finished following path!")
                break                
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('snake_path_follower')
    try:
        path_follower = SnakePathGenerator()
        path_follower.run()
    except rospy.ROSInterruptException:
        pass