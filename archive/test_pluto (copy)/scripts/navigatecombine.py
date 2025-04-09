#!/usr/bin/env python3
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Point
from nav_msgs.msg import Path
from local_planner import execute_planning

class NavigationSystem:
    def __init__(self):
        rospy.init_node('pluto_navigation_system')
        
        # Subscribers
        self.goal_sub = rospy.Subscriber('/goal_pose', PoseStamped, self.goal_callback)
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.robot_pose_callback)
        
        # Publishers
        self.path_pub = rospy.Publisher('/planned_path', Path, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)
        
        self.k_theta = 0.2
        self.k_y = 0.7
        self.k_dey = 0.2
        self.k_det = 0.7                
        self.v_max = 1.0  
        self.v_min = 0.1  
        self.dt = 0.1
        self.kw = 0.2
        self.prev_omega = 0
        self.goal_distance_threshold = 0.5
        
        self.current_goal = None
        self.current_pose = None
        self.current_path = None
        self.control_rate = rospy.Rate(50)  
        
        rospy.loginfo("Pluto Navigation System initialized")

    def goal_callback(self, msg):
        self.current_goal = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo("New goal received: {}".format(self.current_goal))
        # self.plan_path()

    def robot_pose_callback(self, msg):
        self.current_pose = msg

    def plan_path(self):
        if self.current_pose is None or self.current_goal is None:
            rospy.logwarn("Cannot plan path - missing pose or goal")
            return
            
        current_position = (self.current_pose.pose.position.x, 
                          self.current_pose.pose.position.y)
        
        rospy.loginfo("Planning new path...")
        path_points, _, _, _ = execute_planning(current_position, self.current_goal)
        
        path_msg = Path()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = "map"
        
        for point in path_points:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.current_path = path_points
        rospy.loginfo("Path published with {} waypoints".format(len(path_msg.poses)))

    def calculate_curvature(self, x, y):
        """Calculate curvature of path"""
        curvatures = np.zeros(len(x))
        for i in range(1, len(x) - 1):
            x1, y1 = x[i - 1], y[i - 1]
            x2, y2 = x[i], y[i]
            x3, y3 = x[i + 1], y[i + 1]

            a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
            c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

            s = (a + b + c) / 2
            A = np.sqrt(s * (s - a) * (s - b) * (s - c))

            if A > 1e-6:
                R = (a * b * c) / (4 * A)
                curvatures[i] = 1 / R
            else:
                curvatures[i] = 0
        return curvatures

    def run_control(self):
        """Main control loop"""
        while not rospy.is_shutdown():
            if self.current_goal and not self.current_path:
                rospy.loginfo("generating path")
                self.plan_path()
                rospy.loginfo(self.current_path)

            if self.current_path is not None and self.current_pose is not None:
                path = self.current_path
                current_pose = self.current_pose
                rospy.loginfo("runnijnngg")
                
                x_points = [point[0] for point in path]
                y_points = [point[1] for point in path]
                # x_points = [pose.pose.position.x for pose in path.poses]
                # y_points = [pose.pose.position.y for pose in path.poses]
                theta_refs = [point[2] for point in path]
                x_robot = current_pose.pose.position.x
                y_robot = current_pose.pose.position.y

                orientation = current_pose.pose.orientation
                theta_robot = self.current_pose.pose.orientation.z
                


                x_goal = x_points[-1]
                y_goal = y_points[-1]

                distance_to_goal = np.sqrt((x_goal - x_robot)**2 + (y_goal - y_robot)**2)

                if distance_to_goal < self.goal_distance_threshold:
                    v = self.v_min + (self.v_max - self.v_min) * (distance_to_goal / self.goal_distance_threshold)
                else:
                    v = self.v_max

                distances = np.sqrt((np.array(x_points) - x_robot)**2 + (np.array(y_points) - y_robot)**2)
                closest_idx = np.argmin(distances)

                if closest_idx < len(x_points):
                    closest_idx += 0
                    Ld = np.sqrt((x_points[closest_idx] - x_points[closest_idx - 2])**2 + 
                         (y_points[closest_idx] - y_points[closest_idx - 2])**2)
                else:
                    Ld = 0

                x_ref, y_ref = x_points[closest_idx], y_points[closest_idx]
                # theta_ref = np.arctan2(y_points[closest_idx] - y_points[closest_idx - 1], x_points[closest_idx] - x_points[closest_idx - 1])
                theta_ref = theta_refs[closest_idx]
                e_y = np.sin(theta_ref) * (x_robot - x_ref) - np.cos(theta_ref) * (y_robot - y_ref)
                e_theta = theta_robot - theta_ref

                rc = self.calculate_curvature(x_points, y_points)
                rc_value = rc[closest_idx] if closest_idx < len(rc) else 0
                wd = self.kw * v * rc_value
                de_y = v * e_theta + self.prev_omega * Ld
                omega = wd + self.k_y * e_y - self.k_theta * e_theta - self.k_dey * de_y - self.k_det * e_theta
                
                cmd_vel = Twist()
                cmd_vel.linear.x = v
                cmd_vel.angular.z = omega
                self.cmd_vel_pub.publish(cmd_vel)
                rospy.loginfo("Angular speed")
                rospy.loginfo(omega)
                self.prev_omega = omega

            self.control_rate.sleep()

if __name__ == '__main__':
    try:
        nav_system = NavigationSystem()
        nav_system.run_control()
    except rospy.ROSInterruptException:
        pass
