#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
import numpy as np
class PathFollower:
    def __init__(self):
        rospy.init_node('path_follower', anonymous=True)

        self.path_sub = rospy.Subscriber('/path', Path, self.path_callback)
        self.gps_sub = rospy.Subscriber('/gps', PoseStamped, self.gps_callback)

        self.cmd_vel_pub = rospy.Publisher('/atrv/cmd_vel', Twist, queue_size=10)

        self.k_theta = 1
        self.k_y = 3
        self.k_dey = 1
        self.k_det = 3
        self.v_max = 1.0  
        self.v_min = 0.1  
        self.dt = 0.1
        self.kw = 0.2
        self.prev_omega = 0
        self.goal_distance_threshold = 0.5  

        self.path = None
        self.current_pose = None

    def path_callback(self, msg):
        self.path = msg.poses

    def gps_callback(self, msg):
        self.current_pose = msg

    def calculate_curvature(self, x, y):
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

    def run(self):
        rate = rospy.Rate(10)  
        while not rospy.is_shutdown():
            if self.path is not None and self.current_pose is not None:
                x_points = [pose.pose.position.x for pose in self.path]
                y_points = [pose.pose.position.y for pose in self.path]

                x_robot = self.current_pose.pose.position.x
                y_robot = self.current_pose.pose.position.y
                theta_robot = self.current_pose.pose.orientation
                theta_robot = theta_robot* np.pi / 180

                x_goal = x_points[-1]
                y_goal = y_points[-1]

                distance_to_goal = np.sqrt((x_goal - x_robot)**2 + (y_goal - y_robot)**2)

                if distance_to_goal < self.goal_distance_threshold:
                    v = self.v_min + (self.v_max - self.v_min) * (distance_to_goal / self.goal_distance_threshold)
                else:
                    v = self.v_max

                distances = np.sqrt((np.array(x_points) - x_robot)**2 + (np.array(y_points) - y_robot)**2)
                closest_idx = np.argmin(distances)

                if closest_idx + 2 < len(x_points):
                    closest_idx += 2
                    Ld = np.sqrt((x_points[closest_idx] - x_points[closest_idx - 2])**2 + (y_points[closest_idx] - y_points[closest_idx - 2])**2)
                else:
                    Ld = 0

                x_ref, y_ref = x_points[closest_idx], y_points[closest_idx]
                theta_ref = np.arctan2(y_points[closest_idx] - y_points[closest_idx - 1], x_points[closest_idx] - x_points[closest_idx - 1])

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

                self.prev_omega = omega

            rate.sleep()

if __name__ == '__main__':
    try:
        pf = PathFollower()
        pf.run()
    except rospy.ROSInterruptException:
        pass