#!/usr/bin/env python3
import rospy
import argparse
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

class LivePlot:
    def __init__(self, plot_local):
        rospy.init_node('live_plot_node', anonymous=True)

        self.robot_pose = None
        self.goal_pose = None
        self.lookahead_pose = None
        self.global_path = None
        self.vel_history = deque(maxlen=100)
        self.error_history = deque(maxlen=100)
        self.robot_poses_x = []
        self.robot_poses_y = []
        self.goal_poses_x = [0]
        self.goal_poses_y = [0]
        self.global_paths = [[0, 0]]

        self.all_x = []
        self.all_y = []

        # Choose topics based on --plot_local flag
        robot_pose_topic = '/local_robot_pose' if plot_local else '/robot_pose'
        goal_pose_topic = '/local_goal_pose' if plot_local else '/goal_pose'
        path_topic = '/planned_path' if plot_local else '/planned_path'
        lookahead_point_topic = '/lookahead_point'
        rospy.Subscriber(robot_pose_topic, PoseStamped, self.robot_pose_cb)
        rospy.Subscriber(goal_pose_topic, PoseStamped, self.goal_pose_cb)
        rospy.Subscriber(lookahead_point_topic, PoseStamped, self.lookahead_cb)
        rospy.Subscriber(path_topic, Path, self.global_path_cb)
        rospy.Subscriber('/atrv/cmd_vel', Twist, self.cmd_vel_cb)

        self.fig, (self.ax_map, self.ax_vel, self.ax_error) = plt.subplots(3, 1, figsize=(8, 10))
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=200)

    def robot_pose_cb(self, msg):
        self.robot_pose = msg

    def goal_pose_cb(self, msg):
        self.goal_pose = msg
    
    def lookahead_cb(self, msg):
        self.lookahead_pose = msg
    
    def global_path_cb(self, msg):
        if not self.global_path:
            self.global_path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]

    def cmd_vel_cb(self, msg):
        self.vel_history.append((msg.linear.x, msg.angular.z))

    def update_plot(self, frame):
        self.ax_map.clear()
        self.ax_vel.clear()

        self.ax_map.set_title('Map View')
        self.ax_map.grid(True)

        # Plot global path
        if self.global_path:
            rospy.loginfo("Received Path")
            path_x, path_y = zip(*self.global_path)
            self.ax_map.plot(path_x, path_y, 'b--', label='Global Path')
            # self.all_x.extend([min(path_x), max(path_x)])
            # self.all_y.extend([min(path_y), max(path_y)])

        # Plot goal
        if self.goal_pose:
            gx = self.goal_pose.pose.position.x
            gy = self.goal_pose.pose.position.y
            self.ax_map.plot(gx, gy, 'ro', label='Goal')
            self.all_x.append(gx)
            self.all_y.append(gy)

        # Plot robot pose
        if self.robot_pose:
            x = self.robot_pose.pose.position.x
            y = self.robot_pose.pose.position.y
            self.robot_poses_x.append(x)
            self.robot_poses_y.append(y)
            yaw = self.robot_pose.pose.orientation.z
            dx = np.cos(yaw)
            dy = np.sin(yaw)
            self.ax_map.arrow(x, y, dx, dy, head_width=1, color='g', label='Robot')
            self.all_x.append(x)
            self.all_y.append(y)
            self.ax_map.plot(self.robot_poses_x, self.robot_poses_y, "m+")

        
        if self.lookahead_pose:
            lx = self.lookahead_pose.pose.position.x
            ly = self.lookahead_pose.pose.position.y
            self.ax_map.plot(lx, ly, 'b*', label='lookahead')

        # Dynamically adjust axis limits with some margin
        if self.all_x and self.all_y:
            margin = 5
            self.ax_map.set_xlim(min(self.all_x) - margin, max(self.all_x) + margin)
            self.ax_map.set_ylim(min(self.all_y) - margin, max(self.all_y) + margin)

        self.ax_map.legend()

        # Plot velocity history
        if self.vel_history:
            vels = np.array(self.vel_history)
            self.ax_vel.plot(vels[:, 0], label='Linear X')
            self.ax_vel.plot(vels[:, 1], label='Angular Z')
            self.ax_vel.set_ylim(-2, 2)
            self.ax_vel.set_title('Velocity Commands')
            self.ax_vel.legend()
            self.ax_vel.grid(True)

        # Compute and store errors if both robot and goal are available
        if self.robot_pose and self.goal_pose:
            rx = self.robot_pose.pose.position.x
            ry = self.robot_pose.pose.position.y
            gx = self.goal_pose.pose.position.x
            gy = self.goal_pose.pose.position.y
            distance_error = np.hypot(gx - rx, gy - ry)

            goal_theta = np.arctan2(gy - ry, gx - rx)
            robot_theta = self.robot_pose.pose.orientation.z  # Assumed yaw
            heading_error = self.angle_diff(goal_theta, robot_theta)

            self.error_history.append((distance_error, heading_error))

        # Plot errors
        self.ax_error.clear()
        if self.error_history:
            errors = np.array(self.error_history)
            self.ax_error.plot(errors[:, 0], label='Distance to Goal (m)')
            self.ax_error.plot(errors[:, 1], label='Heading Error (rad)')
            self.ax_error.set_title('Error to Goal')
            self.ax_error.set_ylim(-np.pi, np.pi)
            self.ax_error.set_ylabel("Error")
            self.ax_error.set_xlabel("Timestep")
            self.ax_error.legend()
            self.ax_error.grid(True)

    def run(self):
        plt.tight_layout()
        plt.show()

    @staticmethod
    def angle_diff(a, b):
        diff = a - b
        return (diff + np.pi) % (2 * np.pi) - np.pi

if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Live plot of robot path, goal, and velocity")
    parser.add_argument("--plot_local", action="store_true", help="Use local coordinates instead of global")
    args, _ = parser.parse_known_args()  # To avoid issues with ROS args

    try:
        plotter = LivePlot(plot_local=args.plot_local)
        plotter.run()
        rospy.spin()
    except KeyboardInterrupt:
        plt.close("all")
        exit()
    except rospy.ROSInterruptException:
        pass
