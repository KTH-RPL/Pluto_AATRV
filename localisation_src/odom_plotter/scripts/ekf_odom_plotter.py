#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

class Simple2DEKFWithIMU:
    def __init__(self):
        rospy.init_node('simple_2d_ekf_imu')

        # State: [x, y, yaw, vx, vy]
        self.x = np.zeros(5)
        self.P = np.eye(5) * 0.1

        # Process noise
        self.Q = np.diag([0.05, 0.05, 0.01, 0.1, 0.1])

        # Measurement noise
        self.R_odom = np.diag([0.05, 0.05, 0.01])
        self.R_imu_yaw = np.array([[0.02]])

        self.last_time = None

        # Store trajectories
        self.odom_traj = []
        self.ekf_traj = []

        # IMU data cache
        self.acc_body = np.zeros(2)
        self.imu_yaw0 = None
        self.yaw = 0.0

        # Subscribers
        rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/vectornav/IMU', Imu, self.imu_callback)

        rospy.on_shutdown(self.plot_trajectory)
        rospy.spin()

    def imu_callback(self, msg):
        # Orientation (yaw)
        q = msg.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

        # Store first yaw as offset
        if self.imu_yaw0 is None:
            self.imu_yaw0 = yaw
        self.yaw = yaw - self.imu_yaw0
        # print(self.yaw)  # debug

        # Use only x acceleration
        ax = msg.linear_acceleration.x

        # Dead-zone filter
        if -0.1 < ax < 0.1:
            ax = 0.0

        # Store accel (only x, no y)
        self.acc_body = np.array([ax, 0.0])

    def odom_callback(self, msg):   
        curr_time = msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = curr_time
            return
        dt = curr_time - self.last_time
        self.last_time = curr_time

        # --- Prediction step using IMU acceleration ---
        # Inject yaw from IMU into state
        self.x[2] = self.yaw
        yaw = self.x[2]

        # Rotate body-frame acceleration to world-frame
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        acc_world = R.dot(self.acc_body)

        # Update velocities
        self.x[3] += acc_world[0] * dt
        self.x[4] += acc_world[1] * dt
        # Update positions
        self.x[0] += self.x[3] * dt
        self.x[1] += self.x[4] * dt

        # Add process noise
        self.P += self.Q

        # --- Odometry measurement ---
        x_odom = msg.pose.pose.position.x
        y_odom = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw_odom = euler_from_quaternion([q.x, q.y, q.z, q.w])

        z = np.array([x_odom, y_odom, yaw_odom])
        self.odom_traj.append([x_odom, y_odom])

        # --- EKF update with odometry ---
        H = np.array([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]])
        y_residual = z - H.dot(self.x)
        S = H.dot(self.P).dot(H.T) + self.R_odom
        K = self.P.dot(H.T).dot(np.linalg.inv(S))
        # Uncomment if you want EKF correction from odometry:
        self.x += K.dot(y_residual)
        self.P = (np.eye(5) - K.dot(H)).dot(self.P)

        self.ekf_traj.append(self.x[:2].copy())

    def plot_trajectory(self):
        if not self.odom_traj:
            print("No trajectory data to plot.")
            return
        odom_traj = np.array(self.odom_traj)
        ekf_traj = np.array(self.ekf_traj)

        plt.figure(figsize=(8,6))
        plt.plot(odom_traj[:,0], odom_traj[:,1], label='Raw Odometry', linestyle='--', marker='o')
        plt.plot(ekf_traj[:,0], ekf_traj[:,1], label='EKF Estimate', linestyle='-', marker='.')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('EKF + IMU Acceleration vs Odometry Trajectory')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    try:
        Simple2DEKFWithIMU()
    except rospy.ROSInterruptException:
        pass
