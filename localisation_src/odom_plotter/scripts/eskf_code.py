#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

class Simple2DESKFWithIMU:
    def __init__(self):
        rospy.init_node('simple_2d_eskf_imu')

        # Nominal state: [x, y, yaw, vx, vy]
        self.x_nom = np.zeros(5)

        # Error-state covariance
        self.P = np.eye(5) * 0.1

        # Process noise (for error dynamics)
        self.Q = np.diag([0.01, 0.01, 0.005, 0.05, 0.05])

        # Measurement noise for odometry [x, y, yaw]
        self.R_odom = np.diag([0.05, 0.05, 0.01])

        self.last_time = None

        # Store trajectories
        self.odom_traj = []
        self.eskf_traj = []

        # IMU cache
        self.acc_body = np.zeros(2)
        self.imu_yaw0 = None
        self.yaw_meas = 0.0

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
        self.yaw_meas = yaw - self.imu_yaw0

        # Use only x acceleration
        ax = msg.linear_acceleration.x

        # Dead-zone filter
        if -0.05 < ax < 0.05:
            ax = 0.0

        # Only x accel in body frame
        self.acc_body = np.array([-ax, 0.0])

    def predict(self, dt):
        """ ESKF prediction: propagate nominal state and error covariance """
        x, y, yaw, vx, vy = self.x_nom

        # Replace yaw with IMU yaw measurement (drift-free nominal yaw)
        yaw = -self.yaw_meas
        self.x_nom[2] = yaw

        # Rotate acceleration into world frame
        R = np.array([[np.cos(yaw), -np.sin(yaw)],
                      [np.sin(yaw),  np.cos(yaw)]])
        acc_world = R.dot(self.acc_body)

        # Propagate nominal state
        vx += acc_world[0] * dt
        vy += acc_world[1] * dt
        x += vx * dt
        y += vy * dt

        self.x_nom = np.array([x, y, yaw, vx, vy])

        # --- Error-state propagation ---
        # Linearized dynamics matrix F
        F = np.eye(5)
        F[0,3] = dt
        F[1,4] = dt
        # coupling yaw error into velocity
        F[3,2] = -np.sin(yaw) * self.acc_body[0] * dt
        F[4,2] =  np.cos(yaw) * self.acc_body[0] * dt

        # Propagate covariance
        self.P = F @ self.P @ F.T + self.Q

    def correct(self, z):
        """ ESKF correction with odometry """
        # Measurement model: h(x) = [x, y, yaw]
        H = np.array([[1,0,0,0,0],
                      [0,1,0,0,0],
                      [0,0,1,0,0]])
        z_hat = H.dot(self.x_nom)
        y_res = z - z_hat  # innovation

        S = H @ self.P @ H.T + self.R_odom
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update error state estimate
        delta_x = K @ y_res

        # Inject correction into nominal state
        self.x_nom[0] += delta_x[0]
        self.x_nom[1] += delta_x[1]
        self.x_nom[2] += delta_x[2]
        self.x_nom[3] += delta_x[3]
        self.x_nom[4] += delta_x[4]

        # Normalize yaw
        self.x_nom[2] = np.arctan2(np.sin(self.x_nom[2]), np.cos(self.x_nom[2]))

        # Update covariance
        I = np.eye(5)
        self.P = (I - K @ H) @ self.P

    def odom_callback(self, msg):
        curr_time = msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = curr_time
            return
        dt = curr_time - self.last_time
        self.last_time = curr_time

        # Prediction step
        self.predict(dt)

        # Odometry measurement
        x_odom = msg.pose.pose.position.x
        y_odom = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw_odom = euler_from_quaternion([q.x, q.y, q.z, q.w])

        z = np.array([x_odom, y_odom, yaw_odom])
        self.odom_traj.append([x_odom, y_odom])

        # Correction step
        self.correct(z)
        self.eskf_traj.append(self.x_nom[:2].copy())

    def plot_trajectory(self):
        if not self.odom_traj:
            print("No trajectory data to plot.")
            return
        odom_traj = np.array(self.odom_traj)
        eskf_traj = np.array(self.eskf_traj)

        plt.figure(figsize=(8,6))
        plt.plot(odom_traj[:,0], odom_traj[:,1], label='Raw Odometry', linestyle='--', marker='o')
        plt.plot(eskf_traj[:,0], eskf_traj[:,1], label='ESKF Estimate', linestyle='-', marker='.')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('ESKF + IMU Acceleration vs Odometry Trajectory')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    try:
        Simple2DESKFWithIMU()
    except rospy.ROSInterruptException:
        pass
