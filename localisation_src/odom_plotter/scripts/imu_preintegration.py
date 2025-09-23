#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tf_trans


class ImuPreintegrationNode:
    def __init__(self):
        rospy.init_node('imu_preintegration_node')
        self.sub = rospy.Subscriber('/vectornav/IMU', Imu, self.imu_callback)

        self.last_time = None
        self.velocity = np.zeros(3)   # [vx, vy, vz]
        self.position = np.zeros(3)   # [x, y, z]
        self.positions = []           # Store trajectory (X,Y,Z)

        rospy.on_shutdown(self.plot_trajectory)
        rospy.loginfo("IMU Preintegration Node started...")

    def imu_callback(self, msg):
        curr_time = msg.header.stamp.to_sec()
        if self.last_time is None:
            self.last_time = curr_time
            return

        dt = curr_time - self.last_time
        self.last_time = curr_time

        # Reject bad timestamps
        if dt <= 0 or dt > 0.2:
            rospy.logwarn(f"Skipping IMU sample, dt={dt:.3f}s")
            return

        # --- Raw acceleration in body frame (includes gravity, per your sample) ---
        acc_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # --- Orientation quaternion (ROS order: x, y, z, w) ---
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]

        # --- Rotation matrix body->world ---
        R = tf_trans.quaternion_matrix(quat)[:3, :3]

        # --- Transform accel to world frame and ADD gravity in world frame ---
        # If accelerometer reports ~[-g] when stationary, the correct world accel is:
        # a_world = R * acc_body + g_world
        g_world = np.array([0.0, 0.0, 9.81])
        acc_world = R.dot(acc_body) + g_world

        # Optional: quick debug (throttled)
        import math
        _, _, yaw = tf_trans.euler_from_quaternion(quat)
        rospy.loginfo_throttle(1.0, f"dt={dt:.3f}s | yaw={math.degrees(yaw):.1f} deg | acc_w=({acc_world[0]:.3f},{acc_world[1]:.3f},{acc_world[2]:.3f})")

        # --- Integrate ---
        self.velocity += acc_world * dt
        self.position += self.velocity * dt

        # Store trajectory
        self.positions.append(self.position.copy())

    def plot_trajectory(self):
        if not self.positions:
            print("No trajectory data to plot.")
            return

        positions = np.array(self.positions)

        plt.figure(figsize=(8, 6))
        plt.plot(positions[:, 0], positions[:, 1], label='Trajectory (X-Y)')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('IMU Preintegrated Odometry Trajectory')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    try:
        node = ImuPreintegrationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
