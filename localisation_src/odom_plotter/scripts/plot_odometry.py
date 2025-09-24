#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt

class ESKF2D:
    def __init__(self):
        # State: [x, y, yaw, vx, vy, yaw_rate]
        self.x = np.zeros(6)
        self.P = np.eye(6) * 0.01

    def predict(self, imu_msg, dt):
        # Simple 2D motion model
        ax = imu_msg.linear_acceleration.x
        ay = imu_msg.linear_acceleration.y
        wz = imu_msg.angular_velocity.z

        # Integrate velocities and positions
        self.x[0] += self.x[3]*dt + 0.5*ax*dt**2
        self.x[1] += self.x[4]*dt + 0.5*ay*dt**2
        self.x[2] += wz*dt
        self.x[3] += ax*dt
        self.x[4] += ay*dt
        self.x[5] = wz  # angular rate

    def update(self, odom_msg):
        # Measurement: x, y, yaw
        z = np.array([odom_msg.pose.pose.position.x,
                      odom_msg.pose.pose.position.y,
                      self.quat_to_yaw(odom_msg.pose.pose.orientation)])
        H = np.zeros((3,6))
        H[0,0] = 1
        H[1,1] = 1
        H[2,2] = 1

        R = np.eye(3)*0.01
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    @staticmethod
    def quat_to_yaw(q):
        # Convert quaternion to yaw angle
        import math
        siny_cosp = 2*(q.w*q.z + q.x*q.y)
        cosy_cosp = 1 - 2*(q.y*q.y + q.z*q.z)
        return math.atan2(siny_cosp, cosy_cosp)

class OdometryFusion2D:
    def __init__(self):
        rospy.init_node('odom_fusion_2d', anonymous=True)
        self.eskf = ESKF2D()
        self.last_imu_time = None

        rospy.Subscriber('/atrv/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/vectornav/IMU', Imu, self.imu_callback)

        self.raw_x, self.raw_y, self.raw_yaw = [], [], []
        self.fused_x, self.fused_y, self.fused_yaw = [], [], []

    def imu_callback(self, msg):
        t = msg.header.stamp.to_sec()
        if self.last_imu_time is None:
            self.last_imu_time = t
            return
        dt = t - self.last_imu_time
        self.last_imu_time = t
        self.eskf.predict(msg, dt)

        self.fused_x.append(self.eskf.x[0])
        self.fused_y.append(self.eskf.x[1])
        self.fused_yaw.append(self.eskf.x[2])

    def odom_callback(self, msg):
        self.raw_x.append(msg.pose.pose.position.x)
        self.raw_y.append(msg.pose.pose.position.y)
        self.raw_yaw.append(self.eskf.quat_to_yaw(msg.pose.pose.orientation))
        self.eskf.update(msg)

    def run(self):
        rospy.loginfo("Fusing 2D odom and IMU. Press Ctrl+C to stop...")
        rospy.spin()

        # Plot
        plt.figure()
        plt.plot(self.raw_x, self.raw_y, 'r-', label='Raw Odom')
        plt.plot(self.fused_x, self.fused_y, 'b-', label='Fused Odom')
        plt.title("2D Odometry vs Fused Odometry")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == '__main__':
    node = OdometryFusion2D()
    node.run()
