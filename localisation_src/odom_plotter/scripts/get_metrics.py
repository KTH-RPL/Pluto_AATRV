#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
import signal
import sys

# storage for values
prob_values = []
prob_timestamps = []
exe_values = []
exe_timestamps = []

def prob_callback(msg):
    prob_values.append(msg.data)
    prob_timestamps.append(rospy.get_time())

def exe_callback(msg):
    exe_values.append(msg.data)
    exe_timestamps.append(rospy.get_time())

def signal_handler(sig, frame):
    rospy.loginfo("Ctrl+C pressed. Plotting data...")

    # Plot transform_probability
    plt.figure()
    plt.plot(prob_timestamps, prob_values, label="Transform Probability")
    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title("/transform_probability over time")
    plt.legend()
    plt.grid(True)

    # Plot exe_time_ms
    plt.figure()
    plt.plot(exe_timestamps, exe_values, label="Execution Time (ms)", color="orange")
    plt.xlabel("Time (s)")
    plt.ylabel("Execution Time (ms)")
    plt.title("/exe_time_ms over time")
    plt.legend()
    plt.grid(True)

    plt.show()
    sys.exit(0)

def listener():
    rospy.init_node('prob_and_exe_listener', anonymous=True)

    rospy.Subscriber("/transform_probability", Float32, prob_callback)
    rospy.Subscriber("/exe_time_ms", Float32, exe_callback)

    # handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    rospy.loginfo("Listening to /transform_probability and /exe_time_ms. Press Ctrl+C to stop and plot.")
    rospy.spin()

if __name__ == '__main__':
    listener()

