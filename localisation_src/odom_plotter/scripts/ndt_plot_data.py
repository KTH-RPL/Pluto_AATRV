#!/usr/bin/env python3
import rosbag
import matplotlib.pyplot as plt

# === CONFIG ===
bag_file = "ndt_debug_data_2025-10-11-12-47-43.bag"

# === Load Data ===
time_exe = []
exe_time = []
time_prob = []
prob = []

with rosbag.Bag(bag_file, 'r') as bag:
    for topic, msg, t in bag.read_messages(topics=['/exe_time_ms', '/transform_probability']):
        if topic == '/exe_time_ms':
            time_exe.append(t.to_sec())
            exe_time.append(msg.data)
        elif topic == '/transform_probability':
            time_prob.append(t.to_sec())
            prob.append(msg.data)

# Normalize time (start at zero)
if time_exe:
    t0_exe = time_exe[0]
    time_exe = [t - t0_exe for t in time_exe]
if time_prob:
    t0_prob = time_prob[0]
    time_prob = [t - t0_prob for t in time_prob]

# === Plot 1: Execution Time ===
plt.figure()
plt.plot(time_exe, exe_time, 'b-', linewidth=1)
plt.title("NDT Execution Time (ms)")
plt.xlabel("Time [s]")
plt.ylabel("Execution Time [ms]")
plt.grid(True)

# === Plot 2: Transform Probability ===
plt.figure()
plt.plot(time_prob, prob, 'r-', linewidth=1)
plt.title("NDT Transform Probability")
plt.xlabel("Time [s]")
plt.ylabel("Probability")
plt.grid(True)

plt.show()

