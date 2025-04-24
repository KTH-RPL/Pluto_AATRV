#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import NavSatFix
import csv
import os
import threading
import time

# --- Configuration ---
DEFAULT_ROS_TOPIC = "/reach/fix"
# Output CSV file path (make sure the directory exists or adjust path)
# Using os.path.expanduser to handle '~' correctly
DEFAULT_CSV_FILE = os.path.expanduser("~/gps_data.csv")
WRITE_INTERVAL_SECONDS = 1.0 # How often to write the full data to CSV

# --- Global Data Storage ---
# Store all points in memory. For very long runs, consider capping size.
gps_points_list = [] # List of [lat, lon, alt] lists
data_lock = threading.Lock()

# --- ROS Subscriber Callback ---
def gps_callback(msg):
    """Callback function for the ROS subscriber."""
    global gps_points_list
    if msg.status.status >= 0: # STATUS_FIX or better
        with data_lock:
            gps_points_list.append([msg.latitude, msg.longitude, msg.altitude])
            # rospy.loginfo_throttle(10, f"Added point. Total points: {len(gps_points_list)}")
    # else:
        # rospy.logwarn_throttle(10, "Received NavSatFix message with no fix.")

# --- CSV Writing Function ---
def write_data_to_csv(file_path):
    """Writes the current gps_points_list to the CSV file."""
    global gps_points_list
    # Safely copy the data under lock
    with data_lock:
        points_to_write = list(gps_points_list) # Create a shallow copy

    # Write outside the lock to minimize lock holding time
    try:
        # Use 'w' to overwrite the file completely each time
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['latitude', 'longitude', 'altitude'])
            # Write data rows
            writer.writerows(points_to_write)
        # rospy.loginfo_throttle(5, f"Successfully wrote {len(points_to_write)} points to {file_path}")
    except IOError as e:
        rospy.logerr(f"Error writing to CSV file {file_path}: {e}")
    except Exception as e:
        rospy.logerr(f"Unexpected error writing CSV: {e}")

# --- Timer Callback for Periodic Writing ---
def timer_callback(event):
    """Called periodically by rospy.Timer to write data."""
    csv_file_path = rospy.get_param('~csv_file', DEFAULT_CSV_FILE)
    write_data_to_csv(csv_file_path)

# --- Main Execution ---
if __name__ == '__main__':
    try:
        # Initialize ROS node (using default signal handling is fine now)
        rospy.init_node('gps_to_csv_writer', anonymous=True)
        rospy.loginfo("ROS CSV writer node initialized.")

        # Get parameters
        gps_topic = rospy.get_param('~gps_topic', DEFAULT_ROS_TOPIC)
        csv_file_path = rospy.get_param('~csv_file', DEFAULT_CSV_FILE)
        write_interval = rospy.get_param('~write_interval', WRITE_INTERVAL_SECONDS)

        rospy.loginfo(f"Subscribing to ROS topic: {gps_topic}")
        rospy.loginfo(f"Writing data to CSV: {csv_file_path}")
        rospy.loginfo(f"Write interval: {write_interval} seconds")

        # Ensure output directory exists (optional, nice to have)
        try:
            os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        except OSError as e:
            rospy.logwarn(f"Could not create directory {os.path.dirname(csv_file_path)}: {e}. Assuming it exists.")


        # Create the subscriber (callback adds to list)
        rospy.Subscriber(gps_topic, NavSatFix, gps_callback)
        rospy.loginfo("ROS subscriber started.")

        # Create a timer to periodically write the data
        # The timer callback runs in a separate thread managed by rospy
        rospy.Timer(rospy.Duration(write_interval), timer_callback)
        rospy.loginfo("CSV writing timer started.")

        # Keep the node alive until shutdown
        rospy.spin()

    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred in CSV writer: {e}")
    finally:
        # Final write on shutdown (optional)
        final_csv_path = rospy.get_param('~csv_file', DEFAULT_CSV_FILE) # Re-get path in case it changed
        print(f"Attempting final write to {final_csv_path}...")
        write_data_to_csv(final_csv_path)
        print("Shutting down CSV writer node.")