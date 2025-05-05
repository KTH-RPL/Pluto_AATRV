#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os # Used for path manipulation

class ImageSaver:
    def __init__(self):
        # Initialize the ROS node
        # anonymous=True ensures the node has a unique name, avoiding conflicts
        rospy.init_node('image_saver_node', anonymous=True)

        # --- Parameters ---
        # Get the image topic from the parameter server or use a default
        self.image_topic = '/rsD455_node0/depth/image_rect_raw'
        # Get the desired save path from the parameter server or use a default
        # Default saves to the user's home directory

        self.save_path = "/home/sankeerth/catkin_ws/src/Pluto_AATRV/data/depth/"
        # --- End Parameters ---

        # Create a CvBridge object
        self.bridge = CvBridge()

        # Flag to ensure we only save one image
        self.image_saved = False

        # Make sure the directory to save the image exists
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
                rospy.loginfo(f"Created directory: {save_dir}")
            except OSError as e:
                rospy.logerr(f"Failed to create directory {save_dir}: {e}")
                # Optional: Exit if directory creation fails
                rospy.signal_shutdown("Failed to create save directory")
                return # Stop initialization

        # Subscribe to the image topic
        rospy.loginfo(f"Subscribing to image topic: {self.image_topic}")
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)

        rospy.loginfo(f"Waiting for image message on {self.image_topic}...")
        rospy.loginfo(f"Image will be saved to: {self.save_path}")
        rospy.loginfo("Node will shut down after saving the first image.")

    def image_callback(self, msg):
        # Check if we have already saved an image
        if self.image_saved:
            return # Do nothing if already saved

        rospy.loginfo("Received an image!")

        try:

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            rospy.loginfo(f"Image encoding: {msg.encoding}, Converted to bgr8")

        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error converting image: {e}")
            return
        except Exception as e:
             rospy.logerr(f"General Error converting image: {e}")
             return
        path = f"{msg.header.seq}_{msg.header.stamp.secs}_{msg.header.stamp.nsecs}_{msg.header.frame_id}.jpg"

        success = cv2.imwrite(self.save_path+path, cv_image)


if __name__ == '__main__':
    try:
        image_saver = ImageSaver()
        # Keep the node running until it's shut down (either by ctrl-c or programmatically)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Image saver node interrupted.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred: {e}")