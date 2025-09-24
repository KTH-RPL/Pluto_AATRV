#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class SimplePathPublisher:
    """
    A ROS 1 node that subscribes to the robot's pose and publishes a simple path.
    """
    def __init__(self):
        """
        Initializes the node, subscriber, and publisher for ROS 1.
        """
        # Initialize the publisher for the /planned_path topic.
        # The message type is Path, which is a sequence of poses.
        self.path_publisher = rospy.Publisher('/planned_path', Path, queue_size=10)

        # Initialize the subscriber to the /robot_pose topic.
        # The message type is PoseStamped, which includes position and orientation.
        self.pose_subscriber = rospy.Subscriber(
            '/robot_pose',
            PoseStamped,
            self.pose_callback)
        
        rospy.loginfo('Simple Path Publisher node has been started.')
        rospy.loginfo('Listening for pose on /robot_pose...')

    def pose_callback(self, msg):
        """
        This callback function is executed whenever a new pose is received.
        It generates and publishes a simple path based on the current pose.
        """
        current_x = msg.pose.position.x
        current_y = msg.pose.position.y
        
        rospy.loginfo('Received robot pose: (x=%.2f, y=%.2f)', current_x, current_y)

        # --- 1. Create the Path Message ---
        path_msg = Path()
        # Set the header of the path message. The frame_id should match the
        # frame of the incoming pose message.
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = msg.header.frame_id

        # --- 2. Create the Poses for the Path ---
        # The path will consist of two points: the robot's current location
        # and a point 1 meter ahead in the x-direction.

        # First pose: The robot's current position.
        start_pose = msg

        # Second pose: The target position (1 meter ahead in x).
        goal_pose = PoseStamped()
        goal_pose.header = start_pose.header # Use the same header
        goal_pose.pose.position.x = current_x + 1.0
        goal_pose.pose.position.y = current_y
        # Keep the orientation the same as the current pose
        goal_pose.pose.orientation = start_pose.pose.orientation

        # --- 3. Add the Poses to the Path ---
        path_msg.poses.append(start_pose)
        path_msg.poses.append(goal_pose)
        
        # --- 4. Publish the Path ---
        self.path_publisher.publish(path_msg)
        rospy.loginfo('Published a 2-point path to /planned_path')


def main():
    """
    Main function to initialize and run the ROS node.
    """
    rospy.init_node('simple_path_publisher')
    try:
        SimplePathPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
