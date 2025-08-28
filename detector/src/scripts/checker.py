#!/home/sankeerth/pluto/bin/python3

import rospy
from visualization_msgs.msg import MarkerArray
import threading

class ObstacleChecker:
    """
    A utility class to check for collisions with dynamically detected obstacles.

    This class subscribes to a MarkerArray topic representing obstacles as flat
    rectangles (CUBE markers) and provides a method to check if a given 2D
    point is in collision. It is designed to be thread-safe.
    """
    def __init__(self, obstacle_topic="/detected_obstacles"):
        """
        Initializes the ObstacleChecker.
        
        Args:
            obstacle_topic (str): The ROS topic publishing obstacle markers.
        """
        self.obstacles = []
        self.lock = threading.Lock()  # To ensure thread-safe access to the obstacles list

        # Subscribe to the obstacle topic
        self.obstacle_subscriber = rospy.Subscriber(
            obstacle_topic,
            MarkerArray,
            self._obstacle_callback,
            queue_size=1
        )
        
        rospy.loginfo(f"ObstacleChecker initialized. Subscribing to {obstacle_topic}")

    def _obstacle_callback(self, msg):
        """
        Internal callback to process incoming MarkerArray messages.
        It updates the internal list of obstacles with rectangular bounds.
        """
        new_obstacles = []
        for marker in msg.markers:
            # *** MODIFICATION: Look for CUBE markers representing rectangular obstacles ***
            if marker.action == marker.ADD and marker.type == marker.CUBE:
                center_x = marker.pose.position.x
                center_y = marker.pose.position.y
                width = marker.scale.x
                height = marker.scale.y
                
                half_w = width / 2.0
                half_h = height / 2.0

                # Store the min/max coordinates for efficient checking
                obstacle_data = {
                    'xmin': center_x - half_w,
                    'xmax': center_x + half_w,
                    'ymin': center_y - half_h,
                    'ymax': center_y + half_h
                }
                new_obstacles.append(obstacle_data)
        
        # Atomically update the list of obstacles
        with self.lock:
            self.obstacles = new_obstacles

    def is_safe(self, x, y):
        """
        Checks if a point (x, y) is inside any of the detected rectangular obstacles.
        This is the main function to be called by a planner.

        Args:
            x (float): The x-coordinate of the point in the robot's frame.
            y (float): The y-coordinate of the point in the robot's frame.

        Returns:
            bool: True if the point is safe (not in an obstacle), False otherwise.
        """
        # Create a local copy of the list to check against, to minimize lock time
        with self.lock:
            current_obstacles = list(self.obstacles)

        for obs in current_obstacles:
            # *** MODIFICATION: Perform an axis-aligned bounding box (AABB) check ***
            # This is a very efficient check for point-in-rectangle collision.
            if obs['xmin'] <= x <= obs['xmax'] and obs['ymin'] <= y <= obs['ymax']:
                return False  # Collision detected

        return True  # Point is safe

# ==============================================================================
# Example Usage: A simple node that demonstrates how to use the ObstacleChecker
# ==============================================================================
def main():
    """
    A standalone node to test the ObstacleChecker class.
    It continuously checks a few points and prints their status.
    """
    rospy.init_node('obstacle_checker_example', anonymous=True)

    # 1. Instantiate the checker
    collision_checker = ObstacleChecker()

    # Give it a moment to receive the first message from the detector
    rospy.sleep(1.0) 

    rate = rospy.Rate(5) # 5 Hz

    while not rospy.is_shutdown():
        # --- Example points to check (in base_link frame) ---
        point_in_front = (1.0, 0.0) 
        point_on_left = (0.5, 1.0)
        point_at_origin = (0.0, 0.0)

        # 2. Use the is_safe method for your planning logic
        is_front_safe = collision_checker.is_safe(point_in_front[0], point_in_front[1])
        is_left_safe = collision_checker.is_safe(point_on_left[0], point_on_left[1])
        is_origin_safe = collision_checker.is_safe(point_at_origin[0], point_at_origin[1])

        rospy.loginfo("--- Collision Check Status ---")
        rospy.loginfo(f"Point {point_in_front} is safe: {is_front_safe}")
        rospy.loginfo(f"Point {point_on_left} is safe: {is_left_safe}")
        rospy.loginfo(f"Point {point_at_origin} is safe: {is_origin_safe}")
        rospy.loginfo("----------------------------")

        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass