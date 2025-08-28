#!/home/sankeerth/pluto/bin/python3

import rospy
from visualization_msgs.msg import MarkerArray
import threading

class ObstacleChecker:
    """
    A utility class to check for collisions with dynamically detected obstacles.
    This version checks a point against a list of 2D polygons.
    """
    def __init__(self, obstacle_topic="/detected_obstacles"):
        self.obstacles = [] # Will be a list of polygons (each a list of points)
        self.lock = threading.Lock()

        self.obstacle_subscriber = rospy.Subscriber(
            obstacle_topic,
            MarkerArray,
            self._obstacle_callback,
            queue_size=1
        )
        rospy.loginfo(f"ObstacleChecker (Polygon version) initialized. Subscribing to {obstacle_topic}")

    def _obstacle_callback(self, msg):
        """
        Internal callback to process incoming MarkerArray messages,
        extracting polygon footprints from LINE_STRIP markers.
        """
        new_obstacles = []
        for marker in msg.markers:
            # *** MODIFICATION: Look for LINE_STRIP markers ***
            if marker.action == marker.ADD and marker.type == marker.LINE_STRIP and len(marker.points) > 2:
                # Extract the (x, y) coordinates of the polygon vertices
                # The last point is a duplicate to close the loop, so we skip it.
                polygon = [(p.x, p.y) for p in marker.points[:-1]]
                new_obstacles.append(polygon)
        
        with self.lock:
            self.obstacles = new_obstacles

    def _point_in_polygon(self, x, y, polygon):
        """
        Checks if a point (x, y) is inside a polygon using the Ray Casting algorithm.
        
        Args:
            x (float): x-coordinate of the point.
            y (float): y-coordinate of the point.
            polygon (list of tuples): A list of (x, y) vertices of the polygon.

        Returns:
            bool: True if the point is inside the polygon, False otherwise.
        """
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def is_safe(self, x, y):
        """
        Checks if a point (x, y) is inside any of the detected polygonal obstacles.
        """
        with self.lock:
            current_obstacles = list(self.obstacles)

        for obs_polygon in current_obstacles:
            # *** MODIFICATION: Perform point-in-polygon check ***
            if self._point_in_polygon(x, y, obs_polygon):
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