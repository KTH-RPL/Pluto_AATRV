import rospy
import math
from geometry_msgs.msg import PoseStamped, Twist, Point
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from pyproj import Proj, transform

# Global parameters
robot_frame_id = "base_link"
goal_tolerance = 0.1  # Tolerance to stop when close to goal
max_speed = 0.5  # Maximum robot speed (m/s)
max_turn_rate = 1.0  # Maximum angular speed (rad/s)

# Geo offset values to simulate movement
geo_offset = [-333520.64199354, -6582420.00142414, 0.0]

# Publishers and Subscribers
gnss_pub = rospy.Publisher("/reach/fix", NavSatFix, queue_size=10)
cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
path_sub = rospy.Subscriber("/local_planner/path", Path, callback_path)

# Marker publishers
robot_marker_pub = rospy.Publisher("/robot_marker", Marker, queue_size=10)
path_marker_pub = rospy.Publisher("/path_marker", MarkerArray, queue_size=10)

# Global state
current_position = (0.0, 0.0, 0.0)  # (x, y, theta)
goal_position = None
path = []

# Define UTM projection (assuming the UTM zone for Stockholm, typically zone 33N)
utm_proj = Proj(proj="utm", zone=33, ellps="WGS84")
latlon_proj = Proj(proj="latlong", datum="WGS84")

# Callback for receiving the path from local planner
def callback_path(msg):
    global path
    path = msg.poses
    rospy.loginfo("Received new path with %d waypoints", len(path))

# Function to calculate Euclidean distance
def euclidean_dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to move the robot towards the goal
def move_robot(target_x, target_y):
    global current_position

    # Calculate the angle to the goal (robot's orientation needed)
    dx = target_x - current_position[0]
    dy = target_y - current_position[1]
    angle_to_goal = math.atan2(dy, dx)

    # Get current robot's orientation
    theta = current_position[2]
    angle_error = angle_to_goal - theta
    if angle_error > math.pi:
        angle_error -= 2 * math.pi
    elif angle_error < -math.pi:
        angle_error += 2 * math.pi

    # Control: create twist message for movement command
    cmd = Twist()

    # Linear velocity (move towards the goal)
    cmd.linear.x = max_speed * min(1.0, euclidean_dist((current_position[0], current_position[1]), (target_x, target_y)) / goal_tolerance)

    # Angular velocity (turn towards the goal)
    cmd.angular.z = max_turn_rate * angle_error

    # Publish the movement command
    cmd_vel_pub.publish(cmd)

# Function to simulate GNSS update using geo_offset
def publish_gnss(x, y, theta):
    # Convert the robot's local position (x, y) to UTM coordinates
    UTM_x_publish = x - geo_offset[0]
    UTM_y_publish = y - geo_offset[1]
    
    # Convert UTM to Lat/Lon
    lon, lat = transform(utm_proj, latlon_proj, UTM_x_publish, UTM_y_publish)
    
    # Simulate GNSS by publishing to the GNSS topic (NavSatFix message)
    gnss_msg = NavSatFix()
    gnss_msg.latitude = lat
    gnss_msg.longitude = lon
    gnss_msg.altitude = 0  # Assuming flat surface
    gnss_pub.publish(gnss_msg)
    rospy.loginfo("Simulating GNSS: Latitude = %.6f, Longitude = %.6f", lat, lon)

# Function to create and publish markers (visualization in RViz)
def create_robot_marker(x, y, theta):
    # Marker for the robot (as a sphere at its position)
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "robot"
    marker.id = 0
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0.1  # Small height for visibility
    marker.scale.x = 0.2  # Size of the sphere
    marker.scale.y = 0.2
    marker.scale.z = 0.2
    marker.color.a = 1.0  # Transparency (1.0 means opaque)
    marker.color.r = 1.0  # Red color
    marker.color.g = 0.0
    marker.color.b = 0.0
    return marker

def create_path_markers(path):
    # Markers for the path (as a series of points)
    markers = MarkerArray()
    for i, waypoint in enumerate(path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "path"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = waypoint.pose.position.x
        marker.pose.position.y = waypoint.pose.position.y
        marker.pose.position.z = 0.1  # Small height for visibility
        marker.scale.x = 0.1  # Size of the sphere
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0  # Transparency (1.0 means opaque)
        marker.color.g = 1.0  # Green color
        markers.markers.append(marker)
    return markers

def visualize_path():
    # Publish robot path visualization markers
    if path:
        path_marker_pub.publish(create_path_markers(path))

def update_robot_position():
    # Visualize the robot's current position
    robot_marker = create_robot_marker(current_position[0], current_position[1], current_position[2])
    robot_marker_pub.publish(robot_marker)

# Main function for simulation
def controller_node():
    rospy.init_node('controller_node')

    rate = rospy.Rate(10)  # 10 Hz loop rate
    while not rospy.is_shutdown():
        if path:
            # Follow the path step by step
            next_goal = path[0]  # Simple approach: following the first waypoint
            target_x = next_goal.pose.position.x
            target_y = next_goal.pose.position.y
            move_robot(target_x, target_y)
        
        # Simulate GNSS and publish it
        publish_gnss(current_position[0], current_position[1], current_position[2])

        # Visualize robot and path
        update_robot_position()
        visualize_path()

        # Sleep until the next cycle
        rate.sleep()

if __name__ == "__main__":
    try:
        controller_node()
    except rospy.ROSInterruptException:
        pass
