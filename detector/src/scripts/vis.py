import rosbag
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon
import numpy as np

def load_ros_data(bag_path):
    """
    Loads and synchronizes /robot_pose and /detected_obstacles messages 
    from a ROS bag file.
    """
    print(f"Loading messages from {bag_path}...")

    try:
        # Get all robot_pose messages with timestamps
        robot_pose_messages = []
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=['/robot_pose']):
                robot_pose_messages.append((msg, t))

        # Get all detected_obstacles messages with timestamps
        detected_obstacles_messages = []
        with rosbag.Bag(bag_path, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=['/detected_obstacles']):
                detected_obstacles_messages.append((msg, t))
    
    except FileNotFoundError:
        print(f"Error: Bag file not found at '{bag_path}'.")
        print("Please update the 'bag_file_path' variable in the script.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading the bag file: {e}")
        return None, None
        
    print("Messages loaded successfully.")

    if not robot_pose_messages or not detected_obstacles_messages:
        print("Error: One or both required topics (/robot_pose, /detected_obstacles) are empty in the bag file.")
        return None, None

    # Synchronize obstacle and pose messages by finding the closest timestamp
    print("Synchronizing obstacle and pose messages...")
    synced_data = []
    for detected_obstacle_msg, obs_time in detected_obstacles_messages:
        # Find the robot_pose message with the closest timestamp
        closest_pose_msg, pose_time = min(
            robot_pose_messages, 
            key=lambda x: abs((obs_time - x[1]).to_sec())
        )
        
        synced_data.append((detected_obstacle_msg, closest_pose_msg))
    
    if not synced_data:
        print("No synchronized obstacle/pose pairs found. Cannot create animation.")
        return None, None

    print(f"Found {len(synced_data)} synchronized frames for animation.")
    
    # Extract the full robot path for background plotting
    full_path = {
        'x': [msg.pose.position.x for msg, _ in robot_pose_messages],
        'y': [msg.pose.position.y for msg, _ in robot_pose_messages]
    }

    return synced_data, full_path

def quaternion_to_yaw(x, y, z, w):
    """
    Convert quaternion to yaw angle.
    """
    # Handle the case where orientation.z might already be yaw
    # This is a fallback calculation
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

def transform_point_to_odom(local_x, local_y, robot_x, robot_y, robot_yaw):
    """
    Transform a point from robot's local frame to odom frame.
    
    Args:
        local_x, local_y: Point coordinates in robot frame
        robot_x, robot_y: Robot position in odom frame
        robot_yaw: Robot orientation (yaw) in odom frame
    
    Returns:
        odom_x, odom_y: Point coordinates in odom frame
    """
    # Rotation matrix
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)
    
    # Transform: rotate then translate
    odom_x = robot_x + (local_x * cos_yaw - local_y * sin_yaw)
    odom_y = robot_y + (local_x * sin_yaw + local_y * cos_yaw)
    
    return odom_x, odom_y

def create_animation(synced_data, full_path):
    """Creates and displays the animation of the robot and obstacles."""
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Robot Path and Detected Obstacles')
    ax.grid(True)

    # Plot the full robot path as a static background
    ax.plot(full_path['x'], full_path['y'], label='Full Robot Path', 
            color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Initialize robot marker (circle) and heading arrow separately
    robot_dot, = ax.plot([], [], 'bo', markersize=10, label='Robot Position')
    robot_arrow = ax.arrow(0, 0, 0, 0, head_width=0.3, head_length=0.3, 
                          fc='blue', ec='blue', linewidth=2)
    
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                       verticalalignment='top', fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    obstacle_patches = []

    # Set plot limits based on the full path for a stable view
    x_range = max(full_path['x']) - min(full_path['x'])
    y_range = max(full_path['y']) - min(full_path['y'])
    
    # Use larger margins - 50% of range or minimum 5 units
    x_margin = max(x_range * 0.5, 5)
    y_margin = max(y_range * 0.5, 5)
    
    ax.set_xlim(min(full_path['x']) - x_margin, max(full_path['x']) + x_margin)
    ax.set_ylim(min(full_path['y']) - y_margin, max(full_path['y']) + y_margin)
    
    # Animation update function, called for each frame
    def update(frame):
        nonlocal obstacle_patches, robot_arrow
        
        obstacle_msg, pose_msg = synced_data[frame]
        
        # Update Robot Position
        pos_x = pose_msg.pose.position.x
        pos_y = pose_msg.pose.position.y
        
        # Try to get yaw - check if it's stored directly or needs conversion
        try:
            # First try: yaw directly in orientation.z (uncommon but mentioned in original)
            yaw = pose_msg.pose.orientation.z
            # Sanity check - if this seems like a quaternion component, convert properly
            if abs(yaw) > 2 * np.pi:  # Likely a quaternion component
                yaw = quaternion_to_yaw(
                    pose_msg.pose.orientation.x,
                    pose_msg.pose.orientation.y,
                    pose_msg.pose.orientation.z,
                    pose_msg.pose.orientation.w
                )
        except:
            # Fallback: proper quaternion conversion
            yaw = quaternion_to_yaw(
                pose_msg.pose.orientation.x,
                pose_msg.pose.orientation.y,
                pose_msg.pose.orientation.z,
                pose_msg.pose.orientation.w
            )
        
        # Update robot position dot
        robot_dot.set_data([pos_x], [pos_y])
        
        # Remove old arrow and create new one
        robot_arrow.remove()
        arrow_length = 0.5
        arrow_dx = arrow_length * np.cos(yaw)
        arrow_dy = arrow_length * np.sin(yaw)
        
        robot_arrow = ax.arrow(pos_x, pos_y, arrow_dx, arrow_dy, 
                              head_width=0.3, head_length=0.3, 
                              fc='blue', ec='blue', linewidth=2)

        # Update Obstacle Polygons by removing old ones and adding new ones
        for patch in obstacle_patches:
            patch.remove()
        obstacle_patches = []
        
        if hasattr(obstacle_msg, 'markers') and obstacle_msg.markers:
            for marker in obstacle_msg.markers:
                if hasattr(marker, 'points') and marker.points:
                    # Transform each point from robot frame to odom frame
                    poly_points_odom = []
                    for p in marker.points:
                        odom_x, odom_y = transform_point_to_odom(
                            p.x, p.y, pos_x, pos_y, yaw
                        )
                        poly_points_odom.append([odom_x, odom_y])
                    
                    poly_points_odom = np.array(poly_points_odom)
                    
                    if len(poly_points_odom) >= 3:  # Need at least 3 points for a polygon
                        polygon = Polygon(poly_points_odom, closed=True, 
                                        color='red', alpha=0.5, edgecolor='darkred')
                        ax.add_patch(polygon)
                        obstacle_patches.append(polygon)
                
        # Update Timestamp Text
        try:
            timestamp = pose_msg.header.stamp.to_sec()
        except:
            timestamp = frame * 0.1  # Fallback
            
        time_text.set_text(f'Frame: {frame+1}/{len(synced_data)}\nTime: {timestamp:.2f}s')

        # Return all artists that have been modified
        return [robot_dot, robot_arrow, time_text] + obstacle_patches

    print("Creating animation... This may take a moment.")
    
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(synced_data),
        blit=False,  # blit=False is necessary when adding/removing patches
        interval=100,  # Milliseconds between frames
        repeat=True
    )

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    print("Animation finished.")

# --- Main execution block ---
if __name__ == "__main__":
    # IMPORTANT: Update this path to the location of your .bag file
    bag_file_path = "/home/sankeerth/Videos/2025-10-02-15-02-35.bag"
    
    synced_animation_data, full_robot_path = load_ros_data(bag_file_path)
    
    if synced_animation_data and full_robot_path:
        create_animation(synced_animation_data, full_robot_path)