#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import os
import time # For timestamp in filename if desired

# Mapping from ROS PointField types to PCL types and sizes
# Add more mappings if your PointCloud2 uses other types
ROS_TO_PCD_TYPE_MAP = {
    PointField.INT8: ('I', 1),
    PointField.UINT8: ('U', 1),
    PointField.INT16: ('I', 2),
    PointField.UINT16: ('U', 2),
    PointField.INT32: ('I', 4),
    PointField.UINT32: ('U', 4),
    PointField.FLOAT32: ('F', 4),
    PointField.FLOAT64: ('F', 8)
}

# ==========================================================
# PCD Header Writing Function
# ==========================================================
def write_pcd_header(f, fields, width, height, viewpoint, points_count, data_type='ascii'):
    """Writes the PCD header."""
    f.write("# .PCD v0.7 - Point Cloud Data file format\n")
    f.write("VERSION 0.7\n")

    field_names = [field.name for field in fields]
    pcd_types = []
    pcd_sizes = []
    pcd_counts = []

    # Process fields for header lines
    current_offset = 0
    processed_fields_str = [] # Store processed field names for FIELDS line
    for field in fields:
        if field.offset != current_offset:
             rospy.logwarn("Fields are not contiguous or ordered correctly, PCD might be invalid")
             # Attempt to handle non-standard packing, assuming order is correct
             # This might fail for complex non-standard clouds

        if field.datatype not in ROS_TO_PCD_TYPE_MAP:
            rospy.logerr(f"Unsupported ROS PointField type: {field.datatype}. Cannot generate PCD.")
            return False # Indicate header writing failure

        pcd_type, pcd_size = ROS_TO_PCD_TYPE_MAP[field.datatype]

        # Handle multi-count fields (e.g., normals, curvature)
        # Assumes count > 1 means multiple elements of the same type
        effective_count = field.count if field.count > 0 else 1
        if effective_count > 1:
            # If a field has multiple elements (like 'normals' often does),
            # PCL usually expects them listed individually in the header
            # (e.g., normal_x, normal_y, normal_z) even if the ROS message
            # packs them under a single name. We'll list the base name here,
            # but the COUNT line reflects the structure.
             processed_fields_str.append(field.name)
             pcd_types.append(pcd_type)
             pcd_sizes.append(str(pcd_size))
             pcd_counts.append(str(effective_count))
        else:
            # Standard case (count=1)
             processed_fields_str.append(field.name)
             pcd_types.append(pcd_type)
             pcd_sizes.append(str(pcd_size))
             pcd_counts.append('1') # Explicitly 1

        # Increment offset check
        current_offset += effective_count * pcd_size


    f.write(f"FIELDS {' '.join(processed_fields_str)}\n")
    f.write(f"SIZE {' '.join(pcd_sizes)}\n")
    f.write(f"TYPE {' '.join(pcd_types)}\n")
    f.write(f"COUNT {' '.join(pcd_counts)}\n") # Correctly handle multi-element fields
    f.write(f"WIDTH {width}\n")
    f.write(f"HEIGHT {height}\n")
    f.write(f"VIEWPOINT {viewpoint}\n") # Typically "0 0 0 1 0 0 0"
    f.write(f"POINTS {points_count}\n")
    f.write(f"DATA {data_type}\n")
    return True # Indicate header writing success

# ==========================================================
# PointCloud Saver Class
# ==========================================================
class PointCloudSaverNoPCL:
    def __init__(self):
        rospy.init_node('pointcloud_saver_no_pcl_node', anonymous=True)

        # --- Parameters ---
        self.pcd_topic = '/ouster/points' # Example topic
        self.save_dir = "/home/sankeerth/catkin_ws/src/Pluto_AATRV/data/pcds/"
        # --- End Parameters ---

        self.cloud_count = 0

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            try:
                os.makedirs(self.save_dir)
                rospy.loginfo(f"Created directory: {self.save_dir}")
            except OSError as e:
                rospy.logerr(f"Failed to create directory {self.save_dir}: {e}")
                rospy.signal_shutdown("Failed to create save directory")
                return

        # Subscribe to the PointCloud2 topic
        rospy.Subscriber(self.pcd_topic, PointCloud2, self.pointcloud_callback)

        rospy.loginfo(f"Subscribed to {self.pcd_topic}")
        rospy.loginfo(f"Saving ASCII PCD files to: {self.save_dir}")
        rospy.loginfo("Press Ctrl+C to stop saving.")

    def pointcloud_callback(self, ros_cloud):
        rospy.logdebug(f"Received point cloud message seq={ros_cloud.header.seq}") # Use debug level

        filename = os.path.join(self.save_dir, f"{ros_cloud.header.seq}_{ros_cloud.header.stamp.secs}_{ros_cloud.header.stamp.nsecs}_{ros_cloud.header.frame_id}.pcd")
        field_names = [field.name for field in ros_cloud.fields]

        try:
            # Use sensor_msgs.point_cloud2.read_points to iterate through points
            # Set skip_nans=True if you want to exclude NaN points automatically.
            # If skip_nans=True, the POINTS count in the header must reflect the
            # number of *valid* points, not width*height.
            # Let's keep skip_nans=False for now to write all points,
            # matching the width*height dimensions in the header.
            point_generator = pc2.read_points(ros_cloud, field_names=field_names, skip_nans=False)

            # Determine total points (width * height)
            total_points = ros_cloud.width * ros_cloud.height

            # --- Write PCD File ---
            with open(filename, 'w') as f:
                # 1. Write Header
                # Assuming default viewpoint, change if needed
                viewpoint = "0 0 0 1 0 0 0"
                header_success = write_pcd_header(f, ros_cloud.fields, ros_cloud.width, ros_cloud.height, viewpoint, total_points, data_type='ascii')

                if not header_success:
                     rospy.logerr(f"Failed to write PCD header for {filename}. Skipping cloud.")
                     # Optionally delete the incomplete file
                     try: os.remove(filename)
                     except OSError: pass
                     return # Don't proceed if header failed

                # 2. Write Data (ASCII)
                points_written = 0
                for point in point_generator:
                    # point is a tuple with values for fields in field_names
                    # Format the point data as a space-separated string
                    # Handle potential NaN/Inf floats if skip_nans=False
                    line = ' '.join(map(str, point))
                    f.write(line + '\n')
                    points_written += 1

                # Verification (optional but good)
                if points_written != total_points:
                     rospy.logwarn(f"Warning: Number of points written ({points_written}) does not match header POINTS ({total_points}) for {filename}. This might happen if read_points internally skips malformed data.")
                     # Consider adjusting the header POINTS value retroactively if this is expected,
                     # but that requires reading the file back or storing points, making it complex.

            rospy.loginfo(f"Saved {points_written} points to ASCII PCD: {filename}")
            self.cloud_count += 1

        except (pc2.PointCloudException, TypeError, ValueError, IOError) as e:
            rospy.logerr(f"Failed to process or save point cloud {self.cloud_count}: {e}")
        except Exception as e:
             rospy.logerr(f"An unexpected error occurred while saving cloud {self.cloud_count}: {e}")


if __name__ == '__main__':
    try:
        pc_saver = PointCloudSaverNoPCL()
        rospy.spin() # Keep the node alive until Ctrl+C
    except rospy.ROSInterruptException:
        rospy.loginfo("PointCloud saver (no PCL) node shutting down.")
    except Exception as e:
        rospy.logerr(f"An unhandled exception occurred in main: {e}")
