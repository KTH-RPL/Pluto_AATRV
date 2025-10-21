#!/home/nuv/catkin_noetic/src/Pluto_AATRV/detector/detector/bin/python3

# ROS1 Imports
import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped

# Standard Python Imports
import cv2
import numpy as np
import math
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
import traceback
import time

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    rospy.logwarn("Numba not available. Install with: pip install numba")

# ==============================================================================
# Numba-optimized helper functions
# ==============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def _process_valid_depth_fast(depth_vals, min_dist, max_dist, bin_width):
        """Fast depth binning using Numba."""
        valid_mask = (depth_vals > min_dist) & (depth_vals < max_dist)
        return valid_mask, ((depth_vals - min_dist) / bin_width).astype(np.int32)

    @jit(nopython=True, parallel=True)
    def _transform_corners_fast(corners_cam, body2cam):
        """Fast matrix multiplication for corner transformation."""
        n = corners_cam.shape[0]
        corners_robot = np.zeros_like(corners_cam)
        for i in prange(n):
            for j in range(4):
                for k in range(4):
                    corners_robot[i, j] += body2cam[j, k] * corners_cam[i, k]
        return corners_robot

else:
    def _process_valid_depth_fast(depth_vals, min_dist, max_dist, bin_width):
        valid_mask = (depth_vals > min_dist) & (depth_vals < max_dist)
        return valid_mask, ((depth_vals - min_dist) / bin_width).astype(np.int32)

    def _transform_corners_fast(corners_cam, body2cam):
        return (body2cam @ corners_cam.T).T

# ==============================================================================
# The UVDetector and helper classes
# ==============================================================================

class UVbox:
    """Helper class for a bounding box in the U-map."""
    def __init__(self, seg_id=0, row=0, left=0, right=0):
        self.id = seg_id
        self.toppest_parent_id = seg_id
        self.bb = (left, row, right - left, 1)

def merge_two_UVbox(father, son):
    f_x, f_y, f_w, f_h = father.bb
    s_x, s_y, s_w, s_h = son.bb
    left = min(f_x, s_x)
    top = min(f_y, s_y)
    right = max(f_x + f_w, s_x + s_w)
    bottom = max(f_y + f_h, s_y + s_h)
    father.bb = (left, top, right - left, bottom - top)
    return father

class UVDetector:
    """A Python implementation of the UV-detector for obstacle detection."""
    def __init__(self, config):
        self.fx = config['depth_intrinsics'][0]
        self.fy = config['depth_intrinsics'][1]
        self.px = config['depth_intrinsics'][2]
        self.py = config['depth_intrinsics'][3]
        self.body2CamDepth = np.array(config['body_to_camera_depth'], dtype=np.float32).reshape(4, 4)
        self.row_downsample = config['uv_detector']['row_downsample']
        self.col_scale = config['uv_detector']['col_scale']
        self.min_dist = config['uv_detector']['min_dist'] * 1000
        self.max_dist = config['uv_detector']['max_dist'] * 1000
        self.threshold_point = config['uv_detector']['threshold_point']
        self.threshold_line = config['uv_detector']['threshold_line']
        self.min_length_line = config['uv_detector']['min_length_line']
        self.depth_scale = config['depth_scale_factor']

    def detect(self, depth_image, robot_pose):
        self.robot_pose = robot_pose
        self.box3Ds_robot_frame = []
        self.depth = depth_image.astype(np.float32)
        self.extract_U_map()
        self.extract_bb()
        self.extract_3Dbox()
        self.transform_boxes_to_robot_frame()
        self.transform_boxes_to_world_frame()
        return self.box3Ds_world_frame

    def extract_U_map(self):
        depth_rescale = cv2.resize(self.depth, (0, 0), fx=self.col_scale, fy=1, interpolation=cv2.INTER_NEAREST)
        hist_size = self.depth.shape[0] // self.row_downsample
        if hist_size == 0: return
        bin_width = math.ceil((self.max_dist - self.min_dist) / float(hist_size))
        if bin_width == 0: return
        
        self.U_map = np.zeros((hist_size, depth_rescale.shape[1]), dtype=np.uint16)
        depth_vals = depth_rescale * (1000.0 / self.depth_scale)
        
        # Use Numba-optimized function
        valid_mask, bin_indices = _process_valid_depth_fast(
            depth_vals, self.min_dist, self.max_dist, bin_width
        )
        
        valid_cols = np.arange(depth_rescale.shape[1])[np.any(valid_mask, axis=0)]
        for col in valid_cols:
            col_valid_mask = valid_mask[:, col]
            col_bins = bin_indices[col_valid_mask, col]
            np.add.at(self.U_map[:, col], col_bins, 1)
        
        self.U_map = self.U_map.astype(np.uint8)
        self.U_map = cv2.GaussianBlur(self.U_map, (5, 9), 10)

    def extract_bb(self):
        self.bounding_box_U = []
        if not hasattr(self, 'U_map'): return
        u_min = self.threshold_point * self.row_downsample
        interest_mask = self.U_map >= u_min
        labels, num_labels = ndi.label(interest_mask, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
        if num_labels == 0: return
        slices = ndi.find_objects(labels)
        line_sums = ndi.sum_labels(self.U_map, labels, index=np.arange(1, num_labels + 1))
        line_maxes = ndi.maximum(self.U_map, labels, index=np.arange(1, num_labels + 1))
        uv_boxes = []
        seg_id = 0
        for i in range(num_labels):
            s = slices[i]
            length_line = s[1].stop - s[1].start
            if length_line > self.min_length_line and (line_maxes[i] == 0 or line_sums[i] > self.threshold_line * line_maxes[i]):
                seg_id += 1
                new_box = UVbox(seg_id, s[0].start, s[1].start, s[1].stop - 1)
                uv_boxes.append(new_box)
        if not uv_boxes: return
        mask = np.zeros(self.U_map.shape, dtype=int)
        for box in uv_boxes:
            x, y, w, h = box.bb
            mask[y, x:x+w] = box.id
        for box in uv_boxes:
            if box.bb[1] > 0:
                parent_ids = set()
                row = box.bb[1]
                for c in range(box.bb[0], box.bb[0] + box.bb[2]):
                    if mask[row - 1, c] != 0:
                        parent_ids.add(uv_boxes[mask[row - 1, c] - 1].toppest_parent_id)
                if parent_ids:
                    min_parent_id = min(parent_ids)
                    box.toppest_parent_id = min_parent_id
                    for parent_id in parent_ids:
                        if parent_id != min_parent_id:
                            for b in uv_boxes:
                                if b.toppest_parent_id == parent_id:
                                    b.toppest_parent_id = min_parent_id
        box_groups = {}
        for box in uv_boxes:
            box_groups.setdefault(box.toppest_parent_id, []).append(box)
        for group_id, group_boxes in box_groups.items():
            if not group_boxes: continue
            merged_box = group_boxes[0]
            for i in range(1, len(group_boxes)):
                merged_box = merge_two_UVbox(merged_box, group_boxes[i])
            if merged_box.bb[2] * merged_box.bb[3] >= 25:
                self.bounding_box_U.append(merged_box.bb)

    def extract_3Dbox(self):
        self.box3Ds_camera_frame = []
        if not hasattr(self, 'U_map') or not hasattr(self, 'bounding_box_U'): return
        depth_resize = cv2.resize(self.depth, (0, 0), fx=self.col_scale, fy=1, interpolation=cv2.INTER_NEAREST)
        histSize = self.depth.shape[0] / self.row_downsample
        if histSize == 0: return
        bin_width = math.ceil((self.max_dist - self.min_dist) / histSize)
        if bin_width == 0: return
        num_check = 15
        depth_vals_scaled = (depth_resize / self.depth_scale) * 1000.0
        for bb in self.bounding_box_U:
            x, y, width, height = bb
            depth_in_near = (y * bin_width + self.min_dist)
            depth_of_depth = height * bin_width
            depth_in_far = depth_of_depth * 1 + depth_in_near
            patch = depth_vals_scaled[:, x : x + width]
            valid_depth_mask = (patch >= depth_in_near) & (patch <= depth_in_far)
            if not np.any(valid_depth_mask): continue
            consecutive_sum = np.lib.stride_tricks.sliding_window_view(valid_depth_mask, num_check, axis=0).sum(axis=2)
            valid_sequences = consecutive_sum == num_check
            if not np.any(valid_sequences): continue
            row_indices, _ = np.where(valid_sequences)
            y_up = np.min(row_indices)
            y_down = np.max(row_indices) + num_check
            if y_down <= y_up: continue
            im_frame_x = (x + width / 2) / self.col_scale
            im_frame_x_width = width / self.col_scale
            Y_w = (depth_in_near + depth_in_far) / 2
            im_frame_y = (y_down + y_up) / 2
            im_frame_y_width = y_down - y_up
            if im_frame_y_width <= 0: continue
            curr_box = {
                'x': (im_frame_x - self.px) * Y_w / self.fx, 'y': (im_frame_y - self.py) * Y_w / self.fy, 'z': Y_w,
                'x_width': im_frame_x_width * Y_w / self.fx, 'y_width': im_frame_y_width * Y_w / self.fy,
                'z_width': depth_in_far - depth_in_near
            }
            for key in curr_box: curr_box[key] /= 1000.0
            self.box3Ds_camera_frame.append(curr_box)

    def transform_boxes_to_robot_frame(self):
        for box_cam in self.box3Ds_camera_frame:
            dx, dy, dz = box_cam['x_width'] / 2, box_cam['y_width'] / 2, box_cam['z_width'] / 2
            corners_cam = np.array([
                [box_cam['x']-dx, box_cam['y']-dy, box_cam['z']-dz, 1],
                [box_cam['x']+dx, box_cam['y']-dy, box_cam['z']-dz, 1],
                [box_cam['x']-dx, box_cam['y']+dy, box_cam['z']-dz, 1],
                [box_cam['x']+dx, box_cam['y']+dy, box_cam['z']-dz, 1],
                [box_cam['x']-dx, box_cam['y']-dy, box_cam['z']+dz, 1],
                [box_cam['x']+dx, box_cam['y']-dy, box_cam['z']+dz, 1],
                [box_cam['x']-dx, box_cam['y']+dy, box_cam['z']+dz, 1],
                [box_cam['x']+dx, box_cam['y']+dy, box_cam['z']+dz, 1]
            ], dtype=np.float32)
            
            # Use Numba-optimized corner transformation
            corners_robot = _transform_corners_fast(corners_cam, self.body2CamDepth)
            
            min_coords = np.min(corners_robot[:,:3], axis=0)
            max_coords = np.max(corners_robot[:,:3], axis=0)
            center_robot = (min_coords + max_coords) / 2
            size_robot = max_coords - min_coords
            
            self.box3Ds_robot_frame.append({
                'x': center_robot[0], 'y': center_robot[1], 'z': center_robot[2],
                'x_width': size_robot[0], 'y_width': size_robot[1], 'z_width': size_robot[2]
            })

    def transform_boxes_to_world_frame(self):
        x_robot, y_robot, theta_robot = self.robot_pose
        cos_theta, sin_theta = np.cos(theta_robot), np.sin(theta_robot)
        self.box3Ds_world_frame = []
        for box in self.box3Ds_robot_frame:
            x_r, y_r = box['x'], box['y']
            x_w = cos_theta * x_r + sin_theta * y_r + x_robot
            y_w = sin_theta * x_r - cos_theta * y_r + y_robot
            self.box3Ds_world_frame.append({
                'x': x_w, 'y': y_w, 'z': box['z'],
                'x_width': box['x_width'], 'y_width': box['y_width'], 'z_width': box['z_width']
            })
        return self.box3Ds_world_frame

# ==============================================================================
# The ROS1 Node with Frequency Limiting
# ==============================================================================

class UvDetectorNodeROS1:
    def __init__(self):
        rospy.init_node('uv_detector_node', anonymous=True)

        # --- Frequency limiting at 2Hz ---
        self.target_hz = rospy.get_param('~target_hz', 2.0)
        self.min_time_between_processing = 1.0 / self.target_hz
        self.last_process_time = 0.0

        # --- Get parameters from ROS Param Server ---
        depth_topic = rospy.get_param('~depth_topic', '/rsD455_node0/depth/image_rect_raw')
        obstacle_topic = rospy.get_param('~obstacle_topic', '/detected_obstacles')
        self.output_frame_id = rospy.get_param('~output_frame_id', 'base_link')
        self.safety_buffer = rospy.get_param('~safety_buffer', 0.1)

        # --- Load detector configuration from parameters ---
        detector_config = {
            'depth_intrinsics': rospy.get_param('~depth_intrinsics'),
            'body_to_camera_depth': rospy.get_param('~body_to_camera_depth'),
            'uv_detector': {
                'row_downsample': rospy.get_param('~uv_detector/row_downsample', 4),
                'col_scale': rospy.get_param('~uv_detector/col_scale', 0.5),
                'min_dist': rospy.get_param('~uv_detector/min_dist', 0.1),
                'max_dist': rospy.get_param('~uv_detector/max_dist', 8.0),
                'threshold_point': rospy.get_param('~uv_detector/threshold_point', 3),
                'threshold_line': rospy.get_param('~uv_detector/threshold_line', 2),
                'min_length_line': rospy.get_param('~uv_detector/min_length_line', 6),
            },
            'depth_scale_factor': rospy.get_param('~depth_scale_factor', 1000.0)
        }
        self.detector = UVDetector(detector_config)
        self.bridge = CvBridge()
        self.current_robot_pose = (0.0, 0.0, 0.0)

        # --- Create subscriber and publisher ---
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.robot_pose_callback, queue_size=1)
        self.subscriber = rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
        self.publisher = rospy.Publisher(obstacle_topic, MarkerArray, queue_size=10)

        rospy.loginfo("UV Detector Node for ROS1 initialized.")
        rospy.loginfo(f"Numba optimization: {'ENABLED' if NUMBA_AVAILABLE else 'DISABLED'}")
        rospy.loginfo(f"Processing frequency: {self.target_hz} Hz")
        rospy.loginfo(f"Subscribing to depth topic: {depth_topic}")
        rospy.loginfo(f"Publishing obstacle markers on: {obstacle_topic}")

    def robot_pose_callback(self, msg):
        """Callback to update the robot's current pose."""
        x = msg.pose.position.x
        y = msg.pose.position.y
        yaw = msg.pose.orientation.z
        self.current_robot_pose = (x, y, yaw)

    def depth_callback(self, msg):
        """Main callback to process incoming depth images with frequency limiting."""
        current_time = time.time()
        
        # Check if enough time has passed since last processing
        if current_time - self.last_process_time < self.min_time_between_processing:
            return  # Skip this frame
        
        self.last_process_time = current_time
        
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        try:
            detected_boxes = self.detector.detect(depth_image, self.current_robot_pose)
            marker_array_msg = self.create_obstacle_markers(detected_boxes, msg.header)
            self.publisher.publish(marker_array_msg)
        
        except Exception as e:
            rospy.logerr(f"Failed to process depth image: {e}")
            traceback.print_exc()

    def create_obstacle_markers(self, boxes, header):
        """Converts a list of 3D boxes into a MarkerArray of 2D flat rectangles (CUBES)."""
        marker_array = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.header.frame_id = "odom"
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, box in enumerate(boxes):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = "odom"
            marker.ns = "uv_obstacles"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = box['x']
            marker.pose.position.y = box['y']
            marker.pose.position.z = 0.05
            marker.pose.orientation.w = 1.0

            marker.scale.x = box['x_width'] + self.safety_buffer
            marker.scale.y = box['y_width'] + self.safety_buffer
            marker.scale.z = 0.1

            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.7

            marker.lifetime = rospy.Duration(0.5)
            
            marker_array.markers.append(marker)
            
        return marker_array

def main():
    try:
        node = UvDetectorNodeROS1()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyError as e:
        rospy.logerr(f"Missing required ROS parameter: {e}. Please check your launch file.")
        rospy.logerr("You must provide 'depth_intrinsics' and 'body_to_camera_depth' parameters.")
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()