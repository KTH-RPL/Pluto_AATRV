#!/home/sankeerth/pluto/bin/python3

# ROS1 Imports
import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point

# Standard Python Imports
import cv2
import numpy as np
import math
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
import traceback

# ==============================================================================
# The UVDetector and helper classes (unchanged from the original script)
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
        self.body2CamDepth = np.array(config['body_to_camera_depth']).reshape(4, 4)
        self.row_downsample = config['uv_detector']['row_downsample']
        self.col_scale = config['uv_detector']['col_scale']
        self.min_dist = config['uv_detector']['min_dist'] * 1000
        self.max_dist = config['uv_detector']['max_dist'] * 1000
        self.threshold_point = config['uv_detector']['threshold_point']
        self.threshold_line = config['uv_detector']['threshold_line']
        self.min_length_line = config['uv_detector']['min_length_line']
        self.depth_scale = config['depth_scale_factor']

    def detect(self, depth_image):
        self.box3Ds_robot_frame = []
        self.depth = depth_image
        self.extract_U_map()
        self.extract_bb()
        self.extract_3Dbox()
        self.transform_boxes_to_robot_frame()
        return self.box3Ds_robot_frame

    def extract_U_map(self):
        depth_rescale = cv2.resize(self.depth, (0, 0), fx=self.col_scale, fy=1, interpolation=cv2.INTER_NEAREST)
        hist_size = self.depth.shape[0] // self.row_downsample
        if hist_size == 0: return
        bin_width = math.ceil((self.max_dist - self.min_dist) / float(hist_size))
        if bin_width == 0: return
        self.U_map = np.zeros((hist_size, depth_rescale.shape[1]), dtype=np.uint16)
        depth_vals = depth_rescale * (1000.0 / self.depth_scale)
        valid_mask = (depth_vals > self.min_dist) & (depth_vals < self.max_dist)
        cols, rows = np.meshgrid(np.arange(depth_rescale.shape[1]), np.arange(depth_rescale.shape[0]))
        valid_cols = cols[valid_mask]
        valid_depths = depth_vals[valid_mask]
        bin_indices = ((valid_depths - self.min_dist) / bin_width).astype(np.int32)
        np.add.at(self.U_map, (bin_indices, valid_cols), 1)
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
        """
        NEW VERSION: Extracts oriented 2D polygons in the camera frame.
        """
        self.polygons_camera_frame = []
        if not hasattr(self, 'U_map') or not hasattr(self, 'bounding_box_U') or not self.bounding_box_U:
            return

        histSize = self.depth.shape[0] / self.row_downsample
        if histSize == 0: return
        bin_width = math.ceil((self.max_dist - self.min_dist) / histSize)
        if bin_width == 0: return

        # We need the labeled U-map to find the points for each blob
        u_min = self.threshold_point * self.row_downsample
        interest_mask = self.U_map >= u_min
        labels, num_labels = ndi.label(interest_mask, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))

        for i in range(1, num_labels + 1):
            # Get all the (row, col) points for the current blob
            points = np.argwhere(labels == i)
            if len(points) < self.min_length_line: # Filter small blobs
                continue

            # Invert points for OpenCV (x, y) convention -> (col, row)
            points_for_cv = points[:, [1, 0]].astype(np.float32)

            # Find the minimum area oriented rectangle
            # rect is ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(points_for_cv)
            
            # Get the 4 corner points of the rectangle
            # box_points are in (col, row) format
            box_points_uv = cv2.boxPoints(rect)

            # Now, project these 4 corners into 3D camera space
            polygon_3d = []
            for u, v_depth in box_points_uv:
                # v_depth is the row in U-Map, representing depth
                # u is the column in U-Map, representing horizontal angle
                
                # Convert back to real-world depth (Z in camera frame)
                Z = (v_depth * bin_width + self.min_dist) / 1000.0 # in meters

                # Convert back to original image column
                u_image = u / self.col_scale
                
                # Use pinhole model to find X
                # X = (u_image - px) * Z / fx
                X = (u_image - self.px) * Z / self.fx
                
                # For the 2D footprint, we assume the obstacle is on the ground plane.
                # A more complex method would find the actual Y height from the depth image,
                # but for navigation, the ground-projected footprint is what matters most.
                # Let's find the camera's height above ground from the transform for a simple projection.
                # Assuming robot frame Z=0 is ground. The camera's Z position in the robot frame
                # corresponds to its height.
                cam_pos_in_robot = np.linalg.inv(self.body2CamDepth) @ np.array([0,0,0,1])
                cam_height = cam_pos_in_robot[2] # This is a simplification
                
                # Project onto a plane slightly below the camera
                # Y = (v_image - py) * Z / fy
                # Here we use a simplification: Y is derived from camera height.
                # Let's assume the points are on the ground relative to the camera.
                # A robust way is to find the ground plane, but we can approximate.
                # For this example, we'll set Y to 0, which projects the obstacle
                # footprint onto the camera's horizontal plane. When transformed to the
                # robot frame, this will approximate the ground footprint if the camera
                # is not heavily pitched.
                Y = 0 # Simplified: Assumes obstacle base is level with camera center
                      # This is a weak point, but better than AABB.

                polygon_3d.append([X, Y, Z, 1.0])
            
            self.polygons_camera_frame.append(np.array(polygon_3d))

    def transform_boxes_to_robot_frame(self):
        """
        NEW VERSION: Transforms the 4 corners of each polygon to the robot frame.
        """
        self.box3Ds_robot_frame = [] # This will now store lists of 4 points
        if not hasattr(self, 'polygons_camera_frame'):
            return
            
        for poly_cam in self.polygons_camera_frame:
            # poly_cam is a 4x4 array (each row is [X, Y, Z, 1])
            # Transform all 4 points at once
            poly_robot = (self.body2CamDepth @ poly_cam.T).T
            
            # We only need the 2D footprint for the checker (X, Y)
            footprint_2d = poly_robot[:, :2]
            self.box3Ds_robot_frame.append(footprint_2d)


# ==============================================================================
# The ROS1 Node
# ==============================================================================

class UvDetectorNodeROS1:
    def __init__(self):
        rospy.init_node('uv_detector_node', anonymous=True)

        # --- Get parameters from ROS Param Server ---
        depth_topic = rospy.get_param('~depth_topic', '/rsD455_node0/depth/image_rect_raw')
        obstacle_topic = rospy.get_param('~obstacle_topic', '/detected_obstacles')
        self.output_frame_id = rospy.get_param('~output_frame_id', 'base_link')
        self.safety_buffer = rospy.get_param('~safety_buffer', 0.1)  # Extra size in meters for each dimension

        # --- Load detector configuration from parameters ---
        # This makes the node highly configurable via launch files
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

        # --- Create subscriber and publisher ---
        self.subscriber = rospy.Subscriber(depth_topic, Image, self.depth_callback, queue_size=1)
        self.publisher = rospy.Publisher(obstacle_topic, MarkerArray, queue_size=10)

        rospy.loginfo("UV Detector Node for ROS1 initialized.")
        rospy.loginfo(f"Subscribing to depth topic: {depth_topic}")
        rospy.loginfo(f"Publishing obstacle markers on: {obstacle_topic}")

    def depth_callback(self, msg):
        """Main callback to process incoming depth images."""
        try:
            # Use cv_bridge to convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        try:
            # Run the detection algorithm
            detected_boxes = self.detector.detect(depth_image)

            # Convert detections to MarkerArray message
            marker_array_msg = self.create_obstacle_markers(detected_boxes, msg.header)

            # Publish the result
            self.publisher.publish(marker_array_msg)
        
        except Exception as e:
            rospy.logerr(f"Failed to process depth image: {e}")
            traceback.print_exc()


    def create_obstacle_markers(self, polygons, header):
        """Converts a list of 4-point polygons into a MarkerArray of LINE_STRIPs."""
        marker_array = MarkerArray()
        
        delete_marker = Marker()
        delete_marker.header.frame_id = self.output_frame_id
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for i, footprint in enumerate(polygons):
            marker = Marker()
            marker.header = header
            marker.header.frame_id = self.output_frame_id
            marker.ns = "uv_obstacles"
            marker.id = i
            # *** MODIFICATION: Use LINE_STRIP to draw a polygon ***
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD

            # A LINE_STRIP connects a series of points
            marker.points = []
            for point_xy in footprint:
                p = Point()
                p.x = point_xy[0]
                p.y = point_xy[1]
                p.z = 0.05  # Place slightly above the ground plane
                marker.points.append(p)
            
            # Add the first point again to close the loop
            p_first = Point()
            p_first.x = footprint[0][0]
            p_first.y = footprint[0][1]
            p_first.z = 0.05
            marker.points.append(p_first)

            # Line width
            marker.scale.x = 0.05  # Thickness of the line

            # Color (e.g., semi-transparent orange)
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.9

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