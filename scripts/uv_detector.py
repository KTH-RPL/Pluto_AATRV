import cv2
import numpy as np
import math
from pathlib import Path
from rosbags.highlevel import AnyReader
import time
import scipy.ndimage as ndi

class UVbox:
    """Helper class for a bounding box in the U-map."""
    def __init__(self, seg_id=0, row=0, left=0, right=0):
        self.id = seg_id
        self.toppest_parent_id = seg_id
        self.bb = (left, row, right - left, 1)

def merge_two_UVbox(father, son):
    """Merges two UVbox objects into a single bounding box."""

    f_x, f_y, f_w, f_h = father.bb
    s_x, s_y, s_w, s_h = son.bb

    left = min(f_x, s_x)
    top = min(f_y, s_y)
    right = max(f_x + f_w, s_x + s_w)
    bottom = max(f_y + f_h, s_y + s_h)
    
    father.bb = (left, top, right - left, bottom - top)
    return father

class UVDetector:
    """
    A Python implementation of the UV-detector for obstacle detection from a depth image.
    This class replicates the core functionality of the C++ UVdetector provided.
    """
    def __init__(self, config):
        """
        Initializes the UVDetector with configuration parameters.
        Args:
            config (dict): A dictionary containing all necessary parameters.
        """
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
        """Performs the full UV detection pipeline on a given depth image."""
        self.bounding_box_U, self.bounding_box_D = [], []
        self.box3Ds_camera_frame, self.box3Ds_robot_frame = [], []
        
        self.depth = depth_image
        self.extract_U_map()
        self.extract_bb()
        self.extract_3Dbox()
        self.transform_boxes_to_robot_frame()
        return self.box3Ds_robot_frame

    def extract_U_map(self):
        """Extracts the U-map from the depth image using vectorized NumPy operations."""
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
        """Extracts bounding boxes from the U-map using vectorized line grouping."""
        if not hasattr(self, 'U_map'): return

        u_min = self.threshold_point * self.row_downsample
        
        interest_mask = self.U_map >= u_min

        labels, num_labels = ndi.label(interest_mask, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
        
        if num_labels == 0:
            return

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
                row = s[0].start
                left_col = s[1].start
                right_col = s[1].stop -1
                new_box = UVbox(seg_id, row, left_col, right_col)
                uv_boxes.append(new_box)

        if not uv_boxes:
            return

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
            
            box_area = merged_box.bb[2] * merged_box.bb[3]
            if box_area >= 25:
                self.bounding_box_U.append(merged_box.bb)

    def extract_3Dbox(self):
        """Extracts 3D bounding boxes using vectorized patch processing."""
        if not hasattr(self, 'U_map'): return
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
            depth_in_far = depth_of_depth * 1.3 + depth_in_near

            patch = depth_vals_scaled[:, x : x + width]
            
            valid_depth_mask = (patch >= depth_in_near) & (patch <= depth_in_far)
            
            consecutive_sum = np.lib.stride_tricks.sliding_window_view(valid_depth_mask, num_check, axis=0).sum(axis=2)
            valid_sequences = consecutive_sum == num_check

            row_indices, _ = np.where(valid_sequences)

            if row_indices.size == 0:
                continue

            y_up = np.min(row_indices)
            y_down = np.max(row_indices) + num_check

            if y_down <= y_up: continue

            self.bounding_box_D.append((int(x / self.col_scale), int(y_up), int(width / self.col_scale), int(y_down - y_up)))

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
        """Transforms the 3D bounding boxes from camera frame to robot frame."""
        for box_cam in self.box3Ds_camera_frame:
            dx, dy, dz = box_cam['x_width'] / 2, box_cam['y_width'] / 2, box_cam['z_width'] / 2
            corners_cam = np.array([
                [box_cam['x']-dx, box_cam['y']-dy, box_cam['z']-dz, 1],[box_cam['x']+dx, box_cam['y']-dy, box_cam['z']-dz, 1],
                [box_cam['x']-dx, box_cam['y']+dy, box_cam['z']-dz, 1],[box_cam['x']+dx, box_cam['y']+dy, box_cam['z']-dz, 1],
                [box_cam['x']-dx, box_cam['y']-dy, box_cam['z']+dz, 1],[box_cam['x']+dx, box_cam['y']-dy, box_cam['z']+dz, 1],
                [box_cam['x']-dx, box_cam['y']+dy, box_cam['z']+dz, 1],[box_cam['x']+dx, box_cam['y']+dy, box_cam['z']+dz, 1]
            ])
            corners_robot = (self.body2CamDepth @ corners_cam.T).T
            min_coords, max_coords = np.min(corners_robot[:,:3], axis=0), np.max(corners_robot[:,:3], axis=0)
            center_robot, size_robot = (min_coords + max_coords) / 2, max_coords - min_coords
            self.box3Ds_robot_frame.append({
                'x': center_robot[0], 'y': center_robot[1], 'z': center_robot[2],
                'x_width': size_robot[0], 'y_width': size_robot[1], 'z_width': size_robot[2]
            })

    def get_visualizations(self):
        """Generates visual representations of the depth map and U-map."""
        u_map_vis = np.zeros((1,1,3), dtype=np.uint8)
        if hasattr(self, 'U_map') and self.U_map.size > 0:
            u_map_vis = cv2.normalize(self.U_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            u_map_vis = cv2.applyColorMap(u_map_vis, cv2.COLORMAP_JET)
            for bb_tuple in self.bounding_box_U:
                cv2.rectangle(u_map_vis, bb_tuple, (0, 255, 0), 1)

        depth_vis = self.depth.copy()
        min_val, max_val, _, _ = cv2.minMaxLoc(depth_vis)
        if max_val > 0: depth_vis = (depth_vis / max_val * 255).astype(np.uint8)
        depth_vis_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_vis, alpha=1), cv2.COLORMAP_BONE)
        for rect_tuple in self.bounding_box_D:
            cv2.rectangle(depth_vis_color, rect_tuple, (0, 255, 0), 2)

        return u_map_vis, depth_vis_color

class Config:
    """Holds all configuration parameters."""
    def __init__(self):
        self.BAG_FILE_PATH = 'Videos/data.bag'
        self.DEPTH_TOPIC = '/rsD455_node0/depth/image_rect_raw'
        
        self.PARAMS = {
            'depth_intrinsics': [392.18686447169443, 393.05974495542637, 323.7744459749295, 236.8676530416521],
            'body_to_camera_depth': [
                -0.0022578997337579657, -0.0009717964977189038, 0.9999969787456161, 0.06848378638223135,
                0.9999625178902136, 0.008356334051211077, 0.0022659426050660346, -0.054542385544686066,
                -0.00835851083968775, 0.9999646130202771, 0.0009528923083064839, 0.09424585041267339,
                0.0, 0.0, 0.0, 1.0
            ],
            'uv_detector': {
                'row_downsample': 4, 'col_scale': 0.5, 'min_dist': 0.1, 'max_dist': 8.0,
                'threshold_point': 3, 'threshold_line': 2, 'min_length_line': 6,
            },
            'depth_scale_factor': 1000.0
        }

def ros_image_to_numpy(msg, depth_scale_factor=1000.0):
    """Converts a ROS Image message to a NumPy array."""
    if msg.encoding == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    elif msg.encoding == '32FC1':
        f32_image = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        return (f32_image * depth_scale_factor).astype(np.uint16)
    else:
        print(f"Unsupported image encoding: {msg.encoding}. Skipping frame.")
        return None

if __name__ == '__main__':
    
    cfg = Config()
    detector = UVDetector(cfg.PARAMS)
    
    bag_path = Path(cfg.BAG_FILE_PATH)
    if not bag_path.exists():
        print(f"Error: Bag file not found at '{bag_path}'")
        exit()

    try:
        with AnyReader([bag_path]) as reader:
            connections = [c for c in reader.connections if c.topic == cfg.DEPTH_TOPIC]
            if not connections:
                print(f"Topic '{cfg.DEPTH_TOPIC}' not found in the bag file.")
                exit()
            
            print(f"Reading depth images from topic '{cfg.DEPTH_TOPIC}' in '{cfg.BAG_FILE_PATH}'...")
            print("Press 'q' in the display window to quit.")

            frame_count = 0
            start_time = time.time()
            last_fps_time = start_time

            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                depth_image = ros_image_to_numpy(msg, cfg.PARAMS['depth_scale_factor'])
                if depth_image is None:
                    continue

                frame_start = time.time()
                detected_boxes = detector.detect(depth_image)
                frame_end = time.time()
                
                frame_count += 1
                current_time = time.time()
                
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - start_time)
                    processing_time_ms = (frame_end - frame_start) * 1000
                    print(f"FPS: {fps:.2f} | Processing time: {processing_time_ms:.2f}ms")
                    last_fps_time = current_time

                u_map_viz, depth_viz = detector.get_visualizations()
                cv2.imshow("Depth with Detections", depth_viz)
                if u_map_viz.size > 1:
                    cv2.imshow("U-Map with Detections", u_map_viz)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            total_time = time.time() - start_time
            if total_time > 0:
                final_fps = frame_count / total_time
                print(f"\nFinal Statistics:")
                print(f"Total frames processed: {frame_count}")
                print(f"Total time: {total_time:.2f}s")
                print(f"Average FPS: {final_fps:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting.")
        cv2.destroyAllWindows()