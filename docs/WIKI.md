# PLUTO AATRV Wiki

**Engineering Project in Robotics, Perception, and Learning (DD2414)**  
**KTH Royal Institute of Technology - Spring 2025**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Getting Started](#getting-started)
4. [Module Documentation](#module-documentation)
5. [API Reference](#api-reference)
6. [Troubleshooting](#troubleshooting)
7. [Development Guide](#development-guide)
8. [Team & Contact](#team--contact)

---

## Project Overview

### About PLUTO AATRV

The PLUTO (Platform for Learning and Understanding Through Operation) AATRV (All-Terrain Robotic Vehicle) is an autonomous navigation system developed for outdoor mobile robotics applications. This project extends the original [PLUTO Robot platform](https://github.com/KTH-RPL/Pluto-ATRV) with enhanced capabilities in perception, localization, planning, and control.

### Key Features

- **Autonomous Navigation**: Full autonomous stack with behavior tree architecture
- **NDT-based Localization**: High-accuracy localization using 3D point cloud maps
- **Global Path Planning**: Polygon-based planning with obstacle avoidance
- **Local Control**: Preview-based trajectory tracking controller
- **Perception System**: UV-map based obstacle detection from depth cameras
- **Robust Behavior Management**: py_trees-based behavior tree for mission control

### System Requirements

- **OS**: Ubuntu 20.04 LTS (or compatible)
- **ROS**: ROS Noetic
- **Sensors**:
  - Ouster OS1 3D LiDAR
  - Intel RealSense Depth Camera (D435/D455)
- **Hardware**: x86_64 CPU (recommended: Intel i7 or better)
- **Dependencies**: Python 3.8+, C++14

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Behavior Tree Controller                 │
│         (behavior_tree_m2.py / behavior_tree_m3.py)         │
└─────────────────┬───────────────────────────────────────────┘
                  │
      ┌───────────┼───────────┬─────────────┐
      │           │           │             │
      ▼           ▼           ▼             ▼
┌──────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
│   NDT    │ │ Global  │ │  Local  │ │ Detector │
│Localizer │ │ Planner │ │ Control │ │ (Percep.)│
└──────────┘ └─────────┘ └─────────┘ └──────────┘
      │           │           │             │
      └───────────┴───────────┴─────────────┘
                  │
      ┌───────────▼───────────┐
      │    Robot Platform     │
      │   (PLUTO Hardware)    │
      └───────────────────────┘
```

### Module Overview

| Module | Package | Language | Purpose |
|--------|---------|----------|---------|
| **Localization** | `ndt_localizer` | C++ | Real-time pose estimation using NDT algorithm |
| **Planning** | `robot_controller` | Python/C++ | Global path planning with polygon maps |
| **Control** | `robot_controller` | C++ | Local trajectory tracking and velocity control |
| **Perception** | `detector` | Python | Obstacle detection from depth images |
| **Behavior Tree** | `robot_controller` | Python | High-level mission control and state management |

---

## Getting Started

### Installation

#### 1. Prerequisites

```bash
# Install ROS Noetic (if not already installed)
sudo apt update
sudo apt install ros-noetic-desktop-full

# Install dependencies
sudo apt install python3-pip python3-catkin-tools
sudo apt install ros-noetic-cv-bridge ros-noetic-pcl-ros
sudo apt install ros-noetic-actionlib ros-noetic-tf2-geometry-msgs

# Install Python dependencies
pip3 install py_trees py_trees_ros scipy numpy matplotlib
```

#### 2. Clone and Build

```bash
# Create catkin workspace
mkdir -p ~/catkin_noetic_ws/src
cd ~/catkin_noetic_ws/src

# Clone repository
git clone <repository-url> Pluto_AATRV
cd ~/catkin_noetic_ws

# Build packages
catkin build  # or catkin_make
source devel/setup.bash
```

#### 3. Map Setup

Place your point cloud map (`.pcd` file) in the localization map directory:

```bash
cp your_map.pcd ~/catkin_noetic_ws/src/Pluto_AATRV/localisation_src/ndt_localizer/map/
```

Update the map path in `localization.launch`:
```xml
<arg name="pcd_path" default="$(find ndt_localizer)/map/your_map.pcd"/>
```

### Quick Start Guide

#### Method 1: Full Autonomous Stack

Follow these steps in sequence (each in a separate terminal):

**Step 1: Configure LiDAR**
```bash
# Press Ctrl+1 to open terminal windows
# Wait for all terminals to open
# Press Ctrl+0 to configure Ouster LiDAR
```

**Step 2: Launch Localization**
```bash
roslaunch ndt_localizer localization.launch
```

**Step 3: Launch Publishers**
```bash
roslaunch robot_controller main_publishers.launch
```

**Step 4: Launch Controller**
```bash
roslaunch robot_controller main_controller_bt.launch
```

**Step 5: Launch Global Planner**
```bash
roslaunch robot_controller global_planner.launch
```

**Step 6: Start Behavior Tree**
```bash
python3 /catkin_noetic_ws/src/Pluto_AATRV/control/scripts/behavior_tree_m2.py
```

#### Method 2: Manual Control (Joystick)

For teleoperation mode:

```bash
roslaunch pluto pluto.launch
```

---

## Module Documentation

### 1. Localization Module (`ndt_localizer`)

#### Overview

The localization module uses the Normal Distributions Transform (NDT) algorithm to estimate the robot's pose in a pre-built 3D point cloud map.

#### Key Components

- **Map Loader**: Loads and publishes the point cloud map
- **NDT Localizer**: Performs scan matching using PCL NDT
- **Pose Publisher**: Publishes localized pose on `/ndt_pose`

#### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/ouster/points` | `sensor_msgs/PointCloud2` | Input: Raw LiDAR scans |
| `/ndt_pose` | `geometry_msgs/PoseStamped` | Output: Localized robot pose |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Input: Initial pose estimate |
| `/is_converged` | `std_msgs/Bool` | Output: Convergence status |

#### Configuration

Key parameters in `ndt_localizer.launch`:

```xml
<arg name="trans_epsilon" default="0.05"/>   <!-- Convergence threshold -->
<arg name="step_size" default="0.1"/>        <!-- Newton step size -->
<arg name="resolution" default="2.0"/>       <!-- Voxel grid resolution (m) -->
<arg name="max_iterations" default="30.0"/>  <!-- Max NDT iterations -->
```

#### Usage Notes

- Provide an initial pose estimate using RViz "2D Pose Estimate" tool
- Localization accuracy depends on map quality and LiDAR density
- Monitor the `/is_converged` topic to ensure localization is stable

---

### 2. Global Planner (`robot_controller`)

#### Overview

The global planner generates collision-free paths through a polygonal map representation. It uses a graph-based approach with visibility graphs and A* search.

#### Key Components

- **Action Server**: `plan_global_path` action interface
- **Polygon Map**: Static obstacle representation
- **Path Planner**: Visibility graph + A* algorithm

#### Action Interface

**Action**: `robot_controller/PlanGlobalPathAction`

**Goal**:
```
geometry_msgs/PoseArray waypoints  # List of waypoints to visit
```

**Result**:
```
nav_msgs/Path global_plan  # Complete path from start to all waypoints
```

**Feedback**:
```
nav_msgs/Path current_segment  # Currently planned segment
nav_msgs/Path global_plan      # Accumulated path so far
```

#### Configuration

Map data is stored in: `milestone2_gmap_dataset/*.ply`

Polygon obstacles can be loaded and visualized in RViz.

#### Usage

The global planner is invoked automatically by the behavior tree when:
1. Robot receives waypoints on `/waypoints` topic
2. No global path currently exists

---

### 3. Local Controller (`robot_controller`)

#### Overview

The local controller tracks the global path using a preview-based control algorithm. It computes velocity commands to follow the planned trajectory while avoiding obstacles.

#### Key Components

- **Control Service**: `run_control` service interface
- **Preview Controller**: Predictive path tracking algorithm
- **Velocity Publisher**: Outputs motor commands

#### Service Interface

**Service**: `robot_controller/RunControl`

**Request**:
```
bool is_last_goal  # Whether this is the final waypoint
```

**Response**:
```
int32 status  # 0=RUNNING, 1=SUCCESS, 2=FAILURE
```

#### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/global_path` | `nav_msgs/Path` | Input: Path to follow |
| `/robot_pose` | `geometry_msgs/PoseStamped` | Input: Current robot pose |
| `/cmd_vel` | `geometry_msgs/Twist` | Output: Velocity commands |

#### Control Algorithm

The controller uses a **preview-based** approach:
1. Look ahead along the path (preview distance)
2. Compute lateral error and heading error
3. Calculate corrective steering and velocity
4. Apply velocity constraints and smoothing

---

### 4. Perception Module (`detector`)

#### Overview

The detector module processes depth camera images to identify obstacles in the robot's path. It uses a UV-map representation for efficient 3D obstacle detection.

#### Key Components

- **UV Detector**: Converts depth images to obstacle bounding boxes
- **3D Box Estimator**: Computes 3D positions and dimensions
- **RViz Visualizer**: Publishes markers for debugging

#### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | Input: Depth image |
| `/robot_pose` | `geometry_msgs/PoseStamped` | Input: Robot pose for transforms |
| `/detected_obstacles` | `visualization_msgs/MarkerArray` | Output: 3D obstacle markers |

#### Algorithm

1. **U-Map Construction**: Project depth pixels to vertical columns
2. **V-Map Construction**: Detect height discontinuities
3. **Segmentation**: Group connected components
4. **3D Estimation**: Back-project to 3D bounding boxes

#### Configuration

Edit `detector/src/scripts/detector.py` for tuning:
- `min_dist`, `max_dist`: Detection range
- `threshold_point`, `threshold_line`: Segmentation sensitivity
- `row_downsample`: Processing speed vs. accuracy

---

### 5. Behavior Tree (`behavior_tree_m2.py`)

#### Overview

The behavior tree orchestrates the entire autonomous mission. It monitors system health, triggers planning, and coordinates control.

#### Tree Structure

```
Main Sequence
├── Check Ouster Points (1 Hz timeout)
├── Check Localization (convergence check)
└── Goal Reached Fallback
    ├── Goal Reached?
    └── Planner and Control Sequence
        ├── Global Planner Fallback
        │   ├── Global Path Exists?
        │   └── Request Global Path
        └── Execute Control
```

#### Behavior Nodes

| Node | Type | Purpose |
|------|------|---------|
| `check_ouster_points` | Condition | Verify LiDAR data stream |
| `check_localization` | Condition | Ensure localization converged |
| `goal_reached` | Condition | Check if final goal reached |
| `global_path_exist` | Condition | Check if path is planned |
| `global_path_client` | Action | Request path from planner |
| `control_planner` | Action | Execute trajectory tracking |

#### State Management

The behavior tree maintains shared state through `goal_robot_condition`:
- Current waypoints
- Robot pose
- Global path
- Localization status
- Sensor health

#### Failure Recovery

- **LiDAR Failure**: Tree returns FAILURE if no `/ouster/points` for 1 second
- **Localization Loss**: Publishes last good pose to `/initialpose` for recovery
- **Planning Failure**: Logs error and returns FAILURE status
- **Control Failure**: Service returns status code for error handling

---

## API Reference

### ROS Topics

#### Subscribed Topics

| Topic | Type | Module | Description |
|-------|------|--------|-------------|
| `/ouster/points` | `sensor_msgs/PointCloud2` | Localization | Raw LiDAR point cloud |
| `/camera/depth/image_rect_raw` | `sensor_msgs/Image` | Perception | Depth camera image |
| `/waypoints` | `nav_msgs/Path` | Behavior Tree | Mission waypoints |
| `/initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Localization | Initial pose for NDT |

#### Published Topics

| Topic | Type | Module | Description |
|-------|------|--------|-------------|
| `/ndt_pose` | `geometry_msgs/PoseStamped` | Localization | Localized robot pose (map frame) |
| `/robot_pose` | `geometry_msgs/PoseStamped` | Publisher Node | Transformed robot pose |
| `/global_path` | `nav_msgs/Path` | Global Planner | Planned path (latched) |
| `/goal_pose` | `geometry_msgs/PoseStamped` | Behavior Tree | Current target waypoint |
| `/cmd_vel` | `geometry_msgs/Twist` | Controller | Velocity commands |
| `/is_converged` | `std_msgs/Bool` | Localization | NDT convergence status |

### ROS Services

#### `/run_control`

Execute trajectory tracking control.

**Type**: `robot_controller/RunControl`

**Request**:
```
bool is_last_goal  # True if this is the final waypoint
```

**Response**:
```
int32 status  # 0=RUNNING, 1=SUCCESS, 2=FAILURE
```

**Usage**:
```python
rospy.wait_for_service('run_control')
run_control = rospy.ServiceProxy('run_control', RunControl)
response = run_control(False)  # Not last goal
```

### ROS Actions

#### `/plan_global_path`

Plan a global path through multiple waypoints.

**Type**: `robot_controller/PlanGlobalPathAction`

**Goal**:
```
geometry_msgs/PoseArray waypoints  # Ordered list of waypoints
```

**Feedback**:
```
nav_msgs/Path current_segment  # Current segment being planned
nav_msgs/Path global_plan      # Accumulated path
```

**Result**:
```
nav_msgs/Path global_plan  # Complete path from start to all goals
```

**Usage**:
```python
client = actionlib.SimpleActionClient('plan_global_path', PlanGlobalPathAction)
client.wait_for_server()

goal = PlanGlobalPathGoal()
goal.waypoints = waypoint_array  # PoseArray

client.send_goal(goal, done_cb=done_callback, feedback_cb=feedback_callback)
client.wait_for_result()
result = client.get_result()
```

---

## Troubleshooting

### Common Issues

#### 1. Localization Not Converging

**Symptoms**: `/is_converged` stays False, robot pose jumps

**Solutions**:
- Provide better initial pose estimate in RViz
- Check that map file matches current environment
- Verify LiDAR is publishing on `/ouster/points`
- Reduce `resolution` parameter for better accuracy (slower)
- Increase `max_iterations` if convergence is slow

#### 2. No Global Path Generated

**Symptoms**: Behavior tree stuck at `global_path_client`

**Solutions**:
- Verify waypoints are published to `/waypoints`
- Check that waypoints are inside the map boundaries
- Ensure global planner action server is running
- Verify polygon map data is loaded correctly
- Check for planning errors in planner logs

#### 3. Robot Not Moving

**Symptoms**: Control node running but no velocity commands

**Solutions**:
- Check `/global_path` topic has valid path data
- Verify `/robot_pose` is publishing current pose
- Ensure controller service is being called
- Check for path tracking errors in controller logs
- Verify motor drivers are enabled

#### 4. LiDAR Connection Lost

**Symptoms**: `/ouster/points` stops publishing

**Solutions**:
- Reconfigure LiDAR (Ctrl+1, then Ctrl+0)
- Check network connection to LiDAR
- Verify LiDAR power supply
- Restart LiDAR driver node

#### 5. Behavior Tree Fails Immediately

**Symptoms**: Tree returns FAILURE at startup

**Solutions**:
- Check that all prerequisite nodes are running
- Verify sensor topics are publishing
- Ensure localization has converged before mission start
- Check for missing dependencies (py_trees, actionlib)

### Debug Tools

#### Visualize in RViz

```bash
rosrun rviz rviz -d /catkin_noetic_ws/src/Pluto_AATRV/control/rviz/pluto.rviz
```

Add displays:
- **PointCloud2**: `/points_map` (map)
- **Path**: `/global_path` (planned path)
- **PoseStamped**: `/ndt_pose`, `/robot_pose`, `/goal_pose`
- **MarkerArray**: `/detected_obstacles` (perception)
- **TF**: Show transforms

#### Monitor Topics

```bash
# Check topic frequencies
rostopic hz /ouster/points
rostopic hz /ndt_pose
rostopic hz /robot_pose

# Echo topic data
rostopic echo /is_converged
rostopic echo /global_path

# List all active topics
rostopic list
```

#### Inspect TF Tree

```bash
rosrun tf view_frames
evince frames.pdf
```

#### Check Node Status

```bash
rosnode list
rosnode info /ndt_localizer
rosnode info /behavior_tree_controller
```

---

## Development Guide

### Adding New Waypoints

Waypoints are published as a `nav_msgs/Path` message to `/waypoints`.

**Example**:
```python
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

rospy.init_node('waypoint_publisher')
pub = rospy.Publisher('/waypoints', Path, queue_size=1, latch=True)

path = Path()
path.header.frame_id = "map"
path.header.stamp = rospy.Time.now()

# Add waypoints
for (x, y) in [(100, 50), (120, 60), (140, 70)]:
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.w = 1.0
    path.poses.append(pose)

rospy.sleep(1)
pub.publish(path)
rospy.loginfo("Published %d waypoints", len(path.poses))
```

### Extending the Behavior Tree

To add new behaviors, edit `behavior_tree_m2.py`:

1. **Create a new behavior class**:
```python
class my_new_behavior(pt.behaviour.Behaviour):
    def __init__(self, c_goal):
        super(my_new_behavior, self).__init__("MyBehavior")
        self.c_goal = c_goal
    
    def update(self):
        # Your logic here
        if condition_met:
            return pt.common.Status.SUCCESS
        else:
            return pt.common.Status.FAILURE
```

2. **Add to the tree structure**:
```python
tree = RSequence(
    name="Main sequence",
    children=[
        check_ouster_points(c_goal),
        my_new_behavior(c_goal),  # <-- Add here
        check_localization(c_goal),
        # ...
    ]
)
```

### Tuning Control Parameters

Edit `final_control_algo.cpp` for controller tuning:

- `preview_distance`: Look-ahead distance (meters)
- `max_linear_velocity`: Maximum forward speed (m/s)
- `max_angular_velocity`: Maximum rotation speed (rad/s)
- `goal_tolerance`: Distance to consider goal reached (meters)

### Creating New Launch Files

Follow the pattern in `control/launch/`:

```xml
<launch>
    <!-- Your nodes here -->
    <node pkg="robot_controller" type="your_node" name="your_node" output="screen">
        <param name="param1" value="value1"/>
        <rosparam file="$(find robot_controller)/config/your_config.yaml"/>
    </node>
</launch>
```

---

## Team & Contact

### Development Team

| Name | Role | Email |
|------|------|-------|
| **Abhinav Srinivas Rajagopal** | Localization & Mapping | asraj@kth.se |
| **Prasetyo Wibowo Laksono Sanjaya** | Path Planning | pwlsa@kth.se |
| **Sarvesh Raj Kumar** | Control | sarvesh@kth.se |
| **Sankeerth Reddy Prodduturi** | Perception | srpr@kth.se |
| **Rajesh Kumar** | Behavior Tree | rajeshk@kth.se |

### Supervisor

- **Waqas Ali** - waqasali@kth.se

### Project Links

- **Original PLUTO Repository**: [KTH-RPL/Pluto-ATRV](https://github.com/KTH-RPL/Pluto-ATRV)
- **Course**: DD2414 - Engineering Project in Robotics, Perception, and Learning
- **Institution**: KTH Royal Institute of Technology

---

## License

This project builds upon the original PLUTO platform. Please refer to the original repository for license information.

---

## Acknowledgments

- KTH RPL Lab for the original PLUTO platform
- Autoware for NDT localization implementation
- py_trees community for behavior tree framework

---

**Last Updated**: October 26, 2025  
**Version**: Milestone 2 (Post Demo)

