cmake_minimum_required(VERSION 3.0.2)
project(robot_controller)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  tf
  tf2
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  nav_msgs
)

# Find Eigen library
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES robot_controller
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs tf tf2 tf2_ros tf2_geometry_msgs visualization_msgs nav_msgs
#  DEPENDS Eigen3
)

# Specify include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# Add your executable
add_executable(control_node
  src/control_node.cpp
  src/final_control_algo.cpp
)

# Link libraries to your executable
target_link_libraries(control_node
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)