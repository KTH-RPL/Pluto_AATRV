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
  actionlib
  actionlib_msgs
  message_generation
)

# Find Eigen library
find_package(Eigen3 REQUIRED)

add_action_files(
  FILES
  PlanGlobalPath.action
  
)

add_service_files(
  FILES
  RunControl.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  actionlib_msgs
  geometry_msgs
  nav_msgs
)

catkin_package(

#  LIBRARIES robot_controller
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs tf tf2 tf2_ros tf2_geometry_msgs visualization_msgs nav_msgs actionlib actionlib_msgs message_runtime
#  DEPENDS Eigen3
)


# Specify include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
 
# Add your executable
add_executable(control_node
  src/preview_test_alone.cpp
  src/final_control_algo.cpp
)

# Link libraries to your executable
target_link_libraries(control_node
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)


# Ensure control_node builds after messages/actions are generated
add_dependencies(control_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# New executable: C++ control service server
add_executable(control_service_server
  src/control_service_server.cpp
  src/final_control_algo.cpp
)

target_link_libraries(control_service_server
  ${catkin_LIBRARIES}
  ${EIGEN3_LIBRARIES}
)

# Ensure control_service_server builds after messages/services are generated
add_dependencies(control_service_server ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

