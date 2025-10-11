cmake_minimum_required(VERSION 2.8.3)
project(ndt_localizer)

set(CMAKE_CXX_STANDARD 14)

#########################
## Find packages
#########################

find_package(catkin REQUIRED COMPONENTS
        roscpp
        tf2
        tf2_ros
        tf2_geometry_msgs
        std_msgs
        geometry_msgs
        sensor_msgs
        nav_msgs
        diagnostic_msgs
        pcl_conversions
        pcl_ros
        message_generation
        ndt_omp
        )

generate_messages(
        DEPENDENCIES
        std_msgs
)

find_package(PCL REQUIRED QUIET)
find_package(OpenMP REQUIRED)

#########################
## Catkin package
#########################

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS
        roscpp
        tf2
        tf2_ros
        tf2_geometry_msgs
        geometry_msgs
        sensor_msgs
        nav_msgs
        diagnostic_msgs
        pcl_conversions
        pcl_ros
        message_generation
        ndt_omp
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

#########################
## Build nodes
#########################

# Points downsampler
add_executable(voxel_grid_filter nodes/points_downsampler.cpp)
add_dependencies(voxel_grid_filter ${catkin_EXPORTED_TARGETS})
target_link_libraries(voxel_grid_filter ${catkin_LIBRARIES})

# Map loader
add_executable(map_loader nodes/map_loader.cpp)
target_link_libraries(map_loader ${catkin_LIBRARIES} ${PCL_LIBRARIES})

# NDT localizer node
add_executable(ndt_localizer_node nodes/ndt.cpp)

if(OPENMP_FOUND)
    target_link_libraries(ndt_localizer_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
else()
    target_link_libraries(ndt_localizer_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
    )
endif()

