#include "final_control_algo.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "standalone_preview_controller");
    
    ROS_INFO("Starting Standalone Preview Controller Node");
    
    PreviewController controller(1.0, 0.1, 5);  // velocity=1.0, dt=0.1, preview_steps=5
    
    controller.run_standalone_control();
    
    return 0;
} 
