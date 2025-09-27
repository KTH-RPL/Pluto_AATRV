#include "final_control_algo.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "standalone_preview_controller");
    double linear_velocity_;
    double dt_;
    int preview_steps_;
    ros::NodeHandle nh_;
    ROS_INFO("Starting Standalone Preview Controller Node");
    nh_.param("preview_controller/linear_velocity", linear_velocity_, 0.3);
    nh_.param("preview_controller/preview_dt", dt_, 0.1);
    nh_.param("preview_controller/preview_steps", preview_steps_, 0);
    PreviewController controller(linear_velocity_, dt_, preview_steps_);  // velocity=1.0, dt=0.1, preview_steps=0
    
    controller.initialize_standalone_operation();
    
    return 0;
} 