#include "ros/ros.h"
#include "robot_controller/RunControl.h"
#include "final_control_algo.h"

// Instantiate your controller
PreviewController controller;

bool runControlCallback(robot_controller::RunControl::Request &req,
    robot_controller::RunControl::Response &res)
{
    ROS_INFO("[ControlService] run_control called with is_last_goal=%s",
             req.is_last_goal ? "true" : "false");

    try {
        bool reached = controller.run_control(req.is_last_goal);

        if (reached) {
            res.status = 1;  // SUCCESS
        } else {
            res.status = 0;  // RUNNING
        }
    } catch (const std::exception &e) {
        ROS_ERROR("[ControlService] Exception in run_control: %s", e.what());
        res.status = 2;  // FAILURE
    }

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "control_service_server");
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("run_control", runControlCallback);

    ROS_INFO("[ControlService] Ready to run control.");
    ros::spin();

    return 0;
}
