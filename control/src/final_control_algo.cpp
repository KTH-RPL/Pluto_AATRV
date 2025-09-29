#include "final_control_algo.h"
#include <Eigen/LU>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <ros/ros.h>
#include <limits>
#include <algorithm>
#include <thread>
#include <cmath>
// Added for DWA trajectory visualization
#include <visualization_msgs/MarkerArray.h> 

PreviewController::PreviewController(double v, double dt, int preview_steps)
    : linear_velocity_(v), dt_(dt), preview_steps_(preview_steps), 
      prev_ey_(0), prev_etheta_(0), prev_omega_(0), nh_(), targetid(0),
      initial_pose_received_(false), path_generated_(false), initial_alignment_(false)
{
    robot_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/atrv/cmd_vel", 10);
    lookahead_point_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("lookahead_point", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("planned_path", 10);
    robot_pose_sub_ = nh_.subscribe("/robot_pose", 10, &PreviewController::robot_pose_callback, this);
    start_moving_sub_ = nh_.subscribe("/start_moving", 10, &PreviewController::start_moving_callback, this);

    // --- INITIALIZE DEBUG PUBLISHERS ---
    cross_track_error_pub_ = nh_.advertise<std_msgs::Float64>("debug/cross_track_error", 10);
    heading_error_pub_ = nh_.advertise<std_msgs::Float64>("debug/heading_error", 10);
    lookahead_heading_error_pub_ = nh_.advertise<std_msgs::Float64>("debug/lookahead_heading_error", 10);
    current_v_pub_ = nh_.advertise<std_msgs::Float64>("debug/current_v", 10);
    current_omega_pub_ = nh_.advertise<std_msgs::Float64>("debug/current_omega", 10);
    path_curvature_pub_ = nh_.advertise<std_msgs::Float64>("debug/path_curvature", 10);
    start_moving_sub_ = nh_.subscribe("/start_moving", 10, &PreviewController::start_moving_callback, this);
    stop_moving_sub_ = nh_.subscribe("/stop_moving", 10, &PreviewController::stop_moving_callback, this);


    start_moving_ = false;
    use_start_stop = true;
    // -----------------------------------

    // --- MODIFIED --- Path generation parameters
    nh_.param<std::string>("preview_controller/path_type", path_type_, "snake"); // "snake" or "straight"
    nh_.param("preview_controller/amplitude", path_amplitude, 4.0);
    nh_.param("preview_controller/wavelength", path_wavelength, 6.0);
    nh_.param("preview_controller/length", path_length, 10.0);
    nh_.param("preview_controller/point_spacing", path_point_spacing, 0.3);
    nh_.param("preview_controller/straight_path_distance", straight_path_distance_, 5.0); // For straight path

    // Params to modify for different scenarios
    // Reference optimal velocity for robot
    nh_.param("preview_controller/linear_velocity", linear_velocity_, 0.3);

    // Frequency of the controller
    nh_.param("preview_controller/preview_dt", dt_, 0.1);

    // Control params 
    nh_.param("preview_controller/max_vel", max_vel_, 0.3);
    nh_.param("preview_controller/max_omega", max_omega_, 0.6);
    nh_.param("preview_controller/vel_acc", vel_acc_, 0.5);
    nh_.param("preview_controller/omega_acc", omega_acc_, 0.4);
    // nh_.param("preview_controller/max_domega", max_domega_, 0.2);

    // Measure robot_radius and set
    nh_.param("preview_controller/robot_radius", robot_radius_, 0.5);
    nh_.param("preview_controller/lookahead_distance", lookahead_distance_, 0.5);

    // Max cross track error so robot moves towards the target first
    nh_.param("preview_controller/max_cte", max_cte, 1.5);

    // Thresh to adjust orientation before moving to target
    nh_.param("preview_controller/max_lookahead_heading_error", max_lookahead_heading_error, 0.2);

    // For optimaizing the cost function, try diff params 
    nh_.param("preview_controller/preview_loop_thresh", preview_loop_thresh, 1e-5);

    // P gain to adjust high CTE
    nh_.param("preview_controller/kp_adjust_cte", kp_adjust_cte, 2.0);

    // Params for collision avoidance
    nh_.param("preview_controller/collision_robot_coeff", collision_robot_coeff, 2.0);
    nh_.param("preview_controller/collision_obstacle_coeff", collision_obstacle_coeff, 2.0);

    // Thresh to stop the robot when close to the goal
    nh_.param("preview_controller/goal_distance_threshold", goal_distance_threshold_, 0.2);

    // Q matrix for the preview controller
    std::vector<double> default_Q = {5.0, 6.0, 5.0};
    nh_.param("preview_controller/Q_params", Q_params_, default_Q);

    // Thresh to switch between DWA and preview control
    nh_.param("preview_controller/obst_cost_thresh", obst_cost_thresh, 100.0);

    // R matrix for the preview controller
    nh_.param("preview_controller/R", R_param_, 1.0);

    // If obstacle cost exceeds this, stop the robot
    nh_.param("preview_controller/stop_cost", stop_robot_cost_thresh, 200.0);

    // Factor to reduce speed when close to goal
    nh_.param("preview_controller/goal_reduce_factor", goal_reduce_factor, 0.5);

    nh_.param("preview_controller/use_start_top", use_start_stop, true);

    // Calculate the velocity and omega acceleration bounds for timestep
    vel_acc_bound = vel_acc_ * dt_;
    omega_acc_bound = omega_acc_ * dt_;
}

void PreviewController::stop_moving_callback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
        start_moving_ = false;
    }
}


void PreviewController::start_moving_callback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
        start_moving_ = true;
    }
}

void PreviewController::robot_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
    robot_x = current_pose.pose.position.x;
    robot_y = current_pose.pose.position.y;
    robot_theta = current_pose.pose.orientation.z;
    
    // --- MODIFIED --- Generate path on first pose received based on path_type_ parameter
    if (!initial_pose_received_) {
        initial_pose_received_ = true;
        ROS_INFO("Initial robot pose received: x=%.2f, y=%.2f, theta=%.2f", robot_x, robot_y, robot_theta);

        if (path_type_ == "snake") {
            ROS_INFO("Path type set to 'snake'. Generating snake path...");
            generate_snake_path(robot_x, robot_y, robot_theta);
        } else if (path_type_ == "straight") {
            ROS_INFO("Path type set to 'straight'. Generating straight line path...");
            generate_straight_path(robot_x, robot_y, robot_theta);
        } else {
            ROS_ERROR("Invalid path_type '%s'. Defaulting to snake path.", path_type_.c_str());
            generate_snake_path(robot_x, robot_y, robot_theta);
        }

        initialize_dwa_controller();
        path_generated_ = true;
        calculate_all_curvatures(); // Precompute curvatures for all path points
        publish_path();  // Publish the generated path
    }

    if (initial_pose_received_) {
        publish_path();  
    }
}

// Need to call this after setting the current_path from execute planner, also need to set max_path_points 
void PreviewController::initialize_dwa_controller() {
    dwa_controller_ptr = new dwa_controller(current_path, targetid, max_path_points);
}

// Generate snake path starting from robot's initial position
void PreviewController::generate_snake_path(double start_x, double start_y, double start_theta) {
    current_path.clear();
    
    // Snake path parameters (Adjust these)
    double amplitude = path_amplitude;
    double wavelength = path_wavelength;
    double length = path_length;
    double point_spacing = path_point_spacing;
    int num_points = static_cast<int>(std::ceil(length / point_spacing)) + 1;
    
    // Generate snake path
    for (int i = 0; i < num_points; ++i) {
        double x = start_x + (length * i) / (num_points - 1);
        double y = start_y + amplitude * std::sin(2.0 * M_PI * (x - start_x) / wavelength);
        
        // Calculate orientation based on path derivative
        double dx = 1.0;  // dx/dt = constant for linear x progression
        double dy = amplitude * (2.0 * M_PI / wavelength) * std::cos(2.0 * M_PI * (x - start_x) / wavelength);
        double theta = std::atan2(dy, dx);
        
        // Normalize theta to [-pi, pi]
        while (theta > M_PI) theta -= 2.0 * M_PI;
        while (theta < -M_PI) theta += 2.0 * M_PI;
        
        current_path.emplace_back(x, y, theta);
    }
    
    max_path_points = current_path.size();
    ROS_INFO("Generated snake path with %d points.", max_path_points);
}

// --- NEW --- Generate a straight line path
void PreviewController::generate_straight_path(double start_x, double start_y, double start_theta) {
    current_path.clear();

    // Parameters for straight path
    double length = straight_path_distance_;
    double point_spacing = path_point_spacing;
    int num_points = static_cast<int>(std::ceil(length / point_spacing)) + 1;

    // Generate straight path points
    for (int i = 0; i < num_points; ++i) {
        double dist_along_path = i * point_spacing;
        // Ensure the last point is exactly at the end
        if (i == num_points - 1) {
            dist_along_path = length;
        }

        double x = start_x + dist_along_path * std::cos(start_theta);
        double y = start_y + dist_along_path * std::sin(start_theta);
        double theta = start_theta; // Orientation is constant for a straight line

        current_path.emplace_back(x, y, theta);
    }

    max_path_points = current_path.size();
    ROS_INFO("Generated straight path with %d points over %.2f meters.", max_path_points, length);
}


// Initialize standalone operation
void PreviewController::initialize_standalone_operation() {
    ROS_INFO("Initializing standalone preview controller...");
    ROS_INFO("Waiting for robot pose...");

    // Wait until a path is generated
    while (ros::ok() && !path_generated_) {
        ros::spinOnce();
        ros::Duration(0.1).sleep();  // sleep 100ms
    }

    if (!path_generated_) {
        ROS_ERROR("Failed to generate path. Exiting.");
        return;
    }

    ROS_INFO("Standalone preview controller initialized successfully!");
    ROS_INFO("Control loop started with dt = %.3f seconds", dt_);

    // Main control loop
    while (ros::ok()) {
        ros::spinOnce();

        if (start_moving_ || !use_start_stop) {
            bool goal_reached = run_control();
            if (goal_reached) {
                ROS_INFO("Goal reached! Stopping control loop.");
                stop_robot();
                break;
            }
        }

        ros::Duration(dt_).sleep();  // control frequency
    }
}


double PreviewController::distancecalc(double x1, double y1, double x2, double y2) {
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}


// Only changes when velocity changes, can put condition on prev vel diff to reduce computation time
void PreviewController::calcGains() {
    A_ << 0, 1, 0,
          0, 0, v_,
          0, 0, 0;
    B_ << 0,
          0,
          1;
    D_ << 0,
          -(v_ * v_),
          -v_;

    Q_ = Eigen::Matrix3d::Zero();
    Q_(0, 0) = Q_params_[0];
    Q_(1, 1) = Q_params_[1];
    Q_(2, 2) = Q_params_[2];
    R_ << R_param_;

    A_ = Eigen::Matrix3d::Identity() + A_ * dt_;
    B_ = B_ * dt_;
    D_ = D_ * dt_;
    Q_ = Q_ * dt_;
    R_ = R_ / dt_;


    // Reduce iterations if needed
    Eigen::Matrix3d P = Q_;
    for (int i = 0; i < 100; ++i) {
        Eigen::Matrix3d P_next = A_.transpose() * P * A_ -
            (A_.transpose() * P * B_) * (R_ + B_.transpose() * P * B_).inverse() * (B_.transpose() * P * A_) + Q_;
        if ((P_next - P).norm() < preview_loop_thresh) break;
        P = P_next;
    }

    Eigen::Matrix3d lambda0 = A_.transpose() * (Eigen::Matrix3d::Identity() + P * B_ * R_.inverse() * B_.transpose()).inverse();
    Kb_ = (R_ + B_.transpose() * P * B_).inverse() * B_.transpose() * P * A_;

    Pc_ = Eigen::MatrixXd::Zero(3, preview_steps_ + 1);
    for (int i = 0; i <= preview_steps_; ++i) {
        Pc_.col(i) = lambda0.pow(i) * P * D_;
    }

    Lmatrix_ = Eigen::MatrixXd::Zero(preview_steps_ + 1, preview_steps_ + 1);
    if (preview_steps_ > 0) {
        Lmatrix_.block(0, 1, preview_steps_, preview_steps_) = Eigen::MatrixXd::Identity(preview_steps_, preview_steps_);
    }

    Eigen::MatrixXd Kf_term = Pc_ * Lmatrix_;
    Kf_term.col(0) += P * D_;

    Kf_ = (R_ + B_.transpose() * P * B_).inverse() * B_.transpose() * Kf_term;
}

// Depends on path size, either calculate all curvature at once, for now called for each point
double PreviewController::calculate_curvature(std::vector<double> x, std::vector<double> y) {
    double curvature = 0.0;
    if (x.size() < 3) return curvature;
    double x1 = x[0];
    double y1 = y[0];
    double x2 = x[1];
    double y2 = y[1];
    double x3 = x[2];
    double y3 = y[2];
    double dx1 = x2 - x1;
    double dy1 = y2 - y1;
    double dx2 = x3 - x2;
    double dy2 = y3 - y2;
    double angle1 = std::atan2(dy1, dx1);
    double angle2 = std::atan2(dy2, dx2);
    double dtheta = angle2 - angle1;
    dtheta = std::atan2(std::sin(dtheta), std::cos(dtheta));
    double dist = std::hypot(dx1, dy1);
    if (dist > 1e-6)
        curvature = dtheta / dist;
    return curvature;
}

// Calculate curvatures for all path points at once
void PreviewController::calculate_all_curvatures() {
    path_curvatures_.clear();
    path_curvatures_.resize(max_path_points, 0.0);
    
    // Calculate curvature for each point (need at least 3 points)
    for (int i = 0; i < max_path_points; ++i) {
        if (i == 0 || i == max_path_points - 1) {
            // First and last points have zero curvature
            path_curvatures_[i] = 0.0;
        } else {
            // Use current point and its neighbors
            std::vector<double> x_vals = {
                current_path[i-1].x,
                current_path[i].x,
                current_path[i+1].x
            };
            std::vector<double> y_vals = {
                current_path[i-1].y,
                current_path[i].y,
                current_path[i+1].y
            };
            path_curvatures_[i] = calculate_curvature(x_vals, y_vals);
        }
    }
    ROS_INFO("Calculated curvatures for %d path points", max_path_points);
}

double PreviewController::cross_track_error(double x_r, double y_r, double x_ref, double y_ref, double theta_ref) {
    double cte = (y_ref - y_r) * std::cos(theta_ref) - (x_ref - x_r) * std::sin(theta_ref);
    return cte;
}

void PreviewController::lookahead_heading_error(double x_ref, double y_ref, double theta_ref) {
    lookahead_heading_error_ = robot_theta - std::atan2(y_ref - robot_y, x_ref - robot_x);
    if (lookahead_heading_error_ > M_PI) {
        lookahead_heading_error_ -= 2 * M_PI;
    } else if (lookahead_heading_error_ < -M_PI) {
        lookahead_heading_error_ += 2 * M_PI;
    }
}

// Need to check if called only once every iteration, else can shift more than the acc_bound
void PreviewController::boundvel(double ref_vel) {
    if (std::abs(ref_vel - v_) < vel_acc_bound) {
        v_ = ref_vel;
        return;
    } else {
        if (v_ > ref_vel)
            v_ = v_ - vel_acc_bound;
        else
            v_ = v_ + vel_acc_bound;
    }
    v_ = std::max(std::min(v_, max_vel_), 0.0);
}

void PreviewController::boundomega(double ref_omega) {
    if (std::abs(ref_omega - omega_) < omega_acc_bound) {
        omega_ = ref_omega;
    } else {
        if (omega_ > ref_omega)
            omega_ = omega_ - omega_acc_bound;
        else
            omega_ = omega_ + omega_acc_bound;
    }
    omega_ = std::max(std::min(omega_, max_omega_), -max_omega_);
}


// Changed from NavigationSystem to PreviewController, main control loop
bool PreviewController::run_control(bool is_last_goal) {
    ROS_INFO("Running run control");
    bool bounded_vel = false;
    bool bounded_omega = false;
    robot_x = current_pose.pose.position.x;
    robot_y = current_pose.pose.position.y;
    robot_theta = current_pose.pose.orientation.z;


    // Changed until point is only in front of robot
    while ((targetid + 1 < max_path_points) && (chkside(current_path[targetid].theta))) {
        targetid++;
    }

    // Gets the lookahead point
    while ((targetid + 1 < max_path_points) && (distancecalc(robot_x, robot_y, current_path[targetid].x, current_path[targetid].y) < lookahead_distance_)) {
        targetid++;
    }
        

    cross_track_error_ = cross_track_error(robot_x, robot_y, current_path[targetid].x, current_path[targetid].y, current_path[targetid].theta);

    // --- PUBLISH CROSS TRACK ERROR ---
    std_msgs::Float64 cte_msg;
    cte_msg.data = cross_track_error_;
    cross_track_error_pub_.publish(cte_msg);
    // ---------------------------------
    lookahead_heading_error(current_path[targetid].x, current_path[targetid].y, current_path[targetid].theta);
    // --- PUBLISH LOOKAHEAD HEADING ERROR ---
    std_msgs::Float64 lhe_msg;
    lhe_msg.data = lookahead_heading_error_;
    lookahead_heading_error_pub_.publish(lhe_msg);
    // ---------------------------------------
    if (!initial_alignment_) {
        if (std::abs(lookahead_heading_error_) < max_lookahead_heading_error) {
            initial_alignment_ = true;
        }
        else {
            boundvel(0.0);
            boundomega(-kp_adjust_cte * lookahead_heading_error_);
            ROS_INFO("Adjusting Lookahead Heading Error: %f", lookahead_heading_error_);
            ROS_INFO("Publishing omega %f", -kp_adjust_cte * lookahead_heading_error_);
            ROS_INFO("Actual omega %f", omega_);
            bounded_vel = true;
            bounded_omega = true;
            publish_cmd_vel();
            return false;
        }
    }

    

    double x_goal = current_path[max_path_points - 1].x;
    double y_goal = current_path[max_path_points - 1].y;
    double goal_distance = distancecalc(robot_x, robot_y, x_goal, y_goal);
    if(goal_distance < 1.0)
        boundvel(goal_distance*linear_velocity_*goal_reduce_factor);
    else
        boundvel(linear_velocity_);
    // Call DWA controller
    DWAResult dwa_result = dwa_controller_ptr->dwa_main_control(robot_x, robot_y, robot_theta, v_, omega_);
    ROS_INFO("obstacle cost if %f", dwa_result.obs_cost);

    // DWA cause obstacle too close, add condition to stop robot if too close to obstacle or in obstacle
    if (dwa_result.obs_cost > obst_cost_thresh) {
        v_ = dwa_result.best_v;
        omega_ = dwa_result.best_omega;
        // if (dwa_result.obs_cost > stop_robot_cost_thresh) {
        //     v_ = 0.0;
        //     omega_ = 0.0;
        //     ROS_WARN("Obstacle too close! Stopping robot.");
        // }
        ROS_INFO("DWA result: v = %f, omega = %f", v_, omega_);
    }     
    else 
    // Call preview Controller
    {
        if (!bounded_vel)
            // Increase vel to the target vel
            boundvel(linear_velocity_);
        // heading_error_ = robot_theta - current_path[targetid].theta; #Check difference with below line
        heading_error_ =  lookahead_heading_error_;
        if (heading_error_ > M_PI) {
            heading_error_ -= 2 * M_PI;
        } else if (heading_error_ < -M_PI) {
            heading_error_ += 2 * M_PI;
        }

        // --- PUBLISH HEADING ERROR ---
        std_msgs::Float64 he_msg;
        he_msg.data = heading_error_;
        heading_error_pub_.publish(he_msg);
        // -----------------------------

        // Calls to compute the omega
        compute_control(cross_track_error_, heading_error_, path_curvature_);
        
        // --- PUBLISH PATH CURVATURE ---
        std_msgs::Float64 pc_msg;
        pc_msg.data = path_curvature_;
        path_curvature_pub_.publish(pc_msg);
        // ------------------------------
        ROS_INFO("Preview Control: v = %f, omega = %f", v_, omega_);
    }

    if(!bounded_omega)
        boundomega(omega_);
   

    if(!bounded_vel)
        boundvel(v_);
    
    // --- PUBLISH CURRENT VELOCITY ---
    std_msgs::Float64 v_msg;
    v_msg.data = v_;
    current_v_pub_.publish(v_msg);
    // --------------------------------
    
    // Publish the cmd_vel (this also clips omega_)
    publish_cmd_vel();
    
    // --- PUBLISH CURRENT (CLIPPED) OMEGA ---
    std_msgs::Float64 omega_msg;
    omega_msg.data = omega_;
    current_omega_pub_.publish(omega_msg);
    // ---------------------------------------

    // Publish the lookahead point
    geometry_msgs::PoseStamped look_pose;
    look_pose.pose.position.x = current_path[targetid].x;
    look_pose.pose.position.y = current_path[targetid].y;
    look_pose.pose.position.z = 0.0;
    look_pose.pose.orientation.z = current_path[targetid].theta;
    publish_look_pose(look_pose);

    // Check if goal is reached
    if (goal_distance < goal_distance_threshold_) {
        stop_robot();
        ROS_INFO("Goal reached!");
        return true;
    }

    return false;
}


// Checks if robot in front of point
bool PreviewController::chkside(double path_theta) {
    if (targetid + 1 >= max_path_points) return false;
    double x1 = current_path[targetid].x;
    double y1 = current_path[targetid].y;
    double m = -1 / std::tan(path_theta);
    double ineq = 0;
    // In case of straight line, use y-intercept
    if (std::fabs(std::tan(path_theta)) < 1e-6) {
        m = std::numeric_limits<double>::infinity();
        ineq = robot_y - y1;

    }
    else {
        ineq = robot_y - (m * robot_x) - y1 + (m * x1);
    }
    bool t = false;
    if (ineq > 0) {
        t = true;
        // Condition changed based on orientation sign
        if (path_theta < 0) {
            t = false;
        }
    } else {
        t = false;
        if (path_theta < 0) {
            t = true;
        }
    }
    return t;
}

// Compute omega from Preview Controller
void PreviewController::compute_control(double cross_track_error, double heading_error, double path_curvature) {
    x_state << cross_track_error, v_ * std::sin(heading_error), heading_error;

    std::vector<double> preview_curv(preview_steps_ + 1);
    
    // Use precomputed curvatures for preview steps
    for (int i = 0; i <= preview_steps_; ++i) {
        int preview_idx = targetid + i;
        if (preview_idx < max_path_points && preview_idx < path_curvatures_.size()) {
            preview_curv[i] = path_curvatures_[preview_idx];
        } else {
            // If beyond path, use zero curvature
            preview_curv[i] = 0.0;
        }
    }
    
    // Set current path curvature for publishing
    if (targetid < path_curvatures_.size()) {
        path_curvature_ = path_curvatures_[targetid];
    } else {
        path_curvature_ = 0.0;
    }

    Eigen::VectorXd curv_vec = Eigen::Map<Eigen::VectorXd>(preview_curv.data(), preview_steps_ + 1);
    calcGains();
    double u_fb = -(Kb_ * x_state)(0);
    double u_ff = -(Kf_ * curv_vec)(0);
    omega_ = u_fb + u_ff;
    ROS_INFO(" Theta error: %f, Omega: %f, ey: %f", heading_error, omega_, cross_track_error);
}

// Stop the robot
void PreviewController::stop_robot() {
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0.0;
    cmd_vel.angular.z = 0.0;
    robot_vel_pub_.publish(cmd_vel);
}

// Publish the cmd_vel
void PreviewController::publish_cmd_vel() {
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = v_;
    // Bound the omega
    if (omega_ < -max_omega_) {
        omega_ = -max_omega_;
    } else if (omega_ > max_omega_) {
        omega_ = max_omega_;
    }
    cmd_vel.angular.z = omega_;
    robot_vel_pub_.publish(cmd_vel);
}

// Publish the lookahead point
void PreviewController::publish_look_pose(geometry_msgs::PoseStamped look_pose) {
    look_pose.header.stamp = ros::Time::now();
    look_pose.header.frame_id = "odom"; //Change the frame if needed
    lookahead_point_pub_.publish(look_pose);
}

// Publish the planned path
void PreviewController::publish_path() {
    nav_msgs::Path path_msg;
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "odom";
    
    for (int i = targetid; i < current_path.size(); ++i) {
        const auto& waypoint = current_path[i];
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "odom";
        pose.pose.position.x = waypoint.x;
        pose.pose.position.y = waypoint.y;
        pose.pose.position.z = 0.0;
        
        // Convert theta to quaternion
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = std::sin(waypoint.theta / 2.0);
        pose.pose.orientation.w = std::cos(waypoint.theta / 2.0);
        
        path_msg.poses.push_back(pose);
    }
    
    path_pub_.publish(path_msg);
    // ROS_INFO("Published planned path with %zu waypoints to 'planned_path' topic", current_path.size());
}


dwa_controller::dwa_controller(const std::vector<Waypoint>& path, int& target_idx, const int& max_points)
    : current_path_(&path), target_idx_(&target_idx), max_path_points_(&max_points),
      min_obs_num(0), min_obs_dist(std::numeric_limits<double>::infinity()), collision_dist(std::numeric_limits<double>::infinity()),
      dx(0), dy(0), dist(0) {    

    // Future prediction horizion, not too big as most costs measured with last point
    nh_.param("dwa_controller/predict_time", predict_time_, 2.0);

    // Coefficient for cost functions
    nh_.param("dwa_controller/path_distance_bias", path_distance_bias_, 20.0);
    nh_.param("dwa_controller/goal_distance_bias", goal_distance_bias_, 0.5);
    nh_.param("dwa_controller/occdist_scale", occdist_scale_, 10.0);
    nh_.param("dwa_controller/speed_ref_bias", speed_ref_bias_, 0.005);
    nh_.param("dwa_controller/away_bias", away_bias_, 20.0);

    // Number of samples for velocity and omega between dynamic window
    nh_.param("dwa_controller/vx_samples", vx_samples_, 3);
    nh_.param("dwa_controller/omega_samples", omega_samples_, 5);
    
    // Same params above
    nh_.param("preview_controller/vel_acc", vel_acc_, 0.5);
    nh_.param("preview_controller/robot_radius", robot_radius_, 0.5);
    nh_.param("preview_controller/omega_acc", omega_acc_, 0.4);
    nh_.param("preview_controller/min_speed", min_speed_, 0.0);
    nh_.param("preview_controller/max_speed", max_speed_, 0.3);
    nh_.param("preview_controller/max_omega", max_omega_, 0.5);
    nh_.param("preview_controller/dt_dwa", dt_dwa_, 0.1);
    nh_.param("preview_controller/linear_velocity", ref_velocity_, 0.3);
    nh_.param("preview_controller/collision_robot_coeff", collision_robot_coeff, 2.0);
    nh_.param("preview_controller/collision_obstacle_coeff", collision_obstacle_coeff, 2.0);
    
    occ_sub_ = nh_.subscribe("/local_costmap", 1, &dwa_controller::costmap_callback, this);
    
    // --- NEW --- Initialize visualization publisher
    traj_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("dwa_trajectories", 1);
}

void dwa_controller::costmap_callback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
    occ_grid_ = *msg;
    costmap_received_ = true;
}

// void dwa_controller::obstacle_callback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
//     obstacles.clear(); // Clear previous obstacles
//     for (const auto& marker : msg->markers) {
//         double x = marker.pose.position.x;
//         double y = marker.pose.position.y;
//         double height = marker.scale.y;
//         double width = marker.scale.x;
//         obstacles.emplace_back(x, y, width, height);
//     }
// }

// Convert world coordinates to costmap indices
bool dwa_controller::worldToCostmap(double x, double y, int& mx, int& my, double robot_x, double robot_y) {
    if (!costmap_received_)
        return false;

    // Calculate relative x and y to robot, get odom pose in base_link
    double rel_x = x - robot_x;
    double rel_y = y - robot_y;

    // Get origin and resolution of costmap 
    double origin_x = occ_grid_.info.origin.position.x;
    double origin_y = occ_grid_.info.origin.position.y;
    double resolution = occ_grid_.info.resolution;

    // Convert to costmap indices
    mx = static_cast<int>((rel_x - origin_x) / resolution);
    my = static_cast<int>((rel_y - origin_y) / resolution);

    // Check if within costmap bounds
    return (mx >= 0 && mx < static_cast<int>(occ_grid_.info.width) &&
            my >= 0 && my < static_cast<int>(occ_grid_.info.height));
}

uint8_t dwa_controller::getCostmapCost(int mx, int my) {
    // Get costmap index and return cost
    int idx = my * occ_grid_.info.width + mx;
    if (idx >= 0 && idx < static_cast<int>(occ_grid_.data.size()))
        return static_cast<uint8_t>(occ_grid_.data[idx]);
    return 100;
}


// bounds and calculates window within acc
std::vector<double> dwa_controller::calc_dynamic_window(double v, double omega) {
    std::vector<double> Vs = {min_speed_, max_speed_, -max_omega_, max_omega_};
    std::vector<double> Vd = {v - vel_acc_ * dt_dwa_, v + vel_acc_ * dt_dwa_, omega - omega_acc_ * dt_dwa_, omega + omega_acc_ * dt_dwa_};
    std::vector<double> vr = {std::min(Vs[0], Vd[0]), std::min(Vs[1], Vd[1]), std::max(Vs[2], Vd[2]), std::min(Vs[3], Vd[3])};
    return vr;
}

// Traj predicted for time step dt_dwa within predict time, dt_dwa can be different from dt_ in Preview controller
std::vector<std::vector<double>> dwa_controller::calc_trajectory(double x, double y, double theta, double v, double omega) {
    std::vector<std::vector<double>> traj;
    traj.push_back({x, y, theta});

    int steps = static_cast<int>(predict_time_ / dt_dwa_);
    for (int i = 0; i < steps; ++i) {
        x += v * std::cos(theta) * dt_dwa_;
        y += v * std::sin(theta) * dt_dwa_;
        theta += omega * dt_dwa_;
        traj.push_back({x, y, theta});
    }
    return traj;
}

// CTE
double dwa_controller::cross_track_error(double x_r, double y_r, double x_ref, double y_ref, double theta_ref) {
    return (y_ref - y_r) * std::cos(theta_ref) - (x_ref - x_r) * std::sin(theta_ref);
}

// CTE cost
double dwa_controller::calc_path_cost() {
    if (traj_list_.empty() || !current_path_ || !max_path_points_ || !target_idx_ || *max_path_points_ == 0)
        return 0.0;

    auto last_point = traj_list_.back();
    double traj_x = last_point[0];
    double traj_y = last_point[1];

    int current_target = *target_idx_;
    if (current_target < *max_path_points_) {
        double x_ref = (*current_path_)[current_target].x;
        double y_ref = (*current_path_)[current_target].y;
        double theta_ref = (*current_path_)[current_target].theta;
        return std::abs(cross_track_error(traj_x, traj_y, x_ref, y_ref, theta_ref));
    }
    return 0.0;
}

// Lookahead cost, more if farther so robot doesn't indefinitely go out of path because of obstacle, increase predict time if needed
double dwa_controller::calc_lookahead_cost() {
    if (traj_list_.empty() || !current_path_ || !target_idx_ || !max_path_points_)
        return 0.0;

    auto last_point = traj_list_.back();
    double traj_x = last_point[0];
    double traj_y = last_point[1];
    int current_target = *target_idx_;

    if (current_target < *max_path_points_) {
        double tx = (*current_path_)[current_target].x;
        double ty = (*current_path_)[current_target].y;
        return std::hypot(traj_x - tx, traj_y - ty);
    }
    return 0.0;
}

// Tries to maintain required vel
double dwa_controller::calc_speed_ref_cost(double v) {
    return std::abs(v - ref_velocity_);
}

// --- MODIFIED ---
// Calculate cost upon each time step.
// CHANGED: Instead of averaging, we now check for immediate collision and return the MAXIMUM cost found.
double dwa_controller::calc_obstacle_cost() {
    if (traj_list_.empty() || !costmap_received_) {
        if (traj_list_.empty()) {
            ROS_INFO("ZERO OBS COST | REASON: Traj List Empty");
        }
        if (!costmap_received_) {
            ROS_INFO("ZERO OBS COST | REASON: No costmap received");
        }
        return 0.0;
    }
    
    double max_cost = 0.0;
    // Define a threshold for what is considered a collision.
    // In ROS costmaps: 100 is LETHAL, 99 is INSCRIBED_INFLATED_OBSTACLE.
    // We treat 99 and above as a collision.
    const uint8_t collision_threshold = 99; 

    // Iterate through trajectory 
    for (const auto& pt : traj_list_) {
        int mx, my;
        // Continue if out of costmap bounds
        if (!worldToCostmap(pt[0], pt[1], mx, my, traj_list_[0][0], traj_list_[0][1]))
            continue;

        uint8_t raw_cost = getCostmapCost(mx, my);

        // 1. Immediate check for collision
        if (raw_cost >= collision_threshold) {
             // Path hits an obstacle or gets too close. Return infinity.
             return std::numeric_limits<double>::infinity();
        }

        // 2. Track the maximum cost encountered on this path
        double normalized_cost = static_cast<double>(raw_cost) / 100.0; // normalize 0–1
        if (normalized_cost > max_cost) {
            max_cost = normalized_cost;
        }
    }

    // Return the maximum cost encountered scaled up.
    // The original code scaled the average 0-1 cost by 200.0. We do the same with the max cost.
    ROS_INFO("obstacle cost= %f,", 200.0 * max_cost);
    return 200.0 * max_cost;
}

// --- MODIFIED ---
// Exponential penalty for moving through high-cost areas.
// CHANGED: Use MAX instead of AVERAGE to avoid diluting the penalty.
double dwa_controller::calc_away_from_obstacle_cost() {
    if (traj_list_.empty() || !costmap_received_) {
        if (traj_list_.empty()) {
            ROS_INFO("ZERO Away from obs COST | REASON: Traj List Empty");
        }
        if (!costmap_received_) {
            ROS_INFO("ZERO Away from obs COST | REASON: No costmap received");
        }
        return 0.0;
    }

    double max_exp_cost = 0.0;

    // Iterate through trajectory
    for (const auto& pt : traj_list_) {
        int mx, my;
        // Continue if out of costmap bounds
        if (!worldToCostmap(pt[0], pt[1], mx, my, traj_list_[0][0], traj_list_[0][1]))
            continue;

        double c = static_cast<double>(getCostmapCost(mx, my)) / 100.0;
        double exp_cost = std::exp(5.0 * c); // exponential penalty
        
        if (exp_cost > max_exp_cost) {
            max_exp_cost = exp_cost;
        }
    }

    // Return max exponential cost instead of average
    return max_exp_cost;
}

// double dwa_controller::obstacle_check(double traj_x, double traj_y, double obs_x, double obs_y, double obs_width, double obs_height, double theta_diff) 
// {
//     dx = traj_x;
//     dy = traj_y;
//     dist = 0;
//     // Modify the  below code to calculate the collision distance based on the theta_diff
//     // Have to deduce an "obstacle radius" based on the closest point maybe
//     if (traj_x > obs_x + (obs_height / 2.0))
//         dx = obs_x + (obs_height / 2.0);
//     if (traj_x < obs_x - (obs_height / 2.0))
//         dx = obs_x - (obs_height / 2.0);
//     if (traj_y > obs_y + (obs_width / 2.0))
//         dy = obs_y + (obs_width / 2.0);
//     if (traj_y < obs_y - (obs_width / 2.0))
//         dy = obs_y - (obs_width / 2.0);
    

//     collision_dist = collision_robot_coeff * robot_radius_ + collision_obstacle_coeff * (std::sqrt(std::pow(dx-obs_x, 2) + std::pow(dy-obs_y, 2)));
    
//     dist = std::sqrt(std::pow(obs_x-traj_x, 2) + std::pow(obs_y-traj_y, 2));
    
//     return dist;
// }



// Main cost calc
DWAResult dwa_controller::dwa_main_control(double x, double y, double theta, double v, double omega) {
    std::vector<double> dw = calc_dynamic_window(v, omega);
    double min_cost = std::numeric_limits<double>::infinity();
    double best_v = v;
    double best_omega = omega;
    int worst_obsi = 0;
    double worst_mindist = std::numeric_limits<double>::infinity();
    double max_obstacle_cost = 0.0;
    double obs_cost;

    // --- NEW --- Visualization Setup
    visualization_msgs::MarkerArray traj_markers;
    int marker_id = 0;
    // --------------------------------

    for (int i = 0; i < vx_samples_; ++i) {
        double v_sample = dw[0] + (dw[1] - dw[0]) * i / std::max(1, vx_samples_ - 1);

        for (int j = 0; j < omega_samples_; ++j) {
            double omega_sample = dw[2] + (dw[3] - dw[2]) * j / std::max(1, omega_samples_ - 1);

            traj_list_ = calc_trajectory(x, y, theta, v_sample, omega_sample);
            double path_cost = calc_path_cost();
            double lookahead_cost = calc_lookahead_cost();
            double speed_ref_cost = calc_speed_ref_cost(v_sample);
            obs_cost = calc_obstacle_cost();

            // Check this, basically if no collision, can increase speed to move, this may cause issue, increase speed_ref_bias to maneuver around obstacles
            // if (obs_cost > 0) speed_ref_cost = 0;

            double away_cost = calc_away_from_obstacle_cost();

            // Experiment by removing some cost if needed
            double total_cost = path_distance_bias_ * path_cost 
            + goal_distance_bias_ * lookahead_cost 
            + occdist_scale_ * obs_cost 
            + speed_ref_bias_ * speed_ref_cost 
            + away_bias_ * away_cost;

            ROS_INFO_NAMED("cost_calculation",
                "--- Trajectory Cost Details ---\n"
                "\tPath Cost      (bias * cost): %.2f * %.2f = %.2f\n"
                "\tLookahead Cost (bias * cost): %.2f * %.2f = %.2f\n"
                "\tObstacle Cost  (bias * cost): %.2f * %.2f = %.2f\n"
                "\tSpeed Ref Cost (bias * cost): %.2f * %.2f = %.2f\n"
                "\tAway Cost      (bias * cost): %.2f * %.2f = %.2f\n"
                "----------------------------------\n"
                "\t>>> Total Cost: %.2f",
                // Arguments for the format string:
                path_distance_bias_, path_cost, path_distance_bias_ * path_cost,
                goal_distance_bias_, lookahead_cost, goal_distance_bias_ * lookahead_cost,
                occdist_scale_, obs_cost, occdist_scale_ * obs_cost,
                speed_ref_bias_, speed_ref_cost, speed_ref_bias_ * speed_ref_cost,
                away_bias_, away_cost, away_bias_ * away_cost,
                total_cost
            );

            // --- NEW --- Visualization Logic
            visualization_msgs::Marker traj_marker;
            // Assuming odom is your fixed frame, change if necessary
            traj_marker.header.frame_id = "odom"; 
            traj_marker.header.stamp = ros::Time::now();
            traj_marker.ns = "dwa_paths";
            traj_marker.id = marker_id++;
            traj_marker.type = visualization_msgs::Marker::LINE_STRIP;
            traj_marker.action = visualization_msgs::Marker::ADD;
            traj_marker.pose.orientation.w = 1.0;
            traj_marker.scale.x = 0.02; // Line width

            // Color based on cost
            if (std::isinf(total_cost)) {
                // RED for collision/infinite cost
                 traj_marker.color.r = 1.0; traj_marker.color.g = 0.0; traj_marker.color.b = 0.0; traj_marker.color.a = 1.0;
            } else {
                // Green -> Yellow gradient for valid paths
                // Normalize cost for coloring. 300.0 is an arbitrary scaling factor for visualization.
                float normalized_cost = std::min(1.0f, static_cast<float>(total_cost / 300.0)); 
                 traj_marker.color.r = normalized_cost;
                 traj_marker.color.g = 1.0 - normalized_cost;
                 traj_marker.color.b = 0.0;
                 traj_marker.color.a = 0.6; // Slightly transparent
            }

            for(const auto& pt : traj_list_) {
                geometry_msgs::Point p;
                p.x = pt[0];
                p.y = pt[1];
                p.z = 0;
                traj_marker.points.push_back(p);
            }
            traj_markers.markers.push_back(traj_marker);
            // --------------------------------

            if (obs_cost > max_obstacle_cost && !std::isinf(obs_cost)) {
                max_obstacle_cost = obs_cost;
                worst_obsi = j;
                worst_mindist = std::min(worst_mindist, obs_cost);
            }

            if (total_cost < min_cost) {
                min_cost = total_cost;
                best_v = v_sample;
                best_omega = omega_sample;
            }
        }
    }
    
    // --- NEW --- Publish the markers
    traj_pub_.publish(traj_markers);
    // ------------------------------

    // create a DWAResult struct to hold the results
    DWAResult result;
    ROS_INFO("obstacle cost DWA %f", obs_cost);
    result.best_v = best_v;
    result.best_omega = best_omega;
    result.obs_cost = max_obstacle_cost;
    return result;
}