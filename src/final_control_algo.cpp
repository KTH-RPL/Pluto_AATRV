#include "final_control_algo.h"
#include <Eigen/LU>
#include <cmath>
#include <unsupported/Eigen/MatrixFunctions>
#include <std_msgs/Float64.h>
#include <ros/ros.h>
#include <limits>
#include <algorithm>
#include <thread>
#include <cmath>

PreviewController::PreviewController(double v, double dt, int preview_steps)
    : linear_velocity_(v), dt_(dt), preview_steps_(preview_steps), 
      prev_ey_(0), prev_etheta_(0), prev_omega_(0), nh_(), targetid(0),
      initial_pose_received_(false), path_generated_(false) // Check on adding NodeHandle here 
{
    robot_vel_pub_ = nh_.advertise<geometry_msgs::Twist>("/atrv/cmd_vel", 10);
    lookahead_point_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("lookahead_point", 10);
    path_pub_ = nh_.advertise<nav_msgs::Path>("planned_path", 10);
    robot_pose_sub_ = nh_.subscribe("/robot_pose", 10, &PreviewController::robot_pose_callback, this);

    // --- INITIALIZE DEBUG PUBLISHERS ---
    cross_track_error_pub_ = nh_.advertise<std_msgs::Float64>("debug/cross_track_error", 10);
    heading_error_pub_ = nh_.advertise<std_msgs::Float64>("debug/heading_error", 10);
    lookahead_heading_error_pub_ = nh_.advertise<std_msgs::Float64>("debug/lookahead_heading_error", 10);
    current_v_pub_ = nh_.advertise<std_msgs::Float64>("debug/current_v", 10);
    current_omega_pub_ = nh_.advertise<std_msgs::Float64>("debug/current_omega", 10);
    path_curvature_pub_ = nh_.advertise<std_msgs::Float64>("debug/path_curvature", 10);
    // -----------------------------------

    // Params to modify for different scenarios
    // Reference optimal velocity for robot
    nh_.param("preview_controller/linear_velocity", linear_velocity_, 1.0);

    // Frequency of the controller
    nh_.param("preview_controller/preview_dt", dt_, 0.1);

    // Control params 
    nh_.param("preview_controller/max_vel", max_vel_, 0.4);
    nh_.param("preview_controller/max_omega", max_omega_, 0.2);
    nh_.param("preview_controller/vel_acc", vel_acc_, 0.5);
    nh_.param("preview_controller/omega_acc", omega_acc_, 0.4);
    // nh_.param("preview_controller/max_domega", max_domega_, 0.2);

    // Measure robot_radius and set
    nh_.param("preview_controller/robot_radius", robot_radius_, 0.5);
    nh_.param("preview_controller/lookahead_distance", lookahead_distance_, 1.0);

    // Max cross track error so robot moves towards the target first
    nh_.param("preview_controller/max_cte", max_cte, 1.5);

    // Thresh to adjust orientation before moving to target
    nh_.param("preview_controller/max_lookahead_heading_error", max_lookahead_heading_error, 0.2);

    // For optimaizing the cost function, try diff params 
    nh_.param("preview_controller/preview_loop_thresh", preview_loop_thresh, 1e-6);

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
    
    // R matrix for the preview controller
    nh_.param("preview_controller/R", R_param_, 1.0);

    // Calculate the velocity and omega acceleration bounds for timestep
    vel_acc_bound = vel_acc_ * dt_;
    omega_acc_bound = omega_acc_ * dt_;
}

void PreviewController::robot_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_pose = *msg;
    robot_x = current_pose.pose.position.x;
    robot_y = current_pose.pose.position.y;
    robot_theta = current_pose.pose.orientation.z;
    
    // Generate path on first pose received
    if (!initial_pose_received_) {
        initial_pose_received_ = true;
        ROS_INFO("Initial robot pose received: x=%.2f, y=%.2f, theta=%.2f", robot_x, robot_y, robot_theta);
        generate_snake_path(robot_x, robot_y, robot_theta);
        initialize_dwa_controller();
        path_generated_ = true;
        publish_path();  // Publish the generated path
        ROS_INFO("Snake path generated with %d waypoints", max_path_points);
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
    double amplitude = 14.0;
    double wavelength = 20.0;
    double length = 40.0;
    double point_spacing = 0.5;
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
    ROS_INFO("Generated snake path with %d points", max_path_points);
}

// Initialize standalone operation
void PreviewController::initialize_standalone_operation() {
    ROS_INFO("Initializing standalone preview controller...");
    ROS_INFO("Waiting for robot pose...");
    
    // Wait for initial pose and path generation
    ros::Rate rate(10);
    while (ros::ok() && !path_generated_) {
        ros::spinOnce();
        rate.sleep();
    }
    
    if (path_generated_) {
        // Start control timer
        control_timer_ = nh_.createTimer(ros::Duration(dt_), 
            [this](const ros::TimerEvent&) { 
                if (path_generated_) {
                    bool goal_reached = this->run_control();
                    if (goal_reached) {
                        ROS_INFO("Goal reached! Stopping control.");
                        this->control_timer_.stop();
                    }
                }
            });
        
        ROS_INFO("Standalone preview controller initialized successfully!");
        ROS_INFO("Control loop started with dt = %.3f seconds", dt_);
    } else {
        ROS_ERROR("Failed to generate path. Exiting.");
    }
}

// Main standalone control loop
void PreviewController::run_standalone_control() {
    initialize_standalone_operation();
    
    // Keep the node running
    ros::spin();
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
    Q_(0, 0) = 5;
    Q_(1, 1) = 6;
    Q_(2, 2) = 5;
    R_ << 1;

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
    bool bounded = false;
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

    // If CTE high, focus on moving to lookahead
    if (std::abs(cross_track_error_) > max_cte) {
        lookahead_heading_error(current_path[targetid].x, current_path[targetid].y, current_path[targetid].theta);
        
        // --- PUBLISH LOOKAHEAD HEADING ERROR ---
        std_msgs::Float64 lhe_msg;
        lhe_msg.data = lookahead_heading_error_;
        lookahead_heading_error_pub_.publish(lhe_msg);
        // ---------------------------------------

        if (lookahead_heading_error_ > max_lookahead_heading_error) {
            boundvel(0.0001);
            boundomega(kp_adjust_cte * lookahead_heading_error_);
            bounded = true;
        } else {
            boundvel(cross_track_error_ * kp_adjust_cte);
            bounded = true;
        }
    }

    // Call DWA controller
    DWAResult dwa_result = dwa_controller_ptr->dwa_main_control(robot_x, robot_y, robot_theta, v_, omega_);

    double x_goal = current_path[max_path_points - 1].x;
    double y_goal = current_path[max_path_points - 1].y;
    double goal_distance = distancecalc(robot_x, robot_y, x_goal, y_goal);


    // Here the obstacle calculation must be done correctly based on perception result
    double obstacle_radius = 0.5;

    // DWA cause obstacle too close, add condition to stop robot if too close to obstacle or in obstacle
    if (dwa_result.obsi_mindist < collision_robot_coeff * robot_radius_ + collision_obstacle_coeff * obstacle_radius) {
        v_ = dwa_result.best_v;
        omega_ = dwa_result.best_omega;
    } 
    
    else 
    // Call preview Controller
    {
        if (!bounded)
            // Increase vel to the target vel
            boundvel(linear_velocity_);
        heading_error_ = robot_theta - current_path[targetid].theta;
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

        std::vector<double> x_vals = {current_path[targetid].x};
        std::vector<double> y_vals = {current_path[targetid].y};

        // Calls to compute the omega
        compute_control(cross_track_error_, heading_error_, path_curvature_);
        
        // --- PUBLISH PATH CURVATURE ---
        std_msgs::Float64 pc_msg;
        pc_msg.data = path_curvature_;
        path_curvature_pub_.publish(pc_msg);
        // ------------------------------
    }

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
    if (std::fabs(std::tan(path_theta)) < 1e-6) {
        m = std::numeric_limits<double>::infinity();
    }
    double ineq = robot_y - (m * robot_x) - y1 + (m * x1);
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
    std::vector<double> x_vals, y_vals;
    for (int i = 0; i <= preview_steps_; ++i) {
        if (targetid + 2 < max_path_points) {
            x_vals.push_back(current_path[targetid + 1].x);
            y_vals.push_back(current_path[targetid + 1].y);
            x_vals.push_back(current_path[targetid + 2].x);
            y_vals.push_back(current_path[targetid + 2].y);
        }
        path_curvature_ = calculate_curvature(x_vals, y_vals);
        x_vals.clear();
        y_vals.clear();
        preview_curv[i] = path_curvature;
    }

    Eigen::VectorXd curv_vec = Eigen::Map<Eigen::VectorXd>(preview_curv.data(), preview_steps_ + 1);
    calcGains();
    double u_fb = -(Kb_ * x_state)(0);
    double u_ff = -(Kf_ * curv_vec)(0);
    omega_ = u_fb + u_ff;
    std::cout << " Theta error: " << heading_error << ", Omega: " << omega_ << ", ey " << cross_track_error << std::endl;
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
    look_pose.header.frame_id = "map"; //Change the frame if needed
    lookahead_point_pub_.publish(look_pose);
}

// Publish the planned path
void PreviewController::publish_path() {
    nav_msgs::Path path_msg;
    path_msg.header.stamp = ros::Time::now();
    path_msg.header.frame_id = "map";
    
    for (const auto& waypoint : current_path) {
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "map";
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
    ROS_INFO("Published planned path with %zu waypoints to 'planned_path' topic", current_path.size());
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
    nh_.param("preview_controller/max_speed", max_speed_, 1.0);
    nh_.param("preview_controller/max_omega", max_omega_, 0.2);
    nh_.param("preview_controller/dt_dwa", dt_dwa_, 0.1);
    nh_.param("preview_controller/linear_velocity", ref_velocity_, 0.8);
    nh_.param("preview_controller/collision_robot_coeff", collision_robot_coeff, 2.0);
    nh_.param("preview_controller/collision_obstacle_coeff", collision_obstacle_coeff, 2.0);
    
    obstacle_sub_ = nh_.subscribe("/detected_obstacles", 10, &dwa_controller::obstacle_callback, this);

}

void dwa_controller::obstacle_callback(const visualization_msgs::MarkerArray::ConstPtr& msg) {
    obstacles.clear(); // Clear previous obstacles
    for (const auto& marker : msg->markers) {
        double x = marker.pose.position.x;
        double y = marker.pose.position.y;
        double height = marker.scale.y;
        double width = marker.scale.x;
        obstacles.emplace_back(x, y, width, height);
    }
}



// bounds and calculates window within acc
std::vector<double> dwa_controller::calc_dynamic_window(double v, double omega) {
    std::vector<double> Vs = {min_speed_, max_speed_, -max_omega_, max_omega_};
    std::vector<double> Vd = {v - vel_acc_ * dt_dwa_, v + vel_acc_ * dt_dwa_, omega - omega_acc_ * dt_dwa_, omega + omega_acc_ * dt_dwa_};
    std::vector<double> vr = {std::max(Vs[0], Vd[0]), std::min(Vs[1], Vd[1]), std::max(Vs[2], Vd[2]), std::min(Vs[3], Vd[3])};
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

// Calculate cost upon each time step and adds
double dwa_controller::calc_obstacle_cost() {
    if (traj_list_.empty())
        return 0.0;

    obs_cost = 0.0;
    min_obs_num = 0;
    min_obs_dist = std::numeric_limits<double>::infinity();

    for (const auto& point : traj_list_) {
        for (size_t i = 0; i < obstacles.size(); ++i) {
            dist = obstacle_check(point[0], point[1], obstacles[i].x, obstacles[i].y, obstacles[i].width, obstacles[i].height);
            
            if (dist < collision_dist) {
                obs_cost += 1.0 - dist / (collision_dist + 0.01);
            }

            if (dist < min_obs_dist) {
                min_obs_dist = dist;
                min_obs_num = i;
            }
        }
    }
    return 500.0 * obs_cost;
}

// Experimental, can remove if not needed, tries to move away from obstacle if getting chased, increase predict time again if needed
double dwa_controller::calc_away_from_obstacle_cost(int obs_idx, double v, double omega) {
    if (traj_list_.empty() || obs_idx >= obstacles.size())
        return 0.0;

    auto last_point = traj_list_.back();
    double traj_x = last_point[0];
    double traj_y = last_point[1];
    double traj_theta = last_point[2];

    double obs_x = obstacles[obs_idx].x;
    double obs_y = obstacles[obs_idx].y;

    double theta_er = std::atan2(obs_y - traj_y, obs_x - traj_x) - traj_theta;
    theta_er = std::atan2(std::sin(theta_er), std::cos(theta_er));

    dist = std::hypot(traj_x - obs_x, traj_y - obs_y);
    return 50.0 / ((std::abs(theta_er) + 1.0) * dist + 0.01);
}

double dwa_controller::obstacle_check(double traj_x, double traj_y, double obs_x, double obs_y, double obs_width, double obs_height) 
{
    dx = traj_x;
    dy = traj_y;
    dist = 0;

    if (traj_x > obs_x + (obs_width / 2.0))
        dx = obs_x + obs_width / 2.0;
    if (traj_x < obs_x - (obs_width / 2.0))
        dx = obs_x - (obs_width / 2.0);
    if (traj_y > obs_y + (obs_height / 2.0))
        dy = obs_y + obs_height / 2.0 - traj_y;
    if (traj_y < obs_y - (obs_height / 2.0))
        dy = obs_y - (obs_height / 2.0);
    

    collision_dist = collision_robot_coeff * robot_radius_ + collision_obstacle_coeff * (std::sqrt(std::pow(dx-obs_x, 2) + std::pow(dy-obs_y, 2)));
    
    dist = std::sqrt(std::pow(obs_x-traj_x, 2) + std::pow(obs_y-traj_y, 2));
    
    return dist;
}



// Main cost calc
DWAResult dwa_controller::dwa_main_control(double x, double y, double theta, double v, double omega) {
    std::vector<double> dw = calc_dynamic_window(v, omega);
    double min_cost = std::numeric_limits<double>::infinity();
    double best_v = v;
    double best_omega = omega;
    int worst_obsi = 0;
    double worst_mindist = std::numeric_limits<double>::infinity();

    for (int i = 0; i < vx_samples_; ++i) {
        double v_sample = dw[0] + (dw[1] - dw[0]) * i / std::max(1, vx_samples_ - 1);

        for (int j = 0; j < omega_samples_; ++j) {
            double omega_sample = dw[2] + (dw[3] - dw[2]) * j / std::max(1, omega_samples_ - 1);

            traj_list_ = calc_trajectory(x, y, theta, v_sample, omega_sample);
            double path_cost = calc_path_cost();
            double lookahead_cost = calc_lookahead_cost();
            double speed_ref_cost = calc_speed_ref_cost(v_sample);
            double obs_cost = calc_obstacle_cost();

            if (obs_cost > 0) speed_ref_cost = 0;

            double away_cost = calc_away_from_obstacle_cost(min_obs_num, v_sample, omega_sample);

            double total_cost = path_distance_bias_ * path_cost + goal_distance_bias_ * lookahead_cost +
                                occdist_scale_ * obs_cost + speed_ref_bias_ * speed_ref_cost + away_bias_ * away_cost;

            if (total_cost < min_cost) {
                min_cost = total_cost;
                best_v = v_sample;
                best_omega = omega_sample;
                worst_obsi = min_obs_num;
                worst_mindist = min_obs_dist;
            }
        }
    }

    return {best_v, best_omega, worst_obsi, worst_mindist};
}