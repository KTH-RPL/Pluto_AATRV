#pragma once

#include <vector>
#include <string>
#include <Eigen/Core>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Bool.h>
#include <limits>

// Waypoint struct for storing path points
struct Waypoint {
    double x;
    double y;
    double theta;
    Waypoint(double x_ = 0.0, double y_ = 0.0, double theta_ = 0.0)
        : x(x_), y(y_), theta(theta_) {}
};

// Obstacle struct (if needed for future development)
struct Obstacle {
    double x;
    double y;
    double width;
    double height;
    Obstacle(double x_ = 0.0, double y_ = 0.0, double width_ = 0.0, double height_ = 0.0)
        : x(x_), y(y_), width(width_), height(height_) {}
};

// Forward declaration of the DWA controller class
class dwa_controller;

// Struct to hold the result from the DWA controller
struct DWAResult {
    double best_v;
    double best_omega;
    double obs_cost;
    double lookahead_x;
    double lookahead_y;
    double lookahead_theta;
    DWAResult(double best_v_ = 0.0, double best_omega_ = 0.0, double obs_cost_ = 0.0, double lookahead_x_ = 0.0, double lookahead_y_ = 0.0, double lookahead_theta_ = 0.0)
    : best_v(best_v_), best_omega(best_omega_), obs_cost(obs_cost_), lookahead_x(lookahead_x_), lookahead_y(lookahead_y_), lookahead_theta(lookahead_theta_) {}
};

// Main controller class combining Preview Control and DWA
class PreviewController {
    public:
        PreviewController(double v = 1.0, double dt = 0.1, int preview_steps = 5);
        bool run_control(bool is_last_goal = false);
        void initialize_dwa_controller();
        void generate_snake_path(double start_x, double start_y, double start_theta);
        void generate_straight_path(double start_x, double start_y, double start_theta);
        void initialize_standalone_operation();
        double cross_track_error(double x_r, double y_r, double x_ref, double y_ref, double theta_ref);

        // Public member variables
        geometry_msgs::PoseStamped current_pose;
        std::vector<Waypoint> current_path;
        int max_path_points;
        dwa_controller* dwa_controller_ptr;

        // ROS Publishers and Subscribers
        ros::Publisher robot_vel_pub_;
        ros::Publisher lookahead_point_pub_;
        ros::Publisher path_pub_;
        ros::Publisher cross_track_error_pub_;
        ros::Publisher heading_error_pub_;
        ros::Publisher lookahead_heading_error_pub_;
        ros::Publisher current_v_pub_;
        ros::Publisher current_omega_pub_;
        ros::Publisher path_curvature_pub_;
        ros::Subscriber robot_pose_sub_;
        ros::Subscriber start_moving_sub_;
        ros::Subscriber stop_moving_sub_;

        // Control flags
        bool start_moving_;
        bool use_start_stop;

        // Callbacks
        void start_moving_callback(const std_msgs::Bool::ConstPtr& msg);
        void stop_moving_callback(const std_msgs::Bool::ConstPtr& msg);
        void robot_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);

    private:
        ros::Subscriber global_path_sub_;
        void global_path_callback(const nav_msgs::Path::ConstPtr& msg);

        // Private methods for controller logic
        void calcGains();
        double calculate_curvature(std::vector<double> x, std::vector<double> y);
        void calculate_all_curvatures();
        void compute_control(double cross_track_error, double heading_error, double path_curvature);
        double distancecalc(double x1, double y1, double x2, double y2);
        bool chkside(double path_theta);
        void lookahead_heading_error(double x_ref, double y_ref, double theta_ref);
        void stop_robot();
        void publish_cmd_vel();
        void publish_look_pose(geometry_msgs::PoseStamped look_pose);
        void publish_path();
        void boundvel(double ref_vel);
        void boundomega(double ref_omega);

        // State and matrix variables
        bool initial_alignment_;
        Eigen::Matrix3d A_;
        Eigen::Vector3d B_;
        Eigen::Vector3d D_;
        Eigen::Matrix3d Q_;
        Eigen::Matrix<double, 1, 1> R_;
        Eigen::RowVector3d Kb_;
        Eigen::MatrixXd Kf_;
        Eigen::MatrixXd Pc_;
        Eigen::MatrixXd Lmatrix_;
        Eigen::Vector3d x_state;

        // Parameters
        std::vector<double> Q_params_;
        double R_param_;
        double v_;
        double omega_;
        double linear_velocity_;
        double dt_;
        int preview_steps_;
        std::string path_type_;
        double straight_path_distance_;
        double path_length;
        double path_wavelength;
        double path_amplitude;
        double path_point_spacing;
        double cross_track_error_;
        double heading_error_;
        double lookahead_heading_error_;
        double prev_ey_;
        double prev_etheta_;
        double prev_omega_;
        double preview_loop_thresh;
        double robot_radius_;
        double path_curvature_;
        std::vector<double> path_curvatures_;
        double collision_robot_coeff;
        double collision_obstacle_coeff;
        double max_vel_;
        double max_omega_;
        double vel_acc_;
        double omega_acc_;
        double lookahead_distance_;
        double max_cte;
        double max_lookahead_heading_error;
        double goal_distance_threshold_;
        double vel_acc_bound;
        double omega_acc_bound;
        double kp_adjust_cte;
        double dwa_activation_cost_thresh_;
        double preview_reactivation_cost_thresh_;
        std::string active_controller_;
        double stop_robot_cost_thresh;
        double goal_reduce_factor;
        double density_check_radius_;
        double density_obstacle_thresh_;
        double density_dwa_activation_thresh_;
        double density_preview_reactivation_thresh_;

        // Robot state
        double robot_x;
        double robot_y;
        double robot_theta;
        int targetid;

        // ROS and state management
        ros::NodeHandle nh_;
        bool initial_pose_received_;
        bool path_generated_;
};

// DWA Controller class definition
class dwa_controller {
    public:
        dwa_controller(const std::vector<Waypoint>& path, int& target_idx, const int& max_points);
        DWAResult dwa_main_control(double x, double y, double theta, double v, double omega);

        // Costmap handling
        void costmap_callback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
        bool costmap_received_ = false;
        double query_cost_at_world(double wx, double wy, double robot_x, double robot_y, double robot_yaw);
        double calculate_local_obstacle_density(double robot_x, double robot_y, double radius, double obstacle_cost_threshold);

    private:
        // ROS components
        ros::NodeHandle nh_;
        ros::Subscriber occ_sub_;
        ros::Publisher traj_pub_;
        nav_msgs::OccupancyGrid occ_grid_;

        // DWA Parameters
        double predict_time_;
        double path_distance_bias_;
        double goal_distance_bias_;
        double occdist_scale_;
        double speed_ref_bias_;
        double away_bias_;
        double lookahead_heading_bias_;
        int vx_samples_;
        int omega_samples_;
        double max_omega_;
        double max_speed_;
        double min_speed_;
        double vel_acc_;
        double omega_acc_;
        double dt_dwa_;
        double robot_radius_;
        double ref_velocity_;
        double lookahead_distance_;
        double lookahead_obstacle_cost_thresh_;

        // Internal state
        double temp_lookahead_x;
        double temp_lookahead_y;
        double temp_lookahead_theta;
        std::vector<std::vector<double>> traj_list_;
        const std::vector<Waypoint>* current_path_;
        int* target_idx_;
        const int* max_path_points_;

        // DWA logic functions
        std::vector<double> calc_dynamic_window(double v, double omega);
        std::vector<std::vector<double>> calc_trajectory(double x, double y, double theta, double v, double omega);
        double calc_obstacle_cost();
        double calc_speed_ref_cost(double v);
        double calc_path_cost();
        double calc_lookahead_cost();
        double calc_away_from_obstacle_cost();
        double cross_track_error(double x_r, double y_r, double x_ref, double y_ref, double theta_ref);
        double calc_lookahead_heading_cost();
        bool chkside(double x1, double y1, double path_theta, double robot_x, double robot_y);
        bool worldToCostmap(double wx, double wy, int& mx, int& my, double robot_x, double robot_y, double robot_yaw);
        uint8_t getCostmapCost(int mx, int my);
        bool worldToMap(double wx, double wy, int& mx, int& my) const;
};
