#pragma once

#include <vector>
#include <Eigen/Core>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <yaml-cpp/yaml.h>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
#include <limits>
#include <std_msgs/Float64.h> // Include for debug message type
#include <nav_msgs/OccupancyGrid.h>

struct Waypoint {
    double x;
    double y;
    double theta;
    Waypoint(double x_ = 0.0, double y_ = 0.0, double theta_ = 0.0)
        : x(x_), y(y_), theta(theta_) {}
};

struct Obstacle {
    double x;
    double y;
    double width;
    double height;
    Obstacle(double x_ = 0.0, double y_ = 0.0, double width_ = 0.0, double height_ = 0.0)
        : x(x_), y(y_), width(width_), height(height_) {}
};

class dwa_controller;

struct DWAResult {
    double best_v;
    double best_omega;
    double obs_cost;
    DWAResult(double best_v_ = 0.0, double best_omega_ = 0.0, double obs_cost_ = 0.0) 
    : best_v(best_v_), best_omega(best_omega_), obs_cost(obs_cost_) {}
};

class PreviewController {
    public:
        PreviewController(double v = 1.0, double dt = 0.1, int preview_steps = 5);
        void initialize_dwa_controller();
        void generate_snake_path(double start_x, double start_y, double start_theta);
        void initialize_standalone_operation();
        void run_standalone_control();
        double cross_track_error(double x_r, double y_r, double x_ref, double y_ref, double theta_ref);
        geometry_msgs::PoseStamped current_pose;
        std::vector<Waypoint> current_path;
        int max_path_points;
        dwa_controller* dwa_controller_ptr;
        ros::Publisher robot_vel_pub_;
        ros::Publisher lookahead_point_pub_;
        ros::Publisher path_pub_;
        
        // --- ADDED DEBUG PUBLISHERS ---
        ros::Publisher cross_track_error_pub_;
        ros::Publisher heading_error_pub_;
        ros::Publisher lookahead_heading_error_pub_;
        ros::Publisher current_v_pub_;
        ros::Publisher current_omega_pub_;
        ros::Publisher path_curvature_pub_;
        // ------------------------------
        
        ros::Subscriber robot_pose_sub_;
        ros::Subscriber start_moving_sub_;
        bool start_moving_ = false;
        void start_moving_callback(const std_msgs::Bool::ConstPtr& msg);
        void robot_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& msg);

    private:
        bool run_control(bool is_last_goal = false);
        void calcGains();
        double calculate_curvature(const std::vector<double> x, const std::vector<double> y);
        void calculate_all_curvatures(); // Calculate curvatures for all path points
        void compute_control(double cross_track_error, double heading_error, double path_curvature);
        double distancecalc(double x1, double y1, double x2, double y2);
        bool chkside(double path_theta);
        void lookahead_heading_error(double x_ref, double y_ref, double theta_ref);
        void stop_robot();
        void publish_cmd_vel();
        void publish_look_pose(const geometry_msgs::PoseStamped look_pose);
        void publish_path();
        void boundvel(double ref_vel);
        void boundomega(double ref_omega);
        int closest_point(double x, double y);

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

        std::vector<double> Q_params_;
        double R_param_;

        Eigen::Vector3d x_state;

        double v_;
        double omega_;
        double linear_velocity_;
        double dt_;
        double preview_dt_;
        int preview_steps_;

        double cross_track_error_;
        double heading_error_;
        double heading_error_dot_;
        double lookahead_heading_error_;
        double prev_ey_;
        double prev_etheta_;
        double prev_omega_;
        double preview_loop_thresh;
        double robot_radius_;
        double path_curvature_;
        std::vector<double> path_curvatures_; // Precomputed curvatures for all path points
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
        double obst_cost_thresh;
        double stop_robot_cost_thresh;
        double robot_x;
        double robot_y;
        double robot_theta;

        int targetid;
        ros::NodeHandle nh_;
        
        // Standalone operation variables
        bool initial_pose_received_;
        bool path_generated_;
        ros::Timer control_timer_;
};

class dwa_controller {
    public:
        dwa_controller();
        dwa_controller(const std::vector<Waypoint>& path, int& target_idx, const int& max_points);
        DWAResult dwa_main_control(double x, double y, double theta, double v, double omega);
        ros::Subscriber occ_sub_;
        nav_msgs::OccupancyGrid occ_grid_;
        void costmap_callback(const nav_msgs::OccupancyGrid::ConstPtr& msg);
        bool costmap_received_ = false;

    private:
        ros::NodeHandle nh_;
        double predict_time_;
        double path_distance_bias_;
        double goal_distance_bias_;
        double occdist_scale_;
        double speed_ref_bias_;
        double away_bias_;
        int vx_samples_;
        int omega_samples_;
        double max_omega_;
        double max_speed_;
        double min_speed_;
        double vel_acc_;
        double omega_acc_;
        double dt_dwa_;
        double max_domega_;
        double robot_radius_;
        double collision_robot_coeff;
        double collision_obstacle_coeff;
        double ref_velocity_;
        double collision_dist;

        double dx;
        double dy;
        double dist;

        double obs_cost;

        int min_obs_num;
        double min_obs_dist;
        std::vector<std::vector<double>> traj_list_;
        std::vector<Obstacle> obstacles;

        const std::vector<Waypoint>* current_path_;
        int* target_idx_;
        const int* max_path_points_;

        std::vector<double> calc_dynamic_window(double v, double omega);
        std::vector<std::vector<double>> calc_trajectory(double x, double y, double theta, double v, double omega);
        double calc_obstacle_cost();
        double calc_speed_ref_cost(double v);
        double calc_heading_cost();
        double calc_path_cost();
        double calc_lookahead_cost();
        double calc_away_from_obstacle_cost();
        double cross_track_error(double x_r, double y_r, double x_ref, double y_ref, double theta_ref);
        bool worldToCostmap(double x, double y, int& mx, int& my, double robot_x, double robot_y);
        uint8_t getCostmapCost(int mx, int my);

        void obstacle_callback(const visualization_msgs::MarkerArray::ConstPtr& msg);
        double obstacle_check(double traj_x, double traj_y, double obs_x, double obs_y, double obs_width, double obs_height, double theta_diff);
};