// bt_controller_node.cpp
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <thread>
#include <mutex>
#include <vector>
#include <memory>

// BehaviorTree.CPP
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/actions/always_failure_node.h>
#include <behaviortree_cpp_v3/actions/always_successful_node.h>

// Include your action definition (same as Python)
#include <robot_controller/PlanGlobalPathAction.h>

// Include your control header (as you described)
#include "final_control_algo.h"

using PlanClient = actionlib::SimpleActionClient<robot_controller::PlanGlobalPathAction>;
using namespace BT;

struct SharedData
{
    std::mutex mutex;

    // subscribers data
    geometry_msgs::PoseStamped robot_pose;
    bool have_robot_pose = false;

    float transform_probability = -1.0f;
    bool have_transform_probability = false;

    std::vector<geometry_msgs::Pose> goals;
    bool have_goals = false;

    std::vector<geometry_msgs::Pose> global_plan;
    bool have_global_plan = false;

    // pose history for localization fallback (simple)
    std::vector< std::pair<double, robot_controller::PlanGlobalPathResult> > dummy_history;

    // multi-goal
    size_t goal_index = 0;

    // publishers
    ros::Publisher initial_pose_pub;
    ros::Publisher goal_pose_pub;

    // node handle reference to create pubs/subs later
    ros::NodeHandle* nh = nullptr;
};

static std::shared_ptr<SharedData> shared_ptr_;

// ----------------- subscribers callbacks -----------------
void robotPoseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    auto s = shared_ptr_;
    std::lock_guard<std::mutex> lk(s->mutex);
    s->robot_pose = *msg;
    s->have_robot_pose = true;
}

void waypointsCallback(const nav_msgs::Path::ConstPtr& msg)
{
    auto s = shared_ptr_;
    std::lock_guard<std::mutex> lk(s->mutex);
    s->goals.clear();
    for (const auto& ps : msg->poses) {
        s->goals.push_back(ps.pose);
    }
    s->have_goals = !s->goals.empty();
    ROS_INFO("[BT] Received %zu goals", s->goals.size());
}

void transformProbCallback(const std_msgs::Float32::ConstPtr& msg)
{
    auto s = shared_ptr_;
    std::lock_guard<std::mutex> lk(s->mutex);
    s->transform_probability = msg->data;
    s->have_transform_probability = true;
}

// ----------------- BT Nodes -----------------

// GoalReached condition
class GoalReachedCondition : public ConditionNode
{
public:
    GoalReachedCondition(const std::string& name, const NodeConfiguration& config)
    : ConditionNode(name, config) {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override
    {
        auto s = shared_ptr_;
        std::lock_guard<std::mutex> lk(s->mutex);

        if (!s->have_goals || !s->have_robot_pose) {
            ROS_WARN_THROTTLE(5.0, "[GoalReached] missing goals or robot_pose");
            return NodeStatus::RUNNING;
        }

        const geometry_msgs::Point& final_goal = s->goals.back().position;
        const geometry_msgs::Point& cur = s->robot_pose.pose.position;
        double dx = cur.x - final_goal.x;
        double dy = cur.y - final_goal.y;
        double dist = std::sqrt(dx*dx + dy*dy);

        if (dist < 0.6) {
            ROS_INFO("[GoalReached] Final goal reached (dist=%.3f)", dist);
            return NodeStatus::SUCCESS;
        } else {
            return NodeStatus::FAILURE;
        }
    }
};

// CheckLocalization condition
class CheckLocalization : public ConditionNode
{
public:
    CheckLocalization(const std::string& name, const NodeConfiguration& config)
    : ConditionNode(name, config), max_score_(4.2f) {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override
    {
        auto s = shared_ptr_;
        {
            std::lock_guard<std::mutex> lk(s->mutex);
            if (!s->have_transform_probability) {
                ROS_WARN_THROTTLE(5.0, "[CheckLocalization] no localization score received");
                return NodeStatus::RUNNING;
            }

            if (s->transform_probability > max_score_) {
                ROS_WARN("[CheckLocalization] Poor localization score: %.3f", s->transform_probability);
                // attempt to find a last good pose in recent history -- simplified:
                // Here we just publish current robot_pose if available (you can refine)
                if (s->have_robot_pose) {
                    geometry_msgs::PoseStamped corrected = s->robot_pose;
                    corrected.header.stamp = ros::Time::now();
                    corrected.header.frame_id = "map";
                    s->initial_pose_pub.publish(corrected);
                    ROS_INFO("[CheckLocalization] Published initial_pose correction");
                }
                return NodeStatus::FAILURE;
            }
        }
        return NodeStatus::SUCCESS;
    }

private:
    float max_score_;
};

// GlobalPathExist condition
class GlobalPathExist : public ConditionNode
{
public:
    GlobalPathExist(const std::string& name, const NodeConfiguration& config)
    : ConditionNode(name, config) {}

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override
    {
        auto s = shared_ptr_;
        std::lock_guard<std::mutex> lk(s->mutex);
        if (s->have_global_plan && !s->global_plan.empty()) {
            return NodeStatus::SUCCESS;
        } else {
            return NodeStatus::FAILURE;
        }
    }
};

// GlobalPathClient action node (async)
class GlobalPathClientNode : public AsyncActionNode
{
public:
    GlobalPathClientNode(const std::string& name, const NodeConfiguration& config)
    : AsyncActionNode(name, config),
      client_("plan_global_path", true),
      sent_goal_(false)
    {
        // wait for server in constructor might block; keep non-blocking here but wait later
    }

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override
    {
        auto s = shared_ptr_;
        // If we already have path, return SUCCESS quickly
        {
            std::lock_guard<std::mutex> lk(s->mutex);
            if (s->have_global_plan && !s->global_plan.empty()) {
                return NodeStatus::SUCCESS;
            }
        }

        if (!sent_goal_) {
            if (!client_.waitForServer(ros::Duration(5.0))) {
                ROS_WARN("[GlobalPathClient] Action server not available yet");
                return NodeStatus::RUNNING;
            }

            // build goal
            robot_controller::PlanGlobalPathGoal goal_msg;
            {
                std::lock_guard<std::mutex> lk(s->mutex);
                if (!s->have_goals) {
                    ROS_WARN("[GlobalPathClient] no goals to send");
                    return NodeStatus::RUNNING;
                }

                // The action goal message in Python used a PoseArray in pluto_goal.goal.
                // Here we construct similarly if PlanGlobalPathGoal::goal is a PoseArray.
                // Replace field names if different.
                goal_msg.goal.header.stamp = ros::Time::now();
                // frame_id left as default
                for (const auto& p : s->goals) {
                    geometry_msgs::PoseStamped ps;
                    ps.header.stamp = ros::Time::now();
                    ps.pose = p;
                    goal_msg.goal.poses.push_back(ps);
                }
            }

            // send goal with callbacks
            client_.sendGoal(goal_msg,
                             boost::bind(&GlobalPathClientNode::doneCb, this, _1, _2),
                             PlanClient::SimpleActiveCallback(),
                             boost::bind(&GlobalPathClientNode::feedbackCb, this, _1));
            sent_goal_ = true;

            // publish current goal pose (if available)
            {
                std::lock_guard<std::mutex> lk(s->mutex);
                if (s->have_goals && s->goal_index < s->goals.size()) {
                    geometry_msgs::PoseStamped goal_pose;
                    goal_pose.header.stamp = ros::Time::now();
                    goal_pose.pose = s->goals[s->goal_index];
                    s->goal_pose_pub.publish(goal_pose);
                    ROS_INFO("[GlobalPathClient] Sent goal %zu/%zu", s->goal_index + 1, s->goals.size());
                }
            }
        }

        // Wait for result - check state
        actionlib::SimpleClientGoalState state = client_.getState();
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
            // handled in doneCb, but ensure success returned
            sent_goal_ = false;
            return NodeStatus::SUCCESS;
        } else if (state == actionlib::SimpleClientGoalState::ABORTED ||
                   state == actionlib::SimpleClientGoalState::REJECTED) {
            ROS_WARN("[GlobalPathClient] planning failed with state %s", state.toString().c_str());
            sent_goal_ = false;
            return NodeStatus::FAILURE;
        }

        return NodeStatus::RUNNING;
    }

    void halt() override
    {
        AsyncActionNode::halt();
        if (client_.isServerConnected()) {
            client_.cancelAllGoals();
        }
        sent_goal_ = false;
    }

private:
    PlanClient client_;
    bool sent_goal_;

    void feedbackCb(const robot_controller::PlanGlobalPathFeedbackConstPtr& fb)
    {
        // You can visualize here similar to Python (calls to gmap_utility not included).
        ROS_INFO_THROTTLE(5.0, "[GlobalPathClient] feedback received");
    }

    void doneCb(const actionlib::SimpleClientGoalState& state,
                const robot_controller::PlanGlobalPathResultConstPtr& result)
    {
        // When action succeeds, save global_plan into shared
        auto s = shared_ptr_;
        std::lock_guard<std::mutex> lk(s->mutex);
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED && result) {
            s->global_plan.clear();
            for (const auto& ps : result->global_plan.poses) {
                s->global_plan.push_back(ps.pose);
            }
            s->have_global_plan = !s->global_plan.empty();
            ROS_INFO("[GlobalPathClient] Received final global plan with %zu poses", s->global_plan.size());
        } else {
            ROS_WARN("[GlobalPathClient] doneCb: state=%s", state.toString().c_str());
        }
    }
};

// ControlPlanner node (calls your PreviewController::run_control)
class ControlPlannerNode : public AsyncActionNode
{
public:
    ControlPlannerNode(const std::string& name, const NodeConfiguration& config)
    : AsyncActionNode(name, config), controller_()
    {
        // controller_ is from your final_control_algo.h
    }

    static PortsList providedPorts() { return {}; }

    NodeStatus tick() override
    {
        auto s = shared_ptr_;
        {
            std::lock_guard<std::mutex> lk(s->mutex);
            if (!s->have_global_plan) {
                ROS_WARN_THROTTLE(3.0, "[ControlPlanner] No global path; cannot run control");
                return NodeStatus::FAILURE;
            }
        }

        // call the control loop once per tick. Your C++ run_control returns true when
        // goal reached; otherwise keep running.
        bool reached = controller_.run_control(false); // matches your signature

        if (reached) {
            ROS_INFO("[ControlPlanner] Controller reports goal reached.");
            return NodeStatus::SUCCESS;
        } else {
            return NodeStatus::RUNNING;
        }
    }

    void halt() override
    {
        AsyncActionNode::halt();
        // If your controller has a stop/abort API, call it here.
    }

private:
    PreviewController controller_;
};

// ----------------- main -----------------
int main(int argc, char** argv)
{
    ros::init(argc, argv, "bt_behavior_tree_controller");
    ros::NodeHandle nh;

    shared_ptr_ = std::make_shared<SharedData>();
    shared_ptr_->nh = &nh;

    // publishers
    shared_ptr_->initial_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/initial_pose", 1, true);
    shared_ptr_->goal_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/goal_pose", 1, true);

    // subscribers
    ros::Subscriber sub_pose = nh.subscribe("/robot_pose", 10, robotPoseCallback);
    ros::Subscriber sub_waypoints = nh.subscribe("/waypoints", 5, waypointsCallback);
    ros::Subscriber sub_tfprob = nh.subscribe("/transform_probability", 10, transformProbCallback);

    // Register nodes in BehaviorTree factory
    BehaviorTreeFactory factory;
    factory.registerNodeType<GoalReachedCondition>("GoalReached");
    factory.registerNodeType<CheckLocalization>("CheckLocalization");
    factory.registerNodeType<GlobalPathExist>("GlobalPathExist");
    factory.registerNodeType<GlobalPathClientNode>("GlobalPathClient");
    factory.registerNodeType<ControlPlannerNode>("ControlPlanner");

    // Build equivalent tree: Selector( GoalReached, Sequence(CheckLocalization, Selector(GlobalPathExist, GlobalPathClient), ControlPlanner) )
    // Using XML is more typical for BT.CPP, but we'll build programmatically here:

    // compose tree nodes
    Tree tree;
    {
        // Root RSequence (here use Sequence)
        auto root = factory.createTreeFromText(R"(
<root main_tree_to_execute = "MainTree" >
  <BehaviorTree ID="MainTree">
    <Sequence name="Main sequence">
      <Fallback name="Goal Reached Fallback">
        <GoalReached />
        <Sequence name="Pluto Sequence">
          <CheckLocalization />
          <Fallback name="Global Planner Fallback">
            <GlobalPathExist />
            <GlobalPathClient />
          </Fallback>
          <ControlPlanner />
        </Sequence>
      </Fallback>
    </Sequence>
  </BehaviorTree>
</root>
)");
        tree = std::move(root);
    }

    ros::Rate loop_rate(10.0); // 10 Hz tick
    while (ros::ok()) {
        // tick root
        tree.rootNode()->executeTick();
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
