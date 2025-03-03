#!/usr/bin/env python3

# DD2421 Introduction to Robotics
# Assignment 1
# Prasetyo Wibowo L. S. <pwlsa@kth.se> 

import rospy
import actionlib
import irob_assignment_1.msg
from irob_assignment_1.srv import GetSetpoint, GetSetpointRequest, GetSetpointResponse
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import tf2_ros
import tf2_geometry_msgs
from math import atan2, hypot, sqrt

# Use to transform between frames
tf_buffer = tf2_ros.Buffer()
listener = None

# The exploration simple action client
goal_client = None

# The collision avoidance service client
control_client = None

# The velocity command publisher
pub = None

# The robots frame
robot_frame_id = "base_link"

# Max linear velocity (m/s)
max_linear_velocity = 0.5
# Max angular velocity (rad/s)
max_angular_velocity = 0.75


def limit_speed(a,b):
    signq = a/abs(a)
    return signq * min(abs(a),abs(b))

def stop_moving():
    rospy.loginfo('========= Stop Moving =========')
    pub.publish(Twist())

def move(path):
    global control_client, robot_frame_id, pub
    
    
    rate = rospy.Rate(10)

    msg = Twist()
    
    while path.poses:            
        # Call service client with path
        res = control_client(path)
        setpoint = res.setpoint #result: setpoint, new_path
        path = res.new_path

        # Transform Setpoint from service client
        try:
            transform = tf_buffer.lookup_transform(robot_frame_id, 'map', rospy.Time())           
            transformed_setpoint = tf2_geometry_msgs.do_transform_point(setpoint, transform)       
            
            # Create Twist message from the transformed Setpoint
            msg.angular.z = limit_speed(4 * atan2(transformed_setpoint.point.y, transformed_setpoint.point.x), max_angular_velocity)
            msg.linear.x = 0 if abs(msg.angular.z) >= max_angular_velocity else limit_speed(0.5 * sqrt(transformed_setpoint.point.x ** 2 + transformed_setpoint.point.y ** 2), max_linear_velocity)

            # Publish Twist
            pub.publish(msg)
            rospy.loginfo(f'v: {round(msg.linear.x,2) } m/s | ω: {round(msg.angular.z,2)} rad/s')

            # Call service client again if the returned path is not empty and do stuff again
            rate.sleep()

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue
        
    # Send 0 control Twist to stop robot
    stop_moving()

    # Get new path from action server
    get_path()

def get_path():
    global goal_client

    # Get path from action server
    rospy.loginfo('========= Searching Path =========')
    goal_client.wait_for_server()
    goal_client.send_goal(None)
    goal_client.wait_for_result()
    res = goal_client.get_result()

    # Call move with path from action server
    if res.path.poses:
        rospy.loginfo('========= Moving =========')
        move(res.path)
    else:
        return 


if __name__ == "__main__":
    # Init node
    rospy.init_node('controller_main', disable_signals=True)
    listener = tf2_ros.TransformListener(tf_buffer)

    # Init publisher
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    # Init simple action client
    goal_client = actionlib.SimpleActionClient('get_next_goal', irob_assignment_1.msg.GetNextGoalAction)

    # Init service client
    control_client = rospy.ServiceProxy('get_setpoint', GetSetpoint)
    
    # Gracefully prepare for ROS shutdown (https://wiki.ros.org/rospy/Overview/Initialization%20and%20Shutdown)
    while not rospy.is_shutdown():
        rospy.on_shutdown(stop_moving)
        try:
            # Call get path
            result = get_path()
            rospy.loginfo('========= ALL DONE! =========')

            # Spin
            rospy.spin()
        except:
            rospy.signal_shutdown('Controller Stopped')

        


