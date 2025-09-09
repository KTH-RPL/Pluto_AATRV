#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Pose, PoseArray

def publish_goal_list():
    rospy.init_node('goal_publisher')
    pub = rospy.Publisher('/goals_array', PoseArray, queue_size=10)
    rate = rospy.Rate(50)  

    
    goals = PoseArray()
    goals.header.frame_id = "map"

    goal_positions = [
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0,10.0),
    ]

    for pos in goal_positions:
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.orientation.w = 1.0  
        goals.poses.append(pose)

    while not rospy.is_shutdown():
        goals.header.stamp = rospy.Time.now()
        pub.publish(goals)
        rospy.loginfo("Published %d goals", len(goals.poses))
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_goal_list()
    except rospy.ROSInterruptException:
        pass
