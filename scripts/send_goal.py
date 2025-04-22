#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped

def publish_goal():
    rospy.init_node('goal_publisher')
    pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(1)
    
    goal = PoseStamped()
    goal.header.frame_id = "map"
    
    while not rospy.is_shutdown():
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = rospy.get_param('~x', 120.0)
        goal.pose.position.y = rospy.get_param('~y', 90.0)
        goal.pose.orientation.w = 1.0
        pub.publish(goal)
        rospy.loginfo_throttle(5, "Published goal: (%.2f, %.2f)", goal.pose.position.x, goal.pose.position.y)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_goal()
    except rospy.ROSInterruptException:
        pass