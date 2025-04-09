#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from vectornav.msg import Ins

firstins = -500.0

def insval(msg):
    global firstins
    if firstins == -500:
        firstins = float(msg.yaw)
        if firstins < 0:
            firstins = 360 + firstins
if __name__ == '__main__':
    rospy.init_node('ins_suscriber')
    global firstins
    ins = rospy.Subscriber('/vectornav/INS', Ins, insval)
    firstINS = rospy.Publisher('/firstins',PoseStamped,queue_size=10)

    
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        global firsins
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = 0
        pose.pose.position.y = 0
        print(firstins)
        pose.pose.orientation.z = float(firstins)
        firstINS.publish(pose)
        firstINS.publish(pose)
        rate.sleep()