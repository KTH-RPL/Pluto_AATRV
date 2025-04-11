#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from vectornav.msg import Ins
import utm 

firstins = -500.0
start_x = None
start_y = None
start_lat = None
start_lon = None

def insval(msg):
    global firstins
    if firstins == -500:
        firstins = float(msg.yaw)
        if firstins < 0:
            firstins += 360

def gnss_callback(msg):
    global start_lat, start_lon, start_x, start_y
    if start_lat is None and msg.status.status >= 0:  # Only if GNSS fix is valid
        start_lat = msg.latitude
        start_lon = msg.longitude
        start_x, start_y = convert_gnss_to_utm(start_lat, start_lon)
        offset=[-333520.64199354, -6582420.00142414, 0.0]
        start_x = start_x + offset[0]
        start_y = start_y + offset[1]

def convert_gnss_to_utm(lat, lon):
    utm_coords = utm.from_latlon(lat, lon)
    x = utm_coords[0]
    y = utm_coords[1]
    return x, y

if __name__ == '__main__':
    rospy.init_node('ins_gnss_subscriber')

    ins = rospy.Subscriber('/vectornav/INS', Ins, insval)
    gnss = rospy.Subscriber('/reach/fix', NavSatFix, gnss_callback)

    firstINS = rospy.Publisher('/pose_offset', PoseStamped, queue_size=10)

    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        if firstins == -500 or start_x is None or start_y is None:
            rate.sleep()
            continue

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = start_x
        pose.pose.position.y = start_y
        pose.pose.orientation.z = float(firstins)

        firstINS.publish(pose)
        rate.sleep()
