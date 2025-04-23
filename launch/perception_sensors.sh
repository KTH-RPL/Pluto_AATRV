#!/bin/bash
source /opt/ros/noetic/setup.bash 
source /home/nuc/catkin_noetic/devel/setup.bash

# Prepare network config for Ouster Lidar
xterm -hold -geometry 80x36+0+0 -e "nmcli con down static" & sleep 0.1 #added for flush does not remove existint connection directly
xterm -geometry 80x36+0+0 -e "nmcli con down static" & sleep 0.1 #added for flush does not remove existint connection directly
xterm -geometry 80x36+0+0 -e "nmcli dev disconnect eno1" & sleep 0.1 #added for flush does not remove existint connection directly
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S ip addr flush dev eno1" & sleep 0.5
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S ip addr flush dev eno1" & sleep 0.5
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S ip addr show dev eno1" &  sleep 0.5 
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S ip addr add 192.168.1.50/24 dev eno1" &  sleep 0.5
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S ip link set eno1 up" &  sleep 0.5
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S ip addr show dev eno1" &  sleep 0.5
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S dnsmasq -C /dev/null -kd -F 10.5.5.50,10.5.5.100 -i eno1 --bind-dynamic" & sleep 5
xterm -geometry 80x36+0+0 -e "echo 'plutonuc' | sudo -S dnsmasq -C /dev/null -kd -F 192.168.1.50,192.168.1.100 -i eno1 --bind-dynamic" & sleep 5

killall xterm


# Launch the ouster
xterm -T "Launch Ouster" -geometry 80x15+0+	 -e "roslaunch ouster_ros sensor.launch " & sleep 2

# Launch the realsense camera
xterm -T "Launch D455" -geometry 80x15+0+350 -e "roslaunch realsense2_camera rs_camera_D455_0.launch" & sleep 2

xterm -T "Perception Rosbag" -geometry 80x15+0+ -e "rosbag record /rsD455_node0/color/image_raw /os_cloud_node/points /rsD455_node0/depth/image_rect_raw /vectornav/Odom /atrv/odom /reach/fix /robot_pose /clock" & sleep 2