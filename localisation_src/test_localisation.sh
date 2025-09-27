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

# Input bag name as argument to this script
BAG_NAME="$1"

if [ -z "$BAG_NAME" ]; then
    echo "Usage: $0 <bag_name>"
    exit 1
fi

# Launch NDT localization first
xterm -T "Launch NDT Localizer" -geometry 80x24+500+0 -e "roslaunch ndt_localizer localisation.launch bag_name:=$BAG_NAME" &

# Wait 7 seconds
sleep 7

# Launch Ouster LiDAR after localization has started
xterm -T "Launch Ouster" -geometry 80x24+0+0 -e "roslaunch ouster_ros sensor.launch" &
