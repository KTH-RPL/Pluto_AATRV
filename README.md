# **PLUTO AATRV \- DD2414 Engineering Project in Robotics, Perception, and Learning**

This repository contains the software stack for the PLUTO robot, developed as part of the **DD2414 \- Engineering Project in Robotics, Perception, and Learning** at KTH Royal Institute of Technology (Spring 2025).

The original PLUTO robot platform can be found here: [PLUTO Robot (Original Git)](https://github.com/KTH-RPL/Pluto-ATRV)

## **Team**

* **Localization & Mapping:** Abhinav Srinivas Rajagopal <asraj@kth.se>
* **Path Planning:** Prasetyo Wibowo Laksono Sanjaya <pwlsa@kth.se>
* **Control:** Sarvesh Raj Kumar <sarvesh@kth.se>
* **Perception:** Sankeerth Reddy Prodduturi <srpr@kth.se>  
* **Behavior Tree:** Rajesh Kumar <rajeshk@kth.se>

### **Team Supervisor**

* Waqas Ali <waqasali@kth.se>

## **Running the Robot**

There are two primary methods for running the robot:

1. **Full Autonomous Stack:** Uses the Behavior Tree, global planner, and all controllers.  
2. **Manual Joystick Control:** For simple teleoperation.

### **Method 1: Full Autonomous Stack (with Behavior Tree)**

Follow these steps sequentially. It is recommended to run each command in a new terminal or terminal tab.

**1\. Configure LIDAR**

* Press Ctrl+1.  
* Wait for all terminal windows to open.  
* Press Ctrl+0.  
* This configures the Ouster LIDAR connection.

**2\. Launch Localization Node**

```
roslaunch ndt_localizer localization.launch
```

**3\. Launch Publishers Node**

```
roslaunch robot_controller main_publishers.launch
```

**4\. Launch Controller Node (for Behavior Tree)**

```
roslaunch robot_controller main_controller_bt.launch
```

**5\. Launch Global Planner Action Server**

```
roslaunch robot_controller global_planner.launch
```

**6\. Run Behavior Tree**

```
python3 /catkin_noetic_ws/src/Pluto_AATRV/control/scripts/behavior_tree_m2.py
```

### **Method 2: Manual Joystick Control**

To drive the robot manually using a joystick, run the following single command:

```
roslaunch pluto pluto.launch
```

## **Changelog After Demo 24 October 2025**
1. Rename test1.launch to main_publishers.launch
2. Rename test_controller_bt.launch to main_controller_bt.launch