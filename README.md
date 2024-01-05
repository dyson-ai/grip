[![Build and Test (humble)](https://github.com/dyson-ai/grip/actions/workflows/build_and_test.yaml/badge.svg)](https://github.com/dyson-ai/grip/actions/workflows/build_and_test.yaml)

# Grip

Grip is a library for prototyping robot manipulation environments and applications. Grip contains tools for designing and prototyping robotic grasping applications, handling and acquiring data from simulated and real RGBD cameras, customising robot kinematics and dynamics, integration and deployment using ROS2. Grip was designed to be a lightweight python package to help prototyping, testing and deploying of robot grasping algorithms.

## Installation <a name="installation"></a>

### Standalone installation thorough pip:

2. Install

```
pip3 install git+https://github.com/dyson-ai/grip.git
```


1. Run simple example (pure-python)

```
python3 -m grip_examples.environments.ex01_pick_and_place
```
This will run a basic pick and place demo. You should see the robot picking and placing a an object on the table (see below).

![Simple python demo](./grip_assets/media/simple_demo.gif)

2. Run simple ROS2 example

```
python3 -m grip_examples.environments.ex04_ros_robot
```

## Development setup

Development with vscode and running more examples:
1. Clone this repository ```git clone https://github.com/dyson-ai/grip.git```

2. Run vscode: ```code grip```

3. F1 -> "dev containers: Reopen in Container"

4. Build repo with colcon: ```colcon build```

5. Run example launch file: ```ros2 launch grip ros_arm.launch.py```

You should be able to see a bullet simulation of a panda arm, and a pre-configured rviz displaying the robot model, state, and point cloud as below. This example node is located at `grip/grip_examples/robot/ex04_ros_robot.py`. For more examples see `grip/grip_examples` and `grip/launch`.

![Basic ROS2 demo](./grip_assets/media/basic_demo_2x.gif)


## Topics, services and actions of simulated ROS robot

The provided ROS interface should allow for ROS standard communication (e.g. if you want to connect the robot to moveit, or subscribe to its topics, etc)

### Robot and camera topics
   ```
   /hand_camera/color/undistorted/camera_info
   /hand_camera/color/undistorted/image_rect
   /hand_camera/depth_registered/points
   /hand_camera/depth_registered/undistorted/camera_info
   /hand_camera/depth_registered/undistorted/image_rect
   /joint_states
   /robot_description
   /tf
   /tf_static
   ```

### Joint trajectory action server

   ```
   /position_joint_trajectory_controller/follow_joint_trajectory
   ```

