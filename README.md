[![Build and Test (humble)](https://github.com/dyson-ai/grip/actions/workflows/build_and_test.yaml/badge.svg)](https://github.com/dyson-ai/grip/actions/workflows/build_and_test.yaml)

# Grip

Grip is a prototyping toolbox for robot manipulation research powered by [Bullet3](https://github.com/bulletphysics/bullet3). It contains a collection of tools for creating robot manipulation environments, loading arbitrary robot arms and grippers through URDF, as well as handling and acquiring data from simulated and real RGBD cameras. It also supports ROS2 allowing created environments or applications to be easily integrated with the ROS2 ecosystem.

## Installation <a name="installation"></a>

### Pure python installation through pip

1. Install

```
pip3 install git+https://github.com/dyson-ai/grip.git
```

2. Run simple example (pure-python)

```
python3 -m grip_examples.environments.ex01_pick_and_place
```

This will run a basic pick and place demo. You should see the robot picking and placing a an object on the table (see below).

![Simple python demo](./grip_assets/media/simple_demo.gif)

3. Another example: loading a Franka Panda robot (pure-python)
      
```
python3 -m grip_examples.robot.ex02_robots
```
This will spawn a Franka Panda robot that can be controlled through joint sliders. See more examples at [grip_examples](https://github.com/dyson-ai/grip/tree/main/grip_examples).


## Development setup

Development with vscode and running more examples:

1. Clone this repository ```git clone https://github.com/dyson-ai/grip.git```

2. Run vscode: ```code grip```

3. F1 -> "dev containers: Reopen in Container"

4. Build repo with colcon: ```colcon build```


## ROS2 examples
A few basic ROS2 examples are showcased below.

### Basic ROS2 example

Launch file:
```
ros2 launch grip ros_arm.launch.py
```

You should be able to see a simulation of a panda arm, and a pre-configured rviz displaying the robot model, state, and point cloud as below. The main node for this example is located at `grip/grip_examples/robot/ex04_ros_robot.py`. For more examples see `grip/grip_examples` and `grip/launch`.

![Basic ROS2 demo](./grip_assets/media/basic_demo_2x.gif)

### Moveit demo

Launch file:
```
ros2 launch grip moveit_demo.launch.py
```
You should be able to see a simulation of a panda arm, and a pre-configured rviz displaying the robot model, state, and point cloud, and moveit planning scene as below. The main node for this example is located at `grip/grip_examples/robot/ex05_ros_robot_moveit.py`. For more examples see `grip/grip_examples` and `grip/launch`.

![Basic ROS2 demo](./grip_assets/media/moveit_demo.gif)


## Topics, services and actions of simulated ROS robot

The provided ROS2 interfaces should allow for ROS standard communication (e.g. if you want to connect the robot to moveit, or subscribe to its topics, tf, services, actions, sensors, etc)

Happy coding!