try:
    from pybullet_planning import (
        plan_joint_motion,
        get_custom_limits,
    )
    from pybullet_planning.utils import set_client
    from pybullet_planning.interfaces.env_manager.pose_transformation import all_between
except Exception as e:
    print(f"Failed to import pybullet_planning: {e}")

import numpy as np
from .path import Path


class PBBasicPlanner:
    """
    This is class supporting basic motion planning in cartesian and configuration space
    with obstacle avoidance on top of the pybullet_planning package, which is a suite of
    utility functions for robot motion planning, manipulation planning and task and
    motion planning package tailored for PyBullet simulation.


    Reference:
        Caelan Reed Garrett. PyBullet Planning. https://pypi.org/project/pybullet-planning/. 2018.

    Args:
        robot (grip.robot.BulletRobot): An instance of a bullet robot
    """

    def __init__(self, robot):
        self.robot = robot

    def format_joint_limits(self):
        """
        Parsing the joint limits for all joint of the target robot
        Slightly postprocessing it by adding a bit buffer to cope with reading error
        from PyBullet (Less optimal solution)

        Returns:
            custom_limits: [dict]
                        dictionary with key as joint name, value as tuple (lower_limit, upper_limit)
            lower_limits: [list]
                        list of lower limit for all joints in order
            upper_limits: [list]
                        list of upper limit for all joints in order
        """
        CLIENT = self.robot.phys_id
        set_client(CLIENT)

        lower_limits, upper_limits = get_custom_limits(
            self.robot.id, self.robot.joint_ids
        )
        # slightly enlarge the limits for pybullet reading error
        # for example0.04-->0.04000000xx
        custom_limits = {}
        lower_limits = list(lower_limits)
        upper_limits = list(upper_limits)
        for i, (joint, ll, ul) in enumerate(
            zip(self.robot.joint_ids, lower_limits, upper_limits)
        ):
            lower_limits[i] = ll - 0.01
            upper_limits[i] = ul + 0.01
            custom_limits[joint] = lower_limits[i], upper_limits[i]
        return custom_limits, lower_limits, upper_limits

    def plan_cartesian_goal(
        self,
        target_pose,
        obstacles=[],
        disabled_collisions=[],
        initial_conf=None,
        **kwargs,
    ):
        """
        Planing with BiRRT planner from initial configuration to final cartesian
        pose. The IK solver used here is from PyBullet which is aversion of Samuel
        Buss Inverse Kinematics library which also supports Null space control if
        lowerLimits, upperLimits, jointRanges, restPoses are specified.

        Args:
            target_pose: [list]
                    [[postion as list of 3 floats], [quaternion as list of 4 floats]
            obstacles: [list]
                    list of object indexes for target objects
            disabled_collisions: [list]
                    list of object indexes for object which temperoarily disable
                    collision checking, for example during grasping, the collision
                    checking on object or surface will be disabled
            initial_conf: [list]
                    joint angles for initial configuration in order. If None will compute
                    from the current configuration.

        Returns:
            path: [list]
                A trajectory from initial config to target pose as a list
                of joint configurations
        """
        # TODO: add disabled_collisions feature?
        CLIENT = self.robot.phys_id
        set_client(CLIENT)

        full_ret = kwargs.get("full_ret", False)
        kwargs.get("total_time", 5)
        dt = kwargs.get("dt", 0.001)

        original_conf = self.robot.angles
        if initial_conf is None:
            initial_conf = original_conf

        self.robot.reset_angles(initial_conf)
        # set_joint_positions(self.robot.id,
        #                    self.robot.joint_ids,
        #                    initial_conf)

        custom_limits, lower_limits, upper_limits = self.format_joint_limits()

        target_conf = self.robot.inverse_kinematics(*target_pose)

        print("Target conf: ", target_conf)
        path = plan_joint_motion(
            self.robot.id,
            self.robot.joint_ids,
            target_conf,
            obstacles=obstacles,
            self_collisions=False,
            custom_limits=custom_limits,
            max_distance=1e-3,
            iterations=1000,
            restarts=100,
            diagnosis=False,
        )

        ## Needs to reset robot state to where the were since pybullet_planner sets the joints of self.robot during planning
        self.robot.reset_angles(original_conf)

        # motion planner
        if (
            path is None
            or any(map(np.isnan, target_conf))
            or (not all_between(lower_limits, target_conf, upper_limits))
        ):
            target_conf = None
            return None
        else:
            if full_ret:
                gpath = Path(points=np.array(path), joint_names=self.robot.joint_names)

                return gpath
            else:
                return path

    def plan_joint_goal(self, target_conf, obstacles=[], initial_conf=None, **kwargs):
        """
        Planing with BiRRT planner from initial configuration to final configuration
        in joint space

        Args:
            target_pose: [list]
                    [[postion as list of 3 floats], [quaternion as list of 4 floats]
            obstacles: [list]
                    list of object indexes for target objects
            initial_conf: [list]
                    joint angles for initial configuration in order. If None will compute
                    from the current configuration.

        Returns:
            path: [list]
                A trajectory from initial config to target pose as a list
                of joint configurations
        """
        CLIENT = self.robot.phys_id
        set_client(CLIENT)

        full_ret = kwargs.get("full_ret", False)
        kwargs.get("total_time", 5)
        dt = kwargs.get("dt", 0.001)

        original_conf = self.robot.angles
        if initial_conf is None:
            initial_conf = original_conf

        self.robot.reset_angles(initial_conf)
        # set_joint_positions(self.robot.id,
        #                    self.robot.joint_ids,
        #                    initial_conf)

        custom_limits, lower_limits, upper_limits = self.format_joint_limits()

        path = plan_joint_motion(
            self.robot.id,
            self.robot.joint_ids,
            target_conf,
            obstacles=obstacles,
            self_collisions=False,
            custom_limits=custom_limits,
            max_distance=1e-2,
            iterations=1000,
            restarts=100,
            diagnosis=False,
        )

        ## Needs to reset robot state to where the were since pybullet_planner sets the joints of self.robot during planning
        self.robot.reset_angles(original_conf)

        # motion planner
        if any(map(np.isnan, target_conf)) or (
            not all_between(lower_limits, target_conf, upper_limits)
        ):
            target_conf = None
            return None
        else:
            if full_ret:
                gpath = Path(points=np.array(path), joint_names=self.robot.joint_names)

                error = np.linalg.norm(initial_conf - target_conf)
                if error > 1e-3:
                    return gpath.retime(self.robot, dt)
                else:
                    return gpath
            else:
                return path
