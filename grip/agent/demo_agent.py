from grip.agent import Agent
from grip.motion import PBBasicPlanner

import numpy as np


class DemoAgent(Agent):
    """Scripted agent that runs a pre-defined pick and place policy"""

    def __init__(self, **kwargs):
        self.env = kwargs.get("env")
        self.reward = kwargs.get("reward", None)

        self.planner = PBBasicPlanner(self.env.get_entity("robot"))

        self.action_idx = -1
        self.steps = 30

        self.finished = False
        self.gripper_actions = []
        self.arm_actions = []
        self.random_offset = np.zeros(3)

        self.env.world.register_keyhandler(ord("r"), self.reset)

    def reset(self):
        self.env.reset()

        self.generate_actions()

    def generate_actions(self, **obs):
        self.gripper_actions = []
        self.arm_actions = []
        self.finished = False
        self.action_idx = 0

        self.random_offset = obs.get("offset", self.random_offset)

        _, ee_ori = self.env.get_entity("robot").ee_pose
        obj_pos, _ = self.env.get_entity("object").pose
        grasp_pos = obj_pos + np.array([0.0, 0.0, 0.05])
        grasp_ori = ee_ori

        place_pos = [0.49892282, 0.3771686, 0.16378036]
        place_ori = [-0.653, 0.271, 0.271, 0.653]

        self.compute_trajectory(grasp_pos, grasp_ori, place_pos, place_ori)

    def compute_trajectory(self, grasp_pos, grasp_ori, place_pos, place_ori):
        grasp_arm_actions, grasp_gripper_actions = self.get_grasp_actions(
            grasp_pos, grasp_ori
        )

        place_arm_actions, place_gripper_actions = self.get_placement_actions(
            place_pos, place_ori, initial_conf=grasp_arm_actions[-1]
        )

        self.arm_actions = grasp_arm_actions + place_arm_actions
        self.gripper_actions = grasp_gripper_actions + place_gripper_actions

    def observe(self, obs):
        self.finished = self.action_idx >= len(self.arm_actions)

        if self.action_idx < 0:
            self.generate_actions(**obs)

        # log.info("OBS: {}".format(obs))

        return self.finished

    def act(self):
        action = {}
        if not self.finished:
            action = {
                "joint_position_cmd": self.arm_actions[self.action_idx],
                "gripper_cmd": self.gripper_actions[self.action_idx],
            }

            self.action_idx += 1

        return action

    def get_grasp_actions(self, grasp_pos, grasp_ori):
        obstacles = [self.env.get_entity(name).id for name in ["tray", "plane"]]
        disabled_collisions = [self.env.get_entity("object").id]

        grasp_path = self.planner.plan_cartesian_goal(
            (grasp_pos, grasp_ori),
            obstacles=obstacles,
            disabled_collisions=disabled_collisions,
        )

        gripper_actions = np.zeros(len(grasp_path))
        gripper_actions = np.hstack([gripper_actions, -np.ones(self.steps) * 0.05])

        grasp_path = grasp_path + [grasp_path[-1]] * self.steps

        lift_pos = grasp_pos
        lift_pos[2] += 0.20
        lift_path = self.planner.plan_cartesian_goal(
            (lift_pos, grasp_ori), obstacles=[], initial_conf=grasp_path[-1]
        )

        gripper_actions = np.hstack([gripper_actions, -np.ones(len(lift_path)) * 0.05])

        grasp_path = grasp_path + lift_path

        return grasp_path, gripper_actions.tolist()

    def get_placement_actions(self, place_pos, place_ori, initial_conf=None):
        obstacles = [self.env.get_entity(name).id for name in ["tray", "plane"]]

        pre_place_path = self.planner.plan_joint_goal(
            self.env.panda.home_positions,
            obstacles=obstacles,
            initial_conf=initial_conf,
        )

        placement_path = self.planner.plan_cartesian_goal(
            (place_pos, place_ori), obstacles=[], initial_conf=pre_place_path[-1]
        )

        placement_path = pre_place_path + placement_path

        gripper_actions = np.zeros(len(placement_path))
        gripper_actions = np.hstack([gripper_actions, np.ones(self.steps) * 0.05])

        placement_path += [placement_path[-1]] * self.steps

        ee_pos, ee_ori = self.env.panda.forward_kinematics(placement_path[-1])
        ee_pos[0] = ee_pos[0] - 0.15
        release_path = self.planner.plan_cartesian_goal(
            (ee_pos, ee_ori), obstacles=[], initial_conf=placement_path[-1]
        )

        gripper_actions = np.hstack(
            [gripper_actions, np.zeros(len(release_path)) * 0.05]
        )

        placement_path += release_path

        gripper_actions = np.hstack([self.gripper_actions, gripper_actions])

        return placement_path, gripper_actions.tolist()
