#!/usr/bin/env python3
import grip
from grip.environments import PickAndPlaceEnvironment, Env
from grip.agent import DemoAgent
from grip.io import log
import numpy as np
from typing import Dict


class DemoReward(grip.environments.Reward):
    """An example reward function.

    This example computes a 1.0 reward if plate is grasped and placed successfully
    """

    def __init__(self, obj_pos: np.ndarray, obj_ori: np.ndarray):
        """

        Args:
            (obj_pos): initial plate position
            (obj_ori): initial plate orientation
        """

        self.obj_pos0 = obj_pos
        self.obj_ori0 = obj_ori

        self.success = False
        self.was_grasped = False
        self.was_placed = False

    def reward(self, env: Env, obs: Dict) -> float:
        """Computes instantaneous reward given current observation and environment

        Args:
            (env): current task environment
            (obs): current observation from envrironment
        Returns:
            (float): 1.0 reward if plate is grasped and placed successfully, 0.0 otherwise
        """

        height_diff = obs["object"]["pose"][0][2] - self.obj_pos0[2]

        obj_entity = env.get_entity("object")
        robot_entity = env.get_entity("robot")
        table_entity = env.get_entity("table")

        has_grasp_contacts = obj_entity.is_contacting(robot_entity)
        has_table_contacts = obj_entity.is_contacting(table_entity)
        log.info("Has contacts with table {}".format(has_table_contacts))

        if not self.was_grasped and has_grasp_contacts:
            robot_entity.attach_object(robot_entity.tip_link, obj_entity)

        if height_diff < 0.1 and self.was_grasped:
            robot_entity.detach_object(obj_entity)

        log.info("Has contacts with plate {}".format(has_grasp_contacts))

        is_grasped = height_diff >= 0.2 and has_grasp_contacts
        is_placed = not has_grasp_contacts and has_table_contacts

        self.was_grasped = self.was_grasped or is_grasped
        self.was_placed = self.was_placed or is_placed
        log.info(
            "was_grasped {} was_placed {}".format(self.was_grasped, self.was_placed)
        )

        return float(self.was_grasped and self.was_placed)


class EnvRunner:
    """example environment runner: constructs environment and agent and can run one interaction step."""

    def __init__(self, *args, **kwargs):
        self.env = PickAndPlaceEnvironment(seed=None, **kwargs)
        self.agent = DemoAgent(env=self.env)

        self.obs = None
        self.init()

        self.env.world.register_keyhandler(ord("r"), self.init)

    def init(self) -> None:
        """Initialises environment
        Probes initial state, initialises and sets reward
        """

        self.env.step(None)

        self.obs = self.env.reset()
        self.agent.reset()

        obj_pos, obj_ori = self.obs["object"]["pose"]
        self.env.set_reward(DemoReward(obj_pos, obj_ori))

    def run(self) -> None:
        """Runs one agent-environment iteration cycle:
        1. Agent gets an observation
        2. Agent computes an action based on observation
        3. Environment evolves one time step given the agent's action
        """
        self.agent.observe(self.obs)

        action = self.agent.act()

        self.obs, reward, _, _ = self.env.step(action)

        grip.io.log.info("Reward: {}".format(reward))

        grip.sleep(self.env.world.timestep)

    def spin(self):
        """Spins and iterates while connected"""

        while self.env.world.is_connected():
            self.run()


if __name__ == "__main__":
    env_runner = EnvRunner(gripper_max_force=50, enable_texture_randomiser=True)

    env_runner.spin()
