#!/usr/bin/env python3

import os, random
import numpy as np


import pybullet as p

import grip
from .base_env import Env
from .. import robot as g
from ..sensors import RGBDCamera
from ..io import get_data_path
import pybullet_data as pd

base_dir = get_data_path()

TABLE_URDF = os.path.join(base_dir, "urdf/workcell/table/model.urdf")
PLANE_URDF = os.path.join(pd.getDataPath(), "plane.urdf")


VISION_META = {
    "view_dist": 1.6,
    "view_pitch": -128.8,
    "view_yaw": -176.6,
    "view_target_pos": [0.14, 0.24, 0.37],
    "downsample_height": 64,
    "downsample_width": 64,
}


class TemplateEnvironment(Env):
    def __init__(self, **kwargs):
        seed = kwargs.get("seed", 77)

        random.seed(seed)
        np.random.seed(seed)

        self.gripper_type = kwargs.get("gripper", "phase4")
        self.gripper_max_force = kwargs.get("gripper_max_force", 15)

        self.time_step = kwargs.get("time_step", 1.0 / 240.0)
        self.sim_steps = kwargs.get("sim_steps", 10)
        self._object_type = kwargs.get("object_type", ["ycb"])
        self._table_base_pos = kwargs.get("table_base_pos", [0, 0, 0])
        self._render_mode = kwargs.get("render_mode", "GUI")
        self._vision_params = kwargs.get("vision_params", VISION_META)

        self._enable_tex_randomiser = kwargs.get("enable_texture_randomiser", False)

        self.camera_mode = kwargs.get("camera_mode", "on_shoulder")

        self._plane_urdf = PLANE_URDF
        self._table_urdf = TABLE_URDF

        self.entities = dict()
        self.sensors = dict()

        self.world = g.BulletWorld(
            phys_opt=self._render_mode, real_time_sim=False, timestep=self.time_step
        )

        if self._enable_tex_randomiser:
            self.textures_paths = grip.io.file_list(
                grip.io.get_package_path("grip_assets", "texture"),
                extension_filter="jpg",
            )
            self.texture_randomiser = grip.environments.TextureRandomiser(
                self.world, self.textures_paths
            )

        self.rf = None

        self.init()

        self.world.register_keyhandler(ord("r"), self.reset)

    def init(self):
        if self._render_mode == "GUI":
            self.world.setDebugCamera(**self._vision_params)

        # spawn plane and table
        self.plane = g.BulletObject(
            urdf_file=self._plane_urdf, fixed=True, world=self.world
        )

        self.table = g.BulletObject(
            urdf_file=self._table_urdf,
            fixed=True,
            position=self._table_base_pos,
            world=self.world,
        )

        table_p, _ = self.table.pose

        self._table_height = table_p[2]

        # spawn robot
        self.panda = g.BulletPandaArmGripper(
            gripper_type=self.gripper_type,
            gripper_max_force=self.gripper_max_force,
            position=table_p,
            phys_id=self.world.id,
            texture_randomiser=self.world.texture_randomiser,
        )

        self.shoulder_camera = RGBDCamera(
            cid=self.world.id,
            name="ShoulderCamera",
            camera_far=10.0,
            proj_from_intrinsics=False,
        )
        self.shoulder_camera.set_camera_params()
        self.wrist_camera = RGBDCamera(
            cid=self.world.id,
            name="WristCamera",
            anchor_robot=self.panda,
            anchor_link="panda_hand",
        )
        if self.camera_mode in ["on_shoulder", "no_camera"]:
            self.shoulder_camera.draw_frustum(duration=0)

        self.add("plane", self.plane)
        self.add("robot", self.panda)
        self.add("table", self.table)

        if self.camera_mode in ["on_shoulder", "all", "no_camera"]:
            self.add("shoulder_camera", self.shoulder_camera)

        if self.camera_mode in ["on_wrist", "all"]:
            self.add("wrist_camera", self.wrist_camera)

    def get_entity(self, entity_name):
        return self.entities[entity_name]

    def add(self, name, entity):
        if isinstance(entity, grip.robot.Entity):
            self.entities[name] = entity

        if isinstance(entity, grip.sensors.Sensor):
            self.sensors[name] = entity

    def remove(self, name):
        entity = self.entities.pop(name)

        self.world.remove_entity(entity.id)

    def reset(self):
        self.world.restore_state()
        self.panda.reset()

        if self._enable_tex_randomiser:
            self.randomise_textures()

        return self.obs()

    def render(self):
        return self.shoulder_camera.obs()

    def step(self, action):
        if action is None:
            action = {}

        joint_position_cmd = action.get("joint_position_cmd", None)
        gripper_cmd = action.get("gripper_cmd", None)

        if joint_position_cmd is not None:
            self.panda.set_angles(joint_position_cmd)

        if gripper_cmd is not None:
            if isinstance(gripper_cmd, str) and gripper_cmd in ["open", "close"]:
                self.panda.gripper.control_fingers(mode=gripper_cmd)
            else:
                self.panda.gripper.apply_gripper_delta_action(gripper_cmd)

        for _ in range(self.sim_steps):
            self.world.step()

        obs = self.obs()

        reward = 0.0 if self.rf is None else self.rf.reward(self, obs)
        done = False
        info = {}

        return obs, reward, done, info

    def set_reward(self, reward):
        self.rf = reward

    def obs(self):
        obs = dict([(name, e.state()) for (name, e) in self.entities.items()])

        if self.camera_mode != "none":
            for name, camera in self.sensors.items():
                rgb, depth, seg = camera.obs()

                obs[name] = {
                    "rgb": rgb,
                    "depth": depth,
                    "seg": seg,
                    "pose": camera.pose,
                }

        return obs

    def sample(self):
        return {
            "joint_position_cmd": np.random.randn(7),
            "gripper_cmd": np.random.randn(),
        }

    def randomise_textures(self):
        for _, e in self.entities.items():
            e.random_texture()


if __name__ == "__main__":
    env = TemplateEnvironment()

    while p.isConnected():
        env.step(None)
