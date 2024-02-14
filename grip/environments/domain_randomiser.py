from ..io import get_data_path, log
from ..robot import BulletRobot, BulletObject, BulletWorld
import pybullet as p
import numpy as np
from typing import Union, Tuple, List
from dataclasses import dataclass

base_dir = get_data_path()


class TextureRandomiser:
    """An object of this class is able to randomise the texture of BulletObject and BulletRobot instances"""

    def __init__(self, world: BulletWorld, texture_paths: List[str]):
        """
        Constructs a TextureRandomiser object.
        It can be used for to randomising the texture of BulletObject and BulletRobot instances.

        Args:
            world: a parent bullet world instance where entities to be randomised exist
            texture_paths: a list of texture paths to be chosen from

        """

        self.world = world
        self.textures_paths = texture_paths
        self.texture_count = len(texture_paths)

        self.world.texture_randomiser = self

        log.info(f"Loaded textures: {self.world.image_textures}")

    def randomise(self, entity: Union[BulletObject, BulletRobot]) -> None:
        """Sets texture of given entity to be a uniformly random one"""

        if isinstance(entity, BulletObject):
            tid = self.random_texture_id()
            p.changeVisualShape(
                entity.id, -1, textureUniqueId=tid, physicsClientId=self.world.id
            )
        elif isinstance(entity, BulletRobot):
            tid = self.random_texture_id()

            shape_data = p.getVisualShapeData(entity.id, physicsClientId=self.world.id)

            for shape in shape_data:
                p.changeVisualShape(
                    entity.id,
                    shape[1],
                    textureUniqueId=tid,
                    physicsClientId=self.world.id,
                )

    def random_texture_id(self) -> int:
        """
        gets random texture unique identifier (chosen uniformly)

        Returns:

            int: uniform randomly selected unique identifier of loaded texture (assuming texture already existing in memory)

        """

        tex_idx = np.random.randint(0, self.texture_count)

        tid = self.world.texture(self.textures_paths[tex_idx])

        return tid


@dataclass
class LightingParameters:
    """This class represents the basic phong illumination model parameters"""

    light_direction: np.ndarray = np.ones(3)
    light_colour: np.ndarray = np.ones(3)
    specular_coef: int = 1
    diffuse_coef: int = 1
    ambient_coef: int = 1


class LightingRandomiser:
    """Randomises phong illumination parameters"""

    DEFAULT_PARAMS = LightingParameters()

    def __init__(
        self,
        pos_range: List[Tuple[float, float]] = [[-1, 1], [-1, 1], [1, 2.5]],
        colour_range: List[Tuple[float, float]] = [[0, 1], [0, 1], [0, 1]],
        specular_range: Tuple[float, float] = [0, 1],
        diffuse_range: Tuple[float, float] = [0, 1],
        ambient_range: Tuple[float, float] = [0, 1],
    ):
        self._pos_range = pos_range
        self._colour_range = colour_range
        self._specular_range = specular_range
        self._diffuse_range = diffuse_range
        self._ambient_range = ambient_range

    def randomise(self) -> LightingParameters:
        """Returns randomised lighting parameters

        Returns:
            (LightingParameters): randomised phong illumination model parameters
        """

        light_direction = [
            np.random.uniform(self._pos_range[0][0], self._pos_range[0][1]),
            np.random.uniform(self._pos_range[1][0], self._pos_range[1][1]),
            np.random.uniform(self._pos_range[2][0], self._pos_range[2][1]),
        ]
        light_color = [
            max(
                0,
                min(
                    1,
                    np.random.uniform(
                        self._colour_range[0][0], self._colour_range[0][1]
                    ),
                ),
            ),
            max(
                0,
                min(
                    1,
                    np.random.uniform(
                        self._colour_range[1][0], self._colour_range[1][1]
                    ),
                ),
            ),
            max(
                0,
                min(
                    1,
                    np.random.uniform(
                        self._colour_range[2][0], self._colour_range[2][1]
                    ),
                ),
            ),
        ]
        specular_coeff = max(
            0,
            min(1, np.random.uniform(self._specular_range[0], self._specular_range[1])),
        )
        diffuse_coeff = max(
            0, min(1, np.random.uniform(self._diffuse_range[0], self._diffuse_range[1]))
        )
        ambient_coeff = max(
            0, min(1, np.random.uniform(self._ambient_range[0], self._ambient_range[1]))
        )
        return LightingParameters(
            light_direction, light_color, specular_coeff, diffuse_coeff, ambient_coeff
        )
