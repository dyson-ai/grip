from .version import __version__

# from gym.envs.registration import register

# register(
#     id='panda-v0',
#     entry_point='grip.environments:PandaGraspingEnv',
# )

from time import sleep

from . import agent
from . import io
from . import environments
from . import math
from . import motion
from . import perception
from . import robot
from . import app
from . import sensors


# explicitly list imports in __all__
__all__ = [
    "io",
    "perception",
    "environments",
    "robot",
    "agent",
    "sensors",
    "motion",
    "math",
    "app",
]
