from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot
from .legged_gym_go1.legged_gym_go1_config import GO1RoughCfg, GO1RoughCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register("legged_gym_go1",LeggedRobot, GO1RoughCfg, GO1RoughCfgPPO)


