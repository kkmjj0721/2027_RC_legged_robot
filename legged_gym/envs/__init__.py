import os

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .base.legged_robot import LeggedRobot
from .legged_gym_go1.legged_gym_go1_config import GO1RoughCfg, GO1RoughCfgPPO

from .base.him_legged_robot import HimLeggedRobot
from .him_go1.him_go1_config import HimGO1RoughCfg, HimGo1RoughCfgPPO

from .him_recovery_high.him_recovery_high_config import HimRecoveryGo1LeggedRobotCfg, HimRecoveryGo1LeggedRobotCfgPPO
from .him_recovery_high.him_recovery_high import HimRecoveryGo1LeggedRobot


from legged_gym.utils.task_registry import task_registry


task_registry.register("legged_gym_go1",LeggedRobot, GO1RoughCfg, GO1RoughCfgPPO)
task_registry.register("him_go1",HimLeggedRobot, HimGO1RoughCfg, HimGo1RoughCfgPPO)
task_registry.register("him_recovery_high", HimRecoveryGo1LeggedRobot, HimRecoveryGo1LeggedRobotCfg, HimRecoveryGo1LeggedRobotCfgPPO)
                                                              