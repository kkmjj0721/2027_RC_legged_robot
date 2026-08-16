import os
import numpy as np
import yaml

from typing import Dict, Any
from fsm.fsm import FSM


class YamlParams:
    """安全读取 YAML 字典参数的封装类"""
    def __init__(self):
        self.config_node: Dict[str, Any] = {}

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.config_node.get(key, default_value)

class RobotCommand:
    pass

class Observations:
    def __init__(self, num_dof: int, num_obs: int):
        commands = np.zeros()
    
    def np2torch(self):
        pass

class RobotState:
    pass




class RL:
    def __init__(self):
        self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../policy"))
        self.params = YamlParams()              # 全局配置参数

        # 解析参数
        self._parse_cfg(self.config_path)

        self.obs = Observations()
        
        self.robot_command = RobotCommand()     # 准备发给机器人的指令

        self.start_state = RobotState()         # 用于状态机切换(如站立)时记录插值起点
        self.robot_state = RobotState()         # 机器人当前物理状态

        self.fsm = FSM()