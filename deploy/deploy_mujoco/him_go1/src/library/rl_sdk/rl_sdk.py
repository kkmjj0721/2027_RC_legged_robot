import os
import numpy as np
import yaml
import torch

from typing import Dict, Any
from fsm.fsm import FSM


class YamlParams:
    """YAML 参数解析包装器
    """
    def __init__(self, config_file: str = None, root_key: str = None):
        self.config_node: Dict[str, Any] = {}
        if config_file:
            self.load(config_file, root_key)

    def load(self, config_file: str, root_key: str = None):
        """读取yaml文件并转为dict"""
        if not os.path.exists(config_file):
            print(f"[ERROR] [YamlParams] Cannot find config file: {config_file}")
            return
            
        with open(config_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            # 如果指定了根节点（例如 'him_go1/himloco'），则只提取该节点下的数据
            if root_key and isinstance(data, dict):
                # 支持类似 "him_go1/himloco" 这种层级路径
                keys = root_key.split('/')
                node = data
                for k in keys:
                    node = node.get(k, {})
                self.config_node = node
            else:
                self.config_node = data if data else {}

    def get(self, key: str, default_value: Any = None) -> Any:
        return self.config_node.get(key, default_value)

class MotorCommand:
    def __init__(self, num_dof: int):
        self.q = np.zeros(num_dof, dtype=np.float32)
        self.dq = np.zeros(num_dof, dtype=np.float32)
        self.tau = np.zeros(num_dof, dtype=np.float32)
        self.kp = np.zeros(num_dof, dtype=np.float32)
        self.kd = np.zeros(num_dof, dtype=np.float32)

class MotorState:
    def __init__(self, num_dof: int):
        self.q = np.zeros(num_dof, dtype=np.float32)
        self.dq = np.zeros(num_dof, dtype=np.float32)
        self.ddq = np.zeros(num_dof, dtype=np.float32)
        self.tau_est = np.zeros(num_dof, dtype=np.float32)

class RobotState:
    def __init__(self):
        self.motor_state = MotorState()

class RobotCommand:
    def __init__(self):
        self.motor_command = MotorState()

class Observations:
    def __init__(self, num_dof: int, num_obs: int):
        self.commands = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_vec = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(num_dof, dtype=np.float32)
        self.dof_vel = np.zeros(num_dof, dtype=np.float32)
        self.last_actions = np.zeros(num_dof, dtype=np.float32)

        self.obs_buf = np.zeros(num_obs, dtype=np.float32)
    
    def update_obs_buf(self):
        self.obs_buf = np.concatenate([
            self.commands,
            self.ang_vel,
            self.gravity_vec,
            self.dof_pos,
            self.dof_vel,
            self.last_actions
        ]).astype(np.float32)

    def np2torch(self):
        self.update_obs_buf()
        self.torch_obs = torch.from_numpy(self.obs_buf).unsqueeze(0)
        return self.torch_obs

class input:
    pass


class Control:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0


class RL:
    def __init__(self):
        self.config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../policy/config.yaml"))
        self.params = YamlParams(self.config_path, root_key="him_go1/himloco")

        self.fsm = FSM()
        self.control = Control()

        self.__init_buffer()

    def __init_buffer(self):
        self.num_dof = self.params.get("num_dof")
        self.obs = Observations(self.num_dof)
        
        self.robot_command = RobotCommand()     # 准备发给机器人的指令
        self.start_state = RobotState()         # 用于状态机切换(如站立)时记录插值起点
        self.robot_state = RobotState()         # 机器人当前物理状态

    def Interpolate(self):
        """ 线性插值
            实现从趴下到起立 or 站立到趴下的功能
        """

    def ComputeOutput(self):
        """ 将神经网络输出的 Actions 转为实际的控制量
        """

    def ComputeObservation(self):
        """ 计算观测值
        """

