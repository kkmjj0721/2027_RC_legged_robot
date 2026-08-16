from src.library.fsm.fsm import *
from library.rl_sdk.rl_sdk import *


class RLFSMStatePassive(FSMState):
    def __init__(self):
        super().__init__(self, "RLFSMStatePassive")

    def enter(self):
        print("已进入被动/安全状态")

    def run(self):
        pass

    def exit(self):
        print("已退出被动/安全状态")

