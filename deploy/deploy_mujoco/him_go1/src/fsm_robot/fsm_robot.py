from src.library.fsm.fsm import FSMState, FSMMode, FSM, FSMFactory, FSMManager, register_fsm_factory


class RLFSMStatePassive(FSMState):
    def __init__(self):
        super().__init__(self, "RLFSMStatePassive")

    def enter(self):
        print("已进入被动/安全状态")

    def run(self):
        pass

    def exit(self):
        print("已退出被动/安全状态")

