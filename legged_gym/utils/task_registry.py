import os
from datetime import datetime
from typing import Tuple

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner, OnPolicyRunnerCTS

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

RUNNER_REGISTRY = {
    "OnPolicyRunner": OnPolicyRunner,
    "HIMOnPolicyRunner": HIMOnPolicyRunner,
    "OnPolicyRunnerCTS": OnPolicyRunnerCTS,
}


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        self.runner_classes = dict(RUNNER_REGISTRY)
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def register_runner(self, name: str, runner_class):
        self.runner_classes[name] = runner_class
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]

    def get_runner_class(self, train_cfg):
        runner_class_name = getattr(train_cfg, "runner_class_name", None)
        if runner_class_name is None and hasattr(train_cfg, "runner"):
            runner_class_name = getattr(train_cfg.runner, "runner_class_name", None)
        if hasattr(train_cfg, "runner"):
            policy_class_name = getattr(train_cfg.runner, "policy_class_name", "")
            algorithm_class_name = getattr(train_cfg.runner, "algorithm_class_name", "")
            if runner_class_name in [None, "OnPolicyRunner"]:
                if "HIM" in policy_class_name or "HIM" in algorithm_class_name:
                    runner_class_name = "HIMOnPolicyRunner"
                elif "CTS" in policy_class_name or "CTS" in algorithm_class_name:
                    runner_class_name = "OnPolicyRunnerCTS"
        if runner_class_name is None:
            runner_class_name = "OnPolicyRunner"
        if runner_class_name not in self.runner_classes:
            raise ValueError(f"Runner class '{runner_class_name}' was not registered")
        return self.runner_classes[runner_class_name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default", train_path = None) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        train_cfg_dict = class_to_dict(train_cfg)
        runner_class = self.get_runner_class(train_cfg)
        runner = runner_class(env, train_cfg_dict, log_dir, device=args.rl_device)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            # load previously trained model
            # resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
            resume_path = train_path
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
