from legged_gym.envs.base.him_legged_robot_config import HimLeggedRobotCfg, HimLeggedRobotCfgPPO


class HimRecoveryGo1LeggedRobotCfg( HimLeggedRobotCfg ):
    class env(HimLeggedRobotCfg.env):
        num_envs = 2048
        num_one_step_observations = 49  # commands6 + ang_vel3 + gravity3 + dof_pos12 + dof_vel12 + action12 + base_height1
        num_observations = num_one_step_observations  
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187  # + lin_vel3 + push_force3 + heights187
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20

    class commands(HimLeggedRobotCfg.commands):
        # [0]vx [1]vy [2]yaw_rate [3]height [4]stand_flag [5]handstand_flag
        num_commands = 6
        heading_command = False
        resampling_time = 5.0
        # 顶层模式 [locomotion, stand, handstand]
        mode_probs = [0.70, 0.20, 0.10]
        # locomotion 内隐式子模式占比（其余为正常平地行走）
        recovery_ratio = 0.3   # 摔倒恢复回合占比
        platform_ratio = 0.3   # 高台攀爬回合占比
        
        class ranges(HimLeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            target_height = [0.24, 0.40]

    class init_state( HimLeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class control( HimLeggedRobotCfg.control ):
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 0.5

    class rewards(HimLeggedRobotCfg.rewards):
        class scales(HimLeggedRobotCfg.rewards.scales):
            action_rate = -0.02
            torques = -1e-5
            dof_vel = -1e-4
            dof_acc = -2.5e-7
            collision = 0.0

            # ---- locomotion ----
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -0.5
            ang_vel_xy = -0.05
            orientation = -0.2
            feet_air_time = 0.3
            base_height = 0.0
            termination = -10.0

            # ---- recovery 子模式 ----
            upright_linear = 3.0
            stand_height = 1.5
            joint_to_default = 1.5
            recovery_success = 1.0
            joint_to_default_penalty = -0.5

            # ---- high platform 子模式 ----
            platform_progress = 0.5
            platform_mount = 50.0
            wall_crossing = 100.0

            # ---- stand / handstand 竖直姿态 ----
            handstand_orientation = -1.0
            handstand_feet_on_air = 0.4
            handstand_feet_height_exp = 5.0


    class asset(HimLeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        name = "go1"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]  # 接触失败（recovery 子模式在 check_termination 中豁免）
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1  # 1 表示关闭自碰撞
        flip_visual_attachments = True


class HimRecoveryGo1LeggedRobotCfgPPO(HimLeggedRobotCfgPPO):
    class runner(HimLeggedRobotCfgPPO.runner):
        experiment_name = 'him_recovery_high'

