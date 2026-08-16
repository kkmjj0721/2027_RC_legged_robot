from legged_gym.envs.base.him_legged_robot_config import HimLeggedRobotCfg, HimLeggedRobotCfgPPO


class HimRecoveryGo1LeggedRobotCfg( HimLeggedRobotCfg ):
    class env( HimLeggedRobotCfg.env ):
        num_envs = 4096
        num_one_step_observations = 49 # command vx vy yaw hight stand handstand
        num_observations = num_one_step_observations
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 12
        env_spacing = 3. 
        send_timeouts = True 
        episode_length_s = 20 

    class commands(HimLeggedRobotCfg.commands):
        # 0:vx, 1:vy, 2:yaw rate, 3:height target,
        # 4:stand flag, 5:handstand flag
        num_commands = 6
        heading_command = False
        resampling_time = 5.0
        # [implicit locomotion, stand, handstand]; only the latter two
        # values are stored in commands, as binary flags at indices 4 and 5.
        mode_probs = [0.70, 0.20, 0.10]

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

    