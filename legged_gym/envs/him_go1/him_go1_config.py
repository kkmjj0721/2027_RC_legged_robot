from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO



class HimGo1cfg(LeggedRobotCfg):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_one_step_observations = 45
        history_lenth = 6
        num_observations = num_one_step_observations * history_lenth
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_actions = 12
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane' # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True

    class init_state( LeggedRobotCfg.init_state ):
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

    class control( LeggedRobotCfg.control ):
            # PD Drive parameters:
            control_type = 'P'
            stiffness = {'joint': 20.}  # [N*m/rad]
            damping = {'joint': 0.5}     # [N*m*s/rad]
            # action scale: target angle = actionScale * action + defaultAngle
            action_scale = 0.25
            # decimation: Number of control action updates @ sim DT per policy DT
            decimation = 4

    class asset( LeggedRobotCfg.asset ):
            file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
            name = "a1"
            foot_name = "foot"
            penalize_contacts_on = ["thigh", "calf"]
            terminate_after_contacts_on = ["base"]
            self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class domain_rand( LeggedRobotCfg.domain_rand ):
        pass

    class rewards( LeggedRobotCfg.rewards ):
        pass

    class normalization( LeggedRobotCfg.normalization ):
        contact_force_range = [0.0, 50.0]
        class obs_scales( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class HimGo1cfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = "HIMOnPolicyRunner"

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        pass