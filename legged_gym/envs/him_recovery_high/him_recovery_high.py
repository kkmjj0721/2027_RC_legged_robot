import torch
from isaacgym import gymtorch
from legged_gym.envs.base.him_legged_robot import HimLeggedRobot
from legged_gym.utils.math import torch_rand_float


class HimRecoveryGo1LeggedRobot(HimLeggedRobot):
    """四任务单模型 locomotion + recovery + platform high + stand + handstand
    """
    def _init_buffers(self):
        super()._init_buffers()

        # 缩放
        self.commands_scale = torch.tensor(
            [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
             1.0, 1.0, 1.0], device=self.device, requires_grad=False)

        # 模式 flag
        self.stand_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.handstand_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.high_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 恢复成功标记（每回合每 env 一次）
        self.already_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 平滑成功门控
        self.recent_max_agitation = torch.zeros(self.num_envs, device=self.device)

        # 平台翻越标记
        self.wall_crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 目标重力：默认正立 [0,0,-1]（用于 orientation 奖励）
        self.target_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        # 基座高度缓冲：基类无此属性，需自行维护；每次 _post_physics_step_callback 刷新
        self.base_height = torch.zeros(self.num_envs, device=self.device)

        # 模式切换触发重置标记：_post_physics_step_callback 置位，check_termination 消费
        self.mode_changed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _compute_current_observation_parts(self, add_actor_noise=True):
        base_ang_vel_obs, projected_gravity_obs = self._get_imu_obs()
        dof_pos_obs, dof_vel_obs = self._get_motor_obs()

        base_height = self.base_height.unsqueeze(1)

        clean_actor_obs = torch.cat((
            self.commands[:, :6] * self.commands_scale,  # 6 命令
            base_ang_vel_obs,
            projected_gravity_obs,
            dof_pos_obs,
            dof_vel_obs,
            self.actions,
            base_height,  # 新增 1 维
        ), dim=-1)  # 49 维

        actor_obs = clean_actor_obs
        if add_actor_noise and self.add_noise:
            actor_obs = actor_obs + (2 * torch.rand_like(actor_obs) - 1) * self.noise_scale_vec[:self.num_one_step_obs]

        critic_obs = torch.cat((
            clean_actor_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.rand_push_force,  # 新增 3 维推动力（特权信息，与 num_one_step_privileged_obs 的 +3 对齐）
        ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1, 1.) * self.obs_scales.height_measurements
            critic_obs = torch.cat((critic_obs, heights), dim=-1)

        return actor_obs, critic_obs

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(49 + 187, device=self.device)
        self.add_noise = cfg.noise.add_noise
        ns = cfg.noise.noise_scales
        lvl = cfg.noise.noise_level
        noise_vec[0:6] = 0.
        noise_vec[6:9] = ns.ang_vel * lvl * self.obs_scales.ang_vel
        noise_vec[9:12] = ns.gravity * lvl
        noise_vec[12:24] = ns.dof_pos * lvl * self.obs_scales.dof_pos
        noise_vec[24:36] = ns.dof_vel * lvl * self.obs_scales.dof_vel
        noise_vec[36:48] = 0.
        noise_vec[48:49] = ns.height_measurements * lvl * 0.2
        noise_vec[49:236] = ns.height_measurements * lvl * self.obs_scales.height_measurements
        return noise_vec

    def _resample_mode(self, env_ids):
        """ 模式采样
            locomotion/stand/handstand 写入 commands[:,4:6] 与布尔缓冲
            必须在 _reset_root_states 之前调用，reset 姿态才能读到正确模式
        """
        mode_probs = self.cfg.commands.mode_probs
        r = torch.rand(len(env_ids), device=self.device)
        stand_mask = (r >= mode_probs[0]) & (r < mode_probs[0] + mode_probs[1])
        hand_mask = r >= (mode_probs[0] + mode_probs[1])

        self.commands[env_ids, 4] = stand_mask.float()
        self.commands[env_ids, 5] = hand_mask.float()
        self.stand_mode[env_ids] = stand_mask
        self.handstand_mode[env_ids] = hand_mask
        self._update_target_gravity()

    def _resample_sub_mode(self, env_ids):
        """ 仅在 reset 时调用：在 locomotion 内采样隐式子模式（recovery/high/normal）
            顶层模式已由 _resample_mode 置好，这里只填 recovery_mode/high_mode
        """ 
        loco_mask = ~self.stand_mode[env_ids] & ~self.handstand_mode[env_ids]
        r = torch.rand(len(env_ids), device=self.device)
        recovery_ratio = self.cfg.commands.recovery_ratio
        platform_ratio = self.cfg.commands.platform_ratio
        self.recovery_mode[env_ids] = loco_mask & (r < recovery_ratio)
        self.high_mode[env_ids] = loco_mask & (r >= recovery_ratio) & \
            (r < recovery_ratio + platform_ratio)

    def _resample_commands(self, env_ids):
        """ 命令采样
            采样速度/角速度/高度指令；stand/handstand 模式速度归零。模式 flag 由 _resample_mode 负责
        """
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["target_height"][0], self.command_ranges["target_height"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)

        # stand/handstand 模式：速度指令清零
        loco_mask = (~self.stand_mode[env_ids] & ~self.handstand_mode[env_ids]).float()
        self.commands[env_ids, 0] *= loco_mask
        self.commands[env_ids, 1] *= loco_mask
        self.commands[env_ids, 2] *= loco_mask

        # 与基类一致：小速度指令清零（避免原地抖动）
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def reset_idx(self, env_ids):
        """ 重置
            先采模式，再 reset
        """
        if len(env_ids) == 0:
            return

        # 先采样顶层模式 + 隐式子模式，_reset_dofs/_reset_root_states 才能读到
        self._resample_mode(env_ids)
        self._resample_sub_mode(env_ids)
        super().reset_idx(env_ids)

        # 模式切换标记在 reset 后清零
        self.mode_changed_buf[env_ids] = False

        # reset 改变了 root_states，需刷新 base_height，否则本步 compute_observations 读到旧值
        self.base_height[env_ids] = self._get_base_heights()[env_ids]

    