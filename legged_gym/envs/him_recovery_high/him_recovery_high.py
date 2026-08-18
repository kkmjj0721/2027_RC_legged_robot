import torch
from isaacgym import gymtorch
from legged_gym.envs.base.him_legged_robot import HimLeggedRobot
from legged_gym.utils.math import torch_rand_float_tensor


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
        self.commands[env_ids, 0] = torch_rand_float_tensor(
            self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float_tensor(
            self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float_tensor(
            self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = torch_rand_float_tensor(
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

    def _post_physics_step_callback(self):
        """每 resampling_time 重采样模式；模式切换触发重置（新姿态从头开始）。"""

        # 刷新基座高度（plane 地形下 = root_states[:,2]）
        self.base_height = self._get_base_heights()

        resample_steps = int(self.cfg.commands.resampling_time / self.dt)
        env_ids = (self.episode_length_buf % resample_steps == 0)\
            .nonzero(as_tuple=False).flatten()

        if len(env_ids) > 0:
            old_stand = self.stand_mode[env_ids].clone()
            old_hand = self.handstand_mode[env_ids].clone()
            self._resample_mode(env_ids)
            changed = (self.stand_mode[env_ids] != old_stand) | \
                      (self.handstand_mode[env_ids] != old_hand)
            # 切换的 env 置 mode_changed_buf，由 check_termination 并入 reset_buf
            self.mode_changed_buf[env_ids] = changed
            # 未切换的 env：同模式内重采样速度/高度
            self._resample_commands(env_ids[~changed])

        # 地形高度测量（沿用基类）
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()

        # 随机推动（沿用基类）
        if (self.cfg.domain_rand.push_robots
                and not getattr(self.cfg.domain_rand, "continuous_push", False)
                and self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _reset_root_states(self, env_ids):
        """locomotion 平走：正常站立；recovery 子模式：摔倒分桶采样。"""
        recovery_envs = env_ids[self.recovery_mode[env_ids]]
        normal_envs = env_ids[~self.recovery_mode[env_ids]]

        if len(normal_envs) > 0:
            super()._reset_root_states(normal_envs)

        if len(recovery_envs) > 0:
            # 分桶采样总倾倒角 theta，拆成 roll/pitch
            bins = torch.linspace(0.0, 3.14159, 8, device=self.device)
            idx = torch.randint(0, len(bins) - 1, (len(recovery_envs),), device=self.device)
            lo = bins[idx]
            hi = bins[idx + 1]
            theta = lo + (hi - lo) * torch.rand(len(recovery_envs), device=self.device)
            phi = torch.rand(len(recovery_envs), device=self.device) * 2 * 3.14159
            roll = theta * torch.cos(phi)
            pitch = theta * torch.sin(phi)
            self._write_recovery_pose(recovery_envs, roll, pitch)
            self.already_succeeded[recovery_envs] = False
            self.recent_max_agitation[recovery_envs] = 0.
            # 把摔倒姿态 push 到模拟器（基类只对 normal_envs push 过）
            env_ids_int32 = recovery_envs.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _write_recovery_pose(self, env_ids, roll, pitch):
        # 用 roll/pitch 构造四元数（落地实现可复用源工程 roll_pitch_yaw_to_quat）
        # 注意：此处未独立采样 yaw（身体朝向），摔倒朝向的 yaw 多样性受限；
        # 落地时应额外采样 yaw 并用完整 ZYX 顺序合成四元数，覆盖「任意朝向侧躺」。
        cy = torch.cos(pitch * 0.5)
        sy = torch.sin(pitch * 0.5)
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        w = cr * cy
        x = sr * cy
        y = cr * sy
        z = -sr * sy
        self.root_states[env_ids, 3:7] = torch.stack([x, y, z, w], dim=1)
        # 位置：xy 取各自环境原点（保持 env 定位），z 取基类初始高度
        self.root_states[env_ids, :2] = self.env_origins[env_ids, :2]
        self.root_states[env_ids, 2] = self.base_init_state[0, 2]
        self.root_states[env_ids, 7:13] = 0.

    def _reset_dofs(self, env_ids):
        # recovery 子模式：default ± 0.3；其余：default * rand(0.5, 1.5)（与基类一致）
        recovery_envs = env_ids[self.recovery_mode[env_ids]]
        normal_envs = env_ids[~self.recovery_mode[env_ids]]

        if len(normal_envs) > 0:
            self.dof_pos[normal_envs] = self.default_dof_pos * torch_rand_float(
                0.5, 1.5, (len(normal_envs), self.num_dof), device=self.device)
            self.dof_vel[normal_envs] = 0.
        if len(recovery_envs) > 0:
            n = len(recovery_envs)
            self.dof_pos[recovery_envs] = self.default_dof_pos + \
                (2 * torch.rand(n, self.num_dof, device=self.device) - 1) * 0.3
            self.dof_vel[recovery_envs] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # ---------- 目标重力切换 ----------
    def _update_target_gravity(self):
        # stand: [1,0,0]；handstand: [-1,0,0]；其余: [0,0,-1]
        self.target_gravity[:] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.target_gravity[self.stand_mode] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        self.target_gravity[self.handstand_mode] = torch.tensor([-1.0, 0.0, 0.0], device=self.device)


    # -------------- reward -----------------
    def _reward_orientation(self):
        # 必须覆写基类：基类 orientation 惩罚水平重力分量（驱动正立 [0,0,-1]），
        # 与 stand/handstand 的竖直姿态目标相反。不加 mask 会在 stand/handstand 下
        # 收到对抗梯度，故用 mode mask 隔离，只对 locomotion 生效。
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * \
            (~self.stand_mode & ~self.handstand_mode).float()

    def _reward_termination(self):
        # 必须覆写基类：基类 `self.reset_buf * ~self.time_out_buf` 会把 recovery 成功、
        # 模式切换（mode_changed_buf）也扣 -10，与「成功重奖」「切换重置不惩罚」意图矛盾。
        # 此处只在真正的失败（接触/超时）发放 termination 惩罚，scale 仍由外部统一乘。
        return self.reset_buf * ~self.time_out_buf * \
            (~self.recovery_mode).float() * (~self.mode_changed_buf).float()

    def _reward_handstand_orientation(self):
        # stand/handstand 竖直姿态；locomotion 用 [0,0,-1] 退化为 orientation
        diff = self.projected_gravity - self.target_gravity
        return torch.sum(torch.square(diff), dim=1) * \
            (self.stand_mode | self.handstand_mode).float()

    def _reward_base_height(self):
        # stand/handstand 高度：exp(-|h-target|*k)
        k = 10.0 * self.stand_mode.float() + 5.0 * self.handstand_mode.float()
        target = self.cfg.rewards.base_height_target
        rew = torch.exp(-torch.abs(self.base_height - target) * k)
        return rew * (self.stand_mode | self.handstand_mode).float()

    def _reward_upright_linear(self):
        # recovery 子模式：正立驱动
        return ((1.0 - self.projected_gravity[:, 2]) * 0.5) * self.recovery_mode.float()

    def _reward_stand_height(self):
        # recovery 子模式：高度驱动（upright gate）
        gate = (self.projected_gravity[:, 2] < -0.7).float()
        h = self.base_height
        target = self.commands[:, 3]
        return (torch.exp(-torch.square(h - target) / 0.05) * gate) * self.recovery_mode.float()

    def _reward_joint_to_default(self):
        # recovery / stand / handstand：关节归位（recovery 用全部，stand 后半身，handstand 前半身）
        d = torch.abs(self.dof_pos - self.default_dof_pos)
        rec = torch.exp(-torch.mean(torch.square(d), dim=1)) * self.recovery_mode.float()
        stand = torch.exp(-torch.sum(d[:, 6:], dim=1)) * self.stand_mode.float()
        hand = torch.exp(-torch.sum(d[:, :6], dim=1)) * self.handstand_mode.float()
        return rec + stand + hand

    def _reward_handstand_feet_on_air(self):
        """
        脚部在空奖励：
        1. 使用 self.contact_forces 判断足部是否接触地面（通过预先设置的阈值）。
        2. 如果所有足部都没有接触地面，则奖励1，否则奖励为0（或取平均）。
        """
        # contact_forces: shape = (num_envs, num_bodies, 3)
        contact = torch.norm(self.contact_forces[:, self.feet_name_reward_indices , :], dim=-1) > 1.0
        # 如果所有足部均未接触地面，reward = 1；也可以使用 mean 得到部分奖励
        reward = (~contact).float().prod(dim=1)
        return reward

    # ---------- 终止 ----------
    def check_termination(self):
        # 1) 接触失败：命中 terminate_after_contacts_on（"base"）即失败；recovery 子模式豁免
        contact = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
            dim=1)
        contact_fail = contact & ~self.recovery_mode

        # 2) recovery 子模式：不因接触地面失败；只有成功或超时
        upright = self.projected_gravity[:, 2] < -0.9
        height_ok = torch.abs(self.base_height - self.commands[:, 3]) < \
            self.cfg.rewards.stand_height_tolerance
        joint_ok = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) < \
            self.cfg.rewards.recovery_joint_tol

        # 平滑门控：agitation = ang_vel_xy² + lin_vel_z²
        agitation = torch.square(self.base_ang_vel[:, 0]) + torch.square(self.base_ang_vel[:, 1]) + \
            torch.square(self.base_lin_vel[:, 2])
        self.recent_max_agitation = torch.maximum(
            self.recent_max_agitation * self.cfg.rewards.smooth_success_decay,
            agitation)
        still = self.recent_max_agitation < self.cfg.rewards.smooth_success_threshold

        recover_success = upright & height_ok & joint_ok & still
        recover_success = recover_success & self.recovery_mode & ~self.already_succeeded

        # 3) 组合写入 self.reset_buf（基类 check_termination 返回 None，不能 super() 复用）
        self.reset_buf = torch.where(self.recovery_mode, recover_success, contact_fail)

        # 4) 模式切换触发重置（由 _post_physics_step_callback 置位）
        self.reset_buf |= self.mode_changed_buf

        # 5) 超时
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

        # 6) 标记成功（供奖励一次性重奖）
        self.already_succeeded |= recover_success

        return self.reset_buf


    