# 2027_RC_legged_robot 域随机化集成方案

本文只写设计文档和参考代码片段，不表示仓库代码已经按这些片段修改。目标是回答：在 `2027_RC_legged_robot` 里应该把 HIMLoco 和 My_unitree_go2_gym 的域随机化思想放到哪里、怎么组织、先做哪些、哪些不要直接照搬。

# **按步骤添加域随机化：**

- 这一部分按照 `/home/kk/下载/训练/训练.md` 那种写法来写：不先讲太多架构，直接告诉你打开哪个文件、找到哪个函数、在哪个位置加什么。

- 注意：下面所有内容都是“应该怎么改”的文档说明，当前没有真的修改 `.py` 代码。

- 我们这里不建议直接把 `HIMLoco` 或 `My_unitree_go2_gym` 的 `legged_robot.py` 整个复制过来，因为你这个项目 `2027_RC_legged_robot` 后面要集成很多算法，公共的域随机化最好统一放在 base 环境里。

## **第一步：先补齐 domain_rand 配置**

- 打开这个文件：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/base/legged_robot_config.py
    ```

- 找到这个位置：

    ```Python
    class LeggedRobotCfg(BaseConfig):
        ...
        class domain_rand:
            ...
    ```

- 当前这个配置块大概在 `91-170` 行；你已经写了很多配置，例如：

    ```Python
    randomize_payload_mass = False
    randomize_link_mass = False
    randomize_com_displacement = False
    randomize_joint_friction = False
    randomize_joint_damping = False
    randomize_joint_armature = False
    randomize_friction = False
    randomize_restitution = False
    push_robots = False
    randomize_pd_gains = False
    randomize_motor_zero_offset = False
    randomize_motor_strength = False
    add_obs_latency = False
    add_cmd_action_latency = False
    ```

- 但是你的 `legged_robot.py` 里已经写到了下面这个字段：

    ```Python
    self.cfg.domain_rand.randomize_calculated_torque
    self.cfg.domain_rand.torque_multiplier_range
    ```

- 所以如果你先不改变量名字，最小需要在 `randomize_motor_strength` 后面加上：

    ```Python
    # 电机输出力矩倍率；短期兼容 legged_robot.py 里已有的名字
    randomize_calculated_torque = False
    torque_multiplier_range = [0.8, 1.2]
    ```

- 但是我更推荐你后面统一成一个名字，也就是只保留：

    ```Python
    # 电机输出强度
    randomize_motor_strength = False
    motor_strength_range = [0.8, 1.2]
    ```

- 然后把 `legged_robot.py` 里所有 `randomize_calculated_torque` 改成 `randomize_motor_strength`，把 `torque_multiplier` 改成 `motor_strength_multiplier`。这样名字更清楚，不会同时存在两套“电机倍率”。

- 在 `range_cmd_action_latency = [2, 4]` 后面再加一个开关，后面如果要把真实随机化参数给 critic 用，就用这个控制：

    ```Python
    # 是否把真实 domain randomization 参数拼到 critic / teacher 的 privileged obs 中
    # 第一阶段先保持 False，避免影响 obs 维度
    add_domain_rand_privileged_obs = False
    ```

## **第二步：在创建环境之前添加创建期张量**

- 打开这个文件：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py
    ```

- 找到这个函数：

    ```Python
    def _create_envs(self):
    ```

- 在这个函数里找到加载机器人 asset 的部分，大概是当前 `425-435` 行：

    ```Python
    robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
    self.num_dof = self.gym.get_asset_dof_count(robot_asset)
    self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
    dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
    rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

    body_names = self.gym.get_asset_rigid_body_names(robot_asset)
    self.dof_names = self.gym.get_asset_dof_names(robot_asset)
    self.num_bodies = len(body_names)
    self.num_dofs = len(self.dof_names)
    ```

- 在 `self.num_dofs = len(self.dof_names)` 后面添加这一句：

    ```Python
    self._init_domain_rand_creation_buffers()
    ```

- 也就是改成这样：

    ```Python
    body_names = self.gym.get_asset_rigid_body_names(robot_asset)
    self.dof_names = self.gym.get_asset_dof_names(robot_asset)
    self.num_bodies = len(body_names)
    self.num_dofs = len(self.dof_names)

    self._init_domain_rand_creation_buffers()

    feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
    ```

- 为什么加在这里：

    - 因为 `_process_rigid_shape_props`、`_process_dof_props`、`_process_rigid_body_props` 都是在 `_create_envs` 里面创建每个 env 的时候调用的；

    - 这些函数里面会用到 `friction_coeffs`、`payload_mass`、`joint_friction_coeffs` 这些张量；

    - 所以必须在 `for i in range(self.num_envs)` 之前创建好。

## **第三步：添加创建期 buffer 函数**

- 还是在这个文件：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py
    ```

- 找到这个位置：

    ```Python
    def _process_rigid_body_props(self, props, env_id):
        ...
        return props

    def _create_envs(self):
        ...
    ```

- 在 `_process_rigid_body_props` 结束后，`_create_envs` 开始前，添加这个函数：

    ```Python
    def _init_domain_rand_creation_buffers(self):
        self.friction_coeffs = torch.ones(
            self.num_envs, 1, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(
            self.num_envs, 1, device=self.device, requires_grad=False)

        self.payload_mass = torch.zeros(
            self.num_envs, 1, device=self.device, requires_grad=False)
        self.link_mass_ratios = torch.ones(
            self.num_envs, self.num_bodies - 1, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False)

        self.joint_friction_coeffs = torch.ones(
            self.num_envs, 1, device=self.device, requires_grad=False)
        self.joint_damping_coeffs = torch.ones(
            self.num_envs, 1, device=self.device, requires_grad=False)
        self.joint_armatures = torch.zeros(
            self.num_envs, 1, device=self.device, requires_grad=False)
    ```

- 这个函数只负责“创建 actor 时会用到的随机化参数”。

- 这里不要放 `p_gains_multiplier`、`motor_zero_offsets`、`latency_buffer`，因为那些是训练过程中每次 reset 要重新采样的运行期参数，放在后面的 `_init_buffers` 更合适。

## **第四步：在 _init_buffers 里添加运行期张量**

- 找到这个函数：

    ```Python
    def _init_buffers(self):
    ```

- 当前这个函数最后会设置默认关节角，大概在 `94-111` 行：

    ```Python
    self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
    for i in range(self.num_dofs):
        ...
    self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
    ```

- 在 `self.default_dof_pos = self.default_dof_pos.unsqueeze(0)` 后面添加：

    ```Python
    self._init_domain_rand_runtime_buffers()
    ```

- 然后在刚才第三步那个函数后面，继续添加这个运行期初始化函数：

    ```Python
    def _init_domain_rand_runtime_buffers(self):
        self.p_gains_multiplier = torch.ones(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.motor_zero_offsets = torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.motor_strength_multiplier = torch.ones(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False)

        self.rand_push_force = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False)
        self.rand_push_torque = torch.zeros(
            self.num_envs, 3, device=self.device, requires_grad=False)

        self._init_latency_buffers()
        self._reset_latency_buffers(torch.arange(self.num_envs, device=self.device))
    ```

- 这里为什么要单独放在 `_init_buffers`：

    - 因为 `p_gains_multiplier`、`motor_zero_offsets`、`motor_strength_multiplier` 是每次 reset 可以重新采样的；

    - 这些值不是创建 actor 时固定写进 Isaac Gym 的属性，而是在 `_compute_torques` 里每一步参与力矩计算；

    - 所以它们属于运行期 buffer。

## **第五步：添加 action latency 和 obs latency 的 buffer**

- 继续在 `legged_robot.py` 里添加，位置放在 `_init_domain_rand_runtime_buffers` 后面即可。

- 先添加初始化函数：

    ```Python
    def _init_latency_buffers(self):
        cfg = self.cfg.domain_rand
        max_cmd_delay = cfg.range_cmd_action_latency[1]
        max_motor_delay = cfg.range_obs_motor_latency[1]
        max_imu_delay = cfg.range_obs_imu_latency[1]

        self.cmd_action_latency_buffer = torch.zeros(
            self.num_envs, self.num_actions, max_cmd_delay + 1,
            device=self.device, requires_grad=False)
        self.obs_motor_latency_buffer = torch.zeros(
            self.num_envs, self.num_actions * 2, max_motor_delay + 1,
            device=self.device, requires_grad=False)
        self.obs_imu_latency_buffer = torch.zeros(
            self.num_envs, 6, max_imu_delay + 1,
            device=self.device, requires_grad=False)

        self.cmd_action_latency_simstep = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.obs_motor_latency_simstep = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.obs_imu_latency_simstep = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
    ```

- 再添加 reset 时重置 latency 的函数：

    ```Python
    def _reset_latency_buffers(self, env_ids):
        cfg = self.cfg.domain_rand

        if cfg.add_cmd_action_latency:
            self.cmd_action_latency_buffer[env_ids] = 0.0
            if cfg.randomize_cmd_action_latency:
                self.cmd_action_latency_simstep[env_ids] = torch.randint(
                    cfg.range_cmd_action_latency[0], cfg.range_cmd_action_latency[1] + 1,
                    (len(env_ids),), device=self.device)
            else:
                self.cmd_action_latency_simstep[env_ids] = cfg.range_cmd_action_latency[1]

        if cfg.add_obs_latency:
            self.obs_motor_latency_buffer[env_ids] = 0.0
            self.obs_imu_latency_buffer[env_ids] = 0.0
            if cfg.randomize_obs_motor_latency:
                self.obs_motor_latency_simstep[env_ids] = torch.randint(
                    cfg.range_obs_motor_latency[0], cfg.range_obs_motor_latency[1] + 1,
                    (len(env_ids),), device=self.device)
            else:
                self.obs_motor_latency_simstep[env_ids] = cfg.range_obs_motor_latency[1]

            if cfg.randomize_obs_imu_latency:
                self.obs_imu_latency_simstep[env_ids] = torch.randint(
                    cfg.range_obs_imu_latency[0], cfg.range_obs_imu_latency[1] + 1,
                    (len(env_ids),), device=self.device)
            else:
                self.obs_imu_latency_simstep[env_ids] = cfg.range_obs_imu_latency[1]
    ```

- 注意这里的延迟单位是 **physics sim step**，不是 policy step。也就是说如果 `range_cmd_action_latency = [1, 3]`，表示延迟 1 到 3 个仿真小步。

## **第六步：每次 reset 的时候重新采样电机和延迟参数**

- 找到这个函数：

    ```Python
    def reset_idx(self, env_ids):
    ```

- 在里面找到这段，大概在当前 `647-652` 行：

    ```Python
    self._reset_dofs(env_ids)
    self._reset_root_states(env_ids)

    self._resample_commands(env_ids)
    ```

- 在 `_reset_root_states(env_ids)` 后面、`_resample_commands(env_ids)` 前面添加：

    ```Python
    self._resample_domain_rand(env_ids)
    ```

- 修改后就是：

    ```Python
    self._reset_dofs(env_ids)
    self._reset_root_states(env_ids)

    self._resample_domain_rand(env_ids)
    self._resample_commands(env_ids)
    ```

- 然后添加这个函数，位置放在 `_reset_latency_buffers` 后面即可：

    ```Python
    def _resample_domain_rand(self, env_ids):
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        cfg = self.cfg.domain_rand

        if cfg.randomize_pd_gains:
            self.p_gains_multiplier[env_ids, :] = torch_rand_float(
                cfg.stiffness_multiplier_range[0], cfg.stiffness_multiplier_range[1],
                (n, self.num_actions), device=self.device)
            self.d_gains_multiplier[env_ids, :] = torch_rand_float(
                cfg.damping_multiplier_range[0], cfg.damping_multiplier_range[1],
                (n, self.num_actions), device=self.device)
        else:
            self.p_gains_multiplier[env_ids, :] = 1.0
            self.d_gains_multiplier[env_ids, :] = 1.0

        if cfg.randomize_motor_zero_offset:
            self.motor_zero_offsets[env_ids, :] = torch_rand_float(
                cfg.motor_zero_offset_range[0], cfg.motor_zero_offset_range[1],
                (n, self.num_actions), device=self.device)
        else:
            self.motor_zero_offsets[env_ids, :] = 0.0

        if cfg.randomize_motor_strength:
            self.motor_strength_multiplier[env_ids, :] = torch_rand_float(
                cfg.motor_strength_range[0], cfg.motor_strength_range[1],
                (n, self.num_actions), device=self.device)
        else:
            self.motor_strength_multiplier[env_ids, :] = 1.0

        self._reset_latency_buffers(env_ids)
    ```

- 这个函数主要负责那些“每个 episode 可以重新随机一次”的东西：

    - PD 增益倍率；
    - 电机零位偏移；
    - 电机输出强度；
    - action latency；
    - obs latency。

## **第七步：把 action latency 接入 step 函数**

- 找到 `step` 函数：

    ```Python
    def step(self, actions):
    ```

- 当前 decimation 循环大概是这样：

    ```Python
    for _ in range(self.cfg.control.decimation):
        self.torques = self._compute_torques(self.actions).view(self.torques.shape)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
    ```

- 先添加一个函数，用来拿延迟后的动作：

    ```Python
    def _get_delayed_actions(self):
        cfg = self.cfg.domain_rand
        if not cfg.add_cmd_action_latency:
            return self.actions

        max_delay = cfg.range_cmd_action_latency[1]
        self.cmd_action_latency_buffer[:, :, 1:] = self.cmd_action_latency_buffer[:, :, :max_delay].clone()
        self.cmd_action_latency_buffer[:, :, 0] = self.actions.clone()
        return self.cmd_action_latency_buffer[
            torch.arange(self.num_envs, device=self.device),
            :,
            self.cmd_action_latency_simstep.long(),
        ]
    ```

- 然后把 `step` 里的循环改成这样：

    ```Python
    for _ in range(self.cfg.control.decimation):
        actions_for_torque = self._get_delayed_actions()
        self.torques = self._compute_torques(actions_for_torque).view(self.torques.shape)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self._update_obs_latency_buffers()
    ```

- 这里有一个很关键的点：`_get_delayed_actions` 返回的是原始 action，不要在这里乘 `action_scale`。

- 因为你的 `_compute_torques` 里面已经有：

    ```Python
    actions_scaled = actions * self.cfg.control.action_scale
    ```

- 如果 latency buffer 里存的是 scaled action，就会重复缩放，动作会变小或者变乱。

## **第八步：把 PD、零位、电机强度真正放进 _compute_torques**

- 找到这个函数：

    ```Python
    def _compute_torques(self, actions):
    ```

- 当前里面是普通 PD：

    ```Python
    actions_scaled = actions * self.cfg.control.action_scale
    control_type = self.cfg.control.control_type
    if control_type=="P":
        torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
    ```

- 把这一段改成带随机化倍率的版本：

    ```Python
    actions_scaled = actions * self.cfg.control.action_scale
    control_type = self.cfg.control.control_type
    p_gains = self.p_gains.unsqueeze(0) * self.p_gains_multiplier
    d_gains = self.d_gains.unsqueeze(0) * self.d_gains_multiplier

    if control_type == "P":
        target_pos = actions_scaled + self.default_dof_pos + self.motor_zero_offsets
        torques = p_gains * (target_pos - self.dof_pos) - d_gains * self.dof_vel
    elif control_type == "V":
        torques = p_gains * (actions_scaled - self.dof_vel) \
            - d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
    elif control_type == "T":
        torques = actions_scaled
    else:
        raise NameError(f"Unknown controller type: {control_type}")

    torques = torques * self.motor_strength_multiplier
    return torch.clip(torques, -self.torque_limits, self.torque_limits)
    ```

- 到这一步以后，下面这几个配置才是真的有效果：

    ```Python
    randomize_pd_gains = True
    randomize_motor_zero_offset = True
    randomize_motor_strength = True
    ```

## **第九步：把 obs latency 接入观测函数**

- 先添加更新观测延迟 buffer 的函数，位置放在 `_get_delayed_actions` 后面即可：

    ```Python
    def _update_obs_latency_buffers(self):
        cfg = self.cfg.domain_rand
        if not cfg.add_obs_latency:
            return

        if cfg.randomize_obs_motor_latency:
            max_delay = cfg.range_obs_motor_latency[1]
            q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dq = self.dof_vel * self.obs_scales.dof_vel
            self.obs_motor_latency_buffer[:, :, 1:] = self.obs_motor_latency_buffer[:, :, :max_delay].clone()
            self.obs_motor_latency_buffer[:, :, 0] = torch.cat((q, dq), dim=1).clone()

        if cfg.randomize_obs_imu_latency:
            max_delay = cfg.range_obs_imu_latency[1]
            imu = torch.cat((
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
            ), dim=1)
            self.obs_imu_latency_buffer[:, :, 1:] = self.obs_imu_latency_buffer[:, :, :max_delay].clone()
            self.obs_imu_latency_buffer[:, :, 0] = imu.clone()
    ```

- 再添加两个读取函数：

    ```Python
    def _get_motor_obs(self):
        cfg = self.cfg.domain_rand
        if cfg.add_obs_latency and cfg.randomize_obs_motor_latency:
            motor_obs = self.obs_motor_latency_buffer[
                torch.arange(self.num_envs, device=self.device),
                :,
                self.obs_motor_latency_simstep.long(),
            ]
            dof_pos_obs = motor_obs[:, :self.num_actions]
            dof_vel_obs = motor_obs[:, self.num_actions:]
        else:
            dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dof_vel_obs = self.dof_vel * self.obs_scales.dof_vel
        return dof_pos_obs, dof_vel_obs

    def _get_imu_obs(self):
        cfg = self.cfg.domain_rand
        if cfg.add_obs_latency and cfg.randomize_obs_imu_latency:
            imu_obs = self.obs_imu_latency_buffer[
                torch.arange(self.num_envs, device=self.device),
                :,
                self.obs_imu_latency_simstep.long(),
            ]
            base_ang_vel_obs = imu_obs[:, :3]
            projected_gravity_obs = imu_obs[:, 3:6]
        else:
            base_ang_vel_obs = self.base_ang_vel * self.obs_scales.ang_vel
            projected_gravity_obs = self.projected_gravity
        return base_ang_vel_obs, projected_gravity_obs
    ```

- 然后找到观测函数：

    ```Python
    def compute_observations(self):
    ```

- 当前里面直接拼的是：

    ```Python
    self.base_ang_vel * self.obs_scales.ang_vel
    self.projected_gravity
    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
    self.dof_vel * self.obs_scales.dof_vel
    ```

- 修改成先从函数里拿观测：

    ```Python
    heights = None
    base_ang_vel_obs, projected_gravity_obs = self._get_imu_obs()
    dof_pos_obs, dof_vel_obs = self._get_motor_obs()
    ```

- 然后 `actor_obs` 改成：

    ```Python
    actor_obs = torch.cat((
        base_ang_vel_obs,
        projected_gravity_obs,
        self.commands[:, :3] * self.commands_scale,
        dof_pos_obs,
        dof_vel_obs,
        self.actions,
    ), dim=-1)
    ```

- 这里顺手也修掉一个问题：`heights = None` 一定要放在函数开头。否则 `measure_heights=False` 的时候，后面 `if heights is not None:` 会访问一个没定义的变量。

## **第十步：把 push 的角速度也加上**

- 找到这个函数：

    ```Python
    def _push_robots(self):
    ```

- 当前只随机了 base 的 xy 线速度：

    ```Python
    max_vel = self.cfg.domain_rand.max_push_vel_xy
    self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
    ```

- 但是你的 config 里已经有：

    ```Python
    max_push_ang_vel = 0.6
    ```

- 所以建议改成：

    ```Python
    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_ang_vel = self.cfg.domain_rand.max_push_ang_vel
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)
        self.root_states[:, 10:13] = torch_rand_float(
            -max_ang_vel, max_ang_vel, (self.num_envs, 3), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    ```

## **第十一步：最后再到具体任务里打开开关**

- 上面的改动都属于 base 能力，默认开关都是 `False`，所以不会直接改变所有任务。

- 只有等 base 里面都接好了，再到具体 task 的 config 里打开。

- 例如你现在要先给 `him_go1` 打开，可以改这里：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/him_go1/him_go1_config.py
    ```

- 找到：

    ```Python
    class domain_rand( LeggedRobotCfg.domain_rand ):
        pass
    ```

- 改成先保守打开：

    ```Python
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.6, 1.4]

        randomize_payload_mass = True
        payload_mass_range = [-0.5, 1.0]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.02, 0.02]

        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        push_robots = True
        push_interval_s = 6.0
        max_push_vel_xy = 0.3
        max_push_ang_vel = 0.4
    ```

- 如果是 GO2W，我建议比 GO1 多开 latency，因为轮腿上实机延迟和电机误差更明显：

    ```Python
    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_friction = True
        friction_range = [0.6, 1.4]

        randomize_payload_mass = True
        payload_mass_range = [-0.5, 1.5]
        randomize_com_displacement = True
        com_displacement_range = [-0.03, 0.03]

        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.02, 0.02]
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]

        add_cmd_action_latency = True
        randomize_cmd_action_latency = True
        range_cmd_action_latency = [0, 2]

        add_obs_latency = True
        randomize_obs_motor_latency = True
        range_obs_motor_latency = [0, 2]
        randomize_obs_imu_latency = True
        range_obs_imu_latency = [0, 1]

        push_robots = True
        push_interval_s = 6.0
        max_push_vel_xy = 0.3
        max_push_ang_vel = 0.4
    ```

## **这部分的添加顺序总结**

- 如果你真正开始改代码，建议就按照这个顺序来：

    1. 先补 `legged_robot_config.py` 里的字段；

    2. 在 `_create_envs` 里添加 `self._init_domain_rand_creation_buffers()`；

    3. 在 `_process_rigid_body_props` 后面添加 `_init_domain_rand_creation_buffers` 函数；

    4. 在 `_init_buffers` 末尾添加 `self._init_domain_rand_runtime_buffers()`；

    5. 添加 `_init_domain_rand_runtime_buffers`、`_init_latency_buffers`、`_reset_latency_buffers`；

    6. 在 `reset_idx` 里添加 `self._resample_domain_rand(env_ids)`；

    7. 添加 `_resample_domain_rand` 函数；

    8. 添加 `_get_delayed_actions`，并修改 `step` 的 decimation loop；

    9. 修改 `_compute_torques`，让 PD、零位、电机强度真正参与计算；

    10. 添加 `_update_obs_latency_buffers`、`_get_motor_obs`、`_get_imu_obs`；

    11. 修改 `compute_observations`；

    12. 最后才到具体 task config 里打开随机化开关。

- 这里不要一上来就把所有开关全部打开。建议先只打开 `friction + payload + push`，确认能正常训练后，再打开 `PD + motor offset + motor strength`，最后再打开 `latency`。

## 1. 结论先行

`2027_RC_legged_robot` 应该采用“**基类统一承载域随机化能力，具体任务配置只打开开关和范围**”的方式：

1. **公共随机化逻辑放在** `legged_gym/envs/base/legged_robot.py`。
2. **默认开关和范围放在** `legged_gym/envs/base/legged_robot_config.py::LeggedRobotCfg.domain_rand`。
3. **GO1 / GO2W / HIM / CTS 等任务只在自己的 config 里覆盖范围**，不要把 My_unitree 那种大量重复环境代码复制到每个 task 文件。
4. **actor obs 不直接暴露真实随机化参数**；如果要给 critic 或 teacher/adaptation 模块使用，单独放到 privileged obs，避免部署时 actor 依赖仿真真值。
5. **先修当前 2027 的随机化骨架缺口，再打开强随机化**：当前配置项比实际接线更完整，若直接打开部分开关会遇到未初始化变量、未使用倍率或配置缺失。

最推荐的落点如下：

| 类型 | 推荐落点 | 调用时机 | 说明 |
| --- | --- | --- | --- |
| 地面摩擦 / 恢复系数 | `_process_rigid_shape_props()` | 创建 actor 前 | 每个 env 初始采样；如需每 episode 重采样，额外实现 `refresh_actor_rigid_shape_props()`。 |
| DOF 摩擦 / 阻尼 / armature | `_process_dof_props()` | 创建 actor 前 | Isaac Gym 里这类物理属性更适合 env 创建时固化，先不要每 episode 动态改。 |
| base payload / link mass / COM | `_process_rigid_body_props()` | 创建 actor 后设置 rigid body props | 需要保存每个 env 的采样值，供 privileged obs / 日志使用。 |
| PD gain / 电机零位 / 电机强度 | `_init_domain_rand_buffers()` + `_resample_domain_rand()` + `_compute_torques()` | 初始化、reset、每步算 torque | 这是部署鲁棒性最直接的一组，必须真正进入 torque 计算。 |
| action latency | `step()` 内 decimation 子步前 | 每个 physics substep | 使用 raw action buffer 或 scaled action buffer 二选一，不能双重 scale。 |
| motor / IMU observation latency | `step()` 内刷新 DOF/root 后更新 buffer，`compute_observations()` 中读取 | 每个 physics substep 更新，policy step 读延迟值 | 这是 My_unitree 的核心价值，但要写成基类工具函数。 |
| push / disturbance | `_post_physics_step_callback()` | 每隔固定 step | push 改 base velocity；continuous force/torque 用 force tensor，先从 push 开始。 |
| terrain / command curriculum | 保持现有 `_update_terrain_curriculum()` / `update_command_curriculum()` | reset | 不属于狭义 domain rand，但和训练鲁棒性一起调。 |

## 2. 三个项目的结构差异

### 2.1 2027_RC_legged_robot 当前状态

当前项目已经有一个比较完整的 `LeggedRobotCfg.domain_rand` 配置块，包含 payload mass、link mass、COM、关节摩擦/阻尼/armature、地面摩擦、restitution、push、PD gain、电机零位、电机强度、obs latency 和 action latency。位置是：

```text
2027_RC_legged_robot/legged_gym/envs/base/legged_robot_config.py:91-170
```

当前 `LeggedRobot` 也已经有一批随机化 hook：

```text
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:233-274   _process_rigid_shape_props
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:276-345   _process_dof_props
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:347-395   _process_rigid_body_props
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:555-560   _push_robots
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:729-753   step
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:788-803   _post_physics_step_callback
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:805-827   _compute_torques
2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py:829-860   compute_observations
```

但当前骨架还没有闭环，主要问题是：

| 问题 | 位置 | 影响 |
| --- | --- | --- |
| `randomize_calculated_torque` / `torque_multiplier_range` 在代码中使用，但 config 中没有定义 | `legged_robot.py:299-302` vs `legged_robot_config.py:91-170` | 打开相关逻辑会 AttributeError。 |
| `torque_multiplier`、`motor_zero_offsets`、`p_gains_multiplier`、`d_gains_multiplier`、`joint_friction_coeffs`、`joint_damping_coeffs`、`joint_armatures` 等 buffer 被使用但未在 `_init_buffers()` 初始化 | `legged_robot.py:299-343` | 打开对应开关会 AttributeError。 |
| `payload_mass`、`link_mass_ratios`、`com_displacements` 被写入但未初始化 | `legged_robot.py:361-388` | 打开 payload/link/com 会 AttributeError。 |
| `randomize_motor_strength` 只有 config，没有进入 `_compute_torques()` | `legged_robot_config.py:151-152`、`legged_robot.py:805-827` | 电机强度随机化开关无实际效果。 |
| `randomize_pd_gains` 和 `randomize_motor_zero_offset` 采样后没有真正进入 `_compute_torques()` | `legged_robot.py:311-317`、`legged_robot.py:805-827` | PD/零位随机化无实际控制效果。 |
| obs/action latency 配置存在，但没有 latency buffer 和接入点 | `legged_robot_config.py:155-170`、`legged_robot.py:729-860` | 延迟随机化目前只是配置占位。 |
| `compute_observations()` 中 `heights` 只在 `measure_heights=True` 分支定义，但 privileged obs 分支无条件检查 `if heights is not None` | `legged_robot.py:840-855` | `measure_heights=False` 且 `privileged_obs_buf` 存在时会访问未定义变量。 |
| `him_go2w` / `cts_go2w` 文件为空，注册表只注册 `legged_gym_go1` | `envs/him_go2w/*.py`、`envs/cts_go2w/*.py`、`envs/__init__.py:9` | 域随机化应先在 base 做完，再按任务补 config/注册。 |

所以 2027 不是“完全没有域随机化”，而是“配置和 hook 已经开始写了，但需要整理成可运行闭环”。

### 2.2 HIMLoco 的做法

HIMLoco 的域随机化更集中在 base 环境和 GO2W config，特点是：

1. `LeggedRobotCfg.domain_rand` 较简洁，包含 payload、COM、link mass、joint friction/damping、ground friction、restitution、motor strength、Kp/Kd、disturbance、push、action_delay。
2. `step()` 中实现 decimation 内的 action delay：每个 env 随机 delay step，用 `last_actions` 到 `actions` 的切换模拟延迟。
3. `reset_idx()` 会重采样 Kp、Kd、motor strength，并调用 `refresh_actor_rigid_shape_props(env_ids)` 使 friction/restitution 可每 episode 更新。
4. `_compute_torques()` 真正使用 `Kp_factors`、`Kd_factors`、`motor_strength_factors`。
5. GO2W 子类的 observation 使用 history / privileged obs，并把 base velocity、disturbance、height、contact force 等信息放到 privileged/history 结构里。

关键证据位置：

```text
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot_config.py:95-148
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py:40-65
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py:111-145
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py:265-299
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py:380-398
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py:430-433
HIMLoco/legged_gym/legged_gym/envs/base/legged_robot.py:553-570
HIMLoco/legged_gym/legged_gym/envs/go2w/go2w_config.py:86-138
HIMLoco/legged_gym/legged_gym/envs/go2w/go2w_legged_robot.py:21-136
```

HIMLoco 值得借鉴的是：**不要只写配置，必须让随机化参数进入 reset、torque、obs 或 sim props**。不建议照搬的是：它的 `action_delay` 是一种简单的 decimation 内动作切换，对真实通信/观测延迟表达不如 My_unitree 完整。

### 2.3 My_unitree_go2_gym 的做法

My_unitree 的域随机化更重，特点是：

1. base config 中包含更完整的真实部署随机化：base/link mass、base COM、PD gain、torque multiplier、电机零位、joint friction/damping/armature、motor obs latency、IMU latency、cmd action latency。
2. `step()` 每个 decimation 子步都先取 delayed action，再计算 torque；每个 physics step 后更新 obs latency buffer。
3. `compute_observations()` 从 latency buffer 中读取 motor/IMU 延迟观测。
4. 一些任务会把 domain randomization info 拼进 privileged obs，服务 critic 或 teacher。
5. 缺点是多个任务环境文件里重复了大量相似逻辑，维护成本高。

关键证据位置：

```text
My_unitree_go2_gym/legged_gym/envs/base/legged_robot_config.py:122-173
My_unitree_go2_gym/legged_gym/envs/base/legged_robot.py:82-145
My_unitree_go2_gym/legged_gym/envs/base/legged_robot.py:282-383
My_unitree_go2_gym/legged_gym/envs/base/legged_robot.py:423-448
My_unitree_go2_gym/legged_gym/envs/base/legged_robot.py:628-648
My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Trot.py:232-290
My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Trot.py:492-513
My_unitree_go2_gym/legged_gym/envs/Go2_MoB/GO2_Trot/GO2_Trot_config.py:120-141
```

My_unitree 值得借鉴的是：**latency buffer 的位置和数据流**。不建议照搬的是：**把相同随机化代码散落到每个任务 env 文件**。2027 当前正在融合多算法，更需要统一接口，否则 HIM/CTS/MGDP/原生 PPO 的 obs 维度和 reset 语义很容易分叉。

## 3. 推荐架构

### 3.1 模块边界

建议在 `LeggedRobot` 内部形成以下私有方法边界：

```text
_init_domain_rand_buffers()       初始化所有域随机化 tensor，默认全 1 或 0
_resample_domain_rand(env_ids)    每 episode 重采样控制器、电机、latency 等可动态变化参数
_reset_latency_buffers(env_ids)   清空并重采样 delay simstep
_get_delayed_actions(actions)     给 step() 使用，返回用于 torque 的 action
_update_obs_latency_buffers()     每个 physics substep 后更新 motor/IMU buffer
_get_motor_obs()                  compute_observations() 读取 motor latency 后的 q/dq
_get_imu_obs()                    compute_observations() 读取 IMU latency 后的 ang_vel/gravity 或 euler
_get_domain_rand_privileged_obs() 可选，把真实随机化参数拼给 critic/teacher
refresh_actor_rigid_shape_props() 可选，让 friction/restitution 每 episode 重采样
```

这些函数全部属于 base 环境，不应该放到 `him_go1`、`go2w`、`cts_go2w` 这类任务文件里。任务文件只覆盖：

```python
class domain_rand(LeggedRobotCfg.domain_rand):
    randomize_friction = True
    friction_range = [0.6, 1.4]
    randomize_payload_mass = True
    payload_mass_range = [-0.5, 1.5]
    # ...
```

### 3.2 actor obs 与 privileged obs 分层

推荐遵守以下规则：

| 信息 | actor obs | privileged obs / critic | 原因 |
| --- | --- | --- | --- |
| command、IMU、joint q/dq、last action | 可以 | 可以 | 部署可获得。 |
| 延迟后的 motor/IMU 观测 | 可以 | 可以 | 部署可获得或可模拟。 |
| base linear velocity | 一般不要，除非部署可估计并计划一致 | 可以 | 真实部署通常不可直接精确获得。 |
| friction、mass、COM、PD multiplier、latency simstep 真值 | 不要 | 可选 | actor 直接看真值会破坏 sim2real 前提。 |
| terrain height samples | 取决于是否有感知输入 | 可以 | 如果部署没有高度图，就不应给 actor。 |
| disturbance force 真值 | 不要 | 可选 | 部署不可直接获得。 |

如果后续做 HIM / adaptation，可以让 estimator 从 history 推断隐变量，而不是把真实 domain rand vector 直接给 actor。

## 4. 具体实施顺序

### Phase A：先补齐当前 2027 的安全骨架

目标：所有 domain rand 开关默认关闭时，行为与现在一致；打开任何单个开关时不 AttributeError。

1. 在 config 中统一命名，建议保留 `randomize_motor_strength`，删除或废弃 `randomize_calculated_torque` 概念；如果短期不改代码名，则必须给 config 补 `randomize_calculated_torque` 和 `torque_multiplier_range`。
2. 在 `_init_buffers()` 末尾调用 `_init_domain_rand_buffers()`。
3. 初始化所有被 hook 写入或读取的 tensor。
4. 修复 `compute_observations()` 的 `heights` 未定义问题。

### Phase B：接入动力学和接触随机化

目标：地面、刚体、DOF 物理参数随机化能够在 env 创建时生效。

优先顺序：

1. friction / restitution。
2. payload mass。
3. link mass。
4. base COM。
5. joint friction / damping / armature。

这部分主要使用现有 hook，不需要改训练算法。

### Phase C：接入控制器和电机随机化

目标：PD gain、电机零位、电机强度实际进入 torque。

1. `_resample_domain_rand(env_ids)` 在 reset 时重采样 `p_gains_multiplier`、`d_gains_multiplier`、`motor_zero_offsets`、`motor_strength_multiplier`。
2. `_compute_torques()` 使用这些 multiplier。
3. 所有 tensor shape 建议用 `(num_envs, num_actions)`，不要用 `(num_envs, 1)`，因为 GO2W 轮腿可能需要分关节/分轮配置。

### Phase D：接入 action / observation latency

目标：模拟部署中的指令链路和传感器链路延迟。

1. action latency 在 `step()` 的 decimation 循环内取 delayed action。
2. motor/IMU obs latency 在每个 physics substep 刷新 tensor 后更新 buffer。
3. `compute_observations()` 只读取 `_get_motor_obs()` / `_get_imu_obs()`，不要到处直接拼 `self.dof_pos` / `self.base_ang_vel`。

### Phase E：打开 push 和外力扰动

目标：增强恢复能力。

1. 先用 `_push_robots()` 改 base linear/angular velocity。
2. 再考虑 continuous force/torque。当前 2027 config 已有 `continuous_push`、`max_push_force`、`max_push_torque`，但代码还没有完整 force tensor 接线。

### Phase F：任务层配置和注册

目标：把能力真正给 GO1/GO2W/HIM/CTS 任务使用。

1. `legged_gym/envs/__init__.py` 注册未来的 `go2w`、`him_go2w`、`cts_go2w`。
2. `him_go2w`、`cts_go2w` 当前为空文件，应先补 task config，再决定是否需要 task env subclass。
3. 如果只是 domain randomization，不需要新 subclass；只有 GO2W 轮腿 torque/obs layout 不同，才新增子类。

## 4.5 按文件精确添加位置

这一节只回答“到底加到哪里”。下面的行号基于当前工作区扫描结果，后续如果文件前面有改动，按函数名和相邻代码定位即可。

### 4.5.1 `legged_gym/envs/base/legged_robot_config.py`

**位置 1：补齐 `domain_rand` 字段**

当前位置：

```text
legged_gym/envs/base/legged_robot_config.py:91-170
class LeggedRobotCfg(BaseConfig):
    class domain_rand:
        ...
```

具体做法：在 `randomize_motor_strength` / `motor_strength_range` 后面，也就是当前约 `151-152` 行后，补上当前代码已经引用但 config 没有定义的字段。最小可运行补丁是：

```python
        # 电机输出力矩倍率；如果后续统一命名，可用 randomize_motor_strength 取代它
        randomize_calculated_torque = False
        torque_multiplier_range = [0.8, 1.2]
```

更推荐的长期做法：不要保留两套名字，把 `legged_robot.py` 里的 `randomize_calculated_torque` 改为 `randomize_motor_strength`，把 `torque_multiplier` 统一改名为 `motor_strength_multiplier`。这样 config 里只保留：

```python
        randomize_motor_strength = False
        motor_strength_range = [0.8, 1.2]
```

**位置 2：新增 privileged 开关**

当前位置：`domain_rand` 末尾，`range_cmd_action_latency = [2, 4]` 后面，也就是当前约 `169-170` 行后。

建议添加：

```python
        # 是否把真实随机化参数拼到 critic / teacher 的 privileged obs 中
        # 第一阶段保持 False，避免修改 obs 维度后影响 PPO/HIM/CTS 接口
        add_domain_rand_privileged_obs = False
```

**位置 3：任务配置覆盖**

当前 GO1 任务位置：

```text
legged_gym/envs/him_go1/him_go1_config.py:58-59
class domain_rand(LeggedRobotCfg.domain_rand):
    pass
```

后续如果要让 HIM GO1 使用随机化，把 `pass` 替换成任务级范围，例如：

```python
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.4]
        randomize_payload_mass = True
        payload_mass_range = [-0.5, 1.0]
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        push_robots = True
        push_interval_s = 6.0
        max_push_vel_xy = 0.3
```

当前 `him_go2w_config.py` 和 `cts_go2w_config.py` 是 0 行空文件。它们不应该先复制整套环境实现，而应该先创建 config 类并只覆盖 `domain_rand`、`asset`、`control`、`env` 等任务差异。

### 4.5.2 `legged_gym/envs/base/legged_robot.py`

**位置 1：创建期 buffer 初始化**

当前位置：`_create_envs()` 中已经加载 asset，并设置：

```text
legged_gym/envs/base/legged_robot.py:425-435
robot_asset = self.gym.load_asset(...)
self.num_dof = self.gym.get_asset_dof_count(robot_asset)
self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
...
self.num_bodies = len(body_names)
self.num_dofs = len(self.dof_names)
```

具体添加位置：在 `self.num_dofs = len(self.dof_names)` 后面、`feet_names = ...` 前面添加：

```python
            self._init_domain_rand_creation_buffers()
```

原因：`_process_rigid_shape_props()`、`_process_dof_props()`、`_process_rigid_body_props()` 会在 `_create_envs()` 的 for-loop 里被调用，它们使用的 `friction_coeffs`、`payload_mass`、`joint_friction_coeffs` 等必须在 for-loop 之前存在。

**位置 2：新增创建期 buffer 函数**

建议添加位置：`_process_rigid_body_props()` 结束后、`_create_envs()` 开始前。当前相邻位置是：

```text
legged_gym/envs/base/legged_robot.py:347-395   def _process_rigid_body_props(...)
legged_gym/envs/base/legged_robot.py:398       def _create_envs(self):
```

也就是在当前 `return props` 后、`def _create_envs(self):` 前添加：

```python
    def _init_domain_rand_creation_buffers(self):
        self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)

        self.payload_mass = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.link_mass_ratios = torch.ones(self.num_envs, self.num_bodies - 1, device=self.device, requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self.joint_friction_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
        self.joint_damping_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
        self.joint_armatures = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
```

**位置 3：运行期 buffer 初始化**

当前位置：`_init_buffers()` 末尾设置默认关节位置：

```text
legged_gym/envs/base/legged_robot.py:94-111
self.default_dof_pos = torch.zeros(...)
...
self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
```

具体添加位置：在 `self.default_dof_pos = self.default_dof_pos.unsqueeze(0)` 后面添加：

```python
        self._init_domain_rand_runtime_buffers()
```

新增函数建议放在 `_init_domain_rand_creation_buffers()` 后面：

```python
    def _init_domain_rand_runtime_buffers(self):
        self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        self.motor_strength_multiplier = torch.ones(self.num_envs, self.num_actions, device=self.device, requires_grad=False)

        self.rand_push_force = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self._init_latency_buffers()
        self._reset_latency_buffers(torch.arange(self.num_envs, device=self.device))
```

注意：不要在 `_init_domain_rand_runtime_buffers()` 里重新清空 `payload_mass`、`link_mass_ratios`、`com_displacements`、`friction_coeffs`。这些是在 env 创建期采样并写入 Isaac Gym props 的真实值，后续如果要给 privileged obs 或日志使用，不能被覆盖掉。

**位置 4：新增 latency buffer 函数**

添加位置：仍放在 `_init_domain_rand_runtime_buffers()` 后面。

需要新增这 4 个函数：

```text
_init_latency_buffers()
_reset_latency_buffers(env_ids)
_get_delayed_actions()
_update_obs_latency_buffers()
```

调用关系必须是：

```text
_init_buffers()
  -> _init_domain_rand_runtime_buffers()
     -> _init_latency_buffers()
     -> _reset_latency_buffers(all envs)

reset_idx(env_ids)
  -> _resample_domain_rand(env_ids)
     -> _reset_latency_buffers(env_ids)

step(actions)
  -> for decimation:
       actions_for_torque = _get_delayed_actions()
       _compute_torques(actions_for_torque)
       simulate
       refresh_dof_state_tensor
       _update_obs_latency_buffers()

compute_observations()
  -> _get_imu_obs()
  -> _get_motor_obs()
```

**位置 5：reset 时重采样运行期随机量**

当前位置：`reset_idx()` 中 reset robot states 后、重采样 command 前：

```text
legged_gym/envs/base/legged_robot.py:647-652
self._reset_dofs(env_ids)
self._reset_root_states(env_ids)

self._resample_commands(env_ids)
```

具体添加位置：在 `_reset_root_states(env_ids)` 后、`_resample_commands(env_ids)` 前添加：

```python
            self._resample_domain_rand(env_ids)
```

新增函数建议放在 runtime buffer 函数后面：

```python
    def _resample_domain_rand(self, env_ids):
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        cfg = self.cfg.domain_rand

        if cfg.randomize_pd_gains:
            self.p_gains_multiplier[env_ids, :] = torch_rand_float(
                cfg.stiffness_multiplier_range[0], cfg.stiffness_multiplier_range[1],
                (n, self.num_actions), device=self.device)
            self.d_gains_multiplier[env_ids, :] = torch_rand_float(
                cfg.damping_multiplier_range[0], cfg.damping_multiplier_range[1],
                (n, self.num_actions), device=self.device)
        else:
            self.p_gains_multiplier[env_ids, :] = 1.0
            self.d_gains_multiplier[env_ids, :] = 1.0

        if cfg.randomize_motor_zero_offset:
            self.motor_zero_offsets[env_ids, :] = torch_rand_float(
                cfg.motor_zero_offset_range[0], cfg.motor_zero_offset_range[1],
                (n, self.num_actions), device=self.device)
        else:
            self.motor_zero_offsets[env_ids, :] = 0.0

        if cfg.randomize_motor_strength:
            self.motor_strength_multiplier[env_ids, :] = torch_rand_float(
                cfg.motor_strength_range[0], cfg.motor_strength_range[1],
                (n, self.num_actions), device=self.device)
        else:
            self.motor_strength_multiplier[env_ids, :] = 1.0

        self._reset_latency_buffers(env_ids)
```

**位置 6：step 中接入 action latency 和 obs latency**

当前位置：`step()` decimation loop：

```text
legged_gym/envs/base/legged_robot.py:738-744
for _ in range(self.cfg.control.decimation):
    self.torques = self._compute_torques(self.actions).view(self.torques.shape)
    self.gym.set_dof_actuation_force_tensor(...)
    self.gym.simulate(self.sim)
    ...
    self.gym.refresh_dof_state_tensor(self.sim)
```

具体替换为：

```python
            for _ in range(self.cfg.control.decimation):
                actions_for_torque = self._get_delayed_actions()
                self.torques = self._compute_torques(actions_for_torque).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
                self._update_obs_latency_buffers()
```

这里 `_get_delayed_actions()` 返回 raw action，不要返回已经乘过 `action_scale` 的 action；因为当前 `_compute_torques()` 里面会乘 `self.cfg.control.action_scale`。

**位置 7：替换 `_compute_torques()` 主体**

当前位置：

```text
legged_gym/envs/base/legged_robot.py:804-826
def _compute_torques(self, actions):
    ...
```

具体替换函数内部 torque 计算，不需要改函数名和调用者：

```python
            actions_scaled = actions * self.cfg.control.action_scale
            control_type = self.cfg.control.control_type
            p_gains = self.p_gains.unsqueeze(0) * self.p_gains_multiplier
            d_gains = self.d_gains.unsqueeze(0) * self.d_gains_multiplier

            if control_type == "P":
                target_pos = actions_scaled + self.default_dof_pos + self.motor_zero_offsets
                torques = p_gains * (target_pos - self.dof_pos) - d_gains * self.dof_vel
            elif control_type == "V":
                torques = p_gains * (actions_scaled - self.dof_vel) \
                    - d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
            elif control_type == "T":
                torques = actions_scaled
            else:
                raise NameError(f"Unknown controller type: {control_type}")

            torques = torques * self.motor_strength_multiplier
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

**位置 8：compute_observations 中接入延迟观测并修 `heights`**

当前位置：

```text
legged_gym/envs/base/legged_robot.py:828-858
def compute_observations(self):
    actor_obs = torch.cat((self.base_ang_vel..., self.projected_gravity, ... self.dof_pos, self.dof_vel ...))
```

具体修改：在函数一开始先添加：

```python
        heights = None
        base_ang_vel_obs, projected_gravity_obs = self._get_imu_obs()
        dof_pos_obs, dof_vel_obs = self._get_motor_obs()
```

然后把 `actor_obs` 里的三处来源替换掉：

```python
        actor_obs = torch.cat((
            base_ang_vel_obs,
            projected_gravity_obs,
            self.commands[:, :3] * self.commands_scale,
            dof_pos_obs,
            dof_vel_obs,
            self.actions,
        ), dim=-1)
```

这样同时解决两个问题：

1. `measure_heights=False` 时 `heights` 未定义。
2. motor/IMU latency 有统一接入口，而不是在 obs 拼接处写分支。

**位置 9：push 使用 angular velocity 配置**

当前位置：

```text
legged_gym/envs/base/legged_robot.py:555-560
def _push_robots(self):
    max_vel = self.cfg.domain_rand.max_push_vel_xy
    self.root_states[:, 7:9] = ...
```

建议补上角速度扰动，因为 config 里已经有 `max_push_ang_vel`：

```python
            max_vel = self.cfg.domain_rand.max_push_vel_xy
            max_ang_vel = self.cfg.domain_rand.max_push_ang_vel
            self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)
            self.root_states[:, 10:13] = torch_rand_float(-max_ang_vel, max_ang_vel, (self.num_envs, 3), device=self.device)
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
```

### 4.5.3 `legged_gym/envs/__init__.py`

当前只有一个注册：

```text
legged_gym/envs/__init__.py:9
task_registry.register("legged_gym_go1", LeggedRobot, GO1RoughCfg, GO1RoughCfgPPO)
```

如果只是给当前 `legged_gym_go1` 增加域随机化，不需要改这里。只有新增 GO2W / HIM / CTS 任务时才改。

GO2W 任务补齐后的推荐注册格式：

```python
from .go2w.go2w_config import GO2WRoughCfg, GO2WRoughCfgPPO
from .go2w.go2w import Go2w

task_registry.register("go2w", Go2w, GO2WRoughCfg, GO2WRoughCfgPPO)
```

如果 `Go2w` 只是继承 `LeggedRobot` 且不改 obs/torque，可以直接注册 `LeggedRobot`。如果要轮关节速度控制或 wheel-specific obs，才新建 `Go2w(LeggedRobot)`。

### 4.5.4 最小落地顺序

如果只想最快跑通，不要一次加完。按这个顺序最稳：

| 顺序 | 文件 | 精确动作 | 完成标志 |
| --- | --- | --- | --- |
| 1 | `legged_robot_config.py` | 补 `randomize_calculated_torque/torque_multiplier_range` 或统一成 `randomize_motor_strength` | 打开 config 不会 AttributeError。 |
| 2 | `legged_robot.py::_create_envs()` | 在 `self.num_dofs = len(self.dof_names)` 后调用 `_init_domain_rand_creation_buffers()` | props hook 使用的 tensor 已存在。 |
| 3 | `legged_robot.py` | 在 `_process_rigid_body_props()` 后添加 `_init_domain_rand_creation_buffers()` | mass/friction/com 记录不再缺失。 |
| 4 | `legged_robot.py::_init_buffers()` | 在 `self.default_dof_pos = self.default_dof_pos.unsqueeze(0)` 后调用 `_init_domain_rand_runtime_buffers()` | PD/zero/strength/latency buffer 已存在。 |
| 5 | `legged_robot.py::reset_idx()` | 在 `_reset_root_states(env_ids)` 后调用 `_resample_domain_rand(env_ids)` | 每个 episode 重新采样控制器和延迟。 |
| 6 | `legged_robot.py::step()` | decimation loop 中用 `_get_delayed_actions()`，刷新 DOF 后调用 `_update_obs_latency_buffers()` | action/obs latency 开关有实际效果。 |
| 7 | `legged_robot.py::_compute_torques()` | 使用 `p_gains_multiplier/d_gains_multiplier/motor_zero_offsets/motor_strength_multiplier` | PD/零位/电机强度随机化有实际效果。 |
| 8 | `legged_robot.py::compute_observations()` | 开头设 `heights = None`，用 `_get_imu_obs()` / `_get_motor_obs()` | obs latency 接入且 `heights` 不再未定义。 |
| 9 | task config | 在具体 task 的 `class domain_rand` 中打开保守范围 | 任务层可控，不污染所有任务。 |

## 5. 推荐代码片段

以下代码是建议片段，不是当前仓库已应用修改。

### 5.1 config：统一 domain_rand 字段

推荐在 `legged_gym/envs/base/legged_robot_config.py::LeggedRobotCfg.domain_rand` 中补齐并统一字段：

```python
class domain_rand:
    # dynamics
    randomize_payload_mass = False
    payload_mass_range = [-2.5, 2.5]

    randomize_link_mass = False
    link_mass_range = [0.9, 1.1]

    randomize_com_displacement = False
    com_displacement_range = [-0.05, 0.05]

    randomize_joint_friction = False
    joint_friction_range = [0.01, 1.15]

    randomize_joint_damping = False
    joint_damping_range = [0.3, 1.5]

    randomize_joint_armature = False
    joint_armature_range = [0.0001, 0.05]

    # contact
    randomize_friction = False
    friction_range = [0.2, 1.3]

    randomize_restitution = False
    restitution_range = [0.0, 0.4]

    # perturbation
    push_robots = False
    push_interval_s = 4.0
    max_push_vel_xy = 0.4
    max_push_ang_vel = 0.6

    continuous_push = False
    push_interval_s = 4.0
    max_push_force = 30.0
    max_push_torque = 5.0

    # actuator / controller
    randomize_pd_gains = False
    stiffness_multiplier_range = [0.8, 1.2]
    damping_multiplier_range = [0.8, 1.2]

    randomize_motor_zero_offset = False
    motor_zero_offset_range = [-0.035, 0.035]

    randomize_motor_strength = False
    motor_strength_range = [0.8, 1.2]

    # latency; units are physics sim steps, not policy steps
    add_obs_latency = False
    randomize_obs_motor_latency = False
    range_obs_motor_latency = [1, 3]
    randomize_obs_imu_latency = False
    range_obs_imu_latency = [1, 3]

    add_cmd_action_latency = False
    randomize_cmd_action_latency = False
    range_cmd_action_latency = [1, 3]

    # privileged domain info, disabled by default
    add_domain_rand_privileged_obs = False
```

如果你暂时想最小改动，也可以保留现有 `randomize_calculated_torque`，但需要补：

```python
randomize_calculated_torque = False
torque_multiplier_range = [0.8, 1.2]
```

长期建议不要同时保留 `randomize_calculated_torque` 和 `randomize_motor_strength` 两套含义接近的字段。

### 5.2 初始化所有域随机化 buffer

推荐在 `LeggedRobot._init_buffers()` 中完成基础状态、`default_dof_pos`、`p_gains`、`d_gains` 初始化后调用：

```python
self._init_domain_rand_buffers()
```

新增函数建议如下：

```python
def _init_domain_rand_buffers(self):
    self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
    self.restitution_coeffs = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)

    self.payload_mass = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
    self.link_mass_ratios = torch.ones(self.num_envs, self.num_bodies - 1, device=self.device, requires_grad=False)
    self.com_displacements = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    self.joint_friction_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
    self.joint_damping_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
    self.joint_armatures = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)

    self.p_gains_multiplier = torch.ones(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
    self.d_gains_multiplier = torch.ones(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
    self.motor_zero_offsets = torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
    self.motor_strength_multiplier = torch.ones(self.num_envs, self.num_actions, device=self.device, requires_grad=False)

    self.rand_push_force = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    self._init_latency_buffers()
```

注意：`_process_dof_props()` 在 `_create_envs()` 阶段调用，早于 `_init_buffers()`。如果你要在 `_process_dof_props()` 中写 `joint_friction_coeffs` 等 tensor，就不能只在 `_init_buffers()` 初始化它们。更稳的方案是：

1. 在 `_create_envs()` 加载 asset、知道 `num_bodies` / `num_dof` / `num_actions` 后，先调用 `_init_domain_rand_creation_buffers()` 初始化创建期要用的 buffer。
2. `_init_buffers()` 再初始化运行期 buffer。

最小改动也可以在 `_process_dof_props()` 里 `env_id == 0` 时懒初始化这些 tensor，但长期可读性不如显式初始化。

### 5.3 创建期 buffer：给 props hook 使用

建议在 `_create_envs()` 获取 `self.num_dof`、`self.num_bodies` 后，进入 for env loop 前调用：

```python
def _init_domain_rand_creation_buffers(self):
    self.friction_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
    self.restitution_coeffs = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)

    self.payload_mass = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
    self.link_mass_ratios = torch.ones(self.num_envs, self.num_bodies - 1, device=self.device, requires_grad=False)
    self.com_displacements = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

    self.joint_friction_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
    self.joint_damping_coeffs = torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
    self.joint_armatures = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
```

然后在 `_create_envs()` 中：

```python
self.num_dof = self.gym.get_asset_dof_count(robot_asset)
self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
self._init_domain_rand_creation_buffers()
```

这样 `_process_rigid_shape_props()`、`_process_dof_props()`、`_process_rigid_body_props()` 中写 tensor 就不会踩未初始化变量。

### 5.4 reset 时重采样运行期随机量

在 `reset_idx()` 中，建议在 `_reset_dofs(env_ids)` 和 `_reset_root_states(env_ids)` 之后、清空 last buffers 之前调用：

```python
self._resample_domain_rand(env_ids)
```

参考实现：

```python
def _resample_domain_rand(self, env_ids):
    if len(env_ids) == 0:
        return

    n = len(env_ids)
    cfg = self.cfg.domain_rand

    if cfg.randomize_pd_gains:
        self.p_gains_multiplier[env_ids, :] = torch_rand_float(
            cfg.stiffness_multiplier_range[0], cfg.stiffness_multiplier_range[1],
            (n, self.num_actions), device=self.device)
        self.d_gains_multiplier[env_ids, :] = torch_rand_float(
            cfg.damping_multiplier_range[0], cfg.damping_multiplier_range[1],
            (n, self.num_actions), device=self.device)
    else:
        self.p_gains_multiplier[env_ids, :] = 1.0
        self.d_gains_multiplier[env_ids, :] = 1.0

    if cfg.randomize_motor_zero_offset:
        self.motor_zero_offsets[env_ids, :] = torch_rand_float(
            cfg.motor_zero_offset_range[0], cfg.motor_zero_offset_range[1],
            (n, self.num_actions), device=self.device)
    else:
        self.motor_zero_offsets[env_ids, :] = 0.0

    if cfg.randomize_motor_strength:
        self.motor_strength_multiplier[env_ids, :] = torch_rand_float(
            cfg.motor_strength_range[0], cfg.motor_strength_range[1],
            (n, self.num_actions), device=self.device)
    else:
        self.motor_strength_multiplier[env_ids, :] = 1.0

    self._reset_latency_buffers(env_ids)
```

如果确实希望 friction/restitution 每个 episode 都换，参考 HIMLoco 的 `refresh_actor_rigid_shape_props(env_ids)`，在 `_resample_domain_rand()` 后调用。注意它需要逐 env 调 Isaac Gym API，成本比只改 tensor 高。

### 5.5 torque 计算：让 PD、电机零位、电机强度生效

当前 2027 的 `_compute_torques()` 只使用 `self.p_gains` 和 `self.d_gains`，没有使用 multiplier。建议改成：

```python
def _compute_torques(self, actions):
    actions_scaled = actions * self.cfg.control.action_scale
    control_type = self.cfg.control.control_type

    p_gains = self.p_gains.unsqueeze(0) * self.p_gains_multiplier
    d_gains = self.d_gains.unsqueeze(0) * self.d_gains_multiplier

    if control_type == "P":
        target_pos = actions_scaled + self.default_dof_pos + self.motor_zero_offsets
        torques = p_gains * (target_pos - self.dof_pos) - d_gains * self.dof_vel
    elif control_type == "V":
        torques = p_gains * (actions_scaled - self.dof_vel) \
            - d_gains * (self.dof_vel - self.last_dof_vel) / self.sim_params.dt
    elif control_type == "T":
        torques = actions_scaled
    else:
        raise NameError(f"Unknown controller type: {control_type}")

    torques = torques * self.motor_strength_multiplier
    return torch.clip(torques, -self.torque_limits, self.torque_limits)
```

如果 GO2W 有轮关节速度控制，建议 GO2W 子类只 override `_compute_torques()` 的“轮腿混合控制”部分，但继续复用同名 `p_gains_multiplier`、`d_gains_multiplier`、`motor_zero_offsets`、`motor_strength_multiplier`。

### 5.6 action latency：不要双重 action_scale

2027 当前 `_compute_torques()` 内部会 `actions * action_scale`。因此建议 action latency buffer 存 **raw clipped action**，不要存 scaled action：

```python
def _init_latency_buffers(self):
    cfg = self.cfg.domain_rand
    max_cmd_delay = cfg.range_cmd_action_latency[1]
    max_motor_delay = cfg.range_obs_motor_latency[1]
    max_imu_delay = cfg.range_obs_imu_latency[1]

    self.cmd_action_latency_buffer = torch.zeros(
        self.num_envs, self.num_actions, max_cmd_delay + 1,
        device=self.device, requires_grad=False)
    self.obs_motor_latency_buffer = torch.zeros(
        self.num_envs, self.num_actions * 2, max_motor_delay + 1,
        device=self.device, requires_grad=False)
    self.obs_imu_latency_buffer = torch.zeros(
        self.num_envs, 6, max_imu_delay + 1,
        device=self.device, requires_grad=False)

    self.cmd_action_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.obs_motor_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.obs_imu_latency_simstep = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
```

```python
def _reset_latency_buffers(self, env_ids):
    cfg = self.cfg.domain_rand

    if cfg.add_cmd_action_latency:
        self.cmd_action_latency_buffer[env_ids] = 0.0
        if cfg.randomize_cmd_action_latency:
            self.cmd_action_latency_simstep[env_ids] = torch.randint(
                cfg.range_cmd_action_latency[0], cfg.range_cmd_action_latency[1] + 1,
                (len(env_ids),), device=self.device)
        else:
            self.cmd_action_latency_simstep[env_ids] = cfg.range_cmd_action_latency[1]

    if cfg.add_obs_latency:
        self.obs_motor_latency_buffer[env_ids] = 0.0
        self.obs_imu_latency_buffer[env_ids] = 0.0
        if cfg.randomize_obs_motor_latency:
            self.obs_motor_latency_simstep[env_ids] = torch.randint(
                cfg.range_obs_motor_latency[0], cfg.range_obs_motor_latency[1] + 1,
                (len(env_ids),), device=self.device)
        else:
            self.obs_motor_latency_simstep[env_ids] = cfg.range_obs_motor_latency[1]
        if cfg.randomize_obs_imu_latency:
            self.obs_imu_latency_simstep[env_ids] = torch.randint(
                cfg.range_obs_imu_latency[0], cfg.range_obs_imu_latency[1] + 1,
                (len(env_ids),), device=self.device)
        else:
            self.obs_imu_latency_simstep[env_ids] = cfg.range_obs_imu_latency[1]
```

```python
def _get_delayed_actions(self):
    cfg = self.cfg.domain_rand
    if not cfg.add_cmd_action_latency:
        return self.actions

    max_delay = cfg.range_cmd_action_latency[1]
    self.cmd_action_latency_buffer[:, :, 1:] = self.cmd_action_latency_buffer[:, :, :max_delay].clone()
    self.cmd_action_latency_buffer[:, :, 0] = self.actions.clone()
    return self.cmd_action_latency_buffer[
        torch.arange(self.num_envs, device=self.device),
        :,
        self.cmd_action_latency_simstep.long(),
    ]
```

`step()` 中只替换 torque 输入：

```python
for _ in range(self.cfg.control.decimation):
    actions_for_torque = self._get_delayed_actions()
    self.torques = self._compute_torques(actions_for_torque).view(self.torques.shape)
    self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
    self.gym.simulate(self.sim)
    if self.device == "cpu":
        self.gym.fetch_results(self.sim, True)
    self.gym.refresh_dof_state_tensor(self.sim)
    self._update_obs_latency_buffers()
```

### 5.7 observation latency：集中封装 motor / IMU 观测

2027 当前 actor obs 是：base angular velocity、projected gravity、commands、dof position error、dof velocity、actions。为保持 obs layout 不大改，建议只把 motor/IMU 来源替换成函数：

```python
def _update_obs_latency_buffers(self):
    cfg = self.cfg.domain_rand
    if not cfg.add_obs_latency:
        return

    if cfg.randomize_obs_motor_latency:
        max_delay = cfg.range_obs_motor_latency[1]
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        self.obs_motor_latency_buffer[:, :, 1:] = self.obs_motor_latency_buffer[:, :, :max_delay].clone()
        self.obs_motor_latency_buffer[:, :, 0] = torch.cat((q, dq), dim=1).clone()

    if cfg.randomize_obs_imu_latency:
        max_delay = cfg.range_obs_imu_latency[1]
        imu = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
        ), dim=1)
        self.obs_imu_latency_buffer[:, :, 1:] = self.obs_imu_latency_buffer[:, :, :max_delay].clone()
        self.obs_imu_latency_buffer[:, :, 0] = imu.clone()
```

```python
def _get_motor_obs(self):
    cfg = self.cfg.domain_rand
    if cfg.add_obs_latency and cfg.randomize_obs_motor_latency:
        motor_obs = self.obs_motor_latency_buffer[
            torch.arange(self.num_envs, device=self.device),
            :,
            self.obs_motor_latency_simstep.long(),
        ]
        dof_pos_obs = motor_obs[:, :self.num_actions]
        dof_vel_obs = motor_obs[:, self.num_actions:]
    else:
        dof_pos_obs = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_obs = self.dof_vel * self.obs_scales.dof_vel
    return dof_pos_obs, dof_vel_obs

def _get_imu_obs(self):
    cfg = self.cfg.domain_rand
    if cfg.add_obs_latency and cfg.randomize_obs_imu_latency:
        imu_obs = self.obs_imu_latency_buffer[
            torch.arange(self.num_envs, device=self.device),
            :,
            self.obs_imu_latency_simstep.long(),
        ]
        base_ang_vel_obs = imu_obs[:, :3]
        projected_gravity_obs = imu_obs[:, 3:6]
    else:
        base_ang_vel_obs = self.base_ang_vel * self.obs_scales.ang_vel
        projected_gravity_obs = self.projected_gravity
    return base_ang_vel_obs, projected_gravity_obs
```

然后 `compute_observations()` 改为：

```python
def compute_observations(self):
    heights = None
    base_ang_vel_obs, projected_gravity_obs = self._get_imu_obs()
    dof_pos_obs, dof_vel_obs = self._get_motor_obs()

    actor_obs = torch.cat((
        base_ang_vel_obs,
        projected_gravity_obs,
        self.commands[:, :3] * self.commands_scale,
        dof_pos_obs,
        dof_vel_obs,
        self.actions,
    ), dim=-1)

    if self.cfg.terrain.measure_heights:
        heights = torch.clip(
            self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
            -1, 1,
        ) * self.obs_scales.height_measurements
        self.obs_buf = torch.cat((actor_obs, heights), dim=-1)
    else:
        self.obs_buf = actor_obs

    if self.privileged_obs_buf is not None:
        critic_parts = [actor_obs, self.base_lin_vel * self.obs_scales.lin_vel]
        if heights is not None:
            critic_parts.append(heights)
        if self.cfg.domain_rand.add_domain_rand_privileged_obs:
            critic_parts.append(self._get_domain_rand_privileged_obs())
        self.privileged_obs_buf = torch.cat(critic_parts, dim=-1)

    if self.add_noise:
        self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
```

### 5.8 privileged domain rand vector

如果未来要做 teacher critic、HIM estimator 或 adaptation，可以把真实随机化参数给 critic，但默认不放 actor：

```python
def _get_domain_rand_privileged_obs(self):
    return torch.cat((
        self.friction_coeffs,
        self.restitution_coeffs,
        self.payload_mass,
        self.com_displacements,
        self.p_gains_multiplier.mean(dim=1, keepdim=True),
        self.d_gains_multiplier.mean(dim=1, keepdim=True),
        self.motor_strength_multiplier.mean(dim=1, keepdim=True),
        self.cmd_action_latency_simstep.float().unsqueeze(1),
        self.obs_motor_latency_simstep.float().unsqueeze(1),
        self.obs_imu_latency_simstep.float().unsqueeze(1),
    ), dim=1)
```

如果使用这个函数，必须同步修改：

```text
LeggedRobotCfg.env.num_privileged_obs
HIM / CTS storage 和 runner 对 critic obs 维度的假设
```

所以建议第一阶段不要打开 `add_domain_rand_privileged_obs`，等 base 随机化稳定后再加。

### 5.9 friction / restitution 每 episode 刷新

如果希望像 HIMLoco 一样每个 reset 都换 friction/restitution，可以加：

```python
def refresh_actor_rigid_shape_props(self, env_ids):
    cfg = self.cfg.domain_rand
    if not (cfg.randomize_friction or cfg.randomize_restitution):
        return

    if cfg.randomize_friction:
        self.friction_coeffs[env_ids] = torch_rand_float(
            cfg.friction_range[0], cfg.friction_range[1],
            (len(env_ids), 1), device=self.device)
    if cfg.randomize_restitution:
        self.restitution_coeffs[env_ids] = torch_rand_float(
            cfg.restitution_range[0], cfg.restitution_range[1],
            (len(env_ids), 1), device=self.device)

    for env_id in env_ids.tolist():
        props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id])
        for shape in props:
            if cfg.randomize_friction:
                shape.friction = self.friction_coeffs[env_id, 0].item()
            if cfg.randomize_restitution:
                shape.restitution = self.restitution_coeffs[env_id, 0].item()
        self.gym.set_actor_rigid_shape_properties(self.envs[env_id], self.actor_handles[env_id], props)
```

注意：当前 2027 `_create_envs()` 用的是 `actor_handles` 列表，不能像 HIMLoco 那样硬编码 actor id `0`，否则多 actor 场景会出错。

## 6. 建议默认随机化范围

下面是适合 2027 先跑通的保守范围。等 baseline 稳定后再扩大。

| 参数 | 第一阶段建议 | 第二阶段建议 | 说明 |
| --- | --- | --- | --- |
| `friction_range` | `[0.6, 1.4]` | `[0.3, 1.8]` | 先覆盖常见地面差异，不要一开始极端低摩擦。 |
| `restitution_range` | `[0.0, 0.2]` | `[0.0, 0.5]` | 腿足接触中过大 restitution 容易引入不真实弹跳。 |
| `payload_mass_range` | `[-0.5, 1.5]` | `[-1.0, 2.5]` | 如果 GO2W 上有额外电池/传感器，正 payload 更重要。 |
| `link_mass_range` | `[0.95, 1.05]` | `[0.9, 1.1]` | 先小范围，避免动力学过散。 |
| `com_displacement_range` | `[-0.02, 0.02]` | `[-0.05, 0.05]` | 轮腿平台 COM 偏移会明显影响稳定性。 |
| `joint_friction_range` | `[0.5, 1.5]` | `[0.1, 2.0]` | Isaac Gym DOF friction 数值对不同 URDF 敏感，先保守。 |
| `joint_damping_range` | `[0.7, 1.3]` | `[0.3, 1.5]` | 阻尼过散会影响 gait 学习。 |
| `joint_armature_range` | `[0.0, 0.02]` | `[0.0001, 0.05]` | 先按小惯量偏差处理。 |
| `stiffness_multiplier_range` | `[0.9, 1.1]` | `[0.8, 1.2]` | 适合 sim2real 的 PD mismatch。 |
| `damping_multiplier_range` | `[0.9, 1.1]` | `[0.8, 1.2]` | 同上。 |
| `motor_zero_offset_range` | `[-0.02, 0.02]` | `[-0.035, 0.035]` | 先小于 My_unitree 的范围。 |
| `motor_strength_range` | `[0.9, 1.1]` | `[0.8, 1.2]` | 直接影响 torque。 |
| `range_cmd_action_latency` | `[0, 2]` sim steps | `[1, 4]` sim steps | 明确单位是 physics step，不是 policy step。 |
| `range_obs_motor_latency` | `[0, 2]` sim steps | `[1, 4]` sim steps | 需要和真实控制频率对齐。 |
| `range_obs_imu_latency` | `[0, 1]` sim steps | `[1, 3]` sim steps | IMU 通常比低层电机链路更快。 |
| `max_push_vel_xy` | `0.3` | `0.6` | 先训练恢复能力，不要把任务变成抗强冲击。 |
| `max_push_ang_vel` | `0.4` | `1.0` | 对轮腿 yaw/roll 恢复更敏感。 |

## 7. 不同任务应该如何打开

### 7.1 原生 GO1 / 基础 locomotion

先只开接触和轻量动力学：

```python
class domain_rand(LeggedRobotCfg.domain_rand):
    randomize_friction = True
    friction_range = [0.6, 1.4]
    randomize_restitution = True
    restitution_range = [0.0, 0.2]
    randomize_payload_mass = True
    payload_mass_range = [-0.5, 1.0]
    push_robots = True
    push_interval_s = 6.0
    max_push_vel_xy = 0.3
```

### 7.2 GO2W / 轮腿任务

GO2W 更需要 actuator 和 latency：

```python
class domain_rand(LeggedRobotCfg.domain_rand):
    randomize_friction = True
    friction_range = [0.6, 1.4]
    randomize_payload_mass = True
    payload_mass_range = [-0.5, 1.5]
    randomize_com_displacement = True
    com_displacement_range = [-0.03, 0.03]

    randomize_pd_gains = True
    stiffness_multiplier_range = [0.9, 1.1]
    damping_multiplier_range = [0.9, 1.1]
    randomize_motor_zero_offset = True
    motor_zero_offset_range = [-0.02, 0.02]
    randomize_motor_strength = True
    motor_strength_range = [0.9, 1.1]

    add_cmd_action_latency = True
    randomize_cmd_action_latency = True
    range_cmd_action_latency = [0, 2]
    add_obs_latency = True
    randomize_obs_motor_latency = True
    range_obs_motor_latency = [0, 2]
    randomize_obs_imu_latency = True
    range_obs_imu_latency = [0, 1]
```

### 7.3 HIM / estimator 路线

HIM 路线不要把真实随机化参数直接给 actor。推荐：

1. actor 使用历史 obs。
2. critic 可以看 base lin vel、height、contact、可选 domain rand vector。
3. estimator 从历史里学 hidden dynamics。
4. latency 和 motor strength 要打开，否则 estimator 学不到真实部署中最关键的 actuator mismatch。

### 7.4 CTS / MoE 路线

CTS/MoE 通常会有 encoder 或 expert routing。建议：

1. 仍复用 base domain rand。
2. 如果 encoder 需要 domain context，只给 teacher/critic 或 privileged encoder，不直接给部署 actor。
3. 记录每个 episode 的随机化参数，方便按摩擦、payload、latency 分桶评估 expert 行为。

## 8. 验证清单

实现时建议按以下顺序验证，不要一次性全开：

1. **全部开关关闭**：小 env 数启动，obs/action/reward shape 与修改前一致。
2. **单开 friction**：打印 `friction_coeffs.min()/max()`，确认每 env 不同且 Isaac Gym props 已设置。
3. **单开 payload/link/COM**：确认 `body_props` 质量没有出现负数，`recomputeInertia=True`。
4. **单开 PD gain / motor strength**：确认 `_compute_torques()` 输出变化，但无 NaN/Inf。
5. **单开 action latency**：确认 delayed action 与 raw action 不同，并且没有重复乘 `action_scale`。
6. **单开 obs latency**：确认 `obs_motor_latency_buffer`、`obs_imu_latency_buffer` shape 正确，actor obs 维度不变。
7. **打开 push**：确认 push interval 单位是 policy step 换算后的 sim/control step，机器人不会每步都被推。
8. **组合打开保守范围**：训练 100-500 iteration，观察 fall rate、episode length、tracking reward 是否崩溃。
9. **分桶评估**：按 friction、payload、latency、motor strength 分桶统计 tracking、fall、torque、action rate。

建议在训练日志里至少记录这些标量：

```text
domain_rand/friction_mean
domain_rand/payload_mass_mean
domain_rand/motor_strength_mean
domain_rand/cmd_latency_mean
domain_rand/obs_motor_latency_mean
domain_rand/obs_imu_latency_mean
```

## 9. 最终推荐路线

最终应该这么做：

1. **不要从 HIMLoco 或 My_unitree 直接复制一个完整 `legged_robot.py` 覆盖 2027**。2027 已经有自己的 base 文件、HIM/CTS/rsl_rl 融合方向和中文注释结构，直接覆盖会破坏现有算法接口。
2. **以 2027 当前 `LeggedRobotCfg.domain_rand` 为主线补闭环**，从 My_unitree 迁移 latency buffer 思路，从 HIMLoco 迁移 reset 重采样和 torque multiplier 真正生效的思路。
3. **把所有公共随机化函数放进 base `LeggedRobot`**，GO2W 只在必须处理轮关节 obs/torque layout 时写子类 override。
4. **先做可运行性修复，再开随机化范围**：初始化 buffer、补 config 字段、修 `heights`、修 torque multiplier，再打开 friction/payload/PD/latency。
5. **critic/teacher 可以接收 domain rand truth，actor 不要直接接收**。后续 HIM 或 CTS 应通过 history/estimator/encoder 学隐变量，这样部署时才成立。
6. **所有随机化参数都要有日志和分桶评估**，否则训练变差时无法判断是摩擦、质量、延迟还是电机随机化导致。

这条路线保留了 2027 自己项目的结构，同时吸收了两个开源项目最有价值的部分：HIMLoco 的 reset/torque 生效链路，以及 My_unitree 的真实部署 latency 建模。
