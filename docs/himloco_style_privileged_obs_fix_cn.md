# HIMLoco 风格观测与 privileged obs 修改方案

本文只写修改方案和参考代码，不表示仓库代码已经按这里修改。目标是说明：现在 `2027_RC_legged_robot` 里 `heights`、actor obs、privileged obs 应该怎么按 HIMLoco 的方式整理，避免 actor 被迫看到地形高度，同时保证 critic 的 privileged obs 维度正确。

# **结论先说：**

- 现在不要再给 actor / critic 单独加很多开关，例如 `actor_observe_heights`、`critic_observe_heights` 这一类。

- HIMLoco 的做法更简单：**先拼一个完整的 `current_obs`，然后用切片区分 actor 和 critic**。

- actor 只取前 `num_one_step_observations` 维。

- critic / privileged obs 取前 `num_one_step_privileged_obs` 维。

- 所以即使 `current_obs` 后面拼了 `base_lin_vel`、`heights`，只要 actor 切片只取前 45 维，actor 就不会看到这些 privileged 信息。

- 你当前项目里还没有真正接入 HIMLoco 那个 3 维 `disturbance` 物理量，所以第一版建议不要硬凑 `45 + 3 + 3 + 187 = 238`，而是先用：

    ```Python
    num_one_step_privileged_obs = 45 + 3 + 187
    num_privileged_obs = num_one_step_privileged_obs
    ```

- 等后面你真正把连续外力 / disturbance 放进物理仿真里，再升级成 HIMLoco 完整版本：

    ```Python
    num_one_step_privileged_obs = 45 + 3 + 3 + 187
    num_privileged_obs = num_one_step_privileged_obs
    ```

# **为什么 HIMLoco 没有很多参数：**

- HIMLoco 不是这样写：

    ```Python
    if actor_observe_heights:
        actor_obs = torch.cat((actor_obs, heights), dim=-1)

    if critic_observe_heights:
        critic_obs = torch.cat((critic_obs, heights), dim=-1)
    ```

- HIMLoco 是这样写：

    ```Python
    current_obs = torch.cat((actor_visible_obs, privileged_extra_obs), dim=-1)

    self.obs_buf = current_obs[:, :self.num_one_step_obs]
    self.privileged_obs_buf = current_obs[:, :self.num_one_step_privileged_obs]
    ```

- 也就是：

    - `current_obs` 可以包含很多信息；

    - actor 只拿前面一段；

    - critic 拿更长的一段；

    - 具体谁能看见什么，由维度顺序决定，而不是由很多布尔开关决定。

# **第一步：先整理 env 维度配置**

- 打开这个文件：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/base/legged_robot_config.py
    ```

- 找到：

    ```Python
    class LeggedRobotCfg(BaseConfig):
        class env:
    ```

- 建议改成下面这种形式：

    ```Python
    class env:
        num_envs = 4096

        # 单帧 actor 观测：不包含 base_lin_vel，不包含 heights
        num_one_step_observations = 45
        num_observations = num_one_step_observations

        # 第一版：actor_obs + base_lin_vel + heights
        # 45 + 3 + 187 = 235
        num_one_step_privileged_obs = num_one_step_observations + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs

        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20
    ```

- 这里的含义是：

    ```Text
    actor obs:
        base_ang_vel              3
        projected_gravity         3
        commands                  3
        dof_pos                  12
        dof_vel                  12
        last_actions             12
        total                    45

    privileged obs:
        actor_obs                45
        base_lin_vel              3
        terrain heights         187
        total                   235
    ```

- 注意：当前你项目中 `legged_robot_config.py` 已经接近这个方案了，但是建议把 `num_one_step_observations` 和 `num_one_step_privileged_obs` 显式写出来，这样后面如果加 history 或 HIM 算法，不容易乱。

# **第二步：决定 measure_heights 的使用方式**

- 如果你要让 critic 看 terrain heights，就必须让环境计算 heights：

    ```Python
    class terrain:
        measure_heights = True
    ```

- 这不代表 actor 一定会看到 heights。

- actor 是否看到 heights，取决于 `self.obs_buf` 最后切了多少维。

- 按本文方案，actor 只切前 45 维，所以 actor 不会看到 heights。

- 如果 `mesh_type = 'plane'`，你当前项目里的 `_get_heights()` 会直接返回全 0 高度，不会要求 `height_samples`：

    ```Python
    if self.cfg.terrain.mesh_type == 'plane':
        return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
    ```

- 所以平地阶段也可以先设置：

    ```Python
    mesh_type = 'plane'
    measure_heights = True
    ```

- 这样可以先把 privileged obs 的 235 维链路跑通。

- 如果你只是做最小 smoke test，不想要 heights，那么要同步改成：

    ```Python
    measure_heights = False
    num_one_step_privileged_obs = num_one_step_observations + 3
    num_privileged_obs = num_one_step_privileged_obs
    ```

- 也就是：

    ```Text
    45 actor obs + 3 base_lin_vel = 48
    ```

- 不要出现这种组合：

    ```Python
    measure_heights = False
    num_privileged_obs = 235
    ```

- 因为代码实际拼不出 235 维，会导致 privileged obs 维度不匹配。

# **第三步：在 LeggedRobot 初始化里记录 one-step 维度**

- 打开这个文件：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py
    ```

- 找到 `LeggedRobot.__init__`：

    ```Python
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        ...
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
    ```

- 在 `super().__init__(...)` 后面添加：

    ```Python
    self.num_one_step_obs = getattr(
        self.cfg.env,
        "num_one_step_observations",
        self.num_obs,
    )

    self.num_one_step_privileged_obs = getattr(
        self.cfg.env,
        "num_one_step_privileged_obs",
        self.num_privileged_obs,
    )
    ```

- 改完以后结构应该像这样：

    ```Python
    self._parse_cfg(self.cfg)
    super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

    self.num_one_step_obs = getattr(
        self.cfg.env,
        "num_one_step_observations",
        self.num_obs,
    )
    self.num_one_step_privileged_obs = getattr(
        self.cfg.env,
        "num_one_step_privileged_obs",
        self.num_privileged_obs,
    )

    if not self.headless:
        self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
    self._init_buffers()
    ```

- 这里为什么放在 `super().__init__` 后面：

    - `BaseTask.__init__` 会先设置 `self.num_obs` 和 `self.num_privileged_obs`；

    - 所以后面用 `self.num_obs` 做 fallback 更安全；

    - 同时这个时候 env 已经创建完，后续 `_init_buffers()` 和 `compute_observations()` 可以直接使用这些维度。

# **第四步：替换 compute_observations 的组织方式**

- 继续打开：

    ```Bash
    2027_RC_legged_robot/legged_gym/envs/base/legged_robot.py
    ```

- 找到：

    ```Python
    def compute_observations(self):
    ```

- 当前这里的问题是：

    - `heights` 只在 `measure_heights=True` 时定义；

    - 后面却直接判断 `if heights is not None`；

    - 当 `measure_heights=False` 时会有 `heights` 未定义的问题；

    - 另外当前代码在 `measure_heights=True` 时会把 heights 拼进 `self.obs_buf`，这样 actor 也会看到 heights，不符合你现在想要的 blind actor 方案。

- 建议把整个函数替换成下面这种结构：

    ```Python
    def compute_observations(self):
        """Computes actor observations and privileged critic observations.

        HIMLoco-style layout:
            current_obs = [actor_obs, base_lin_vel, heights]
            obs_buf = current_obs[:, :num_one_step_obs]
            privileged_obs_buf = current_obs[:, :num_one_step_privileged_obs]
        """
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

        if self.add_noise:
            actor_obs += (
                2 * torch.rand_like(actor_obs) - 1
            ) * self.noise_scale_vec[:self.num_one_step_obs]

        current_obs = torch.cat((
            actor_obs,
            self.base_lin_vel * self.obs_scales.lin_vel,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(
                self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                -1,
                1.,
            ) * self.obs_scales.height_measurements

            if self.add_noise:
                heights += (
                    2 * torch.rand_like(heights) - 1
                ) * self.cfg.noise.noise_scales.height_measurements \
                  * self.cfg.noise.noise_level \
                  * self.obs_scales.height_measurements

            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = current_obs[:, :self.num_one_step_obs]

        if self.privileged_obs_buf is not None:
            if current_obs.shape[1] != self.num_one_step_privileged_obs:
                raise RuntimeError(
                    f"privileged obs dim mismatch: "
                    f"got {current_obs.shape[1]}, "
                    f"expected {self.num_one_step_privileged_obs}. "
                    f"Check measure_heights and cfg.env.num_one_step_privileged_obs."
                )
            self.privileged_obs_buf = current_obs[:, :self.num_one_step_privileged_obs]
    ```

- 这段代码的关键点：

    - `actor_obs` 永远是 45 维；

    - `self.obs_buf` 永远只切前 45 维；

    - `heights` 可以存在于 `current_obs` 后半段；

    - critic 可以通过 `privileged_obs_buf` 看到 heights；

    - actor 不会看到 heights；

    - 如果配置维度和实际拼接维度对不上，会直接报清楚的错，不会静默训练坏数据。

# **第五步：为什么这里不用 noise_scale_vec[45:232]**

- 你当前项目的 actor obs 是 45 维，所以：

    ```Python
    self.noise_scale_vec = torch.zeros_like(self.obs_buf[0])
    ```

- 这里 `noise_scale_vec` 的长度就是 45。

- 如果 actor 不看 heights，就不要再依赖：

    ```Python
    self.noise_scale_vec[45:232]
    ```

- 因为这个切片对 45 维 actor obs 没有实际意义。

- 所以在上面的推荐代码里，actor 噪声走：

    ```Python
    self.noise_scale_vec[:self.num_one_step_obs]
    ```

- heights 噪声单独用标量配置：

    ```Python
    self.cfg.noise.noise_scales.height_measurements \
        * self.cfg.noise.noise_level \
        * self.obs_scales.height_measurements
    ```

- 这样结构更清楚，也不会要求 actor obs 必须包含 heights。

# **第六步：如果后面要完全对齐 HIMLoco 的 238 维**

- HIMLoco 的 privileged obs 是：

    ```Text
    actor_obs + base_lin_vel + disturbance + heights
    45        + 3            + 3           + 187 = 238
    ```

- 你当前项目里虽然已经有：

    ```Python
    self.rand_push_force = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    self.rand_push_torque = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
    ```

- 但是当前 `_push_robots()` 实际是直接改 `root_states` 的速度，并没有持续把 `rand_push_force` 施加到物理仿真里。

- 所以第一版不建议把 `rand_push_force` 直接当成 HIMLoco 的 `disturbance` 来凑维度。

- 如果后面你真的实现了连续外力，例如每一步通过 Isaac Gym force tensor 给 base 或 body 加外力，那么再把配置改成：

    ```Python
    num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
    num_privileged_obs = num_one_step_privileged_obs
    ```

- 然后在 `current_obs` 里改成：

    ```Python
    privileged_disturbance = self.rand_push_force

    current_obs = torch.cat((
        actor_obs,
        self.base_lin_vel * self.obs_scales.lin_vel,
        privileged_disturbance,
    ), dim=-1)

    if self.cfg.terrain.measure_heights:
        current_obs = torch.cat((current_obs, heights), dim=-1)
    ```

- 注意顺序不要乱，保持：

    ```Text
    [actor_obs, base_lin_vel, disturbance, heights]
    ```

- 这样后面如果你接 HIMLoco estimator，也能保持和它的 velocity target 约定一致。

# **第七步：训练前应该检查什么**

- 改完后先不要直接长时间训练，先做最小启动检查。

- 如果采用本文主线 235 维方案，配置应该满足：

    ```Python
    num_observations = 45
    num_privileged_obs = 235
    measure_heights = True
    ```

- 如果采用 smoke test 48 维方案，配置应该满足：

    ```Python
    num_observations = 45
    num_privileged_obs = 48
    measure_heights = False
    ```

- 不要混用：

    ```Python
    num_privileged_obs = 235
    measure_heights = False
    ```

- 也不要混用：

    ```Python
    num_observations = 45
    self.obs_buf = torch.cat((actor_obs, heights), dim=-1)
    ```

- 因为这会让 actor obs 实际变成 232 维，但配置还是 45 维。

- 建议先跑：

    ```Bash
    python legged_gym/scripts/train.py --task=legged_gym_go1 --headless --num_envs=16 --max_iterations=1
    ```

- 如果启动阶段报：

    ```Text
    privileged obs dim mismatch
    ```

- 就说明 `measure_heights` 和 `num_one_step_privileged_obs` 没有对应起来。

# **最终推荐改法**

- 当前阶段按 HIMLoco 的“切片方法”改，不按 HIMLoco 硬凑全部维度。

- 第一版推荐：

    ```Text
    obs_buf = 45
    privileged_obs_buf = 235
    current_obs = [actor_obs(45), base_lin_vel(3), heights(187)]
    ```

- 等连续外力 / disturbance 真的进物理仿真以后，再升级：

    ```Text
    obs_buf = 45
    privileged_obs_buf = 238
    current_obs = [actor_obs(45), base_lin_vel(3), disturbance(3), heights(187)]
    ```

- 这样 actor 仍然是 blind policy，部署时不依赖地形高度；critic 在训练时能看到真实速度和地形高度，符合 asymmetric actor-critic 的用法，也和 HIMLoco 的组织方式一致。
