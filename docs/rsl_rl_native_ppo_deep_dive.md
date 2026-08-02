# rsl_rl 原生 PPO 深度解析：算法、网络、缓冲区与启动设计

本文面向希望从零理解本仓库 `rsl_rl` 原生 PPO 实现的人。它不是只列 API 的速查表，而是把强化学习里的数学动机、PPO 的工程拆分、网络结构、rollout 缓冲区、runner 启动路径、训练循环和保存推理路径串成一条完整链路。本文讨论的“原生 PPO”特指以下源码组成的默认 on-policy 训练栈：

- `rsl_rl/rsl_rl/runners/on_policy_runner.py` 中的 `OnPolicyRunner`
- `rsl_rl/rsl_rl/algorithms/ppo.py` 中的 `PPO`
- `rsl_rl/rsl_rl/modules/actor_critic.py` 中的 `ActorCritic`
- `rsl_rl/rsl_rl/storage/rollout_storage.py` 中的 `RolloutStorage`
- `rsl_rl/rsl_rl/env/vec_env.py` 中的 `VecEnv` 抽象接口
- `legged_gym/envs/base/legged_robot_config.py` 中 `LeggedRobotCfgPPO` 的默认配置片段

## 1. 从整体上看：原生 PPO 在仓库里解决什么问题

`rsl_rl` 的原生 PPO 是一个典型的并行仿真、on-policy、连续动作控制训练框架。它假设环境一次并行运行很多个 `env`，每个环境每一步返回观测、奖励、终止标记和附加信息。策略网络根据当前观测采样动作，价值网络根据 critic 观测估计状态价值，runner 把一段固定长度的 rollout 收集进缓冲区，然后 PPO 算法用这批刚收集的数据做若干 epoch 的 mini-batch 更新。更新结束后缓冲区清空，再收集下一批新数据。

这种设计里的核心边界很清楚：`OnPolicyRunner` 负责“什么时候采样、什么时候更新、什么时候记录日志和保存模型”；`PPO` 负责“如何把旧策略采样出来的数据变成损失函数并更新参数”；`ActorCritic` 负责“策略分布和价值函数的神经网络表达”；`RolloutStorage` 负责“把时间维、环境维、动作、价值、log probability、return、advantage 等训练所需张量存起来并切成 batch”。这些模块合起来就是一个标准 Actor-Critic PPO 系统。

如果用一句话概括训练链路：`OnPolicyRunner.learn()` 先从 `VecEnv` 取观测，调用 `PPO.act()` 让 `ActorCritic` 生成动作并记录旧策略统计量，再把环境返回通过 `PPO.process_env_step()` 放入 `RolloutStorage`，rollout 满后调用 `PPO.compute_returns()` 计算 GAE 和 return，最后 `PPO.update()` 读取 mini-batch 做 PPO-Clip 更新。

## 2. 强化学习基本对象：MDP、轨迹、回报和价值

PPO 背后默认的数学抽象是马尔可夫决策过程。一个 MDP 可以写成五元组 `M=(S,A,P,r,γ)`。`S` 是状态空间，`A` 是动作空间，`P(s'|s,a)` 是状态转移概率，`r(s,a)` 是即时奖励，`γ ∈ [0,1)` 是折扣因子。腿式机器人控制里真实状态往往包含机器人姿态、关节角速度、接触状态、地形、外部扰动等；actor 看到的观测可能只是其中一部分，所以工程上常把 actor 观测和 critic privileged observation 分开。

策略 `π_θ(a|s)` 是参数为 `θ` 的动作分布。在连续动作 PPO 中，策略不是直接输出唯一动作，而是输出一个高斯分布的均值，标准差由可学习参数或状态相关网络给出。本仓库 `ActorCritic` 的做法是 actor MLP 输出均值 `mean`，并用 `self.std` 作为每个动作维度的可学习标准差，构造 `torch.distributions.Normal(mean, mean*0. + self.std)`。动作采样来自该分布，log probability 和 entropy 也来自该分布。

一条轨迹可表示为 `τ=(s_0,a_0,r_0,s_1,a_1,r_1,…)`。从时间 `t` 开始的折扣回报是：

```text
G_t = ∑_{k=0}^{∞} γ^k r_{t+k}.
```

价值函数 `V^π(s)` 是在状态 `s` 按策略 `π` 行动时的期望回报。动作价值函数 `Q^π(s,a)` 是先执行动作 `a` 再按策略行动的期望回报。优势函数 `A^π(s,a)=Q^π(s,a)-V^π(s)` 衡量某个动作相对于当前状态平均水平好多少。Actor-Critic 的基本思想就是：actor 学策略，critic 学价值，critic 估出的 advantage 作为 actor 更新方向的低方差信号。

## 3. Policy Gradient 与 Actor-Critic：为什么需要价值网络

目标函数可写为 `J(θ)=E_{τ∼π_θ}[G_0]`。Policy Gradient Theorem 给出一个可采样估计的梯度形式：

```text
∇_θ J(θ) = E_{s,a∼π_θ}[∇_θ log π_θ(a|s) A^π_θ(s,a)].
```

直觉是：如果某个动作的优势为正，就提高该动作在该状态下的概率；如果优势为负，就降低概率。这个公式直接可用，但如果用 Monte Carlo 回报估计 advantage，方差会很大。价值网络就是为了给回报提供 baseline。减去 baseline 不改变期望梯度，但可以显著降低方差。本仓库的 `ActorCritic` 把 actor 和 critic 做成两个分离 MLP，actor 输入 `num_actor_obs`，critic 输入 `num_critic_obs`。当环境提供 privileged observation 时，critic 可以看到更多信息；如果没有，就和 actor 使用同一观测。

Actor-Critic 的工程优势是在线并行控制非常自然。runner 每一步都能同时拿到 action、value 和 log probability；storage 只要把这些旧策略统计量保存下来，更新时就能比较新旧策略概率比。PPO 正是建立在这个“旧策略采样、新策略复用多轮但限制步长”的结构上。

## 4. PPO 的核心动机：稳定地复用 on-policy 数据

普通 policy gradient 每收集一批数据通常只做一次更新，因为数据来自旧策略，而策略一旦改变，数据分布就不再严格 on-policy。TRPO 用 trust region 约束 KL 散度，让新策略不要离旧策略太远；PPO 则用更简单的 clipped surrogate 近似这种约束。PPO 的重要性采样比率为：

```text
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t) = exp(log π_θ(a_t|s_t) - log π_old(a_t|s_t)).
```

如果 `A_t>0`，我们希望增大动作概率，但不希望增大太多；如果 `A_t<0`，我们希望降低动作概率，也不希望降低太多。PPO-Clip 的目标是：

```text
L^CLIP(θ)=E_t[min(r_t(θ)A_t, clip(r_t(θ),1-ε,1+ε)A_t)].
```

本仓库代码以最小化 loss 的方式实现，所以写成负号：`surrogate = -advantages * ratio`，`surrogate_clipped = -advantages * clamp(ratio, 1-clip_param, 1+clip_param)`，然后取 `torch.max`。这和最大化上式中的 `min` 等价，因为最小化负目标时要取更保守、更大的损失项。这个细节是读代码时最容易混淆的地方：论文里通常是最大化，PyTorch 训练里通常是最小化。

## 5. GAE：优势估计如何在偏差和方差之间折中

`RolloutStorage.compute_returns()` 实现的是 Generalized Advantage Estimation。先定义 TD 残差：

```text
δ_t = r_t + γ V(s_{t+1}) - V(s_t).
```

GAE 把未来多步 TD 残差按 `γλ` 衰减累加：

```text
Â_t = δ_t + γλ δ_{t+1} + (γλ)² δ_{t+2} + ⋯.
```

其中 `λ` 越接近 1，越接近 Monte Carlo 回报，方差更高但偏差更低；`λ` 越接近 0，越接近一步 TD，方差更低但偏差更高。代码里的 `gamma` 和 `lam` 分别由算法配置传入，默认配置中 `gamma=0.99`、`lam=0.95`。在腿式机器人任务里，奖励密集、并行环境多、rollout 短，因此 GAE 可以在短 horizon 里提供稳定的 advantage。

`compute_returns()` 从最后一个 step 倒序遍历。遇到终止状态时，`next_is_not_terminal = 1.0 - dones[step].float()` 会把跨 episode 的 bootstrap 切断。最后 `returns = advantages + values`，并对 advantage 做全 batch 标准化：

```text
Â ← (Â - μ_A) / (σ_A + 10^-8).
```

标准化不会改变优势的相对排序，但会让 actor loss 的尺度更稳定，避免某些 batch 因 reward scale 过大导致梯度爆炸。

## 6. `ActorCritic` 网络：最小但完整的连续控制策略

`rsl_rl/rsl_rl/modules/actor_critic.py` 中的 `ActorCritic` 是一个非 recurrent 的 actor-critic 模块。构造函数接受 `num_actor_obs`、`num_critic_obs`、`num_actions`、`actor_hidden_dims`、`critic_hidden_dims`、`activation` 和 `init_noise_std`。actor 和 critic 都是 `nn.Sequential` MLP。actor 的最后一层输出 `num_actions` 维动作均值；critic 的最后一层输出一维 value。

默认配置来自 `LeggedRobotCfgPPO.policy`，actor 和 critic hidden dims 通常是 `[512, 256, 128]`，激活函数是 `elu`，初始动作噪声标准差是 `1.0`。这种结构没有共享 trunk。分离 actor 和 critic 的优点是 actor 只服务于可执行观测，critic 可以服务于 privileged observation；缺点是参数更多，且 actor/critic 表征不能共享。对于机器人控制，critic 看到 privileged observation 时，分离结构更干净，因为共享 trunk 会把训练时专有的信息泄露到部署 actor 的路径中。

策略分布通过 `update_distribution(observations)` 构造：先 `mean = self.actor(observations)`，再 `Normal(mean, mean*0. + self.std)`。`mean*0. + self.std` 的写法让标准差广播到 batch 形状，同时不依赖观测。`self.std` 是 `nn.Parameter`，会随 PPO 主优化器一起更新。`act()` 调用 `update_distribution()` 后 `sample()`；`get_actions_log_prob(actions)` 返回每个动作维度 log prob 求和；`entropy` 返回各维 entropy 求和。

这个设计的推理路径也很清楚：`act_inference(observations)` 只返回 actor 均值，不采样。训练时使用随机动作促进探索，部署时通常使用确定性均值动作提升稳定性。保存模型时 runner 保存 actor_critic 的 `state_dict`，因此 `std` 参数也会被保存。

## 7. `RolloutStorage` 缓冲区：为什么保存旧均值、旧标准差和旧 log prob

PPO 更新必须知道数据来自哪个旧策略。`RolloutStorage` 的 `Transition` 临时对象保存单步采样结果：actor observation、critic observation、action、reward、done、value、action log probability、action mean、action sigma，以及 recurrent 时的 hidden states。runner 每走一步，`PPO.act()` 先填 transition 的 action/value/log_prob/mu/sigma/obs，环境 step 后 `PPO.process_env_step()` 再填 reward/done，并调用 `storage.add_transitions()` 把 transition 拷进大张量。

大张量的第一维是时间 `num_transitions_per_env`，第二维是并行环境 `num_envs`。例如 observations 形状是 `[T, N, obs_dim]`，actions 是 `[T, N, action_dim]`，values、returns、advantages 是 `[T, N, 1]`。这种布局符合 rollout 收集时按时间推进的写入模式。mini-batch 更新前，普通 feed-forward 生成器把前两维 flatten 成 `[T*N, ...]`，再用随机索引切 mini-batch。

保存 `old_actions_log_prob` 是计算 ratio 的最直接方式。保存 `old_mu` 和 `old_sigma` 则用于 KL 自适应学习率。代码里 KL 近似计算的是两个对角高斯分布之间的解析 KL：

```text
D_KL(N(μ_old, σ_old) ‖ N(μ, σ)) = ∑_i [log(σ_i / σ_{old,i}) + (σ_{old,i}² + (μ_{old,i} - μ_i)²) / (2σ_i²) - 1/2].
```

如果只保存旧 log prob，也能算 PPO ratio，但无法直接算新旧高斯分布的 KL。因此 storage 同时保存旧均值和旧标准差。这个设计让 PPO 更新不仅有 clip，还可以通过 `desired_kl` 调整学习率。

## 8. `PPO` 类：算法对象如何组织一次训练更新

`rsl_rl/rsl_rl/algorithms/ppo.py` 中的 `PPO` 负责保存超参数、持有 actor_critic、optimizer 和 storage。构造函数里的关键参数包括 `num_learning_epochs`、`num_mini_batches`、`clip_param`、`gamma`、`lam`、`value_loss_coef`、`entropy_coef`、`learning_rate`、`max_grad_norm`、`use_clipped_value_loss`、`schedule` 和 `desired_kl`。它用 `optim.Adam(self.actor_critic.parameters(), lr=learning_rate)` 更新整个 actor_critic，包括 actor、critic 和 `std`。

`act(obs, critic_obs)` 是 rollout 阶段的入口。它先在 recurrent 情况下保存 hidden states，然后调用 `actor_critic.act(obs)` 采样 action，调用 `actor_critic.evaluate(critic_obs)` 得到 value，再从当前分布里取 action log probability、mean 和 std。注意这些都 `.detach()`，因为 rollout 收集不构建梯度图，storage 保存的是旧策略统计量，不应把 rollout 的计算图跨越到 update。

`process_env_step(rewards, dones, infos)` 负责把环境反馈接到 transition 上。如果 `infos` 包含 `time_outs`，代码会把 timeout 状态的 reward 加上 `γ V(s_t)` 的 bootstrap 项。这里的含义是：时间限制截断不是任务自然终止，不能把它当作失败 terminal，否则会系统性低估接近最大 episode 长度的状态价值。这个工程细节对 Isaac Gym 风格的定长 episode 很重要。

`compute_returns(last_critic_obs)` 在 rollout 结束时用最后一个 critic observation 计算 `last_values`，再委托 storage 反向计算 GAE。`update()` 读取 mini-batch，重新用当前策略计算新 log prob、新 value、新 distribution 参数和 entropy，然后构造 loss 并反向传播。

## 9. PPO loss 的三个主要部分

本实现总 loss 是：

```text
L = L_policy + c_v L_value - c_e H(π_θ),
```

对应代码是 `surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()`。三项意义如下。

第一，`surrogate_loss` 是 PPO-Clip 策略损失。ratio 来自新旧 log prob 差的指数。clip 范围由 `clip_param` 控制，默认常见值是 0.2。clip 不保证严格 KL 上界，但能抑制单个 batch 上过大的策略更新。

第二，`value_loss` 是价值函数均方误差。如果 `use_clipped_value_loss=True`，代码先构造 `value_clipped = target_values + clamp(value_batch - target_values, -clip, clip)`，然后取 unclipped value loss 和 clipped value loss 的最大值。这借鉴 PPO 论文中的 value clipping 思路，避免 critic 单次更新跨度太大。这里的 `target_values_batch` 是 storage 中保存的旧 value，不是 return；`returns_batch` 是 GAE return。critic 被要求拟合 return，但被限制相对于旧 value 的改变量。

第三，entropy bonus 鼓励探索。连续高斯策略的 entropy 与标准差相关。`entropy_coef` 越大，策略越倾向维持较大噪声；越小，策略更快收敛到确定性动作。在机器人控制中，过大的 entropy 可能导致动作抖动，过小则可能过早收敛到局部行为。

最后代码用 `nn.utils.clip_grad_norm_` 做梯度裁剪。梯度裁剪不是 PPO 特有，但在并行仿真和非平稳 reward 下能防止少数异常 batch 把参数推飞。

## 10. 自适应 KL 学习率：clip 之外的步长控制

当 `schedule == 'adaptive'` 且 `desired_kl` 不为 `None` 时，`PPO.update()` 会计算新旧策略高斯分布的平均 KL。如果 `kl_mean > desired_kl * 2.0`，学习率除以 1.5，最低到 `1e-5`；如果 `kl_mean < desired_kl / 2.0` 且大于 0，学习率乘以 1.5，最高到 `1e-2`。然后更新 optimizer param group 中的 `lr`。

这是一种非常实用的工程反馈控制。clip objective 控制的是样本级 ratio，KL schedule 控制的是分布级平均变化。两者不是互相替代，而是从不同角度限制策略步长。对于高维连续动作，某些维度的均值或标准差变化可能在 ratio 上不直观，KL 提供了更全局的诊断。

需要注意，代码里的 KL 计算在 `torch.inference_mode()` 下执行，不参与梯度。它只调学习率，不作为 loss penalty。相比 KL penalty PPO，这种方式更简单，不需要调 penalty coefficient，但它的控制是滞后的：先用当前 mini-batch 看到 KL，再影响后续 mini-batch 的学习率。

## 11. `OnPolicyRunner`：训练启动和主循环设计

`OnPolicyRunner.__init__()` 接受 `env`、`train_cfg`、`log_dir` 和 `device`。它从 `train_cfg` 中拆出 `runner`、`algorithm`、`policy` 三段配置。critic observation 维度通过 `env.num_privileged_obs` 决定；如果环境没有 privileged observation，就使用 `env.num_obs`。然后通过 `eval(self.cfg["policy_class_name"])` 实例化 policy 类，默认是 `ActorCritic`；通过 `eval(self.cfg["algorithm_class_name"])` 实例化算法类，默认是 `PPO`。

这里的 `eval` 是一种轻量动态注册方式。优点是配置里改字符串就能切换 `ActorCritic`、`ActorCriticRecurrent` 或其他算法；缺点是类型安全弱，字符串写错会运行时报错。`rsl_rl/rsl_rl/runners/__init__.py` 和相关 import 确保这些类名在 runner 文件作用域可见。

runner 随后调用 `self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])`。这里 actor obs、critic obs、action shape 都以 list 形式传入，使 storage 用 `*obs_shape` 展开成张量尾维。最后 runner 初始化日志状态，调用 `env.reset()`。

`learn()` 是完整训练循环。每个 iteration 先进入 rollout 阶段，在 `torch.inference_mode()` 中循环 `num_steps_per_env` 次：调用 `self.alg.act(obs, critic_obs)`，执行 `env.step(actions)`，把新 obs、critic_obs、reward、done 移到 device，调用 `self.alg.process_env_step(...)` 写入 storage，同时维护 episode reward/length buffer。rollout 结束后计算 return，再退出 inference mode 调用 `self.alg.update()`。

## 12. 采样、学习、日志和保存的时间边界

runner 把一次 iteration 拆成 collection time 和 learn time。collection time 是环境交互和 storage 写入时间；learn time 是 GAE 后 PPO 多 epoch 更新的时间。TensorBoard 记录 value loss、surrogate loss、learning rate、mean noise std、fps、collection time、learning time、mean reward 和 mean episode length。控制台也打印这些统计量。

保存逻辑在每个 iteration 检查 `if it % self.save_interval == 0`，保存 `model_{it}.pt`；训练结束再保存当前 iteration 的模型。checkpoint 字典包含 `model_state_dict`、`optimizer_state_dict`、`iter` 和 `infos`。加载时恢复 actor_critic 参数，按需恢复 optimizer，并设置 `current_learning_iteration`。

`get_inference_policy(device=None)` 会把 actor_critic 切到 eval，必要时迁移 device，然后返回 `actor_critic.act_inference`。这意味着部署侧只需要拿到这个 callable，给 observations 返回动作均值。训练时的 storage、optimizer、PPO loss 都不会进入推理路径。

## 13. `VecEnv` 接口：runner 对环境的最小假设

`rsl_rl/rsl_rl/env/vec_env.py` 定义了抽象 `VecEnv`。原生 runner 期望环境具有 `num_envs`、`num_obs`、`num_privileged_obs`、`num_actions`、`max_episode_length`、`episode_length_buf` 等属性，并提供 `step(actions)`、`reset(env_ids)`、`get_observations()` 和 `get_privileged_observations()`。原生 `OnPolicyRunner` 的 `env.step(actions)` 预期返回五元组：`obs, privileged_obs, rewards, dones, infos`。

这个接口刻意很薄。它不规定仿真器、机器人模型、奖励项或 reset 细节，只规定 PPO 训练需要的最小数据。这样 `rsl_rl` 可以被 `legged_gym` 这类环境库调用，也可以理论上接其他向量化环境。对算法而言，环境只要提供连续动作控制的数据流即可。

## 14. 默认配置如何映射到代码

`legged_gym/envs/base/legged_robot_config.py` 中 `LeggedRobotCfgPPO` 给出默认训练配置：`runner_class_name='OnPolicyRunner'`，policy 默认 `ActorCritic`，algorithm 默认 `PPO`。policy 部分设置网络隐藏层、激活函数、初始噪声；algorithm 部分设置 value loss 系数、clip、entropy、epoch、mini-batch、learning rate、schedule、gamma、lambda、desired_kl 和梯度裁剪；runner 部分设置 `num_steps_per_env`、`max_iterations`、保存间隔和 resume 相关字段。

这些配置不是独立存在的文本，而是直接决定类构造参数。比如 `actor_hidden_dims` 传给 `ActorCritic`，`clip_param` 传给 `PPO`，`num_steps_per_env` 决定 storage 的时间长度。理解配置和源码之间的映射，比单独背超参数更重要，因为训练行为通常是多个超参数共同作用的结果。

## 15. Recurrent 支持为什么在 storage 中存在

虽然本文重点是原生 feed-forward PPO，但 `RolloutStorage` 和 `PPO.update()` 中保留了 recurrent 分支。如果 `actor_critic.is_recurrent` 为真，`PPO.act()` 会保存 hidden states，`update()` 会调用 `reccurent_mini_batch_generator()`。该生成器通过 `split_and_pad_trajectories` 按 done 切轨迹并 padding，返回 masks 和对应 hidden state batch。

这说明 `rsl_rl` 的原生 PPO 设计不是只服务 MLP。它把“普通 flatten mini-batch”和“按 episode 片段组织 RNN mini-batch”放在同一个 storage 抽象里。对于腿式机器人，如果观测历史不足或需要记忆接触/地形状态，RNN actor-critic 是一种可扩展路径。但默认 `ActorCritic` 的 `is_recurrent=False`，所以普通 PPO 不走这个路径。

## 16. 数据流逐步追踪：从观测到一次梯度更新

一次完整的数据流可以分解为十个步骤。第一，runner 从环境取 `obs` 和 `privileged_obs`，得到 `critic_obs`。第二，`PPO.act()` 用 actor 分布采样 action，并用 critic 估 value。第三，`PPO.act()` 保存旧策略的 action log prob、mean、std。第四，runner 用 action 推进环境。第五，`process_env_step()` 保存 reward、done，并处理 timeout bootstrap。第六，storage 把 transition 拷贝到第 `step` 个时间槽。第七，rollout 满后 `compute_returns()` 倒序算 GAE。第八，`mini_batch_generator()` flatten 并随机切 batch。第九，`PPO.update()` 用当前网络重算 log prob/value/entropy。第十，loss backward，梯度裁剪，Adam step，storage clear。

这条链路体现了 PPO 最重要的工程不变量：更新时需要同时看到“旧策略采样时的信息”和“当前策略重算的信息”。旧信息来自 storage，新信息来自 actor_critic 当前参数。ratio、KL、clipped objective 都建立在这个比较上。

## 17. 设计取舍：为什么这个实现简洁而有效

本实现没有复杂的分布封装、没有多优化器、没有 replay buffer、没有 target network，也没有异步 actor。它选择了高度同步的 on-policy pipeline：并行环境一次收集固定长度 rollout，然后 GPU 上集中更新。这种设计很适合 Isaac Gym/legged_gym 一类高吞吐仿真，因为瓶颈通常不是环境 step，而是如何稳定利用大量并行样本。

简洁也带来限制。on-policy 数据不能长期复用，样本效率低于 off-policy；固定对角高斯难以表达动作维度之间相关性；可学习全局标准差不能随状态动态调整探索强度；`eval` 动态类名缺少静态检查；advantage 全 batch 标准化在多任务或 reward scale 极不均匀时可能掩盖差异。但对许多机器人 locomotion 基准来说，这些取舍是合理的，因为稳定性、吞吐和实现可控性往往比极限样本效率更重要。

## 18.1 设计札记：观测与特权观测

actor observation 是部署时可获得的信息，critic observation 可以包含训练时才可获得的 privileged signal。原生 PPO 通过 `num_privileged_obs` 决定 critic 输入维度，避免把不可部署信息泄露进 actor。

从数学角度看，观测与特权观测并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 观测与特权观测 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，观测与特权观测还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.2 设计札记：动作分布

连续控制中的动作不是分类采样，而是高斯分布采样。actor 输出均值，`self.std` 给出每个动作维的尺度，log probability 对动作维求和后成为 PPO ratio 的基础。

从数学角度看，动作分布并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 动作分布 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，动作分布还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.3 设计札记：旧策略统计量

PPO 的名字里虽然没有显式 old policy 对象，但 storage 中的 old log prob、old mu、old sigma 就是旧策略的快照。没有这些量，clip 和 KL 都无法正确计算。

从数学角度看，旧策略统计量并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 旧策略统计量 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，旧策略统计量还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.4 设计札记：GAE 倒序递推

倒序递推让每个时间步复用后一个时间步的 advantage 累积值。`done` mask 切断跨 episode bootstrap，从而保证不同 episode 不互相污染。

从数学角度看，GAE 倒序递推并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 GAE 倒序递推 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，GAE 倒序递推还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.5 设计札记：timeout bootstrap

时间限制截断和真实失败终止不同。代码对 `infos[time_outs]` 增加 value bootstrap，体现了工程上对 TimeLimit truncation 的处理。

从数学角度看，timeout bootstrap并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 timeout bootstrap 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，timeout bootstrap还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.6 设计札记：value clipping

critic 过快变化会让 advantage 和 return 目标在多 epoch 更新中变得不稳定。value clipping 用旧 value 限制新 value 的单步偏移。

从数学角度看，value clipping并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 value clipping 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，value clipping还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.7 设计札记：entropy bonus

entropy 是策略分布不确定性的度量。训练早期它帮助探索，训练后期如果系数过大则可能保留不必要的噪声。

从数学角度看，entropy bonus并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 entropy bonus 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，entropy bonus还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.8 设计札记：KL 自适应学习率

clip 控制样本 ratio，KL 控制整体分布距离。二者合用时，即使一个机制不够敏感，另一个也能提供步长反馈。

从数学角度看，KL 自适应学习率并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 KL 自适应学习率 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，KL 自适应学习率还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.9 设计札记：mini-batch 多 epoch

同一批 rollout 数据会被随机切分并训练多个 epoch。这提高样本利用率，但必须靠 clip 和 KL 防止策略离采样分布太远。

从数学角度看，mini-batch 多 epoch并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 mini-batch 多 epoch 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，mini-batch 多 epoch还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.10 设计札记：梯度裁剪

机器人奖励和接触动力学可能产生尖峰梯度。全局范数裁剪是低成本保护，尤其适合多 loss 混合的 actor-critic。

从数学角度看，梯度裁剪并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 梯度裁剪 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，梯度裁剪还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.11 设计札记：日志指标

surrogate loss、value loss、mean action noise std、fps 和 episode reward 共同描述训练状态。单看 reward 不足以判断 PPO 是否健康。

从数学角度看，日志指标并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 日志指标 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，日志指标还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.12 设计札记：保存与加载

checkpoint 同时保存模型和 optimizer。恢复 optimizer 能保留 Adam 动量状态，继续训练更平滑；只加载模型则更适合部署或微调。

从数学角度看，保存与加载并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 保存与加载 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，保存与加载还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.13 设计札记：推理策略

`act_inference` 返回均值动作而非采样动作。部署控制器通常需要确定性和可重复性，训练探索噪声不应直接带到实机执行。

从数学角度看，推理策略并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 推理策略 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，推理策略还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.14 设计札记：并行环境维度

storage 的 `[T,N,...]` 布局保留时间和环境维。写入时按时间推进，训练时 flatten 成样本池，这是并行 on-policy PPO 的常见模式。

从数学角度看，并行环境维度并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 并行环境维度 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，并行环境维度还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.15 设计札记：recurrent 预留

虽然默认是 MLP，storage 仍保存 hidden states 并支持 padding trajectory。这说明框架在设计上考虑了记忆策略扩展。

从数学角度看，recurrent 预留并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 recurrent 预留 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，recurrent 预留还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.16 设计札记：配置驱动实例化

`policy_class_name` 和 `algorithm_class_name` 让 runner 不硬编码具体类。代价是字符串错误要到运行时才暴露。

从数学角度看，配置驱动实例化并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 配置驱动实例化 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，配置驱动实例化还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.17 设计札记：device 迁移

runner 和算法都显式把 tensor 移到 device。大规模 GPU 仿真中，避免 CPU/GPU 往返是吞吐稳定的基础。

从数学角度看，device 迁移并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 device 迁移 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，device 迁移还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.18 设计札记：critic 训练目标

critic 拟合的是 GAE return，不是即时奖励。它学习“从这个状态往后还能得到多少折扣回报”。

从数学角度看，critic 训练目标并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 critic 训练目标 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，critic 训练目标还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.19 设计札记：actor 更新方向

优势为正的动作会被提高概率，优势为负的动作会被降低概率。clip 只是限制这个方向上的过度移动。

从数学角度看，actor 更新方向并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 actor 更新方向 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，actor 更新方向还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.20 设计札记：设计边界

runner 不关心 loss 公式，PPO 不关心环境奖励细节，storage 不关心网络结构。这种分层让后续 HIM 扩展可以只替换必要模块。

从数学角度看，设计边界并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 设计边界 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，设计边界还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.21 设计札记：观测与特权观测

actor observation 是部署时可获得的信息，critic observation 可以包含训练时才可获得的 privileged signal。原生 PPO 通过 `num_privileged_obs` 决定 critic 输入维度，避免把不可部署信息泄露进 actor。

从数学角度看，观测与特权观测并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 观测与特权观测 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，观测与特权观测还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.22 设计札记：动作分布

连续控制中的动作不是分类采样，而是高斯分布采样。actor 输出均值，`self.std` 给出每个动作维的尺度，log probability 对动作维求和后成为 PPO ratio 的基础。

从数学角度看，动作分布并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 动作分布 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，动作分布还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.23 设计札记：旧策略统计量

PPO 的名字里虽然没有显式 old policy 对象，但 storage 中的 old log prob、old mu、old sigma 就是旧策略的快照。没有这些量，clip 和 KL 都无法正确计算。

从数学角度看，旧策略统计量并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 旧策略统计量 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，旧策略统计量还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.24 设计札记：GAE 倒序递推

倒序递推让每个时间步复用后一个时间步的 advantage 累积值。`done` mask 切断跨 episode bootstrap，从而保证不同 episode 不互相污染。

从数学角度看，GAE 倒序递推并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 GAE 倒序递推 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，GAE 倒序递推还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.25 设计札记：timeout bootstrap

时间限制截断和真实失败终止不同。代码对 `infos[time_outs]` 增加 value bootstrap，体现了工程上对 TimeLimit truncation 的处理。

从数学角度看，timeout bootstrap并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 timeout bootstrap 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，timeout bootstrap还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.26 设计札记：value clipping

critic 过快变化会让 advantage 和 return 目标在多 epoch 更新中变得不稳定。value clipping 用旧 value 限制新 value 的单步偏移。

从数学角度看，value clipping并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 value clipping 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，value clipping还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.27 设计札记：entropy bonus

entropy 是策略分布不确定性的度量。训练早期它帮助探索，训练后期如果系数过大则可能保留不必要的噪声。

从数学角度看，entropy bonus并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 entropy bonus 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，entropy bonus还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.28 设计札记：KL 自适应学习率

clip 控制样本 ratio，KL 控制整体分布距离。二者合用时，即使一个机制不够敏感，另一个也能提供步长反馈。

从数学角度看，KL 自适应学习率并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 KL 自适应学习率 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，KL 自适应学习率还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.29 设计札记：mini-batch 多 epoch

同一批 rollout 数据会被随机切分并训练多个 epoch。这提高样本利用率，但必须靠 clip 和 KL 防止策略离采样分布太远。

从数学角度看，mini-batch 多 epoch并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 mini-batch 多 epoch 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，mini-batch 多 epoch还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 18.30 设计札记：梯度裁剪

机器人奖励和接触动力学可能产生尖峰梯度。全局范数裁剪是低成本保护，尤其适合多 loss 混合的 actor-critic。

从数学角度看，梯度裁剪并不是孤立技巧，而是 policy gradient 在有限样本、函数逼近和并行仿真条件下的稳定化手段。PPO 的真实难点不在于写出一个 ratio，而在于保证 ratio 对应的旧策略统计量、advantage 的尺度、critic 的 bootstrap、action distribution 的熵和 optimizer 的步长全部处于可解释范围。仓库实现把这些约束拆成多个小部件：旧统计量由 `RolloutStorage` 保存，优势由 `compute_returns` 标准化，策略分布由 `ActorCritic.update_distribution` 统一创建，步长由 clip、KL schedule 和 gradient clipping 共同约束。

工程上设计 梯度裁剪 时要一直追问三个问题。第一，这个量是在 rollout 阶段决定，还是在 update 阶段重算；如果阶段搞错，就可能把新策略的信息混进旧策略样本。第二，这个量是否应该 detach；如果 rollout 图没有断开，显存和梯度语义都会出问题。第三，这个量的 batch 维度是否与 `[T,N]` 展平后的索引一致；如果索引错位，PPO 仍会运行，但 ratio、advantage 和 action 不再对应同一个时间步，训练会表现为无规律退化。

放到腿式机器人控制里，梯度裁剪还要面对接触不连续、奖励项尺度差异、episode 因 timeout 被截断、并行环境 reset 不同步等问题。原生 PPO 的实现没有把这些问题隐藏成复杂框架，而是用少量清晰张量显式表达。读代码时可以把每个张量都问成一句话：它来自哪个时间点，它属于旧策略还是新策略，它是否跨越 terminal，它是否会进入梯度。只要这四个问题回答清楚，PPO 主循环就不再神秘。

## 19. 排错和阅读建议

阅读原生 PPO 时建议先沿着运行时调用顺序看，而不是按文件名看。第一遍只跟踪 `OnPolicyRunner.learn()`，弄清每个 iteration 何时收集数据、何时更新。第二遍进入 `PPO.act()` 和 `PPO.update()`，把 transition 保存的旧策略量与 update 重算的新策略量一一对应。第三遍看 `RolloutStorage.compute_returns()` 和 `mini_batch_generator()`，确认 return、advantage、old log prob、old mu、old sigma 如何进入 loss。最后再看 `ActorCritic`，理解分布对象怎样把 mean/std、sample、log_prob、entropy 统一起来。

如果训练出现异常，常见检查顺序是：确认 observations/action shapes 是否匹配；确认 rewards/dones 是否在 device 上；确认 `infos['time_outs']` 的 shape 是否能与 values 对齐；确认 `advantages.std()` 不为 0；确认 `std` 没有变成负数或 NaN；确认 KL 自适应学习率没有长期卡在上下界；确认保存的 checkpoint 中 `iter` 和 optimizer 状态符合预期。

## 20. 总结

本仓库的原生 PPO 是一个清晰的 on-policy Actor-Critic 实现。它没有过度抽象，但保留了 PPO 训练所需的关键机制：连续高斯策略、价值函数、GAE、clip surrogate、value clipping、entropy bonus、KL 自适应学习率、梯度裁剪、并行 rollout storage、TensorBoard 日志和 checkpoint。理解它的关键是把数学对象和工程张量对应起来：策略分布对应 actor 的 Normal，价值函数对应 critic 输出，优势对应 storage 中标准化后的 advantages，旧策略对应 storage 保存的 log prob/mu/sigma，信任域近似对应 clip 和 KL schedule。掌握这条链路后，再看 HIMLoco/HIMPPO 的扩展就会自然得多，因为 HIM 的主干仍然是这套 PPO，只是在观测表征和辅助估计上加入了面向部分可观测腿式运动控制的设计。
