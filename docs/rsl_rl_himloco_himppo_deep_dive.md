# rsl_rl HIMLoco / HIMPPO 深度解析：历史编码、隐变量估计、缓冲区与启动设计

本文解析本仓库 `rsl_rl` 中 HIMLoco 风格的 PPO 扩展实现。这里的 HIMLoco 不是一个单独的环境文件；当前仓库里 `legged_gym/envs/him_go1/him_go2.py`、`legged_gym/envs/him_go1/him_go2_config.py`、`legged_gym/envs/him_go2w/him_go2w.py` 和 `legged_gym/envs/him_go2w/him_go2w_config.py` 是空文件。因此本文把“himloco”严格限定为 `rsl_rl` 里已经存在的 HIM 训练栈：

- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py` 中的 `HIMOnPolicyRunner`
- `rsl_rl/rsl_rl/algorithms/him_ppo.py` 中的 `HIMPPO`
- `rsl_rl/rsl_rl/modules/him_actor_critic.py` 中的 `HIMActorCritic`
- `rsl_rl/rsl_rl/modules/him_estimator.py` 中的 `HIMEstimator`
- `rsl_rl/rsl_rl/storage/him_rollout_storage.py` 中的 `HIMRolloutStorage`

HIM 版本不是推翻原生 PPO，而是在 PPO 主干上增加“历史观测编码”和“隐变量/速度估计”辅助学习。它仍然使用 clipped surrogate、value loss、entropy bonus、GAE、KL 自适应学习率和 on-policy rollout；不同点是 actor 不直接吃完整历史观测，而是把历史观测送入 estimator，得到三维速度估计和一个 16 维 latent，再把“当前单步观测 + 估计速度 + latent”拼接给 actor。critic 仍然使用 privileged observation。estimator 自己通过速度 MSE 和 prototype swap loss 训练。

## 1. HIM 要解决的问题：腿式运动控制中的部分可观测性

腿式机器人控制往往不是完全可观测 MDP，而更接近 POMDP。真实状态包括基座线速度、角速度、接触力、摩擦、地形高度、外部扰动、执行器延迟等；actor 在部署时未必能直接获得全部状态。解决 POMDP 的常见方法有三类：第一，给 actor 输入一段历史观测，让网络从时间序列中推断隐藏状态；第二，使用 RNN/GRU/LSTM 显式维护记忆；第三，引入 estimator，把历史压缩成更小的 belief 或 latent，再给策略使用。HIMActorCritic 采用第三类思路。

在本实现中，`HIMActorCritic` 通过 `history_size = int(num_actor_obs / num_one_step_obs)` 假设 actor observation 是若干个单步观测拼接形成的历史。`num_one_step_obs` 表示一帧观测长度，`num_actor_obs` 是历史总长度。estimator 的 encoder 输入完整历史，输出前三维速度 `vel` 和后续 latent `z`。actor 的输入维度被写成 `num_one_step_obs + 3 + 16`，也就是当前单步观测、三维速度估计、16 维隐变量。

这种结构的含义是：历史观测不直接全部进入 actor MLP，而先被压缩成少量任务相关信息。这样 actor 输入更紧凑，部署时计算更轻；同时 estimator 可以用辅助目标单独训练，使 latent 不只是被 PPO sparse/noisy gradient 间接塑造。

## 2. 从原生 PPO 到 HIMPPO：不变的主干和新增的估计分支

`HIMPPO` 和原生 `PPO` 的构造参数几乎一致：都有 `clip_param`、`gamma`、`lam`、`value_loss_coef`、`entropy_coef`、`learning_rate`、`max_grad_norm`、`schedule` 和 `desired_kl`。它同样持有 `actor_critic`、`storage`、`optimizer` 和 `transition`。主 PPO loss 的计算也同样包含 ratio、clipped surrogate、value clipping、entropy 和 gradient clipping。

真正的差异有三处。第一，storage 换成 `HIMRolloutStorage`，transition 多了 `next_critic_observations`。第二，runner 的环境 step 返回值比原生 PPO 多：`termination_ids` 和 `termination_privileged_obs`，用于构造终止时正确的 next critic observation。第三，`HIMPPO.update()` 在每个 mini-batch 中额外调用 `self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)`，得到 `estimation_loss` 和 `swap_loss`。

因此 HIM 可以理解为“双优化器耦合”的训练：PPO 主优化器更新 actor、critic 以及 estimator 参数所在 actor_critic 的整体参数；estimator 自己还有一个 Adam optimizer，在 mini-batch 内先用辅助 loss 更新 estimator。需要注意，`HIMActorCritic.update_distribution()` 里 estimator 前向被包在 `torch.no_grad()` 中，因此 PPO actor loss 不会通过策略分布反向更新 estimator；estimator 主要由自身的 `update()` 训练。critic loss 也不依赖 estimator，因为 critic 直接吃 privileged observation。

## 3. POMDP、历史和 belief 表征的数学直觉

在 POMDP 中，智能体每一步看到的是观测 `o_t`，而不是完整状态 `s_t`。单个 `o_t` 可能无法确定真实速度、接触相位或地形属性。历史 `h_t=(o_{t-K+1},…,o_t)` 包含状态转移留下的痕迹，可以作为 belief state 的近似。理想情况下，策略写成 `π(a_t|h_t)`。但直接把长历史输入大 actor 会增加维度和训练难度，所以 HIM 使用 encoder `e_φ(h_t)` 得到低维表示：

```text
(v̂_t,z_t)=e_φ(h_t),
```

然后策略变成：

```text
π_θ(a_t|o_t, v̂_t, z_t).
```

这里 `v̂_t` 是显式速度估计，`z_t` 是没有人工命名的隐变量。速度估计让网络学习一个有物理意义的辅助任务；隐变量则吸收速度之外的可辨识因素，比如地形、摩擦、接触模式或动力学扰动。虽然代码没有直接命名 latent 的语义，但 prototype 和 swap loss 鼓励它形成有结构的聚类式表征。

## 4. `HIMActorCritic` 网络结构：actor 输入为什么是当前观测加估计量

`rsl_rl/rsl_rl/modules/him_actor_critic.py` 中 `HIMActorCritic` 的构造函数多了 `num_one_step_obs`。它先计算 `history_size`，再设定 actor 输入维度为 `num_one_step_obs + 3 + 16`，critic 输入维度仍为 `num_critic_obs`。estimator 是 `HIMEstimator(temporal_steps=self.history_size, num_one_step_obs=num_one_step_obs)`。

actor 的 MLP 和原生 `ActorCritic` 类似，默认隐藏层更大时也可以通过配置传入。critic 同样是 MLP，输出一维 value。策略噪声仍由 `self.std` 表示，分布仍是对角高斯 Normal。也就是说 HIM 的 action distribution 设计没有改变，改变的是 actor 的输入表征。

`update_distribution(obs_history)` 是理解 HIMActorCritic 的关键。它在 `torch.no_grad()` 下调用 `vel, latent = self.estimator(obs_history)`，然后取 `obs_history[:, :self.num_one_step_obs]` 作为当前单步观测，与 `vel`、`latent` 拼接成 actor input。这里默认历史观测的前 `num_one_step_obs` 维代表当前单步观测；如果环境拼接历史的顺序不同，就会直接影响策略输入语义。因此 HIM 对 observation layout 有强假设。

`act_inference(obs_history, observations=None)` 也使用 estimator，但不加 `no_grad()` 包装；推理时通常在外层 eval/no_grad 环境中调用。它返回 actor 均值动作。部署时只需要历史观测，不需要 privileged critic observation，这符合“critic 可用训练特权信息、actor 部署只用可观测历史”的原则。

## 5. `HIMEstimator`：encoder、target、prototype 和两个辅助损失

`rsl_rl/rsl_rl/modules/him_estimator.py` 中 `HIMEstimator` 是 HIM 的核心新增模块。构造函数参数包括 `temporal_steps`、`num_one_step_obs`、encoder hidden dims、target hidden dims、learning rate、gradient clipping、prototype 数量和 temperature。encoder 输入维度是 `temporal_steps * num_one_step_obs`，输出维度是 `enc_hidden_dims[-1] + 3`。默认 `enc_hidden_dims[-1]` 是 16，所以 encoder 输出前三维速度、后 16 维 latent。

Target 网络输入单步 next observation，输出 16 维 target latent。Prototype 是 `nn.Embedding(num_prototype, enc_hidden_dims[-1])`，默认 32 个 prototype，每个 prototype 是 16 维向量。训练时先把 encoder latent `z_s` 和 target latent `z_t` 做 L2 normalize，再把 prototype 权重 normalize，然后计算分数矩阵：

```text
score_s=z_s Wᵀ, score_t=z_t Wᵀ.
```

`update()` 中的速度监督来自 `next_critic_obs[:, self.num_one_step_obs:self.num_one_step_obs+3]`，目标 next observation 来自 `next_critic_obs[:, 3:self.num_one_step_obs+3]`。这说明代码假设 privileged observation 的某些切片具有固定语义：从 `num_one_step_obs` 开始的三维是速度监督目标，从索引 3 到 `num_one_step_obs+3` 是 target 网络的 next observation。文档和使用者必须明确：如果环境 privileged observation 布局不同，estimator 的监督信号就会错位。

速度损失是：

```text
L_vel=‖v̂_t-v_t‖_2².
```

Swap loss 使用 Sinkhorn 得到平衡 assignment `q_s`、`q_t`，再让 source 分配监督 target prediction、target 分配监督 source prediction：

```text
L_swap = -1/2 E[q_s log p_t + q_t log p_s].
```

其中 `p_s=softmax(score_s/T)`，`p_t=softmax(score_t/T)`。这种结构与 SwAV 一类自监督聚类思想相似：不要求每个 latent 对应人工标签，而是通过 prototype assignment 让不同 view 的表示在聚类空间中互相预测，同时用 Sinkhorn 防止所有样本坍缩到同一个 prototype。

## 6. Sinkhorn：为什么要平衡 assignment

`sinkhorn(out, eps=0.05, iters=3)` 先对分数取指数并转置成 `[K,B]`，其中 `K` 是 prototype 数量，`B` 是 batch size。然后把整个矩阵归一化，再交替归一化行和列：每行总量被调成 `1/K`，每列总量被调成 `1/B`，最后返回 `[B,K]` 的 assignment。数学上这是一个近似求解熵正则最优传输分配的问题。

如果没有平衡约束，自监督聚类很容易坍缩：网络把所有样本都分到一个 prototype，也能在某些损失下得到表面上低损失。Sinkhorn 的行约束要求每个 prototype 接收相近总质量，列约束要求每个样本的 assignment 有正确归一化尺度。这样 latent 被迫利用多个 prototype，表征空间更有结构。

在 HIM 中，prototype 不一定对应可命名类别。它可能把不同速度、接触模式、地形反应或动力学条件分成若干簇。对 actor 来说，重要的不是 prototype 名称，而是 latent 能否提供对控制有用的上下文。

## 7. `HIMRolloutStorage`：为什么要保存 next privileged observation

`HIMRolloutStorage` 和原生 `RolloutStorage` 的主体很像，仍然保存 observations、privileged observations、rewards、actions、dones、actions log prob、values、returns、advantages、mu 和 sigma。新增的是 `next_privileged_observations`，transition 也新增 `next_critic_observations`。

这个新增字段只服务 estimator 更新。`HIMPPO.update()` 的 mini-batch 生成器返回 `next_critic_obs_batch`，然后 `HIMEstimator.update(obs_batch, next_critic_obs_batch, lr=...)` 从中取速度监督和 next observation view。原生 PPO 不需要 next critic obs，因为 GAE 只需要 rollout 末尾的 last value 和 storage 内保存的 value/reward/done；HIM estimator 需要每个样本对应的 next privileged observation 来构造辅助目标。

storage 的 mini-batch 逻辑仍是 flatten `[T,N]` 后随机取索引。关键是当前 obs、当前 critic obs、action、old log prob、advantage、return 和 next critic obs 必须用同一个 `batch_idx` 取出。这样 estimator 的 input history 和 target next observation 才来自同一条 transition。

## 8. `HIMOnPolicyRunner`：终止观测替换为什么重要

`HIMOnPolicyRunner.learn()` 与原生 runner 最大的差异在环境 step 返回值：

`obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)`。

随后 runner 构造：

`next_critic_obs = critic_obs.clone().detach()`，并执行 `next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()`。

这段代码处理的是并行环境 reset 后 next observation 的歧义。很多向量化环境在某个子环境 done 后会立即 reset，并把返回的 `obs` 填成新 episode 的初始观测。但 estimator 的 transition 辅助目标需要的是“当前动作之后、终止瞬间”的 next privileged observation，而不是 reset 后新 episode 的第一个观测。`termination_privileged_obs` 就是为了保留这个终止观测。如果不替换，done 样本的 next target 会跨 episode 错位，速度监督和 latent swap 训练都会被污染。

这也是 HIM storage 必须保存 next privileged obs 的原因：HIM 的辅助学习比 PPO 主损失更依赖 transition 的 next state 语义。PPO 主损失主要关心当前 `(obs, action, old_log_prob, advantage, return)`；HIM estimator 还关心 `(history_t, privileged_{t+1})`。

## 9. HIMPPO 更新顺序：辅助估计和主 PPO 的耦合方式

`HIMPPO.update()` 对每个 mini-batch 先调用 `self.actor_critic.act(obs_batch)` 重建当前策略分布，再计算 log prob、value、entropy 和 KL。之后调用 estimator update，随后计算 PPO surrogate 和 value loss，最后用主 optimizer 更新 actor_critic。这个顺序意味着 estimator 在同一个 mini-batch 上先被辅助 optimizer 推一步，然后 actor/critic 再按 PPO loss 更新。

不过，由于 `HIMActorCritic.update_distribution()` 对 estimator 前向使用 `torch.no_grad()`，actor 的 PPO loss 不会把梯度传回 estimator。因此 estimator 的主要学习信号来自 `estimation_loss + swap_loss`。主 optimizer 参数列表包含 actor_critic 的全部参数，但在 actor loss 中 estimator 没有梯度；critic loss 也不经过 estimator。除非其他路径产生梯度，否则主 optimizer 对 estimator 参数没有实际更新。这种设计把“学表征”和“用表征控制”分开，降低 PPO noisy gradient 对 estimator 的干扰。

返回值上，代码累加了 `mean_estimation_loss` 和 `mean_swap_loss`，但最后返回的是 `estimation_loss, swap_loss` 而不是均值变量。这是当前实现中的一个小不一致：日志里 runner 接收 `mean_estimation_loss, mean_swap_loss`，但算法返回最后一个 mini-batch 的 loss 值。文档应如实说明这一点，而不是把它解释成真正均值。

## 10. HIM 的数学目标可以怎样理解

HIM 的总训练目标可概念化为：

```text
min_θ,ψ,φ L_PPO(θ,ψ; e_φ(h)) + α L_vel(φ) + β L_swap(φ),
```

其中 `θ` 是 actor 参数，`ψ` 是 critic 参数，`φ` 是 estimator 参数。实际代码中并没有把辅助损失加到 PPO 主 loss 里一起 backward，而是 estimator 自己先 backward 和 step。因此更准确的工程描述是“同一批 on-policy 数据上交替执行 estimator auxiliary update 和 PPO actor-critic update”。

速度 MSE 给 latent encoder 一个物理锚点：历史里必须包含足够信息预测身体速度。Swap loss 给 latent 空间一个结构锚点：历史 view 和 next-observation view 在 prototype 空间中应该互相一致。PPO loss 给 actor 一个任务锚点：使用 estimator 输出后，动作必须提高长期回报。三个目标结合起来，HIM 试图把“状态估计”和“控制策略”放在一个训练循环里共同成长。

## 11. 与 RNN 策略的对比

同样面对部分可观测性，RNN 策略会把历史压缩进隐藏状态，并让 actor/critic 端到端学习。HIM 的做法更显式：历史一次性拼接为 `obs_history`，estimator 输出低维速度和 latent，actor 只看压缩结果。这种方式的优点是推理结构简单，不需要维护 recurrent hidden state；训练 storage 也不需要 padding trajectory；辅助损失更容易约束表征。缺点是历史长度固定，encoder 只能看窗口内信息；如果关键状态需要更长记忆，固定窗口可能不足。

对腿式运动控制来说，固定历史窗口常常已经能推断速度、延迟、接触相位和短期扰动，因为机器人动力学有连续性。HIM 因此是一种介于 frame stacking 和 recurrent policy 之间的设计：比纯 frame stacking 更有结构，比 RNN 更容易批量训练和部署。

## 12. 环境接口和配置前提

HIM runner 对环境的要求比原生 runner 更强。除了 `num_envs`、`num_obs`、`num_privileged_obs`、`num_actions`、`get_observations()`、`get_privileged_observations()` 等基础接口，还要求环境提供 `num_one_step_obs`，并且 `step(actions)` 返回终止 id 和终止 privileged observation。`rsl_rl/rsl_rl/env/vec_env.py` 的抽象接口没有声明这些额外返回值，因此 HIM runner 实际上约定了一个更具体的环境协议。

这对实现环境的人非常重要。只把 runner_class_name 改成 `HIMOnPolicyRunner` 不够；环境 observation layout、privileged observation layout、termination observation 返回都要匹配 HIMEstimator 的切片假设。如果 `num_one_step_obs` 错，`history_size` 就错；如果 privileged obs 中速度切片错，MSE 监督目标就错；如果 done 后没有 termination privileged obs，next target 就可能来自 reset 后状态。

## 13. 训练日志和 checkpoint 的 HIM 差异

`HIMOnPolicyRunner.log()` 在原生指标之外增加 `Loss/Estimation Loss` 和 `Loss/Swap Loss`。这两个指标应与 PPO 指标一起看。如果 reward 上升但 estimation loss 长期很高，说明 actor 可能没有充分利用速度估计，或者速度切片语义不对。如果 swap loss 很快异常接近零，也要警惕 prototype collapse 或 assignment 退化；需要结合 latent 分布进一步诊断。

保存时，HIM checkpoint 除了 `model_state_dict` 和主 optimizer，还保存 `estimator_optimizer_state_dict`。加载时如果 `load_optimizer=True`，会恢复两个 optimizer。这个细节必要，因为 estimator 有独立 Adam 状态。如果只保存 actor_critic 参数而不保存 estimator optimizer，resume 后辅助学习的动量和自适应二阶矩会丢失，短期训练曲线可能抖动。

## 14. HIM 的部署路径

部署时调用 `HIMOnPolicyRunner.get_inference_policy()` 返回 `HIMActorCritic.act_inference`。输入是历史观测 `obs_history`，输出是 actor 均值动作。critic、storage、HIMPPO update、estimator optimizer 都不参与部署。部署系统需要做的主要工作是按照训练时相同顺序维护历史观测窗口，并保证当前单步观测位于 `obs_history[:, :num_one_step_obs]`，因为 actor 输入会取这一段。

这点经常被忽视。HIM 的部署不是只加载网络就行，还要复刻 observation stacking 约定。如果训练时历史顺序是“当前帧在前，旧帧在后”，部署也必须一样；如果训练时做了归一化或裁剪，部署也必须一样。否则 estimator 看到的时间结构会变，速度和 latent 都会偏离训练分布。

## 15. HIM 与原生 PPO 的共同不变量

尽管 HIM 新增很多估计逻辑，它仍然继承原生 PPO 的不变量。old log probability 必须对应采样动作；advantages 必须对应同一 transition；returns 必须由 rollout reward/value/done 计算；ratio 必须用新策略 log prob 和旧 log prob；KL 必须用同一 batch 的 old/new Gaussian 参数；gradient clipping 保护主网络；storage 每轮 update 后 clear。HIM 的额外 next critic obs 不能破坏这些主 PPO 对齐关系。

因此调试 HIM 时应该先确认 PPO 主干健康，再看 estimator。若 surrogate/value loss 已经异常，问题可能不在 HIM latent，而在 rollout、reward、done 或配置。只有主干稳定时，estimation_loss 和 swap_loss 的诊断才有意义。

## 16.1 发散设计思考：历史观测窗口

HIM 假设 actor observation 是固定长度历史。历史提供速度和隐状态线索，但也要求环境和部署端严格保持拼接顺序。

从算法设计角度看，历史观测窗口体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 历史观测窗口 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，历史观测窗口可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，历史观测窗口还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.2 发散设计思考：速度辅助监督

三维速度估计让 estimator 学到物理可解释量。它把纯任务回报之外的监督信号注入表征学习。

从算法设计角度看，速度辅助监督体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 速度辅助监督 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，速度辅助监督可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，速度辅助监督还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.3 发散设计思考：latent 表征

16 维 latent 没有手工语义，但通过 prototype swap loss 获得结构约束，可表达摩擦、地形或接触模式等隐藏因素。

从算法设计角度看，latent 表征体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 latent 表征 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，latent 表征可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，latent 表征还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.4 发散设计思考：prototype 机制

prototype 是可学习聚类中心。latent 与 prototype 点积形成分类分数，Sinkhorn 让 assignment 更均衡。

从算法设计角度看，prototype 机制体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 prototype 机制 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，prototype 机制可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，prototype 机制还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.5 发散设计思考：Sinkhorn 平衡

交替归一化防止所有样本落到同一个 prototype。它把自监督目标从普通分类变成带平衡约束的分配问题。

从算法设计角度看，Sinkhorn 平衡体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 Sinkhorn 平衡 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，Sinkhorn 平衡可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，Sinkhorn 平衡还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.6 发散设计思考：next privileged obs

estimator 更新依赖 transition 后继状态。done 时必须使用终止瞬间 privileged obs，而不是 reset 后新 episode 观测。

从算法设计角度看，next privileged obs体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 next privileged obs 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，next privileged obs可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，next privileged obs还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.7 发散设计思考：双优化器

HIMPPO 主 optimizer 更新 actor_critic，estimator 还有自己的 optimizer。辅助训练和 PPO 更新在同一 mini-batch 上交替发生。

从算法设计角度看，双优化器体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 双优化器 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，双优化器可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，双优化器还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.8 发散设计思考：no_grad estimator

actor 构造分布时对 estimator 使用 no_grad，说明 PPO 策略梯度不直接塑造 estimator，降低梯度耦合噪声。

从算法设计角度看，no_grad estimator体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 no_grad estimator 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，no_grad estimator可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，no_grad estimator还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.9 发散设计思考：critic privileged

critic 使用 privileged observation 学价值，actor 使用历史估计量决策。这个分工保留了 asymmetric actor-critic 的部署安全性。

从算法设计角度看，critic privileged体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 critic privileged 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，critic privileged可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，critic privileged还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.10 发散设计思考：环境协议

HIM runner 要求环境返回 termination_ids 和 termination_privileged_obs。抽象 VecEnv 没有声明这一点，实际使用时要额外实现。

从算法设计角度看，环境协议体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 环境协议 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，环境协议可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，环境协议还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.11 发散设计思考：观测切片假设

HIMEstimator 对 next_critic_obs 的速度和 next_obs 切片写死。环境 observation 布局必须与代码假设一致。

从算法设计角度看，观测切片假设体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 观测切片假设 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，观测切片假设可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，观测切片假设还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.12 发散设计思考：日志解释

Estimation loss 和 Swap loss 是表征学习健康度指标，但不能脱离 reward、KL、value loss 单独判断。

从算法设计角度看，日志解释体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 日志解释 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，日志解释可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，日志解释还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.13 发散设计思考：部署一致性

推理只用历史观测，但历史构造、归一化和当前帧位置必须与训练完全一致。

从算法设计角度看，部署一致性体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 部署一致性 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，部署一致性可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，部署一致性还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.14 发散设计思考：与 RNN 对比

HIM 用固定窗口加 encoder 替代 recurrent hidden state，训练更简单，记忆长度也更固定。

从算法设计角度看，与 RNN 对比体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 与 RNN 对比 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，与 RNN 对比可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，与 RNN 对比还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.15 发散设计思考：PPO 主干继承

HIM 没有改变 clipped surrogate、GAE、KL schedule 和 Gaussian policy 的核心逻辑，只扩展输入表征与辅助损失。

从算法设计角度看，PPO 主干继承体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 PPO 主干继承 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，PPO 主干继承可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，PPO 主干继承还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.16 发散设计思考：历史观测窗口

HIM 假设 actor observation 是固定长度历史。历史提供速度和隐状态线索，但也要求环境和部署端严格保持拼接顺序。

从算法设计角度看，历史观测窗口体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 历史观测窗口 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，历史观测窗口可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，历史观测窗口还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.17 发散设计思考：速度辅助监督

三维速度估计让 estimator 学到物理可解释量。它把纯任务回报之外的监督信号注入表征学习。

从算法设计角度看，速度辅助监督体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 速度辅助监督 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，速度辅助监督可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，速度辅助监督还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.18 发散设计思考：latent 表征

16 维 latent 没有手工语义，但通过 prototype swap loss 获得结构约束，可表达摩擦、地形或接触模式等隐藏因素。

从算法设计角度看，latent 表征体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 latent 表征 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，latent 表征可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，latent 表征还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.19 发散设计思考：prototype 机制

prototype 是可学习聚类中心。latent 与 prototype 点积形成分类分数，Sinkhorn 让 assignment 更均衡。

从算法设计角度看，prototype 机制体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 prototype 机制 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，prototype 机制可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，prototype 机制还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.20 发散设计思考：Sinkhorn 平衡

交替归一化防止所有样本落到同一个 prototype。它把自监督目标从普通分类变成带平衡约束的分配问题。

从算法设计角度看，Sinkhorn 平衡体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 Sinkhorn 平衡 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，Sinkhorn 平衡可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，Sinkhorn 平衡还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.21 发散设计思考：next privileged obs

estimator 更新依赖 transition 后继状态。done 时必须使用终止瞬间 privileged obs，而不是 reset 后新 episode 观测。

从算法设计角度看，next privileged obs体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 next privileged obs 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，next privileged obs可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，next privileged obs还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.22 发散设计思考：双优化器

HIMPPO 主 optimizer 更新 actor_critic，estimator 还有自己的 optimizer。辅助训练和 PPO 更新在同一 mini-batch 上交替发生。

从算法设计角度看，双优化器体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 双优化器 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，双优化器可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，双优化器还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.23 发散设计思考：no_grad estimator

actor 构造分布时对 estimator 使用 no_grad，说明 PPO 策略梯度不直接塑造 estimator，降低梯度耦合噪声。

从算法设计角度看，no_grad estimator体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 no_grad estimator 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，no_grad estimator可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，no_grad estimator还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.24 发散设计思考：critic privileged

critic 使用 privileged observation 学价值，actor 使用历史估计量决策。这个分工保留了 asymmetric actor-critic 的部署安全性。

从算法设计角度看，critic privileged体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 critic privileged 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，critic privileged可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，critic privileged还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.25 发散设计思考：环境协议

HIM runner 要求环境返回 termination_ids 和 termination_privileged_obs。抽象 VecEnv 没有声明这一点，实际使用时要额外实现。

从算法设计角度看，环境协议体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 环境协议 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，环境协议可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，环境协议还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.26 发散设计思考：观测切片假设

HIMEstimator 对 next_critic_obs 的速度和 next_obs 切片写死。环境 observation 布局必须与代码假设一致。

从算法设计角度看，观测切片假设体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 观测切片假设 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，观测切片假设可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，观测切片假设还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.27 发散设计思考：日志解释

Estimation loss 和 Swap loss 是表征学习健康度指标，但不能脱离 reward、KL、value loss 单独判断。

从算法设计角度看，日志解释体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 日志解释 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，日志解释可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，日志解释还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.28 发散设计思考：部署一致性

推理只用历史观测，但历史构造、归一化和当前帧位置必须与训练完全一致。

从算法设计角度看，部署一致性体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 部署一致性 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，部署一致性可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，部署一致性还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.29 发散设计思考：与 RNN 对比

HIM 用固定窗口加 encoder 替代 recurrent hidden state，训练更简单，记忆长度也更固定。

从算法设计角度看，与 RNN 对比体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 与 RNN 对比 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，与 RNN 对比可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，与 RNN 对比还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.30 发散设计思考：PPO 主干继承

HIM 没有改变 clipped surrogate、GAE、KL schedule 和 Gaussian policy 的核心逻辑，只扩展输入表征与辅助损失。

从算法设计角度看，PPO 主干继承体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 PPO 主干继承 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，PPO 主干继承可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，PPO 主干继承还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.31 发散设计思考：历史观测窗口

HIM 假设 actor observation 是固定长度历史。历史提供速度和隐状态线索，但也要求环境和部署端严格保持拼接顺序。

从算法设计角度看，历史观测窗口体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 历史观测窗口 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，历史观测窗口可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，历史观测窗口还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.32 发散设计思考：速度辅助监督

三维速度估计让 estimator 学到物理可解释量。它把纯任务回报之外的监督信号注入表征学习。

从算法设计角度看，速度辅助监督体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 速度辅助监督 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，速度辅助监督可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，速度辅助监督还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.33 发散设计思考：latent 表征

16 维 latent 没有手工语义，但通过 prototype swap loss 获得结构约束，可表达摩擦、地形或接触模式等隐藏因素。

从算法设计角度看，latent 表征体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 latent 表征 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，latent 表征可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，latent 表征还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.34 发散设计思考：prototype 机制

prototype 是可学习聚类中心。latent 与 prototype 点积形成分类分数，Sinkhorn 让 assignment 更均衡。

从算法设计角度看，prototype 机制体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 prototype 机制 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，prototype 机制可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，prototype 机制还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 16.35 发散设计思考：Sinkhorn 平衡

交替归一化防止所有样本落到同一个 prototype。它把自监督目标从普通分类变成带平衡约束的分配问题。

从算法设计角度看，Sinkhorn 平衡体现的是“把不可直接观测的控制上下文变成可学习中间量”。在腿式机器人任务中，actor 直接根据单帧观测输出动作，经常会把速度误差、接触相位、地面扰动和动力学延迟混在一起处理。HIM 的 encoder 把历史中的时间信息压缩出来，让 actor 接收更接近 belief state 的输入。这样做不是为了追求网络形式复杂，而是为了把任务分解成两个更容易学习的问题：先估计隐藏上下文，再基于上下文做控制。

实现 Sinkhorn 平衡 时最重要的风险是语义错位。HIMEstimator 的监督目标来自 `next_critic_obs` 的固定切片，actor 的当前观测来自 `obs_history` 的前 `num_one_step_obs` 维，terminal 样本的 next privileged observation 来自 runner 替换逻辑。任何一个位置错了，模型仍然可以反向传播，loss 也会有数值，但学到的物理语义已经偏离。相比普通 PPO，HIM 的调试难点恰恰在于这种“静默错位”：张量 shape 对了，不代表时间语义和物理语义对了。

数学上，Sinkhorn 平衡可以被看成 POMDP belief approximation 的一部分。历史 `h_t` 不是状态本身，但包含状态转移的可辨识信息；encoder `e_φ(h_t)` 不是精确贝叶斯滤波器，但能在任务数据分布上学习一个足够好用的低维统计量。速度 MSE 提供有监督 anchor，swap loss 提供无监督结构，PPO reward 提供控制目标。三者的结合让 latent 既不能完全脱离物理，也不被单一物理标签限制。

工程上，Sinkhorn 平衡还带来部署约束。部署端必须维护和训练一致的 history buffer，保证当前帧、旧帧、归一化和裁剪都一致。如果训练中 actor 输入是“当前观测 + 估计速度 + latent”，部署中就不能临时改成完整历史或不同顺序的堆叠。HIM 的优势来自结构化约束，结构化约束也意味着接口契约更严格。

## 17. 代码阅读顺序建议

建议先读 `HIMOnPolicyRunner.learn()`，因为它揭示了 HIM 和原生 PPO 的接口差异：环境 step 多返回终止观测，runner 构造 `next_critic_obs`。再读 `HIMRolloutStorage`，确认 next privileged observation 如何随 transition 保存和 mini-batch 返回。第三步读 `HIMPPO.update()`，看 estimator update 插入在 PPO 主损失之前。第四步读 `HIMActorCritic.update_distribution()`，理解 actor 输入如何由当前单步观测、速度估计和 latent 拼接。最后读 `HIMEstimator.update()`，把速度切片、target view、prototype、Sinkhorn 和 swap loss 串起来。

这种阅读顺序从运行时数据流出发，能避免一开始陷入 estimator 数学细节。HIM 的核心不是某一个公式，而是“历史观测如何变成 actor 可用上下文，以及这个上下文如何被辅助目标训练”。

## 18. 常见错误和检查清单

第一，检查 `num_actor_obs` 是否能被 `num_one_step_obs` 整除，否则 `history_size` 的语义不成立。第二，检查历史拼接顺序，确认当前观测确实在前 `num_one_step_obs` 维。第三，检查 privileged observation 中速度切片是否与 `next_critic_obs[:, num_one_step_obs:num_one_step_obs+3]` 一致。第四，检查 `termination_ids` 是否是一维索引或可用于 tensor indexing 的 id 集合。第五，检查 `termination_privileged_obs` 是否对应 reset 前的终止状态。第六，检查 estimator optimizer 是否在 checkpoint 中保存和恢复。第七，检查 `mean_estimation_loss` 日志是否实际反映平均值；当前代码返回最后一个 mini-batch 的 `estimation_loss` 和 `swap_loss`，这是需要读者注意的实现细节。

## 19. 总结

HIMLoco/HIMPPO 在本仓库中可以理解为“PPO 主干 + 历史状态估计器 + prototype 自监督表征”的组合。PPO 主干负责稳定优化连续控制策略，critic 使用 privileged observation 提供价值监督；HIMEstimator 从历史观测中估计速度和 latent，使 actor 在部署可观测信息下获得更接近真实状态的上下文；HIMRolloutStorage 和 HIMOnPolicyRunner 则为 estimator 提供 next privileged observation，尤其处理 done/reset 边界。相比原生 PPO，HIM 的代码复杂度主要来自时序语义和辅助目标，而不是 PPO 公式本身。只要把当前观测、历史观测、当前 privileged obs、next privileged obs、terminal privileged obs 的时间关系理清，HIM 的训练流程就是一条可解释的扩展链路。
