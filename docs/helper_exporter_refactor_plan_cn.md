# helper/exporter 导出逻辑重构方案（详细版）

## 1. 目标

当前项目同时融合了多套训练算法和网络结构：

- 原生 `legged_gym`：PPO、Recurrent PPO。
- `go2_rl_gym`：CTS、MoE-CTS、MCP-CTS、AC-MoE、Dual-MoE 等。
- 当前项目：HIM / himloco 风格的 `HIMActorCritic`、`HIMPPO`、`HIMOnPolicyRunner`。

这些算法的导出方式不一致。普通 PPO 只需要导出 `actor`，Recurrent PPO 需要导出 RNN hidden state，CTS/MoE 需要维护历史观测和 encoder，HIM 则需要把 `estimator.encoder + actor` 包成一个推理图。

因此后续重构的目标是：

1. `helpers.py` 只保留通用 helper，不再承载 JIT/ONNX 导出逻辑。
2. 所有 JIT / ONNX / PKL 导出统一放到 `legged_gym/utils/exporter.py`。
3. `play.py` 和 runner 不直接判断 `actor_critic`、`model`、`estimator`、`student_encoder` 等内部字段。
4. 不同训练任务通过配置选择 runner 和导出类型，自动检测只作为兜底。
5. 新增算法时，只需要在 `exporter.py` 增加 wrapper 和分支，不再污染 helper、play 或训练逻辑。

本文件只描述修改方案和参考代码，不表示当前代码已经完成这些修改。

## 2. 已检查文件

当前项目：

```text
legged_gym/utils/helpers.py
legged_gym/utils/exporter.py
legged_gym/utils/__init__.py
legged_gym/scripts/play.py
legged_gym/utils/task_registry.py
rsl_rl/rsl_rl/runners/on_policy_runner.py
rsl_rl/rsl_rl/runners/him_on_policy_runner.py
rsl_rl/rsl_rl/runners/on_policy_runner_cts.py
rsl_rl/rsl_rl/modules/actor_critic.py
rsl_rl/rsl_rl/modules/actor_critic_recurrent.py
rsl_rl/rsl_rl/modules/actor_critic_cts.py
rsl_rl/rsl_rl/modules/actor_critic_moe_cts.py
rsl_rl/rsl_rl/modules/him_actor_critic.py
rsl_rl/rsl_rl/modules/him_estimator.py
```

参考项目：

```text
/home/kk/github/go2_rl_gym/legged_gym/utils/helpers.py
/home/kk/github/go2_rl_gym/legged_gym/utils/exporter.py
/home/kk/github/go2_rl_gym/legged_gym/scripts/play.py
/home/kk/github/go2_rl_gym/legged_gym/utils/task_registry.py
/home/kk/github/legged_gym/legged_gym/utils/helpers.py
```

未发现当前项目存在 `ARCHITECTURE_CONTEXT.md`。

## 3. 现有实现分析

### 3.1 原生 legged_gym 的导出方式

原生 `legged_gym` 把导出逻辑放在 `helpers.py` 里，核心函数是：

```python
def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
```

它只考虑两类情况：

- `actor_critic.actor`：普通 MLP policy。
- `actor_critic.memory_a`：LSTM/Recurrent policy。

这个设计适合原生 PPO，但不适合当前融合项目。原因是 CTS/MoE/HIM 的推理图不只是 actor。

### 3.2 go2_rl_gym 的导出方式

`go2_rl_gym` 已经把导出逻辑迁移到了：

```text
legged_gym/utils/exporter.py
```

它的 `helpers.py` 中旧的 `export_policy_as_jit()` 和 `PolicyExporterLSTM` 已被注释掉，说明它已经完成了职责拆分：

```text
helpers.py  -> 通用 helper
exporter.py -> policy 导出
```

`go2_rl_gym` 的 `exporter.py` 支持：

| 结构 | 识别字段 | JIT 导出行为 | ONNX 导出行为 |
|---|---|---|---|
| PPO | `actor` | 直接导出 actor | 使用 `forward_ppo` |
| Recurrent PPO | `is_recurrent`, `memory_a.rnn` | 导出 RNN + actor + hidden state | ONNX 支持有限 |
| CTS | `student_encoder` | 维护 history，encoder 输出 latent，再拼 obs 输入 actor | flatten history 后导出 |
| MoE-CTS | `student_moe_encoder` | 输出 action + weights + latent | 多输出 |
| MCP-CTS | `actor_mcp` | actor 接收 full obs/no-goal obs 两路输入 | 多输出 |
| AC-MoE | `actor_moe` | actor 输出 action + weights | 当前 JIT 有分支 |
| Dual-MoE | `student_moe_encoder` + `actor_moe` | student moe + actor moe 双权重输出 | 当前 JIT 有分支 |

`go2_rl_gym` 的 `play.py` 也比当前项目更通用：

```python
if hasattr(runner.alg, 'actor_critic'):
    model = runner.alg.actor_critic
else:
    model = runner.alg.model

export_policy_as_jit(model, path)
export_policy_as_onnx(model, path)
export_policy_as_pkl(model, path)
```

这说明它已经意识到不同 runner 的模型字段不同：

- PPO/HIM 风格：`runner.alg.actor_critic`
- CTS/MoE 风格：`runner.alg.model`

但是这段逻辑仍然写在 `play.py` 中。当前项目建议进一步封装到 `exporter.py`。

### 3.3 当前项目 exporter.py 的状态

当前项目的：

```text
legged_gym/utils/exporter.py
```

与 `go2_rl_gym` 的 `exporter.py` 内容一致，已经具备 CTS/MoE/MCP 等导出能力。

现有导出入口：

```python
def export_policy_as_jit(policy: object, path: str, normalizer: Optional[object] = None, filename="policy.pt"):
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: Optional[object] = None, filename="policy.onnx", verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_policy_as_pkl(policy: nn.Module, path: str, filename="policy.pkl"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model_dict = policy.state_dict()
    torch.save(model_dict, os.path.join(path, filename))
```

当前问题是：

- 它还不支持 HIM/himloco。
- 它没有统一的 `detect_export_type()`。
- 它没有 `resolve_policy_from_runner()`。
- 它没有高层 `export_policy()`，导致 `play.py` 仍要自己组合 JIT/ONNX/PKL。

### 3.4 当前项目 helpers.py 的状态

当前项目的 `helpers.py` 仍然包含 HIM 导出逻辑：

```python
def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'estimator'):
        exporter = PolicyExporterHIM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_1.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)
```

HIM JIT wrapper：

```python
class PolicyExporterHIM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
        self.num_one_step_obs = _resolve_him_num_one_step_obs(actor_critic)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        current_obs = obs_history[:, :self.num_one_step_obs]
        return self.actor(torch.cat((current_obs, vel, z), dim=1))
```

HIM ONNX wrapper：

```python
class ONNXPolicyExporterHIM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).eval()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).eval()
        self.num_one_step_obs = _resolve_him_num_one_step_obs(actor_critic)

    def forward(self, obs_history):
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        current_obs = obs_history[:, :self.num_one_step_obs]
        return self.actor(torch.cat((current_obs, vel, z), dim=1))
```

这部分逻辑本身是有价值的，但位置不对。它应该迁移到 `exporter.py`，并纳入统一分发。

### 3.5 当前项目 play.py 的状态

当前 `play.py` 导入方式：

```python
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import export_policy_as_onnx
```

问题：

- JIT 从 `legged_gym.utils` 导入，但当前 `utils/__init__.py` 又从 `helpers.py` re-export。
- ONNX 直接从 `helpers.py` 导入。
- 两者都没有统一进入 `exporter.py`。

当前导出调用：

```python
export_policy_as_jit(ppo_runner.alg.actor_critic, path)
```

问题：

- 硬编码 `actor_critic`。
- 对 `runner.alg.model` 的 CTS/MoE 不兼容。
- 没有 ONNX/PKL 的统一格式控制。
- `play.py` 被迫知道算法内部结构。

### 3.6 当前项目 task_registry.py 的状态

当前 `task_registry.py` 中 runner 创建被硬编码为：

```python
runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
```

问题：

- 即使 config 中存在 `runner_class_name = "OnPolicyRunner"`，这里也不会使用。
- PPO、HIM、CTS/MoE 无法通过同一个 registry 选择不同 runner。
- 后续新增训练任务时，需要改 Python 代码，而不是只改 config。

## 4. 当前主要问题总结

### 问题 1：导出职责分散

当前导出逻辑分布在：

```text
helpers.py    -> HIM JIT/ONNX
exporter.py   -> go2 风格 CTS/MoE/PPO 导出
play.py       -> 自己判断导出模型字段
runner_cts.py -> 训练中 RoboGauge 自动 JIT 导出
```

这会导致后续新增算法时很容易出现重复实现和行为不一致。

### 问题 2：helpers.py 职责过重

`helpers.py` 应该是通用工具，不应该依赖具体网络结构：

- 不应该知道 `actor_critic.estimator.encoder`。
- 不应该知道 HIM 的 `num_one_step_obs`。
- 不应该维护 ONNX dummy input。
- 不应该决定导出文件名。

这些都属于 exporter 的职责。

### 问题 3：play.py 不应该理解算法内部结构

`play.py` 的职责应该是：

1. 创建 env。
2. 创建 runner。
3. 加载 policy。
4. 调用统一导出函数。
5. 执行 rollout。

它不应该写：

```python
if hasattr(runner.alg, "actor_critic"):
    ...
else:
    ...
```

更不应该直接知道 HIM、CTS、MoE 的字段。

### 问题 4：导出分发不应该依赖新增 config 字段

不同训练任务可以继续使用已有的 `runner_class_name`、`policy_class_name` 和 `algorithm_class_name` 选择训练路径，但导出路径不应该再要求新增 `export_policy_type` 或 `export_formats` 这类配置字段。

导出应该由 `exporter.py` 根据实际 policy 结构自动识别，例如 `estimator`、`student_encoder`、`student_moe_encoder`、`actor_mcp`、`actor_moe`、`memory_a`、`actor` 等字段。后续新增算法时，只需要在 exporter 中补一个明确分支或注册函数，不需要到各个 config 文件里补导出参数。

## 5. 目标设计

### 5.1 文件职责

重构后的职责建议如下：

```text
legged_gym/utils/helpers.py
  只保留通用函数：
  - class_to_dict
  - update_class_from_dict
  - update_cfg_from_args
  - get_load_path
  - get_args
  - set_seed
  - parse_sim_params

legged_gym/utils/exporter.py
  统一导出中心：
  - detect_export_type
  - resolve_policy_from_runner
  - export_policy
  - export_policy_as_jit
  - export_policy_as_onnx
  - export_policy_as_pkl
  - _TorchPolicyExporter
  - _OnnxPolicyExporter
  - _TorchHIMPolicyExporter
  - _OnnxHIMPolicyExporter

legged_gym/scripts/play.py
  调用 exporter.py，不直接关心模型内部字段。

legged_gym/utils/task_registry.py
  根据 train_cfg.runner_class_name 选择 runner。

rsl_rl/rsl_rl/runners/*.py
  只负责训练、保存 checkpoint、可选触发统一 exporter。
```

### 5.2 导出调用链

建议最终调用链：

```text
play.py
  -> task_registry.make_alg_runner(...)
  -> runner.get_inference_policy(...)
  -> resolve_policy_from_runner(runner)
  -> export_policy(policy, path, formats, export_type)
      -> export_policy_as_jit(...)
      -> export_policy_as_onnx(...)
      -> export_policy_as_pkl(...)
```

训练中自动导出，例如 RoboGauge：

```text
on_policy_runner_cts.py
  -> export_policy_as_jit(self.alg.model, jit_dir, filename=...)
```

后续可以保留这个调用，但它应该继续从 `legged_gym.utils.exporter` 导入，不要回到 `helpers.py`。

## 6. 模型结构与导出类型映射

| export_type | 适用模型 | runner 字段 | policy 关键字段 | 输入 | 输出 |
|---|---|---|---|---|---|
| `ppo` | 原生 ActorCritic | `alg.actor_critic` | `actor` | 当前 obs | actions |
| `recurrent` | ActorCriticRecurrent | `alg.actor_critic` | `memory_a.rnn`, `actor` | 当前 obs + 内部 hidden | actions |
| `him` | HIMActorCritic | `alg.actor_critic` | `estimator.encoder`, `actor`, `num_one_step_obs` | obs_history | actions |
| `cts` | ActorCriticCTS | `alg.model` | `student_encoder`, `history`, `actor` | 当前 obs 或 stack obs | actions / latent |
| `moe_cts` | ActorCriticMoECTS | `alg.model` | `student_moe_encoder`, `history`, `actor` | 当前 obs 或 stack obs | actions, weights, latent |
| `mcp_cts` | MCPCTS | `alg.model` | `actor_mcp`, `obs_no_goal_mask` | 当前 obs 或 stack obs | actions, weights |
| `ac_moe_cts` | ACMoE | `alg.model` | `actor_moe`, `student_encoder` | 当前 obs 或 stack obs | actions, weights, latent |
| `dual_moe_cts` | DualMoE | `alg.model` | `student_moe_encoder`, `actor_moe` | 当前 obs 或 stack obs | actions, student weights, actor weights, latent |

导出类型默认由 `exporter.py` 自动检测。调用方仍可在代码里显式传 `export_type` 做特殊覆盖，但不要求在 config 文件中增加导出字段。

### 6.1 关于 normalization 的专项分析

用户提到的“CTS 训练用了归一化，HIM 和原生似乎没有”，这里需要拆成三类概念。它们对导出的影响不同，不能混在一起处理。

| 类型 | 当前代码位置 | 是否随训练学习/更新 | 是否必须额外传给 exporter | 结论 |
|---|---|---:|---:|---|
| env 固定尺度缩放 | `legged_gym/envs/base/legged_robot.py` 的 `obs_scales`、`clip_observations` | 否 | 否 | env 输出给 policy 的 obs 已经缩放/clip，导出模型不需要再内置 |
| encoder latent norm | CTS/MoE 的 `L2Norm` / `SimNorm`，HIM estimator 的 `F.normalize` | 否，属于网络 forward 结构 | 否 | 已经是模型图的一部分，deepcopy encoder 时会一起导出 |
| empirical normalizer / RunningMeanStd | exporter 参数 `normalizer` 支持，但当前训练 runner 没有创建/保存/传入；HIM 文件里有 `RunningMeanStd` 类但未接入 | 如果启用才会更新 | 启用后必须传 | 当前项目暂时不需要；如果未来启用，JIT/ONNX 都必须包含同一个 normalizer |

#### 6.1.1 env 固定尺度缩放不是 exporter normalizer

当前 env 在计算 obs 时已经做固定尺度缩放：

```python
self.obs_buf = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,
    self.base_ang_vel * self.obs_scales.ang_vel,
    self.projected_gravity,
    self.commands[:, :3] * self.commands_scale,
    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
    self.dof_vel * self.obs_scales.dof_vel,
    self.actions,
), dim=-1)
```

step 返回前又做 clip：

```python
clip_obs = self.cfg.normalization.clip_observations
self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
if self.privileged_obs_buf is not None:
    self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
```

这类 normalization 是环境观测定义的一部分，不是 policy 网络的一部分。导出 policy 时通常不应该把这部分再次放进 JIT/ONNX，否则部署端如果已经按 env 逻辑构造 obs，就会重复缩放。

结论：

- 如果部署端输入的是“已经按训练 env 规则构造好的 obs”，exporter 不需要处理 `obs_scales`。
- 如果部署端输入的是“原始机器人状态”，则应该另写 deployment preprocessor，把 `obs_scales`、clip、obs 拼接顺序放在那里，而不是塞进 policy exporter。

#### 6.1.2 CTS/MoE 的 norm_type 是 latent norm，已经在模型内部

当前 CTS 模型里：

```python
assert norm_type in ["l2norm", "simnorm"]
...
encoder_layers.append(nn.Linear(student_encoder_hidden_dims[l], latent_dim))
if norm_type == "l2norm":
    encoder_layers.append(L2Norm())
elif norm_type == "simnorm":
    encoder_layers.append(SimNorm())
self.student_encoder = nn.Sequential(*encoder_layers)
```

MoE-CTS 的 student encoder 也是一样的思路：

```python
class StudentMoEEncoder(nn.Module):
    def __init__(..., norm_type="l2norm"):
        self.norm_layer = L2Norm() if norm_type == "l2norm" else SimNorm()
        self.moe = MoE(...)

    def forward(self, obs):
        latent, weights = self.moe(obs)
        latent = self.norm_layer(latent)
        return latent, weights
```

这不是输入观测的 running mean/std normalizer，而是 encoder 输出 latent 的确定性网络层：

- `L2Norm` 调用 `F.normalize(x, p=2.0, dim=-1)`。
- `SimNorm` 对 latent 分块后做 `softmax`。

因为这些层已经属于 `student_encoder` / `student_moe_encoder` / `teacher_encoder`，当前 exporter 在 `copy.deepcopy(policy.student_encoder)` 或 `copy.deepcopy(policy.student_moe_encoder)` 时会一起带走它们。

结论：

- CTS/MoE 的 `norm_type` 不需要额外传 `normalizer`。
- 导出时真正要保证的是：JIT/ONNX wrapper 必须调用和 `act_inference()` 一样的 `student_encoder` 或 `student_moe_encoder` 路径。
- 如果 wrapper 绕过 encoder 或手写 latent 计算，就会漏掉 L2Norm/SimNorm。

#### 6.1.3 HIM 的 normalize 也是 latent norm，不是输入 normalizer

HIM 的 `HIMEstimator.forward()` 中：

```python
parts = self.encoder(obs_history.detach())
vel, z = parts[..., :3], parts[..., 3:]
z = F.normalize(z, dim=-1, p=2)
return vel.detach(), z.detach()
```

当前 `helpers.py` 里的 HIM exporter 也手写了同样逻辑：

```python
parts = self.estimator(obs_history)[:, 0:19]
vel, z = parts[..., :3], parts[..., 3:]
z = F.normalize(z, dim=-1, p=2.0)
current_obs = obs_history[:, :self.num_one_step_obs]
return self.actor(torch.cat((current_obs, vel, z), dim=1))
```

HIM 文件里虽然定义了：

```python
class RunningMeanStd:
    ...

class Normalization:
    ...
```

但当前检查到的 `HIMActorCritic` / `HIMPPO` / `HIMOnPolicyRunner` 没有实例化或调用这个 `Normalization`。也就是说，它现在是未接入训练链路的工具类/遗留代码。

结论：

- HIM 当前不需要额外导出 empirical normalizer。
- HIM exporter 必须保留 `F.normalize(z)`，因为这是 estimator latent 逻辑的一部分。
- 如果未来真的接入 `Normalization(shape=...)` 并在训练中更新 running mean/std，则必须保存、加载并导出这个 normalizer。

#### 6.1.4 当前 exporter 的 normalizer 参数是什么

当前 `legged_gym/utils/exporter.py` 的 JIT exporter 支持外部 normalizer：

```python
if normalizer:
    self.normalizer = copy.deepcopy(normalizer)
else:
    self.normalizer = torch.nn.Identity()
```

然后在 forward 里先执行：

```python
x = self.normalizer(x)
```

这个 `normalizer` 指的是类似 Isaac Lab / RSL-RL 里的 empirical observation normalizer，不是 CTS 的 `L2Norm`，也不是 env 的 `obs_scales`。

当前代码中没有看到 runner 创建、更新、保存、加载这种 empirical normalizer，也没有任何导出调用传入非空 normalizer：

```python
export_policy_as_jit(model, path)
export_policy_as_onnx(model, path)
```

因此当前项目的正确策略是：

- 默认 `normalizer=None`，导出时使用 `Identity()`。
- 不要因为 CTS 有 `norm_type` 就传 external normalizer。
- 不要因为 env 有 `obs_scales` 就传 external normalizer。

#### 6.1.5 需要注意：当前 ONNX exporter 忽略了 normalizer 参数

当前 `_OnnxPolicyExporter.__init__()` 虽然接收 `normalizer=None`，但实际写的是：

```python
self.normalizer = torch.nn.Identity()
```

也就是说，ONNX 分支目前不会 deepcopy 传入的 normalizer。当前项目没有使用 empirical normalizer，所以这个问题暂时不影响现有导出；但如果未来启用 observation running mean/std，这就是一个必须修复的问题。

建议把 `_OnnxPolicyExporter.__init__()` 改成与 JIT 一致：

```python
if normalizer:
    self.normalizer = copy.deepcopy(normalizer).cpu().eval()
else:
    self.normalizer = torch.nn.Identity()
```

完整上下文示例：

```python
class _OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_dim = None
        self.num_actions = 12

        if normalizer:
            self.normalizer = copy.deepcopy(normalizer).cpu().eval()
        else:
            self.normalizer = torch.nn.Identity()

        # 后续继续保留原来的 policy 分支逻辑
        ...
```

如果未来添加 HIM empirical normalizer，则 HIM ONNX wrapper 也要同样支持：

```python
class _OnnxHIMPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None, input_dim=None, verbose=False):
        ...
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer).cpu().eval()
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, obs_history):
        obs_history = self.normalizer(obs_history)
        ...
```

但注意：只有训练时真的对 `obs_history` 使用同一个 normalizer，导出时才应该这样做。当前 HIM 没有接入，所以不要提前加到 forward 路径里改变行为。

#### 6.1.6 本项目当前导出是否要因为 CTS normalization 修改？

短结论：

```text
当前不需要因为 CTS 的 norm_type=L2Norm/SimNorm 额外修改导出 normalizer 传参。
```

原因：

1. CTS 的 normalization 是 encoder 内部 latent 层，已经在 `student_encoder` / `student_moe_encoder` 里。
2. exporter 当前 deepcopy encoder，因此会保留 L2Norm/SimNorm。
3. 当前训练代码没有 empirical observation normalizer。
4. 当前导出调用没有传 normalizer，也没有地方可取可保存的 normalizer。

真正建议修改的是：

1. 文档和代码里明确区分 `latent norm` 与 `empirical obs normalizer`。
2. `_OnnxPolicyExporter` 应补齐 normalizer 支持，防止未来启用 empirical normalizer 后 JIT/ONNX 行为不一致。
3. 高层 `export_policy()` 接口保留 `normalizer=None` 参数，但当前 PPO/HIM/CTS/MoE 默认都传 `None`。
4. 如果未来某个 runner 真的引入 empirical normalizer，则必须同时修改 checkpoint 保存/加载和导出调用。

## 7. 具体修改方案与参考代码

以下代码是建议落地方式。实施时应按小步提交，先迁移 HIM，再改入口，再改 registry。

### 7.1 修改 exporter.py 的 import

在 `legged_gym/utils/exporter.py` 顶部加入：

```python
import copy
import os
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn
```

如果原文件已经有 `copy/os/torch/nn/Optional`，只需要补：

```python
from typing import Sequence
import torch.nn.functional as F
```

`F` 是 HIM latent normalize 需要的。

### 7.2 增加导出类型检测

建议添加：

```python
def detect_export_type(policy: object, export_type: str = "auto") -> str:
    """Resolve policy export type.

    Explicit config always wins. Auto detection is only a compatibility fallback.
    Detection order must go from specific structures to generic PPO actor.
    """
    if export_type and export_type != "auto":
        return export_type

    if hasattr(policy, "estimator"):
        return "him"

    if hasattr(policy, "actor_mcp"):
        return "mcp_cts"

    if hasattr(policy, "student_moe_encoder") and hasattr(policy, "actor_moe"):
        return "dual_moe_cts"

    if hasattr(policy, "actor_moe"):
        return "ac_moe_cts"

    if hasattr(policy, "student_moe_encoder"):
        return "moe_cts"

    if hasattr(policy, "student_encoder"):
        return "cts"

    if getattr(policy, "is_recurrent", False) or hasattr(policy, "memory_a"):
        return "recurrent"

    if hasattr(policy, "actor"):
        return "ppo"

    raise ValueError(
        f"Unsupported policy structure for export: {type(policy).__name__}. "
        "Pass export_type explicitly or add a new exporter branch."
    )
```

为什么这个顺序重要：

- HIM 也有 `actor`，必须先于 PPO 判断。
- MoE/CTS 也有 `actor`，必须先于 PPO 判断。
- Dual-MoE 比 MoE 更特殊，必须先判断。

### 7.3 增加 runner policy 解析

建议添加：

```python
def resolve_policy_from_runner(runner: object) -> nn.Module:
    """Return the model object that should be exported from a runner.

    PPO/HIM algorithms store the policy in `alg.actor_critic`.
    CTS/MoE algorithms store the policy in `alg.model`.
    """
    if not hasattr(runner, "alg"):
        raise ValueError(f"Runner does not have alg field: {type(runner).__name__}")

    alg = runner.alg

    if hasattr(alg, "actor_critic"):
        return alg.actor_critic

    if hasattr(alg, "model"):
        return alg.model

    raise ValueError(
        f"Unsupported runner algorithm type: {type(alg).__name__}. "
        "Expected alg.actor_critic or alg.model."
    )
```

这样 `play.py` 不再需要知道 `actor_critic` / `model` 差异。

### 7.4 迁移 HIM 维度推断 helper

从 `helpers.py` 移到 `exporter.py`：

```python
def _first_linear_in_features(module: nn.Module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            return layer.in_features
    return None


def _last_linear_out_features(module: nn.Module):
    out_features = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            out_features = layer.out_features
    return out_features


def _resolve_him_num_one_step_obs(actor_critic: object) -> int:
    if hasattr(actor_critic, "num_one_step_obs"):
        return int(actor_critic.num_one_step_obs)

    actor_input_dim = _first_linear_in_features(actor_critic.actor)
    estimator_output_dim = _last_linear_out_features(actor_critic.estimator.encoder)

    if actor_input_dim is not None:
        if estimator_output_dim is None:
            # HIM default: 3 velocity dims + 16 latent dims.
            estimator_output_dim = 19

        num_one_step_obs = actor_input_dim - estimator_output_dim
        if num_one_step_obs > 0:
            return int(num_one_step_obs)

    if hasattr(actor_critic, "num_obs"):
        return int(actor_critic.num_obs)

    raise ValueError(
        "Unable to infer HIM one-step observation dimension: "
        f"actor_input_dim={actor_input_dim}, estimator_output_dim={estimator_output_dim}"
    )
```

建议再加 HIM ONNX input dim 推断：

```python
def _resolve_him_input_dim(actor_critic: object, input_dim: Optional[int] = None) -> int:
    if input_dim is not None:
        return int(input_dim)

    if hasattr(actor_critic, "num_actor_obs"):
        return int(actor_critic.num_actor_obs)

    if hasattr(actor_critic, "history_size") and hasattr(actor_critic, "num_one_step_obs"):
        return int(actor_critic.history_size * actor_critic.num_one_step_obs)

    if hasattr(actor_critic, "estimator"):
        estimator = actor_critic.estimator
        if hasattr(estimator, "temporal_steps") and hasattr(estimator, "num_one_step_obs"):
            return int(estimator.temporal_steps * estimator.num_one_step_obs)

    raise ValueError(
        "Unable to infer HIM ONNX input dimension. "
        "Pass input_dim explicitly to export_policy_as_onnx(...)."
    )
```

### 7.5 新增 HIM JIT exporter

建议放在 `_TorchPolicyExporter` 附近：

```python
class _TorchHIMPolicyExporter(torch.nn.Module):
    """Exporter for HIM/himloco actor_critic.

    The exported graph contains:
    obs_history -> estimator.encoder -> velocity + normalized latent -> actor -> actions
    """

    def __init__(self, actor_critic: object):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor).cpu().eval()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).cpu().eval()
        self.num_one_step_obs = _resolve_him_num_one_step_obs(actor_critic)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        current_obs = obs_history[:, :self.num_one_step_obs]
        actor_input = torch.cat((current_obs, vel, z), dim=1)
        return self.actor(actor_input)

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(os.path.join(path, filename))
```

注意：

- `actor` 和 `estimator.encoder` 都要 `deepcopy()`，避免导出过程改变训练模型。
- `eval()` 用于避免 Dropout/BatchNorm 影响推理。
- JIT 和 ONNX 的 forward 必须完全一致。

### 7.6 新增 HIM ONNX exporter

```python
class _OnnxHIMPolicyExporter(torch.nn.Module):
    """ONNX exporter for HIM/himloco actor_critic."""

    def __init__(self, actor_critic: object, input_dim: Optional[int] = None, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor).cpu().eval()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).cpu().eval()
        self.num_one_step_obs = _resolve_him_num_one_step_obs(actor_critic)
        self.input_dim = _resolve_him_input_dim(actor_critic, input_dim)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        current_obs = obs_history[:, :self.num_one_step_obs]
        actor_input = torch.cat((current_obs, vel, z), dim=1)
        return self.actor(actor_input)

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        self.to("cpu")

        dummy_input = torch.zeros(1, self.input_dim, device="cpu")

        torch.onnx.export(
            self,
            dummy_input,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            verbose=self.verbose,
            input_names=["obs_history"],
            output_names=["actions"],
            dynamic_axes={
                "obs_history": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
        )
```

这里保留 `obs_history` 作为输入名，因为 HIM 的输入确实是完整历史观测，不是单步 obs。

### 7.7 改造 export_policy_as_jit

原函数：

```python
def export_policy_as_jit(policy, path, normalizer=None, filename="policy.pt"):
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)
```

建议改成：

```python
def export_policy_as_jit(
    policy: object,
    path: str,
    normalizer: Optional[object] = None,
    filename: str = "policy.pt",
    export_type: str = "auto",
):
    """Export policy into a TorchScript/JIT file."""
    resolved_type = detect_export_type(policy, export_type)

    if resolved_type == "him":
        policy_exporter = _TorchHIMPolicyExporter(policy)
    else:
        policy_exporter = _TorchPolicyExporter(policy, normalizer)

    policy_exporter.export(path, filename)
```

为什么只对 HIM 单独分支：

- 现有 `_TorchPolicyExporter` 已经支持 PPO/Recurrent/CTS/MoE/MCP。
- HIM 当前没有被 `_TorchPolicyExporter` 正确支持，因为它需要 `estimator.encoder + actor` 的组合图。

### 7.8 改造 export_policy_as_onnx

原函数：

```python
def export_policy_as_onnx(policy, path, normalizer=None, filename="policy.onnx", verbose=False):
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)
```

建议改成：

```python
def export_policy_as_onnx(
    policy: object,
    path: str,
    normalizer: Optional[object] = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
    export_type: str = "auto",
    input_dim: Optional[int] = None,
):
    """Export policy into an ONNX file."""
    os.makedirs(path, exist_ok=True)
    resolved_type = detect_export_type(policy, export_type)

    if resolved_type == "him":
        policy_exporter = _OnnxHIMPolicyExporter(
            policy,
            input_dim=input_dim,
            verbose=verbose,
        )
    else:
        policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)

    policy_exporter.export(path, filename)
```

### 7.9 增加统一 export_policy

建议新增高层入口：

```python
def export_policy(
    policy: object,
    path: str,
    formats: Sequence[str] = ("jit",),
    export_type: str = "auto",
    normalizer: Optional[object] = None,
    onnx_input_dim: Optional[int] = None,
    verbose: bool = False,
):
    """Export a policy in one or more formats.

    Args:
        policy: Policy module resolved from runner.
        path: Export directory.
        formats: Any combination of "jit", "onnx", "pkl".
        export_type: Explicit export type or "auto".
        normalizer: Optional empirical normalizer.
        onnx_input_dim: Optional ONNX dummy input dim, mainly for HIM fallback.
        verbose: ONNX export verbosity.
    """
    os.makedirs(path, exist_ok=True)

    normalized_formats = {fmt.lower() for fmt in formats}

    unknown = normalized_formats - {"jit", "onnx", "pkl"}
    if unknown:
        raise ValueError(f"Unsupported export formats: {sorted(unknown)}")

    if "jit" in normalized_formats:
        export_policy_as_jit(
            policy,
            path,
            normalizer=normalizer,
            filename="policy.pt",
            export_type=export_type,
        )

    if "onnx" in normalized_formats:
        export_policy_as_onnx(
            policy,
            path,
            normalizer=normalizer,
            filename="policy.onnx",
            verbose=verbose,
            export_type=export_type,
            input_dim=onnx_input_dim,
        )

    if "pkl" in normalized_formats:
        export_policy_as_pkl(
            policy,
            path,
            filename="policy.pkl",
        )
```

如果需要兼容旧文件名，可以在 config 中额外加 `export_jit_filename`、`export_onnx_filename`，但第一阶段不建议扩展太多。

## 8. play.py 建议改法

当前：

```python
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import export_policy_as_onnx
```

建议改成：

```python
from legged_gym.utils import get_args, task_registry, Logger
from legged_gym.utils.exporter import resolve_policy_from_runner, export_policy
```

当前导出代码：

```python
if EXPORT_POLICY:
    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit(ppo_runner.alg.actor_critic, path)
    print('Exported policy as jit script to: ', path)
```

建议改成：

```python
if EXPORT_POLICY:
    path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "policies",
    )

    export_policy(runner, path)
    print("Exported policy to: ", path)
```

同时建议把变量名从 `ppo_runner` 改成更中性的 `runner`：

```python
runner, train_cfg = task_registry.make_alg_runner(
    env=env,
    name=args.task,
    args=args,
    train_cfg=train_cfg,
)
policy = runner.get_inference_policy(device=env.device)
```

原因：当前项目不一定都是 PPO runner。

## 9. utils/__init__.py 建议改法

当前：

```python
from .helpers import class_to_dict, get_load_path, get_args, export_policy_as_jit, set_seed, update_class_from_dict
```

建议改成：

```python
from .helpers import class_to_dict, get_load_path, get_args, set_seed, update_class_from_dict
from .exporter import (
    export_policy,
    export_policy_as_jit,
    export_policy_as_onnx,
    export_policy_as_pkl,
    resolve_policy_from_runner,
)
from .task_registry import task_registry
from .logger import Logger
from .math import *
from .terrain import Terrain
```

这样旧代码如果写：

```python
from legged_gym.utils import export_policy_as_jit
```

仍然能工作，但真实实现来自 `exporter.py`，不是 `helpers.py`。

## 10. helpers.py 清理方案

迁移完成后，从 `helpers.py` 删除以下内容：

```text
export_policy_as_jit
export_policy_as_onnx
PolicyExporterHIM
ONNXPolicyExporterHIM
_first_linear_in_features
_last_linear_out_features
_resolve_him_num_one_step_obs
```

再检查并删除不再使用的 import：

```python
import copy
import torch.nn.functional as F
```

如果 `torch` 只被导出逻辑使用，也可以删除。但要先运行：

```bash
rg "torch" legged_gym/utils/helpers.py
```

确认其他 helper 没有依赖。

## 11. task_registry.py 建议改法

当前硬编码：

```python
runner = HIMOnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
```

建议改为显式 registry：

```python
from rsl_rl.runners import OnPolicyRunner, HIMOnPolicyRunner, OnPolicyRunnerCTS


RUNNER_REGISTRY = {
    "OnPolicyRunner": OnPolicyRunner,
    "HIMOnPolicyRunner": HIMOnPolicyRunner,
    "OnPolicyRunnerCTS": OnPolicyRunnerCTS,
}
```

在 `make_alg_runner()` 中：

```python
train_cfg_dict = class_to_dict(train_cfg)

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

if runner_class_name not in RUNNER_REGISTRY:
    raise ValueError(
        f"Runner class '{runner_class_name}' was not registered"
    )

runner_cls = RUNNER_REGISTRY[runner_class_name]
runner = runner_cls(env, train_cfg_dict, log_dir, device=args.rl_device)
```

说明：

- 不建议用 `eval()`，避免字符串执行风险。
- runner 类型优先使用已有 `runner_class_name`，缺失或保持默认时可根据 `policy_class_name` / `algorithm_class_name` 推断 HIM/CTS。
- 这样 PPO/HIM/CTS/MoE 任务可以共用同一个 `task_registry.py`。

## 12. 任务配置与导出关系

这次修改不要求在 config 中新增 `export_policy_type`、`export_formats` 或类似字段。config 仍然只负责已有训练结构选择：

```python
runner_class_name = "OnPolicyRunner"

class runner:
    policy_class_name = "ActorCritic"
    algorithm_class_name = "PPO"
```

HIM/CTS/MoE 任务也只需要使用已有字段表达训练结构，例如：

```python
runner_class_name = "HIMOnPolicyRunner"

class runner:
    policy_class_name = "HIMActorCritic"
    algorithm_class_name = "HIMPPO"
```

```python
runner_class_name = "OnPolicyRunnerCTS"

class runner:
    policy_class_name = "ActorCriticMoECTS"
    algorithm_class_name = "MoECTS"
```

导出类型由 `exporter.py` 根据实例化后的 policy 自动判断：

```text
HIMActorCritic       -> hasattr(policy, "estimator")
ActorCriticCTS      -> hasattr(policy, "student_encoder")
ActorCriticMoECTS   -> hasattr(policy, "student_moe_encoder")
ActorCritic         -> hasattr(policy, "actor")
ActorCriticRecurrent -> is_recurrent / memory_a
```

后续新增算法时，应优先在 `exporter.py` 中补检测分支和 wrapper，而不是在每个 config 文件中补导出字段。

## 13. 迁移顺序

### Step 1：只迁移 HIM 导出到 exporter.py

动作：

1. 在 `exporter.py` 中增加 HIM 维度推断 helper。
2. 增加 `_TorchHIMPolicyExporter`。
3. 增加 `_OnnxHIMPolicyExporter`。
4. 给 `export_policy_as_jit()` 和 `export_policy_as_onnx()` 增加 `export_type` 参数。
5. 顺手修正 `_OnnxPolicyExporter` 对 `normalizer` 参数的处理，使其和 JIT 分支一致；当前项目默认仍传 `None`。

此时先不要动 `task_registry.py`。

验收：

```bash
rg "class _TorchHIMPolicyExporter|class _OnnxHIMPolicyExporter" legged_gym/utils/exporter.py
rg "copy.deepcopy\\(normalizer\\)" legged_gym/utils/exporter.py
python -m py_compile legged_gym/utils/exporter.py
```

### Step 2：更新导入入口

动作：

1. `utils/__init__.py` 从 `exporter.py` re-export 导出函数。
2. `play.py` 不再从 `helpers.py` 导入 ONNX 导出函数。

验收：

```bash
rg "helpers import export_policy" legged_gym rsl_rl
rg "export_policy_as_jit|export_policy_as_onnx|export_policy_as_pkl" legged_gym rsl_rl
```

预期：

- 不再出现 `from legged_gym.utils.helpers import export_policy_as_onnx`。
- 导出函数集中来自 `legged_gym.utils.exporter` 或 `legged_gym.utils` re-export。

### Step 3：封装 runner policy 解析

动作：

1. 在 `exporter.py` 增加 `resolve_policy_from_runner()`。
2. `play.py` 用 `resolve_policy_from_runner(runner)`。

验收：

- HIM/PPO runner 返回 `alg.actor_critic`。
- CTS/MoE runner 返回 `alg.model`。

### Step 4：增加 export_policy 高层入口

动作：

1. 在 `exporter.py` 增加 `export_policy()`。
2. `play.py` 只调用 `export_policy()`。
3. 默认导出 JIT；如需 ONNX/PKL，由调用代码显式传 `formats`，不要求 config 新增字段。

验收：

- `export_policy(runner, path)` 只导出 JIT。
- `export_policy(runner, path, formats=("jit", "onnx", "pkl"))` 同时导出三个格式。

### Step 5：清理 helpers.py

动作：

1. 删除 HIM 导出类和函数。
2. 删除无用 import。

验收：

```bash
rg "PolicyExporterHIM|ONNXPolicyExporterHIM|export_policy_as_onnx|export_policy_as_jit" legged_gym/utils/helpers.py
python -m py_compile legged_gym/utils/helpers.py
```

预期：

- `helpers.py` 内无任何导出逻辑。

### Step 6：改造 task_registry.py

动作：

1. 增加 `RUNNER_REGISTRY`。
2. 优先用已有 `train_cfg.runner_class_name` 创建 runner。
3. 新 runner 可通过 `task_registry.register_runner()` 在代码中注册。

验收：

- PPO 任务能创建 `OnPolicyRunner`。
- HIM 任务能创建 `HIMOnPolicyRunner`。
- CTS/MoE 任务能创建 `OnPolicyRunnerCTS`。

## 14. 验证代码建议

### 14.1 HIM JIT 输出对齐

可以写一个临时验证脚本，不提交到仓库也可以：

```python
import os
import torch

from legged_gym.utils.exporter import export_policy_as_jit


def verify_him_jit(actor_critic, export_dir):
    actor_critic.eval()
    obs_dim = actor_critic.num_actor_obs
    obs_history = torch.randn(1, obs_dim, device="cpu")

    with torch.no_grad():
        ref = actor_critic.cpu().act_inference(obs_history)

    export_policy_as_jit(
        actor_critic,
        export_dir,
        filename="policy.pt",
        export_type="him",
    )

    jit_model = torch.jit.load(os.path.join(export_dir, "policy.pt")).cpu()
    with torch.no_grad():
        out = jit_model(obs_history)

    max_err = (ref - out).abs().max().item()
    print("HIM JIT max error:", max_err)
    assert max_err < 1e-4
```

### 14.2 HIM ONNX 输出对齐

需要安装 `onnxruntime`：

```python
import os
import numpy as np
import torch
import onnxruntime as ort

from legged_gym.utils.exporter import export_policy_as_onnx


def verify_him_onnx(actor_critic, export_dir):
    actor_critic.eval().cpu()
    obs_dim = actor_critic.num_actor_obs
    obs_history = torch.randn(1, obs_dim, device="cpu")

    with torch.no_grad():
        ref = actor_critic.act_inference(obs_history).numpy()

    export_policy_as_onnx(
        actor_critic,
        export_dir,
        filename="policy.onnx",
        export_type="him",
        input_dim=obs_dim,
    )

    session = ort.InferenceSession(os.path.join(export_dir, "policy.onnx"))
    out = session.run(
        ["actions"],
        {"obs_history": obs_history.numpy()},
    )[0]

    max_err = np.max(np.abs(ref - out))
    print("HIM ONNX max error:", max_err)
    assert max_err < 1e-4
```

### 14.3 CTS/MoE JIT reset 验证

CTS/MoE JIT exporter 内部维护 `history`，部署时必须支持 reset：

```python
def verify_cts_jit_reset(jit_model):
    if hasattr(jit_model, "reset"):
        jit_model.reset()

    obs = torch.zeros(1, obs_dim)
    actions_1 = jit_model(obs)

    if hasattr(jit_model, "reset"):
        jit_model.reset()

    actions_2 = jit_model(obs)
```

如果同样输入、同样 reset 后输出不一致，需要检查 exporter 内部状态更新逻辑。

## 15. 关键风险点

### 15.1 ONNX 输入布局风险

当前 `_OnnxPolicyExporter.flatten_obs()` 写死了：

```python
term_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
```

这明显绑定某类 Go2 观测结构。它不应被 HIM 使用，也不一定适合所有 PPO/CTS 任务。后续如果更多任务要导 ONNX，建议把 `term_dims` 变成 config 参数或从 env cfg 推断。

### 15.2 empirical normalizer 的 JIT/ONNX 一致性风险

当前项目没有启用 empirical observation normalizer，所以默认 `normalizer=None` 是正确的。不要因为 CTS 的 `norm_type="l2norm"` 或 env 的 `obs_scales` 就额外传 normalizer。

但 exporter API 已经暴露了 `normalizer` 参数，后续一旦某个 runner 真正引入 running mean/std normalizer，就必须保证：

- checkpoint 保存 normalizer 状态。
- runner load 后能恢复 normalizer。
- JIT 和 ONNX 导出都 deepcopy 同一个 normalizer。
- 部署端不要再重复做同一层 running mean/std normalization。

当前 JIT 分支会使用传入的 `normalizer`，ONNX 分支应补齐相同行为。否则同一个 policy 导出的 JIT 和 ONNX 会出现系统性输出偏差。

### 15.3 HIM JIT/ONNX forward 必须一致

HIM JIT 和 ONNX 都要执行同样逻辑：

```text
obs_history
  -> estimator.encoder
  -> vel + normalized latent
  -> current_obs + vel + latent
  -> actor
  -> actions
```

不要让 JIT 调 `estimator.encoder`，ONNX 调 `estimator.forward()`，否则 detach、normalize 或输出切片行为可能不同。

### 15.4 文件名兼容风险

历史实现中：

- 原生普通 actor JIT：`policy_1.pt`
- HIM JIT：`policy.pt`
- go2 exporter：`policy.pt`

建议统一为：

```text
policy.pt
policy.onnx
policy.pkl
```

如果部署脚本已经依赖 `policy_1.pt`，应通过 `filename` 参数兼容，而不是在 exporter 内硬编码旧名字。

### 15.5 自动检测误判风险

`hasattr()` 适合兜底，不适合长期作为主要分发机制。原因：

- 很多模型都有 `actor`。
- 新算法可能同时有 `student_encoder` 和 `actor_moe`。
- wrapper 字段名变化会导致静默走错分支。

当前实现默认自动检测，不要求 config 新增导出字段。如果某个新增算法字段重叠导致误判，应在 `detect_export_type()` 中把更特殊的结构放在更通用的 `actor` 分支之前；必要时由调用代码显式传 `export_type`，不要把导出策略扩散到每个任务 config。

### 15.6 runner 选择风险

如果 `task_registry.py` 继续硬编码 `HIMOnPolicyRunner`，即使 exporter 完成重构，也无法真正支持多任务。因为 CTS/MoE 需要 `OnPolicyRunnerCTS`，普通 PPO 需要 `OnPolicyRunner`。

所以 runner registry 是第二阶段必须完成的配套改造。

## 16. 最终完成标准

完成后应满足：

```bash
rg "PolicyExporterHIM|ONNXPolicyExporterHIM" legged_gym/utils/helpers.py
```

无结果。

```bash
rg "helpers import export_policy" legged_gym rsl_rl
```

无结果。

```bash
rg "def export_policy|def detect_export_type|def resolve_policy_from_runner" legged_gym/utils/exporter.py
```

能找到对应函数。

```bash
rg "self.normalizer = torch.nn.Identity\\(\\)" legged_gym/utils/exporter.py
```

需要确认 JIT 和 ONNX 分支都只在 `normalizer is None` 时使用 `Identity()`；如果传入 normalizer，两边都应 deepcopy 同一个 normalizer。

```bash
python -m py_compile \
  legged_gym/utils/exporter.py \
  legged_gym/utils/helpers.py \
  legged_gym/scripts/play.py \
  legged_gym/utils/task_registry.py
```

无语法错误。

至少验证以下任务：

- 一个 PPO 任务：能导出 JIT。
- 一个 HIM 任务：能导出 JIT 和 ONNX，并与 `act_inference()` 对齐。
- 一个 CTS/MoE 任务：能导出 JIT，部署端能 reset history。

## 17. 推荐落地原则

1. 先迁移，不重写：HIM 现有 wrapper 逻辑是可用的，先移动到 `exporter.py`，不要同时改算法逻辑。
2. 先自动检测，再代码覆盖：默认由 `detect_export_type()` 判断；特殊情况由调用代码传 `export_type`，不新增导出 config 字段。
3. 先 JIT，再 ONNX：JIT 更接近 PyTorch 行为，先验证 JIT，再验证 ONNX。
4. 先 play，再训练中导出：先把手动 play 导出打通，再考虑 RoboGauge 或训练中周期导出。
5. 不在 helpers.py 写任何网络结构判断：后续只要看到 `estimator`、`actor_moe`、`student_encoder` 这类字段判断出现在 helpers，就说明职责又跑偏了。

最终形态应该是：

```text
新增算法 -> 增加 exporter wrapper / 检测分支 -> 必要时注册 runner -> play/export_policy 调用
```

而不是：

```text
新增算法 -> 修改 helpers.py -> 修改 play.py -> 修改多个 runner -> 手动判断各种字段
```
