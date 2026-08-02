# Policy 导出链路说明

本文解释当前项目里 policy 导出的完整链路：从 `play.py` 触发导出，到 `task_registry.py` 创建 runner，再到 `exporter.py` 自动识别不同训练算法的网络结构并导出 JIT / ONNX / PKL。

重点目标不是重新设计训练流程，而是把“导出到底导出了什么、为什么不同算法走不同 wrapper、后续新增算法应该改哪里”讲清楚。

## 0. 快速结论

当前导出设计应遵守几个原则：

1. `helpers.py` 不再承担导出职责。导出逻辑集中放在 `legged_gym/utils/exporter.py`。
2. config 文件里不新增 `export_policy_type = "xxx"`、`export_formats = [...]` 这类导出专用字段。
3. `play.py` 不关心 PPO / HIM / CTS / MoE-CTS 的具体网络结构，只调用统一入口 `export_policy(runner, path)`。
4. `exporter.py` 通过 policy 对象本身的字段自动判断导出类型，例如 `estimator`、`student_encoder`、`student_moe_encoder`、`memory_a`、`actor`。
5. 原生 PPO MLP 的 JIT 导出已经有链路。
6. 原生 RNN / LSTM policy 的 JIT 导出已经有链路，但 ONNX 没有完整支持 hidden state 输入输出。
7. CTS / MoE-CTS 里的 `L2Norm` / `SimNorm` 是 encoder 内部 latent normalization，不等于 observation running mean/std normalizer；一般不需要额外修改导出 normalizer。
8. 当前 ONNX 的普通 PPO 路径仍带有 go2 stack observation 假设，如果要严格支持原生 PPO direct observation ONNX，需要单独修正。

## 1. 文件职责

### 1.1 `legged_gym/scripts/play.py`

`play.py` 是目前最直接的导出入口。它做三件事：

1. 创建环境。
2. 创建并恢复 runner。
3. 如果 `EXPORT_POLICY = True`，调用 `export_policy(runner, path)`。

关键代码形态如下：

```python
env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
obs = env.get_observations()

train_cfg.runner.resume = True
runner, train_cfg = task_registry.make_alg_runner(
    env=env,
    name=args.task,
    args=args,
    train_cfg=train_cfg,
)

policy = runner.get_inference_policy(device=env.device)

if EXPORT_POLICY:
    path = os.path.join(
        LEGGED_GYM_ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "exported",
        "policies",
    )
    export_policy(runner, path)
```

这里容易混淆的是：`policy = runner.get_inference_policy(...)` 是仿真 rollout 时实际调用的推理函数；`export_policy(runner, path)` 导出时传的是整个 runner，不是这个 `policy` 函数。

原因是导出不应该只拿到一个函数闭包。导出需要拿到完整模型对象，比如：

- 原生 PPO：`runner.alg.actor_critic`
- HIM：`runner.alg.actor_critic`
- CTS / MoE-CTS：`runner.alg.model`

这些完整对象里包含 actor、encoder、RNN、history buffer、estimator 等结构。只拿 `act_inference` 函数不方便可靠地反推出这些模块。

### 1.2 `legged_gym/utils/task_registry.py`

`task_registry.py` 负责根据训练配置创建正确的 runner。

当前 registry 里有一个 runner registry：

```python
RUNNER_REGISTRY = {
    "OnPolicyRunner": OnPolicyRunner,
    "HIMOnPolicyRunner": HIMOnPolicyRunner,
    "OnPolicyRunnerCTS": OnPolicyRunnerCTS,
}
```

创建 runner 时走：

```python
runner_class = self.get_runner_class(train_cfg)
runner = runner_class(env, train_cfg_dict, log_dir, device=args.rl_device)
```

`get_runner_class(train_cfg)` 的作用是：

1. 优先读取已有配置里的 `runner_class_name`。
2. 如果没有明确 runner，或者仍是默认 `OnPolicyRunner`，再根据 `policy_class_name` / `algorithm_class_name` 推断：
   - 名字里包含 `HIM`：使用 `HIMOnPolicyRunner`
   - 名字里包含 `CTS`：使用 `OnPolicyRunnerCTS`
   - 否则默认 `OnPolicyRunner`
3. 如果后续新增 runner，可以调用 `register_runner(name, runner_class)` 注册。

这一步只负责“训练/加载时创建什么 runner”，不负责导出格式选择。

### 1.3 `legged_gym/utils/exporter.py`

`exporter.py` 是导出中心。它承担四类职责：

1. 从 runner 中解析真正要导出的模型。
2. 自动识别 policy 类型。
3. 按类型选择 TorchScript / ONNX wrapper。
4. 写出 `.pt` / `.onnx` / `.pkl` 文件。

对外主要入口是：

```python
export_policy(policy_or_runner, path, formats=("jit",), normalizer=None)
export_policy_as_jit(policy_or_runner, path, normalizer=None, filename="policy.pt")
export_policy_as_onnx(policy_or_runner, path, normalizer=None, filename="policy.onnx")
export_policy_as_pkl(policy_or_runner, path, filename="policy.pkl")
```

默认 `export_policy(...)` 只导出 JIT：

```python
def export_policy(
    policy: object,
    path: str,
    formats: Sequence[str] = ("jit",),
    normalizer: Optional[object] = None,
    export_type="auto",
    input_dim=None,
):
    policy = resolve_policy_from_runner(policy)
    if isinstance(formats, str):
        formats = (formats,)
    for export_format in formats:
        if export_format == "jit":
            export_policy_as_jit(policy, path, normalizer, export_type=export_type)
        elif export_format == "onnx":
            export_policy_as_onnx(policy, path, normalizer, export_type=export_type, input_dim=input_dim)
        elif export_format == "pkl":
            export_policy_as_pkl(policy, path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
```

这里是有意不从 config 读取 `export_formats`。如果某个脚本需要导出 ONNX / PKL，可以在脚本里显式传：

```python
export_policy(runner, path, formats=("jit", "onnx", "pkl"))
```

不要把这个选择写回每个训练 config。训练 config 应该描述训练算法和网络结构，不应该混入部署导出格式。

### 1.4 `rsl_rl/rsl_rl/runners/on_policy_runner_cts.py`

除了 `play.py` 的手动导出，CTS runner 里还有一个训练过程中的自动导出点：

```python
def update_robogauge(self, it, last_model):
    ...
    if it % 500 == 0 or last_model:
        jit_dir = os.path.join(self.log_dir, "jit_models")
        jit_path = os.path.join(jit_dir, f"policy_jit_{it}.pt")
        export_policy_as_jit(self.alg.model, jit_dir, filename=f"policy_jit_{it}.pt")
        ...
```

这个路径只导出 JIT，主要用于 RoboGauge 提交评测。它同样调用 `exporter.py`，没有把导出逻辑写在 runner 内部。

这说明当前项目里导出入口有两类：

| 入口 | 触发时机 | 默认格式 | 作用 |
| --- | --- | --- | --- |
| `play.py` | 手动 play / 部署前导出 | JIT | 生成 `logs/<experiment>/exported/policies/policy.pt` |
| `OnPolicyRunnerCTS.update_robogauge()` | CTS 训练过程中按间隔或最后一次模型 | JIT | 生成 `jit_models/policy_jit_<it>.pt` 并提交 RoboGauge |

两者的共同点是：真正的 wrapper 和模型拆分逻辑都在 `exporter.py`。

## 2. 整体调用链

可以把当前导出理解成下面这条链：

```text
play.py
  -> task_registry.get_cfgs(task)
  -> task_registry.make_env(...)
  -> task_registry.make_alg_runner(...)
       -> get_runner_class(train_cfg)
       -> runner_class(env, train_cfg_dict, log_dir, device)
       -> runner.load(checkpoint)
  -> runner.get_inference_policy(...)
       -> 只用于 play rollout
  -> export_policy(runner, export_dir)
       -> resolve_policy_from_runner(runner)
       -> detect_export_type(policy)
       -> export_policy_as_jit / export_policy_as_onnx / export_policy_as_pkl
       -> _TorchPolicyExporter / _TorchHIMPolicyExporter
       -> torch.jit.script(...).save(...)
```

CTS 训练过程中的 RoboGauge 自动导出链路更短：

```text
OnPolicyRunnerCTS.learn(...)
  -> save(...)
  -> update_robogauge(...)
  -> export_policy_as_jit(self.alg.model, jit_dir, filename=...)
       -> detect_export_type(model)
       -> _TorchPolicyExporter
       -> torch.jit.script(...).save(...)
```

其中有两个容易混淆的点。

第一个点：`runner.get_inference_policy()` 和 `export_policy()` 不是同一个概念。

- `get_inference_policy()` 返回一个 callable，例如 `actor_critic.act_inference`。
- `export_policy()` 需要完整模型对象，然后重新封装一个部署用 wrapper。

第二个点：训练时的 actor-critic 不一定等于部署时的 forward 图。

例如：

- 原生 PPO 训练时有 actor 和 critic，部署只需要 actor。
- HIM 训练时有 estimator、actor、critic、estimator optimizer，部署只需要 estimator encoder + actor。
- CTS 训练时有 teacher encoder、student encoder、critic，部署只需要 student encoder + actor。
- RNN 训练时 actor 依赖 `memory_a` 的 hidden state，部署 wrapper 需要自己维护 hidden/cell state。

所以 exporter 不是简单 `torch.jit.script(policy)`，而是把训练模型拆成部署所需的子图。

## 3. Runner 到模型对象的解析

统一入口允许传 runner，也允许直接传 policy。解析逻辑在 `resolve_policy_from_runner()`：

```python
def resolve_policy_from_runner(runner: object):
    if not hasattr(runner, "alg"):
        return runner

    if hasattr(runner.alg, "actor_critic"):
        return runner.alg.actor_critic
    if hasattr(runner.alg, "model"):
        return runner.alg.model

    raise ValueError(f"Unsupported runner algorithm type: {type(runner.alg).__name__}")
```

也就是说：

| 训练链路 | runner 类型 | 算法对象字段 | 实际导出对象 |
| --- | --- | --- | --- |
| 原生 PPO MLP | `OnPolicyRunner` | `runner.alg.actor_critic` | `ActorCritic` |
| 原生 PPO RNN/LSTM | `OnPolicyRunner` | `runner.alg.actor_critic` | `ActorCriticRecurrent` |
| HIM | `HIMOnPolicyRunner` | `runner.alg.actor_critic` | `HIMActorCritic` |
| CTS | `OnPolicyRunnerCTS` | `runner.alg.model` | `ActorCriticCTS` |
| MoE-CTS | `OnPolicyRunnerCTS` | `runner.alg.model` | `ActorCriticMoECTS` |

这个函数是后续扩展 runner 时必须考虑的接口。如果新算法的模型不是放在 `alg.actor_critic` 或 `alg.model`，有两种做法：

1. 最好让新算法复用这两个字段之一。
2. 如果必须使用新字段，就在 `resolve_policy_from_runner()` 里增加明确分支。

## 4. Policy 类型自动识别

当前识别函数是 `detect_export_type(policy, export_type="auto")`：

```python
def detect_export_type(policy: object, export_type: str = "auto") -> str:
    if export_type and export_type != "auto":
        return export_type

    if hasattr(policy, "estimator"):
        return "him"

    if hasattr(policy, "student_moe_encoder"):
        return "moe_cts"

    if hasattr(policy, "student_encoder"):
        return "cts"

    if getattr(policy, "is_recurrent", False) or hasattr(policy, "memory_a"):
        return "recurrent"

    if hasattr(policy, "actor"):
        return "ppo"

    raise ValueError(...)
```

检测顺序很重要：必须从“更特殊的结构”到“更通用的结构”。

原因是很多模型都有 `actor`：

- 原生 PPO 有 `actor`
- RNN PPO 有 `actor`
- HIM 有 `actor`
- CTS 有 `actor`
- MoE-CTS 有 `actor`

如果先判断 `hasattr(policy, "actor")`，所有这些模型都会被误判成普通 PPO。因此当前顺序先判断 `estimator`、`student_moe_encoder`、`student_encoder`、`memory_a`，最后才判断 `actor`。

后续新增算法时，也应该按这个规则加分支：

```python
if hasattr(policy, "my_special_encoder"):
    return "my_algo"

if hasattr(policy, "actor"):
    return "ppo"
```

不要反过来。

## 5. JIT 导出链路

JIT 是当前默认导出格式。`play.py` 里 `export_policy(runner, path)` 默认等价于：

```python
export_policy(runner, path, formats=("jit",))
```

内部最终调用：

```python
export_policy_as_jit(policy, path)
```

JIT 路径会先解析 runner：

```python
policy = resolve_policy_from_runner(policy)
```

再判断是否是 HIM：

```python
if detect_export_type(policy, export_type) == "him":
    policy_exporter = _TorchHIMPolicyExporter(policy, normalizer)
else:
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
```

也就是说当前 JIT wrapper 分两大类：

1. HIM 专用 wrapper：`_TorchHIMPolicyExporter`
2. 其他通用 wrapper：`_TorchPolicyExporter`

最后保存：

```python
traced_script_module = torch.jit.script(self)
traced_script_module.save(path)
```

注意这里变量名叫 `traced_script_module`，但实际用的是 `torch.jit.script()`，不是 `torch.jit.trace()`。这对有状态 wrapper 比较重要，因为 CTS history、RNN hidden state 这种逻辑更适合 script，而不是单纯 trace 一个固定输入路径。

## 6. 原生 PPO MLP 导出

### 6.1 训练时结构

原生 PPO 的模型是 `rsl_rl/rsl_rl/modules/actor_critic.py` 里的 `ActorCritic`。

核心结构：

```python
class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_actor_obs, num_critic_obs, num_actions, ...):
        mlp_input_dim_a = num_actor_obs
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        ...
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
```

训练时还有 critic、action std、Normal distribution 等内容，但部署推理只需要 actor 的 deterministic mean action。

### 6.2 JIT 导出时结构

`_TorchPolicyExporter` 看到普通 PPO 时会复制 actor：

```python
elif hasattr(policy, "actor"):
    self.actor = copy.deepcopy(policy.actor)
```

如果不是 recurrent，它的 forward 是：

```python
def forward(self, x):
    return self.actor(self.normalizer(x))
```

因此原生 PPO MLP 的 JIT 导出等价于：

```text
obs -> optional normalizer -> actor MLP -> actions
```

这一点是已经覆盖的。

### 6.3 ONNX 导出当前状态

当前 `_OnnxPolicyExporter.forward_ppo()` 是：

```python
def forward_ppo(self, x):
    x = self.normalizer(x)
    history, obs_dim = self.flatten_obs(x)
    last_obs = history[:, -obs_dim:]
    return self.actor(last_obs)
```

这不是严格的原生 PPO direct observation 路径。它假设输入是按 term 堆叠的 stack observation，然后通过 `flatten_obs()` 取最后一帧 observation。

如果目标是导出原生 `ActorCritic` 的 ONNX，并且部署端输入就是 `[batch, num_obs]` 的直接观测，那么更合理的 forward 应该是：

```python
def forward_ppo(self, x):
    x = self.normalizer(x)
    return self.actor(x)
```

所以当前结论是：

- 原生 PPO MLP JIT：已支持。
- 原生 PPO MLP ONNX：有入口，但目前带 stack observation 假设；严格 direct obs ONNX 需要修正。

## 7. 原生 RNN / LSTM 导出

### 7.1 训练时结构

原生 recurrent policy 是 `rsl_rl/rsl_rl/modules/actor_critic_recurrent.py` 里的 `ActorCriticRecurrent`。

它继承 `ActorCritic`，但把 actor 和 critic 的输入维度改成 RNN hidden size：

```python
class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(..., rnn_type="lstm", rnn_hidden_size=256, rnn_num_layers=1, ...):
        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            ...
        )
        self.memory_a = Memory(num_actor_obs, type=rnn_type, ...)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, ...)
```

actor 推理路径是：

```python
def act_inference(self, observations):
    input_a = self.memory_a(observations)
    return super().act_inference(input_a.squeeze(0))
```

也就是：

```text
obs -> memory_a RNN/LSTM -> actor MLP -> actions
```

### 7.2 JIT 导出时结构

`_TorchPolicyExporter` 通过这两个条件识别 recurrent：

```python
self.is_recurrent = getattr(policy, "is_recurrent", False)
...
if self.is_recurrent:
    self.rnn = copy.deepcopy(policy.memory_a.rnn)
```

然后注册 hidden state 和 cell state：

```python
self.register_buffer(
    "hidden_state",
    torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
)
self.register_buffer(
    "cell_state",
    torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size),
)
self.forward = self.forward_lstm
self.reset = self.reset_memory
```

forward 逻辑是：

```python
def forward_lstm(self, x):
    x = self.normalizer(x)
    x, (h, c) = self.rnn(
        x.unsqueeze(0),
        (self.hidden_state, self.cell_state),
    )
    self.hidden_state[:] = h
    self.cell_state[:] = c
    x = x.squeeze(0)
    return self.actor(x)
```

因此 RNN / LSTM 的 JIT 导出链路是：

```text
obs
  -> optional normalizer
  -> exported internal LSTM state
  -> actor MLP
  -> actions
```

部署端每一步只传当前 obs，hidden/cell state 保存在导出的 TorchScript module 内部。

reset 时应该调用导出 module 的 reset 接口，把 hidden/cell 清零。当前 wrapper 有：

```python
def reset_memory(self):
    self.hidden_state[:] = 0.0
    self.cell_state[:] = 0.0
```

建议后续把它显式标注为 TorchScript export：

```python
@torch.jit.export
def reset_memory(self):
    self.hidden_state[:] = 0.0
    self.cell_state[:] = 0.0
```

这样部署侧可以更明确地调用 reset 方法。

### 7.3 ONNX 导出当前状态

RNN / LSTM ONNX 不能简单复用普通 MLP ONNX。

原因是 ONNX 本身没有 Python module 内部可变状态这个概念。一个严谨的 recurrent ONNX 应该把 hidden state / cell state 作为输入和输出：

```text
inputs:
  obs
  hidden_state_in
  cell_state_in

outputs:
  actions
  hidden_state_out
  cell_state_out
```

而当前 `_OnnxPolicyExporter` 虽然在 recurrent 情况下复制了：

```python
if self.is_recurrent:
    self.rnn = copy.deepcopy(policy.memory_a.rnn)
```

但 `forward_ppo()` 没有调用这个 RNN，也没有导出 hidden/cell 输入输出。

所以当前结论是：

- 原生 RNN / LSTM JIT：已有基本链路。
- 原生 RNN / LSTM ONNX：没有完整支持，继续使用会有静默错误风险。

建议短期做法是：如果检测到 `recurrent` 并请求 ONNX，直接报错，让用户使用 JIT。

```python
resolved_type = detect_export_type(policy, export_type)
if resolved_type == "recurrent":
    raise NotImplementedError(
        "ONNX export for recurrent policies is not supported; use JIT export."
    )
```

长期如果确实需要 RNN ONNX，就单独写 `_OnnxRecurrentPolicyExporter`，明确 hidden/cell 的输入输出协议。

## 8. HIM 导出

### 8.1 训练时结构

HIM policy 是 `HIMActorCritic`。它包含：

- `estimator`
- `actor`
- `critic`
- `std`

构造时 actor 输入维度是：

```python
mlp_input_dim_a = num_one_step_obs + 3 + 16
```

其中：

- `num_one_step_obs`：当前时刻 observation。
- `3`：估计出来的 base velocity。
- `16`：HIM latent。

训练/推理时的 inference 路径是：

```python
def act_inference(self, obs_history, observations=None):
    vel, latent = self.estimator(obs_history)
    actions_mean = self.actor(
        torch.cat((obs_history[:, :self.num_one_step_obs], vel, latent), dim=-1)
    )
    return actions_mean
```

也就是：

```text
obs_history
  -> estimator
       -> vel
       -> latent
  -> concat(current_obs, vel, latent)
  -> actor
  -> actions
```

### 8.2 为什么 HIM 需要专用 exporter

普通 `_TorchPolicyExporter` 主要处理 `actor`、`student_encoder`、`student_moe_encoder`、`memory_a`。

HIM 的关键不是直接 `actor(obs)`，而是：

1. 从历史观测 `obs_history` 中取当前帧。
2. 用 estimator 从历史观测估计 `vel` 和 `latent`。
3. 拼接 `current_obs + vel + latent`。
4. 再送入 actor。

所以 HIM 使用 `_TorchHIMPolicyExporter` 和 `_OnnxHIMPolicyExporter`。

当前 JIT HIM wrapper 结构：

```python
class _TorchHIMPolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic, normalizer=None):
        self.actor = copy.deepcopy(actor_critic.actor).cpu().eval()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).cpu().eval()
        self.num_one_step_obs = _resolve_him_num_one_step_obs(actor_critic)
        self.normalizer = _copy_normalizer(normalizer)

    def forward(self, obs_history):
        obs_history = self.normalizer(obs_history)
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        current_obs = obs_history[:, :self.num_one_step_obs]
        return self.actor(torch.cat((current_obs, vel, z), dim=1))
```

这里 `parts[:, 0:19]` 对应 `3 + 16`。如果以后 HIM latent 维度不是 16，这一段应该改成从 estimator / actor 结构动态推断，而不是硬编码 19。

### 8.3 HIM ONNX

HIM ONNX 也走同样的逻辑，只是需要知道输入维度：

```python
self.input_dim = _resolve_him_input_dim(actor_critic, input_dim)
obs = torch.zeros(1, self.input_dim)
```

输入名字是：

```python
input_names=["obs_history"]
output_names=["actions"]
```

HIM ONNX 当前比 RNN ONNX 简单，因为它没有可变 hidden state；只要输入 `obs_history` layout 正确，就可以形成静态计算图。

## 9. CTS / MoE-CTS 导出

### 9.1 CTS 训练时结构

CTS 模型是 `ActorCriticCTS`。

它包含：

- `teacher_encoder`
- `student_encoder`
- `actor`
- `critic`
- `history`

训练时 teacher 可以用 privileged observation；部署时没有 privileged observation，所以导出必须走 student 分支。

部署推理路径是：

```python
def act_inference(self, obs):
    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
    latent = self.student_encoder(self.history.flatten(1))
    x = torch.cat([latent, obs], dim=1)
    actions_mean = self.actor(x)
    return actions_mean
```

所以 CTS 部署图是：

```text
current obs
  -> update internal history
  -> student_encoder(history)
  -> concat(latent, current obs)
  -> actor
  -> actions
```

### 9.2 CTS JIT

`_TorchPolicyExporter` 检测到 `student_encoder`：

```python
if hasattr(policy, "student_encoder"):
    self.student_encoder = copy.deepcopy(policy.student_encoder).cpu()
    self.history = torch.zeros(
        [1, policy.history.shape[1], policy.history.shape[2]],
        device="cpu",
    )
    self.forward = self.forward_cts
```

forward：

```python
def forward_cts(self, x):
    x = self.normalizer(x)
    self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
    latent = self.student_encoder(self.history.flatten(1))
    x = torch.cat([latent, x], dim=1)
    return self.actor(x), (None, latent)
```

注意这里返回的是：

```text
actions, (None, latent)
```

这和普通 PPO 只返回 `actions` 不一样。部署端如果只需要 action，需要取第一个返回值。

### 9.3 MoE-CTS 训练时结构

MoE-CTS 是 `ActorCriticMoECTS`。

它包含：

- `teacher_encoder`
- `student_moe_encoder`
- `actor`
- `critic`
- `history`

部署推理路径是：

```python
def act_inference(self, obs):
    self.history = torch.cat([self.history[:, 1:], obs.unsqueeze(1)], dim=1)
    latent, _ = self.student_moe_encoder(self.history.flatten(1))
    x = torch.cat([latent, obs], dim=1)
    actions_mean = self.actor(x)
    return actions_mean
```

JIT wrapper 检测到 `student_moe_encoder` 后会复制它，并根据是否存在 `obs_no_goal_mask` 选择不同 forward：

```python
if hasattr(policy, "student_moe_encoder"):
    self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder).cpu()
    if hasattr(policy, "obs_no_goal_mask"):
        self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
    self.history_length = policy.history.shape[1]
    self.history = torch.zeros([1, policy.history.shape[1], policy.history.shape[2]], device="cpu")
    self.forward = self.forward_moe_no_goal_cts
    if not hasattr(policy, "obs_no_goal_mask"):
        self.forward = self.forward_moe_cts
```

普通 MoE-CTS forward：

```python
def forward_moe_cts(self, x):
    x = self.normalizer(x)
    self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
    latent, weights = self.student_moe_encoder(self.history.flatten(1))
    x = torch.cat([latent, x], dim=1)
    return self.actor(x), (weights, latent)
```

返回：

```text
actions, (weights, latent)
```

这里多返回 `weights` 是为了部署调试或评估专家权重。如果部署端只需要 action，仍然取第一个返回值。

### 9.4 CTS / MoE-CTS ONNX

当前 ONNX CTS / MoE-CTS 路径不维护内部 history，而是假设输入已经是 stack observation：

```python
def forward_cts(self, x):
    x = self.normalizer(x)
    history, obs_dim = self.flatten_obs(x)
    last_obs = history[:, -obs_dim:]
    latent = self.student_encoder(history)
    x = torch.cat([latent, last_obs], dim=1)
    return self.actor(x)
```

这里的关键是 `flatten_obs(x)`。

它假设输入 `x` 不是一帧 obs，而是按 term 堆叠的多帧 observation。当前 hardcode 的 term 维度是：

```python
term_dims = [3, 3, 3, num_actions, num_actions, num_actions]
```

含义大致是：

```text
base_lin_vel
base_ang_vel
commands
dof_pos
dof_vel
actions
```

所以 ONNX CTS / MoE-CTS 的输入协议和 JIT 不一样：

| 格式 | 输入 | 是否内部维护 history |
| --- | --- | --- |
| JIT CTS / MoE-CTS | 单帧 obs | 是 |
| ONNX CTS / MoE-CTS | stack obs | 否，调用方提前拼好 history |

这一点在部署时必须明确，否则会出现“JIT 正常、ONNX 行为不一致”的问题。

## 10. Normalizer 与 CTS 归一化的区别

这里最容易误解。

项目里至少有三类“归一化”概念：

1. 环境 observation scale。
2. CTS / MoE-CTS encoder 内部的 `L2Norm` / `SimNorm`。
3. exporter 参数里的 empirical observation normalizer。

它们不是同一个东西。

### 10.1 环境 observation scale

`legged_gym/envs/base/legged_robot.py` 里构造 observation 时已经乘了 `obs_scales`：

```python
self.obs_buf = torch.cat((
    self.base_lin_vel * self.obs_scales.lin_vel,
    self.base_ang_vel * self.obs_scales.ang_vel,
    ...
), dim=-1)
```

这是环境输出 observation 的一部分。训练时 actor 看到的是已经 scale 后的 obs，部署时也应该输入同样定义的 obs。

这个不是 exporter 的 `normalizer`。

### 10.2 CTS / MoE-CTS 的 `L2Norm` / `SimNorm`

CTS 里有：

```python
encoder_layers.append(L2Norm())
# 或
encoder_layers.append(SimNorm())
```

MoE-CTS 里也有类似结构。

这些 norm 是 `student_encoder` 或 `student_moe_encoder` 网络内部的层，用于 latent 表征。导出时：

```python
self.student_encoder = copy.deepcopy(policy.student_encoder).cpu()
self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder).cpu()
```

所以这些 norm 层已经跟着 encoder 一起导出了。

因此，不需要因为 CTS 配置里有 `norm_type="l2norm"` 或 `norm_type="simnorm"`，就额外给 exporter 传 normalizer。

### 10.3 HIM 的 latent normalize

HIM estimator 内部会做：

```python
z = F.normalize(z, dim=-1, p=2)
```

导出 wrapper 里也复现了这一点：

```python
z = F.normalize(z, dim=-1, p=2.0)
```

这也是 latent normalization，不是 observation running mean/std。

### 10.4 exporter 的 `normalizer`

`exporter.py` 里的 normalizer 是这个：

```python
def _copy_normalizer(normalizer):
    if normalizer is not None:
        normalizer = copy.deepcopy(normalizer)
        if hasattr(normalizer, "cpu"):
            normalizer = normalizer.cpu()
        if hasattr(normalizer, "eval"):
            normalizer.eval()
        return normalizer
    return torch.nn.Identity()
```

它的语义是：如果训练框架维护了一个 empirical observation normalizer，例如 running mean/std，那么导出时可以把同一个 normalizer 包进部署图。

当前项目里没有看到 runner 创建、更新、保存、加载这个 external empirical normalizer。默认传 `None`，因此导出使用 `Identity()`。

结论：

- 当前 PPO / HIM / CTS / MoE-CTS 默认都不需要额外传 exporter normalizer。
- CTS 的 `norm_type` 已经在 encoder 内部，不需要额外处理。
- 如果未来真的引入 running mean/std observation normalizer，必须同时修改 checkpoint 保存/加载、runner、export 调用，确保训练和部署使用同一个 normalizer。

## 11. JIT 与 ONNX 的输入协议对比

当前各类型输入协议如下：

| 类型 | JIT 输入 | JIT 状态 | ONNX 输入 | ONNX 状态 |
| --- | --- | --- | --- | --- |
| 原生 PPO MLP | 当前 obs | 无状态 | 当前实现偏 stack obs 假设 | 无状态 |
| 原生 RNN/LSTM | 当前 obs | wrapper 内部维护 hidden/cell | 当前不完整 | 应显式 hidden/cell 输入输出 |
| HIM | obs history | 无可变状态 | obs history | 无可变状态 |
| CTS | 当前 obs | wrapper 内部维护 history | stack obs | 调用方维护 history |
| MoE-CTS | 当前 obs | wrapper 内部维护 history | stack obs | 调用方维护 history |

这张表是部署时最重要的接口说明。

如果部署端用 JIT：

- PPO：每步喂当前 obs。
- RNN：每步喂当前 obs，episode reset 时调用 module reset。
- CTS / MoE-CTS：每步喂当前 obs，episode reset 时调用 module reset 清 history。
- HIM：每步喂 obs history，调用方自己提供历史。

如果部署端用 ONNX：

- PPO direct obs 需要修正当前 forward。
- RNN 不建议使用当前 ONNX。
- CTS / MoE-CTS 要提前拼好 stack obs。
- HIM 输入 obs history。

## 12. PKL 导出是什么

`export_policy_as_pkl()` 当前做的事情很简单：

```python
def export_policy_as_pkl(policy, path, filename="policy.pkl"):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy = resolve_policy_from_runner(policy)
    model_dict = policy.state_dict()
    torch.save(model_dict, os.path.join(path, filename))
```

它保存的是模型参数 `state_dict()`，不是一个可直接调用的部署模型。

因此三种格式的含义不同：

| 格式 | 保存内容 | 是否包含部署 forward wrapper | 典型用途 |
| --- | --- | --- | --- |
| JIT `.pt` | TorchScript module | 是 | C++ / Python 直接部署推理 |
| ONNX `.onnx` | ONNX 静态图 | 是，但受输入协议限制 | ONNX Runtime / TensorRT 等部署 |
| PKL `.pkl` | `state_dict()` 参数 | 否 | 后续重新实例化模型并加载权重 |

如果部署端希望“加载后直接 `module(obs)`”，应该使用 JIT 或 ONNX，不应该使用 PKL。PKL 需要知道原始 Python 模型类和构造参数，先创建模型，再 `load_state_dict()`。

## 13. 当前缺口

### 13.1 原生 PPO ONNX direct obs 缺口

问题：

```python
def forward_ppo(self, x):
    history, obs_dim = self.flatten_obs(x)
    last_obs = history[:, -obs_dim:]
    return self.actor(last_obs)
```

这适合 stack obs，不适合最普通的 direct obs。

最小修复：

```python
def forward_ppo(self, x):
    x = self.normalizer(x)
    return self.actor(x)
```

如果项目里确实还要保留 go2 stack obs PPO ONNX，则不要直接覆盖，可以拆成两个 forward：

```python
def forward_ppo(self, x):
    x = self.normalizer(x)
    return self.actor(x)

def forward_ppo_stack(self, x):
    x = self.normalizer(x)
    history, obs_dim = self.flatten_obs(x)
    last_obs = history[:, -obs_dim:]
    return self.actor(last_obs)
```

然后在 `detect_export_type()` 中通过更明确的结构或显式参数选择。

### 13.2 RNN ONNX 缺口

当前 ONNX exporter 不应该静默导出 recurrent policy。

短期修复：

```python
def export_policy_as_onnx(...):
    policy = resolve_policy_from_runner(policy)
    resolved_type = detect_export_type(policy, export_type)
    if resolved_type == "recurrent":
        raise NotImplementedError(
            "ONNX export for recurrent policies is not supported; use JIT export."
        )
    ...
```

长期实现：

```python
class _OnnxRecurrentPolicyExporter(torch.nn.Module):
    def forward(self, obs, hidden_state, cell_state):
        obs = self.normalizer(obs)
        out, (hidden_state, cell_state) = self.rnn(
            obs.unsqueeze(0),
            (hidden_state, cell_state),
        )
        actions = self.actor(out.squeeze(0))
        return actions, hidden_state, cell_state
```

导出时输出名应包含：

```python
input_names = ["obs", "hidden_state_in", "cell_state_in"]
output_names = ["actions", "hidden_state_out", "cell_state_out"]
```

### 13.3 HIM latent 维度硬编码

HIM exporter 当前写了：

```python
parts = self.estimator(obs_history)[:, 0:19]
vel, z = parts[..., :3], parts[..., 3:]
```

这依赖 latent 是 16 维。如果以后 HIM latent 维度改了，应该用 estimator 输出维度推断：

```python
parts = self.estimator(obs_history)
vel, z = parts[..., :3], parts[..., 3:]
```

如果担心 estimator 输出里包含额外字段，再用 actor 输入维度和 `num_one_step_obs` 推导 `3 + latent_dim`。

### 13.4 detect_export_type 对未来算法还不够细

当前检测能覆盖本项目已有的：

- `him`
- `moe_cts`
- `cts`
- `recurrent`
- `ppo`

但 `_TorchPolicyExporter` 里还有 `actor_mcp`、`actor_moe`、`dual_moe_cts` 等 forward 分支。若后续真的加入这些算法，建议把 `detect_export_type()` 也补成更细的顺序，例如：

```python
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
```

这样类型名和实际 forward 分支更一致。

## 14. 后续新增算法时怎么接入

新增算法时，不建议改 `play.py`，也不建议在 config 里加导出字段。推荐按下面顺序接入。

### 14.1 先明确训练模型对象放在哪里

runner 里最好保持以下约定之一：

```python
self.alg.actor_critic = model
```

或者：

```python
self.alg.model = model
```

这样 `resolve_policy_from_runner()` 不用改。

如果用了其他字段，例如：

```python
self.alg.policy_net = model
```

就需要在 `resolve_policy_from_runner()` 增加：

```python
if hasattr(runner.alg, "policy_net"):
    return runner.alg.policy_net
```

### 14.2 再明确部署 forward 图

不要直接假设训练模型的 `act_inference()` 就适合导出。先写清楚部署需要什么：

```text
输入是什么？
是否需要 history？
history 由 wrapper 维护还是调用方维护？
是否需要 hidden state？
是否需要 privileged observation？
输出只要 actions，还是还要 latent / weights？
```

例如：

```text
新算法 MyAlgo:
  input: current obs
  internal state: last 10 obs history
  deploy graph:
    obs -> update history -> my_encoder(history) -> actor(latent, obs) -> actions
  output:
    actions only
```

### 14.3 在 exporter 增加检测分支

```python
def detect_export_type(policy, export_type="auto"):
    if export_type and export_type != "auto":
        return export_type

    if hasattr(policy, "my_encoder"):
        return "my_algo"

    ...
```

### 14.4 增加 wrapper

如果只是类似 CTS 的结构，可以扩展 `_TorchPolicyExporter`：

```python
if hasattr(policy, "my_encoder"):
    self.my_encoder = copy.deepcopy(policy.my_encoder).cpu()
    self.actor = copy.deepcopy(policy.actor).cpu()
    self.history = torch.zeros(...)
    self.forward = self.forward_my_algo
```

```python
def forward_my_algo(self, x):
    x = self.normalizer(x)
    self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
    latent = self.my_encoder(self.history.flatten(1))
    return self.actor(torch.cat([latent, x], dim=1))
```

如果结构和现有差异很大，建议新建专用 wrapper：

```python
class _TorchMyAlgoPolicyExporter(torch.nn.Module):
    ...
```

然后在 `export_policy_as_jit()` 中分流：

```python
resolved_type = detect_export_type(policy, export_type)
if resolved_type == "my_algo":
    policy_exporter = _TorchMyAlgoPolicyExporter(policy, normalizer)
elif resolved_type == "him":
    policy_exporter = _TorchHIMPolicyExporter(policy, normalizer)
else:
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
```

### 14.5 如果需要新 runner，再注册 runner

如果新算法需要新的 runner class：

```python
task_registry.register_runner("MyAlgoRunner", MyAlgoRunner)
```

或者把它加入默认 registry：

```python
RUNNER_REGISTRY = {
    "OnPolicyRunner": OnPolicyRunner,
    "HIMOnPolicyRunner": HIMOnPolicyRunner,
    "OnPolicyRunnerCTS": OnPolicyRunnerCTS,
    "MyAlgoRunner": MyAlgoRunner,
}
```

但这依然是训练/加载路径选择，不是导出格式选择。

## 15. 推荐的实现边界

为了保持项目清晰，建议边界如下：

| 文件 | 应该放什么 | 不应该放什么 |
| --- | --- | --- |
| `helpers.py` | 参数解析、配置更新、路径、seed、通用 helper | 网络结构判断、JIT/ONNX wrapper |
| `exporter.py` | 导出类型识别、runner 到 policy 解析、JIT/ONNX/PKL wrapper | 训练 loop、环境创建、CLI 参数解析 |
| `task_registry.py` | task/env/runner 注册与创建 | 具体导出 format 判断 |
| `play.py` | 触发导出、指定导出目录 | 判断 `student_encoder` / `estimator` 等模型内部字段 |
| config 文件 | 训练算法、网络结构、runner 类型 | `export_policy_type`、`export_formats` |

一句话总结：

```text
config 决定训练什么；
task_registry 决定用什么 runner 加载；
exporter 根据实际模型结构决定怎么导出；
play 只负责触发导出。
```

## 16. 当前最应该优先修的两点

如果下一步继续改代码，我建议优先做两个小修，不引入复杂抽象。

第一，修正或拆分原生 PPO ONNX：

```python
def forward_ppo(self, x):
    x = self.normalizer(x)
    return self.actor(x)
```

第二，禁止 recurrent ONNX 静默导出：

```python
resolved_type = detect_export_type(policy, export_type)
if resolved_type == "recurrent":
    raise NotImplementedError(
        "ONNX export for recurrent policies is not supported; use JIT export."
    )
```

这两个改动能避免最危险的问题：导出成功但部署行为和训练推理不一致。

## 17. 检查导出是否正确

导出后不要只看文件是否生成，至少检查以下几点。

### 17.1 原生 PPO MLP JIT

检查输入输出：

```python
module = torch.jit.load("policy.pt")
obs = torch.zeros(1, env.num_obs)
actions = module(obs)
assert actions.shape[-1] == env.num_actions
```

预期：

```text
obs -> actor -> actions
```

### 17.2 RNN JIT

检查连续调用和 reset：

```python
module = torch.jit.load("policy.pt")
obs = torch.zeros(1, env.num_obs)

a0 = module(obs)
a1 = module(obs)
module.reset()
a2 = module(obs)
```

RNN 是有状态的，所以 `a0` 和 `a1` 不一定相同；reset 后状态应清零。

### 17.3 CTS / MoE-CTS JIT

检查返回值：

```python
out = module(obs)
actions = out[0]
extra = out[1]
```

不要把整个 tuple 当作 action。

### 17.4 HIM JIT / ONNX

检查输入维度是 `obs_history`，不是单帧 obs：

```python
obs_history = torch.zeros(1, env.num_obs)
actions = module(obs_history)
```

其中 `env.num_obs` 对 HIM 通常已经是 history 展平后的 observation。

## 18. 最终理解

这套导出链路的核心不是“所有算法共用同一个 forward”，而是“所有算法共用同一个导出入口”。

统一入口是：

```python
export_policy(runner, path)
```

但内部会按实际模型结构分成不同部署图：

```text
PPO MLP:
  obs -> actor -> actions

PPO RNN:
  obs -> rnn state -> actor -> actions

HIM:
  obs_history -> estimator -> current_obs + vel + latent -> actor -> actions

CTS:
  obs -> internal history -> student_encoder -> latent + obs -> actor -> actions

MoE-CTS:
  obs -> internal history -> student_moe_encoder -> latent + obs -> actor -> actions
```

因此后续新增算法时，关键不是给 config 增加导出字段，而是在 `exporter.py` 里把“这个算法部署时真正需要的计算图”表达清楚。
