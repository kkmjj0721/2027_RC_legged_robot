# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


PolicyExporterFactory = Callable[..., torch.nn.Module]
PolicyExportAdapter = Dict[str, Optional[PolicyExporterFactory]]

_LEGACY_AUTO_EXPORT_TYPE = "generic"
_POLICY_EXPORT_ADAPTERS: Dict[str, PolicyExportAdapter] = {}


def _normalize_export_type(export_type) -> str:
    if export_type is None:
        return "auto"
    return str(export_type).strip().lower()


def register_policy_export_adapter(
    export_type: str,
    *,
    jit_factory: Optional[PolicyExporterFactory] = None,
    onnx_factory: Optional[PolicyExporterFactory] = None,
):
    """Register exporter factories for an explicit policy export type."""
    resolved_type = _normalize_export_type(export_type)
    if not resolved_type or resolved_type == "auto":
        raise ValueError("export_type must be an explicit non-auto name.")
    if jit_factory is None and onnx_factory is None:
        raise ValueError(f"No exporter factories registered for export_type={resolved_type!r}.")
    if jit_factory is not None and not callable(jit_factory):
        raise TypeError(f"jit_factory for export_type={resolved_type!r} is not callable.")
    if onnx_factory is not None and not callable(onnx_factory):
        raise TypeError(f"onnx_factory for export_type={resolved_type!r} is not callable.")
    _POLICY_EXPORT_ADAPTERS[resolved_type] = {
        "jit": jit_factory,
        "onnx": onnx_factory,
    }


def get_policy_export_adapter(export_type: str) -> PolicyExportAdapter:
    """Return the registered exporter adapter for an explicit export type."""
    resolved_type = _normalize_export_type(export_type)
    if resolved_type == "auto":
        resolved_type = _LEGACY_AUTO_EXPORT_TYPE
    if resolved_type not in _POLICY_EXPORT_ADAPTERS:
        supported = ", ".join(sorted(_POLICY_EXPORT_ADAPTERS))
        raise ValueError(
            f"Unsupported export_type={export_type!r}. "
            f"Register an adapter or pass one of: {supported}"
        )
    return _POLICY_EXPORT_ADAPTERS[resolved_type]


def detect_export_type(policy: object, export_type: str = "auto") -> str:
    """Resolve the explicitly requested policy export type.

    ``auto`` is retained only as a legacy API value and maps to the generic
    exporter adapter. This function intentionally does not inspect policy
    internals; specialized graphs such as HIM must pass export_type explicitly.
    """
    del policy
    resolved_type = _normalize_export_type(export_type)
    if not resolved_type or resolved_type == "auto":
        resolved_type = _LEGACY_AUTO_EXPORT_TYPE
    get_policy_export_adapter(resolved_type)
    return resolved_type


def _build_policy_exporter(export_type: str, export_format: str, policy: object, **kwargs):
    adapter = get_policy_export_adapter(export_type)
    exporter_factory = adapter.get(export_format)
    if exporter_factory is None:
        raise NotImplementedError(
            f"{export_format.upper()} export is not registered for export_type={export_type!r}."
        )
    return exporter_factory(policy=policy, **kwargs)


def resolve_policy_from_runner(runner: object):
    """Return the model object that should be exported from a runner."""
    if not hasattr(runner, "alg"):
        return runner

    if hasattr(runner.alg, "actor_critic"):
        return runner.alg.actor_critic
    if hasattr(runner.alg, "model"):
        return runner.alg.model

    raise ValueError(f"Unsupported runner algorithm type: {type(runner.alg).__name__}")


def export_policy_as_jit(
    policy: object,
    path: str,
    normalizer: Optional[object] = None,
    filename="policy.pt",
    export_type="auto",
):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy = resolve_policy_from_runner(policy)
    resolved_type = detect_export_type(policy, export_type)
    policy_exporter = _build_policy_exporter(
        resolved_type,
        "jit",
        policy,
        normalizer=normalizer,
    )
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object,
    path: str,
    normalizer: Optional[object] = None,
    filename="policy.onnx",
    verbose=False,
    export_type="auto",
    input_dim=None,
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy = resolve_policy_from_runner(policy)
    resolved_type = detect_export_type(policy, export_type)
    policy_exporter = _build_policy_exporter(
        resolved_type,
        "onnx",
        policy,
        normalizer=normalizer,
        input_dim=input_dim,
        verbose=verbose,
    )
    policy_exporter.export(path, filename)


def export_policy_as_pkl(
    policy: nn.Module, path: str, filename="policy.pkl"
):
    """Export policy into a Torch pkl file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported pkl file. Defaults to "policy.pkl".
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy = resolve_policy_from_runner(policy)
    model_dict = policy.state_dict()
    torch.save(model_dict, os.path.join(path, filename))


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


"""
Helper Classes - Private.
"""


def _copy_normalizer(normalizer):
    if normalizer is not None:
        normalizer = copy.deepcopy(normalizer)
        if hasattr(normalizer, "cpu"):
            normalizer = normalizer.cpu()
        if hasattr(normalizer, "eval"):
            normalizer.eval()
        return normalizer
    return torch.nn.Identity()


def _first_linear_in_features(module):
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            return layer.in_features
    return None


def _last_linear_out_features(module):
    out_features = None
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            out_features = layer.out_features
    return out_features


def _resolve_him_num_one_step_obs(actor_critic):
    if hasattr(actor_critic, "num_one_step_obs"):
        return int(actor_critic.num_one_step_obs)

    actor_input_dim = _first_linear_in_features(actor_critic.actor)
    estimator_output_dim = _last_linear_out_features(actor_critic.estimator.encoder)
    if actor_input_dim is not None:
        if estimator_output_dim is None:
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


def _resolve_him_input_dim(actor_critic, input_dim=None):
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
    raise ValueError("Unable to infer HIM ONNX input dimension.")


class _TorchHIMPolicyExporter(torch.nn.Module):
    """Exporter of HIM actor-critic into JIT file."""

    def __init__(self, actor_critic, normalizer=None):
        super().__init__()
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

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxHIMPolicyExporter(torch.nn.Module):
    """Exporter of HIM actor-critic into ONNX file."""

    def __init__(self, actor_critic, normalizer=None, input_dim=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.actor = copy.deepcopy(actor_critic.actor).cpu().eval()
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder).cpu().eval()
        self.num_one_step_obs = _resolve_him_num_one_step_obs(actor_critic)
        self.input_dim = _resolve_him_input_dim(actor_critic, input_dim)
        self.normalizer = _copy_normalizer(normalizer)

    def forward(self, obs_history):
        obs_history = self.normalizer(obs_history)
        parts = self.estimator(obs_history)[:, 0:19]
        vel, z = parts[..., :3], parts[..., 3:]
        z = F.normalize(z, dim=-1, p=2.0)
        current_obs = obs_history[:, :self.num_one_step_obs]
        return self.actor(torch.cat((current_obs, vel, z), dim=1))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)

        torch.onnx.export(
            self,
            obs,
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


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file."""

    def __init__(self, policy, normalizer=None):
        super().__init__()
        self.is_recurrent = getattr(policy, "is_recurrent", False)
        # copy policy parameters
        if hasattr(policy, "student_encoder"):
            self.student_encoder = copy.deepcopy(policy.student_encoder).cpu()
            self.history = torch.zeros([1, policy.history.shape[1], policy.history.shape[2]], device='cpu')
            self.forward = self.forward_cts
        if hasattr(policy, "student_moe_encoder"):
            self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder).cpu()
            if hasattr(policy, "obs_no_goal_mask"):
                self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.history_length = policy.history.shape[1]
            self.history = torch.zeros([1, policy.history.shape[1], policy.history.shape[2]], device='cpu')
            self.forward = self.forward_moe_no_goal_cts
            if not hasattr(policy, "obs_no_goal_mask"):
                self.forward = self.forward_moe_cts
        if hasattr(policy, "actor_mcp"):
            self.actor = copy.deepcopy(policy.actor_mcp)
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.forward = self.forward_mcp_cts
        elif hasattr(policy, "actor_moe"):
            self.actor = copy.deepcopy(policy.actor_moe)
            self.forward = self.forward_ac_moe
        elif hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
        elif hasattr(policy, "student"):
            self.actor = copy.deepcopy(policy.student)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_s.rnn)
        else:
            raise ValueError("Policy does not have an actor/student module.")
        if hasattr(policy, "student_moe_encoder") and hasattr(policy, "actor_moe"):
            self.forward = self.forward_dual_moe_cts
        # set up recurrent network
        if self.is_recurrent:
            self.rnn.cpu()
            self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
            self.forward = self.forward_lstm
            self.reset = self.reset_memory
        # copy normalizer if exists
        self.normalizer = _copy_normalizer(normalizer)

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x)

    def forward(self, x):
        return self.actor(self.normalizer(x))
    
    def forward_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (None, latent)
    
    def forward_moe_no_goal_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        history_no_goal = self.history.reshape(1, self.history_length, -1)[:, :, self.obs_no_goal_mask].reshape(1, -1)
        latent, weights = self.student_moe_encoder(self.history.flatten(1), history_no_goal)
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (weights, latent)

    def forward_moe_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent, weights = self.student_moe_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (weights, latent)
    
    def forward_mcp_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        x_no_goal = x[:, self.obs_no_goal_mask]
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        x_no_goal = torch.cat([latent, x_no_goal], dim=1)
        mean_action, _, weights = self.actor(x, x_no_goal)
        return mean_action, (weights, latent)

    def forward_ac_moe(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        mean, weights = self.actor(x)
        return mean, (weights, latent)

    def forward_dual_moe_cts(self, x):  # x is single observations
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent, student_weights = self.student_moe_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        mean, actor_weights = self.actor(x)
        return mean, (student_weights, actor_weights, latent)

    @torch.jit.export
    def reset(self):
        if hasattr(self, 'history'):
            self.history = torch.zeros_like(self.history)

    def reset_memory(self):
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.input_dim = None
        self.num_actions = 12
        self.is_recurrent = getattr(policy, "is_recurrent", False)
        self.normalizer = _copy_normalizer(normalizer)
        
        # copy policy parameters
        if hasattr(policy, 'student_encoder'):
            self.student_encoder = copy.deepcopy(policy.student_encoder)
            self.forward = self.forward_cts
            self.input_dim = self.student_encoder[0].in_features
            
        elif hasattr(policy, "student_moe_encoder"):
            self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)
            self.history_length = policy.history.shape[1]
            self.forward = self.forward_moe_no_goal_cts
            self.input_dim = self.history_length * policy.history.shape[2]
            if hasattr(policy, "obs_no_goal_mask"):
                self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            else:
                self.forward = self.forward_moe_cts
        
        else:  # PPO
            self.forward = self.forward_ppo
            
        if hasattr(policy, "actor_mcp"):
            self.actor = copy.deepcopy(policy.actor_mcp)
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()
            self.history_length = policy.history.shape[1]
            self.forward = self.forward_mcp_cts
        elif hasattr(policy, "actor"):
            self.actor = copy.deepcopy(policy.actor)
            if self.is_recurrent:
                self.rnn = copy.deepcopy(policy.memory_a.rnn)
            if self.input_dim is None:
                 self.input_dim = self.actor[0].in_features
        else:
            raise ValueError("Policy does not have an actor/student module.")

    def flatten_obs(self, x):  # flatten stack obs by terms to stack by step frames
        term_dims = [3, 3, 3, self.num_actions, self.num_actions, self.num_actions]
        obs_dim = sum(term_dims)
        if x.shape[1] % obs_dim != 0:
            raise ValueError(f"x.shape[1] ({x.shape[1]}) 不是 obs_dim ({obs_dim}) 的整数倍")
            
        frames = x.shape[1] // obs_dim
        split_sizes = [dim * frames for dim in term_dims]
        # [B, dim0*frames], [B, dim1*frames], ...
        term_chunks = torch.split(x, split_sizes, dim=1)

        # [ [B, frames, dim0], [B, frames, dim1], ... ]
        frame_terms_reshaped = [
            chunk.view(-1, frames, dim) 
            for chunk, dim in zip(term_chunks, term_dims)
        ]

        history_by_frame = []
        for i in range(frames):
            # [ [B, dim0], [B, dim1], ... ]
            terms_for_this_frame = [ftr[:, i, :] for ftr in frame_terms_reshaped]
            history_by_frame.append(torch.cat(terms_for_this_frame, dim=1))
        # [B, (Frame0_AllTerms), (Frame1_AllTerms), ...]
        history = torch.cat(history_by_frame, dim=1)
        return history, obs_dim
    
    def forward_ppo(self, x):  # x is stack observations by terms
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)
        last_obs = history[:, -obs_dim:]
        return self.actor(last_obs)

    def forward_cts(self, x):  # x is stack observations by terms
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]
        latent = self.student_encoder(history)
        x = torch.cat([latent, last_obs], dim=1)
        
        return self.actor(x)

    def forward_moe_no_goal_cts(self, x):
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]
        history_3d = history.view(-1, self.history_length, obs_dim)
        history_no_goal = history_3d[:, :, self.obs_no_goal_mask].reshape(x.shape[0], -1)

        latent, weights = self.student_moe_encoder(history, history_no_goal)
        x = torch.cat([latent, last_obs], dim=1)

        return self.actor(x), weights, latent

    def forward_moe_cts(self, x):
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]

        latent, weights = self.student_moe_encoder(history)
        x = torch.cat([latent, last_obs], dim=1)

        return self.actor(x), weights, latent
    
    def forward_mcp_cts(self, x):
        x = self.normalizer(x)
        history, obs_dim = self.flatten_obs(x)

        last_obs = history[:, -obs_dim:]
        obs_no_goal = last_obs[:, self.obs_no_goal_mask]
        latent = self.student_encoder(history)
        x_in = torch.cat([latent, last_obs], dim=1)
        x_no_goal_in = torch.cat([latent, obs_no_goal], dim=1)
        
        mean_action, _, weights = self.actor(x_in, x_no_goal_in)
        return mean_action, weights

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)
        
        output_names = ["actions"]
        if self.forward == self.forward_moe_no_goal_cts:
            output_names.append("weights")
            output_names.append("latent")
        if self.forward == self.forward_mcp_cts:
            output_names.append("weights")

        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=output_names,
            dynamic_axes={},
        )


def _make_torch_policy_exporter(policy, normalizer=None, **_kwargs):
    return _TorchPolicyExporter(policy, normalizer)


def _make_onnx_policy_exporter(policy, normalizer=None, verbose=False, **_kwargs):
    return _OnnxPolicyExporter(policy, normalizer, verbose)


def _make_torch_him_policy_exporter(policy, normalizer=None, **_kwargs):
    return _TorchHIMPolicyExporter(policy, normalizer)


def _make_onnx_him_policy_exporter(policy, normalizer=None, input_dim=None, verbose=False, **_kwargs):
    return _OnnxHIMPolicyExporter(policy, normalizer, input_dim, verbose)


def _register_builtin_policy_export_adapters():
    generic_export_types = (
        "generic",
        "default",
        "ppo",
        "recurrent",
        "cts",
        "moe_cts",
        "mcp_cts",
        "ac_moe_cts",
        "dual_moe_cts",
    )
    for export_type in generic_export_types:
        register_policy_export_adapter(
            export_type,
            jit_factory=_make_torch_policy_exporter,
            onnx_factory=_make_onnx_policy_exporter,
        )
    register_policy_export_adapter(
        "him",
        jit_factory=_make_torch_him_policy_exporter,
        onnx_factory=_make_onnx_him_policy_exporter,
    )


_register_builtin_policy_export_adapters()
