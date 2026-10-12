"""Small, explicit standard-LoRA implementation for the PRISM controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(
        self, base: nn.Linear, *, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        device = base.weight.device
        self.lora_A = nn.Parameter(
            torch.empty((self.rank, base.in_features), device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((base.out_features, self.rank), device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        adapter_inputs = self.dropout(inputs).to(dtype=self.lora_A.dtype)
        adapter_hidden = F.linear(adapter_inputs, self.lora_A)
        adapter_output = F.linear(adapter_hidden, self.lora_B)
        return base_output + adapter_output.to(dtype=base_output.dtype) * self.scaling


@dataclass(frozen=True)
class LoRAInjection:
    module_names: tuple[str, ...]
    trainable_parameters: int
    frozen_parameters: int


def _parent_and_leaf(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
        if not isinstance(parent, nn.Module):
            raise RuntimeError(f"LoRA parent is not a module: {dotted_name}")
    return parent, parts[-1]


def inject_all_backbone_linears(
    model: nn.Module, *, rank: int, alpha: float, dropout: float
) -> LoRAInjection:
    """Inject LoRA only into Linear modules strictly under ``model.layers``."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("model.layers.") and isinstance(module, nn.Linear)
    ]
    if not targets:
        raise RuntimeError("no backbone Linear modules were found for LoRA")
    names = [name for name, _module in targets]
    if len(names) != len(set(names)) or any(
        name == "lm_head" or name.startswith("visual.") or ".visual." in name
        for name in names
    ):
        raise RuntimeError("LoRA target scope escaped the text backbone")
    for name, module in targets:
        parent, leaf = _parent_and_leaf(model, name)
        if getattr(parent, leaf) is not module:
            raise RuntimeError(f"LoRA target changed during injection: {name}")
        setattr(
            parent,
            leaf,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    if trainable <= 0 or frozen <= 0:
        raise RuntimeError("LoRA injection did not preserve trainable/frozen parameter groups")
    return LoRAInjection(tuple(names), trainable, frozen)


def adapter_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}.lora_A"] = module.lora_A.detach().cpu().contiguous().clone()
            state[f"{name}.lora_B"] = module.lora_B.detach().cpu().contiguous().clone()
    if not state:
        raise RuntimeError("model has no LoRA adapter state")
    return state


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected: set[str] = set()
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not isinstance(module, LoRALinear):
            continue
        for suffix, parameter in (("lora_A", module.lora_A), ("lora_B", module.lora_B)):
            key = f"{name}.{suffix}"
            expected.add(key)
            value = state.get(key)
            if value is None or value.shape != parameter.shape:
                raise RuntimeError(f"adapter tensor is absent or has wrong shape: {key}")
            parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
    if set(state) != expected:
        raise RuntimeError("adapter state keys differ from injected modules")
