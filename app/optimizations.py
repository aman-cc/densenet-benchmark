from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from torch import nn


OptimizationName = Literal["baseline", "amp", "channels_last", "compile", "jit"]


def prepare_model(model: nn.Module, optimization: OptimizationName, device: torch.device) -> nn.Module:
	model = model.to(device)
	if optimization in ("channels_last",):
		model = model.to(memory_format=torch.channels_last)
	if optimization == "compile":
		try:
			model = torch.compile(model)  # type: ignore[attr-defined]
		except Exception:
			pass
	return model.eval()


def inference_context(optimization: OptimizationName, device: torch.device):
	if optimization in ("amp",):
		return torch.autocast(device_type=device.type, dtype=torch.float16)
	# no-op context
	from contextlib import nullcontext
	return nullcontext()


def maybe_convert_input(x: torch.Tensor, optimization: OptimizationName) -> torch.Tensor:
	if optimization in ("channels_last",):
		return x.to(memory_format=torch.channels_last)
	return x


def maybe_jit(model: nn.Module, example: torch.Tensor, optimization: OptimizationName) -> nn.Module:
	if optimization == "jit":
		try:
			model = torch.jit.trace(model, example)
			torch.jit.freeze(model)
		except Exception:
			pass
	return model
