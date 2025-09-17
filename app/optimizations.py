from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from torch import nn
import torch_tensorrt

OptimizationName = Literal["baseline", "amp", "channels_last", "compile", "jit", "trt"]


def prepare_model(model: nn.Module, optimization: OptimizationName, device: torch.device, input_shape = (1,3,224,224)) -> nn.Module:
    model = model.to(device)
    if optimization in ("channels_last",):
        model = model.to(memory_format=torch.channels_last)
    elif optimization == "compile":
        model = torch.compile(model)  # type: ignore[attr-defined]
    elif optimization in ("quantize"):
        model = model.half()
    elif optimization == "trt":
        model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(input_shape, dtype=torch.float32)],
            enabled_precisions={torch.float32},
            workspace_size=1 << 30,
        )
    elif optimization == "trt_fp16":
        model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input(input_shape, dtype=torch.float16)],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,
        )
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
    elif optimization in ("quantize", "trt_fp16"):
        return x.half()
    return x


def maybe_jit(model: nn.Module, example: torch.Tensor, optimization: OptimizationName) -> nn.Module:
    if optimization == "jit":
        try:
            model = torch.jit.trace(model, example)
            torch.jit.freeze(model)
        except Exception:
            pass
    return model
