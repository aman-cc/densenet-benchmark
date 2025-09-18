from __future__ import annotations

import os
from typing import Callable, Literal, Optional, Tuple

import torch
from torch import nn
from loguru import logger

AVAIL_CUDA = False
if torch.cuda.is_available():
    import torch_tensorrt
    AVAIL_CUDA = True

OptimizationName = Literal["baseline", "amp", "channels_last", "compile", "jit", "trt"]


def prepare_model(model: nn.Module, optimization: OptimizationName, device: torch.device, input_shape: Tuple = (1,3,224,224)) -> nn.Module:
    """Method to prepare model for the specified optimization"""
    model = model.to(device)
    if optimization in ("channels_last",):
        model = model.to(memory_format=torch.channels_last)
    elif optimization == "compile":
        model = torch.compile(model)  # type: ignore[attr-defined]
    elif optimization in ("quantize"):
        model = model.half()
    elif optimization == "trt":
        if AVAIL_CUDA:
            model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(input_shape, dtype=torch.float32)],
                enabled_precisions={torch.float32},
                workspace_size=1 << 30,
            )
        else:
            logger.info("CUDA not avaiable, skipping TensorRT optimization")
    elif optimization == "trt_fp16":
        if AVAIL_CUDA:
            model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(input_shape, dtype=torch.float16)],
                enabled_precisions={torch.float16},
                workspace_size=1 << 30,
            )
        else:
            logger.info("CUDA not avaiable, skipping TensorRT optimization")
    model = model.eval()
    os.makedirs("results/models", exist_ok=True)
    torch.save(model, f"results/models/{optimization}_{device}.pth")
    return model


def inference_context(optimization: OptimizationName, device: torch.device):
    """Method to get inference context for the specified optimization"""
    if optimization in ("amp",):
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    # no-op context
    from contextlib import nullcontext

    return nullcontext()


def maybe_convert_input(x: torch.Tensor, optimization: OptimizationName) -> torch.Tensor:
    """Method to get input tensor for the specified optimization"""
    if optimization in ("channels_last",):
        return x.to(memory_format=torch.channels_last)
    elif optimization in ("quantize", "trt_fp16"):
        return x.half()
    return x


def maybe_jit(model: nn.Module, example: torch.Tensor, optimization: OptimizationName) -> nn.Module:
    """Method to get trace jot model"""
    if optimization == "jit":
        try:
            model = torch.jit.trace(model, example)
            torch.jit.freeze(model)
        except Exception:
            pass
    return model
