from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import psutil

try:
	import pynvml  # type: ignore
	pynvml.nvmlInit()
	_NVML_AVAILABLE = True
except Exception:
	_NVML_AVAILABLE = False


@dataclass
class SystemMetrics:
	ram_usage_mb: float
	vram_usage_mb: float
	cpu_utilization_pct: float
	gpu_utilization_pct: float
	peak_vram_allocated_mb: Optional[float] = None


def get_system_metrics(device_index: int = 0) -> SystemMetrics:
	process = psutil.Process(os.getpid())
	ram_bytes = process.memory_info().rss
	ram_mb = ram_bytes / (1024 ** 2)
	cpu_pct = psutil.cpu_percent(interval=0.0)
	gpu_util = 0.0
	vram_mb = 0.0
	peak_vram = None
	if _NVML_AVAILABLE:
		h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
		util = pynvml.nvmlDeviceGetUtilizationRates(h)
		mem = pynvml.nvmlDeviceGetMemoryInfo(h)
		gpu_util = float(util.gpu)
		vram_mb = float(mem.used) / (1024 ** 2)
	return SystemMetrics(
		ram_usage_mb=ram_mb,
		vram_usage_mb=vram_mb,
		cpu_utilization_pct=cpu_pct,
		gpu_utilization_pct=gpu_util,
		peak_vram_allocated_mb=peak_vram,
	)


def timed(fn):
	def wrapper(*args, **kwargs):
		start = time.perf_counter()
		result = fn(*args, **kwargs)
		end = time.perf_counter()
		return result, (end - start)
	return wrapper
