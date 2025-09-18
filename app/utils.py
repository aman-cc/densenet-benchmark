from __future__ import annotations

import os
import requests
import time
from dataclasses import dataclass
from typing import Optional

from tqdm import tqdm
from loguru import logger
import zipfile

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
    ram_mb = ram_bytes / (1024**2)
    cpu_pct = psutil.cpu_percent(interval=0.0)
    gpu_util = 0.0
    vram_mb = 0.0
    peak_vram = None
    if _NVML_AVAILABLE:
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        gpu_util = float(util.gpu)
        vram_mb = float(mem.used) / (1024**2)
    return SystemMetrics(
        ram_usage_mb=round(ram_mb, 2),
        vram_usage_mb=round(vram_mb, 2),
        cpu_utilization_pct=round(cpu_pct, 2),
        gpu_utilization_pct=round(gpu_util, 2),
        peak_vram_allocated_mb=peak_vram,
    )


def timed(fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        return result, (end - start)

    return wrapper

def download_file(url: str, destination: str) -> None:
    try:
        head_response = requests.head(url, allow_redirects=True)
        head_response.raise_for_status()
        total_size_in_bytes = int(head_response.headers.get('Content-Length', 0))
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in tqdm(
                    response.iter_content(chunk_size=8192),
                    total=total_size_in_bytes // 8192,
                    unit='KB',
                    unit_scale=True,
                    desc="Downloading File"
                ):
                # for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info(f"Downloaded successfully to {destination}")
    except Exception as e:
        logger.info(f"Download failed: {e}")

def unzip_file(zip_path: str, extract_dir: str) -> None:
    """Method to extract zipfile"""
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"Unzipped {zip_path} to {extract_dir}")

