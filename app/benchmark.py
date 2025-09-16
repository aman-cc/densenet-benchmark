from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import DenseNet121_Weights

from data import get_val_loader
from optimizations import OptimizationName, inference_context, maybe_convert_input, maybe_jit, prepare_model
from utils import SystemMetrics, get_system_metrics

torch.set_float32_matmul_precision("high")

BATCH_SIZES = [1, 4, 8, 16, 32]


@dataclass
class BenchmarkRow:
    model_variant: str
    batch_size: int
    device: str
    ram_usage_mb: float
    vram_usage_mb: float
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    latency_ms: float
    throughput_samples_sec: float
    accuracy_top1: float
    accuracy_top5: float
    model_size_mb: float
    optimization_technique: str


def measure_accuracy(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, optimization: OptimizationName
) -> Tuple[float, float]:
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            images = maybe_convert_input(images, optimization)
            with inference_context(optimization, device):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, pred1 = probs.topk(1, dim=1)
            _, pred5 = probs.topk(5, dim=1)
            top1_correct += (pred1.squeeze(1) == targets).sum().item()
            top5_correct += sum(targets[i].item() in pred5[i].tolist() for i in range(targets.size(0)))
            total += targets.size(0)
    if total == 0:
        return 0.0, 0.0
    return top1_correct / total, top5_correct / total


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    # assume fp32 params by default
    bytes_total = total_params * 4
    return bytes_total / (1024**2)


def run_single(
    optimization: OptimizationName,
    batch_size: int,
    device: torch.device,
    writer: SummaryWriter,
    results_csv: Path,
    logdir_profiles: Path,
) -> BenchmarkRow:
    # Data
    val_loader, _ = get_val_loader(batch_size=batch_size)

    # Model
    model = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    model = prepare_model(model, optimization, device)

    # maybe JIT after an example batch
    example = torch.randn(batch_size, 3, 224, 224, device=device)
    example = maybe_convert_input(example, optimization)
    model = maybe_jit(model, example, optimization)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            with inference_context(optimization, device):
                _ = model(example)

    # Profiling and latency measurement
    steps = 20
    latencies: List[float] = []
    num_samples = 0
    profile_dir = logdir_profiles / f"{optimization}_bs{batch_size}_{int(time.time())}"
    profile_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                (
                    torch.profiler.ProfilerActivity.CUDA
                    if device.type == "cuda"
                    else torch.profiler.ProfilerActivity.CPU
                ),
            ],
            schedule=torch.profiler.schedule(wait=2, warmup=2, active=steps, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            for i in range(steps + 4):
                images = torch.randn(batch_size, 3, 224, 224, device=device)
                images = maybe_convert_input(images, optimization)
                start = time.perf_counter()
                with inference_context(optimization, device):
                    outputs = model(images)
                torch.cuda.synchronize() if device.type == "cuda" else None
                end = time.perf_counter()
                if i >= 4:  # after wait+warmup
                    latencies.append((end - start) * 1000.0)
                    num_samples += images.size(0)
                prof.step()

    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
    throughput = (num_samples / (sum(latencies) / 1000.0)) if latencies else 0.0

    # Accuracy pass on small subset
    acc_top1, acc_top5 = measure_accuracy(model, val_loader, device, optimization)

    # System metrics
    metrics: SystemMetrics = get_system_metrics(device_index=0)

    row = BenchmarkRow(
        model_variant="densenet121",
        batch_size=batch_size,
        device=str(device),
        ram_usage_mb=metrics.ram_usage_mb,
        vram_usage_mb=metrics.vram_usage_mb,
        cpu_utilization_pct=metrics.cpu_utilization_pct,
        gpu_utilization_pct=metrics.gpu_utilization_pct,
        latency_ms=avg_latency_ms,
        throughput_samples_sec=throughput,
        accuracy_top1=acc_top1,
        accuracy_top5=acc_top5,
        model_size_mb=estimate_model_size_mb(model),
        optimization_technique=optimization,
    )

    # Log to TensorBoard
    writer.add_scalar(f"{optimization}/latency_ms", row.latency_ms, batch_size)
    writer.add_scalar(f"{optimization}/throughput", row.throughput_samples_sec, batch_size)
    writer.add_scalar(f"{optimization}/acc_top1", row.accuracy_top1, batch_size)
    writer.add_scalar(f"{optimization}/acc_top5", row.accuracy_top5, batch_size)

    # Append to CSV
    file_exists = results_csv.exists()
    with results_csv.open("a", newline="") as f:
        writer_csv = csv.DictWriter(
            f,
            fieldnames=[
                "model_variant",
                "batch_size",
                "device",
                "ram_usage_mb",
                "vram_usage_mb",
                "cpu_utilization_pct",
                "gpu_utilization_pct",
                "latency_ms",
                "throughput_samples_sec",
                "accuracy_top1",
                "accuracy_top5",
                "model_size_mb",
                "optimization_technique",
            ],
        )
        if not file_exists:
            writer_csv.writeheader()
        writer_csv.writerow(asdict(row))

    return row


def run_all(
    output_dir: str,
    logdir: str,
    device_str: str,
    optimizations: Iterable[OptimizationName],
    batch_sizes: Iterable[int] = BATCH_SIZES,
) -> List[BenchmarkRow]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tensorboard"))
    results_csv = Path(output_dir) / "benchmark_results.csv"

    rows: List[BenchmarkRow] = []
    for opt in optimizations:
        for bs in batch_sizes:
            row = run_single(
                optimization=opt,
                batch_size=bs,
                device=device,
                writer=writer,
                results_csv=results_csv,
                logdir_profiles=profiles_dir,
            )
            print(f"{opt} | bs={bs} | latency={row.latency_ms:.2f}ms | thrpt={row.throughput_samples_sec:.2f} it/s")
            rows.append(row)
    writer.flush()
    writer.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,amp,channels_last,compile,jit,quantize",
        help="Comma-separated optimization modes",
    )
    args = parser.parse_args()
    optimizations = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = run_all(
        output_dir=args.output_dir,
        logdir=args.logdir,
        device_str=args.device,
        optimizations=optimizations,  # type: ignore[arg-type]
    )
    # Console summary
    print("\nSummary:")
    for r in rows:
        print(
            f"{r.optimization_technique:>12} | bs={r.batch_size:<2} | lat={r.latency_ms:>8.2f}ms | thrput={r.throughput_samples_sec:>8.2f} | top1={r.accuracy_top1:.3f} | top5={r.accuracy_top5:.3f}"
        )


if __name__ == "__main__":
    main()
