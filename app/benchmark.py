from __future__ import annotations

import argparse
import csv
import os
import os.path as osp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import DenseNet121_Weights
from tqdm import tqdm
from loguru import logger

from data import get_val_loader
from optimizations import OptimizationName, inference_context, maybe_convert_input, maybe_jit, prepare_model
from utils import SystemMetrics, get_system_metrics

torch.set_float32_matmul_precision("high")

BATCH_SIZES = [1, 4, 8, 16, 32]


def save_ground_truth_predictions(
    model: torch.nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    optimization: OptimizationName,
    ground_truth_csv: Path
) -> None:
    """Save model predictions as ground truth labels to CSV"""
    predictions = []
    sample_ids = []

    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(loader)):
            images = images.to(device, non_blocking=True)
            images = maybe_convert_input(images, optimization)
            with inference_context(optimization, device):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, pred1 = probs.topk(1, dim=1)
            _, pred5 = probs.topk(5, dim=1)
            # Store predictions for each sample in the batch
            for i in range(pred1.size(0)):
                sample_id = batch_idx * loader.batch_size + i
                sample_ids.append(sample_id)
                predictions.append({
                    'sample_id': sample_id,
                    'pred_top1': pred1[i].item(),
                    'pred_top5': pred5[i].tolist()
                })

    # Save to CSV
    with ground_truth_csv.open("w", newline="") as f:
        fieldnames = ['sample_id', 'pred_top1', 'pred_top5']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def load_ground_truth_predictions(ground_truth_csv: Path) -> Dict[int, Dict]:
    """Load ground truth predictions from CSV"""
    ground_truth = {}

    if not ground_truth_csv.exists():
        logger.warning(f"Ground truth file {ground_truth_csv} does not exist")
        return ground_truth

    with ground_truth_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = int(row['sample_id'])
            ground_truth[sample_id] = {
                # 'true_label': int(row['true_label']),
                'pred_top1': int(row['pred_top1']),
                'pred_top5': eval(row['pred_top5'])  # Convert string representation back to list
            }

    return ground_truth


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
    model: torch.nn.Module, loader: DataLoader, device: torch.device, optimization: OptimizationName,
    ground_truth: Dict[int, Dict] = None
) -> Tuple[float, float]:
    """Method to get model accuracy metrics for the model"""
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            images = maybe_convert_input(images, optimization)
            with inference_context(optimization, device):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, pred1 = probs.topk(1, dim=1)
            _, pred5 = probs.topk(5, dim=1)

            for i in range(pred1.size(0)):
                sample_id = batch_idx * loader.batch_size + i
                if sample_id in ground_truth:
                    # Compare against ground truth predictions
                    gt_pred1 = ground_truth[sample_id]['pred_top1']
                    gt_pred5 = ground_truth[sample_id]['pred_top5']

                    if pred1[i].item() == gt_pred1:
                        top1_correct += 1
                    if pred1[i].item() in gt_pred5:
                        top5_correct += 1
                    total += 1

    if total == 0:
        return 0.0, 0.0
    return top1_correct / total, top5_correct / total


def estimate_model_size_mb(model: torch.nn.Module) -> float:
    """Method to get model size for the model"""
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
    is_first_run: bool = False,
) -> BenchmarkRow:
    """Method to inference on the specified optimization, batch and device"""
    # Data
    val_loader = get_val_loader(batch_size=batch_size)

    # Model
    model = torchvision.models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    example = torch.randn(batch_size, 3, 224, 224, device=device)
    model = prepare_model(model, optimization, device, input_shape=tuple(example.shape))

    # maybe JIT after an example batch
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
    ground_truth_csv = Path(osp.join(osp.dirname(results_csv), "pred_label.csv"))

    if is_first_run:
        # Save ground truth predictions for first run
        save_ground_truth_predictions(model, val_loader, device, optimization, ground_truth_csv)
        val_loader = get_val_loader(batch_size=batch_size)

    ground_truth = load_ground_truth_predictions(ground_truth_csv)
    acc_top1, acc_top5 = measure_accuracy(model, val_loader, device, optimization, ground_truth)

    # System metrics
    metrics: SystemMetrics = get_system_metrics(device_index=0)
    logged_device = f"{device}_{torch.cuda.get_device_name(0)}" if str(device) == "cuda" else str(device)

    row = BenchmarkRow(
        model_variant="densenet121",
        batch_size=batch_size,
        device=logged_device,
        ram_usage_mb=metrics.ram_usage_mb,
        vram_usage_mb=metrics.vram_usage_mb,
        cpu_utilization_pct=metrics.cpu_utilization_pct,
        gpu_utilization_pct=metrics.gpu_utilization_pct,
        latency_ms=round(avg_latency_ms, 2),
        throughput_samples_sec=round(throughput, 2),
        accuracy_top1=round(acc_top1, 4),
        accuracy_top5=round(acc_top5, 4),
        model_size_mb=round(estimate_model_size_mb(model), 2),
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
    """Method to inference on list of specified optimizations"""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logdir).mkdir(parents=True, exist_ok=True)
    profiles_dir = Path(output_dir) / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(logdir, "tensorboard"))
    results_csv = Path(output_dir) / "benchmark_results.csv"
    ground_truth_csv = Path(output_dir) / "ground_truth_predictions.csv"

    rows: List[BenchmarkRow] = []
    is_first_run = True

    for opt in optimizations:
        batch_sizes = [1] if opt in ["trt", "trt_fp16"] else BATCH_SIZES
        for bs in batch_sizes:
            row = run_single(
                optimization=opt,
                batch_size=bs,
                device=device,
                writer=writer,
                results_csv=results_csv,
                logdir_profiles=profiles_dir,
                is_first_run=is_first_run,
            )

            # Load ground truth after first run
            if is_first_run:
                is_first_run = False

            logger.info(f"{opt} | bs={bs} | acc={row.accuracy_top1:.2f} | latency={row.latency_ms:.2f}ms | thrpt={row.throughput_samples_sec:.2f} it/s")
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
        default="baseline,amp,channels_last,compile,quantize,trt",
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
    logger.info("\nSummary:")
    for r in rows:
        logger.info(
            f"{r.optimization_technique:>12} | bs={r.batch_size:<2} | lat={r.latency_ms:>8.2f}ms | thrput={r.throughput_samples_sec:>8.2f} | top1={r.accuracy_top1:.3f} | top5={r.accuracy_top5:.3f}"
        )


if __name__ == "__main__":
    main()
