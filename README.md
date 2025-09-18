# DenseNet Benchmark & Optimization Suite

## Project Overview
Production-ready benchmarking and optimization of DenseNet-121 using PyTorch Profiler and TensorBoard for production deployment.

## Setup Instructions
- Docker and Docker Compose required
- NVIDIA GPU with CUDA support recommended

## Usage Guide
```bash
./build_and_run.sh --output-dir ./results --gpu-enabled true
```

## Optimization Approaches
- **Baseline**: Standard DenseNet-121 without optimizations
    - Standard DenseNet-121 model without any optimizations.
    - Serves as a reference point to measure the impact of other optimizations.
- **AMP**: Automatic Mixed Precision for reduced memory usage
    - Uses FP16/FP32 mixed precision for computations during training or inference.
    - Reduces GPU memory usage and improves throughput without significant loss in accuracy.
- **Channels Last**: Memory format optimization for better GPU utilization
    - Changes the memory layout of tensors to NHWC (channels last) instead of NCHW.
    - Improves GPU memory access patterns and increases performance, especially for convolutional networks.
- **Compile**: PyTorch 2.0 compilation for improved performance
    - Uses PyTorch 2.0’s ahead-of-time compilation to optimize model execution.
    - Speeds up forward and backward passes by fusing operations and reducing Python overhead.
- **JIT**: TorchScript compilation for deployment optimization
    - Converts the PyTorch model into a TorchScript graph for deployment.
    - Enables optimizations like operator fusion, removes Python runtime overhead, and allows exporting models for C++ inference.
- **QUANTIZE**: FP16 Quantization for deployment optimization
    - Converts model weights and computations to half-precision (FP16) for inference.
    - Reduces GPU memory footprint and increases inference throughput with minimal accuracy impact.
- **TRT**: TensorRT compilation for deployment optimization
    - Converts the PyTorch model to a TensorRT engine optimized for NVIDIA GPUs.
    - Enables FP16/INT8 acceleration, kernel fusion, and reduced latency for deployment.

## Results Summary
- CSV at `./results/benchmark_results.csv`
- TensorBoard logs and profiling traces at `./logs/tensorboard/`
- Models saved at `./results/models/`
- Metrics available at http://localhost:6006/#scalars
- Profiling results viewer available at http://localhost:6007/#pytorch_profiler

## Performance Analysis
TBD after running.

## Trade-offs Discussion
TBD after running.


