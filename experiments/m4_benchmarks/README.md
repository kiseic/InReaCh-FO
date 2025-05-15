# Adaptive Vision-Based Anomaly Detection - M4 Benchmarks

This directory contains benchmarks for evaluating various anomaly detection models on the MacBook Pro M4 hardware.

## Overview

The benchmark suite includes:

1. **Model Latency Comparison** - Measures inference speed of each model
2. **Dataset Performance Evaluation** - Measures AUROC on MVTec-AD2, VisA, and VIADUCT datasets
3. **Drift Recovery Comparison** - Evaluates models' ability to recover from domain shifts
4. **LAFT Language Instruction Evaluation** - Tests language-guided adaptation performance

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- transformers
- diffusers
- timm

## Quick Start

To run all benchmarks, use the following command:

```bash
python experiments/m4_benchmarks/run_all_benchmarks.py
```

For a quick evaluation with reduced samples (useful for testing):

```bash
python experiments/m4_benchmarks/run_all_benchmarks.py --fast-mode
```

## Individual Benchmarks

### Model Latency Benchmark

Measures inference time for all models:

```bash
python experiments/m4_benchmarks/model_latency_benchmark.py
```

Options:
- `--runs`: Number of inference runs (default: 100)
- `--warmup`: Number of warmup runs (default: 10)
- `--device`: Device to use (mps, cuda, cpu)
- `--skip-{modelname}`: Skip specific models

### Dataset Performance Benchmark

Evaluates models on anomaly detection datasets:

```bash
python experiments/m4_benchmarks/dataset_performance_benchmark.py --dataset mvtec_ad2
```

Options:
- `--dataset`: Dataset to evaluate (mvtec_ad2, visa, viaduct)
- `--num-categories`: Number of categories to test (default: all)
- `--fast-mode`: Use fewer images for quicker testing

### Drift Recovery Benchmark

Tests model adaptation to changing conditions:

```bash
python experiments/m4_benchmarks/drift_recovery_benchmark.py --drift-type lighting
```

Options:
- `--drift-type`: Type of drift (lighting, scratch, texture, defects)
- `--duration`: Length of test sequence in frames
- `--intensity`: Drift intensity (0.0-1.0)
- `--save-video`: Save test sequences as videos

### LAFT Instruction Benchmark

Evaluates language-guided adaptation:

```bash
python experiments/m4_benchmarks/laft_instruction_benchmark.py
```

Options:
- `--categories`: Categories to test (comma-separated)
- `--max-instructions`: Maximum instructions per type to test
- `--max-images`: Maximum images to test per category

## Output

All benchmarks save results to `experiments/m4_benchmarks/results/` by default:

- CSV files with metrics
- JSON files with detailed results
- PNG and SVG charts visualizing performance
- Log files with run information

## Advanced Usage

For running on specific hardware or with custom settings:

```bash
# Using MPS acceleration on M4 Macs
python experiments/m4_benchmarks/run_all_benchmarks.py --device mps

# Testing only specific models
python experiments/m4_benchmarks/run_all_benchmarks.py --skip-samlad --skip-laft

# Running a specific benchmark with custom settings
python experiments/m4_benchmarks/dataset_performance_benchmark.py --dataset mvtec_ad2 --num-categories 3 --device mps
```

## Result Analysis

After running the benchmarks, you can find visual charts in the results directory that show:

1. Latency comparison across models
2. AUROC performance on different datasets
3. Drift recovery capability plots
4. Instruction effectiveness for LAFT

## References

- MVTec-AD2 dataset: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- VisA dataset: [https://github.com/amazon-science/spot-diff](https://github.com/amazon-science/spot-diff)
- VIADUCT dataset: [Paper](https://doi.org/10.1109/WACV45572.2020.9093444)