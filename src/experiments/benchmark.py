#!/usr/bin/env python3
"""
Benchmark script for Adaptive Vision-Based Anomaly Detection experiments.

This script provides a unified interface for running benchmarks on
different components and datasets, collecting metrics, and saving results.
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Conditionally import torch if available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Create results directory if it doesn't exist
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class Benchmark:
    """Base benchmark class for all experiments."""
    
    def __init__(
        self,
        name: str,
        output_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        Initialize benchmark.
        
        Args:
            name: Benchmark name
            output_dir: Directory to save results
            device: Device to run experiments on ('cpu', 'cuda', 'mps')
        """
        self.name = name
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._validate_device(device)
        self.metrics = {}
        self.start_time = None
        
    def _validate_device(self, device: str) -> str:
        """Validate and return the appropriate device."""
        if not HAS_TORCH:
            return "cpu"
            
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            return "cpu"
            
        if device == "mps" and not (hasattr(torch.backends, "mps") 
                                    and torch.backends.mps.is_available()):
            print("MPS requested but not available, falling back to CPU")
            return "cpu"
            
        return device
        
    def start(self):
        """Start the benchmark and record time."""
        self.start_time = time.time()
        print(f"Starting benchmark: {self.name}")
        
    def end(self):
        """End the benchmark and record elapsed time."""
        if self.start_time is None:
            raise ValueError("Benchmark was never started")
            
        elapsed = time.time() - self.start_time
        self.metrics["elapsed_time"] = elapsed
        print(f"Benchmark {self.name} completed in {elapsed:.2f}s")
        
    def record_metric(self, name: str, value: Union[float, List, Dict, np.ndarray]):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
        """
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(value, np.ndarray):
            value = value.tolist()
            
        # Convert torch tensors to lists if torch is available
        if HAS_TORCH and isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy().tolist()
            
        self.metrics[name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return self.metrics
        
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save results to a JSON file.
        
        Args:
            filename: Optional filename override
            
        Returns:
            Path to saved results file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.json"
            
        output_path = self.output_dir / filename
        
        # Add metadata
        results = {
            "benchmark": self.name,
            "device": self.device,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to {output_path}")
        return str(output_path)
        
    def visualize(self, output_format: str = "png"):
        """
        Create visualization of benchmark results.
        
        Args:
            output_format: Output format (png, svg, pdf)
        """
        # Implemented by subclasses
        raise NotImplementedError("Visualization must be implemented by subclasses")


class LatencyBenchmark(Benchmark):
    """Benchmark for measuring latency of different components."""
    
    def __init__(
        self,
        model_name: str,
        batch_size: int = 1,
        num_frames: int = 100,
        resolution: Tuple[int, int] = (224, 224),
        warmup: int = 10,
        **kwargs
    ):
        """
        Initialize latency benchmark.
        
        Args:
            model_name: Name of the model to benchmark
            batch_size: Batch size to use
            num_frames: Number of frames to process
            resolution: Input resolution (width, height)
            warmup: Number of warmup iterations
        """
        super().__init__(name=f"latency_{model_name}", **kwargs)
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.resolution = resolution
        self.warmup = warmup
        
        self.timings = []
        self.memory_usage = []
        
    def run_inference(self, model, inputs):
        """
        Run inference on inputs and measure time.
        
        Args:
            model: Model to benchmark
            inputs: Input data
            
        Returns:
            Inference outputs
        """
        if HAS_TORCH:
            torch.cuda.synchronize() if self.device == "cuda" else None
            
        start = time.time()
        outputs = model(inputs)
        
        if HAS_TORCH:
            torch.cuda.synchronize() if self.device == "cuda" else None
            
        end = time.time()
        self.timings.append((end - start) * 1000)  # Convert to ms
        
        # Measure memory if using CUDA
        if HAS_TORCH and self.device == "cuda":
            memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            self.memory_usage.append(memory)
            torch.cuda.reset_peak_memory_stats()
            
        return outputs
        
    def analyze_timings(self):
        """Analyze timing measurements."""
        timings = np.array(self.timings)
        
        # Calculate statistics
        mean = np.mean(timings)
        median = np.median(timings)
        p95 = np.percentile(timings, 95)
        p99 = np.percentile(timings, 99)
        min_t = np.min(timings)
        max_t = np.max(timings)
        
        # Record metrics
        self.record_metric("timings", timings)
        self.record_metric("mean_time_ms", mean)
        self.record_metric("median_time_ms", median)
        self.record_metric("p95_time_ms", p95)
        self.record_metric("p99_time_ms", p99)
        self.record_metric("min_time_ms", min_t)
        self.record_metric("max_time_ms", max_t)
        
        # Record memory usage if available
        if self.memory_usage:
            self.record_metric("memory_usage_mb", np.mean(self.memory_usage))
            
        # Calculate FPS
        fps = 1000 / mean
        self.record_metric("fps", fps)
        
        print(f"Latency statistics for {self.model_name}:")
        print(f"  Mean: {mean:.2f} ms")
        print(f"  P95: {p95:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        
        return {
            "mean": mean,
            "median": median,
            "p95": p95,
            "p99": p99,
            "min": min_t,
            "max": max_t,
            "fps": fps
        }
        
    def visualize(self, output_format: str = "png"):
        """
        Create visualization of latency benchmark results.
        
        Args:
            output_format: Output format (png, svg, pdf)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot timings
            timings = np.array(self.metrics.get("timings", []))
            if len(timings) > 0:
                # Plot histogram
                sns.histplot(timings, ax=ax1, kde=True)
                ax1.set_title(f"{self.model_name} Latency Distribution")
                ax1.set_xlabel("Time (ms)")
                ax1.set_ylabel("Count")
                
                # Add lines for mean and p95
                mean = self.metrics.get("mean_time_ms", 0)
                p95 = self.metrics.get("p95_time_ms", 0)
                
                ax1.axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f} ms')
                ax1.axvline(p95, color='g', linestyle='--', label=f'P95: {p95:.2f} ms')
                ax1.legend()
                
                # Plot timings over iterations
                ax2.plot(timings, marker='.', alpha=0.5)
                ax2.set_title(f"{self.model_name} Latency Over Time")
                ax2.set_xlabel("Iteration")
                ax2.set_ylabel("Time (ms)")
                ax2.set_ylim(bottom=0)
                
                # Add horizontal line for mean
                ax2.axhline(mean, color='r', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.output_dir / f"latency_{self.model_name}.{output_format}"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
            
            plt.close(fig)
            return str(output_path)
            
        except ImportError:
            print("matplotlib or seaborn not available for visualization")
            return None


class DriftRecoveryBenchmark(Benchmark):
    """Benchmark for measuring recovery from distribution drift."""
    
    def __init__(
        self,
        adaptation_method: str,
        drift_type: str = "lighting",
        num_frames: int = 500,
        **kwargs
    ):
        """
        Initialize drift recovery benchmark.
        
        Args:
            adaptation_method: Adaptation method to benchmark
            drift_type: Type of drift to simulate
            num_frames: Number of frames to process
        """
        super().__init__(name=f"drift_{drift_type}_{adaptation_method}", **kwargs)
        self.adaptation_method = adaptation_method
        self.drift_type = drift_type
        self.num_frames = num_frames
        
        self.auroc_scores = []
        self.recovery_frame = None
        
    def record_auroc(self, frame: int, auroc: float):
        """
        Record AUROC score for a frame.
        
        Args:
            frame: Frame index
            auroc: AUROC score
        """
        self.auroc_scores.append((frame, auroc))
        
    def detect_recovery(self, threshold: float = 0.9):
        """
        Detect recovery frame (when AUROC exceeds threshold).
        
        Args:
            threshold: AUROC threshold for recovery
            
        Returns:
            Frame index where recovery is detected
        """
        for frame, auroc in self.auroc_scores:
            if auroc >= threshold:
                self.recovery_frame = frame
                break
                
        return self.recovery_frame
        
    def analyze_results(self):
        """Analyze drift recovery results."""
        # Convert scores to numpy arrays
        frames, scores = zip(*self.auroc_scores) if self.auroc_scores else ([], [])
        frames = np.array(frames)
        scores = np.array(scores)
        
        # Detect recovery if not already done
        if self.recovery_frame is None:
            self.detect_recovery()
            
        # Extract key values
        initial_auroc = scores[0] if len(scores) > 0 else 0
        final_auroc = scores[-1] if len(scores) > 0 else 0
        auroc_at_64 = scores[min(64, len(scores)-1)] if len(scores) > 0 else 0
        
        # Record metrics
        self.record_metric("frames", frames)
        self.record_metric("auroc_scores", scores)
        self.record_metric("initial_auroc", initial_auroc)
        self.record_metric("final_auroc", final_auroc)
        self.record_metric("auroc_at_64", auroc_at_64)
        self.record_metric("recovery_frame", self.recovery_frame)
        
        # Calculate recovery stats
        if self.recovery_frame is not None:
            recovery_time = self.recovery_frame
            delta_auroc = auroc_at_64 - initial_auroc
            
            self.record_metric("recovery_time", recovery_time)
            self.record_metric("delta_auroc_at_64", delta_auroc)
            
            print(f"Drift recovery results for {self.adaptation_method}:")
            print(f"  Recovery detected at frame: {self.recovery_frame}")
            print(f"  ΔAUROC@64f: {delta_auroc:.4f}")
            print(f"  Final AUROC: {final_auroc:.4f}")
            
            return {
                "recovery_frame": self.recovery_frame,
                "delta_auroc_at_64": delta_auroc,
                "final_auroc": final_auroc
            }
        else:
            print(f"No recovery detected for {self.adaptation_method}")
            return None
            
    def visualize(self, output_format: str = "png"):
        """
        Create visualization of drift recovery benchmark results.
        
        Args:
            output_format: Output format (png, svg, pdf)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot AUROC scores
            frames = self.metrics.get("frames", [])
            scores = self.metrics.get("auroc_scores", [])
            
            if frames and scores:
                ax.plot(frames, scores, marker='.', alpha=0.7)
                ax.set_title(f"Drift Recovery: {self.adaptation_method} ({self.drift_type})")
                ax.set_xlabel("Frames Since Drift Onset")
                ax.set_ylabel("AUROC Score")
                ax.set_ylim(0.6, 1.0)
                
                # Add vertical line at recovery frame if detected
                recovery_frame = self.metrics.get("recovery_frame")
                if recovery_frame:
                    ax.axvline(recovery_frame, color='g', linestyle='--', 
                              label=f'Recovery: Frame {recovery_frame}')
                    
                # Add vertical line at frame 64
                ax.axvline(64, color='r', linestyle='--', label='Frame 64')
                
                # Add horizontal line at 0.9 AUROC
                ax.axhline(0.9, color='k', linestyle=':', alpha=0.5, label='AUROC = 0.9')
                
                # Annotate ΔAUROC@64f
                delta_auroc = self.metrics.get("delta_auroc_at_64", 0)
                auroc_at_64 = next((s for f, s in zip(frames, scores) if f >= 64), None)
                if auroc_at_64:
                    ax.annotate(f'ΔAUROC@64f: +{delta_auroc:.4f}',
                               xy=(64, auroc_at_64),
                               xytext=(100, auroc_at_64 - 0.05),
                               arrowprops=dict(facecolor='black', shrink=0.05, width=1))
                
                ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            output_path = self.output_dir / f"drift_recovery_{self.adaptation_method}.{output_format}"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
            
            plt.close(fig)
            return str(output_path)
            
        except ImportError:
            print("matplotlib or seaborn not available for visualization")
            return None


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="Run benchmarks for Adaptive Vision-Based Anomaly Detection")
    parser.add_argument("--benchmark", type=str, choices=["latency", "drift", "all"], default="all",
                       help="Benchmark to run")
    parser.add_argument("--model", type=str, default="transfusion_lite",
                       help="Model to benchmark")
    parser.add_argument("--adaptation", type=str, default="inreach_fo",
                       help="Adaptation method to benchmark")
    parser.add_argument("--drift-type", type=str, default="lighting",
                       help="Type of drift to simulate")
    parser.add_argument("--frames", type=int, default=500,
                       help="Number of frames to process")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size to use")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda", "mps"],
                       help="Device to run benchmarks on")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations of results")
    parser.add_argument("--output-format", type=str, default="png",
                       choices=["png", "svg", "pdf"],
                       help="Output format for visualizations")
    
    args = parser.parse_args()
    
    # Run latency benchmark
    if args.benchmark in ["latency", "all"]:
        print(f"Running latency benchmark for {args.model}")
        latency_bench = LatencyBenchmark(
            model_name=args.model,
            batch_size=args.batch_size,
            num_frames=args.frames,
            device=args.device,
            output_dir=args.output_dir
        )
        
        latency_bench.start()
        # TODO: Implement actual model loading and inference
        time.sleep(1)  # Placeholder for now
        latency_bench.end()
        
        latency_bench.record_metric("test_metric", 123.45)
        results_path = latency_bench.save_results()
        
        if args.visualize:
            latency_bench.visualize(output_format=args.output_format)
            
    # Run drift recovery benchmark
    if args.benchmark in ["drift", "all"]:
        print(f"Running drift recovery benchmark for {args.adaptation}")
        drift_bench = DriftRecoveryBenchmark(
            adaptation_method=args.adaptation,
            drift_type=args.drift_type,
            num_frames=args.frames,
            device=args.device,
            output_dir=args.output_dir
        )
        
        drift_bench.start()
        # TODO: Implement actual drift simulation and recovery
        time.sleep(1)  # Placeholder for now
        
        # Sample simulated recovery data
        np.random.seed(42)
        for i in range(args.frames):
            # Simulated AUROC that improves over time
            if args.adaptation == "inreach_fo":
                auroc = 0.75 + 0.2 * (1 - np.exp(-i / 64.0)) + 0.02 * np.random.randn()
            else:  # Baseline
                auroc = 0.75 + 0.2 * (1 - np.exp(-i / 250.0)) + 0.02 * np.random.randn()
                
            auroc = np.clip(auroc, 0, 1.0)
            drift_bench.record_auroc(i, auroc)
            
        drift_bench.end()
        
        drift_bench.analyze_results()
        results_path = drift_bench.save_results()
        
        if args.visualize:
            drift_bench.visualize(output_format=args.output_format)
    
    print("Benchmarks completed successfully")

if __name__ == "__main__":
    main()