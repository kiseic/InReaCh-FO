#!/usr/bin/env python3
"""
Visualization utilities for Adaptive Vision-Based Anomaly Detection results.

This module provides functions for creating comprehensive visualizations
of experimental results, including latency benchmarks, drift recovery,
and model performance comparisons.
"""

import os
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    print("Warning: Visualization libraries not available.")

# Define constants
RESULTS_DIR = Path("results")
OUTPUT_DIR = Path("visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure visualization style
if HAS_VIZ:
    plt.style.use('seaborn-v0_8')
    sns.set_context("paper", font_scale=1.2)
    
    # Custom color palettes
    MODEL_COLORS = {
        "transfusion": "#3498db",  # Blue
        "inreach": "#2ecc71",      # Green
        "samlad": "#e74c3c",       # Red
        "laft": "#9b59b6"          # Purple
    }
    
    # Custom colormap for anomaly visualization
    anomaly_cmap = LinearSegmentedColormap.from_list(
        "anomaly_cmap", ["#f1c40f", "#e74c3c", "#8e44ad"]
    )


def load_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment results from a JSON file.
    
    Args:
        results_path: Path to the results JSON file
        
    Returns:
        Dictionary containing the experiment results
    """
    results_path = Path(results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
        
    with open(results_path, 'r') as f:
        results = json.load(f)
        
    return results


def visualize_latency(
    results: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> Optional[str]:
    """
    Create latency comparison visualization.
    
    Args:
        results: Dictionary containing latency benchmark results
        output_path: Path to save the visualization
        show: Whether to display the plot
        
    Returns:
        Path to the saved visualization or None
    """
    if not HAS_VIZ:
        print("Visualization libraries not available")
        return None
        
    # Extract latency results
    # Check for nested results structure
    if "results" in results and "latency" in results["results"]:
        latency_results = results["results"]["latency"]
    elif "latency" in results:
        latency_results = results["latency"]
    else:
        print("No latency results found")
        return None
    
    if not latency_results:
        print("Empty latency results")
        return None
        
    # Create figure with multiple plots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Latency bar chart - Grid position 1
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    
    models = []
    mean_times = []
    p95_times = []
    fps_values = []
    
    for model_name, metrics in latency_results.items():
        models.append(model_name)
        mean_times.append(metrics.get("mean_time_ms", 0))
        p95_times.append(metrics.get("p95_time_ms", 0))
        fps_values.append(metrics.get("fps", 0))
        
    # Plot bar chart
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, mean_times, width, label='Mean Latency (ms)', 
                   color=[MODEL_COLORS.get(m, "#3498db") for m in models])
    bars2 = ax1.bar(x + width/2, p95_times, width, label='P95 Latency (ms)',
                   color=[MODEL_COLORS.get(m, "#3498db") for m in models], alpha=0.7)
    
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Model Latency Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    
    # Add FPS as text
    for i, v in enumerate(fps_values):
        ax1.text(i, mean_times[i] + 5, f"{v:.1f} FPS", 
               ha='center', va='bottom', fontsize=9, color='green')
        
    # Add 50ms latency threshold line
    ax1.axhline(50, color='red', linestyle='--', alpha=0.7, 
                label='Real-time threshold (50ms)')
    ax1.legend()
    
    # 2. FPS comparison - Grid position 2
    ax2 = plt.subplot2grid((2, 3), (0, 2))
    
    # Plot horizontal bar chart for FPS
    ax2.barh(models, fps_values, color=[MODEL_COLORS.get(m, "#3498db") for m in models])
    ax2.set_xlabel('Frames Per Second')
    ax2.set_title('Model FPS Comparison')
    
    # Add 20 FPS threshold line for real-time operation
    ax2.axvline(20, color='red', linestyle='--', alpha=0.7,
                label='Real-time threshold (20 FPS)')
    
    # Add FPS values as text
    for i, v in enumerate(fps_values):
        ax2.text(v + 1, i, f"{v:.1f}", va='center')
        
    # 3. Memory usage - Grid position 3 (if available)
    ax3 = plt.subplot2grid((2, 3), (1, 0))
    
    memory_values = []
    for model_name, metrics in latency_results.items():
        memory = metrics.get("memory_usage_mb", 0)
        memory_values.append(memory)
        
    # Plot bar chart for memory usage
    ax3.bar(models, memory_values, color=[MODEL_COLORS.get(m, "#3498db") for m in models])
    ax3.set_ylabel('Memory Usage (MB)')
    ax3.set_title('Model Memory Comparison')
    ax3.set_xticklabels(models, rotation=45, ha='right')
    
    # 4. Timing distribution - Grid position 4
    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
    
    # Collect all timing distributions
    timing_data = []
    for model_name, metrics in latency_results.items():
        timings = metrics.get("timings", [])
        if timings:
            for t in timings:
                timing_data.append((model_name, t))
                
    if timing_data:
        # Convert to DataFrame for seaborn
        import pandas as pd
        df = pd.DataFrame(timing_data, columns=["Model", "Latency (ms)"])
        
        # Plot violin plot
        sns.violinplot(x="Model", y="Latency (ms)", data=df, ax=ax4,
                      palette=MODEL_COLORS)
        ax4.set_title('Latency Distribution')
        ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, "No timing distribution data available",
                ha='center', va='center')
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = OUTPUT_DIR / "latency_comparison.png"
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return str(output_path)


def visualize_drift_recovery(
    results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> Optional[str]:
    """
    Create drift recovery visualization.
    
    Args:
        results: Dictionary containing drift recovery benchmark results
        output_path: Path to save the visualization
        show: Whether to display the plot
        
    Returns:
        Path to the saved visualization or None
    """
    if not HAS_VIZ:
        print("Visualization libraries not available")
        return None
        
    # Extract drift recovery results
    # Check for nested results structure
    if "results" in results and "drift_recovery" in results["results"]:
        drift_results = results["results"]["drift_recovery"]
    elif "drift_recovery" in results:
        drift_results = results["drift_recovery"]
    else:
        print("No drift recovery results found")
        return None
    
    if not drift_results:
        print("Empty drift recovery results")
        return None
        
    # Create figure with multiple plots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. AUROC curve - Grid position 1
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    frames = drift_results.get("frames", [])
    scores = drift_results.get("auroc_scores", [])
    recovery_frame = drift_results.get("recovery_frame")
    
    if frames and scores:
        ax1.plot(frames, scores, marker='.', alpha=0.7, color=MODEL_COLORS.get("inreach", "green"))
        ax1.set_title("Drift Recovery Performance")
        ax1.set_xlabel("Frames Since Drift Onset")
        ax1.set_ylabel("AUROC Score")
        ax1.set_ylim(0.5, 1.0)
        
        # Add vertical line at recovery frame if detected
        if recovery_frame:
            ax1.axvline(recovery_frame, color='g', linestyle='--', 
                      label=f'Recovery: Frame {recovery_frame}')
            
        # Add vertical line at frame 64
        ax1.axvline(64, color='r', linestyle='--', label='Frame 64')
        
        # Add horizontal line at 0.9 AUROC
        ax1.axhline(0.9, color='k', linestyle=':', alpha=0.5, label='AUROC = 0.9')
        
        # Annotate ΔAUROC@64f if available
        delta_auroc = drift_results.get("delta_auroc_at_64", 0)
        auroc_at_64 = next((s for f, s in zip(frames, scores) if f >= 64), None)
        if auroc_at_64:
            ax1.annotate(f'ΔAUROC@64f: +{delta_auroc:.4f}',
                       xy=(64, auroc_at_64),
                       xytext=(100, auroc_at_64 - 0.05),
                       arrowprops=dict(facecolor='black', shrink=0.05, width=1))
        
        ax1.legend()
        
    else:
        ax1.text(0.5, 0.5, "No AUROC data available",
                ha='center', va='center')
    
    # 2. Recovery time comparison - Grid position 3
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    
    # Add a simple bar chart showing recovery frame
    if recovery_frame is not None:
        ax2.bar(["InReaCh-FO"], [recovery_frame], color=MODEL_COLORS.get("inreach", "green"))
        ax2.set_ylabel("Recovery Time (frames)")
        ax2.set_title("Adaptation Recovery Speed")
        ax2.text(0, recovery_frame + 5, f"{recovery_frame} frames", 
                ha='center', va='bottom')
    else:
        ax2.text(0.5, 0.5, "No recovery frame data available",
                ha='center', va='center')
    
    # 3. AUROC improvement visualization - Grid position 4
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    
    # Key metrics
    metrics = [
        ("Initial AUROC", drift_results.get("initial_auroc", 0)),
        ("AUROC@64f", drift_results.get("auroc_at_64", 0)),
        ("Final AUROC", drift_results.get("final_auroc", 0))
    ]
    
    labels, values = zip(*metrics)
    
    ax3.bar(labels, values, color=["#3498db", "#f39c12", "#2ecc71"])
    ax3.set_ylim(0.5, 1.0)
    ax3.set_title("AUROC Improvement")
    
    # Add values as text
    for i, v in enumerate(values):
        ax3.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = OUTPUT_DIR / "drift_recovery.png"
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return str(output_path)


def visualize_combined_results(
    results: Dict[str, Any],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False
) -> Optional[str]:
    """
    Create a comprehensive visualization of all experiment results.
    
    Args:
        results: Dictionary containing all experiment results
        output_path: Path to save the visualization
        show: Whether to display the plot
        
    Returns:
        Path to the saved visualization or None
    """
    if not HAS_VIZ:
        print("Visualization libraries not available")
        return None
        
    # Check if we have results to visualize
    has_latency = ("results" in results and "latency" in results["results"]) or ("latency" in results)
    has_drift = ("results" in results and "drift_recovery" in results["results"]) or ("drift_recovery" in results)
    
    if not (has_latency or has_drift):
        print("No results to visualize")
        return None
        
    # Create figure with multiple plots
    fig = plt.figure(figsize=(20, 12))
    
    # Set up grid layout
    if has_latency and has_drift:
        # 1. Main title and experiment info
        ax_title = plt.subplot2grid((4, 4), (0, 0), colspan=4)
        ax_title.axis("off")
        
        # Add title and info
        experiment_name = results.get("experiment_name", "Adaptive Vision-Based Anomaly Detection")
        timestamp = results.get("timestamp", "")
        device = results.get("device", "cpu")
        
        title_text = f"{experiment_name}\n"
        info_text = f"Device: {device} | Date: {timestamp}"
        
        ax_title.text(0.5, 0.5, title_text, ha='center', va='center', 
                     fontsize=16, fontweight='bold')
        ax_title.text(0.5, 0.2, info_text, ha='center', va='center', 
                     fontsize=12, color='gray')
        
        # 2. Latency comparison - Top row
        ax_latency = plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=1)
        
        latency_results = results.get("latency", {})
        models = []
        mean_times = []
        fps_values = []
        
        for model_name, metrics in latency_results.items():
            models.append(model_name)
            mean_times.append(metrics.get("mean_time_ms", 0))
            fps_values.append(metrics.get("fps", 0))
            
        x = np.arange(len(models))
        ax_latency.bar(x, mean_times, color=[MODEL_COLORS.get(m, "#3498db") for m in models])
        ax_latency.set_xticks(x)
        ax_latency.set_xticklabels(models, rotation=45, ha='right')
        ax_latency.set_ylabel("Latency (ms)")
        ax_latency.set_title("Model Latency Comparison")
        
        # Add FPS as text
        for i, v in enumerate(fps_values):
            ax_latency.text(i, mean_times[i] + 2, f"{v:.1f} FPS", 
                          ha='center', va='bottom', fontsize=9, color='green')
        
        # 3. FPS comparison - Top right
        ax_fps = plt.subplot2grid((4, 4), (1, 2), colspan=2, rowspan=1)
        ax_fps.barh(models, fps_values, color=[MODEL_COLORS.get(m, "#3498db") for m in models])
        ax_fps.set_xlabel("Frames Per Second")
        ax_fps.set_title("Model FPS Comparison")
        
        # Add 20 FPS threshold line
        ax_fps.axvline(20, color='red', linestyle='--', alpha=0.7)
        
        # 4. Drift recovery curve - Bottom row, left
        ax_drift = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)
        
        drift_results = results.get("drift_recovery", {})
        frames = drift_results.get("frames", [])
        scores = drift_results.get("auroc_scores", [])
        recovery_frame = drift_results.get("recovery_frame")
        
        if frames and scores:
            ax_drift.plot(frames, scores, marker='.', alpha=0.7, 
                         color=MODEL_COLORS.get("inreach", "green"))
            ax_drift.set_title("Drift Recovery Performance")
            ax_drift.set_xlabel("Frames Since Drift Onset")
            ax_drift.set_ylabel("AUROC Score")
            ax_drift.set_ylim(0.5, 1.0)
            
            # Add vertical line at recovery frame if detected
            if recovery_frame:
                ax_drift.axvline(recovery_frame, color='g', linestyle='--', 
                              label=f'Recovery: Frame {recovery_frame}')
                
            # Add vertical line at frame 64
            ax_drift.axvline(64, color='r', linestyle='--', label='Frame 64')
            
            # Add horizontal line at 0.9 AUROC
            ax_drift.axhline(0.9, color='k', linestyle=':', alpha=0.5, label='AUROC = 0.9')
            ax_drift.legend()
            
        # 5. AUROC improvement visualization - Bottom right
        ax_auroc = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=1)
        
        # Key metrics
        metrics = [
            ("Initial AUROC", drift_results.get("initial_auroc", 0)),
            ("AUROC@64f", drift_results.get("auroc_at_64", 0)),
            ("Final AUROC", drift_results.get("final_auroc", 0))
        ]
        
        labels, values = zip(*metrics)
        
        ax_auroc.bar(labels, values, color=["#3498db", "#f39c12", "#2ecc71"])
        ax_auroc.set_ylim(0.5, 1.0)
        ax_auroc.set_title("AUROC Improvement")
        
        # Add values as text
        for i, v in enumerate(values):
            ax_auroc.text(i, v + 0.02, f"{v:.3f}", ha='center')
            
        # 6. System diagram - Bottom right
        ax_diagram = plt.subplot2grid((4, 4), (3, 2), colspan=2, rowspan=1)
        ax_diagram.axis("off")
        
        # Simple system diagram
        components = [
            "Camera Input",
            "StreamingAdapter",
            "TransFusion-Lite",
            "InReaCh-FO",
            "SAM-LAD",
            "LAFT + Phi-4-mini"
        ]
        
        # Create a flow diagram
        component_positions = np.linspace(0.1, 0.9, len(components))
        for i, comp in enumerate(components):
            # Draw component box
            ax_diagram.text(component_positions[i], 0.5, comp, 
                           ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", 
                                    fc=plt.cm.tab10(i % 10), 
                                    ec="black", alpha=0.7))
            
            # Draw arrow to next component
            if i < len(components) - 1:
                ax_diagram.annotate("", 
                                  xy=(component_positions[i+1] - 0.05, 0.5),
                                  xytext=(component_positions[i] + 0.05, 0.5),
                                  arrowprops=dict(arrowstyle="->", color="black"))
        
        ax_diagram.set_title("System Components")
        
    elif has_latency:
        # Only latency results available
        visualize_latency(results, output_path=output_path, show=show)
        return output_path
        
    elif has_drift:
        # Only drift recovery results available
        visualize_drift_recovery(results, output_path=output_path, show=show)
        return output_path
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = OUTPUT_DIR / "experiment_results.png"
    else:
        output_path = Path(output_path)
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
        
    return str(output_path)


def main():
    """Main entry point for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize experiment results for Adaptive Vision-Based Anomaly Detection"
    )
    parser.add_argument("--results", type=str, required=True,
                       help="Path to results JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for visualizations")
    parser.add_argument("--type", type=str, default="all",
                       choices=["latency", "drift", "all"],
                       help="Type of visualization to generate")
    parser.add_argument("--show", action="store_true",
                       help="Show visualizations interactively")
    
    args = parser.parse_args()
    
    try:
        # Load results
        results = load_results(args.results)
        
        # Determine output path
        output_dir = Path(args.output) if args.output else OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        if args.type == "latency" or args.type == "all":
            latency_path = visualize_latency(
                results, 
                output_path=output_dir / "latency_comparison.png",
                show=args.show
            )
            if latency_path:
                print(f"Latency visualization saved to: {latency_path}")
                
        if args.type == "drift" or args.type == "all":
            drift_path = visualize_drift_recovery(
                results,
                output_path=output_dir / "drift_recovery.png",
                show=args.show
            )
            if drift_path:
                print(f"Drift recovery visualization saved to: {drift_path}")
                
        if args.type == "all":
            combined_path = visualize_combined_results(
                results,
                output_path=output_dir / "experiment_results.png",
                show=args.show
            )
            if combined_path:
                print(f"Combined visualization saved to: {combined_path}")
                
    except Exception as e:
        print(f"Error during visualization: {e}")
    

if __name__ == "__main__":
    main()