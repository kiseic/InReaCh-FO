import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import time
import logging
import os

logger = logging.getLogger(__name__)

def find_bn_layers(model: nn.Module) -> List[nn.Module]:
    """
    Find all BatchNorm layers in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of BatchNorm layers
    """
    bn_layers = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_layers.append(module)
    return bn_layers

def get_bn_statistics(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Get BatchNorm statistics for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary mapping layer names to statistics
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            stats[name] = {
                "running_mean": module.running_mean.clone().detach().cpu(),
                "running_var": module.running_var.clone().detach().cpu(),
                "weight": module.weight.clone().detach().cpu() if module.weight is not None else None,
                "bias": module.bias.clone().detach().cpu() if module.bias is not None else None
            }
    return stats

def plot_bn_statistics(
    stats: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot BatchNorm statistics.
    
    Args:
        stats: BatchNorm statistics dictionary from get_bn_statistics
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("BatchNorm Layer Statistics", fontsize=16)
    
    # Flatten stats for plotting
    layer_names = list(stats.keys())
    mean_values = [stats[name]["running_mean"].mean().item() for name in layer_names]
    var_values = [stats[name]["running_var"].mean().item() for name in layer_names]
    
    # Sort by layer names
    indices = np.argsort(layer_names)
    layer_names = [layer_names[i] for i in indices]
    mean_values = [mean_values[i] for i in indices]
    var_values = [var_values[i] for i in indices]
    
    # Plot mean and variance
    axes[0, 0].bar(layer_names, mean_values)
    axes[0, 0].set_title("Mean Values")
    axes[0, 0].set_xticklabels([])
    
    axes[0, 1].bar(layer_names, var_values)
    axes[0, 1].set_title("Variance Values")
    axes[0, 1].set_xticklabels([])
    
    # Get distribution of first few layers
    if layer_names:
        sample_layers = layer_names[:min(5, len(layer_names))]
        
        for i, layer in enumerate(sample_layers):
            mean_dist = stats[layer]["running_mean"].numpy()
            var_dist = stats[layer]["running_var"].numpy()
            
            axes[1, 0].hist(mean_dist, bins=20, alpha=0.5, label=layer)
            axes[1, 1].hist(var_dist, bins=20, alpha=0.5, label=layer)
    
    axes[1, 0].set_title("Mean Distribution (Sample Layers)")
    axes[1, 0].legend()
    
    axes[1, 1].set_title("Variance Distribution (Sample Layers)")
    axes[1, 1].legend()
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

def compare_bn_statistics(
    stats1: Dict[str, Dict[str, torch.Tensor]],
    stats2: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Compare BatchNorm statistics from two models.
    
    Args:
        stats1: First model's BatchNorm statistics
        stats2: Second model's BatchNorm statistics
        save_path: Optional path to save the plot
        show: Whether to display the plot
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("BatchNorm Statistics Comparison", fontsize=16)
    
    # Find common layers
    common_layers = [layer for layer in stats1.keys() if layer in stats2]
    
    if not common_layers:
        logger.warning("No common BatchNorm layers found")
        return
    
    # Calculate differences
    mean_diffs = [
        torch.mean(torch.abs(stats1[layer]["running_mean"] - stats2[layer]["running_mean"])).item()
        for layer in common_layers
    ]
    
    var_diffs = [
        torch.mean(torch.abs(stats1[layer]["running_var"] - stats2[layer]["running_var"])).item()
        for layer in common_layers
    ]
    
    # Sort by layer names
    indices = np.argsort(common_layers)
    common_layers = [common_layers[i] for i in indices]
    mean_diffs = [mean_diffs[i] for i in indices]
    var_diffs = [var_diffs[i] for i in indices]
    
    # Plot differences
    axes[0, 0].bar(common_layers, mean_diffs)
    axes[0, 0].set_title("Mean Absolute Differences")
    axes[0, 0].set_xticklabels([])
    
    axes[0, 1].bar(common_layers, var_diffs)
    axes[0, 1].set_title("Variance Absolute Differences")
    axes[0, 1].set_xticklabels([])
    
    # Plot correlation
    if common_layers:
        all_means1 = torch.cat([stats1[layer]["running_mean"] for layer in common_layers])
        all_means2 = torch.cat([stats2[layer]["running_mean"] for layer in common_layers])
        
        all_vars1 = torch.cat([stats1[layer]["running_var"] for layer in common_layers])
        all_vars2 = torch.cat([stats2[layer]["running_var"] for layer in common_layers])
        
        axes[1, 0].scatter(all_means1.numpy(), all_means2.numpy(), alpha=0.5)
        axes[1, 0].set_title("Mean Correlation")
        axes[1, 0].set_xlabel("Model 1 Means")
        axes[1, 0].set_ylabel("Model 2 Means")
        
        axes[1, 1].scatter(all_vars1.numpy(), all_vars2.numpy(), alpha=0.5)
        axes[1, 1].set_title("Variance Correlation")
        axes[1, 1].set_xlabel("Model 1 Variances")
        axes[1, 1].set_ylabel("Model 2 Variances")
        
        # Add x=y line
        min_mean = min(all_means1.min().item(), all_means2.min().item())
        max_mean = max(all_means1.max().item(), all_means2.max().item())
        axes[1, 0].plot([min_mean, max_mean], [min_mean, max_mean], 'r--')
        
        min_var = min(all_vars1.min().item(), all_vars2.min().item())
        max_var = max(all_vars1.max().item(), all_vars2.max().item())
        axes[1, 1].plot([min_var, max_var], [min_var, max_var], 'r--')
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path)
        
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

def save_adaptation_metrics(
    metrics: Dict[str, Any],
    save_path: str,
    save_plots: bool = True
):
    """
    Save adaptation metrics to file.
    
    Args:
        metrics: Metrics dictionary from InReachFO.get_metrics()
        save_path: Path to save metrics (without extension)
        save_plots: Whether to generate and save plots
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save JSON metrics
    import json
    
    # Convert numpy and tensor values to Python types
    clean_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            clean_metrics[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    clean_metrics[key][k] = v.tolist() if hasattr(v, 'tolist') else float(v)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.ndarray, torch.Tensor)):
                    clean_metrics[key][k] = [item.tolist() if hasattr(item, 'tolist') else float(item) for item in v]
                else:
                    clean_metrics[key][k] = v
        elif isinstance(value, (np.ndarray, torch.Tensor)):
            clean_metrics[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
        else:
            clean_metrics[key] = value
    
    # Save metrics as JSON
    with open(f"{save_path}.json", 'w') as f:
        json.dump(clean_metrics, f, indent=2)
    
    # Create and save plots if requested
    if save_plots and "confidence" in metrics and "trend" in metrics["confidence"]:
        # Create confidence trend plot
        plt.figure(figsize=(10, 6))
        plt.plot(metrics["confidence"]["trend"])
        plt.title("Confidence Trend")
        plt.xlabel("Update")
        plt.ylabel("Confidence")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}_confidence.png")
        plt.close()
        
    if save_plots and "update_times" in metrics:
        # Create update time histogram
        plt.figure(figsize=(10, 6))
        plt.hist(metrics["update_times"].get("all", []), bins=20)
        plt.title("Update Time Distribution")
        plt.xlabel("Time (ms)")
        plt.ylabel("Count")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}_update_times.png")
        plt.close()

def measure_adaptation_speed(
    model: nn.Module,
    adapter,
    test_data: torch.Tensor,
    metric_fn,
    n_frames: int = 100,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Measure adaptation speed on a test dataset.
    
    Args:
        model: Base model (without adaptation)
        adapter: InReachFO adapter
        test_data: Test data tensor
        metric_fn: Function to calculate metrics
        n_frames: Number of frames to process
        device: Device to use
        
    Returns:
        Dictionary with adaptation metrics
    """
    # Ensure model and adapter are in eval mode
    model.eval()
    if hasattr(adapter, 'eval'):
        adapter.eval()
        
    # Move model and adapter to device
    model.to(device)
    if hasattr(adapter, 'to'):
        adapter.to(device)
        
    # Prepare data
    if test_data.device != device:
        test_data = test_data.to(device)
        
    # Initialize metrics
    base_metrics = []
    adapted_metrics = []
    
    # Process frames
    with torch.no_grad():
        for i in range(min(n_frames, len(test_data))):
            # Get input
            inputs = test_data[i:i+1]
            
            # Forward pass with base model
            base_outputs = model(inputs)
            base_metric = metric_fn(base_outputs)
            base_metrics.append(base_metric)
            
            # Forward pass with adapted model
            adapted_outputs = adapter(inputs)
            adapted_metric = metric_fn(adapted_outputs)
            adapted_metrics.append(adapted_metric)
            
    # Calculate improvement over time
    improvements = [adapted - base for base, adapted in zip(base_metrics, adapted_metrics)]
    
    # Find convergence point (when improvement stabilizes)
    window_size = 5
    convergence_point = n_frames
    
    for i in range(window_size, len(improvements) - window_size):
        window = improvements[i-window_size:i+window_size]
        std = np.std(window)
        
        if std < 0.01:  # Low standard deviation indicates stabilization
            convergence_point = i
            break
            
    # Create results
    results = {
        "base_metrics": base_metrics,
        "adapted_metrics": adapted_metrics,
        "improvements": improvements,
        "convergence_point": convergence_point,
        "mean_improvement": np.mean(improvements[-10:]),  # Mean of last 10 frames
        "max_improvement": max(improvements),
        "adaptation_frames": adapter.frame_counter
    }
    
    return results