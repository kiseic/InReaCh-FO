import torch
import torch.nn as nn
import numpy as np
import os
import logging
import json
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

def get_available_device() -> str:
    """
    Get the best available device for model inference.
    
    Returns:
        Device string (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def save_model_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    path: str = "checkpoints/transfusion_lite",
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optional optimizer to save
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
        metadata: Additional metadata to save
        
    Returns:
        Path to saved checkpoint
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create checkpoint dict
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "loss": loss
    }
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    # Add metadata if provided
    if metadata is not None:
        checkpoint["metadata"] = metadata
    
    # Save checkpoint
    torch.save(checkpoint, path)
    logger.info(f"Model checkpoint saved to {path}")
    
    return path

def load_model_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load checkpoint into
        path: Path to checkpoint
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to
        
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    # Determine device if not provided
    if device is None:
        device = get_available_device()
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load optimizer state if provided
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    logger.info(f"Model checkpoint loaded from {path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model, checkpoint

def export_onnx_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    path: str,
    device: str = "cpu",
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        input_shape: Input shape for example input
        path: Path to save ONNX model
        device: Device to use for export
        dynamic_axes: Dynamic axes for ONNX export
        
    Returns:
        Path to saved ONNX model
    """
    import onnx
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create example input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Set default dynamic axes if not provided
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes
    )
    
    # Verify exported model
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    
    logger.info(f"Model exported to ONNX format at {path}")
    
    return path

def create_model_card(
    model_name: str,
    model_desc: str,
    model_path: str,
    metrics: Dict[str, Any],
    params: Dict[str, Any]
) -> str:
    """
    Create a model card for a saved model.
    
    Args:
        model_name: Name of the model
        model_desc: Description of the model
        model_path: Path to saved model
        metrics: Model performance metrics
        params: Model parameters
        
    Returns:
        Path to saved model card
    """
    # Create model card in markdown format
    model_card = f"""# {model_name}

## Description

{model_desc}

## Model Details

- **Model Type**: TransFusion-Lite v1.0-rc2
- **Framework**: PyTorch
- **Backbone**: {params.get('backbone', 'vit_base_patch16_224')}
- **Dimensions**: {params.get('feature_size', 16)}x{params.get('feature_size', 16)}
- **Diffusion Steps**: {params.get('n_steps', 4)}

## Performance Metrics

| Metric | Value |
| ------ | ----- |
"""
    
    # Add metrics to table
    for name, value in metrics.items():
        if isinstance(value, dict):
            for subname, subvalue in value.items():
                model_card += f"| {name}/{subname} | {subvalue} |\n"
        else:
            model_card += f"| {name} | {value} |\n"
    
    # Add usage information
    model_card += """
## Usage

```python
from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
import torch

# Initialize model
model = TransFusionLite(
    backbone="vit_base_patch16_224",
    pretrained=True,
    n_steps=4
)

# Load weights
checkpoint = torch.load("path/to/checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Create processor
processor = TransFusionProcessor(model, threshold=0.5)

# Process image
result = processor.process(image_tensor)
```
"""
    
    # Save model card
    card_path = os.path.join(os.path.dirname(model_path), "model_card.md")
    with open(card_path, "w") as f:
        f.write(model_card)
    
    logger.info(f"Model card saved to {card_path}")
    
    return card_path