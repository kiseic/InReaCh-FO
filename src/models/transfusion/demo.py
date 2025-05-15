#!/usr/bin/env python3
"""
TransFusion-Lite Demo for Adaptive Vision-Based Anomaly Detection.

This script demonstrates the TransFusion-Lite model for anomaly detection
using a camera or video file input and visualizing real-time results.
"""

import os
import sys
import argparse
import torch
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.transfusion.utils import get_available_device, load_model_checkpoint, create_model_card
from src.streaming.adapter import StreamingAdapter

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('transfusion_demo.log')
    ]
)
logger = logging.getLogger("transfusion_demo")

def create_debug_overlay(
    frame: np.ndarray,
    stats: Dict[str, Any],
    score: float,
    threshold: float = 0.5,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Create debug overlay with statistics.
    
    Args:
        frame: Input frame
        stats: Dictionary with statistics
        score: Anomaly score
        threshold: Anomaly threshold
        position: Text position
        font_scale: Font scale
        color: Text color
        thickness: Text thickness
        
    Returns:
        Frame with debug overlay
    """
    # Make a copy of the frame
    display = frame.copy()
    
    # Add header background
    header_height = 40
    cv2.rectangle(display, (0, 0), (frame.shape[1], header_height), (0, 0, 0), -1)
    
    # Add FPS
    fps = stats.get("fps", 0)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(display, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Add score
    score_text = f"Score: {score:.3f}"
    score_pos = (position[0] + 150, position[1])
    # Color based on threshold
    score_color = (0, 255, 0) if score < threshold else (0, 0, 255)
    cv2.putText(display, score_text, score_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, score_color, thickness)
    
    # Add process time
    process_time = stats.get("process_time", 0)
    time_text = f"Process: {process_time:.1f} ms"
    time_pos = (position[0] + 300, position[1])
    cv2.putText(display, time_text, time_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Add device info
    device = stats.get("device", "unknown")
    device_pos = (position[0] + 500, position[1])
    cv2.putText(display, device, device_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    # Add TransFusion-Lite label
    title = "TransFusion-Lite Anomaly Detection"
    title_pos = (frame.shape[1] - 400, position[1])
    cv2.putText(display, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 165, 0), thickness)
    
    return display

def display_gallery(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 100
) -> np.ndarray:
    """
    Display a gallery of images with titles.
    
    Args:
        images: List of images to display
        titles: List of titles for images
        figsize: Figure size
        dpi: DPI for rendering
        
    Returns:
        Rendered image as numpy array
    """
    # Create figure
    fig, axes = plt.subplots(1, len(images), figsize=figsize, dpi=dpi)
    
    # Handle single image case
    if len(images) == 1:
        axes = [axes]
    
    # Add images to figure
    for i, (img, title) in enumerate(zip(images, titles)):
        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3 and img.dtype == np.uint8:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close figure to avoid memory leak
    plt.close(fig)
    
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img

def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(description="TransFusion-Lite Demo")
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file path')
    parser.add_argument('--resolution', type=str, default='640x480', help='Input resolution (WxH)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold (0-1)')
    parser.add_argument('--model-path', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--backbone', type=str, default='vit_base_patch16_224', help='ViT backbone')
    parser.add_argument('--steps', type=int, default=4, help='Number of diffusion steps')
    parser.add_argument('--feature-size', type=int, default=16, help='Feature map size')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--output', type=str, default='', help='Path to save output video')
    parser.add_argument('--export-onnx', action='store_true', help='Export model to ONNX format')
    parser.add_argument('--visualize-features', action='store_true', help='Visualize feature maps')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
    parser.add_argument('--benchmark-frames', type=int, default=1000, help='Number of frames for benchmark')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Get device
    device = get_available_device()
    logger.info(f"Using device: {device}")
    
    # Initialize model
    logger.info(f"Initializing TransFusion-Lite with backbone: {args.backbone}")
    model = TransFusionLite(
        input_shape=(224, 224, 3),  # Standard input size
        backbone=args.backbone,
        pretrained=True,
        n_steps=args.steps,
        feature_size=args.feature_size,
        device=device
    )
    
    # Load checkpoint if provided
    if args.model_path:
        logger.info(f"Loading model checkpoint: {args.model_path}")
        try:
            model, checkpoint = load_model_checkpoint(model, args.model_path, device=device)
            logger.info(f"Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    
    # Export model to ONNX if requested
    if args.export_onnx:
        try:
            from src.models.transfusion.utils import export_onnx_model
            logger.info("Exporting model to ONNX format...")
            output_path = f"models/transfusion_lite_{args.backbone}_{args.steps}steps.onnx"
            export_onnx_model(
                model,
                input_shape=(1, 3, 224, 224),
                path=output_path,
                device=device
            )
            logger.info(f"Model exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
    
    # Create processor
    processor = TransFusionProcessor(
        model, 
        threshold=args.threshold,
        image_size=(224, 224),  # Standard input size
        device=device
    )
    
    # Parse source (camera index or video file)
    try:
        source = int(args.source)  # Try as camera index
    except ValueError:
        source = args.source  # Use as file path
    
    # Initialize streaming adapter
    try:
        logger.info(f"Initializing streaming adapter with source: {source}")
        adapter = StreamingAdapter(
            source=source,
            resolution=(width, height),
            target_size=(224, 224),  # ViT input size
            batch_size=args.batch_size
        )
    except Exception as e:
        logger.error(f"Failed to initialize streaming adapter: {e}")
        return
    
    # Create video writer if output path provided
    video_writer = None
    if args.output:
        logger.info(f"Creating video writer: {args.output}")
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.output, 
            fourcc, 
            30.0,  # FPS
            (width, height)
        )
    
    # Process frames if benchmark mode
    if args.benchmark:
        run_benchmark(
            model=model,
            processor=processor,
            adapter=adapter,
            batch_size=args.batch_size,
            num_frames=args.benchmark_frames,
            device=device
        )
        return
    
    # Main processing loop
    try:
        frame_count = 0
        process_times = []
        
        while True:
            # Get a batch of frames
            batch = adapter.get_micro_batch()
            if batch is None:
                logger.warning("Failed to get micro-batch")
                break
            
            # Process batch
            start_time = time.time()
            result = processor.process(batch)
            end_time = time.time()
            process_time = (end_time - start_time) * 1000  # ms
            process_times.append(process_time)
            
            # Get results
            anomaly_maps = result["anomaly_maps"]
            normalized_maps = result["normalized_maps"]
            binary_masks = result["binary_masks"]
            scores = result["scores"]
            
            # Get adapter statistics
            adapter_stats = adapter.get_stats()
            
            # Create visualizations for each frame in batch
            for i in range(batch.shape[0]):
                frame = (batch[i] * 255).astype(np.uint8)
                anomaly_map = normalized_maps[i].cpu().numpy()
                binary_mask = binary_masks[i].cpu().numpy()
                score = scores[i].item()
                
                # Create visualization
                vis_frame = processor.visualize(
                    frame, 
                    anomaly_map=anomaly_map, 
                    binary_mask=binary_mask,
                    alpha=0.5,
                    colormap="jet"
                )
                
                # Add stats overlay
                stats = {
                    "fps": adapter_stats["fps"],
                    "process_time": process_time / args.batch_size,
                    "device": device
                }
                
                # Create debug overlay
                debug_frame = create_debug_overlay(
                    vis_frame, 
                    stats, 
                    score,
                    threshold=args.threshold
                )
                
                # Create feature visualization if requested
                if args.visualize_features and i == 0:  # Only for first frame in batch
                    # Extract feature maps for visualization
                    features = result["features"][i].detach().cpu().numpy()
                    latent = result["latent"][i].detach().cpu().numpy()
                    
                    # Create feature visualization
                    feature_vis = visualize_feature_maps(features, latent)
                    
                    # Resize feature visualization to match frame size
                    feature_vis = cv2.resize(feature_vis, (width, height))
                    
                    # Display side by side
                    debug_frame = np.hstack((debug_frame, feature_vis))
                
                # Display frame if GUI enabled
                if not args.no_gui:
                    cv2.imshow("TransFusion-Lite Demo", debug_frame)
                    
                # Write frame to video if output path provided
                if video_writer is not None:
                    # Ensure frame has correct size
                    if debug_frame.shape[1] != width or debug_frame.shape[0] != height:
                        debug_frame = cv2.resize(debug_frame, (width, height))
                    video_writer.write(debug_frame)
                
                frame_count += 1
                
                # Print stats every 100 frames
                if frame_count % 100 == 0:
                    avg_time = sum(process_times[-100:]) / min(100, len(process_times))
                    p95_time = sorted(process_times[-100:])[-5]  # 95th percentile of last 100
                    effective_fps = 1000 / avg_time * args.batch_size
                    
                    logger.info(f"Frame {frame_count}: Avg={avg_time:.1f}ms, P95={p95_time:.1f}ms, FPS={effective_fps:.1f}")
            
            # Exit on 'q' press
            if not args.no_gui and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        # Release resources
        adapter.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print final stats
        if process_times:
            # Remove first 10% of measurements as warmup
            warmup = max(1, len(process_times) // 10)
            process_times = process_times[warmup:]
            
            avg_time = sum(process_times) / len(process_times)
            p95_time = sorted(process_times)[int(len(process_times) * 0.95)]
            effective_fps = 1000 / avg_time * args.batch_size
            
            logger.info("\nPerformance Summary:")
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Average process time: {avg_time:.2f} ms per batch")
            logger.info(f"P95 process time: {p95_time:.2f} ms per batch")
            logger.info(f"Effective FPS: {effective_fps:.2f}")
            
            # Get model metrics
            model_metrics = model.get_metrics()
            logger.info(f"Model metrics: {model_metrics}")
            
            # Create model card
            params = {
                "backbone": args.backbone,
                "feature_size": args.feature_size,
                "n_steps": args.steps
            }
            
            metrics = {
                "avg_process_time_ms": avg_time,
                "p95_process_time_ms": p95_time,
                "effective_fps": effective_fps,
                "model_metrics": model_metrics
            }
            
            try:
                if args.model_path:
                    card_path = create_model_card(
                        model_name="TransFusion-Lite",
                        model_desc="Lightweight diffusion-based anomaly detection model with 4-step distillation",
                        model_path=args.model_path,
                        metrics=metrics,
                        params=params
                    )
                    logger.info(f"Model card created at {card_path}")
            except Exception as e:
                logger.error(f"Failed to create model card: {e}")
        
        logger.info("Demo completed successfully")


def run_benchmark(
    model: TransFusionLite,
    processor: TransFusionProcessor,
    adapter: StreamingAdapter,
    batch_size: int = 5,
    num_frames: int = 1000,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run performance benchmark.
    
    Args:
        model: TransFusion-Lite model
        processor: TransFusion processor
        adapter: Streaming adapter
        batch_size: Batch size
        num_frames: Number of frames to process
        device: Device to run on
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running benchmark with {num_frames} frames on {device}...")
    
    # Reset metrics
    model.reset_metrics()
    processor.reset_metrics()
    
    # Prepare variables
    frame_count = 0
    batch_times = []
    step_times = []
    
    # Run warmup
    logger.info("Running warmup...")
    for _ in range(5):
        batch = adapter.get_micro_batch()
        if batch is None:
            logger.error("Failed to get micro-batch during warmup")
            return {}
        processor.process(batch)
    
    # Clear metrics after warmup
    model.reset_metrics()
    processor.reset_metrics()
    
    # Run benchmark
    logger.info("Starting benchmark measurement...")
    start_time = time.time()
    
    while frame_count < num_frames:
        # Get a batch of frames
        batch = adapter.get_micro_batch()
        if batch is None:
            break
            
        # Process batch
        batch_start = time.time()
        result = processor.process(batch)
        batch_end = time.time()
        
        batch_time = (batch_end - batch_start) * 1000  # ms
        batch_times.append(batch_time)
        
        # Increment frame count
        frame_count += batch.shape[0]
        
        # Print progress
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count}/{num_frames} frames")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    p95_batch_time = sorted(batch_times)[int(len(batch_times) * 0.95)]
    effective_fps = frame_count / total_time
    
    # Get model metrics
    model_metrics = model.get_metrics()
    processor_metrics = processor.get_metrics()
    
    # Combine all metrics
    benchmark_results = {
        "frames_processed": frame_count,
        "total_time_s": total_time,
        "avg_batch_time_ms": avg_batch_time,
        "p95_batch_time_ms": p95_batch_time,
        "effective_fps": effective_fps,
        "per_frame_time_ms": avg_batch_time / batch_size,
        "device": device,
        "batch_size": batch_size,
        "model_metrics": model_metrics,
        "processor_metrics": processor_metrics
    }
    
    # Print results
    logger.info("\nBenchmark Results:")
    logger.info(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    logger.info(f"Average batch time: {avg_batch_time:.2f} ms")
    logger.info(f"P95 batch time: {p95_batch_time:.2f} ms")
    logger.info(f"Effective FPS: {effective_fps:.2f}")
    logger.info(f"Per-frame processing time: {avg_batch_time / batch_size:.2f} ms")
    
    # Create JSON report
    report_path = f"benchmark_results_{device}_{batch_size}batch.json"
    with open(report_path, 'w') as f:
        import json
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark report saved to {report_path}")
    
    return benchmark_results


def visualize_feature_maps(
    features: np.ndarray, 
    latent: np.ndarray, 
    n_channels: int = 4
) -> np.ndarray:
    """
    Visualize feature maps from TransFusion-Lite.
    
    Args:
        features: Feature tensor from ViT
        latent: Latent tensor from diffusion process
        n_channels: Number of channels to visualize
        
    Returns:
        Visualization as numpy array
    """
    # Get feature dimensions
    if len(features.shape) < 3:
        # Linear features, reshape to 2D
        features_vis = np.zeros((224, 224, 3), dtype=np.uint8)
    else:
        # Create visualization of feature maps
        channels_to_show = min(n_channels, features.shape[0])
        
        # Select channels to visualize
        selected_channels = []
        for i in range(channels_to_show):
            # Select evenly spaced channels
            channel_idx = i * features.shape[0] // channels_to_show
            channel = features[channel_idx]
            
            # Normalize channel to [0, 1]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                channel = (channel - channel_min) / (channel_max - channel_min)
            
            # Convert to uint8
            channel = (channel * 255).astype(np.uint8)
            
            # Add to list
            selected_channels.append(channel)
            
        # Create visualization grid
        grid_size = int(np.ceil(np.sqrt(len(selected_channels))))
        grid_height = grid_size * 50
        grid_width = grid_size * 50
        features_vis = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Fill grid with feature maps
        for i, channel in enumerate(selected_channels):
            # Calculate position in grid
            row = i // grid_size
            col = i % grid_size
            
            # Resize channel
            resized = cv2.resize(channel, (50, 50))
            
            # Convert to color (jet colormap)
            colored = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
            
            # Add to grid
            features_vis[row*50:(row+1)*50, col*50:(col+1)*50] = colored
    
    # Create similar visualization for latent features
    latent_vis = np.zeros_like(features_vis)
    
    if len(latent.shape) >= 3:
        # Select channels to visualize
        channels_to_show = min(n_channels, latent.shape[0])
        
        selected_channels = []
        for i in range(channels_to_show):
            # Select evenly spaced channels
            channel_idx = i * latent.shape[0] // channels_to_show
            channel = latent[channel_idx]
            
            # Normalize channel to [0, 1]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                channel = (channel - channel_min) / (channel_max - channel_min)
            
            # Convert to uint8
            channel = (channel * 255).astype(np.uint8)
            
            # Add to list
            selected_channels.append(channel)
            
        # Create visualization grid
        grid_size = int(np.ceil(np.sqrt(len(selected_channels))))
        
        # Fill grid with feature maps
        for i, channel in enumerate(selected_channels):
            # Calculate position in grid
            row = i // grid_size
            col = i % grid_size
            
            # Resize channel
            resized = cv2.resize(channel, (50, 50))
            
            # Convert to color (plasma colormap)
            colored = cv2.applyColorMap(resized, cv2.COLORMAP_PLASMA)
            
            # Add to grid
            latent_vis[row*50:(row+1)*50, col*50:(col+1)*50] = colored
    
    # Create comparison visualization
    vis_width = features_vis.shape[1] * 2 + 10  # Add 10px gap
    vis_height = features_vis.shape[0]
    
    comparison = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
    
    # Add feature visualization
    comparison[:, :features_vis.shape[1]] = features_vis
    
    # Add latent visualization
    comparison[:, features_vis.shape[1] + 10:] = latent_vis
    
    # Add labels
    cv2.putText(comparison, "Features", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(comparison, "Latent", (features_vis.shape[1] + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return comparison


if __name__ == "__main__":
    main()