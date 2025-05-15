import os
import sys
import argparse
import torch
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.transfusion.utils import get_available_device, load_model_checkpoint
from src.models.inreach.model import InReachFO, AdaptiveThreshold
from src.models.inreach.utils import (
    get_bn_statistics, 
    plot_bn_statistics, 
    save_adaptation_metrics
)
from src.streaming.adapter import StreamingAdapter
from src.streaming.utils import display_stream, create_debug_overlay

def simulate_drift(frame, drift_type="lighting", intensity=0.5):
    """
    Simulate different types of drift in an image.
    
    Args:
        frame: Input frame (numpy array)
        drift_type: Type of drift to simulate
        intensity: Drift intensity (0-1)
        
    Returns:
        Frame with simulated drift
    """
    # Make a copy of the frame
    drift_frame = frame.copy()
    
    if drift_type == "lighting":
        # Simulate lighting changes
        gamma = 1.0 + intensity  # Increase brightness
        drift_frame = np.power(drift_frame, 1/gamma)
    elif drift_type == "contrast":
        # Simulate contrast changes
        mean = np.mean(drift_frame, axis=(0, 1), keepdims=True)
        drift_frame = (1 - intensity) * mean + intensity * drift_frame
    elif drift_type == "noise":
        # Add noise
        noise = np.random.normal(0, intensity * 0.1, drift_frame.shape)
        drift_frame = np.clip(drift_frame + noise, 0, 1)
    elif drift_type == "blur":
        # Add blur
        kernel_size = int(1 + intensity * 10) * 2 + 1  # Odd kernel size
        drift_frame = cv2.GaussianBlur(drift_frame, (kernel_size, kernel_size), 0)
    
    return drift_frame

def main():
    parser = argparse.ArgumentParser(description="InReaCh-FO Adaptation Demo")
    parser.add_argument('--source', type=int, default=0, help='Camera index')
    parser.add_argument('--resolution', type=str, default='640x480', help='Input resolution (WxH)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size')
    parser.add_argument('--model-path', type=str, default='', help='Path to model checkpoint')
    parser.add_argument('--update-freq', type=int, default=8, help='Update frequency in frames')
    parser.add_argument('--alpha', type=float, default=0.9, help='EMA factor for BN updates')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--drift-mode', type=str, default='manual', help='Drift mode: none, manual, or auto')
    parser.add_argument('--drift-type', type=str, default='lighting', help='Drift type for auto mode')
    parser.add_argument('--output', type=str, default='', help='Path to save output video')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Get device
    device = get_available_device()
    print(f"Using device: {device}")
    
    # Initialize base model
    base_model = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    base_model.to(device)
    
    # Load checkpoint if provided
    if args.model_path:
        base_model, _ = load_model_checkpoint(base_model, args.model_path, device=device)
        
    # Create adapter with base model
    adapter = InReachFO(
        model=base_model,
        alpha=args.alpha,
        update_freq=args.update_freq,
        confidence_thresh=0.7,
        track_metrics=True,
        device=device
    )
    
    # Create processors for base and adapted models
    base_processor = TransFusionProcessor(base_model, threshold=2.0, device=device)
    adaptive_threshold = AdaptiveThreshold(initial_threshold=2.0, alpha=0.95)
    
    # Initialize streaming adapter
    stream = StreamingAdapter(
        source=args.source,
        resolution=(width, height),
        target_size=(224, 224),  # ViT input size
        batch_size=args.batch_size
    )
    
    # Create video writer if output path provided
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            args.output, 
            fourcc, 
            30.0, 
            (width * 2, height)  # Side-by-side comparison
        )
    
    # Performance tracking
    frame_count = 0
    process_times = []
    base_scores = []
    adapted_scores = []
    thresholds = []
    current_drift_intensity = 0.0
    drift_direction = 1
    
    # Create plot window for metrics
    if not args.no_gui:
        plt.figure(figsize=(10, 6))
        plt.ion()  # Enable interactive mode
    
    # Main processing loop
    try:
        while True:
            # Get a batch of frames
            batch = stream.get_micro_batch()
            if batch is None:
                print("Failed to get micro-batch")
                break
            
            # Apply drift if in auto mode
            if args.drift_mode == "auto":
                # Update drift intensity
                current_drift_intensity += 0.005 * drift_direction
                if current_drift_intensity >= 1.0 or current_drift_intensity <= 0.0:
                    drift_direction *= -1
                    current_drift_intensity = max(0.0, min(1.0, current_drift_intensity))
                
                # Apply drift to each frame
                for i in range(batch.shape[0]):
                    batch[i] = simulate_drift(batch[i], args.drift_type, current_drift_intensity)
            
            # Convert numpy batch to torch tensor
            batch_tensor = torch.from_numpy(batch).float()
            batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
            
            # Process batch with base model
            start_time = time.time()
            base_result = base_processor.process(batch_tensor, update_stats=False)
            base_time = time.time() - start_time
            
            # Process batch with adapted model
            start_time = time.time()
            adapted_outputs = adapter(batch_tensor)
            
            # Extract adapted outputs (format depends on model implementation)
            if isinstance(adapted_outputs, tuple):
                anomaly_maps = adapted_outputs[0]
                features = adapted_outputs[1]
                latents = adapted_outputs[2]
            else:
                anomaly_maps = adapted_outputs
                features = None
                latents = None
                
            # Update adaptive threshold
            threshold = adaptive_threshold.update(anomaly_maps.detach().cpu())
            adapted_time = time.time() - start_time
            
            # Calculate total process time
            process_time = (base_time + adapted_time) * 1000  # ms
            process_times.append(process_time)
            
            # Get normalized anomaly maps
            base_anomaly_maps = base_result["normalized_maps"]
            
            # Process first frame for visualization
            frame_idx = 0
            base_frame = (batch[frame_idx] * 255).astype(np.uint8)
            
            # Get anomaly maps and scores
            base_map = base_anomaly_maps[frame_idx].cpu().numpy()
            base_score = torch.mean(base_anomaly_maps[frame_idx]).item()
            
            adapted_map = anomaly_maps[frame_idx].cpu().numpy()
            adapted_map = adaptive_threshold.get_normalized_scores(torch.tensor(adapted_map)).numpy()
            adapted_score = torch.mean(anomaly_maps[frame_idx]).item()
            
            # Track scores
            base_scores.append(base_score)
            adapted_scores.append(adapted_score)
            thresholds.append(threshold)
            
            # Create visualizations
            base_vis = base_processor.visualize(
                base_frame, 
                base_map, 
                base_map > base_processor.threshold
            )
            
            adapted_vis = base_processor.visualize(
                base_frame, 
                adapted_map, 
                adapted_map > threshold
            )
            
            # Add metrics overlays
            base_stats = {
                "model": "Base Model",
                "score": base_score,
                "threshold": base_processor.threshold
            }
            
            adapted_stats = {
                "model": "Adapted Model",
                "score": adapted_score,
                "threshold": threshold,
                "update_freq": args.update_freq,
                "alpha": args.alpha,
                "updates": adapter.frame_counter // args.update_freq,
                "conf": adapter.running_conf
            }
            
            base_vis = create_debug_overlay(base_vis, base_stats, base_score)
            adapted_vis = create_debug_overlay(adapted_vis, adapted_stats, adapted_score)
            
            # Combine visualizations side by side
            combined_vis = np.hstack((base_vis, adapted_vis))
            
            # Display if GUI enabled
            if not args.no_gui:
                cv2.imshow("InReaCh-FO Demo", combined_vis)
                
                # Update plot every 10 frames
                if frame_count % 10 == 0 and frame_count > 0:
                    plt.clf()
                    plt.plot(base_scores[-100:], 'b-', label='Base Model')
                    plt.plot(adapted_scores[-100:], 'g-', label='Adapted Model')
                    plt.plot(thresholds[-100:], 'r--', label='Threshold')
                    if args.drift_mode == "auto":
                        drift_history = [simulate_drift(np.zeros((1, 1, 3)), args.drift_type, i/100).mean() 
                                        for i in range(min(100, len(base_scores)))]
                        plt.plot(drift_history, 'y-', label='Drift Intensity')
                    plt.legend()
                    plt.title("Anomaly Scores")
                    plt.xlabel("Frame")
                    plt.ylabel("Score")
                    plt.grid(True)
                    plt.draw()
                    plt.pause(0.001)
                
            # Write frame to video if output path provided
            if video_writer is not None:
                video_writer.write(combined_vis)
            
            frame_count += batch.shape[0]
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                avg_time = sum(process_times) / len(process_times)
                p95_time = sorted(process_times)[int(len(process_times) * 0.95)]
                print(f"Frame {frame_count}: Avg Process Time = {avg_time:.2f} ms, P95 = {p95_time:.2f} ms")
                
                # Print adaptation metrics
                metrics = adapter.get_metrics()
                if "update_times" in metrics and "mean" in metrics["update_times"]:
                    print(f"Average update time: {metrics['update_times']['mean']:.2f} ms")
                if "confidence" in metrics and "mean" in metrics["confidence"]:
                    print(f"Average confidence: {metrics['confidence']['mean']:.4f}")
            
            # Exit on 'q' press
            if not args.no_gui and cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Release resources
        stream.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Save adaptation metrics
        metrics = adapter.get_metrics()
        if args.output:
            output_dir = os.path.dirname(args.output)
            metrics_path = os.path.join(output_dir, "adaptation_metrics")
            save_adaptation_metrics(metrics, metrics_path, save_plots=True)
        
        # Print final stats
        if process_times:
            avg_time = sum(process_times) / len(process_times)
            p95_time = sorted(process_times)[int(len(process_times) * 0.95)]
            effective_fps = 1000 / avg_time * args.batch_size
            print("\nPerformance Summary:")
            print(f"Processed {frame_count} frames")
            print(f"Average process time: {avg_time:.2f} ms per batch")
            print(f"P95 process time: {p95_time:.2f} ms per batch")
            print(f"Effective FPS: {effective_fps:.2f}")
            
            if len(base_scores) > 0 and len(adapted_scores) > 0:
                # Calculate average improvement
                improvement = sum(adapted_scores) / len(adapted_scores) - sum(base_scores) / len(base_scores)
                print(f"Average score improvement: {improvement:.4f}")
        
        print("Cleanup complete")

if __name__ == "__main__":
    main()