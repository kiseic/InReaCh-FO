import os
import sys
import argparse
import cv2
import numpy as np
import time
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("samlad-demo")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.samlad.model import SAMLAD
from src.streaming.adapter import StreamingAdapter
from src.streaming.utils import create_debug_overlay

# Check if SAM is available
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_SAM = True
except ImportError:
    logger.warning("Warning: Segment Anything Model (SAM) not found. Running in fallback mode.")
    HAS_SAM = False

def simulate_anomaly(image, anomaly_type="missing", intensity=0.5):
    """
    Simulate different types of logical anomalies.
    
    Args:
        image: Input image
        anomaly_type: Type of anomaly to simulate
        intensity: Anomaly intensity
        
    Returns:
        Image with simulated anomaly
    """
    result = image.copy()
    height, width = image.shape[:2]
    
    if anomaly_type == "missing":
        # Simulate missing object by masking out part of the image
        mask_size = int(min(width, height) * intensity * 0.3)
        mask_x = np.random.randint(0, width - mask_size)
        mask_y = np.random.randint(0, height - mask_size)
        
        # White rectangle
        result[mask_y:mask_y+mask_size, mask_x:mask_x+mask_size] = 1.0
        
    elif anomaly_type == "extra":
        # Simulate extra object by adding a new shape
        obj_size = int(min(width, height) * intensity * 0.2)
        obj_x = np.random.randint(0, width - obj_size)
        obj_y = np.random.randint(0, height - obj_size)
        
        # Draw random colored shape
        color = np.random.random(3)
        shape_type = np.random.choice(["circle", "rectangle", "triangle"])
        
        if shape_type == "circle":
            cv2.circle(
                result,
                (obj_x + obj_size//2, obj_y + obj_size//2),
                obj_size//2,
                color,
                -1
            )
        elif shape_type == "rectangle":
            cv2.rectangle(
                result,
                (obj_x, obj_y),
                (obj_x + obj_size, obj_y + obj_size),
                color,
                -1
            )
        else:  # triangle
            points = np.array([
                [obj_x + obj_size//2, obj_y],
                [obj_x, obj_y + obj_size],
                [obj_x + obj_size, obj_y + obj_size]
            ], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(result, [points], color)
            
    elif anomaly_type == "moved":
        # Simulate moved object by applying perspective transform
        # Define transform matrix
        src_points = np.float32([
            [0, 0],
            [width-1, 0],
            [width-1, height-1],
            [0, height-1]
        ])
        
        # Apply intensity-based distortion
        dst_points = np.float32([
            [width * intensity * 0.2, height * intensity * 0.1],
            [width * (1 - intensity * 0.2), height * intensity * 0.1],
            [width * (1 - intensity * 0.1), height * (1 - intensity * 0.1)],
            [width * intensity * 0.1, height * (1 - intensity * 0.1)]
        ])
        
        # Apply perspective transform
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(result, M, (width, height))
        
    elif anomaly_type == "color":
        # Simulate color abnormality
        # Apply color shift to the entire image
        hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + intensity * 90) % 180  # Shift hue
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + intensity * 0.5), 0, 1)  # Increase saturation
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
    return result

def download_sam_checkpoint(checkpoint_dir, sam_type="vit_h"):
    """
    Download SAM checkpoint if not already downloaded.
    
    Args:
        checkpoint_dir: Directory to save the checkpoint
        sam_type: SAM model type (vit_h, vit_l, vit_b)
        
    Returns:
        Path to the checkpoint file
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # URLs for different model types
    urls = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    }
    
    # Check if requested model type is supported
    if sam_type not in urls:
        logger.error(f"Unsupported SAM model type: {sam_type}")
        return None
        
    # Generate checkpoint path
    checkpoint_path = os.path.join(checkpoint_dir, f"sam_{sam_type}.pth")
    
    # Download if not exists
    if not os.path.exists(checkpoint_path):
        try:
            logger.info(f"Downloading SAM {sam_type} checkpoint...")
            
            # Use curl or wget to download
            if os.system(f"curl -L {urls[sam_type]} -o {checkpoint_path}") != 0:
                if os.system(f"wget {urls[sam_type]} -O {checkpoint_path}") != 0:
                    # If both fail, try with Python
                    import requests
                    response = requests.get(urls[sam_type], stream=True)
                    if response.status_code == 200:
                        with open(checkpoint_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    else:
                        logger.error(f"Failed to download checkpoint: {response.status_code}")
                        return None
                        
            logger.info(f"Downloaded SAM checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            return None
    else:
        logger.info(f"Using existing SAM checkpoint at {checkpoint_path}")
        
    return checkpoint_path

def load_test_image(path, resolution=(640, 480)):
    """
    Load a test image from a file.
    
    Args:
        path: Path to the image file
        resolution: Target resolution
        
    Returns:
        Loaded image
    """
    if not os.path.exists(path):
        logger.error(f"Image file not found: {path}")
        # Create blank image
        return np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
        
    # Load image
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Failed to load image: {path}")
        return np.zeros((resolution[1], resolution[0], 3), dtype=np.float32)
        
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, resolution)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img

def main():
    parser = argparse.ArgumentParser(description="SAM-LAD Demo")
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file or image file')
    parser.add_argument('--resolution', type=str, default='640x480', help='Input resolution (WxH)')
    parser.add_argument('--sam-checkpoint', type=str, default='', help='Path to SAM checkpoint')
    parser.add_argument('--sam-type', type=str, default='vit_h', choices=['vit_b', 'vit_l', 'vit_h'], help='SAM model type')
    parser.add_argument('--reference', type=str, default='', help='Path to reference model (load or save)')
    parser.add_argument('--save-reference', action='store_true', help='Save reference model')
    parser.add_argument('--anomaly-mode', type=str, default='none', choices=['none', 'manual', 'auto'], help='Anomaly mode')
    parser.add_argument('--anomaly-type', type=str, default='missing', choices=['missing', 'extra', 'moved', 'color'], help='Anomaly type for auto mode')
    parser.add_argument('--output', type=str, default='', help='Path to save output video')
    parser.add_argument('--min-mask-area', type=int, default=100, help='Minimum mask area')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark mode')
    parser.add_argument('--use-tracking', action='store_true', help='Enable object tracking')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on')
    parser.add_argument('--sample-image', type=str, default='', help='Sample image path to use instead of camera')
    parser.add_argument('--test-modes', action='store_true', help='Test all anomaly modes automatically')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Download SAM checkpoint if not provided
    if not args.sam_checkpoint and HAS_SAM:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../models/weights'))
        args.sam_checkpoint = download_sam_checkpoint(models_dir, args.sam_type)
    
    # Initialize SAM-LAD
    sam_checkpoint = args.sam_checkpoint if HAS_SAM and args.sam_checkpoint else None
    sam_lad = SAMLAD(
        sam_checkpoint=sam_checkpoint,
        sam_type=args.sam_type,
        device=args.device,
        min_mask_area=args.min_mask_area,
        reference_path=args.reference if os.path.exists(args.reference) else None,
        use_tracking=args.use_tracking
    )
    
    # Initialize camera/video source
    is_image_mode = False
    if args.sample_image:
        # Use a still image
        is_image_mode = True
        sample_image = load_test_image(args.sample_image, (width, height))
        logger.info(f"Using sample image: {args.sample_image}")
    elif args.source.isdigit():
        # Camera index
        source = int(args.source)
        logger.info(f"Using camera source: {source}")
    else:
        # Video file
        source = args.source
        if not os.path.exists(source):
            logger.error(f"Video file not found: {source}")
            return
        logger.info(f"Using video source: {source}")
    
    # Initialize streaming adapter (if not in image mode)
    if not is_image_mode:
        stream = StreamingAdapter(
            source=source,
            resolution=(width, height),
            target_size=(width, height),  # Keep original size for SAM
            batch_size=1  # Process one frame at a time
        )
    
    # Create video writer if output path provided
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            args.output, 
            fourcc, 
            30.0, 
            (width, height)
        )
    
    # Performance tracking
    frame_count = 0
    process_times = []
    scores = []
    current_anomaly_intensity = 0.0
    anomaly_direction = 1
    reference_saved = False
    
    # Benchmarking variables
    if args.benchmark:
        start_time = time.time()
        benchmark_frames = 100
        logger.info(f"Running benchmark mode with {benchmark_frames} frames")
    
    # Test modes variables
    test_modes = []
    if args.test_modes:
        test_modes = ["none", "missing", "extra", "moved", "color"]
        current_mode_idx = 0
        frames_per_mode = 100
        mode_frames = 0
        logger.info(f"Testing all anomaly modes, {frames_per_mode} frames per mode")
    
    # Allow keyboard control of anomaly intensity
    intensity_keys = {
        ord('0'): 0.0,
        ord('1'): 0.1,
        ord('2'): 0.2,
        ord('3'): 0.3,
        ord('4'): 0.4,
        ord('5'): 0.5,
        ord('6'): 0.6,
        ord('7'): 0.7,
        ord('8'): 0.8,
        ord('9'): 0.9,
        ord('m'): 1.0  # Max
    }
    
    # Anomaly type keys
    type_keys = {
        ord('n'): "none",
        ord('x'): "missing",
        ord('e'): "extra",
        ord('v'): "moved",
        ord('c'): "color"
    }
    
    # Current manual settings
    current_manual_intensity = 0.0
    current_manual_type = "none"
    
    # Main processing loop
    try:
        while True:
            # Get a frame
            if is_image_mode:
                frame = sample_image.copy()
            else:
                batch = stream.get_micro_batch()
                if batch is None:
                    logger.info("End of video or camera failure")
                    break
                # Take the first frame
                frame = batch[0]
            
            # Apply anomaly if in auto mode or test_modes mode
            anomaly_desc = "none"
            if args.test_modes:
                # Switch mode if needed
                if mode_frames >= frames_per_mode:
                    current_mode_idx = (current_mode_idx + 1) % len(test_modes)
                    mode_frames = 0
                    logger.info(f"Switching to anomaly mode: {test_modes[current_mode_idx]}")
                
                # Get current mode
                current_mode = test_modes[current_mode_idx]
                mode_frames += 1
                
                # Apply anomaly if not in "none" mode
                if current_mode != "none" and frame_count > 30:
                    # Calculate intensity based on frame position in the mode
                    phase = mode_frames / frames_per_mode
                    # Ramp up and down within the mode
                    if phase < 0.5:
                        current_anomaly_intensity = phase * 2  # 0 to 1
                    else:
                        current_anomaly_intensity = 2 - phase * 2  # 1 to 0
                    
                    # Apply anomaly
                    frame = simulate_anomaly(frame, current_mode, current_anomaly_intensity)
                    anomaly_desc = f"{current_mode} ({current_anomaly_intensity:.2f})"
                    
            elif args.anomaly_mode == "auto" and frame_count > 30:
                # Update anomaly intensity with oscillation
                if frame_count % 30 == 0:  # Change every 30 frames
                    current_anomaly_intensity += 0.05 * anomaly_direction
                    if current_anomaly_intensity >= 1.0 or current_anomaly_intensity <= 0.0:
                        anomaly_direction *= -1
                        current_anomaly_intensity = max(0.0, min(1.0, current_anomaly_intensity))
                
                # Apply anomaly
                frame = simulate_anomaly(frame, args.anomaly_type, current_anomaly_intensity)
                anomaly_desc = f"{args.anomaly_type} ({current_anomaly_intensity:.2f})"
                
            elif args.anomaly_mode == "manual" and current_manual_intensity > 0:
                # Apply manual anomaly
                frame = simulate_anomaly(frame, current_manual_type, current_manual_intensity)
                anomaly_desc = f"{current_manual_type} ({current_manual_intensity:.2f})"
            
            # Save reference from first clean frames
            if not reference_saved and args.save_reference and frame_count == 15:
                # Process the frame
                results = sam_lad.process_image(frame)
                
                # Save reference if requested
                if args.reference:
                    sam_lad.save_reference(args.reference)
                    reference_saved = True
                    logger.info(f"Saved reference model to {args.reference}")
                
            # Process the frame
            start_time_frame = time.time()
            results = sam_lad.process_image(frame)
            process_time = (time.time() - start_time_frame) * 1000  # ms
            process_times.append(process_time)
            
            # Track anomaly score
            scores.append(results["anomaly_score"])
            
            # Get stats
            if is_image_mode:
                adapter_stats = {"fps": 0}
            else:
                adapter_stats = stream.get_stats()
            
            # Create visualization
            viz_frame = sam_lad.visualize_results(
                (frame * 255).astype(np.uint8),  # Convert to uint8
                results,
                show_masks=True,
                show_relationships=True,
                show_clusters=True
            )
            
            # Add stats overlay
            stats = {
                "fps": adapter_stats["fps"] if not is_image_mode else 1000 / (process_time + 1e-6),
                "process_time": process_time,
                "frame": frame_count,
                "objects": results["n_objects"],
                "anomaly": f"{results['anomaly_score']:.2f}"
            }
            
            if args.anomaly_mode == "auto" or args.test_modes or (args.anomaly_mode == "manual" and current_manual_intensity > 0):
                stats["anomaly_type"] = anomaly_desc
                
            debug_frame = create_debug_overlay(viz_frame, stats, results["anomaly_score"])
            
            # Add help text
            cv2.putText(
                debug_frame,
                "Keys: 0-9 (intensity), n,x,e,v,c (type), q (quit), r (reset)",
                (10, debug_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
            
            # Display frame
            cv2.imshow("SAM-LAD Demo", debug_frame)
            
            # Write frame to video if output path provided
            if video_writer is not None:
                video_writer.write(debug_frame)
            
            frame_count += 1
            
            # Print stats every 10 frames
            if frame_count % 10 == 0:
                avg_time = sum(process_times[-10:]) / min(10, len(process_times))
                avg_score = sum(scores[-10:]) / min(10, len(scores))
                logger.info(f"Frame {frame_count}: Avg Process Time = {avg_time:.2f} ms, Avg Score = {avg_score:.2f}")
            
            # Check for benchmark completion
            if args.benchmark and frame_count >= benchmark_frames:
                break
                
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on 'q' press
            if key == ord('q'):
                break
                
            # Update manual intensity
            if key in intensity_keys:
                current_manual_intensity = intensity_keys[key]
                logger.info(f"Set anomaly intensity to {current_manual_intensity}")
                
            # Update anomaly type
            if key in type_keys:
                current_manual_type = type_keys[key]
                logger.info(f"Set anomaly type to {current_manual_type}")
                
            # Reset reference model on 'r' press
            if key == ord('r'):
                sam_lad.object_relations = None
                logger.info("Reset reference model")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Release resources
        if not is_image_mode:
            stream.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Save reference if requested and not already saved
        if args.save_reference and args.reference and not reference_saved:
            sam_lad.save_reference(args.reference)
            logger.info(f"Saved reference model to {args.reference}")
        
        # Print final stats
        if process_times:
            avg_time = sum(process_times) / len(process_times)
            p95_time = sorted(process_times)[int(len(process_times) * 0.95)]
            logger.info("\nPerformance Summary:")
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Average process time: {avg_time:.2f} ms per frame")
            logger.info(f"P95 process time: {p95_time:.2f} ms per frame")
            logger.info(f"Average FPS: {1000 / avg_time:.2f}")
            
            # Get detailed metrics
            metrics = sam_lad.get_metrics()
            logger.info("\nDetailed Metrics:")
            for key, values in metrics.items():
                logger.info(f"  {key.capitalize()}:")
                for stat, value in values.items():
                    logger.info(f"    {stat}: {value:.2f} ms")
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()