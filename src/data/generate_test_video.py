#!/usr/bin/env python3
"""
Test video generator for Adaptive Vision-Based Anomaly Detection experiments.

This script generates synthetic videos with various patterns and anomalies
for testing the anomaly detection algorithms.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict

# Create test data directory if it doesn't exist
TEST_DATA_DIR = Path("data/test_videos")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

def create_normal_frame(
    width: int = 640,
    height: int = 480,
    frame_idx: int = 0,
    pattern_type: str = "grid"
) -> np.ndarray:
    """
    Create a synthetic normal frame with a regular pattern.
    
    Args:
        width: Frame width
        height: Frame height
        frame_idx: Frame index for time-varying patterns
        pattern_type: Type of pattern ('grid', 'texture', 'gradient')
        
    Returns:
        Synthetic frame as RGB numpy array (0-255)
    """
    # Create base frame (gray background)
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    if pattern_type == "grid":
        # Create grid pattern
        grid_size = 40
        for i in range(0, height, grid_size):
            cv2.line(frame, (0, i), (width, i), (150, 150, 150), 1)
        for j in range(0, width, grid_size):
            cv2.line(frame, (j, 0), (j, height), (150, 150, 150), 1)
            
        # Add some objects in grid cells
        for i in range(grid_size//2, height, grid_size):
            for j in range(grid_size//2, width, grid_size):
                # Alternate between circles and squares
                if (i // grid_size + j // grid_size) % 2 == 0:
                    # Circle
                    cv2.circle(frame, (j, i), 10, (100, 100, 240), -1)
                else:
                    # Square
                    cv2.rectangle(frame, (j-10, i-10), (j+10, i+10), (240, 100, 100), -1)
                    
    elif pattern_type == "texture":
        # Create noise texture
        noise = np.random.randint(0, 50, (height, width), dtype=np.uint8)
        frame[:,:,0] = frame[:,:,0] - noise//2
        frame[:,:,1] = frame[:,:,1] - noise
        frame[:,:,2] = frame[:,:,2] - noise//3
        
        # Add regular pattern
        pattern_size = 20
        for i in range(0, height, pattern_size):
            for j in range(0, width, pattern_size):
                if (i // pattern_size + j // pattern_size) % 2 == 0:
                    # Add texture pattern
                    y_start, y_end = i, min(i + pattern_size, height)
                    x_start, x_end = j, min(j + pattern_size, width)
                    frame[y_start:y_end, x_start:x_end, 0] += 30
                    
    elif pattern_type == "gradient":
        # Create gradient background
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        gradient_x = x / width * 255
        gradient_y = y / height * 255
        
        frame[:,:,0] = gradient_x.astype(np.uint8)
        frame[:,:,1] = gradient_y.astype(np.uint8)
        frame[:,:,2] = ((gradient_x + gradient_y) / 2).astype(np.uint8)
        
        # Add moving objects
        t = frame_idx * 0.05
        for i in range(5):
            x_pos = int(width/2 + width/4 * np.sin(t + i*0.7))
            y_pos = int(height/2 + height/4 * np.cos(t + i*0.7))
            size = 15 + i * 5
            cv2.circle(frame, (x_pos, y_pos), size, (255, 255, 255), -1)
            cv2.circle(frame, (x_pos, y_pos), size, (0, 0, 0), 2)
    
    # Add subtle time-varying element
    t = frame_idx * 0.02
    wave_pos = int(width/2 + width/4 * np.sin(t))
    cv2.line(frame, (wave_pos, 0), (wave_pos, height), (220, 220, 220), 3)
    
    return frame

def create_anomaly(
    frame: np.ndarray,
    anomaly_type: str = "scratch",
    intensity: float = 1.0,
    position: Optional[Tuple[int, int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add an anomaly to a frame.
    
    Args:
        frame: Input frame
        anomaly_type: Type of anomaly ('scratch', 'spot', 'missing', 'color')
        intensity: Anomaly intensity (0-1)
        position: Optional position override (x, y)
        
    Returns:
        Tuple of (frame with anomaly, ground truth mask)
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    result = frame.copy()
    
    # Set random position if not specified
    if position is None:
        x = np.random.randint(width // 4, width * 3 // 4)
        y = np.random.randint(height // 4, height * 3 // 4)
    else:
        x, y = position
    
    if anomaly_type == "scratch":
        # Create a scratch
        length = int(100 * intensity)
        thickness = max(1, int(3 * intensity))
        angle = np.random.rand() * 360
        
        end_x = int(x + length * np.cos(np.radians(angle)))
        end_y = int(y + length * np.sin(np.radians(angle)))
        
        cv2.line(result, (x, y), (end_x, end_y), (50, 50, 50), thickness)
        cv2.line(mask, (x, y), (end_x, end_y), 255, thickness * 3)
        
    elif anomaly_type == "spot":
        # Create a spot
        radius = int(20 * intensity)
        color = np.random.randint(0, 100, 3).tolist()
        
        cv2.circle(result, (x, y), radius, color, -1)
        cv2.circle(mask, (x, y), radius, 255, -1)
        
    elif anomaly_type == "missing":
        # Create a missing part (black region)
        size = int(30 * intensity)
        cv2.rectangle(result, (x - size, y - size), (x + size, y + size), (0, 0, 0), -1)
        cv2.rectangle(mask, (x - size, y - size), (x + size, y + size), 255, -1)
        
    elif anomaly_type == "color":
        # Create a color anomaly
        radius = int(30 * intensity)
        # Invert colors in the circle
        roi = result[y-radius:y+radius, x-radius:x+radius]
        if roi.size > 0:  # Check if ROI is valid
            roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(roi_mask, (radius, radius), radius, 255, -1)
            roi_masked = cv2.bitwise_and(roi, roi, mask=roi_mask)
            
            # Invert the masked region
            roi_inverted = cv2.bitwise_not(roi_masked)
            # Create inverted output by combining inverted ROI with original
            roi_output = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(roi_mask))
            roi_output = cv2.add(roi_output, roi_inverted)
            
            # Update result
            result[y-radius:y+radius, x-radius:x+radius] = roi_output
            cv2.circle(mask, (x, y), radius, 255, -1)
    
    return result, mask

def create_drift(
    frame: np.ndarray,
    drift_type: str = "lighting",
    intensity: float = 0.5
) -> np.ndarray:
    """
    Apply drift effect to a frame.
    
    Args:
        frame: Input frame
        drift_type: Type of drift ('lighting', 'blur', 'noise', 'color')
        intensity: Drift intensity (0-1)
        
    Returns:
        Frame with drift applied
    """
    result = frame.copy()
    
    if drift_type == "lighting":
        # Apply brightness drift
        gamma = 1.0 + (intensity * 0.5)  # 1.0 - 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype(np.uint8)
        result = cv2.LUT(result, table)
        
    elif drift_type == "blur":
        # Apply blur drift
        kernel_size = int(1 + intensity * 9) * 2 + 1  # Odd kernel from 3 to 21
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
    elif drift_type == "noise":
        # Apply noise drift
        noise = np.random.normal(0, intensity * 50, frame.shape).astype(np.int16)
        # Add noise to image
        result = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
    elif drift_type == "color":
        # Apply color drift by changing color balance
        result = result.astype(np.float32)
        # Modify color channels
        result[:,:,0] = np.clip(result[:,:,0] * (1.0 + intensity * 0.5), 0, 255)  # Increase blue
        result[:,:,1] = np.clip(result[:,:,1] * (1.0 - intensity * 0.3), 0, 255)  # Decrease green
        result = result.astype(np.uint8)
    
    return result

def generate_video(
    output_path: str,
    duration: int = 10,
    fps: int = 30,
    resolution: Tuple[int, int] = (640, 480),
    pattern_type: str = "grid",
    anomaly_config: Optional[Dict] = None,
    drift_config: Optional[Dict] = None
):
    """
    Generate a test video with optional anomalies and drift.
    
    Args:
        output_path: Path to save the video
        duration: Video duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
        pattern_type: Base pattern type
        anomaly_config: Anomaly configuration, or None for normal video
        drift_config: Drift configuration, or None for stable video
    """
    width, height = resolution
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    mask_writer = None
    
    # Create mask video if anomalies are present
    if anomaly_config:
        mask_path = output_path.replace('.avi', '_mask.avi')
        mask_writer = cv2.VideoWriter(mask_path, fourcc, fps, resolution, isColor=False)
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Generate frames
    for frame_idx in tqdm(range(total_frames), desc=f"Generating {output_path}"):
        # Create base frame
        frame = create_normal_frame(width, height, frame_idx, pattern_type)
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Apply anomaly if configured
        if anomaly_config and frame_idx >= anomaly_config.get("start_frame", 0):
            # Check if anomaly should be present in this frame
            if anomaly_config.get("intermittent", False):
                if frame_idx % anomaly_config.get("interval", 30) < anomaly_config.get("duration", 15):
                    frame, mask = create_anomaly(
                        frame,
                        anomaly_config.get("type", "scratch"),
                        anomaly_config.get("intensity", 1.0)
                    )
            else:
                frame, mask = create_anomaly(
                    frame,
                    anomaly_config.get("type", "scratch"),
                    anomaly_config.get("intensity", 1.0)
                )
        
        # Apply drift if configured
        if drift_config and frame_idx >= drift_config.get("start_frame", 0):
            intensity = drift_config.get("intensity", 0.5)
            
            # For gradual drift, calculate intensity ramp
            if drift_config.get("gradual", False):
                ramp_duration = drift_config.get("ramp_duration", int(fps * 2))  # Default 2s ramp
                progress = min(1.0, (frame_idx - drift_config.get("start_frame", 0)) / ramp_duration)
                intensity *= progress
                
            frame = create_drift(
                frame,
                drift_config.get("type", "lighting"),
                intensity
            )
        
        # Write frame
        video_writer.write(frame)
        
        # Write mask if anomaly writer exists
        if mask_writer:
            mask_writer.write(mask)
    
    # Release writers
    video_writer.release()
    if mask_writer:
        mask_writer.release()
    
    print(f"Video saved to {output_path}")
    if anomaly_config:
        print(f"Mask saved to {mask_path}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate test videos for anomaly detection")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path (default: auto-generated in data/test_videos)")
    parser.add_argument("--pattern", type=str, default="grid",
                       choices=["grid", "texture", "gradient"],
                       help="Base pattern type")
    parser.add_argument("--duration", type=int, default=10,
                       help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second")
    parser.add_argument("--width", type=int, default=640,
                       help="Video width")
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--anomaly-type", type=str, default=None,
                       choices=["scratch", "spot", "missing", "color", None],
                       help="Type of anomaly to add (None for normal video)")
    parser.add_argument("--anomaly-intensity", type=float, default=1.0,
                       help="Anomaly intensity (0-1)")
    parser.add_argument("--anomaly-start", type=float, default=3.0,
                       help="When to start anomaly (seconds)")
    parser.add_argument("--drift-type", type=str, default=None,
                       choices=["lighting", "blur", "noise", "color", None],
                       help="Type of drift to add (None for stable video)")
    parser.add_argument("--drift-intensity", type=float, default=0.5,
                       help="Drift intensity (0-1)")
    parser.add_argument("--drift-start", type=float, default=5.0,
                       help="When to start drift (seconds)")
    parser.add_argument("--gradual-drift", action="store_true",
                       help="Apply drift gradually")
    
    args = parser.parse_args()
    
    # Calculate frame indices from time
    anomaly_start_frame = int(args.anomaly_start * args.fps)
    drift_start_frame = int(args.drift_start * args.fps)
    
    # Configure anomaly if specified
    anomaly_config = None
    if args.anomaly_type:
        anomaly_config = {
            "type": args.anomaly_type,
            "intensity": args.anomaly_intensity,
            "start_frame": anomaly_start_frame
        }
    
    # Configure drift if specified
    drift_config = None
    if args.drift_type:
        drift_config = {
            "type": args.drift_type,
            "intensity": args.drift_intensity,
            "start_frame": drift_start_frame,
            "gradual": args.gradual_drift,
            "ramp_duration": args.fps * 2  # 2 second ramp
        }
    
    # Generate default output path if not specified
    if not args.output:
        parts = []
        parts.append(args.pattern)
        
        if anomaly_config:
            parts.append(f"{anomaly_config['type']}_i{anomaly_config['intensity']}")
            
        if drift_config:
            drift_str = f"{drift_config['type']}_i{drift_config['intensity']}"
            if drift_config['gradual']:
                drift_str += "_gradual"
            parts.append(drift_str)
            
        filename = f"synthetic_{'_'.join(parts)}.avi"
        args.output = str(TEST_DATA_DIR / filename)
    
    # Generate the video
    generate_video(
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        resolution=(args.width, args.height),
        pattern_type=args.pattern,
        anomaly_config=anomaly_config,
        drift_config=drift_config
    )

if __name__ == "__main__":
    main()