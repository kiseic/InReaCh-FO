#!/usr/bin/env python3
"""
Demo script for StreamingAdapter component.

This script demonstrates the StreamingAdapter with various input sources
and visualizes the micro-batching process.
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.streaming.adapter import StreamingAdapter

def visualize_batch(batch, window_name="Micro-Batch"):
    """
    Visualize a batch of frames in a grid layout.
    
    Args:
        batch: Batch of frames to visualize
        window_name: Name of the window to display
    """
    # Determine grid layout
    batch_size = batch.shape[0]
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    rows, cols = grid_size, grid_size
    
    # Get frame dimensions
    frame_height, frame_width = batch.shape[1:3]
    
    # Create canvas
    canvas_height = rows * frame_height
    canvas_width = cols * frame_width
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Fill canvas with frames
    for i in range(batch_size):
        row = i // cols
        col = i % cols
        
        # Convert frame to uint8 if float
        if batch.dtype == np.float32 or batch.dtype == np.float64:
            frame = (batch[i] * 255).astype(np.uint8)
        else:
            frame = batch[i]
            
        # Place frame in canvas
        y_start = row * frame_height
        y_end = (row + 1) * frame_height
        x_start = col * frame_width
        x_end = (col + 1) * frame_width
        
        canvas[y_start:y_end, x_start:x_end] = frame
    
    # Display canvas
    cv2.imshow(window_name, canvas)

def create_stats_display(stats, frame):
    """
    Create a stats display overlay on a frame.
    
    Args:
        stats: Statistics dictionary
        frame: Frame to overlay on
        
    Returns:
        Frame with stats overlay
    """
    # Create a copy to avoid modifying the original
    display = frame.copy()
    
    # Draw background for text
    cv2.rectangle(display, (10, 10), (300, 130), (0, 0, 0), -1)
    cv2.rectangle(display, (10, 10), (300, 130), (255, 255, 255), 1)
    
    # Draw statistics
    cv2.putText(display, f"FPS: {stats['fps']:.2f}", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(display, f"Buffer Size: {stats['buffer_size']}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.putText(display, f"Frames Processed: {stats['frames_processed']}", (20, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Get frame time stats
    if stats['frame_times']['mean'] > 0:
        mean_ms = stats['frame_times']['mean'] * 1000
        p95_ms = stats['frame_times']['p95'] * 1000
        
        cv2.putText(display, f"Avg Time: {mean_ms:.1f} ms", (150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(display, f"P95 Time: {p95_ms:.1f} ms", (150, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return display

def main():
    parser = argparse.ArgumentParser(description="StreamingAdapter Demo")
    parser.add_argument("--source", type=str, default="0", 
                       help="Source (camera index or video file)")
    parser.add_argument("--resolution", type=str, default="640x480",
                       help="Input resolution (WxH)")
    parser.add_argument("--target-size", type=str, default="480x480",
                       help="Processing size (WxH)")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Micro-batch size")
    parser.add_argument("--fps", type=int, default=30,
                       help="Target FPS")
    parser.add_argument("--roi", type=str, default=None,
                       help="Region of interest (x,y,w,h)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video file")
    
    args = parser.parse_args()
    
    # Parse resolution and target size
    width, height = map(int, args.resolution.split("x"))
    target_width, target_height = map(int, args.target_size.split("x"))
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        roi = tuple(map(int, args.roi.split(",")))
    
    # Parse source (camera index or video file)
    try:
        source = int(args.source)  # Try to convert to camera index
    except ValueError:
        source = args.source  # Use as file path
    
    # Create video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(
            args.output, 
            fourcc, 
            args.fps, 
            (target_width, target_height)
        )
    
    # Initialize adapter
    adapter = StreamingAdapter(
        source=source,
        resolution=(width, height),
        target_size=(target_width, target_height),
        batch_size=args.batch_size,
        fps=args.fps,
        roi=roi
    )
    
    try:
        # Print video properties
        print(f"Video properties:")
        print(f"  Resolution: {adapter.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{adapter.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  FPS: {adapter.cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  Frame count: {adapter.cap.get(cv2.CAP_PROP_FRAME_COUNT)}")
        
        # Test if we can read a single frame
        ret, frame = adapter.cap.read()
        if ret:
            print(f"Successfully read a frame with shape: {frame.shape}")
            # Reset video position to start
            adapter.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            print("Failed to read even a single frame from the video")
        
        while True:
            # Get batch
            batch = adapter.get_micro_batch()
            if batch is None:
                print("Failed to get micro-batch")
                break
            
            # Get statistics
            stats = adapter.get_stats()
            
            # Visualize batch
            visualize_batch(batch)
            
            # Process first frame for stats display
            first_frame = (batch[0] * 255).astype(np.uint8)
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display
            
            # Create stats display
            stats_display = create_stats_display(stats, first_frame)
            
            # Show stats
            cv2.imshow("Statistics", stats_display)
            
            # Write frames if output specified
            if writer:
                for i in range(batch.shape[0]):
                    frame = (batch[i] * 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving
                    writer.write(frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        # Release resources
        adapter.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()