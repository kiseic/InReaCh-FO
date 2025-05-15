#!/usr/bin/env python3
"""
Simple test script for video reading and preprocessing.
"""

import os
import sys
import cv2
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test video reading")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.video):
        print(f"Error: File not found: {args.video}")
        return
        
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Failed to open video: {args.video}")
        return
    
    # Get video properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    print(f"Video properties:")
    print(f"  Path: {args.video}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frame count: {frame_count}")
    print(f"  Duration: {frame_count/fps:.2f} seconds")
    
    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame_normalized = frame_rgb / 255.0
        
        frames.append(frame_normalized)
        
        # Print progress every 10 frames
        if len(frames) % 10 == 0:
            print(f"Read {len(frames)} frames...")
    
    # Close video
    cap.release()
    
    print(f"Successfully read {len(frames)} frames")
    
    # Try to create a micro-batch
    if len(frames) >= 5:
        micro_batch = np.stack(frames[:5], axis=0)
        print(f"Created micro-batch with shape {micro_batch.shape}")
    
    # Save first frame as an image for inspection
    if frames:
        first_frame = (frames[0] * 255).astype(np.uint8)
        first_frame_bgr = cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
        
        # Save to temporary file
        output_path = "first_frame.png"
        cv2.imwrite(output_path, first_frame_bgr)
        print(f"Saved first frame to {output_path}")

if __name__ == "__main__":
    main()