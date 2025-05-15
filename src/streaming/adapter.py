#!/usr/bin/env python3
"""
Streaming Adapter implementation for Adaptive Vision-Based Anomaly Detection.

This module provides the StreamingAdapter class for handling camera or video inputs
with 5-frame micro-batching processing.
"""

import cv2
import numpy as np
import time
import logging
import os
from typing import List, Tuple, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

class StreamingAdapter:
    """
    StreamingAdapter handles camera input processing with 5-frame micro-batching.
    It provides preprocessing, buffering, and frame synchronization.
    """
    
    def __init__(
        self, 
        source: Union[int, str] = 0, 
        resolution: Tuple[int, int] = (640, 480),
        target_size: Tuple[int, int] = (480, 480),
        batch_size: int = 5,
        fps: int = 30,
        roi: Optional[Tuple[int, int, int, int]] = None,
        preprocess_fn = None
    ):
        """
        Initialize the streaming adapter.
        
        Args:
            source: Camera index or video file path
            resolution: Input resolution (width, height)
            target_size: Target size for processing (width, height)
            batch_size: Number of frames to batch together
            fps: Target frames per second
            roi: Region of interest (x, y, width, height) or None for full frame
            preprocess_fn: Optional custom preprocessing function
        """
        self.source = source
        self.resolution = resolution
        self.target_size = target_size
        self.batch_size = batch_size
        self.fps = fps
        self.roi = roi
        self.preprocess_fn = preprocess_fn
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(source)
        self.configure_capture()
        
        # Buffer for frames
        self.buffer = []
        
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        self.current_fps = 0
        self.frames_processed = 0
        
        logger.info(f"StreamingAdapter initialized with source={source}, "
                  f"batch_size={batch_size}, target_size={target_size}")
        
    def configure_capture(self):
        """Configure video capture properties."""
        if not self.cap.isOpened():
            # Try to provide more details about the error
            if isinstance(self.source, str) and os.path.exists(self.source):
                file_size = os.path.getsize(self.source)
                readable = os.access(self.source, os.R_OK)
                readable_str = "readable" if readable else "not readable"
                error_msg = f"Failed to open video file: {self.source} (size: {file_size} bytes, {readable_str})"
            else:
                error_msg = f"Failed to open video source: {self.source}"
                
            raise RuntimeError(error_msg)
            
        # Set capture properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Check if properties were set correctly
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        logger.info(f"Video source configured with resolution={actual_width}x{actual_height}, " 
                    f"fps={actual_fps}, frames={frame_count}")
        
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply standard preprocessing to a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame (RGB format, normalized)
        """
        # Apply ROI if specified
        if self.roi is not None:
            x, y, w, h = self.roi
            frame = frame[y:y+h, x:x+w]
            
        # Resize to target size
        frame = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        frame = frame / 255.0
        
        # Apply custom preprocessing if provided
        if self.preprocess_fn is not None:
            frame = self.preprocess_fn(frame)
            
        return frame
    
    def get_micro_batch(self) -> Optional[np.ndarray]:
        """
        Get a micro-batch of frames.
        
        Returns:
            Batch of frames as numpy array with shape (batch_size, height, width, channels)
            or None if frames cannot be read
        """
        # Simpler implementation that reads frames directly into a batch
        start_time = time.time()
        frames = []
        
        # Read batch_size frames or until end of video
        for _ in range(self.batch_size):
            ret, frame = self.cap.read()
            if not ret:
                # End of video or read error
                break
                
            # Track FPS
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.frame_times.append(elapsed)
            self.frame_times = self.frame_times[-30:]  # Keep last 30 frames
            self.current_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times)) if self.frame_times else 0
            
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            frames.append(processed_frame)
            self.frames_processed += 1
        
        # Check if we got any frames
        if not frames:
            logger.info("No frames read from source")
            return None
            
        # If we got some frames but not enough, pad with the last frame
        if len(frames) < self.batch_size:
            logger.info(f"Only got {len(frames)} frames, padding to {self.batch_size}")
            last_frame = frames[-1]
            while len(frames) < self.batch_size:
                frames.append(last_frame)
        
        # Create batch
        batch = np.stack(frames, axis=0)
        
        # Record time
        elapsed = time.time() - start_time
        logger.debug(f"Micro-batch created in {elapsed:.4f}s with shape {batch.shape}")
        
        return batch
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get adapter statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "fps": self.current_fps,
            "buffer_size": len(self.buffer),
            "frames_processed": self.frames_processed,
            "frame_times": {
                "mean": np.mean(self.frame_times) if self.frame_times else 0,
                "p95": np.percentile(self.frame_times, 95) if self.frame_times else 0
            }
        }
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("StreamingAdapter released")
        
    def __del__(self):
        """Ensure resources are released on garbage collection."""
        self.release()


class BatchProcessor:
    """
    Helper class for processing and visualizing micro-batches.
    """
    
    def __init__(self, adapter: StreamingAdapter):
        """
        Initialize the batch processor.
        
        Args:
            adapter: StreamingAdapter instance
        """
        self.adapter = adapter
        
    def process_next_batch(self, model, visualize: bool = False):
        """
        Process the next batch of frames.
        
        Args:
            model: Model to use for prediction
            visualize: Whether to visualize results
            
        Returns:
            Tuple of (batch, predictions, visualization) if successful, or None
        """
        # Get next batch
        batch = self.adapter.get_micro_batch()
        if batch is None:
            return None
            
        # Process batch with model
        try:
            import torch
            
            # Convert numpy to torch tensor
            if isinstance(batch, np.ndarray):
                batch_tensor = torch.from_numpy(batch).float()
                
                # Move to appropriate device
                if hasattr(model, "device"):
                    device = model.device
                elif next(model.parameters(), None) is not None:
                    device = next(model.parameters()).device
                else:
                    device = "cpu"
                    
                batch_tensor = batch_tensor.to(device)
                
                # Get predictions
                with torch.no_grad():
                    predictions = model(batch_tensor)
                    
            else:
                # If not numpy, assume model can handle it directly
                predictions = model(batch)
                
        except ImportError:
            # Fallback if torch is not available
            predictions = model(batch)
            
        if visualize:
            # Create visualization of the batch and predictions
            visualization = self._create_visualization(batch, predictions)
            return batch, predictions, visualization
            
        return batch, predictions, None
        
    def _create_visualization(self, batch, predictions):
        """
        Create visualization of batch and predictions.
        
        Args:
            batch: Input batch
            predictions: Model predictions
            
        Returns:
            Visualization image
        """
        # Basic implementation - can be overridden in subclasses
        # Just show the first frame and its prediction
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with 2 columns
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Show first frame
            axes[0].imshow(batch[0])
            axes[0].set_title("Input Frame")
            axes[0].axis("off")
            
            # Show prediction (assume it's an image-like output)
            if isinstance(predictions, tuple) and len(predictions) > 0:
                pred = predictions[0]
            else:
                pred = predictions
                
            import torch
            if isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()
                
            # If prediction is a mask or heatmap (2D)
            if len(pred.shape) == 2 or (len(pred.shape) == 3 and pred.shape[0] == 1):
                if len(pred.shape) == 3:
                    pred = pred[0]
                im = axes[1].imshow(pred, cmap="jet")
                plt.colorbar(im, ax=axes[1])
            elif len(pred.shape) == 3:
                # RGB image
                axes[1].imshow(pred)
            elif len(pred.shape) == 1:
                # 1D prediction (e.g., class scores)
                axes[1].bar(range(len(pred)), pred)
                axes[1].set_xlabel("Class")
                axes[1].set_ylabel("Score")
                
            axes[1].set_title("Prediction")
            
            # Convert to image
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return image
            
        except ImportError:
            # Fallback if matplotlib is not available
            # Create a simple grid of the first frame and prediction
            frame = (batch[0] * 255).astype(np.uint8)
            
            # Create placeholder for prediction viz
            pred_viz = np.zeros_like(frame)
            
            # Create side-by-side visualization
            viz = np.hstack((frame, pred_viz))
            
            # Add text
            cv2.putText(viz, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(viz, "Prediction", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            return viz