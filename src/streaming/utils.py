import cv2
import numpy as np
import time
from typing import Tuple, Optional, List

def display_stream(frames: np.ndarray, window_name: str = "Stream"):
    """
    Display a batch of frames in a window.
    
    Args:
        frames: Batch of frames (batch_size, height, width, channels)
        window_name: Name of the window
    """
    # Create a grid layout
    batch_size = frames.shape[0]
    rows = int(np.ceil(np.sqrt(batch_size)))
    cols = int(np.ceil(batch_size / rows))
    
    # Get frame dimensions
    height, width = frames.shape[1:3]
    
    # Create a grid canvas
    grid = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    
    # Fill the grid with frames
    for i in range(batch_size):
        row = i // cols
        col = i % cols
        
        # Convert to uint8 if float
        if frames.dtype == np.float32 or frames.dtype == np.float64:
            frame = (frames[i] * 255).astype(np.uint8)
        else:
            frame = frames[i]
            
        grid[row*height:(row+1)*height, col*width:(col+1)*width] = frame
    
    # Display the grid
    cv2.imshow(window_name, grid)
    
def create_debug_overlay(frame: np.ndarray, stats: dict, anomaly_score: Optional[float] = None) -> np.ndarray:
    """
    Create a debug overlay on a frame.
    
    Args:
        frame: Input frame
        stats: Statistics to display
        anomaly_score: Optional anomaly score
        
    Returns:
        Frame with debug overlay
    """
    # Create a copy to avoid modifying the original
    debug_frame = frame.copy()
    
    # Draw FPS
    fps_text = f"FPS: {stats.get('fps', 0):.1f}"
    cv2.putText(debug_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw buffer size
    buffer_text = f"Buffer: {stats.get('buffer_size', 0)}"
    cv2.putText(debug_frame, buffer_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw anomaly score if provided
    if anomaly_score is not None:
        score_text = f"Anomaly: {anomaly_score:.4f}"
        color = (0, 255, 0) if anomaly_score < 0.5 else (0, 0, 255)
        cv2.putText(debug_frame, score_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return debug_frame

def benchmark_adapter(adapter, num_batches: int = 100) -> dict:
    """
    Benchmark the adapter performance.
    
    Args:
        adapter: StreamingAdapter instance
        num_batches: Number of batches to process
        
    Returns:
        Dictionary with benchmark results
    """
    start_time = time.time()
    batch_times = []
    
    for _ in range(num_batches):
        batch_start = time.time()
        batch = adapter.get_micro_batch()
        if batch is None:
            break
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    p95_batch_time = np.percentile(batch_times, 95) if batch_times else 0
    
    return {
        "total_time": total_time,
        "num_batches": len(batch_times),
        "avg_batch_time": avg_batch_time,
        "p95_batch_time": p95_batch_time,
        "effective_fps": len(batch_times) * adapter.batch_size / total_time if total_time > 0 else 0
    }

def create_test_video(output_path: str, resolution: Tuple[int, int] = (640, 480), 
                      duration: int = 10, fps: int = 30):
    """
    Create a test video for offline testing.
    
    Args:
        output_path: Path to save the video
        resolution: Video resolution (width, height)
        duration: Video duration in seconds
        fps: Frames per second
    """
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
    
    # Calculate number of frames
    num_frames = duration * fps
    
    # Generate frames
    for i in range(num_frames):
        # Create a gradient frame with a moving circle
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(resolution[1]):
            for x in range(resolution[0]):
                frame[y, x, 0] = int(255 * y / resolution[1])  # Blue gradient
                frame[y, x, 1] = int(255 * x / resolution[0])  # Green gradient
                frame[y, x, 2] = int(128 + 127 * np.sin(i / fps))  # Red oscillation
        
        # Add moving circle
        center_x = int(resolution[0] / 2 + resolution[0] / 4 * np.sin(i * 2 * np.pi / fps))
        center_y = int(resolution[1] / 2 + resolution[1] / 4 * np.cos(i * 2 * np.pi / fps))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Test video created at {output_path}")