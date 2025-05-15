import os
import sys
import argparse
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("laft-demo")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.transfusion.utils import get_available_device, load_model_checkpoint
from src.models.laft.model import LAFTPhi4
from src.streaming.adapter import StreamingAdapter
from src.streaming.utils import create_debug_overlay

def detect_phi_mini_path():
    """
    Try to automatically detect Phi-4-mini GGUF model path.
    
    Returns:
        Path to model if found, else None
    """
    # Common locations
    possible_paths = [
        # Home directory model locations
        os.path.expanduser("~/models/phi-4-mini/phi-4-mini.q4_k_m.gguf"),
        os.path.expanduser("~/models/phi-4-mini.q4_k_m.gguf"),
        os.path.expanduser("~/models/phi-4-mini.gguf"),
        os.path.expanduser("~/Models/phi-4-mini.gguf"),
        os.path.expanduser("~/ai/models/phi-4-mini.gguf"),
        os.path.expanduser("~/ai/phi-4-mini.gguf"),
        
        # Project directory models
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/weights/phi-4-mini.gguf")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../models/phi-4-mini.gguf")),
        
        # Common system paths
        "/usr/local/share/models/phi-4-mini.gguf",
        "/opt/models/phi-4-mini.gguf",
    ]
    
    # Try variations with different quantization formats
    quantizations = ["q4_k_m", "q5_k_m", "q6_k", "q8_0", ""]
    full_paths = []
    
    for base_path in possible_paths:
        full_paths.append(base_path)  # Original path
        # Add variations with different quantization
        for quant in quantizations:
            if quant:
                # For paths without explicit quantization, try adding it
                if ".gguf" in base_path and not any(q in base_path for q in quantizations):
                    variant = base_path.replace(".gguf", f".{quant}.gguf")
                    full_paths.append(variant)
    
    # Check all paths
    for path in full_paths:
        if os.path.exists(path):
            logger.info(f"Found Phi-4-mini model at: {path}")
            return path
            
    logger.warning("Could not automatically find Phi-4-mini model")
    return None

def main():
    parser = argparse.ArgumentParser(description="LAFT + Phi-4-mini Demo")
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file path')
    parser.add_argument('--resolution', type=str, default='640x480', help='Input resolution (WxH)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--model-path', type=str, default='', help='Path to TransFusion model checkpoint')
    parser.add_argument('--phi-model-path', type=str, default='', help='Path to Phi-4-mini GGUF model')
    parser.add_argument('--clip-model-name', type=str, default='openai/clip-vit-base-patch16', 
                      help='CLIP model name or path')
    parser.add_argument('--instruction', type=str, default='', 
                      help='Initial LAFT instruction (e.g., "Ignore oil stains; detect scratches")')
    parser.add_argument('--output', type=str, default='', help='Path to save output video')
    parser.add_argument('--use-float16', action='store_true', help='Use float16 precision for CLIP')
    parser.add_argument('--no-phi', action='store_true', help='Run without Phi-4-mini LLM')
    parser.add_argument('--sensitivity', type=str, default='medium', choices=['low', 'medium', 'high'],
                      help='Default sensitivity level')
    parser.add_argument('--cache-dir', type=str, default='', help='Cache directory for models')
    parser.add_argument('--save-state', type=str, default='', help='Path to save LAFT state')
    parser.add_argument('--load-state', type=str, default='', help='Path to load LAFT state')
    parser.add_argument('--sample-image', type=str, default='', 
                      help='Use a sample image instead of camera/video')
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Get device
    device = get_available_device()
    logger.info(f"Using device: {device}")
    
    # Initialize TransFusion model (base model)
    base_model = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    base_model.to(device)
    base_model.eval()  # Inference mode
    
    # Load checkpoint if provided
    if args.model_path:
        base_model, _ = load_model_checkpoint(base_model, args.model_path, device=device)
        logger.info(f"Loaded TransFusion model checkpoint from {args.model_path}")
    
    # Auto-detect Phi-4-mini model if not provided
    phi_model_path = args.phi_model_path
    if not phi_model_path and not args.no_phi:
        phi_model_path = detect_phi_mini_path()
        
    # Initialize LAFT + Phi-4-mini
    phi_model_path = None if args.no_phi else phi_model_path
    laft = LAFTPhi4(
        clip_model_name=args.clip_model_name,
        phi_model_path=phi_model_path,
        device=device,
        feature_dim=base_model.vit_feature_dim,
        use_float16=args.use_float16,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    
    # Load saved state if provided
    if args.load_state and os.path.exists(args.load_state):
        if laft.load_state(args.load_state):
            logger.info(f"Loaded LAFT state from {args.load_state}")
        else:
            logger.error(f"Failed to load LAFT state from {args.load_state}")
    
    # Create processor for anomaly detection
    processor = TransFusionProcessor(base_model, threshold=2.0, device=device)
    
    # Initialize image or video source
    is_image_mode = False
    if args.sample_image:
        # Use a sample image instead of camera/video
        if not os.path.exists(args.sample_image):
            logger.error(f"Sample image not found: {args.sample_image}")
            return
            
        logger.info(f"Using sample image: {args.sample_image}")
        sample_image = cv2.imread(args.sample_image)
        sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
        sample_image = cv2.resize(sample_image, (width, height))
        sample_image = sample_image.astype(np.float32) / 255.0
        is_image_mode = True
    else:
        # Initialize streaming adapter for camera or video
        source = int(args.source) if args.source.isdigit() else args.source
        
        if not args.source.isdigit() and not os.path.exists(source):
            logger.error(f"Video file not found: {source}")
            return
            
        stream = StreamingAdapter(
            source=source,
            resolution=(width, height),
            target_size=(224, 224),  # ViT input size
            batch_size=args.batch_size
        )
        logger.info(f"Initialized source: {source}")
    
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
        logger.info(f"Output video will be saved to: {args.output}")
    
    # Apply initial instruction if provided
    current_instruction = args.instruction
    if current_instruction:
        laft.adjust_feature_space(current_instruction)
        logger.info(f"Applied initial instruction: '{current_instruction}'")
    
    # Performance tracking
    frame_count = 0
    process_times = []
    scores = []
    explanations = []
    last_explanation_time = 0
    explanation_interval = 3.0  # seconds
    
    # Text input mode
    text_input = ""
    input_mode = False
    help_mode = False
    
    # Main processing loop
    try:
        logger.info("Starting main processing loop")
        while True:
            # Get input frame
            if is_image_mode:
                # Use the sample image
                frame = sample_image.copy()
            else:
                # Get a batch of frames
                batch = stream.get_micro_batch()
                if batch is None:
                    logger.warning("Failed to get micro-batch")
                    break
                
                # Use first frame from batch
                frame = batch[0]
            
            # Convert to torch tensor
            frame_tensor = torch.from_numpy(frame).float()
            frame_tensor = frame_tensor.permute(0, 3, 1, 2) if frame_tensor.dim() == 4 else frame_tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
            
            # Process frame for anomaly detection
            start_time = time.time()
            with torch.no_grad():
                # Extract base features
                features = base_model.vit(frame_tensor.to(device))
                
                # Apply LAFT transformation
                transformed_features = laft.transform_features(features)
                
                # Process with TransFusion (manually using transformed features)
                reshaped_features = base_model._reshape_features(transformed_features)
                
                # Apply diffusion process
                latent = reshaped_features.clone()
                for i in range(base_model.n_steps):
                    t = torch.tensor([i / base_model.n_steps], device=device)
                    latent = base_model.unet(latent, t).sample
                
                # Calculate anomaly map
                anomaly_map = torch.sum((reshaped_features - latent) ** 2, dim=1)
                
                # Normalize and resize
                anomaly_map = F.interpolate(
                    anomaly_map.unsqueeze(1), 
                    size=(frame_tensor.shape[2], frame_tensor.shape[3]), 
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            process_time = (time.time() - start_time) * 1000  # ms
            process_times.append(process_time)
            
            # Calculate score
            score = torch.mean(anomaly_map).item()
            scores.append(score)
            
            # Generate explanation periodically
            current_time = time.time()
            if current_time - last_explanation_time > explanation_interval:
                explanation = laft.generate_explanation(
                    anomaly_score=score,
                    anomaly_map=anomaly_map[0],
                    image_features=features
                )
                explanations.append(explanation)
                last_explanation_time = current_time
            else:
                explanation = explanations[-1] if explanations else ""
            
            # Create visualization
            frame_vis = (frame * 255).astype(np.uint8)
            
            # Normalize anomaly map for visualization
            anomaly_np = anomaly_map[0].cpu().numpy()
            norm_map = (anomaly_np - np.min(anomaly_np)) / (np.max(anomaly_np) - np.min(anomaly_np) + 1e-8)
            heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Resize heatmap to match frame if needed
            if heatmap.shape[:2] != frame_vis.shape[:2]:
                heatmap = cv2.resize(heatmap, (frame_vis.shape[1], frame_vis.shape[0]))
            
            # Blend with original image
            alpha = 0.6
            blend = cv2.addWeighted(frame_vis, 1-alpha, heatmap, alpha, 0)
            
            # Add text overlay
            stats = {
                "score": f"Score: {score:.4f}",
                "instruction": f"LAFT: {current_instruction if current_instruction else 'None'}",
                "process_time": f"Time: {process_time:.1f} ms"
            }
            
            # Create debug overlay
            debug_frame = create_debug_overlay(blend, stats, score)
            
            # Add explanation
            if explanation:
                # Draw background for text
                text_y = debug_frame.shape[0] - 80
                cv2.rectangle(debug_frame, (10, text_y - 30), (debug_frame.shape[1] - 10, text_y + 40), (0, 0, 0), -1)
                
                # Draw explanation with wrapping
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                max_width = debug_frame.shape[1] - 40
                
                # Split explanation into words and add them line by line
                words = explanation.split()
                lines = []
                current_line = ""
                
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
                    
                    if size[0] <= max_width:
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                        
                if current_line:
                    lines.append(current_line)
                
                # Draw lines
                for i, line in enumerate(lines[:2]):  # Limit to 2 lines
                    y = text_y + i * 25
                    cv2.putText(debug_frame, line, (20, y), font, font_scale, (255, 255, 255), thickness)
            
            # Add text input UI if in input mode
            if input_mode:
                # Draw background for text input
                cv2.rectangle(debug_frame, (10, 10), (debug_frame.shape[1] - 10, 50), (0, 0, 0), -1)
                cv2.putText(debug_frame, f"Enter instruction: {text_input}", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add instructions
                cv2.putText(debug_frame, "Press Enter to confirm, ESC to cancel", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show help if in help mode
            if help_mode:
                # Create semi-transparent overlay
                help_overlay = np.zeros_like(debug_frame)
                cv2.rectangle(help_overlay, (0, 0), (debug_frame.shape[1], debug_frame.shape[0]), (0, 0, 0), -1)
                debug_frame = cv2.addWeighted(debug_frame, 0.3, help_overlay, 0.7, 0)
                
                # Add help text
                help_text = [
                    "LAFT + Phi-4-mini Demo - Keyboard Controls",
                    "",
                    "i - Enter instruction input mode",
                    "s - Save current LAFT state",
                    "r - Reset LAFT to default state",
                    "1,2,3 - Set sensitivity (low, medium, high)",
                    "q - Quit the application",
                    "h - Toggle this help screen",
                    "",
                    "Current instruction: " + (current_instruction if current_instruction else "None"),
                    f"Sensitivity: {laft.adaptation_info['sensitivity']}",
                    f"Focus: {', '.join(laft.adaptation_info['focus']) if laft.adaptation_info['focus'] else 'None'}",
                    f"Ignore: {', '.join(laft.adaptation_info['ignore']) if laft.adaptation_info['ignore'] else 'None'}"
                ]
                
                y = 50
                for line in help_text:
                    cv2.putText(debug_frame, line, (50, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y += 30
            
            # Display frame
            cv2.imshow("LAFT + Phi-4-mini Demo", debug_frame)
            
            # Write frame to video if output path provided
            if video_writer is not None:
                video_writer.write(debug_frame)
                
            frame_count += 1
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                logger.info("Exiting on user request (q key)")
                break
            elif key == ord('i') and not input_mode and not help_mode:
                # Enter text input mode
                input_mode = True
                text_input = ""
                logger.info("Entered instruction input mode")
            elif key == ord('h'):
                # Toggle help mode
                help_mode = not help_mode
            elif key == ord('s') and not input_mode and not help_mode and args.save_state:
                # Save state
                laft.save_state(args.save_state)
                logger.info(f"Saved LAFT state to {args.save_state}")
            elif key == ord('r') and not input_mode and not help_mode:
                # Reset LAFT
                laft.reset()
                current_instruction = ""
                logger.info("Reset LAFT to default state")
            elif key in [ord('1'), ord('2'), ord('3')] and not input_mode and not help_mode:
                # Set sensitivity
                sensitivity = {ord('1'): "low", ord('2'): "medium", ord('3'): "high"}[key]
                laft.adaptation_info["sensitivity"] = sensitivity
                logger.info(f"Set sensitivity to {sensitivity}")
                
                # Update projection if we have a current instruction
                if laft.current_instruction:
                    laft.adjust_feature_space(laft.current_instruction)
            elif input_mode:
                if key == 13:  # Enter key
                    # Apply instruction and exit input mode
                    current_instruction = text_input
                    laft.adjust_feature_space(current_instruction)
                    input_mode = False
                    logger.info(f"Applied new instruction: '{current_instruction}'")
                elif key == 27:  # Escape key
                    # Cancel input
                    input_mode = False
                    logger.info("Cancelled instruction input")
                elif key == 8:  # Backspace
                    # Remove last character
                    text_input = text_input[:-1]
                elif 32 <= key <= 126:  # Printable ASCII characters
                    # Add character to input
                    text_input += chr(key)
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                avg_time = sum(process_times[-30:]) / min(30, len(process_times))
                avg_score = sum(scores[-30:]) / min(30, len(scores))
                logger.info(f"Frame {frame_count}: Avg Process Time = {avg_time:.2f} ms, Avg Score = {avg_score:.2f}")
                
                if explanations:
                    logger.info(f"Latest explanation: {explanations[-1]}")
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        # Release resources
        if not is_image_mode:
            stream.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Save final state if requested
        if args.save_state:
            laft.save_state(args.save_state)
            logger.info(f"Saved final LAFT state to {args.save_state}")
        
        # Print final stats
        if process_times:
            avg_time = sum(process_times) / len(process_times)
            p95_time = sorted(process_times)[int(len(process_times) * 0.95)]
            logger.info("\nPerformance Summary:")
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Average process time: {avg_time:.2f} ms per frame")
            logger.info(f"P95 process time: {p95_time:.2f} ms per frame")
            logger.info(f"Effective FPS: {1000 / avg_time:.2f}")
            
            # Get LAFT metrics
            metrics = laft.get_metrics()
            if metrics:
                logger.info("\nLAFT Metrics:")
                for key, values in metrics.items():
                    if isinstance(values, dict):
                        logger.info(f"  {key}:")
                        for stat, value in values.items():
                            logger.info(f"    {stat}: {value}")
                    else:
                        logger.info(f"  {key}: {values}")
        
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()