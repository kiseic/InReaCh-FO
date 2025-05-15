import os
import sys
import asyncio
import logging
import time
import json
import base64
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import models
from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.transfusion.utils import get_available_device
from src.models.inreach.model import InReachFO
from src.models.samlad.model import SAMLAD
from src.models.laft.model import LAFTPhi4
from src.streaming.adapter import StreamingAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Adaptive Vision-Based Anomaly Detection")

# Get base dir for templates and static files
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
TEMPLATE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Set up templates and static files
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def send_json(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")
                
    async def send_text(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message: {e}")

# Initialize connection manager
manager = ConnectionManager()

# Stream processing state
processing_state = {
    "running": False,
    "camera_index": 0,
    "resolution": (640, 480),
    "frame_count": 0,
    "scores": [],
    "process_times": [],
    "current_instruction": "",
    "latest_explanation": "",
    "adaptive_mode": True,
}

# Initialize models
device = get_available_device()
models = {
    "transfusion": None,
    "processor": None,
    "inreach": None,
    "samlad": None,
    "laft": None,
    "stream": None,
}

async def init_models(
    model_path: Optional[str] = None,
    phi_model_path: Optional[str] = None,
    sam_checkpoint: Optional[str] = None
):
    """Initialize all models"""
    global models, device
    
    # Initialize TransFusion model
    models["transfusion"] = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    models["transfusion"].to(device)
    
    # Load checkpoint if provided
    if model_path and os.path.exists(model_path):
        from src.models.transfusion.utils import load_model_checkpoint
        models["transfusion"], _ = load_model_checkpoint(
            models["transfusion"], model_path, device=device
        )
    
    # Initialize processor
    models["processor"] = TransFusionProcessor(
        models["transfusion"], threshold=2.0, device=device
    )
    
    # Initialize InReaCh-FO
    models["inreach"] = InReachFO(
        model=models["transfusion"],
        alpha=0.9,
        update_freq=8,
        confidence_thresh=0.7,
        track_metrics=True,
        device=device
    )
    
    # Initialize SAM-LAD if checkpoint provided
    if sam_checkpoint and os.path.exists(sam_checkpoint):
        models["samlad"] = SAMLAD(
            sam_checkpoint=sam_checkpoint,
            device=device
        )
    
    # Initialize LAFT + Phi-4-mini
    models["laft"] = LAFTPhi4(
        clip_model_path="openai/clip-vit-g-14",
        phi_model_path=phi_model_path if phi_model_path and os.path.exists(phi_model_path) else None,
        device=device,
        feature_dim=models["transfusion"].vit_feature_dim
    )
    
    logger.info("All models initialized successfully")

async def process_stream(websocket: WebSocket):
    """Process streaming frames and send results via WebSocket"""
    global models, processing_state
    
    # Initialize streaming adapter
    models["stream"] = StreamingAdapter(
        source=processing_state["camera_index"],
        resolution=processing_state["resolution"],
        target_size=(224, 224),  # ViT input size
        batch_size=1
    )
    
    # Reset state
    processing_state["frame_count"] = 0
    processing_state["scores"] = []
    processing_state["process_times"] = []
    
    # Processing loop
    while processing_state["running"]:
        # Get frame
        batch = models["stream"].get_micro_batch()
        if batch is None:
            await asyncio.sleep(0.1)
            continue
        
        # Convert to torch tensor
        import torch
        batch_tensor = torch.from_numpy(batch).float()
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        # Process frame with or without adaptation
        start_time = time.time()
        if processing_state["adaptive_mode"]:
            # Process with adaptation
            with torch.no_grad():
                # Forward through InReaCh-FO (which handles adaptive BN updates)
                outputs = models["inreach"](batch_tensor.to(device))
                anomaly_maps, features, latents = outputs
        else:
            # Process without adaptation
            with torch.no_grad():
                outputs = models["transfusion"](batch_tensor.to(device))
                anomaly_maps, features, latents = outputs
        
        # Apply LAFT if instruction is set
        if processing_state["current_instruction"]:
            with torch.no_grad():
                # Apply LAFT transformation to features
                transformed_features = models["laft"].transform_features(features)
                
                # Reshape and process with TransFusion
                reshaped_features = models["transfusion"]._reshape_features(transformed_features)
                
                # Apply diffusion process
                latent = reshaped_features.clone()
                for i in range(models["transfusion"].n_steps):
                    t = torch.tensor([i / models["transfusion"].n_steps], device=device)
                    latent = models["transfusion"].unet(latent, t).sample
                
                # Calculate anomaly map
                anomaly_maps = torch.sum((reshaped_features - latent) ** 2, dim=1)
                
        process_time = (time.time() - start_time) * 1000  # ms
        processing_state["process_times"].append(process_time)
        
        # Calculate score
        score = torch.mean(anomaly_maps[0]).item()
        processing_state["scores"].append(score)
        
        # Generate explanation periodically (every 3 seconds)
        if models["laft"] and processing_state["frame_count"] % 90 == 0:  # Assuming 30 FPS
            explanation = models["laft"].generate_explanation(
                features,
                anomaly_maps[0],
                score
            )
            processing_state["latest_explanation"] = explanation
        
        # Create visualization
        frame = (batch[0] * 255).astype(np.uint8)
        
        # Normalize anomaly map for visualization
        anomaly_np = anomaly_maps[0].cpu().numpy()
        norm_map = (anomaly_np - np.min(anomaly_np)) / (np.max(anomaly_np) - np.min(anomaly_np) + 1e-8)
        heatmap = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with original image
        alpha = 0.6
        blend = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
        
        # Encode as base64 for transmission
        _, buffer = cv2.imencode('.jpg', blend)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Send data
        await websocket.send_json({
            "type": "frame",
            "frame": img_str,
            "score": score,
            "process_time": process_time,
            "frame_count": processing_state["frame_count"],
            "explanation": processing_state["latest_explanation"],
            "adaptive_mode": processing_state["adaptive_mode"],
            "instruction": processing_state["current_instruction"]
        })
        
        processing_state["frame_count"] += 1
        
        # Don't overwhelm the connection
        await asyncio.sleep(0.01)
    
    # Clean up
    if models["stream"]:
        models["stream"].release()
        models["stream"] = None

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page"""
    # Create simple HTML page if it doesn't exist
    html_path = TEMPLATE_DIR / "index.html"
    if not html_path.exists():
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Adaptive Vision-Based Anomaly Detection</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                .header {
                    text-align: center;
                    margin-bottom: 20px;
                }
                .video-container {
                    display: flex;
                    justify-content: center;
                    margin-bottom: 20px;
                }
                #video {
                    max-width: 100%;
                    border: 1px solid #ddd;
                }
                .controls {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin-bottom: 20px;
                }
                .metrics {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                }
                .metric {
                    flex: 1;
                    background-color: #f0f0f0;
                    padding: 10px;
                    border-radius: 5px;
                }
                .explanation {
                    background-color: #f9f9f9;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                button {
                    padding: 8px 15px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #45a049;
                }
                input[type="text"] {
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    width: 300px;
                }
                .toggle {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Adaptive Vision-Based Anomaly Detection</h1>
                    <p>Real-time anomaly detection with adaptation</p>
                </div>
                
                <div class="video-container">
                    <img id="video" src="" alt="Video stream">
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <h3>Anomaly Score</h3>
                        <p id="score">0.00</p>
                    </div>
                    <div class="metric">
                        <h3>Processing Time</h3>
                        <p id="process_time">0 ms</p>
                    </div>
                    <div class="metric">
                        <h3>Frame Count</h3>
                        <p id="frame_count">0</p>
                    </div>
                </div>
                
                <div class="explanation">
                    <h3>Explanation</h3>
                    <p id="explanation">No anomalies detected yet.</p>
                </div>
                
                <div class="controls">
                    <button id="startBtn">Start</button>
                    <button id="stopBtn">Stop</button>
                    <div class="toggle">
                        <label for="adaptiveToggle">Adaptive Mode:</label>
                        <input type="checkbox" id="adaptiveToggle" checked>
                    </div>
                </div>
                
                <div class="controls">
                    <input type="text" id="instructionInput" placeholder="Enter LAFT instruction (e.g., 'Ignore oil stains; detect scratches')">
                    <button id="applyInstructionBtn">Apply</button>
                </div>
            </div>
            
            <script>
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                const videoElement = document.getElementById('video');
                const scoreElement = document.getElementById('score');
                const processTimeElement = document.getElementById('process_time');
                const frameCountElement = document.getElementById('frame_count');
                const explanationElement = document.getElementById('explanation');
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const adaptiveToggle = document.getElementById('adaptiveToggle');
                const instructionInput = document.getElementById('instructionInput');
                const applyInstructionBtn = document.getElementById('applyInstructionBtn');
                
                ws.onopen = function(event) {
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'frame') {
                        videoElement.src = 'data:image/jpeg;base64,' + data.frame;
                        scoreElement.textContent = data.score.toFixed(4);
                        processTimeElement.textContent = data.process_time.toFixed(2) + ' ms';
                        frameCountElement.textContent = data.frame_count;
                        
                        if (data.explanation) {
                            explanationElement.textContent = data.explanation;
                        }
                    }
                };
                
                ws.onclose = function(event) {
                    console.log('WebSocket disconnected');
                };
                
                startBtn.addEventListener('click', function() {
                    ws.send(JSON.stringify({
                        action: 'start',
                        adaptive_mode: adaptiveToggle.checked
                    }));
                });
                
                stopBtn.addEventListener('click', function() {
                    ws.send(JSON.stringify({
                        action: 'stop'
                    }));
                });
                
                adaptiveToggle.addEventListener('change', function() {
                    ws.send(JSON.stringify({
                        action: 'set_adaptive_mode',
                        enabled: adaptiveToggle.checked
                    }));
                });
                
                applyInstructionBtn.addEventListener('click', function() {
                    const instruction = instructionInput.value.trim();
                    if (instruction) {
                        ws.send(JSON.stringify({
                            action: 'set_instruction',
                            instruction: instruction
                        }));
                    }
                });
            </script>
        </body>
        </html>
        """
        
        with open(html_path, "w") as f:
            f.write(html_content)
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        # Initialize models if not already initialized
        if models["transfusion"] is None:
            await init_models()
        
        # Process messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "start":
                # Start processing
                processing_state["running"] = True
                processing_state["adaptive_mode"] = message.get("adaptive_mode", True)
                # Start processing in background
                asyncio.create_task(process_stream(websocket))
                
            elif message.get("action") == "stop":
                # Stop processing
                processing_state["running"] = False
                
            elif message.get("action") == "set_adaptive_mode":
                # Set adaptive mode
                processing_state["adaptive_mode"] = message.get("enabled", True)
                
            elif message.get("action") == "set_instruction":
                # Set LAFT instruction
                instruction = message.get("instruction", "")
                if instruction and models["laft"]:
                    models["laft"].adjust_feature_space(instruction)
                    processing_state["current_instruction"] = instruction
                    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
        manager.disconnect(websocket)
        # Stop processing if running
        processing_state["running"] = False
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        manager.disconnect(websocket)
        # Stop processing if running
        processing_state["running"] = False

@app.get("/status")
async def get_status():
    """Get current processing status"""
    return {
        "running": processing_state["running"],
        "frame_count": processing_state["frame_count"],
        "adaptive_mode": processing_state["adaptive_mode"],
        "current_instruction": processing_state["current_instruction"],
        "models_initialized": models["transfusion"] is not None
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    # Create static and template directories
    STATIC_DIR.mkdir(exist_ok=True)
    TEMPLATE_DIR.mkdir(exist_ok=True)
    logger.info("Server started")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    # Stop processing if running
    processing_state["running"] = False
    # Release streaming if active
    if models["stream"]:
        models["stream"].release()
    logger.info("Server shutdown")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI App for Adaptive Anomaly Detection")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--model-path", type=str, default="", help="Path to TransFusion model checkpoint")
    parser.add_argument("--phi-model-path", type=str, default="", help="Path to Phi-4-mini GGUF model")
    parser.add_argument("--sam-checkpoint", type=str, default="", help="Path to SAM checkpoint")
    
    args = parser.parse_args()
    
    # Run the app
    uvicorn.run(
        "app:app", 
        host=args.host, 
        port=args.port, 
        reload=True
    )