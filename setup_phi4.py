#!/usr/bin/env python3
"""
This script will download and test the Phi-4-mini model with LAFTPhi4 class.
"""

import os
import sys
import logging
import torch
import requests
from pathlib import Path
from tqdm import tqdm
import time

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup_phi4")

def download_file(url, dest_path):
    """
    Download a file with progress bar
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(dest_path):
        logger.info(f"File already exists at {dest_path}")
        return True
        
    # Start download
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        file_size = int(response.headers.get('content-length', 0))
        
        # Show progress bar
        logger.info(f"Downloading {url} to {dest_path}")
        logger.info(f"File size: {file_size / (1024 * 1024):.2f} MB")
        
        progress = tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading")
        
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
        
        progress.close()
        logger.info(f"Download completed: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def download_phi4_model():
    """
    Download the Phi-4-mini model
    """
    # Setup paths
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    phi4_dir = models_dir / "phi4-mini"
    phi4_dir.mkdir(exist_ok=True)
    
    # Define model URL and path
    # Using Q4_K_M quantization which is 2.49 GB
    model_url = "https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/resolve/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    model_path = phi4_dir / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
    
    # Download model
    success = download_file(model_url, model_path)
    
    return str(model_path) if success else None

def test_phi4_model(model_path):
    """
    Test the downloaded Phi-4-mini model with LAFTPhi4
    """
    from src.models.laft.model import LAFTPhi4
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return False
    
    # Get device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Initialize LAFTPhi4
        logger.info("Initializing LAFTPhi4...")
        laft_phi4 = LAFTPhi4(
            clip_model_name="openai/clip-vit-base-patch16",
            phi_model_path=model_path,
            device=device,
            feature_dim=512
        )
        
        # Test feature transformation
        logger.info("Testing feature transformation...")
        test_features = torch.randn(1, 512, device=device)
        
        # Adjust feature space with instruction
        instruction = "Detect scratches on metal surfaces with high sensitivity and ignore dust particles"
        logger.info(f"Adjusting feature space with instruction: {instruction}")
        laft_phi4.adjust_feature_space(instruction)
        
        # Transform features
        transformed = laft_phi4.transform_features(test_features)
        logger.info(f"Transformed features shape: {transformed.shape}")
        
        # Test explanation generation
        logger.info("Testing explanation generation with Phi-4-mini...")
        for score in [0.2, 0.6, 0.8]:
            start_time = time.time()
            explanation = laft_phi4.generate_explanation(score)
            gen_time = time.time() - start_time
            logger.info(f"Score: {score:.2f} -> Explanation: {explanation}")
            logger.info(f"Generation time: {gen_time:.2f} seconds")
        
        logger.info("LAFTPhi4 test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing LAFTPhi4: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Download model
    logger.info("Step 1: Downloading Phi-4-mini model...")
    model_path = download_phi4_model()
    
    if not model_path:
        logger.error("Failed to download model. Exiting.")
        return 1
        
    # Test model
    logger.info("Step 2: Testing Phi-4-mini model with LAFTPhi4...")
    success = test_phi4_model(model_path)
    
    if success:
        logger.info("All tests passed! Phi-4-mini is ready to use.")
        return 0
    else:
        logger.error("Tests failed. Check the logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())