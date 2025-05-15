#!/usr/bin/env python3
"""
Test script for LAFTPhi4 functionality.
This will test if LAFTPhi4 works even without a Phi4 model by using fallback mechanisms.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_phi4")

def main():
    # Import LAFTPhi4 from model module
    from src.models.laft.model import LAFTPhi4, HAS_TRANSFORMERS, HAS_LLAMA_CPP
    
    # Check available dependencies
    logger.info(f"Transformers available: {HAS_TRANSFORMERS}")
    logger.info(f"llama-cpp-python available: {HAS_LLAMA_CPP}")
    
    # Initialize LAFTPhi4 without the Phi4 model
    # It should fall back to basic explanation generation
    device = "cpu"
    laft_phi4 = LAFTPhi4(
        clip_model_name="openai/clip-vit-base-patch16",
        phi_model_path=None,  # No model provided, should use fallback
        device=device,
        feature_dim=512
    )
    
    # Test feature transformation
    logger.info("Testing feature transformation...")
    test_features = torch.randn(1, 512, device=device)
    
    # Try to adjust feature space with an instruction
    instruction = "Detect scratches and ignore dust particles with high sensitivity"
    logger.info(f"Adjusting feature space with instruction: {instruction}")
    laft_phi4.adjust_feature_space(instruction)
    
    # Transform features
    transformed = laft_phi4.transform_features(test_features)
    logger.info(f"Original features shape: {test_features.shape}")
    logger.info(f"Transformed features shape: {transformed.shape}")
    
    # Calculate difference to see if transformation was applied
    diff = torch.norm(transformed - test_features).item()
    logger.info(f"L2 difference after transformation: {diff:.6f}")
    
    # Test explanation generation without Phi model (should use fallback)
    logger.info("Testing explanation generation...")
    for score in [0.2, 0.4, 0.6, 0.8]:
        explanation = laft_phi4.generate_explanation(score)
        logger.info(f"Score: {score:.2f} -> Explanation: {explanation}")
    
    # Get metrics
    metrics = laft_phi4.get_metrics()
    logger.info(f"Metrics: {metrics}")
    
    logger.info("LAFTPhi4 test completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())