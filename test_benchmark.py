#!/usr/bin/env python3
"""
Mini test benchmark to verify model implementations.
Only runs InReachFO on one category.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_benchmark")

def main():
    # Import DatasetLoader and evaluate_inreach from benchmark script
    from experiments.m4_benchmarks.dataset_performance_benchmark import DatasetLoader, evaluate_inreach
    
    # Device selection (CPU for stability)
    device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Select dataset and category
    dataset_name = "mvtec_ad2"
    category = "bottle"  # Just one category for quick test
    
    # Create dataset loader
    dataset = DatasetLoader(dataset_name)
    logger.info(f"Loaded dataset: {dataset_name} with categories: {dataset.categories}")
    
    # Run benchmark on InReachFO only
    logger.info(f"Testing InReachFO on category: {category}")
    result = evaluate_inreach(dataset, [category], device, fast_mode=True)
    
    # Output results
    logger.info(f"Benchmark complete. Results:")
    logger.info(f"Overall AUROC: {result['overall_auroc']:.4f}")
    
    for cat, metrics in result["categories"].items():
        logger.info(f"Category {cat}:")
        logger.info(f"  AUROC: {metrics['auroc']:.4f}")
        logger.info(f"  Pixel AUROC: {metrics['pixel_auroc']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())