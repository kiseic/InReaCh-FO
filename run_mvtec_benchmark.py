#!/usr/bin/env python3
"""
MVTec-AD2 benchmark script for all models.
Runs a benchmark on MVTec-AD2 categories with all implemented models.
Uses fast mode to limit the number of images processed.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import torch

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mvtec_benchmark")

def main():
    # Import benchmark components
    from experiments.m4_benchmarks.dataset_performance_benchmark import (
        DatasetLoader, 
        evaluate_transfusion, 
        evaluate_inreach, 
        evaluate_samlad, 
        evaluate_laft
    )
    
    # Device selection (CPU for stability)
    device = "cpu"
    
    # Create output directory
    output_dir = Path("results/benchmarks/mvtec_ad2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Fast mode to limit sample count
    fast_mode = True
    
    # Dataset setup
    dataset_name = "mvtec_ad2"
    dataset = DatasetLoader(dataset_name)
    
    # Limit to available categories - take only the first 3 to keep it manageable
    categories = dataset.categories[:3]
    logger.info(f"Running benchmark on dataset: {dataset_name}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Using device: {device}, fast_mode: {fast_mode}")
    
    # Results to collect
    benchmark_results = []
    
    # Run InReachFO benchmark
    logger.info("Running InReachFO benchmark...")
    inreach_results = evaluate_inreach(dataset, categories, device, fast_mode=fast_mode)
    benchmark_results.append(inreach_results)
    
    # Run LAFT benchmark
    logger.info("Running LAFT benchmark...")
    laft_results = evaluate_laft(dataset, categories, device, fast_mode=fast_mode)
    benchmark_results.append(laft_results)
    
    # Run TransFusion-Lite benchmark
    logger.info("Running TransFusion-Lite benchmark...")
    transfusion_results = evaluate_transfusion(dataset, categories, device, fast_mode=fast_mode)
    benchmark_results.append(transfusion_results)
    
    # Run SAM-LAD benchmark
    logger.info("Running SAM-LAD benchmark...")
    samlad_results = evaluate_samlad(dataset, categories, device, fast_mode=fast_mode)
    benchmark_results.append(samlad_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"{dataset_name}_performance_detailed_{timestamp}.json"
    
    with open(result_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    
    # Save summary
    summary_path = output_dir / f"{dataset_name}_performance_summary_{timestamp}.txt"
    
    with open(summary_path, "w") as f:
        f.write(f"=== {dataset_name} Benchmark Results ===\n\n")
        
        for result in benchmark_results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Overall AUROC: {result['overall_auroc']:.4f}\n\n")
            
            f.write("Category Results:\n")
            for category, metrics in result["categories"].items():
                f.write(f"  {category}:\n")
                f.write(f"    AUROC: {metrics['auroc']:.4f}\n")
                f.write(f"    Pixel AUROC: {metrics['pixel_auroc']:.4f}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    # Print results summary
    logger.info(f"Results saved to: {result_path}")
    logger.info(f"Summary saved to: {summary_path}")
    
    # Print high-level results
    logger.info("Benchmark Results Summary:")
    for result in benchmark_results:
        logger.info(f"{result['model']}: Overall AUROC = {result['overall_auroc']:.4f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())