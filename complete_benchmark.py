#!/usr/bin/env python3
"""
Complete benchmark script for anomaly detection models.
Supports configurable dataset selection, category limits, and model selection.
Generates both numerical results and visualizations.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import torch

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("complete_benchmark")

def run_benchmark(dataset_name, output_dir, device, category_limit=None, 
                  skip_transfusion=False, skip_inreach=False, 
                  skip_samlad=False, skip_laft=False, fast_mode=False):
    """
    Run benchmark on specified dataset and models.
    
    Args:
        dataset_name: Name of the dataset (mvtec_ad2, visa, viaduct)
        output_dir: Path to save results
        device: Device to run on (cpu, cuda, mps)
        category_limit: Maximum number of categories to benchmark
        skip_transfusion: Skip TransFusion-Lite model
        skip_inreach: Skip InReachFO model
        skip_samlad: Skip SAM-LAD model
        skip_laft: Skip LAFT model
        fast_mode: Use fast mode (limit sample count)
        
    Returns:
        Path to the benchmark results JSON file
    """
    # Import benchmark components
    from experiments.m4_benchmarks.dataset_performance_benchmark import (
        DatasetLoader, 
        evaluate_transfusion, 
        evaluate_inreach, 
        evaluate_samlad, 
        evaluate_laft
    )
    
    # Create output directory
    dataset_output_dir = Path(output_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset setup
    dataset = DatasetLoader(dataset_name)
    
    # Limit categories if requested
    if category_limit is not None and category_limit > 0:
        categories = dataset.categories[:category_limit]
    else:
        categories = dataset.categories
    
    logger.info(f"Running benchmark on dataset: {dataset_name}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Using device: {device}, fast_mode: {fast_mode}")
    
    # Results to collect
    benchmark_results = []
    
    # Run benchmarks for each selected model
    if not skip_inreach:
        logger.info("Running InReachFO benchmark...")
        inreach_results = evaluate_inreach(dataset, categories, device, fast_mode=fast_mode)
        benchmark_results.append(inreach_results)
    
    if not skip_laft:
        logger.info("Running LAFT benchmark...")
        laft_results = evaluate_laft(dataset, categories, device, fast_mode=fast_mode)
        benchmark_results.append(laft_results)
    
    if not skip_transfusion:
        logger.info("Running TransFusion-Lite benchmark...")
        transfusion_results = evaluate_transfusion(dataset, categories, device, fast_mode=fast_mode)
        benchmark_results.append(transfusion_results)
    
    if not skip_samlad:
        logger.info("Running SAM-LAD benchmark...")
        samlad_results = evaluate_samlad(dataset, categories, device, fast_mode=fast_mode)
        benchmark_results.append(samlad_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = dataset_output_dir / f"{dataset_name}_performance_detailed_{timestamp}.json"
    
    with open(result_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    
    # Save summary
    summary_path = dataset_output_dir / f"{dataset_name}_performance_summary_{timestamp}.txt"
    
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
    
    # Print high-level results
    logger.info(f"Results saved to: {result_path}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("Benchmark Results Summary:")
    for result in benchmark_results:
        logger.info(f"{result['model']}: Overall AUROC = {result['overall_auroc']:.4f}")
    
    return str(result_path)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Complete anomaly detection benchmark")
    
    # Dataset selection
    parser.add_argument("--datasets", type=str, default="mvtec_ad2", 
                        help="Datasets to benchmark (comma-separated): mvtec_ad2,visa,viaduct")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results/benchmarks",
                        help="Directory to save benchmark results")
    
    # Model selection
    parser.add_argument("--skip-transfusion", action="store_true",
                        help="Skip benchmarking TransFusion-Lite model")
    parser.add_argument("--skip-inreach", action="store_true",
                        help="Skip benchmarking InReach-FO model")
    parser.add_argument("--skip-samlad", action="store_true",
                        help="Skip benchmarking SAM-LAD model")
    parser.add_argument("--skip-laft", action="store_true",
                        help="Skip benchmarking LAFT model")
    
    # Performance options
    parser.add_argument("--device", type=str, default="",
                        help="Device to run on (cpu, cuda, mps). Empty for auto-selection.")
    parser.add_argument("--fast-mode", action="store_true",
                        help="Use fast mode (limit sample count)")
    parser.add_argument("--category-limit", type=int, default=None,
                        help="Maximum number of categories to benchmark per dataset")
    
    # Process options
    parser.add_argument("--generate-tables", action="store_true",
                        help="Generate league tables and visualizations")
    parser.add_argument("--benchmark-results", type=str, default=None,
                        help="Use existing benchmark results instead of running new benchmarks")
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Process dataset list
    datasets = [ds.strip() for ds in args.datasets.split(",")]
    logger.info(f"Selected datasets: {datasets}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List of benchmark result files
    result_files = []
    
    # Run benchmarks if needed
    if args.benchmark_results is None:
        for dataset_name in datasets:
            logger.info(f"Starting benchmark for dataset: {dataset_name}")
            
            result_file = run_benchmark(
                dataset_name=dataset_name,
                output_dir=args.output_dir,
                device=device,
                category_limit=args.category_limit,
                skip_transfusion=args.skip_transfusion,
                skip_inreach=args.skip_inreach,
                skip_samlad=args.skip_samlad,
                skip_laft=args.skip_laft,
                fast_mode=args.fast_mode
            )
            
            result_files.append(result_file)
    else:
        # Use existing benchmark results
        result_files = [f.strip() for f in args.benchmark_results.split(",")]
        logger.info(f"Using existing benchmark results: {result_files}")
    
    # Generate tables and visualizations if requested
    if args.generate_tables:
        logger.info("Generating league tables and visualizations")
        
        # Import functions from benchmark_subset.py
        from benchmark_subset import (
            collect_benchmark_results,
            calculate_metrics,
            create_league_tables,
            create_visualizations
        )
        
        # Process benchmark results
        collected_data = collect_benchmark_results(result_files)
        metrics = calculate_metrics(collected_data)
        
        # Create league tables
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        tables = create_league_tables(metrics, str(tables_dir))
        
        # Create visualizations
        visualizations_dir = output_dir / "visualizations"
        visualizations_dir.mkdir(exist_ok=True)
        visualizations = create_visualizations(collected_data, metrics, str(visualizations_dir))
        
        logger.info(f"League tables saved to: {tables_dir}")
        logger.info(f"Visualizations saved to: {visualizations_dir}")
    
    logger.info("Benchmark complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())