#!/usr/bin/env python3
"""
M4ベンチマーク実行スクリプト

このスクリプトは、M4環境でのモデル評価のために以下のベンチマークを順に実行します：
1. モデル別レイテンシ比較
2. データセット別性能評価（AUROC）
3. ドリフトリカバリ比較実験
4. LAFTの言語指示ベース適応評価

環境変数やコマンドライン引数を使用して、特定のベンチマークのみの実行やパラメータ調整が可能です。
"""

import os
import sys
import argparse
import subprocess
import logging
import time
from pathlib import Path

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("benchmark_run.log")
    ]
)
logger = logging.getLogger("benchmark_runner")

def run_command(command, desc=None):
    """コマンドを実行し、結果を返す"""
    if desc:
        logger.info(f"実行中: {desc}")
    logger.info(f"コマンド: {command}")
    
    try:
        start_time = time.time()
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True)
        elapsed = time.time() - start_time
        
        logger.info(f"完了（{elapsed:.2f}秒）")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"エラー: {e}")
        logger.error(f"標準出力: {e.stdout}")
        logger.error(f"標準エラー: {e.stderr}")
        return False, e.stderr

def run_latency_benchmark(args):
    """モデル別レイテンシ比較ベンチマーク"""
    logger.info("===== モデル別レイテンシ比較ベンチマーク =====")
    
    # コマンドの構築
    cmd = ["python", "experiments/m4_benchmarks/model_latency_benchmark.py"]
    
    # オプションの追加
    if args.device:
        cmd.append(f"--device {args.device}")
    if args.output_dir:
        cmd.append(f"--output-dir {args.output_dir}")
    if args.fast_mode:
        cmd.append("--runs 20")
        cmd.append("--warmup 5")
    
    # モデルスキップオプション
    if args.skip_transfusion:
        cmd.append("--skip-transfusion")
    if args.skip_inreach:
        cmd.append("--skip-inreach")
    if args.skip_samlad:
        cmd.append("--skip-samlad")
    if args.skip_laft:
        cmd.append("--skip-laft")
    
    # コマンド実行
    success, output = run_command(" ".join(cmd), "レイテンシベンチマーク")
    return success

def run_dataset_benchmark(args):
    """データセット別性能評価（AUROC）"""
    logger.info("===== データセット別性能評価（AUROC）=====")
    
    # 実行するデータセット
    datasets = ["mvtec_ad2"]
    if args.dataset:
        datasets = [args.dataset]
    elif not args.fast_mode:
        datasets = ["mvtec_ad2", "visa"]
    
    all_success = True
    
    for dataset in datasets:
        logger.info(f"データセット: {dataset}")
        
        # コマンドの構築
        cmd = ["python", "experiments/m4_benchmarks/dataset_performance_benchmark.py"]
        cmd.append(f"--dataset {dataset}")
        
        # オプションの追加
        if args.device:
            cmd.append(f"--device {args.device}")
        if args.output_dir:
            cmd.append(f"--output-dir {args.output_dir}")
        if args.fast_mode:
            cmd.append("--fast-mode")
            cmd.append("--num-categories 3")
        
        # モデルスキップオプション
        if args.skip_transfusion:
            cmd.append("--skip-transfusion")
        if args.skip_inreach:
            cmd.append("--skip-inreach")
        if args.skip_samlad:
            cmd.append("--skip-samlad")
        if args.skip_laft:
            cmd.append("--skip-laft")
        
        # コマンド実行
        success, output = run_command(" ".join(cmd), f"{dataset}データセット評価")
        all_success = all_success and success
    
    return all_success

def run_drift_benchmark(args):
    """ドリフトリカバリ比較実験"""
    logger.info("===== ドリフトリカバリ比較実験 =====")
    
    # 実行するドリフトタイプ
    drift_types = ["lighting"]
    if args.drift_type:
        drift_types = [args.drift_type]
    elif not args.fast_mode:
        drift_types = ["lighting", "scratch", "texture"]
    
    all_success = True
    
    for drift_type in drift_types:
        logger.info(f"ドリフトタイプ: {drift_type}")
        
        # コマンドの構築
        cmd = ["python", "experiments/m4_benchmarks/drift_recovery_benchmark.py"]
        cmd.append(f"--drift-type {drift_type}")
        cmd.append("--save-video")  # ビデオを保存
        
        # オプションの追加
        if args.device:
            cmd.append(f"--device {args.device}")
        if args.output_dir:
            cmd.append(f"--output-dir {args.output_dir}")
        if args.fast_mode:
            cmd.append("--duration 60")
        
        # モデルスキップオプション
        if args.skip_transfusion:
            cmd.append("--skip-transfusion")
        if args.skip_inreach:
            cmd.append("--skip-inreach")
        if args.skip_samlad:
            cmd.append("--skip-samlad")
        if args.skip_laft:
            cmd.append("--skip-laft")
        
        # コマンド実行
        success, output = run_command(" ".join(cmd), f"{drift_type}ドリフト評価")
        all_success = all_success and success
    
    return all_success

def run_laft_instruction_benchmark(args):
    """LAFTの言語指示ベース適応評価"""
    logger.info("===== LAFTの言語指示ベース適応評価 =====")
    
    # コマンドの構築
    cmd = ["python", "experiments/m4_benchmarks/laft_instruction_benchmark.py"]
    
    # オプションの追加
    if args.device:
        cmd.append(f"--device {args.device}")
    if args.output_dir:
        cmd.append(f"--output-dir {args.output_dir}")
    if args.fast_mode:
        cmd.append("--max-instructions 2")
        cmd.append("--max-images 20")
    
    # カテゴリの指定
    if args.categories:
        cmd.append(f"--categories {args.categories}")
    
    # コマンド実行
    success, output = run_command(" ".join(cmd), "LAFT指示評価")
    return success

def main():
    parser = argparse.ArgumentParser(description="M4ベンチマーク実行スクリプト")
    parser.add_argument("--benchmark", type=str, default="all",
                       choices=["all", "latency", "dataset", "drift", "laft"],
                       help="実行するベンチマーク")
    parser.add_argument("--device", type=str, default="",
                       help="使用するデバイス (mps, cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="./experiments/m4_benchmarks/results",
                       help="結果の保存先ディレクトリ")
    parser.add_argument("--fast-mode", action="store_true",
                       help="高速モード（少数のサンプルで評価）")
    parser.add_argument("--dataset", type=str, default="",
                       help="評価するデータセット (mvtec_ad2, visa, viaduct)")
    parser.add_argument("--drift-type", type=str, default="",
                       choices=["lighting", "scratch", "texture", "defects"],
                       help="評価するドリフトの種類")
    parser.add_argument("--categories", type=str, default="",
                       help="LAFTで評価するカテゴリ（カンマ区切り）")
    parser.add_argument("--skip-transfusion", action="store_true",
                       help="TransFusion-Liteの評価をスキップ")
    parser.add_argument("--skip-inreach", action="store_true",
                       help="InReaCh-FOの評価をスキップ")
    parser.add_argument("--skip-samlad", action="store_true",
                       help="SAM-LADの評価をスキップ")
    parser.add_argument("--skip-laft", action="store_true",
                       help="LAFTの評価をスキップ")
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"結果を保存するディレクトリ: {output_dir}")
    
    # ベンチマークの選択と実行
    success = True
    benchmark_results = {}
    
    start_time = time.time()
    
    if args.benchmark in ["all", "latency"]:
        latency_success = run_latency_benchmark(args)
        benchmark_results["latency"] = latency_success
        success = success and latency_success
    
    if args.benchmark in ["all", "dataset"]:
        dataset_success = run_dataset_benchmark(args)
        benchmark_results["dataset"] = dataset_success
        success = success and dataset_success
    
    if args.benchmark in ["all", "drift"]:
        drift_success = run_drift_benchmark(args)
        benchmark_results["drift"] = drift_success
        success = success and drift_success
    
    if args.benchmark in ["all", "laft"]:
        laft_success = run_laft_instruction_benchmark(args)
        benchmark_results["laft"] = laft_success
        success = success and laft_success
    
    total_time = time.time() - start_time
    
    # 結果の出力
    logger.info("===== ベンチマーク実行結果 =====")
    for benchmark, result in benchmark_results.items():
        status = "成功" if result else "失敗"
        logger.info(f"{benchmark}: {status}")
    
    logger.info(f"合計実行時間: {total_time:.2f}秒")
    logger.info(f"全体結果: {'成功' if success else '一部失敗'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())