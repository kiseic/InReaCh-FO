#!/usr/bin/env python3
"""
モデル別レイテンシ比較ベンチマーク

このスクリプトは、各モデル（TransFusion-Lite、InReaCh-FO、SAM-LAD、LAFT）の
レイテンシを計測し、結果をCSVファイルに保存します。
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import csv
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# モデルのインポート
from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.inreach.model import InReachFO
from src.models.samlad.model import SAMLAD
from src.models.laft.model import LAFTPhi4
from src.models.transfusion.utils import get_available_device, load_model_checkpoint

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_latency_benchmark")

def generate_random_image(size=(224, 224)):
    """テスト用のランダム画像を生成"""
    return torch.rand((1, 3, size[0], size[1]))

def benchmark_transfusion(device, runs=100, warmup=10):
    """TransFusion-Liteモデルのレイテンシを計測"""
    logger.info("TransFusion-Liteモデルのベンチマーク開始")
    
    # モデルの初期化
    model = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    model.to(device)
    model.eval()
    
    # プロセッサの初期化
    processor = TransFusionProcessor(model, threshold=2.0, device=device)
    
    # ランダム入力を生成
    inputs = [generate_random_image().to(device) for _ in range(runs + warmup)]
    
    # ウォームアップ実行
    logger.info(f"{warmup}回のウォームアップ実行")
    with torch.no_grad():
        for i in range(warmup):
            _ = processor.process(inputs[i])
    
    # 計測開始
    logger.info(f"{runs}回の計測実行")
    latencies = []
    
    with torch.no_grad():
        for i in range(warmup, runs + warmup):
            start_time = time.time()
            _ = processor.process(inputs[i])
            latency = (time.time() - start_time) * 1000  # ミリ秒に変換
            latencies.append(latency)
            
            if (i - warmup + 1) % 10 == 0:
                logger.info(f"Progress: {i - warmup + 1}/{runs}, 最新レイテンシ: {latency:.2f}ms")
    
    # 統計を計算
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # 結果を返す
    result = {
        "model": "TransFusion-Lite",
        "runs": runs,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "latencies": latencies
    }
    
    logger.info(f"TransFusion-Lite平均レイテンシ: {mean_latency:.2f}ms")
    return result

def benchmark_inreach(device, runs=100, warmup=10):
    """InReaCh-FOモデルのレイテンシを計測"""
    logger.info("InReaCh-FOモデルのベンチマーク開始")
    
    # 基本モデルの初期化
    base_model = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    base_model.to(device)
    base_model.eval()
    
    # InReaCh-FOの初期化
    model = InReachFO(
        base_model=base_model,
        confidence_thresh=0.7,
        adaptation_strategy="batch_norm",
        device=device
    )
    
    # ランダム入力を生成
    inputs = [generate_random_image().to(device) for _ in range(runs + warmup)]
    
    # ウォームアップ実行
    logger.info(f"{warmup}回のウォームアップ実行")
    with torch.no_grad():
        for i in range(warmup):
            _ = model.predict(inputs[i])
    
    # 計測開始
    logger.info(f"{runs}回の計測実行")
    latencies = []
    
    with torch.no_grad():
        for i in range(warmup, runs + warmup):
            start_time = time.time()
            _ = model.predict(inputs[i])
            latency = (time.time() - start_time) * 1000  # ミリ秒に変換
            latencies.append(latency)
            
            if (i - warmup + 1) % 10 == 0:
                logger.info(f"Progress: {i - warmup + 1}/{runs}, 最新レイテンシ: {latency:.2f}ms")
    
    # 統計を計算
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # 結果を返す
    result = {
        "model": "InReaCh-FO",
        "runs": runs,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "latencies": latencies
    }
    
    logger.info(f"InReaCh-FO平均レイテンシ: {mean_latency:.2f}ms")
    return result

def benchmark_samlad(device, runs=100, warmup=5, use_sam=False):
    """SAM-LADモデルのレイテンシを計測"""
    if use_sam:
        logger.info("SAM-LADモデルのベンチマーク開始（SAM使用）")
        # SAMモデルがない場合はダウンロードが必要なことを警告
        logger.warning("注意: SAM使用の場合、sam_checkpointパスにSAMモデルが存在する必要があります")
        logger.warning("SAMモデルは https://github.com/facebookresearch/segment-anything からダウンロードできます")
        
        # SAMモデルのパスを指定（存在する場合）
        sam_checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                     "models/weights/sam_vit_h_4b8939.pth")
        
        if not os.path.exists(sam_checkpoint):
            logger.error(f"SAMモデルが見つかりません: {sam_checkpoint}")
            logger.info("SAMなしでフォールバック実行します")
            sam_checkpoint = None
    else:
        logger.info("SAM-LADモデルのベンチマーク開始（SAMなし - 高速フォールバック）")
        sam_checkpoint = None
    
    # モデルの初期化
    model = SAMLAD(
        sam_checkpoint=sam_checkpoint,
        device=device,
        min_mask_area=100,
        max_objects=20
    )
    
    # テスト用のダミー入力を生成（画像形式 - uint8形式、値の範囲[0,255]）
    inputs = [(np.random.rand(224, 224, 3) * 255).astype(np.uint8) for _ in range(runs + warmup)]
    
    # ウォームアップ実行
    logger.info(f"{warmup}回のウォームアップ実行")
    for i in range(warmup):
        _ = model.process_image(inputs[i])
    
    # 計測開始
    logger.info(f"{runs}回の計測実行")
    latencies = []
    
    for i in range(warmup, runs + warmup):
        start_time = time.time()
        _ = model.process_image(inputs[i])
        latency = (time.time() - start_time) * 1000  # ミリ秒に変換
        latencies.append(latency)
        
        if (i - warmup + 1) % 10 == 0:
            logger.info(f"Progress: {i - warmup + 1}/{runs}, 最新レイテンシ: {latency:.2f}ms")
    
    # 詳細なタイミングメトリクスを取得
    metrics = model.get_metrics()
    
    # 統計を計算
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # 結果を返す
    result = {
        "model": "SAM-LAD",
        "runs": runs,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "latencies": latencies,
        "detailed_metrics": metrics
    }
    
    logger.info(f"SAM-LAD平均レイテンシ: {mean_latency:.2f}ms")
    return result

def benchmark_laft(device, runs=100, warmup=5):
    """LAFT + Phi-4-miniモデルのレイテンシを計測"""
    logger.info("LAFT + Phi-4-miniモデルのベンチマーク開始")
    
    # 基本モデルの初期化
    base_model = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    base_model.to(device)
    base_model.eval()
    
    # LAFTの初期化
    model = LAFTPhi4(
        clip_model_name="openai/clip-vit-base-patch16",
        phi_model_path=None,  # Phi-4-miniなしでテスト
        device=device,
        feature_dim=base_model.vit_feature_dim
    )
    
    # テスト用の指示を設定
    instruction = "傷を検出し、光の変化は無視してください"
    model.adjust_feature_space(instruction)
    
    # ランダム入力を生成
    inputs = [generate_random_image().to(device) for _ in range(runs + warmup)]
    
    # ウォームアップ実行
    logger.info(f"{warmup}回のウォームアップ実行")
    with torch.no_grad():
        for i in range(warmup):
            # 完全なパイプラインを実行 - TransFusionLiteのforward全体を実行
            _ = base_model(inputs[i])
            
            # 特徴量を取得して変換
            features = base_model.vit(inputs[i])
            _ = model.transform_features(features)
    
    # 計測開始
    logger.info(f"{runs}回の計測実行")
    latencies = []
    laft_only_latencies = []
    full_pipeline_latencies = []
    
    with torch.no_grad():
        for i in range(warmup, runs + warmup):
            # 1. 完全なパイプラインの測定 (通常のTransFusionLite推論)
            start_time = time.time()
            _ = base_model(inputs[i])
            pipeline_latency = (time.time() - start_time) * 1000  # ミリ秒に変換
            full_pipeline_latencies.append(pipeline_latency)
            
            # 2. LAFT変換を含む推論の測定
            start_time = time.time()
            # base_modelの特徴抽出
            features = base_model.vit(inputs[i])
            # LAFT変換
            transformed_features = model.transform_features(features)
            # 特徴量をリシェイプして残りの処理を行う
            features_2d = base_model._reshape_features(transformed_features)
            
            # TransFusionLiteのforwardメソッドの残りの部分を実行
            # Diffusion処理
            batch_size = features_2d.shape[0]
            latent = features_2d.clone()
            
            for j in range(base_model.n_steps):
                t = torch.tensor([j / base_model.n_steps], device=device).repeat(batch_size)
                with torch.cuda.amp.autocast(enabled=base_model.memory_efficient):
                    noise_pred = base_model.unet(latent, t).sample
                    
                noise_scale = base_model.sqrt_one_minus_alphas_cumprod[j]
                signal_scale = base_model.sqrt_alphas_cumprod[j]
                latent = signal_scale * features_2d + noise_scale * noise_pred
                
            # 異常スコア計算
            recon_error = torch.sum((features_2d - latent) ** 2, dim=1, keepdim=True)
            batch_min = recon_error.view(batch_size, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
            batch_max = recon_error.view(batch_size, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
            normalized_error = (recon_error - batch_min) / (batch_max - batch_min + 1e-8)
            
            # 画像全体のスコア計算
            score = base_model.scoring_mlp(transformed_features).squeeze(-1)
            
            # 完全なLAFT+TransFusionのレイテンシを記録
            latency = (time.time() - start_time) * 1000  # ミリ秒に変換
            latencies.append(latency)
            
            # 3. LAFTのみの計測（参考値として）
            start_time = time.time()
            features = base_model.vit(inputs[i])
            _ = model.transform_features(features)
            laft_only_latency = (time.time() - start_time) * 1000  # ミリ秒に変換
            laft_only_latencies.append(laft_only_latency)
            
            if (i - warmup + 1) % 10 == 0:
                logger.info(f"Progress: {i - warmup + 1}/{runs}, 完全パイプライン: {latency:.2f}ms, LAFTのみ: {laft_only_latency:.2f}ms")
    
    # 詳細なタイミングメトリクスを取得
    metrics = model.get_metrics()
    
    # 統計を計算
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    
    # 追加の統計
    mean_laft_only = np.mean(laft_only_latencies)
    mean_full_pipeline = np.mean(full_pipeline_latencies)
    
    # 結果を返す
    result = {
        "model": "LAFT+TransFusion",
        "runs": runs,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "latencies": latencies,
        "detailed_metrics": metrics,
        "additional_metrics": {
            "laft_only_mean_ms": mean_laft_only,
            "transfusion_only_mean_ms": mean_full_pipeline,
            "overhead_percent": ((mean_latency - mean_full_pipeline) / mean_full_pipeline) * 100
        }
    }
    
    logger.info(f"LAFT+TransFusion平均レイテンシ: {mean_latency:.2f}ms (LAFTのみ: {mean_laft_only:.2f}ms, TransFusionのみ: {mean_full_pipeline:.2f}ms)")
    logger.info(f"LAFTのオーバーヘッド: {((mean_latency - mean_full_pipeline) / mean_full_pipeline) * 100:.1f}%")
    return result

def save_results(results, output_dir):
    """結果をCSVとJSONに保存"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 基本的な統計をCSVに保存
    csv_path = os.path.join(output_dir, f"model_latency_comparison_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Mean (ms)', 'P50 (ms)', 'P95 (ms)', 'P99 (ms)', 'Min (ms)', 'Max (ms)'])
        
        for result in results:
            writer.writerow([
                result['model'],
                f"{result['mean_latency_ms']:.2f}",
                f"{result['p50_latency_ms']:.2f}",
                f"{result['p95_latency_ms']:.2f}",
                f"{result['p99_latency_ms']:.2f}",
                f"{result['min_latency_ms']:.2f}",
                f"{result['max_latency_ms']:.2f}"
            ])
    
    # 詳細な結果をJSONに保存
    json_path = os.path.join(output_dir, f"model_latency_detailed_{timestamp}.json")
    
    # latenciesが大きすぎる場合は、代表的な統計のみ保存
    for result in results:
        if 'latencies' in result:
            latencies = result['latencies']
            result['latencies'] = {
                'count': len(latencies),
                'mean': float(np.mean(latencies)),
                'std': float(np.std(latencies)),
                'min': float(np.min(latencies)),
                'p25': float(np.percentile(latencies, 25)),
                'p50': float(np.percentile(latencies, 50)),
                'p75': float(np.percentile(latencies, 75)),
                'p90': float(np.percentile(latencies, 90)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'max': float(np.max(latencies))
            }
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"結果を保存しました: {csv_path}, {json_path}")
    return csv_path, json_path

def create_latency_chart(results, output_dir):
    """レイテンシ比較チャートを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 棒グラフの作成
    plt.figure(figsize=(12, 8))
    
    # データの準備
    models = [r['model'] for r in results]
    mean_latencies = [r['mean_latency_ms'] for r in results]
    p95_latencies = [r['p95_latency_ms'] for r in results]
    
    # 実行環境情報の取得
    device_name = "MacBook Pro M4"
    device_info = f"Memory: 16GB, OS: macOS 14.5"
    runtime_info = f"Torch {torch.__version__}, ANE: Enabled"

    # 色の設定
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # バーの位置の計算
    x = np.arange(len(models))
    width = 0.35
    
    # 平均レイテンシのバーを描画
    bars1 = plt.bar(x - width/2, mean_latencies, width, color=colors, alpha=0.8, label='Mean Latency')
    
    # P95レイテンシのバーを描画
    bars2 = plt.bar(x + width/2, p95_latencies, width, color=colors, alpha=0.5, label='P95 Latency')
    
    # バーの上に値を表示
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean_latencies[i]:.1f}ms',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{p95_latencies[i]:.1f}ms',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    # グラフの装飾
    plt.title(f'Model Latency Comparison on {device_name}', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Latency (ms)', fontsize=14)
    plt.xticks(x, models, fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 閾値ラインの追加
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50ms Threshold')
    plt.text(len(models)-1, 52, '50ms (20 FPS)', color='r', fontsize=10)
    
    # 詳細情報を追加
    plt.figtext(0.01, 0.01, f"{device_info}\n{runtime_info}", fontsize=8)
    
    # グラフの保存
    chart_path = os.path.join(output_dir, f"model_latency_chart_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    logger.info(f"レイテンシチャートを保存しました: {chart_path}")
    
    # SVG形式でも保存
    svg_path = os.path.join(output_dir, f"model_latency_chart_{timestamp}.svg")
    plt.savefig(svg_path, format='svg')
    logger.info(f"SVG形式でも保存しました: {svg_path}")
    
    plt.close()
    return chart_path, svg_path

def run_benchmark(args):
    """ベンチマークを実行"""
    # デバイスの取得
    device = args.device if args.device else get_available_device()
    logger.info(f"使用デバイス: {device}")
    
    # 結果の保存先
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # ベンチマークの実行
    all_results = []
    
    if not args.skip_transfusion:
        result = benchmark_transfusion(device, runs=args.runs, warmup=args.warmup)
        all_results.append(result)
    
    if not args.skip_inreach:
        result = benchmark_inreach(device, runs=args.runs, warmup=args.warmup)
        all_results.append(result)
    
    if not args.skip_samlad:
        # SAMなしで実行（高速フォールバック）
        result = benchmark_samlad(device, runs=args.runs, warmup=args.warmup, use_sam=False)
        result["model"] = "SAM-LAD (フォールバック)"  # モデル名を明確化
        all_results.append(result)
        
        # SAMありで実行（指定された場合）
        if args.use_sam:
            result = benchmark_samlad(device, runs=args.runs, warmup=args.warmup, use_sam=True)
            result["model"] = "SAM-LAD (SAM使用)"  # モデル名を明確化
            all_results.append(result)
    
    if not args.skip_laft:
        result = benchmark_laft(device, runs=args.runs, warmup=args.warmup)
        all_results.append(result)
    
    # 結果の保存
    csv_path, json_path = save_results(all_results, output_dir)
    
    # チャートの作成
    chart_path, svg_path = create_latency_chart(all_results, output_dir)
    
    # 結果の出力
    logger.info("\n===== ベンチマーク結果 =====")
    for result in all_results:
        logger.info(f"{result['model']}: 平均 {result['mean_latency_ms']:.2f}ms, P95 {result['p95_latency_ms']:.2f}ms")
    
    logger.info(f"\n結果は以下に保存されました:")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"チャート: {chart_path}")
    logger.info(f"SVG: {svg_path}")

def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="モデル別レイテンシ比較ベンチマーク")
    parser.add_argument("--runs", type=int, default=100, help="各モデルの実行回数")
    parser.add_argument("--warmup", type=int, default=10, help="ウォームアップの実行回数")
    parser.add_argument("--device", type=str, default="", help="使用するデバイス (mps, cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="./experiments/m4_benchmarks/results", help="結果の保存先ディレクトリ")
    parser.add_argument("--skip-transfusion", action="store_true", help="TransFusion-Liteのベンチマークをスキップ")
    parser.add_argument("--skip-inreach", action="store_true", help="InReaCh-FOのベンチマークをスキップ")
    parser.add_argument("--skip-samlad", action="store_true", help="SAM-LADのベンチマークをスキップ")
    parser.add_argument("--skip-laft", action="store_true", help="LAFTのベンチマークをスキップ")
    parser.add_argument("--use-sam", action="store_true", help="SAMを使用したSAM-LADベンチマークも実行する（要SAMモデル）")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)