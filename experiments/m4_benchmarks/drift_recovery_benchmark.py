#!/usr/bin/env python3
"""
ドリフトリカバリ比較ベンチマーク

このスクリプトは、各モデル（TransFusion-Lite、InReaCh-FO、SAM-LAD、LAFT）の
ドメインシフト（光条件変化、傷の特性変化など）からの回復能力を評価します。
"""

import os
import sys
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import cv2

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# モデルのインポート
from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.inreach.model import InReachFO
from src.models.samlad.model import SAMLAD
from src.models.laft.model import LAFTPhi4
from src.models.transfusion.utils import get_available_device, load_model_checkpoint
# Using existing test videos instead of generating new ones
# from src.data.generate_test_video import generate_synthetic_video

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("drift_recovery_benchmark")

def load_test_video(drift_type="lighting", test_videos_dir=None):
    """テスト動画をロードしてシーケンスとして返す"""
    if test_videos_dir is None:
        test_videos_dir = Path("data/test_videos")
    
    # 利用可能なテスト動画を探す
    if drift_type == "lighting":
        video_path = test_videos_dir / "synthetic_grid_lighting_i0.5_gradual.avi"
    elif drift_type == "scratch":
        video_path = test_videos_dir / "synthetic_grid_scratch_i1.0.avi"
    else:
        # デフォルトの動画
        video_paths = list(test_videos_dir.glob("*.avi"))
        if not video_paths:
            raise ValueError(f"テスト動画が見つかりません: {test_videos_dir}")
        video_path = video_paths[0]
        if "mask" in str(video_path):
            # maskファイル以外を選択
            for path in video_paths:
                if "mask" not in str(path):
                    video_path = path
                    break
    
    logger.info(f"テスト動画をロード中: {video_path}")
    
    if not video_path.exists():
        raise FileNotFoundError(f"テスト動画が見つかりません: {video_path}")
    
    # 動画をフレームに分解
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"動画を開けませんでした: {video_path}")
    
    sequence = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 0-1に正規化
        frame = frame.astype(np.float32) / 255.0
        
        sequence.append(frame)
    
    cap.release()
    logger.info(f"ロード完了: {len(sequence)}フレーム")
    return sequence


def load_anomaly_mask(sequence, drift_type="lighting", start_frame=30, anomaly_intensity=1.0):
    """シーケンスに対応するマスクをロードまたは生成"""
    # まずマスク動画が存在するか確認
    test_videos_dir = Path("data/test_videos")
    mask_path = None
    
    if drift_type == "scratch":
        possible_mask_path = test_videos_dir / "synthetic_grid_scratch_i1.0_mask.avi"
        if possible_mask_path.exists():
            mask_path = possible_mask_path
    
    if mask_path:
        logger.info(f"マスク動画をロード中: {mask_path}")
        cap = cv2.VideoCapture(str(mask_path))
        if cap.isOpened():
            masks = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # グレースケールに変換
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 0-1に正規化
                mask = frame.astype(np.float32) / 255.0
                masks.append(mask)
            cap.release()
            logger.info(f"マスクロード完了: {len(masks)}フレーム")
            
            # シーケンスとマスクの長さを合わせる
            if len(masks) != len(sequence):
                logger.warning(f"マスクとシーケンスの長さが一致しません: {len(masks)} vs {len(sequence)}")
                if len(masks) > len(sequence):
                    masks = masks[:len(sequence)]
                else:
                    # 足りない分は最後のマスクを繰り返す
                    last_mask = masks[-1]
                    while len(masks) < len(sequence):
                        masks.append(last_mask.copy())
            return masks
    
    # マスク動画がない場合は生成
    logger.info(f"マスクを生成します（マスク動画が見つかりませんでした）")
    # マスクを生成して返す
    return create_anomaly_mask(sequence, drift_type, start_frame, anomaly_intensity)

def create_anomaly_mask(sequence, drift_type="lighting", start_frame=30, anomaly_intensity=1.0):
    """異常マスクの生成（真のアノマリー位置）"""
    masks = []
    
    # 基本設定
    image_size = sequence[0].shape[:2]
    
    for t in range(len(sequence)):
        # 初期フレームは異常なし
        if t < start_frame:
            mask = np.zeros(image_size, dtype=np.float32)
        else:
            # 異常の種類に基づいてマスク生成
            mask = np.zeros(image_size, dtype=np.float32)
            
            if drift_type == "scratch":
                # 傷の異常（中央に傷）
                center_x, center_y = image_size[1] // 2, image_size[0] // 2
                length = int(min(image_size) * 0.4 * anomaly_intensity)
                thickness = int(3 * anomaly_intensity)
                
                # 線を描画
                cv2.line(mask, 
                         (center_x - length//2, center_y - length//2),
                         (center_x + length//2, center_y + length//2),
                         1.0, thickness)
            
            elif drift_type == "defects":
                # 欠陥の異常（一定のパターン）
                # 中央付近に円形の欠陥
                center_x, center_y = image_size[1] // 2, image_size[0] // 2
                radius = int(min(image_size) * 0.1 * anomaly_intensity)
                cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
            
            else:
                # その他のドリフトタイプでは特定の位置に異常を生成
                x1, y1 = int(image_size[1] * 0.3), int(image_size[0] * 0.3)
                x2, y2 = int(image_size[1] * 0.7), int(image_size[0] * 0.7)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1.0, 2)
        
        masks.append(mask)
    
    return masks


def evaluate_drift_recovery(model_name, sequence, masks, device, anomaly_start_frame=30):
    """ドリフトリカバリの評価"""
    logger.info(f"{model_name}モデルのドリフトリカバリ評価開始")
    
    # モデルの初期化
    if model_name == "TransFusion-Lite":
        # 標準TransFusion-Lite（適応なし）
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
        
        # 評価関数
        def process_frame(frame):
            # numpy -> torch
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # リサイズ（ViTモデルは224x224を期待）
            if frame_tensor.shape[2] != 224 or frame_tensor.shape[3] != 224:
                frame_tensor = F.interpolate(
                    frame_tensor, 
                    size=(224, 224), 
                    mode='bilinear',
                    align_corners=False
                )
            
            with torch.no_grad():
                output = processor.process(frame_tensor)
                
            anomaly_score = output["scores"].cpu().numpy().item()
            anomaly_map = output["anomaly_maps"].cpu().numpy().squeeze()
            anomaly_map = anomaly_map / anomaly_map.max()  # 0-1に正規化
            
            return anomaly_score, anomaly_map
    
    elif model_name == "InReaCh-FO":
        # InReaCh-FO（オンライン適応あり）
        base_model = TransFusionLite(
            backbone="vit_base_patch16_224",
            pretrained=True,
            n_steps=4,
            device=device
        )
        base_model.to(device)
        base_model.eval()
        
        # プロセッサの初期化
        processor = TransFusionProcessor(base_model, threshold=2.0, device=device)
        
        # InReaCh-FOの初期化
        inreach = InReachFO(
            base_model=base_model,
            confidence_thresh=0.7,
            adaptation_strategy="batch_norm",
            device=device
        )
        
        # 評価関数
        def process_frame(frame):
            # numpy -> torch
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # リサイズ（ViTモデルは224x224を期待）
            if frame_tensor.shape[2] != 224 or frame_tensor.shape[3] != 224:
                frame_tensor = F.interpolate(
                    frame_tensor, 
                    size=(224, 224), 
                    mode='bilinear',
                    align_corners=False
                )
            
            with torch.no_grad():
                # 特徴抽出
                features = base_model.vit(frame_tensor)
                
                # 通常推論
                output = processor.process(frame_tensor)
                
                # 適応処理（異常度が低い場合のみ）
                anomaly_score = output["scores"].cpu().numpy().item()
                if anomaly_score < 1.5:  # 異常度が低い場合のみ適応
                    inreach.adapt(frame_tensor, output)
                
            anomaly_map = output["anomaly_maps"].cpu().numpy().squeeze()
            anomaly_map = anomaly_map / anomaly_map.max()  # 0-1に正規化
            
            return anomaly_score, anomaly_map
    
    elif model_name == "SAM-LAD":
        # SAM-LAD
        model = SAMLAD(
            sam_checkpoint=None,  # SAMなしでテスト
            device=device,
            min_mask_area=100,
            max_objects=20,
            use_tracking=True  # トラッキングを有効化
        )
        
        # 評価関数
        def process_frame(frame):
            # すでに0-1の範囲のnumpy配列
            result = model.process_image(frame)
            
            if "anomaly_score" in result:
                anomaly_score = result["anomaly_score"]
            else:
                anomaly_score = result.get("anomaly_probability", 0.0)
                
            if "anomaly_map" in result:
                anomaly_map = result["anomaly_map"]
                if anomaly_map is not None:
                    anomaly_map = anomaly_map / (anomaly_map.max() + 1e-8)  # 0-1に正規化
                else:
                    anomaly_map = np.zeros(frame.shape[:2], dtype=np.float32)
            else:
                anomaly_map = np.zeros(frame.shape[:2], dtype=np.float32)
                
            return anomaly_score, anomaly_map
    
    elif model_name == "LAFT":
        # LAFT + Phi-4-mini
        base_model = TransFusionLite(
            backbone="vit_base_patch16_224",
            pretrained=True,
            n_steps=4,
            device=device
        )
        base_model.to(device)
        base_model.eval()
        
        # LAFTの初期化
        laft = LAFTPhi4(
            clip_model_name="openai/clip-vit-base-patch16",
            phi_model_path=None,  # Phi-4-miniなしでテスト
            device=device,
            feature_dim=base_model.vit_feature_dim
        )
        
        # 指示の設定
        instruction = "表面の傷や欠けを検出し、照明の変化や色の変動は無視してください"
        laft.adjust_feature_space(instruction)
        logger.info(f"LAFT指示: {instruction}")
        
        # 評価関数
        def process_frame(frame):
            # numpy -> torch
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 特徴抽出
                features = base_model.vit(frame_tensor)
                
                # LAFT変換
                transformed_features = laft.transform_features(features)
                
                # 変換された特徴を使って残りの処理を実行
                output = base_model.process_features(transformed_features)
                
            anomaly_score = output["anomaly_score"].cpu().numpy().item()
            anomaly_map = output["anomaly_map"].cpu().numpy().squeeze()
            anomaly_map = anomaly_map / anomaly_map.max()  # 0-1に正規化
            
            return anomaly_score, anomaly_map
    
    else:
        raise ValueError(f"サポートされていないモデル: {model_name}")
    
    # シーケンス全体の評価
    scores = []
    precision_history = []
    recall_history = []
    f1_history = []
    
    # 各フレームの評価
    for i, (frame, mask) in enumerate(zip(sequence, masks)):
        # フレーム処理
        anomaly_score, anomaly_map = process_frame(frame)
        scores.append(anomaly_score)
        
        # ピクセルレベルの評価（異常検出の精度）
        if np.max(mask) > 0:
            # 異常マップをバイナリマスクに変換
            threshold = 0.5
            anomaly_binary = (anomaly_map > threshold).astype(float)
            
            # マスクのサイズを異常マップに合わせる
            if mask.shape != anomaly_map.shape:
                logger.warning(f"異常マップとマスクのサイズが一致しません: {anomaly_map.shape} vs {mask.shape}")
                import cv2
                mask = cv2.resize(mask, (anomaly_map.shape[1], anomaly_map.shape[0]))
            
            # 真陽性、偽陽性、偽陰性
            tp = np.sum((anomaly_binary == 1) & (mask > 0))
            fp = np.sum((anomaly_binary == 1) & (mask == 0))
            fn = np.sum((anomaly_binary == 0) & (mask > 0))
            
            # 精度、再現率、F1スコア
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            # 異常がない場合（評価不可）
            precision = float('nan')
            recall = float('nan')
            f1 = float('nan')
        
        precision_history.append(precision)
        recall_history.append(recall)
        f1_history.append(f1)
        
        # 進捗
        if i % 10 == 0:
            logger.debug(f"フレーム {i+1}/{len(sequence)} 処理完了")
    
    # 結果のまとめ
    # 異常開始前後の区間を定義
    pre_anomaly = slice(0, anomaly_start_frame)
    post_anomaly = slice(anomaly_start_frame, len(sequence))
    
    # 区間ごとのスコア統計
    pre_anomaly_scores = np.array(scores)[pre_anomaly]
    post_anomaly_scores = np.array(scores)[post_anomaly]
    
    pre_anomaly_mean = np.mean(pre_anomaly_scores)
    post_anomaly_mean = np.mean(post_anomaly_scores)
    
    # F1スコアの時間変化
    f1_values = np.array(f1_history)[post_anomaly]
    f1_values = f1_values[~np.isnan(f1_values)]  # NaNを除外
    mean_f1 = np.mean(f1_values) if len(f1_values) > 0 else 0.0
    
    # 異常検出の安定性（スコアの標準偏差）
    score_stability = np.std(scores)
    
    # 適応速度の計算
    # 1. 異常開始後のスコア変化率
    anomaly_response_time = 0
    for i in range(anomaly_start_frame, len(scores)):
        if scores[i] > 1.5 * pre_anomaly_mean:  # 異常として検出する閾値
            anomaly_response_time = i - anomaly_start_frame
            break
    
    # 総合結果
    result = {
        "model": model_name,
        "scores": scores,
        "pre_anomaly_mean": float(pre_anomaly_mean),
        "post_anomaly_mean": float(post_anomaly_mean),
        "anomaly_response_time": anomaly_response_time,
        "mean_f1_score": float(mean_f1),
        "score_stability": float(score_stability),
        "precision_history": precision_history,
        "recall_history": recall_history,
        "f1_history": f1_history
    }
    
    # 要約
    detection_ratio = post_anomaly_mean / pre_anomaly_mean
    logger.info(f"{model_name} 結果:")
    logger.info(f"  異常前平均スコア: {pre_anomaly_mean:.4f}")
    logger.info(f"  異常後平均スコア: {post_anomaly_mean:.4f}")
    logger.info(f"  検出比率: {detection_ratio:.4f}")
    logger.info(f"  異常反応時間: {anomaly_response_time}フレーム")
    logger.info(f"  平均F1スコア: {mean_f1:.4f}")
    
    return result


def save_results(results, output_dir, drift_type):
    """結果をCSVとJSONに保存"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 基本的な統計をCSVに保存
    csv_path = os.path.join(output_dir, f"drift_recovery_{drift_type}_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model',
            'Pre-Anomaly Score',
            'Post-Anomaly Score',
            'Detection Ratio',
            'Response Time',
            'Mean F1 Score',
            'Score Stability'
        ])
        
        for result in results:
            model_name = result['model']
            pre_score = result['pre_anomaly_mean']
            post_score = result['post_anomaly_mean']
            detection_ratio = post_score / pre_score if pre_score > 0 else float('inf')
            
            writer.writerow([
                model_name,
                f"{pre_score:.4f}",
                f"{post_score:.4f}",
                f"{detection_ratio:.4f}",
                f"{result['anomaly_response_time']}",
                f"{result['mean_f1_score']:.4f}",
                f"{result['score_stability']:.4f}"
            ])
    
    # 詳細な結果をJSONに保存
    json_path = os.path.join(output_dir, f"drift_recovery_detailed_{drift_type}_{timestamp}.json")
    
    # スコア履歴がサイズが大きい場合は間引く
    processed_results = []
    for result in results:
        processed_result = result.copy()
        
        # スコアの間引き
        if len(result['scores']) > 100:
            indices = np.linspace(0, len(result['scores'])-1, 100, dtype=int)
            processed_result['scores'] = [float(result['scores'][i]) for i in indices]
        else:
            processed_result['scores'] = [float(score) for score in result['scores']]
            
        # 履歴の間引き
        for key in ['precision_history', 'recall_history', 'f1_history']:
            if len(result[key]) > 100:
                indices = np.linspace(0, len(result[key])-1, 100, dtype=int)
                processed_result[key] = [float(result[key][i]) if not np.isnan(result[key][i]) else None for i in indices]
            else:
                processed_result[key] = [float(val) if not np.isnan(val) else None for val in result[key]]
        
        processed_results.append(processed_result)
    
    with open(json_path, 'w') as f:
        json.dump(processed_results, f, indent=2)
    
    logger.info(f"結果を保存しました: {csv_path}, {json_path}")
    return csv_path, json_path


def create_drift_recovery_chart(results, output_dir, drift_type):
    """ドリフトリカバリ比較チャートを作成"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 12))
    
    # 実行環境情報の取得
    device_name = "MacBook Pro M4"
    device_info = f"Memory: 16GB, OS: macOS 14.5"
    runtime_info = f"Torch {torch.__version__}, ANE: Enabled"
    
    # モデル名と色
    models = [r['model'] for r in results]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # 1. スコアの時間推移
    plt.subplot(2, 2, 1)
    
    for i, result in enumerate(results):
        scores = result['scores']
        x = np.arange(len(scores))
        plt.plot(x, scores, label=result['model'], color=colors[i % len(colors)])
        
        # 異常開始点
        anomaly_start_frame = 30  # 仮定
        plt.axvline(x=anomaly_start_frame, color='gray', linestyle='--', alpha=0.7)
        plt.text(anomaly_start_frame + 1, max(scores) * 0.9, "Anomaly Start", 
                 fontsize=9, color='gray')
    
    plt.title('Anomaly Scores Over Time', fontsize=14)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. 検出性能比較（棒グラフ）
    plt.subplot(2, 2, 2)
    
    # データ準備
    x = np.arange(len(models))
    width = 0.4
    
    detection_ratios = [r['post_anomaly_mean'] / r['pre_anomaly_mean'] if r['pre_anomaly_mean'] > 0 else 10 
                        for r in results]
    f1_scores = [r['mean_f1_score'] for r in results]
    
    # 検出比率の棒グラフ
    bars1 = plt.bar(x - width/2, detection_ratios, width, label='Detection Ratio', 
                    color=[colors[i % len(colors)] for i in range(len(models))])
    
    # F1スコアの棒グラフ
    bars2 = plt.bar(x + width/2, f1_scores, width, label='F1 Score',
                   alpha=0.7, color=[colors[i % len(colors)] for i in range(len(models))])
    
    # バーの上に値を表示
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{detection_ratios[i]:.1f}',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{f1_scores[i]:.2f}',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    plt.title('Detection Performance', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(x, models)
    plt.ylim(0, max(max(detection_ratios), max(f1_scores)) * 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. 反応時間と安定性
    plt.subplot(2, 2, 3)
    
    response_times = [r['anomaly_response_time'] for r in results]
    stabilities = [r['score_stability'] for r in results]
    
    # 散布図
    for i, model in enumerate(models):
        plt.scatter(response_times[i], stabilities[i], color=colors[i % len(colors)], 
                   s=100, label=model)
        plt.text(response_times[i] + 0.5, stabilities[i] + 0.02,
                model, fontsize=10)
    
    plt.title('Response Time vs. Stability', fontsize=14)
    plt.xlabel('Response Time (frames)', fontsize=12)
    plt.ylabel('Score Stability (std)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 4. F1スコアの推移
    plt.subplot(2, 2, 4)
    
    for i, result in enumerate(results):
        f1_history = np.array(result['f1_history'])
        
        # NaNを除外して平均化
        valid_indices = ~np.isnan(f1_history)
        x_valid = np.arange(len(f1_history))[valid_indices]
        f1_valid = f1_history[valid_indices]
        
        if len(f1_valid) > 0:
            plt.plot(x_valid, f1_valid, label=result['model'], color=colors[i % len(colors)])
        
    plt.title('F1 Score Over Time', fontsize=14)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # タイトル
    plt.suptitle(f'Drift Recovery Comparison: {drift_type.capitalize()} Drift', fontsize=16)
    
    # 詳細情報を追加
    plt.figtext(0.01, 0.01, f"{device_info}\n{runtime_info}", fontsize=8)
    
    # グラフの保存
    chart_path = os.path.join(output_dir, f"drift_recovery_chart_{drift_type}_{timestamp}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(chart_path, dpi=300)
    logger.info(f"ドリフトリカバリチャートを保存しました: {chart_path}")
    
    # SVG形式でも保存
    svg_path = os.path.join(output_dir, f"drift_recovery_chart_{drift_type}_{timestamp}.svg")
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
    
    # ドリフトシーケンスのロード
    sequence = load_test_video(
        drift_type=args.drift_type
    )
    
    # 異常マスクのロードまたは生成
    try:
        masks = load_anomaly_mask(
            sequence,
            drift_type=args.drift_type,
            start_frame=args.anomaly_start,
            anomaly_intensity=args.anomaly_intensity
        )
    except:
        # マスクのロードに失敗した場合は生成
        masks = create_anomaly_mask(
            sequence,
            drift_type=args.drift_type,
            start_frame=args.anomaly_start,
            anomaly_intensity=args.anomaly_intensity
        )
    
    logger.info(f"ドリフトシーケンス生成完了（{len(sequence)}フレーム）")
    
    # ビデオとして保存（オプション）
    if args.save_video:
        video_path = os.path.join(output_dir, f"drift_sequence_{args.drift_type}.avi")
        
        # 0-1の範囲から0-255の範囲に変換
        sequence_uint8 = [(frame * 255).astype(np.uint8) for frame in sequence]
        
        # OpenCVでビデオ書き込み
        frame_size = sequence[0].shape[:2][::-1]  # (width, height)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, frame_size)
        
        for frame in sequence_uint8:
            # RGB -> BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
            
        out.release()
        logger.info(f"シーケンスをビデオとして保存: {video_path}")
        
        # マスクビデオも保存
        mask_video_path = os.path.join(output_dir, f"drift_mask_{args.drift_type}.avi")
        out = cv2.VideoWriter(mask_video_path, fourcc, 10.0, frame_size)
        
        for mask in masks:
            # 0-1の範囲から0-255の範囲に変換
            mask_frame = (mask * 255).astype(np.uint8)
            
            # グレースケールをBGRに変換
            mask_bgr = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
            out.write(mask_bgr)
            
        out.release()
        logger.info(f"マスクをビデオとして保存: {mask_video_path}")
    
    # ベンチマークの実行
    all_results = []
    
    if not args.skip_transfusion:
        result = evaluate_drift_recovery("TransFusion-Lite", sequence, masks, device, args.anomaly_start)
        all_results.append(result)
    
    if not args.skip_inreach:
        result = evaluate_drift_recovery("InReaCh-FO", sequence, masks, device, args.anomaly_start)
        all_results.append(result)
    
    if not args.skip_samlad:
        result = evaluate_drift_recovery("SAM-LAD", sequence, masks, device, args.anomaly_start)
        all_results.append(result)
    
    if not args.skip_laft:
        result = evaluate_drift_recovery("LAFT", sequence, masks, device, args.anomaly_start)
        all_results.append(result)
    
    # 結果の保存
    csv_path, json_path = save_results(all_results, output_dir, args.drift_type)
    
    # チャートの作成
    chart_path, svg_path = create_drift_recovery_chart(all_results, output_dir, args.drift_type)
    
    # 結果の出力
    logger.info("\n===== ベンチマーク結果 =====")
    for result in all_results:
        detection_ratio = result['post_anomaly_mean'] / result['pre_anomaly_mean']
        logger.info(f"{result['model']}: 検出比 {detection_ratio:.2f}, F1 {result['mean_f1_score']:.3f}")
    
    logger.info(f"\n結果は以下に保存されました:")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"JSON: {json_path}")
    logger.info(f"チャート: {chart_path}")
    logger.info(f"SVG: {svg_path}")


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="ドリフトリカバリ比較ベンチマーク")
    parser.add_argument("--drift-type", type=str, default="lighting",
                        choices=["lighting", "scratch", "texture", "defects"],
                        help="評価するドリフトの種類")
    parser.add_argument("--duration", type=int, default=100,
                        help="シーケンスの長さ（フレーム数）")
    parser.add_argument("--intensity", type=float, default=0.5,
                        help="ドリフト強度（0.0-1.0）")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="ノイズレベル（0.0-1.0）")
    parser.add_argument("--anomaly-start", type=int, default=30,
                        help="異常開始フレーム")
    parser.add_argument("--anomaly-intensity", type=float, default=1.0,
                        help="異常強度（0.0-1.0）")
    parser.add_argument("--save-video", action="store_true",
                        help="ドリフトシーケンスをビデオとして保存")
    parser.add_argument("--device", type=str, default="",
                        help="使用するデバイス (mps, cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="./experiments/m4_benchmarks/results",
                        help="結果の保存先ディレクトリ")
    parser.add_argument("--skip-transfusion", action="store_true",
                        help="TransFusion-Liteの評価をスキップ")
    parser.add_argument("--skip-inreach", action="store_true",
                        help="InReaCh-FOの評価をスキップ")
    parser.add_argument("--skip-samlad", action="store_true",
                        help="SAM-LADの評価をスキップ")
    parser.add_argument("--skip-laft", action="store_true",
                        help="LAFTの評価をスキップ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)