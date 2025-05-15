#!/usr/bin/env python3
"""
LAFTの言語指示ベース適応評価ベンチマーク

このスクリプトは、LAFT + Phi-4-miniモデルの言語指示に基づく適応能力を
異なる指示パターンで評価し、結果を比較します。
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from tqdm import tqdm
import cv2

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# モデルのインポート
from src.models.transfusion.model import TransFusionLite, TransFusionProcessor
from src.models.laft.model import LAFTPhi4
from src.models.transfusion.utils import get_available_device, load_model_checkpoint
try:
    from src.data.download import get_dataset_path
except ImportError:
    # Fallback to alternative implementation
    def get_dataset_path(dataset_name):
        return Path(f"data/{dataset_name}")

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("laft_instruction_benchmark")


class MVTecDataset:
    """MVTec AD2データセットローダー"""
    
    def __init__(self, root_path=None):
        """
        MVTec AD2データセットの初期化
        
        Args:
            root_path: データセットのルートパス（Noneの場合はデフォルトパスを使用）
        """
        self.dataset_name = "mvtec_ad2"
        self.root_path = root_path or get_dataset_path(self.dataset_name)
        
        # データセットの検証
        if not os.path.exists(self.root_path):
            raise ValueError(f"データセットパスが存在しません: {self.root_path}")
            
        logger.info(f"MVTec AD2データセットを読み込み中: {self.root_path}")
        
        # 画像サイズ設定
        self.image_size = (224, 224)
        
        # MVTec AD2のカテゴリとその異常タイプ
        self.categories = {
            "bottle": ["broken_large", "broken_small", "contamination"],
            "cable": ["bent_wire", "cable_swap", "cut_inner_insulation", "cut_outer_insulation", "missing_cable", "missing_wire", "poke_insulation"],
            "capsule": ["crack", "faulty_imprint", "poke", "scratch", "squeeze"],
            "carpet": ["color", "cut", "hole", "metal_contamination", "thread"],
            "grid": ["bent", "broken", "glue", "metal_contamination", "thread"],
            "hazelnut": ["crack", "cut", "hole", "print"],
            "leather": ["color", "cut", "fold", "glue", "poke"],
            "metal_nut": ["bent", "color", "flip", "scratch"],
            "pill": ["color", "combined", "contamination", "crack", "faulty_imprint", "pill_type", "scratch"],
            "screw": ["crossed_thread", "manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"],
            "tile": ["crack", "glue_strip", "gray_stroke", "oil", "rough"],
            "toothbrush": ["defective"],
            "transistor": ["bent_lead", "cut_lead", "damaged_case", "misplaced"],
            "wood": ["color", "combined", "hole", "liquid", "scratch"],
            "zipper": ["broken_teeth", "combined", "fabric_border", "fabric_interior", "split_teeth", "squeezed_teeth"]
        }
        
        # 存在するカテゴリの検証
        self.valid_categories = []
        for category in self.categories.keys():
            category_path = os.path.join(self.root_path, category)
            if os.path.exists(category_path):
                self.valid_categories.append(category)
                
        if not self.valid_categories:
            raise ValueError(f"有効なカテゴリが見つかりません: {self.root_path}")
            
        logger.info(f"有効なカテゴリ: {', '.join(self.valid_categories)}")
    
    def get_category_path(self, category):
        """カテゴリのパスを取得"""
        if category not in self.valid_categories:
            raise ValueError(f"カテゴリ {category} は利用できません")
            
        return os.path.join(self.root_path, category)
    
    def get_normal_images(self, category, split="test", max_images=None):
        """正常画像のパスリストを取得"""
        category_path = self.get_category_path(category)
        normal_dir = os.path.join(category_path, split, "good")
        
        if not os.path.exists(normal_dir):
            logger.warning(f"正常画像ディレクトリが見つかりません: {normal_dir}")
            return []
        
        # PNG画像のみ取得
        normal_images = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith(".png")]
        
        # 最大数の制限（必要な場合）
        if max_images is not None and len(normal_images) > max_images:
            normal_images = normal_images[:max_images]
            
        return normal_images
    
    def get_anomaly_images(self, category, anomaly_type=None, max_images=None):
        """異常画像のパスリストを取得"""
        category_path = self.get_category_path(category)
        test_path = os.path.join(category_path, "test")
        
        if not os.path.exists(test_path):
            logger.warning(f"テストディレクトリが見つかりません: {test_path}")
            return [], []
        
        # 異常タイプの選択
        if anomaly_type:
            if anomaly_type not in self.categories[category]:
                logger.warning(f"異常タイプ {anomaly_type} は {category} カテゴリに存在しません")
                return [], []
            anomaly_types = [anomaly_type]
        else:
            anomaly_types = [d for d in os.listdir(test_path) 
                           if os.path.isdir(os.path.join(test_path, d)) and d != "good"]
        
        # 画像とマスクのパスのリスト
        image_paths = []
        mask_paths = []
        
        for atype in anomaly_types:
            anomaly_dir = os.path.join(test_path, atype)
            if not os.path.exists(anomaly_dir):
                continue
                
            # PNG画像のみ取得
            anomaly_images = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir) if f.endswith(".png")]
            
            # 対応するマスクを取得
            for img_path in anomaly_images:
                img_name = os.path.basename(img_path)
                mask_dir = os.path.join(category_path, "ground_truth", atype)
                mask_path = os.path.join(mask_dir, img_name)
                
                if os.path.exists(mask_path):
                    image_paths.append(img_path)
                    mask_paths.append(mask_path)
        
        # 最大数の制限（必要な場合）
        if max_images is not None and len(image_paths) > max_images:
            # 同じインデックスを使用
            indices = list(range(len(image_paths)))[:max_images]
            image_paths = [image_paths[i] for i in indices]
            mask_paths = [mask_paths[i] for i in indices]
            
        return image_paths, mask_paths
    
    def load_image(self, image_path):
        """画像の読み込み"""
        import cv2
        
        if not os.path.exists(image_path):
            raise ValueError(f"画像パスが存在しません: {image_path}")
            
        # 画像読み込み
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
            
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # リサイズ（必要な場合）
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size)
            
        # 正規化（0-1の範囲に）
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def load_mask(self, mask_path):
        """マスクの読み込み"""
        import cv2
        
        if not os.path.exists(mask_path):
            raise ValueError(f"マスクパスが存在しません: {mask_path}")
            
        # マスク読み込み
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"マスクの読み込みに失敗しました: {mask_path}")
            
        # リサイズ（必要な場合）
        if mask.shape != self.image_size:
            mask = cv2.resize(mask, self.image_size)
            
        # 正規化（0-1の範囲に）
        mask = mask.astype(np.float32) / 255.0
        
        return mask
    
    def preprocess_for_model(self, image):
        """モデル用の前処理"""
        # numpy -> torch
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        
        # すでに0-1の範囲なのでそのまま返す
        return image_tensor


def generate_instructions():
    """様々な指示パターンを生成"""
    # 基本指示
    basic_instructions = [
        "異常を検出する",
        "製品の欠陥を見つける",
        "損傷や不良を検出する",
        "表面の異常を検出する",
        "不良品を識別する"
    ]
    
    # 詳細指示
    detailed_instructions = [
        "表面の傷や割れを検出し、色の変化は無視してください",
        "製品の物理的な欠陥を検出し、照明の変化による影響は無視してください",
        "重要な部品の欠損を検出し、向きの違いや位置のずれは許容してください",
        "パターンの乱れや不整合を検出し、自然な表面の質感の違いは無視してください",
        "構造的な損傷を検出し、光の反射や影は無視してください"
    ]
    
    # カテゴリ固有の指示
    category_specific_instructions = {
        "bottle": "ボトルの表面の傷や欠陥を検出し、ラベルの問題は無視してください",
        "cable": "ケーブルの被覆の損傷や露出した導線を検出し、曲がりは無視してください",
        "capsule": "カプセルの欠陥や変形を検出し、色の変化は無視してください",
        "carpet": "カーペットの汚れや織りの不良を検出し、パターンの変化は無視してください",
        "grid": "グリッドの破損や欠けを検出し、光の反射は無視してください",
        "hazelnut": "ヘーゼルナッツの傷や腐敗を検出し、大きさの違いは無視してください",
        "leather": "レザーの傷や染みを検出し、自然な質感の違いは無視してください",
        "metal_nut": "ナットの欠けや変形を検出し、錆や色の変化は無視してください",
        "pill": "錠剤の欠けや変色を検出し、向きの違いは無視してください",
        "screw": "ねじの変形や破損を検出し、表面の光沢は無視してください",
        "tile": "タイルの亀裂や欠けを検出し、パターンの違いは無視してください",
        "toothbrush": "歯ブラシの毛の異常を検出し、色の違いは無視してください",
        "transistor": "トランジスタの破損やはんだ不良を検出し、向きは無視してください",
        "wood": "木材の傷や節を検出し、木目の自然な変化は無視してください",
        "zipper": "ジッパーの歯の欠損や変形を検出し、金属の光沢は無視してください"
    }
    
    # 欠陥タイプ固有の指示
    defect_specific_instructions = {
        "scratches": "表面の傷だけを検出し、他の異常は無視してください",
        "dents": "へこみや凹みだけを検出し、表面の傷は無視してください",
        "discoloration": "色の変化や変色だけを検出し、物理的な損傷は無視してください",
        "missing_parts": "欠けた部品や欠損だけを検出し、表面の異常は無視してください",
        "cracks": "割れや亀裂だけを検出し、色の変化や汚れは無視してください",
        "contamination": "汚れや異物だけを検出し、製品自体の異常は無視してください"
    }
    
    # 特定の状況に対する特化指示
    contextual_instructions = [
        "照明変化の下での異常検出: 光の状態が変わっても欠陥を検出し、影や反射は無視",
        "回転・向きの影響を排除: 製品の向きや角度に関わらず構造的欠陥を検出",
        "背景の変化に影響されない検出: 製品自体の異常のみを検出し、背景や周囲の変化は無視",
        "表面テクスチャの自然な変動を許容: 製造上の重大な欠陥のみを検出し、自然な変化は無視",
        "特定の部位に焦点: 製品の重要部分のみを詳細に検査し、その他の領域は無視"
    ]
    
    # 矛盾する指示（テスト用）
    contradictory_instructions = [
        "すべての異常を検出する一方で、誤検出を絶対に避けてください",
        "表面の傷を検出しつつも、表面の変化はすべて無視してください",
        "色の変化を検出してください、ただし色の違いは無視してください",
        "製品の欠陥をすべて検出してください、ただし細かい欠陥は無視してください",
        "高精度で異常を検出しつつ、疑わしいケースはすべて正常と判断してください"
    ]
    
    # すべての指示をまとめる
    all_instructions = {
        "basic": basic_instructions,
        "detailed": detailed_instructions,
        "category_specific": category_specific_instructions,
        "defect_specific": defect_specific_instructions,
        "contextual": contextual_instructions,
        "contradictory": contradictory_instructions
    }
    
    return all_instructions


def evaluate_instruction_accuracy(base_model, laft_model, dataset, category, instruction, device, max_images=50):
    """特定の指示の精度評価"""
    logger.info(f"カテゴリ {category} で指示を評価: {instruction}")
    
    # LAFT特徴空間の調整
    start_time = time.time()
    laft_model.adjust_feature_space(instruction)
    adjustment_time = time.time() - start_time
    
    # トランスフォーメーションマトリックスの取得
    projection_matrix = laft_model.projection.weight.detach()
    
    # 正常・異常画像の取得
    normal_images = dataset.get_normal_images(category, max_images=max_images//2)
    anomaly_image_paths, anomaly_mask_paths = dataset.get_anomaly_images(category, max_images=max_images//2)
    
    # 結果保存用
    normal_scores = []
    anomaly_scores = []
    anomaly_maps = []
    anomaly_masks = []
    
    # 正常画像の評価
    for img_path in normal_images:
        # 画像読み込み
        image = dataset.load_image(img_path)
        image_tensor = dataset.preprocess_for_model(image).to(device)
        
        # ベースモデル特徴抽出
        with torch.no_grad():
            features = base_model.vit(image_tensor)
            
            # LAFT変換
            transformed_features = laft_model.transform_features(features)
            
            # 変換された特徴を使ってベースモデルで推論
            output = base_model.process_features(transformed_features)
            
        # 画像レベルのスコア
        score = output["anomaly_score"].cpu().numpy().item()
        normal_scores.append(score)
    
    # 異常画像の評価
    for img_path, mask_path in zip(anomaly_image_paths, anomaly_mask_paths):
        # 画像読み込み
        image = dataset.load_image(img_path)
        image_tensor = dataset.preprocess_for_model(image).to(device)
        
        # マスク読み込み
        mask = dataset.load_mask(mask_path)
        
        # ベースモデル特徴抽出
        with torch.no_grad():
            features = base_model.vit(image_tensor)
            
            # LAFT変換
            transformed_features = laft_model.transform_features(features)
            
            # 変換された特徴を使ってベースモデルで推論
            output = base_model.process_features(transformed_features)
            
        # 画像レベルのスコア
        score = output["anomaly_score"].cpu().numpy().item()
        anomaly_scores.append(score)
        
        # 異常マップと真のマスク
        anomaly_map = output["anomaly_map"].cpu().numpy().squeeze()
        anomaly_map = anomaly_map / (anomaly_map.max() + 1e-8)  # 0-1に正規化
        anomaly_maps.append(anomaly_map)
        anomaly_masks.append(mask)
    
    # 結果の算出
    # 1. 画像レベルのAUROC
    y_true = [0] * len(normal_scores) + [1] * len(anomaly_scores)
    y_scores = normal_scores + anomaly_scores
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except:
        logger.warning(f"AUROCの計算に失敗しました")
        auroc = float('nan')
    
    # 2. ピクセルレベルのAUROC
    pixel_auroc = float('nan')
    if anomaly_maps and anomaly_masks:
        try:
            # すべての画像のピクセルを結合
            all_pixels_pred = np.concatenate([m.flatten() for m in anomaly_maps])
            all_pixels_true = np.concatenate([(m > 0).flatten() for m in anomaly_masks])
            
            pixel_auroc = roc_auc_score(all_pixels_true, all_pixels_pred)
        except:
            logger.warning(f"ピクセルAUROCの計算に失敗しました")
    
    # 3. 正常・異常の平均スコア
    normal_mean = np.mean(normal_scores) if normal_scores else float('nan')
    anomaly_mean = np.mean(anomaly_scores) if anomaly_scores else float('nan')
    
    # 4. 分離度（異常と正常のスコア差）
    separation = anomaly_mean - normal_mean if not np.isnan(normal_mean) and not np.isnan(anomaly_mean) else float('nan')
    
    # Projectionマトリックスの特性
    if projection_matrix is not None:
        proj_norm = torch.norm(projection_matrix).item()
        proj_complexity = torch.std(projection_matrix).item() / torch.mean(torch.abs(projection_matrix)).item()
    else:
        proj_norm = float('nan')
        proj_complexity = float('nan')
    
    # 結果の返却
    result = {
        "category": category,
        "instruction": instruction,
        "auroc": auroc,
        "pixel_auroc": pixel_auroc,
        "normal_mean": float(normal_mean),
        "anomaly_mean": float(anomaly_mean),
        "separation": float(separation),
        "projection_norm": float(proj_norm),
        "projection_complexity": float(proj_complexity),
        "adjustment_time": adjustment_time,
        "normal_count": len(normal_scores),
        "anomaly_count": len(anomaly_scores)
    }
    
    logger.info(f"評価結果: AUROC={auroc:.4f}, PixelAUROC={pixel_auroc:.4f}, 分離度={separation:.4f}")
    
    return result


def run_instruction_benchmark(args):
    """指示ベンチマークの実行"""
    # デバイスの取得
    device = args.device if args.device else get_available_device()
    logger.info(f"使用デバイス: {device}")
    
    # 結果の保存先
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # MVTecデータセットの初期化
    dataset = MVTecDataset(args.dataset_path)
    
    # モデルの初期化
    # ベースモデル
    base_model = TransFusionLite(
        backbone="vit_base_patch16_224",
        pretrained=True,
        n_steps=4,
        device=device
    )
    base_model.to(device)
    base_model.eval()
    
    # LAFTの初期化
    laft_model = LAFTPhi4(
        clip_model_name="openai/clip-vit-base-patch16",
        phi_model_path=None,  # Phi-4-miniなしでテスト
        device=device,
        feature_dim=base_model.vit_feature_dim
    )
    
    # 指示のロード
    all_instructions = generate_instructions()
    
    # テスト対象のカテゴリー
    if args.categories:
        categories = args.categories.split(",")
        # 有効なカテゴリのみフィルタ
        categories = [c for c in categories if c in dataset.valid_categories]
    else:
        # デフォルトでいくつかのカテゴリを選択
        categories = ["bottle", "carpet", "grid", "leather", "metal_nut"]
        # 有効なカテゴリのみフィルタ
        categories = [c for c in categories if c in dataset.valid_categories]
        
        # 利用可能なカテゴリが少ない場合は追加
        if len(categories) < 3 and len(dataset.valid_categories) >= 3:
            categories = dataset.valid_categories[:3]
    
    logger.info(f"評価するカテゴリ: {', '.join(categories)}")
    
    # 指示評価のためのカテゴリ選択
    if not categories:
        raise ValueError("評価するカテゴリがありません")
    
    # 結果保存用
    all_results = []
    
    # 指示タイプごとの評価
    for instruction_type, instructions in all_instructions.items():
        logger.info(f"指示タイプ: {instruction_type} の評価")
        
        # カテゴリ固有指示の特別処理
        if instruction_type == "category_specific":
            for category in categories:
                if category in instructions:
                    instruction = instructions[category]
                    result = evaluate_instruction_accuracy(
                        base_model, laft_model, dataset, category, instruction, device, args.max_images)
                    result["instruction_type"] = instruction_type
                    all_results.append(result)
        else:
            # その他の指示タイプ
            # 各カテゴリで各指示を評価
            for category in categories:
                for instruction in instructions[:args.max_instructions]:
                    result = evaluate_instruction_accuracy(
                        base_model, laft_model, dataset, category, instruction, device, args.max_images)
                    result["instruction_type"] = instruction_type
                    all_results.append(result)
    
    # 結果の保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSVに保存
    csv_path = os.path.join(output_dir, f"laft_instruction_benchmark_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Category', 'Instruction Type', 'Instruction', 
            'AUROC', 'Pixel AUROC', 'Separation',
            'Normal Mean', 'Anomaly Mean',
            'Projection Norm', 'Projection Complexity',
            'Adjustment Time (s)'
        ])
        
        for result in all_results:
            writer.writerow([
                result['category'],
                result['instruction_type'],
                result['instruction'],
                f"{result['auroc']:.4f}",
                f"{result['pixel_auroc']:.4f}",
                f"{result['separation']:.4f}",
                f"{result['normal_mean']:.4f}",
                f"{result['anomaly_mean']:.4f}",
                f"{result['projection_norm']:.4f}",
                f"{result['projection_complexity']:.4f}",
                f"{result['adjustment_time']:.4f}"
            ])
    
    # JSONに保存
    json_path = os.path.join(output_dir, f"laft_instruction_benchmark_detailed_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 結果の分析とチャート作成
    create_instruction_charts(all_results, output_dir, timestamp)
    
    logger.info(f"結果を保存しました: {csv_path}, {json_path}")
    
    return all_results


def create_instruction_charts(results, output_dir, timestamp):
    """指示評価の結果チャートを作成"""
    # 実行環境情報の取得
    device_name = "MacBook Pro M4"
    device_info = f"Memory: 16GB, OS: macOS 14.5"
    runtime_info = f"Torch {torch.__version__}, ANE: Enabled"
    
    # 1. 指示タイプ別の性能比較
    plt.figure(figsize=(12, 10))
    
    # 指示タイプごとのデータ
    instruction_types = sorted(set(r["instruction_type"] for r in results))
    auroc_by_type = {t: [] for t in instruction_types}
    pixel_auroc_by_type = {t: [] for t in instruction_types}
    separation_by_type = {t: [] for t in instruction_types}
    
    for result in results:
        t = result["instruction_type"]
        if not np.isnan(result["auroc"]):
            auroc_by_type[t].append(result["auroc"])
        if not np.isnan(result["pixel_auroc"]):
            pixel_auroc_by_type[t].append(result["pixel_auroc"])
        if not np.isnan(result["separation"]):
            separation_by_type[t].append(result["separation"])
    
    # 平均値の計算
    auroc_means = [np.mean(auroc_by_type[t]) if auroc_by_type[t] else 0 for t in instruction_types]
    pixel_auroc_means = [np.mean(pixel_auroc_by_type[t]) if pixel_auroc_by_type[t] else 0 for t in instruction_types]
    separation_means = [np.mean(separation_by_type[t]) if separation_by_type[t] else 0 for t in instruction_types]
    
    # グラフの描画
    plt.subplot(2, 2, 1)
    x = np.arange(len(instruction_types))
    width = 0.35
    
    plt.bar(x - width/2, auroc_means, width, label='AUROC', color='#3498db')
    plt.bar(x + width/2, pixel_auroc_means, width, label='Pixel AUROC', color='#2ecc71')
    
    plt.xlabel('Instruction Type')
    plt.ylabel('AUROC')
    plt.title('AUROC by Instruction Type')
    plt.xticks(x, [t.capitalize() for t in instruction_types], rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 2. カテゴリごとの指示効果
    plt.subplot(2, 2, 2)
    
    categories = sorted(set(r["category"] for r in results))
    auroc_by_category = {c: [] for c in categories}
    
    for result in results:
        c = result["category"]
        if not np.isnan(result["auroc"]):
            auroc_by_category[c].append(result["auroc"])
    
    # カテゴリごとの平均AUROC
    category_means = [np.mean(auroc_by_category[c]) if auroc_by_category[c] else 0 for c in categories]
    category_stds = [np.std(auroc_by_category[c]) if len(auroc_by_category[c]) > 1 else 0 for c in categories]
    
    # エラーバー付きの棒グラフ
    plt.bar(categories, category_means, yerr=category_stds, capsize=5, color='#e74c3c')
    plt.xlabel('Category')
    plt.ylabel('AUROC')
    plt.title('AUROC by Category')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    # 3. 指示の複雑さと性能の関係
    plt.subplot(2, 2, 3)
    
    # 指示の長さを複雑さの指標とする
    instruction_lengths = [len(r["instruction"]) for r in results]
    aurocs = [r["auroc"] for r in results]
    
    # 散布図
    plt.scatter(instruction_lengths, aurocs, alpha=0.6, c='#9b59b6')
    
    # 傾向線
    if len(instruction_lengths) > 1:
        z = np.polyfit(instruction_lengths, aurocs, 1)
        p = np.poly1d(z)
        plt.plot(sorted(instruction_lengths), p(sorted(instruction_lengths)), "r--", alpha=0.7)
    
    plt.xlabel('Instruction Length (chars)')
    plt.ylabel('AUROC')
    plt.title('AUROC vs Instruction Complexity')
    plt.grid(True, alpha=0.3)
    
    # 4. Projectionマトリックスの特性と性能
    plt.subplot(2, 2, 4)
    
    complexity = [r["projection_complexity"] for r in results]
    separation = [r["separation"] for r in results]
    
    # 散布図
    plt.scatter(complexity, separation, alpha=0.6, c='#f39c12')
    
    # 傾向線
    if len(complexity) > 1:
        mask = ~np.isnan(complexity) & ~np.isnan(separation)
        if np.sum(mask) > 1:
            z = np.polyfit(np.array(complexity)[mask], np.array(separation)[mask], 1)
            p = np.poly1d(z)
            x_range = np.array(sorted([c for c, s in zip(complexity, separation) if not np.isnan(c) and not np.isnan(s)]))
            if len(x_range) > 0:
                plt.plot(x_range, p(x_range), "r--", alpha=0.7)
    
    plt.xlabel('Projection Matrix Complexity')
    plt.ylabel('Separation (Anomaly - Normal Score)')
    plt.title('Separation vs Matrix Complexity')
    plt.grid(True, alpha=0.3)
    
    # タイトルと追加情報
    plt.suptitle('LAFT Instruction-based Adaptation Evaluation', fontsize=16)
    plt.figtext(0.01, 0.01, f"{device_info}\n{runtime_info}", fontsize=8)
    
    # 保存
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    chart_path = os.path.join(output_dir, f"laft_instruction_chart_{timestamp}.png")
    plt.savefig(chart_path, dpi=300)
    
    # SVG形式でも保存
    svg_path = os.path.join(output_dir, f"laft_instruction_chart_{timestamp}.svg")
    plt.savefig(svg_path, format='svg')
    
    plt.close()
    
    # 5. 最良の指示ランキング
    plt.figure(figsize=(12, 8))
    
    # AUROCが高い上位10件の指示
    sorted_results = sorted(results, key=lambda x: x["auroc"] if not np.isnan(x["auroc"]) else 0, reverse=True)
    top_n = min(10, len(sorted_results))
    top_instructions = sorted_results[:top_n]
    
    # 短縮した指示テキスト
    instruction_texts = [r["instruction"][:50] + "..." if len(r["instruction"]) > 50 else r["instruction"] 
                         for r in top_instructions]
    auroc_values = [r["auroc"] for r in top_instructions]
    
    # 横棒グラフ
    bars = plt.barh(range(top_n), auroc_values, color='#2ecc71')
    plt.yticks(range(top_n), instruction_texts)
    plt.xlabel('AUROC')
    plt.title('Top Performing Instructions')
    plt.xlim(0.5, 1.0)
    plt.grid(axis='x', alpha=0.3)
    
    # バーの上に値を表示
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{auroc_values[i]:.3f}', va='center')
    
    # 追加情報
    plt.figtext(0.01, 0.01, f"{device_info}\n{runtime_info}", fontsize=8)
    
    # 保存
    plt.tight_layout()
    chart_path = os.path.join(output_dir, f"laft_instruction_ranking_{timestamp}.png")
    plt.savefig(chart_path, dpi=300)
    
    # SVG形式でも保存
    svg_path = os.path.join(output_dir, f"laft_instruction_ranking_{timestamp}.svg")
    plt.savefig(svg_path, format='svg')
    
    plt.close()
    
    logger.info(f"チャートを保存しました: {chart_path}, {svg_path}")


def parse_args():
    """コマンドライン引数のパース"""
    parser = argparse.ArgumentParser(description="LAFTの言語指示ベース適応評価ベンチマーク")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="MVTec AD2データセットのパス")
    parser.add_argument("--categories", type=str, default=None,
                        help="評価するカテゴリ（カンマ区切り）")
    parser.add_argument("--max-instructions", type=int, default=3,
                        help="各タイプで評価する最大指示数")
    parser.add_argument("--max-images", type=int, default=30,
                        help="各カテゴリで評価する最大画像数")
    parser.add_argument("--device", type=str, default="",
                        help="使用するデバイス (mps, cuda, cpu)")
    parser.add_argument("--output-dir", type=str, default="./experiments/m4_benchmarks/results",
                        help="結果の保存先ディレクトリ")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_instruction_benchmark(args)