#!/usr/bin/env python3
"""
SAMモデルのロードとAPIテスト

このスクリプトは、SAMモデルの正しいロードとAPIの挙動を確認するためのテストスクリプトです。
SAM-LADとSegment Anything Model (SAM) の互換性問題をデバッグするのに役立ちます。
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sam_load_test")

def check_sam_installation():
    """SAMのインストール状況を確認"""
    try:
        import segment_anything
        logger.info(f"SAMのインストールを確認: OK")
        return True
    except ImportError:
        logger.error("SAMがインストールされていません。以下のコマンドでインストールしてください:")
        logger.error("pip install git+https://github.com/facebookresearch/segment-anything.git")
        return False

def load_sam_model(model_path, device="cpu"):
    """SAMモデルをロードしてテスト"""
    try:
        logger.info(f"SAMモデルをロード: {model_path}")
        logger.info(f"使用デバイス: {device}")
        
        from segment_anything import sam_model_registry, SamPredictor
        
        # モデルタイプの確認（ファイル名から推測）
        model_type = "vit_h"  # デフォルト
        if "vit_b" in model_path:
            model_type = "vit_b"
        elif "vit_l" in model_path:
            model_type = "vit_l"
        
        logger.info(f"モデルタイプ: {model_type}")
        
        # モデルのロード
        start_time = time.time()
        sam = sam_model_registry[model_type](checkpoint=model_path)
        sam.to(device)
        load_time = time.time() - start_time
        logger.info(f"モデルロード時間: {load_time:.2f}秒")
        
        # モデル情報の表示
        num_params = sum(p.numel() for p in sam.parameters())
        logger.info(f"モデルパラメータ数: {num_params:,}")
        
        # Predictorの作成
        predictor = SamPredictor(sam)
        
        # APIバージョン情報の表示
        logger.info("SAM APIメソッド一覧:")
        for method_name in dir(predictor):
            if not method_name.startswith('_'):
                method = getattr(predictor, method_name)
                if callable(method):
                    logger.info(f" - {method_name}()")
        
        return sam, predictor
    
    except Exception as e:
        logger.error(f"SAMモデルのロード中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_sam_predictor(predictor):
    """SAM Predictorの動作テスト"""
    if predictor is None:
        return False
    
    # テスト画像を生成
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    logger.info(f"テスト画像形状: {test_image.shape}")
    
    try:
        # 画像をセット
        logger.info("画像をセット")
        start_time = time.time()
        predictor.set_image(test_image)
        set_image_time = time.time() - start_time
        logger.info(f"画像セット時間: {set_image_time:.2f}秒")
        
        # 自動セグメンテーションのテスト
        logger.info("自動セグメンテーションをテスト")
        start_time = time.time()
        result = predictor.predict(multimask_output=True, return_logits=True)
        predict_time = time.time() - start_time
        logger.info(f"予測時間: {predict_time:.2f}秒")
        
        # 結果の確認
        logger.info("予測結果の確認")
        if 'masks' in result:
            masks = result['masks']
            logger.info(f"マスク形状: {masks.shape}")
            logger.info(f"マスク数: {len(masks)}")
        
        if 'scores' in result:
            scores = result['scores']
            logger.info(f"スコア: {scores}")
        
        # プロンプトを使用したセグメンテーションのテスト
        logger.info("プロンプトを使用したセグメンテーションをテスト")
        input_points = np.array([[256, 256]])
        input_labels = np.array([1])
        
        start_time = time.time()
        result = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        prompt_predict_time = time.time() - start_time
        logger.info(f"プロンプト予測時間: {prompt_predict_time:.2f}秒")
        
        # 結果の確認
        if 'masks' in result:
            masks = result['masks']
            logger.info(f"プロンプトマスク形状: {masks.shape}")
            
        logger.info("SAM Predictorのテストが完了しました")
        return True
    
    except Exception as e:
        logger.error(f"SAM Predictorのテスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_samlad_with_sam(model_path, device="cpu"):
    """SAM-LADでSAMを使用したテスト"""
    try:
        from src.models.samlad.model import SAMLAD
        
        # SAM-LADモデルの初期化（SAMありで）
        logger.info(f"SAM-LADモデルを初期化（SAM使用）")
        model = SAMLAD(
            sam_checkpoint=model_path,
            device=device,
            min_mask_area=100,
            max_objects=20
        )
        
        # SAMの存在確認
        if model.predictor is not None:
            logger.info("SAMモデルが正常にロードされました")
        else:
            logger.error("SAMモデルのロードに失敗しました")
            return False
        
        # テスト画像を生成
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 画像処理を実行
        logger.info("SAM-LADでテスト画像を処理")
        start_time = time.time()
        results = model.process_image(test_image)
        process_time = time.time() - start_time
        logger.info(f"処理時間: {process_time:.2f}秒")
        
        # 結果を確認
        num_objects = results.get('n_objects', 0)
        logger.info(f"検出オブジェクト数: {num_objects}")
        
        # マスク情報を表示
        if 'masks' in results:
            masks = results['masks']
            logger.info(f"マスク数: {len(masks)}")
            
            for i, mask in enumerate(masks[:3]):  # 最初の3つのみ表示
                logger.info(f"マスク {i+1} 形状: {mask.shape}, タイプ: {mask.dtype}")
        
        logger.info("SAM-LADとSAMの統合テストが成功しました")
        return True
    
    except Exception as e:
        logger.error(f"SAM-LADとSAMの統合テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_samlad_with_without_sam(model_path, device="cpu"):
    """SAMあり/なしのSAM-LADを比較"""
    try:
        from src.models.samlad.model import SAMLAD
        
        # テスト画像を生成
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 1. SAMなしのSAM-LAD
        logger.info("1. SAMなしのSAM-LADをテスト")
        model_no_sam = SAMLAD(
            sam_checkpoint=None,
            device=device,
            min_mask_area=100,
            max_objects=20
        )
        
        start_time = time.time()
        results_no_sam = model_no_sam.process_image(test_image)
        time_no_sam = time.time() - start_time
        logger.info(f"SAMなし処理時間: {time_no_sam:.2f}秒")
        logger.info(f"検出オブジェクト数: {results_no_sam.get('n_objects', 0)}")
        
        # 2. SAMありのSAM-LAD
        logger.info("2. SAMありのSAM-LADをテスト")
        model_with_sam = SAMLAD(
            sam_checkpoint=model_path,
            device=device,
            min_mask_area=100,
            max_objects=20
        )
        
        start_time = time.time()
        results_with_sam = model_with_sam.process_image(test_image)
        time_with_sam = time.time() - start_time
        logger.info(f"SAMあり処理時間: {time_with_sam:.2f}秒")
        logger.info(f"検出オブジェクト数: {results_with_sam.get('n_objects', 0)}")
        
        # 比較
        logger.info("\n--- 比較結果 ---")
        logger.info(f"処理時間比: SAMあり/SAMなし = {time_with_sam/time_no_sam:.2f}倍")
        logger.info(f"オブジェクト数比: SAMあり/SAMなし = {results_with_sam.get('n_objects', 0)/max(1, results_no_sam.get('n_objects', 0)):.2f}倍")
        
        # マスクの違いを確認
        if 'masks' in results_no_sam and 'masks' in results_with_sam:
            masks_no_sam = results_no_sam['masks']
            masks_with_sam = results_with_sam['masks']
            
            logger.info(f"マスク数: SAMなし={len(masks_no_sam)}, SAMあり={len(masks_with_sam)}")
            
            # マスクの質の違いを評価（簡易的に）
            if masks_no_sam and masks_with_sam:
                avg_area_no_sam = np.mean([np.sum(m) for m in masks_no_sam])
                avg_area_with_sam = np.mean([np.sum(m) for m in masks_with_sam])
                
                logger.info(f"平均マスク面積: SAMなし={avg_area_no_sam:.1f}ピクセル, SAMあり={avg_area_with_sam:.1f}ピクセル")
        
        return True
    
    except Exception as e:
        logger.error(f"SAM-LAD比較中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="SAMモデルのロードとAPIテスト")
    parser.add_argument("--model-path", type=str, default=None, 
                       help="SAMモデルファイルのパス")
    parser.add_argument("--device", type=str, default="", 
                       help="使用するデバイス (cpu, cuda, mps)")
    parser.add_argument("--test-samlad", action="store_true",
                       help="SAM-LADとの統合テストを実行する")
    parser.add_argument("--compare", action="store_true",
                       help="SAMあり/なしのSAM-LADを比較する")
    args = parser.parse_args()
    
    # デバイス設定
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"使用デバイス: {device}")
    
    # SAMのインストール確認
    if not check_sam_installation():
        sys.exit(1)
    
    # モデルパスの設定
    model_path = args.model_path
    if model_path is None:
        # デフォルトパスを探索
        default_paths = [
            os.path.join(project_root, "models/weights/sam_vit_h_4b8939.pth"),
            os.path.join(project_root, "models/sam_vit_h_4b8939.pth"),
            os.path.join(project_root, "sam_vit_h_4b8939.pth")
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"SAMモデルを発見: {model_path}")
                break
        
        if model_path is None:
            logger.error("SAMモデルファイルが見つかりません。--model-pathオプションでパスを指定してください。")
            sys.exit(1)
    
    # ファイルサイズの確認
    file_size_mb = os.path.getsize(model_path) / (1024*1024)
    logger.info(f"モデルファイルサイズ: {file_size_mb:.1f} MB")
    
    if file_size_mb < 100:
        logger.warning(f"モデルファイルが小さすぎる可能性があります（{file_size_mb:.1f} MB）。完全なSAMモデルは約2.4GBです。")
    
    # SAMモデルのロードとテスト
    sam, predictor = load_sam_model(model_path, device)
    
    if sam is None or predictor is None:
        logger.error("SAMモデルのロードに失敗しました")
        sys.exit(1)
    
    # Predictorのテスト
    if not test_sam_predictor(predictor):
        logger.error("SAM Predictorのテストに失敗しました")
        sys.exit(1)
    
    # SAM-LADとの統合テスト（指定された場合）
    if args.test_samlad:
        if not test_samlad_with_sam(model_path, device):
            logger.error("SAM-LADとSAMの統合テストに失敗しました")
            sys.exit(1)
    
    # SAMあり/なしの比較（指定された場合）
    if args.compare:
        if not compare_samlad_with_without_sam(model_path, device):
            logger.error("SAMあり/なしの比較に失敗しました")
            sys.exit(1)
    
    logger.info("すべてのテストが正常に完了しました")

if __name__ == "__main__":
    main()