#!/usr/bin/env python3
"""
パフォーマンスベンチマークスクリプト

MVTec-AD2、VisA、VIADUCTデータセットに対する各異常検知モデルの評価を行います。
"""

import os
import sys
import json
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dataset_performance_benchmark")

class DatasetLoader:
    """データセットローダー"""
    
    def __init__(self, dataset_name, dataset_dir=None):
        """
        Args:
            dataset_name: データセット名 ('mvtec_ad2', 'visa', 'viaduct')
            dataset_dir: データセットのディレクトリ（Noneの場合はデフォルト）
        """
        self.dataset_name = dataset_name
        
        # デフォルトのディレクトリ
        if dataset_dir is None:
            self.dataset_dir = Path(f"data/{dataset_name}")
        else:
            self.dataset_dir = Path(dataset_dir)
        
        # データセットが存在するか確認
        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")
            
        logger.info(f"{dataset_name} データセットを読み込み中: {self.dataset_dir}")
        
        # データセットの構造に合わせたパス設定
        if dataset_name == "mvtec_ad2":
            self.categories = self._get_mvtec_categories()
        elif dataset_name == "visa":
            self.categories = self._get_visa_categories()
        elif dataset_name == "viaduct":
            self.categories = self._get_viaduct_categories()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        logger.info(f"データセット {dataset_name} を読み込みました")
    
    def _get_mvtec_categories(self):
        """MVTec-AD2のカテゴリ一覧を取得"""
        categories = []
        
        for item in self.dataset_dir.iterdir():
            if item.is_dir() and (item / "train").exists() and (item / "test").exists():
                categories.append(item.name)
                
        return sorted(categories)
    
    def _get_visa_categories(self):
        """VisAのカテゴリ一覧を取得"""
        categories = []
        
        # VisAディレクトリの特定
        visa_dir = self.dataset_dir / "VisA"
        if visa_dir.exists():
            for item in visa_dir.iterdir():
                if item.is_dir() and (item / "normal").exists() and (item / "abnormal").exists():
                    categories.append(item.name)
        
        return sorted(categories)
    
    def _get_viaduct_categories(self):
        """VIADUCTのカテゴリ一覧を取得"""
        categories = []
        
        # VIADUCTのカテゴリは特定のサブディレクトリ構造に基づく
        data_dir = self.dataset_dir / "data"
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir():
                    categories.append(item.name)
                    
        return sorted(categories)
    
    def get_test_images(self, category):
        """
        テスト画像のパスとラベルを取得
        
        Args:
            category: カテゴリ名
            
        Returns:
            (画像パスのリスト, ラベルのリスト, マスクパスのリスト)
        """
        image_paths = []
        labels = []  # 0: normal, 1: anomaly
        mask_paths = []  # 異常のセグメンテーションマスク（正常の場合はNone）
        
        if self.dataset_name == "mvtec_ad2":
            category_path = self.dataset_dir / category
            test_path = category_path / "test"
            
            # 正常テスト画像
            normal_dir = test_path / "good"
            for img_path in normal_dir.glob("*.png"):
                image_paths.append(str(img_path))
                labels.append(0)
                mask_paths.append(None)
                
            # 異常テスト画像
            anomaly_types = [d for d in os.listdir(test_path) 
                           if os.path.isdir(os.path.join(test_path, d)) and d != "good"]
            
            for anomaly_type in anomaly_types:
                anomaly_dir = test_path / anomaly_type
                gt_dir = category_path / "ground_truth" / anomaly_type
                
                for img_path in anomaly_dir.glob("*.png"):
                    image_paths.append(str(img_path))
                    labels.append(1)
                    
                    # 対応するマスクパスを取得
                    img_name = img_path.stem
                    mask_path = gt_dir / f"{img_name}_mask.png"
                    
                    if mask_path.exists():
                        mask_paths.append(str(mask_path))
                    else:
                        mask_paths.append(None)
                        
        elif self.dataset_name == "visa":
            category_path = self.dataset_dir / "VisA" / category
            
            # 正常テスト画像
            normal_dir = category_path / "normal"
            for img_path in normal_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(0)
                    mask_paths.append(None)
                    
            # 異常テスト画像
            anomaly_dir = category_path / "abnormal"
            mask_dir = category_path / "abnormal_mask"
            
            for img_path in anomaly_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
                    labels.append(1)
                    
                    # 対応するマスクを探す
                    img_name = img_path.stem
                    mask_path = mask_dir / f"{img_name}.png"
                    
                    if mask_path.exists():
                        mask_paths.append(str(mask_path))
                    else:
                        mask_paths.append(None)
                    
        elif self.dataset_name == "viaduct":
            # VIADUCT特有の構造に応じて実装
            pass
            
        return image_paths, labels, mask_paths
    
    def get_train_images(self, category):
        """
        トレーニング画像のパスを取得（通常は正常画像のみ）
        
        Args:
            category: カテゴリ名
            
        Returns:
            画像パスのリスト
        """
        image_paths = []
        
        if self.dataset_name == "mvtec_ad2":
            normal_dir = self.dataset_dir / category / "train" / "good"
            
            for img_path in normal_dir.glob("*.png"):
                image_paths.append(str(img_path))
                
        elif self.dataset_name == "visa":
            normal_dir = self.dataset_dir / "VisA" / category / "normal"
            
            for img_path in normal_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    image_paths.append(str(img_path))
        
        return image_paths

# モデル評価関数
def evaluate_transfusion(dataset, categories, device, fast_mode=False):
    """
    TransFusion-Liteモデルの評価
    
    Args:
        dataset: DatasetLoaderインスタンス
        categories: 評価するカテゴリのリスト
        device: 使用するデバイス
        fast_mode: 高速モード（サンプル数制限）
        
    Returns:
        評価結果の辞書
    """
    from src.models.transfusion.model import TransFusionLite
    
    logger.info("TransFusion-Liteモデルの評価開始")
    model = TransFusionLite(device=device)
    
    results = {
        "model": "TransFusion-Lite",
        "overall_auroc": 0.0,
        "categories": {}
    }
    
    all_scores = []
    all_labels = []
    
    for category in tqdm(categories, desc="カテゴリ評価"):
        logger.info(f"カテゴリ: {category} の評価中")
        
        # トレーニング（正常）画像の取得
        train_images = dataset.get_train_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            train_sample_count = min(50, len(train_images))
            train_images = train_images[:train_sample_count]
        
        # モデルのフィット
        model.fit(train_images)
        
        # テスト画像の取得
        test_images, test_labels, test_masks = dataset.get_test_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            max_samples = 20
            # ラベルの分布を保持するためにバランスよく選択
            normal_indices = [i for i, label in enumerate(test_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(test_labels) if label == 1]
            
            selected_normal = normal_indices[:max_samples//2]
            selected_anomaly = anomaly_indices[:max_samples//2]
            selected_indices = sorted(selected_normal + selected_anomaly)
            
            test_images = [test_images[i] for i in selected_indices]
            test_labels = [test_labels[i] for i in selected_indices]
            test_masks = [test_masks[i] for i in selected_indices]
        
        # 予測と評価
        anomaly_scores = []
        pixel_masks = []
        
        for image_path in test_images:
            score, mask = model.predict(image_path)
            anomaly_scores.append(score)
            pixel_masks.append(mask)
        
        # AUROCの計算
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(test_labels, anomaly_scores)
        except:
            auroc = float('nan')
        
        # PixelワイズAUROCの計算（あれば）
        pixel_auroc = calculate_pixel_auroc(pixel_masks, test_masks, test_labels)
        
        # 結果の保存
        results["categories"][category] = {
            "auroc": auroc,
            "pixel_auroc": pixel_auroc
        }
        
        logger.info(f"カテゴリ: {category} - AUROC: {auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}")
        
        # 全体評価用にスコアとラベルを収集
        all_scores.extend(anomaly_scores)
        all_labels.extend(test_labels)
    
    # 全体AUROCの計算
    try:
        overall_auroc = roc_auc_score(all_labels, all_scores)
    except:
        overall_auroc = float('nan')
    
    results["overall_auroc"] = overall_auroc
    return results

def evaluate_inreach(dataset, categories, device, fast_mode=False):
    """
    InReaCh-FOモデルの評価
    
    Args:
        dataset: DatasetLoaderインスタンス
        categories: 評価するカテゴリのリスト
        device: 使用するデバイス
        fast_mode: 高速モード（サンプル数制限）
        
    Returns:
        評価結果の辞書
    """
    from src.models.inreach.model import InReachFO
    
    logger.info("InReaCh-FOモデルの評価開始")
    model = InReachFO(device=device)
    
    results = {
        "model": "InReaCh-FO",
        "overall_auroc": 0.0,
        "categories": {}
    }
    
    all_scores = []
    all_labels = []
    
    for category in tqdm(categories, desc="カテゴリ評価"):
        logger.info(f"カテゴリ: {category} の評価中")
        
        # トレーニング（正常）画像の取得
        train_images = dataset.get_train_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            train_sample_count = min(50, len(train_images))
            train_images = train_images[:train_sample_count]
        
        # モデルのフィット
        model.fit(train_images)
        
        # テスト画像の取得
        test_images, test_labels, test_masks = dataset.get_test_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            max_samples = 20
            # ラベルの分布を保持するためにバランスよく選択
            normal_indices = [i for i, label in enumerate(test_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(test_labels) if label == 1]
            
            selected_normal = normal_indices[:max_samples//2]
            selected_anomaly = anomaly_indices[:max_samples//2]
            selected_indices = sorted(selected_normal + selected_anomaly)
            
            test_images = [test_images[i] for i in selected_indices]
            test_labels = [test_labels[i] for i in selected_indices]
            test_masks = [test_masks[i] for i in selected_indices]
        
        # 予測と評価
        anomaly_scores = []
        pixel_masks = []
        
        for image_path in test_images:
            score, mask = model.predict(image_path)
            anomaly_scores.append(score)
            pixel_masks.append(mask)
        
        # AUROCの計算
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(test_labels, anomaly_scores)
        except:
            auroc = float('nan')
        
        # PixelワイズAUROCの計算（あれば）
        pixel_auroc = calculate_pixel_auroc(pixel_masks, test_masks, test_labels)
        
        # 結果の保存
        results["categories"][category] = {
            "auroc": auroc,
            "pixel_auroc": pixel_auroc
        }
        
        logger.info(f"カテゴリ: {category} - AUROC: {auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}")
        
        # 全体評価用にスコアとラベルを収集
        all_scores.extend(anomaly_scores)
        all_labels.extend(test_labels)
    
    # 全体AUROCの計算
    try:
        overall_auroc = roc_auc_score(all_labels, all_scores)
    except:
        overall_auroc = float('nan')
    
    results["overall_auroc"] = overall_auroc
    return results

def evaluate_samlad(dataset, categories, device, fast_mode=False):
    """
    SAM-LADモデルの評価
    
    Args:
        dataset: DatasetLoaderインスタンス
        categories: 評価するカテゴリのリスト
        device: 使用するデバイス
        fast_mode: 高速モード（サンプル数制限）
        
    Returns:
        評価結果の辞書
    """
    from src.models.samlad.model import SAMLAD
    
    logger.info("SAM-LADモデルの評価開始")
    model = SAMLAD(device=device)
    
    results = {
        "model": "SAM-LAD",
        "overall_auroc": 0.0,
        "categories": {}
    }
    
    all_scores = []
    all_labels = []
    
    for category in tqdm(categories, desc="カテゴリ評価"):
        logger.info(f"カテゴリ: {category} の評価中")
        
        # トレーニング（正常）画像の取得
        train_images = dataset.get_train_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            train_sample_count = min(50, len(train_images))
            train_images = train_images[:train_sample_count]
        
        # モデルのフィット
        model.fit(train_images)
        
        # テスト画像の取得
        test_images, test_labels, test_masks = dataset.get_test_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            max_samples = 20
            # ラベルの分布を保持するためにバランスよく選択
            normal_indices = [i for i, label in enumerate(test_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(test_labels) if label == 1]
            
            selected_normal = normal_indices[:max_samples//2]
            selected_anomaly = anomaly_indices[:max_samples//2]
            selected_indices = sorted(selected_normal + selected_anomaly)
            
            test_images = [test_images[i] for i in selected_indices]
            test_labels = [test_labels[i] for i in selected_indices]
            test_masks = [test_masks[i] for i in selected_indices]
        
        # 予測と評価
        anomaly_scores = []
        pixel_masks = []
        
        for image_path in test_images:
            score, mask = model.predict(image_path)
            anomaly_scores.append(score)
            pixel_masks.append(mask)
        
        # AUROCの計算
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(test_labels, anomaly_scores)
        except:
            auroc = float('nan')
        
        # PixelワイズAUROCの計算（あれば）
        pixel_auroc = calculate_pixel_auroc(pixel_masks, test_masks, test_labels)
        
        # 結果の保存
        results["categories"][category] = {
            "auroc": auroc,
            "pixel_auroc": pixel_auroc
        }
        
        logger.info(f"カテゴリ: {category} - AUROC: {auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}")
        
        # 全体評価用にスコアとラベルを収集
        all_scores.extend(anomaly_scores)
        all_labels.extend(test_labels)
    
    # 全体AUROCの計算
    try:
        overall_auroc = roc_auc_score(all_labels, all_scores)
    except:
        overall_auroc = float('nan')
    
    results["overall_auroc"] = overall_auroc
    return results

def evaluate_laft(dataset, categories, device, fast_mode=False):
    """
    LAFTモデルの評価
    
    Args:
        dataset: DatasetLoaderインスタンス
        categories: 評価するカテゴリのリスト
        device: 使用するデバイス
        fast_mode: 高速モード（サンプル数制限）
        
    Returns:
        評価結果の辞書
    """
    from src.models.laft.model import LAFT
    
    logger.info("LAFTモデルの評価開始")
    model = LAFT(device=device)
    
    results = {
        "model": "LAFT",
        "overall_auroc": 0.0,
        "categories": {}
    }
    
    all_scores = []
    all_labels = []
    
    for category in tqdm(categories, desc="カテゴリ評価"):
        logger.info(f"カテゴリ: {category} の評価中")
        
        # トレーニング（正常）画像の取得
        train_images = dataset.get_train_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            train_sample_count = min(50, len(train_images))
            train_images = train_images[:train_sample_count]
        
        # モデルのフィット
        model.fit(train_images)
        
        # テスト画像の取得
        test_images, test_labels, test_masks = dataset.get_test_images(category)
        
        # 高速モードの場合はサンプル数を制限
        if fast_mode:
            max_samples = 20
            # ラベルの分布を保持するためにバランスよく選択
            normal_indices = [i for i, label in enumerate(test_labels) if label == 0]
            anomaly_indices = [i for i, label in enumerate(test_labels) if label == 1]
            
            selected_normal = normal_indices[:max_samples//2]
            selected_anomaly = anomaly_indices[:max_samples//2]
            selected_indices = sorted(selected_normal + selected_anomaly)
            
            test_images = [test_images[i] for i in selected_indices]
            test_labels = [test_labels[i] for i in selected_indices]
            test_masks = [test_masks[i] for i in selected_indices]
        
        # 予測と評価
        anomaly_scores = []
        pixel_masks = []
        
        for image_path in test_images:
            score, mask = model.predict(image_path)
            anomaly_scores.append(score)
            pixel_masks.append(mask)
        
        # AUROCの計算
        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(test_labels, anomaly_scores)
        except:
            auroc = float('nan')
        
        # PixelワイズAUROCの計算（あれば）
        pixel_auroc = calculate_pixel_auroc(pixel_masks, test_masks, test_labels)
        
        # 結果の保存
        results["categories"][category] = {
            "auroc": auroc,
            "pixel_auroc": pixel_auroc
        }
        
        logger.info(f"カテゴリ: {category} - AUROC: {auroc:.4f}, Pixel AUROC: {pixel_auroc:.4f}")
        
        # 全体評価用にスコアとラベルを収集
        all_scores.extend(anomaly_scores)
        all_labels.extend(test_labels)
    
    # 全体AUROCの計算
    try:
        overall_auroc = roc_auc_score(all_labels, all_scores)
    except:
        overall_auroc = float('nan')
    
    results["overall_auroc"] = overall_auroc
    return results

def calculate_pixel_auroc(pred_masks, gt_masks, labels):
    """
    ピクセルワイズAUROCの計算
    
    Args:
        pred_masks: 予測マスクのリスト
        gt_masks: 正解マスクのリスト（正常の場合はNone）
        labels: 画像ラベル（0: 正常, 1: 異常）
    
    Returns:
        ピクセルワイズAUROC
    """
    # 有効なマスクペアのみ抽出
    valid_pairs = [(pred, gt) for pred, gt, label in zip(pred_masks, gt_masks, labels)
                  if gt is not None and label == 1]
    
    if not valid_pairs:
        return float('nan')
    
    from sklearn.metrics import roc_auc_score
    
    # 正解マスクと予測マスクのフラット化
    all_pred = []
    all_gt = []
    
    for pred_mask, gt_mask in valid_pairs:
        try:
            # マスクの読み込み
            import cv2
            import numpy as np
            
            # 予測マスク
            if isinstance(pred_mask, str):
                pred = cv2.imread(pred_mask, cv2.IMREAD_GRAYSCALE)
            elif isinstance(pred_mask, np.ndarray):
                pred = pred_mask
            else:
                continue
                
            # 正解マスク
            if isinstance(gt_mask, str):
                gt = cv2.imread(gt_mask, cv2.IMREAD_GRAYSCALE)
            elif isinstance(gt_mask, np.ndarray):
                gt = gt_mask
            else:
                continue
                
            # サイズが異なる場合はリサイズ
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
                
            # 二値化（正解マスク）
            _, gt_binary = cv2.threshold(gt, 1, 1, cv2.THRESH_BINARY)
            
            # フラット化して追加
            all_pred.extend(pred.flatten())
            all_gt.extend(gt_binary.flatten())
        except Exception as e:
            logger.warning(f"Error processing mask pair: {e}")
            continue
    
    if not all_pred:
        return float('nan')
    
    # AUROCの計算
    try:
        return roc_auc_score(all_gt, all_pred)
    except:
        return float('nan')

def run_benchmark(args):
    """
    ベンチマークの実行
    
    Args:
        args: 実行パラメータ
    """
    # デバイスの特定
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"使用デバイス: {device}")
    
    # データセットのロード
    dataset = DatasetLoader(args.dataset)
    
    if args.num_categories is None:
        categories = dataset.categories
    else:
        categories = dataset.categories[:args.num_categories]
    
    logger.info(f"評価するカテゴリ: {', '.join(categories)}")
    
    # 評価結果
    benchmark_results = []
    
    # TransFusion-Liteの評価
    if not args.skip_transfusion:
        transfusion_results = evaluate_transfusion(dataset, categories, device, args.fast_mode)
        benchmark_results.append(transfusion_results)
    
    # InReaCh-FOの評価
    if not args.skip_inreach:
        inreach_results = evaluate_inreach(dataset, categories, device, args.fast_mode)
        benchmark_results.append(inreach_results)
    
    # SAM-LADの評価
    if not args.skip_samlad:
        samlad_results = evaluate_samlad(dataset, categories, device, args.fast_mode)
        benchmark_results.append(samlad_results)
    
    # LAFTの評価
    if not args.skip_laft:
        laft_results = evaluate_laft(dataset, categories, device, args.fast_mode)
        benchmark_results.append(laft_results)
    
    # 結果の保存
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(args.output_dir, f"{args.dataset}_performance_detailed_{timestamp}.json")
    
    with open(result_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_performance_summary_{timestamp}.txt")
    
    with open(summary_path, "w") as f:
        f.write(f"=== {args.dataset} Benchmark Results ===\n\n")
        
        for result in benchmark_results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Overall AUROC: {result['overall_auroc']:.4f}\n\n")
            
            f.write("Category Results:\n")
            for category, metrics in result["categories"].items():
                f.write(f"  {category}:\n")
                f.write(f"    AUROC: {metrics['auroc']:.4f}\n")
                f.write(f"    Pixel AUROC: {metrics['pixel_auroc']:.4f}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    logger.info(f"結果を保存しました: {result_path}, {summary_path}")

def run_benchmark_on_categories(args):
    """
    特定のカテゴリに対するベンチマークの実行
    
    Args:
        args: 実行パラメータ（categories属性を含む）
    """
    # デバイスの特定
    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'is_available') and torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    logger.info(f"使用デバイス: {device}")
    
    # データセットのロード
    dataset = DatasetLoader(args.dataset)
    
    # 指定されたカテゴリを使用
    categories = args.categories
    
    logger.info(f"評価するカテゴリ: {', '.join(categories)}")
    
    # 評価結果
    benchmark_results = []
    
    # TransFusion-Liteの評価
    if not args.skip_transfusion:
        transfusion_results = evaluate_transfusion(dataset, categories, device, args.fast_mode)
        benchmark_results.append(transfusion_results)
    
    # InReaCh-FOの評価
    if not args.skip_inreach:
        inreach_results = evaluate_inreach(dataset, categories, device, args.fast_mode)
        benchmark_results.append(inreach_results)
    
    # SAM-LADの評価
    if not args.skip_samlad:
        samlad_results = evaluate_samlad(dataset, categories, device, args.fast_mode)
        benchmark_results.append(samlad_results)
    
    # LAFTの評価
    if not args.skip_laft:
        laft_results = evaluate_laft(dataset, categories, device, args.fast_mode)
        benchmark_results.append(laft_results)
    
    # 結果の保存
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(args.output_dir, f"{args.dataset}_performance_detailed_{timestamp}.json")
    
    with open(result_path, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    
    summary_path = os.path.join(args.output_dir, f"{args.dataset}_performance_summary_{timestamp}.txt")
    
    with open(summary_path, "w") as f:
        f.write(f"=== {args.dataset} Benchmark Results ===\n\n")
        
        for result in benchmark_results:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Overall AUROC: {result['overall_auroc']:.4f}\n\n")
            
            f.write("Category Results:\n")
            for category, metrics in result["categories"].items():
                f.write(f"  {category}:\n")
                f.write(f"    AUROC: {metrics['auroc']:.4f}\n")
                f.write(f"    Pixel AUROC: {metrics['pixel_auroc']:.4f}\n")
            
            f.write("\n" + "-"*50 + "\n\n")
    
    logger.info(f"結果を保存しました: {result_path}, {summary_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="データセットパフォーマンスベンチマーク")
    parser.add_argument("--dataset", type=str, required=True, choices=["mvtec_ad2", "visa", "viaduct"],
                      help="評価するデータセット")
    parser.add_argument("--num-categories", type=int, default=None,
                      help="評価するカテゴリ数（Noneの場合はすべて）")
    parser.add_argument("--output-dir", type=str, default="./results",
                      help="結果を保存するディレクトリ")
    parser.add_argument("--device", type=str, default="",
                      help="使用するデバイス（cuda, mps, cpu）")
    parser.add_argument("--skip-transfusion", action="store_true",
                      help="TransFusion-Liteの評価をスキップ")
    parser.add_argument("--skip-inreach", action="store_true",
                      help="InReaCh-FOの評価をスキップ")
    parser.add_argument("--skip-samlad", action="store_true",
                      help="SAM-LADの評価をスキップ")
    parser.add_argument("--skip-laft", action="store_true",
                      help="LAFTの評価をスキップ")
    parser.add_argument("--fast-mode", action="store_true",
                      help="高速モード（サンプル数を制限）")
    
    args = parser.parse_args()
    
    run_benchmark(args)