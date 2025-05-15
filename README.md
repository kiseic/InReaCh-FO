# 画像認識 (Image Recognition) プロジェクト

このプロジェクトは、異常検出と適応型画像認識のための実装と実験を含んでいます。

## 主要モデル

1. **TransFusion-Lite**: 4ステップの蒸留を用いた効率的なdiffusionベースの異常検出器
2. **InReaCh-FO**: リアルタイムドメイン適応のためのForward-Onlyの適応アルゴリズム
3. **SAM-LAD**: Segment Anything Model (SAM) を使用した論理的異常検出
4. **LAFT+Phi-4-mini**: Phi-4-mini LLMを使用した言語適応型特徴変換

## データセット

1. **DatasetsDemo-Drift**: 1,000フレームの照明シフトシーケンス。照明条件の変化に対するモデルの回復性能をテストするために使用。
2. **Scratch/Oil Toy**: 600枚の静止画像で、テクスチャと論理的混合検出のためのデータセット。

## 概要

このシステムは次のような特徴を持っています：

- **StreamingAdapter**: 効率的な5フレームマイクロバッチングによるビデオ入力処理
- **TransFusion-Lite**: 4ステップ蒸留によるDiffusionベースの異常検出
- **InReaCh-FO**: オンラインドメイン適応のためのForward-Only適応アルゴリズム
- **SAM-LAD**: Segment Anything Modelによる論理的異常検出
- **LAFT + Phi-4-mini UI**: 言語適応型特徴変換による自然言語制御

## 特徴

- **リアルタイム性能**: すべてのコンポーネントでフレームあたり50ms未満のレイテンシ
- **ドメイン適応**: 64フレーム以内に分布ドリフトから回復
- **マイクロバッチング**: 5フレームバッチによる効率的な処理
- **総合的なベンチマーク**: レイテンシとドリフト回復ベンチマークを含む
- **可視化ツール**: 結果の詳細な可視化を生成

## リポジトリ構造

```
.
├── data/                      # データセットとテスト用ビデオのディレクトリ
├── results/                   # 実験結果
├── src/                       # ソースコード
│   ├── data/                  # データセットユーティリティとジェネレータ
│   ├── experiments/           # ベンチマークフレームワーク
│   ├── models/                # モデル実装
│   │   ├── inreach/           # InReaCh-FO適応アルゴリズム
│   │   ├── laft/              # LAFT + Phi-4-mini
│   │   ├── samlad/            # SAM-LAD実装
│   │   └── transfusion/       # TransFusion-Liteモデル
│   ├── streaming/             # ビデオ入力処理
│   └── visualization/         # 可視化ユーティリティ
├── visualizations/            # 生成された可視化
├── experiments/m4_benchmarks/ # ベンチマーク実験
├── main.py                    # メイン実験ランナー
├── generate_benchmark_figures.py # ベンチマーク図の生成スクリプト
├── generate_scratch_oil_dataset.py # Scratch/Oil Toyデータセット生成
├── fix_samlad_sam_integration.py # SAM-LADとSAMの統合修正
├── REPORT.md                  # 詳細な実験レポート
└── requirements.txt           # Python依存関係
```

## セットアップ

1. リポジトリをクローン:
   ```bash
   git clone https://github.com/yourusername/image_recognition.git
   cd image_recognition
   ```

2. 仮想環境のセットアップ:
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. 依存関係のインストール:
   ```bash
   uv pip install -r requirements.txt
   ```

4. SAMのインストール（オプション）:
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

5. SAMモデルのダウンロード（オプション）:
   ```bash
   python download_sam_with_progress.py
   ```

## 使用方法

### ベンチマークの実行

すべてのモデルのレイテンシを比較：

```bash
python experiments/m4_benchmarks/model_latency_benchmark.py
```

SAM統合版のSAM-LADも含めてベンチマーク：

```bash
python experiments/m4_benchmarks/model_latency_benchmark.py --use-sam
```

オプション:
- `--runs`: 各モデルの実行回数
- `--warmup`: ウォームアップの実行回数
- `--device`: 使用するデバイス (mps, cuda, cpu)
- `--output-dir`: 結果の保存先ディレクトリ
- `--skip-*`: 特定モデルのベンチマークをスキップ

### ベンチマーク図の生成

ベンチマーク視覚化を生成する（Fig-2a、Fig-2b、Fig-2c、Fig-2d）：

```bash
python generate_benchmark_figures.py
```

### Scratch/Oil Toyデータセットの生成

カスタムデータセットを生成する：

```bash
python generate_scratch_oil_dataset.py --num-images 600
```

オプション:
- `--num-images`: 生成する画像の総数
- `--width`/`--height`: 画像の解像度
- `--normal-ratio`: 通常画像の比率

### SAM-LAD と SAM の統合修正

SAM-LADとSegment Anything Model (SAM) の互換性を修正：

```bash
python fix_samlad_sam_integration.py
```

### 実験実行

完全な実験ワークフローを実行：

```bash
python main.py --name full_experiment --device cpu
```

オプション:
- `--name`: 実験名（出力ディレクトリに使用）
- `--video`: 入力ビデオファイルへのパス（オプション）
- `--device`: 実行デバイス (cpu, cuda, mps)
- `--batch-size`: 処理のバッチサイズ（デフォルト: 5）
- `--output-dir`: 結果保存ディレクトリ

## 実験結果

詳細な実験結果と分析については[実験レポート](REPORT.md)を参照してください。

## データセット詳細

### DatasetsDemo-Drift

- 1,000フレームの照明シフトシーケンス
- 段階的に照明条件が変化するシンセティックなビデオ
- モデルのドリフト回復性能をテストするために設計
- 詳細な照明変化パターンで実世界の環境変化をシミュレート

### Scratch/Oil Toy

- 600枚の合成画像（512x512ピクセル）
- 3種類のテクスチャ背景（タイル、金属、布）
- 2種類の異常タイプ：
  - 線状の傷（異なる角度と長さ）
  - 不規則な形状の油染み
- テクスチャ（ピクセルレベル）と論理的（意味レベル）な検出の比較に使用
- 「油染みを無視する」などの言語指示の効果をテスト

## ライセンス

このプロジェクトはMITライセンスの下で提供されています。LICENSEファイルを参照してください。

## 謝辞

- 異常検出ベンチマーク用のMVTec-AD2、VisA、VIADUCTデータセット
- ディープラーニングフレームワークを提供するPyTorchチーム
- 強力なセグメンテーション機能を持つSegment Anything Model (SAM)