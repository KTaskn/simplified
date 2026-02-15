# 日本語画像キャプション生成モデル

CLIP (frozen) + Transformer Decoder による画像→日本語キャプション生成モデル。
STAIR Captions / やさしい日本語版 STAIR Captions を学習データとして使用。

## アーキテクチャ

```
[画像] → CLIP ViT-B/32 (frozen) → 射影層 (Linear+LN) → Transformer Decoder → [日本語キャプション]
```

| コンポーネント | 詳細 | 学習 |
|-------------|------|------|
| 画像エンコーダ | CLIP ViT-B/32 (768次元) | frozen |
| 射影層 | Linear(768→decoder_dim) + LayerNorm | ゼロから |
| テキストデコーダ | Transformer Decoder (サイズ可変) | ゼロから |
| トークナイザ | SentencePiece unigram (vocab_size可変) | ゼロから |

### モデルサイズプリセット

| サイズ | decoder_dim | layers | heads | ff_dim | パラメータ数(概算) |
|--------|-------------|--------|-------|--------|------------------|
| base   | 512         | 6      | 8     | 2048   | ~30M             |
| small  | 384         | 4      | 6     | 1536   | ~12M             |
| tiny   | 256         | 3      | 4     | 1024   | ~5M              |
| micro  | 128         | 2      | 4     | 512    | ~1.5M            |

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. COCO画像のダウンロード

train2014 (~13GB) + val2014 (~6GB) をダウンロードする。

```bash
bash ../download_coco.sh ../coco_images
```

## 学習手順

### 3. トークナイザの学習

```bash
# オリジナル版 (STAIR Captions)
python tokenizer/train_tokenizer.py --variant stair

# 平易化版 (Snow Simplified)
python tokenizer/train_tokenizer.py --variant snow

# vocab_sizeを指定
python tokenizer/train_tokenizer.py --variant snow --vocab_size 4000
```

出力先: `../outputs/tokenizer/`

### 4. モデルの学習

```bash
# オリジナル版 (baseサイズ)
python train.py --variant stair

# 平易化版 (tinyサイズ, vocab_size=4000)
python train.py --variant snow --model_size tiny --vocab_size 4000
```

オプション:
- `--model_size {base,small,tiny,micro}` : モデルサイズ (default: base)
- `--vocab_size 4000` : トークナイザのvocab_size (default: 8000)
- `--epochs 30` : エポック数
- `--batch_size 1024` : バッチサイズ
- `--lr 5e-4` : 学習率
- `--resume path/to/checkpoint.pt` : チェックポイントから再開

チェックポイント・TensorBoardログの出力先: `../outputs/{run_name}/`
- run_name例: `run_snow_tiny`, `run_snow_tiny_v4000`

### 5. TensorBoardで学習経過を確認

```bash
tensorboard --logdir ../outputs/
```

## 評価・推論

### 6. BLEU評価

```bash
python evaluate.py --variant snow --model_size tiny --checkpoint ../outputs/run_snow_tiny/checkpoints/best.pt
```

オプション:
- `--num_samples 100` : 評価画像数を制限
- `--batch_size 512` : 評価時のバッチサイズ
- `--vocab_size 4000` : vocab_sizeが異なる場合
- `--tokenizer_model_path path/to/sp.model` : トークナイザを直接指定

### 7. 任意の画像でキャプション生成

```bash
python generate.py --variant snow --model_size tiny \
    --checkpoint ../outputs/run_snow_tiny/checkpoints/best.pt \
    path/to/image.jpg
```

複数キャプション生成:
```bash
python generate.py --variant snow --model_size tiny \
    --checkpoint ../outputs/run_snow_tiny/checkpoints/best.pt \
    --num_captions 5 --temperature 0.8 --top_k 50 \
    img1.jpg img2.jpg
```

## グリッドサーチ

モデルサイズ × vocab_size の全組み合わせを一括で学習・評価する。

```bash
bash run_grid.sh snow
```

スクリプト内の配列を編集して実験範囲を調整可能:
```bash
MODEL_SIZES=("base" "small" "tiny" "micro")
VOCAB_SIZES=(8000 4000 2000 1000)
```

## ファイル構成

```
image_captioning/
├── config.py                  # ハイパーパラメータ・パス設定・モデルサイズプリセット
├── tokenizer/
│   └── train_tokenizer.py     # SentencePieceトークナイザ学習
├── dataset.py                 # PyTorch Dataset
├── model.py                   # CLIP + 射影層 + Transformer Decoder
├── train.py                   # 学習ループ
├── evaluate.py                # BLEU評価
├── generate.py                # 推論 (greedy / temperature sampling)
├── run_grid.sh                # モデルサイズ×vocab_sizeグリッドサーチ
├── requirements.txt           # 依存パッケージ
└── README.md
```

## 必要環境

- Python 3.10+
- CUDA対応GPU (VRAM 24GB推奨: RTX 3090/4090)
- ディスク容量: COCO画像 ~19GB + モデルチェックポイント
