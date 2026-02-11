# 日本語画像キャプション生成モデル

CLIP (frozen) + Transformer Decoder による画像→日本語キャプション生成モデル。
STAIR Captions / やさしい日本語版 STAIR Captions を学習データとして使用。

## アーキテクチャ

```
[画像] → CLIP ViT-B/32 (frozen) → 射影層 (Linear+LN) → Transformer Decoder (6層) → [日本語キャプション]
```

| コンポーネント | 詳細 | 学習 |
|-------------|------|------|
| 画像エンコーダ | CLIP ViT-B/32 (768次元) | frozen |
| 射影層 | Linear(768→512) + LayerNorm | ゼロから |
| テキストデコーダ | Transformer Decoder 6層, 8head, ff=2048 | ゼロから |
| トークナイザ | SentencePiece unigram (vocab=8000) | ゼロから |

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
```

出力先: `../outputs/tokenizer/`

### 4. モデルの学習

```bash
# オリジナル版
python train.py --variant stair

# 平易化版
python train.py --variant snow
```

オプション:
- `--epochs 30` : エポック数
- `--batch_size 64` : バッチサイズ
- `--lr 1e-4` : 学習率
- `--resume path/to/checkpoint.pt` : チェックポイントから再開

チェックポイント・TensorBoardログの出力先: `../outputs/run_{variant}/`

### 5. TensorBoardで学習経過を確認

```bash
tensorboard --logdir ../outputs/run_stair/logs
```

## 評価・推論

### 6. BLEU評価

```bash
python evaluate.py --variant stair --checkpoint ../outputs/run_stair/checkpoints/best.pt
```

`--num_samples 100` で評価画像数を制限可能。

### 7. 任意の画像でキャプション生成

```bash
python generate.py --variant stair --checkpoint ../outputs/run_stair/checkpoints/best.pt path/to/image.jpg
```

複数画像の指定も可能:

```bash
python generate.py --checkpoint ../outputs/run_stair/checkpoints/best.pt img1.jpg img2.jpg img3.jpg
```

## ファイル構成

```
image_captioning/
├── config.py                  # ハイパーパラメータ・パス設定
├── tokenizer/
│   └── train_tokenizer.py     # SentencePieceトークナイザ学習
├── dataset.py                 # PyTorch Dataset
├── model.py                   # CLIP + 射影層 + Transformer Decoder
├── train.py                   # 学習ループ
├── evaluate.py                # BLEU評価
├── generate.py                # 推論
├── requirements.txt           # 依存パッケージ
└── README.md
```

## 必要環境

- Python 3.10+
- CUDA対応GPU (VRAM 24GB推奨: RTX 3090/4090)
- ディスク容量: COCO画像 ~19GB + モデルチェックポイント
