# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Japanese Image Captioning Model using frozen CLIP (ViT-B/32) encoder + Transformer Decoder. Trained on STAIR Captions v1.2 dataset (original and "Snow" simplified Japanese variant using Easy Japanese 2,000 word vocabulary).

### Architecture

```
[Image] → CLIP ViT-B/32 (frozen) → Projection (Linear+LN) → Transformer Decoder → [Japanese Caption]
```

- **Image Encoder**: CLIP ViT-B/32 (768-dim, frozen during training)
- **Projection Layer**: Linear(768 → decoder_dim) + LayerNorm (trainable)
- **Text Decoder**: Transformer Decoder with configurable size (trainable)
- **Tokenizer**: SentencePiece unigram model (trainable, vocab_size configurable)

### Model Size Presets

Defined in [config.py:9-14](image_captioning/config.py#L9-L14):
- `base`: 512-dim, 6 layers, 8 heads (~30M params)
- `small`: 384-dim, 4 layers, 6 heads (~12M params)
- `tiny`: 256-dim, 3 layers, 4 heads (~5M params)
- `micro`: 128-dim, 2 layers, 4 heads (~1.5M params)

## Common Commands

All commands should be run from `image_captioning/` directory unless noted.

### Setup

```bash
# Install dependencies
pip install -r image_captioning/requirements.txt

# Download COCO images (train2014 ~13GB + val2014 ~6GB)
bash download_coco.sh coco_images
```

### Training Pipeline

```bash
# 1. Train tokenizer (required first)
python tokenizer/train_tokenizer.py --variant snow --vocab_size 4000

# 2. (Optional) Cache CLIP features to speed up training
python cache_features.py --variant snow --split both

# 3. Train model
python train.py --variant snow --model_size tiny --vocab_size 4000 --use_cache --auto_batch

# With custom hyperparameters
python train.py --variant snow --model_size tiny --vocab_size 4000 \
    --epochs 30 --batch_size 1024 --lr 5e-4 \
    --label_smoothing 0.1 --patience 3 --use_cache --auto_batch

# Resume from checkpoint
python train.py --variant snow --model_size tiny --vocab_size 4000 \
    --resume ../outputs/run_snow_tiny_v4000/checkpoints/latest.pt --use_cache
```

### Evaluation and Inference

```bash
# Evaluate with BLEU on validation set
python evaluate.py --variant snow --model_size tiny --vocab_size 4000 \
    --checkpoint ../outputs/run_snow_tiny_v4000/checkpoints/best.pt

# Limit evaluation samples for faster testing
python evaluate.py --variant snow --model_size tiny \
    --checkpoint ../outputs/run_snow_tiny/checkpoints/best.pt --num_samples 100

# Generate captions for arbitrary images
python generate.py --variant snow --model_size tiny \
    --checkpoint ../outputs/run_snow_tiny/checkpoints/best.pt \
    path/to/image.jpg

# Generate multiple captions with sampling
python generate.py --variant snow --model_size tiny \
    --checkpoint ../outputs/run_snow_tiny/checkpoints/best.pt \
    --num_captions 5 --temperature 0.8 --top_k 50 \
    img1.jpg img2.jpg
```

### Grid Search

Run comprehensive model_size × vocab_size experiments:

```bash
# Edit run_grid.sh to configure search space
bash run_grid.sh snow

# Results saved to ../outputs/grid_results_snow.csv and eval_results_snow.csv
```

### Monitoring

```bash
# View training progress with TensorBoard
tensorboard --logdir ../outputs/
```

## Key Implementation Details

### Dataset Variants

- **`stair`**: Original STAIR Captions v1.2 (natural Japanese)
- **`snow`**: Simplified Japanese version using 2,000-word Easy Japanese vocabulary

Specified via `--variant` flag. JSON paths auto-resolved in [config.py:80-89](image_captioning/config.py#L80-L89).

### Cached vs Non-Cached Training

The codebase supports two training modes:

1. **Non-cached** (`CocoCaptionDataset`): Loads images on-the-fly, runs CLIP encoding during training
2. **Cached** (`CachedCaptionDataset` + `--use_cache`): Pre-computes all CLIP features as float16 tensors

Cached mode significantly speeds up training (~3-5x faster) and reduces GPU memory for CLIP, allowing larger batch sizes. Always use `cache_features.py` before `--use_cache` flag.

### Auto Batch Sizing

`--auto_batch` flag performs binary search to find maximum batch size that fits in GPU memory. Uses safety factor of 0.85 to avoid OOM during training. Implementation in [train.py:31-110](image_captioning/train.py#L31-L110).

### Output Structure

All outputs saved to `../outputs/`:
- Tokenizers: `../outputs/tokenizer/sp_{variant}_{vocab_size}.model`
- Checkpoints: `../outputs/run_{variant}_{model_size}_v{vocab_size}/checkpoints/{best,latest}.pt`
- TensorBoard: `../outputs/run_{variant}_{model_size}_v{vocab_size}/logs/`
- Grid results: `../outputs/grid_results_{variant}.csv` and `eval_results_{variant}.csv`

Run name format defined in [config.py:104-108](image_captioning/config.py#L104-L108).

### Special Tokens

Defined in [config.py:42-46](image_captioning/config.py#L42-L46):
- PAD: 0
- BOS: 1
- EOS: 2
- UNK: 3

### Model Architecture Notes

- CLIP vision encoder is frozen (`requires_grad=False`) and set to eval mode during training
- Cross-attention uses all CLIP patch tokens (49 patches + 1 CLS token = 50 total)
- Decoder uses causal masking for autoregressive generation
- Teacher forcing during training: input is `tokens[:-1]`, target is `tokens[1:]`

### Training Features

- **AMP**: Mixed precision training enabled by default (`use_amp=True`)
- **Label smoothing**: Configurable via `--label_smoothing` (default 0.1)
- **Early stopping**: Configurable via `--patience` (default 3 epochs)
- **Gradient clipping**: Fixed at 1.0
- **Scheduler**: OneCycleLR with cosine annealing
- **Optimizer**: AdamW with weight_decay=0.01

### Generation Modes

In [model.py:158-206](image_captioning/model.py#L158-L206):
- Greedy decoding: `temperature <= 0`
- Sampling: `temperature > 0` with optional `top_k` filtering

## Important Path Conventions

- COCO images must be in subdirectories: `coco_images/train2014/` and `coco_images/val2014/`
- Caption JSONs follow COCO annotation format with `images` and `annotations` keys
- Image filenames in JSONs automatically determine train2014 vs val2014 subdirectory

## Common Issues

1. **Tokenizer not found**: Run `tokenizer/train_tokenizer.py` first before training
2. **Cache not found with `--use_cache`**: Run `cache_features.py` first
3. **OOM during training**: Use `--auto_batch` flag or manually reduce `--batch_size`
4. **Mismatched vocab_size**: Ensure `--vocab_size` matches between tokenizer training, model training, and evaluation