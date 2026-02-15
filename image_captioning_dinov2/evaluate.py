"""Evaluate trained model with BLEU scores on validation set (DINOv2 version)."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import sacrebleu
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor

from config import Config
from model import ImageCaptionModel


class UniqueImageDataset(Dataset):
    """Dataset that loads each unique image only once (no duplicate per caption)."""

    def __init__(self, caption_json: str, image_dir: str, encoder_model_name: str):
        with open(caption_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.image_dir = Path(image_dir)
        self.image_processor = AutoImageProcessor.from_pretrained(encoder_model_name)

        # Deduplicate: one entry per image
        seen = set()
        self.image_ids = []
        self.id_to_filename = {}
        for img in data["images"]:
            if img["id"] not in seen:
                seen.add(img["id"])
                self.image_ids.append(img["id"])
                self.id_to_filename[img["id"]] = img["file_name"]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.id_to_filename[image_id]
        image_subdir = "train2014" if "train" in filename else "val2014"
        image_path = self.image_dir / image_subdir / filename
        image = Image.open(image_path).convert("RGB")

        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        return pixel_values, image_id


def evaluate(cfg: Config, checkpoint_path: str, num_samples: int | None = None, tokenizer_model_path: str = "", batch_size: int | None = None):
    tokenizer_model_path = tokenizer_model_path or cfg.tokenizer_model_path
    print(f"Using tokenizer model: {tokenizer_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

    # Load model
    print("Loading model...")
    model = ImageCaptionModel(
        encoder_model_name=cfg.encoder_model_name,
        vocab_size=cfg.sp_vocab_size,
        decoder_dim=cfg.decoder_dim,
        decoder_layers=cfg.decoder_layers,
        decoder_heads=cfg.decoder_heads,
        decoder_ff_dim=cfg.decoder_ff_dim,
        decoder_dropout=cfg.decoder_dropout,
        encoder_embed_dim=cfg.encoder_embed_dim,
        max_length=cfg.max_caption_length,
        pad_token_id=cfg.pad_token_id,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f}")

    # Load validation data for reference captions
    with open(cfg.val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    # Build image_id -> list of reference captions
    image_refs = defaultdict(list)
    for ann in val_data["annotations"]:
        image_refs[ann["image_id"]].append(ann["caption"])

    # Unique image dataset
    val_dataset = UniqueImageDataset(
        caption_json=cfg.val_json,
        image_dir=cfg.coco_image_dir,
        encoder_model_name=cfg.encoder_model_name,
    )

    if num_samples:
        val_dataset.image_ids = val_dataset.image_ids[:num_samples]

    eval_batch_size = batch_size or cfg.batch_size
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    # Generate captions in batches
    hypotheses = []
    references_list = []

    print(f"Generating captions for {len(val_dataset)} unique images (batch_size={eval_batch_size})...")
    for pixel_values, image_ids in tqdm(val_loader):
        pixel_values = pixel_values.to(device)
        generated_ids = model.generate(
            pixel_values,
            bos_token_id=cfg.bos_token_id,
            eos_token_id=cfg.eos_token_id,
            max_length=cfg.max_gen_length,
        )

        for ids, img_id in zip(generated_ids, image_ids.tolist()):
            ids = [t for t in ids if t not in (cfg.bos_token_id, cfg.eos_token_id)]
            hypothesis = sp.decode(ids)
            hypotheses.append(hypothesis)
            references_list.append(image_refs[img_id])

    # Compute BLEU
    max_refs = max(len(r) for r in references_list)
    refs_transposed = []
    for ref_idx in range(max_refs):
        ref_column = []
        for refs in references_list:
            if ref_idx < len(refs):
                ref_column.append(refs[ref_idx])
            else:
                ref_column.append("")
        refs_transposed.append(ref_column)

    bleu = sacrebleu.corpus_bleu(hypotheses, refs_transposed, tokenize="ja-mecab")

    print(f"\n=== Evaluation Results ({cfg.dataset_variant}) ===")
    print(f"Images evaluated: {len(hypotheses)}")
    print(f"BLEU: {bleu}")

    # Show some examples
    print("\n--- Sample Generations ---")
    for i in range(min(10, len(hypotheses))):
        print(f"\nHypothesis: {hypotheses[i]}")
        print(f"References: {references_list[i][:3]}")

    # Append BLEU result to eval CSV
    results_csv = Path(cfg.output_dir) / f"eval_results_{cfg.dataset_variant}.csv"
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["variant", "model_size", "vocab_size",
                         "bleu", "bleu1", "bleu2", "bleu3", "bleu4",
                         "num_images", "checkpoint"])
        w.writerow([cfg.dataset_variant, cfg.model_size, cfg.sp_vocab_size,
                     f"{bleu.score:.2f}",
                     f"{bleu.precisions[0]:.2f}", f"{bleu.precisions[1]:.2f}",
                     f"{bleu.precisions[2]:.2f}", f"{bleu.precisions[3]:.2f}",
                     len(hypotheses), checkpoint_path])

    return bleu


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["stair", "snow"], default="stair")
    parser.add_argument("--model_size", choices=["base", "small", "tiny", "micro"], default="base")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of images to evaluate")
    parser.add_argument("--tokenizer_model_path", type=str, default="", help="Path to SentencePiece model")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for evaluation")
    parser.add_argument("--vocab_size", type=int, default=None, help="SentencePiece vocab size (default: 8000)")
    args = parser.parse_args()

    kwargs = {"dataset_variant": args.variant, "model_size": args.model_size}
    if args.vocab_size is not None:
        kwargs["sp_vocab_size"] = args.vocab_size
    cfg = Config(**kwargs)
    evaluate(cfg, args.checkpoint, args.num_samples, args.tokenizer_model_path, args.batch_size)
