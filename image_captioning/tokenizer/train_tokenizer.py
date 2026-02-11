"""Train a SentencePiece tokenizer on STAIR/Snow captions."""

import argparse
import json
import tempfile
from pathlib import Path

import sentencepiece as spm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config


def extract_captions(json_path: str) -> list[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [ann["caption"] for ann in data["annotations"]]


def train_tokenizer(cfg: Config):
    output_dir = Path(cfg.output_dir) / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect captions from both train and val
    captions = extract_captions(cfg.train_json)
    captions += extract_captions(cfg.val_json)
    print(f"Total captions: {len(captions)}")

    # Write captions to a temp file for SentencePiece
    tmp_path = output_dir / "captions_corpus.txt"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for cap in captions:
            f.write(cap.strip() + "\n")

    model_prefix = str(output_dir / cfg.tokenizer_prefix)

    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=model_prefix,
        vocab_size=cfg.sp_vocab_size,
        model_type=cfg.sp_model_type,
        pad_id=cfg.pad_token_id,
        bos_id=cfg.bos_token_id,
        eos_id=cfg.eos_token_id,
        unk_id=cfg.unk_token_id,
        character_coverage=0.9995,
        num_threads=8,
    )

    print(f"Tokenizer saved to {model_prefix}.model")

    # Verify
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    test_sentences = [
        "山の中を赤い電車が走っている",
        "男がスケートボードでジャンプしている",
        "犬が公園で遊んでいる",
    ]
    print("\n--- Tokenizer verification ---")
    for sent in test_sentences:
        ids = sp.encode(sent, out_type=int)
        pieces = sp.encode(sent, out_type=str)
        decoded = sp.decode(ids)
        print(f"Original:  {sent}")
        print(f"Pieces:    {pieces}")
        print(f"IDs:       {ids}")
        print(f"Decoded:   {decoded}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["stair", "snow"], default="stair",
                        help="Dataset variant to train tokenizer on")
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    cfg = Config(dataset_variant=args.variant, sp_vocab_size=args.vocab_size)
    train_tokenizer(cfg)
