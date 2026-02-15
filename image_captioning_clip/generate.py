"""Generate captions for given images using a trained model."""

import argparse
from pathlib import Path

import sentencepiece as spm
import torch
from PIL import Image
from transformers import CLIPProcessor

from config import Config
from model import ImageCaptionModel


def generate_captions(
    model: ImageCaptionModel,
    image_path: str,
    clip_processor: CLIPProcessor,
    sp: spm.SentencePieceProcessor,
    cfg: Config,
    device: torch.device,
    num_captions: int = 1,
    temperature: float = 1.0,
    top_k: int = 0,
) -> list[str]:
    image = Image.open(image_path).convert("RGB")
    pixel_values = clip_processor(images=image, return_tensors="pt").pixel_values
    # Repeat for num_captions (batch generation)
    pixel_values = pixel_values.repeat(num_captions, 1, 1, 1).to(device)

    generated_ids = model.generate(
        pixel_values,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        max_length=cfg.max_gen_length,
        temperature=temperature,
        top_k=top_k,
    )

    captions = []
    for ids in generated_ids:
        ids = [t for t in ids if t not in (cfg.bos_token_id, cfg.eos_token_id)]
        captions.append(sp.decode(ids))
    return captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s)")
    parser.add_argument("--variant", choices=["stair", "snow"], default="stair")
    parser.add_argument("--model_size", choices=["base", "small", "tiny", "micro"], default="base")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer_model_path", type=str, default="", help="Path to SentencePiece model")
    parser.add_argument("--num_captions", type=int, default=1, help="Number of captions per image")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling (0=disabled)")
    parser.add_argument("--vocab_size", type=int, default=None, help="SentencePiece vocab size (default: 8000)")
    args = parser.parse_args()

    kwargs = {"dataset_variant": args.variant, "model_size": args.model_size}
    if args.vocab_size is not None:
        kwargs["sp_vocab_size"] = args.vocab_size
    cfg = Config(**kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer_model_path = args.tokenizer_model_path or cfg.tokenizer_model_path
    sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

    # Load CLIP processor
    clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

    # Load model
    model = ImageCaptionModel(
        clip_model_name=cfg.clip_model_name,
        vocab_size=cfg.sp_vocab_size,
        decoder_dim=cfg.decoder_dim,
        decoder_layers=cfg.decoder_layers,
        decoder_heads=cfg.decoder_heads,
        decoder_ff_dim=cfg.decoder_ff_dim,
        decoder_dropout=cfg.decoder_dropout,
        clip_embed_dim=cfg.clip_embed_dim,
        max_length=cfg.max_caption_length,
        pad_token_id=cfg.pad_token_id,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print(f"Model loaded (epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.4f})")
    print(f"Variant: {cfg.dataset_variant}")
    print(f"Captions per image: {args.num_captions}, temperature: {args.temperature}, top_k: {args.top_k}\n")

    for image_path in args.images:
        captions = generate_captions(
            model, image_path, clip_processor, sp, cfg, device,
            num_captions=args.num_captions,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"{image_path}")
        for i, cap in enumerate(captions, 1):
            print(f"  [{i}] {cap}")
        print()


if __name__ == "__main__":
    main()
