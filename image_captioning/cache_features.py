"""Pre-compute and cache CLIP image features for faster training."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from config import Config


class CocoImageDataset(Dataset):
    """Simple dataset that loads COCO images by image_id."""

    def __init__(self, image_ids: list[int], id_to_filename: dict[int, str],
                 image_dir: str, clip_processor: CLIPProcessor):
        self.image_ids = image_ids
        self.id_to_filename = id_to_filename
        self.image_dir = Path(image_dir)
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        filename = self.id_to_filename[image_id]
        image_subdir = "train2014" if "train" in filename else "val2014"
        image_path = self.image_dir / image_subdir / filename
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)
        return image_id, pixel_values


def cache_features(cfg: Config, split: str, batch_size: int = 256):
    cache_dir = Path(cfg.output_dir) / "clip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}.pt"

    if cache_path.exists():
        print(f"Cache already exists: {cache_path}")
        print("Delete it to regenerate.")
        return

    # Load image list from annotation JSON
    json_path = cfg.train_json if split == "train" else cfg.val_json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_ids = list(id_to_filename.keys())
    print(f"Split: {split}, Images: {len(image_ids)}")

    # Load CLIP
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(cfg.clip_model_name, use_safetensors=True)
    clip_vision = clip_model.vision_model.to(device)
    clip_vision.eval()
    del clip_model

    clip_processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

    dataset = CocoImageDataset(image_ids, id_to_filename, cfg.coco_image_dir, clip_processor)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    # Extract features
    cache = {}
    with torch.no_grad():
        for batch_ids, pixel_values in tqdm(loader, desc=f"Caching {split}"):
            pixel_values = pixel_values.to(device)
            output = clip_vision(pixel_values=pixel_values)
            features = output.last_hidden_state.half().cpu()  # (B, 50, 768) in float16

            for i, img_id in enumerate(batch_ids.tolist()):
                cache[img_id] = features[i]

    print(f"Cached {len(cache)} images -> {cache_path}")
    torch.save(cache, cache_path)

    # Report size
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Cache file size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "both"], default="both")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--variant", choices=["stair", "snow"], default="stair",
                        help="Dataset variant (for JSON path resolution)")
    args = parser.parse_args()

    cfg = Config(dataset_variant=args.variant)

    if args.split in ("train", "both"):
        cache_features(cfg, "train", args.batch_size)
    if args.split in ("val", "both"):
        cache_features(cfg, "val", args.batch_size)
