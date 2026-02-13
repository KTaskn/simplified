"""PyTorch Dataset for COCO images + STAIR/Snow Japanese captions."""

import json
from pathlib import Path

import sentencepiece as spm
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class CocoCaptionDataset(Dataset):
    """Dataset that pairs COCO images with Japanese captions.

    Each __getitem__ returns one (image, caption) pair.
    Since each image has ~5 captions, the dataset size equals the number of annotations.
    """

    def __init__(
        self,
        caption_json: str,
        image_dir: str,
        sp_model_path: str,
        clip_model_name: str,
        max_length: int = 64,
        split: str = "train",
    ):
        with open(caption_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.annotations = data["annotations"]
        # Build image_id -> file_name mapping
        self.id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

        self.image_dir = Path(image_dir)
        self.split = split
        self.max_length = max_length

        # SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        # CLIP image processor
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        caption = ann["caption"]

        # Load image
        filename = self.id_to_filename[image_id]
        # COCO images are in train2014/ or val2014/ subdirectories
        image_subdir = "train2014" if "train" in filename else "val2014"
        image_path = self.image_dir / image_subdir / filename
        image = Image.open(image_path).convert("RGB")

        # Process image for CLIP
        pixel_values = self.clip_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenize caption: <bos> tokens <eos>
        token_ids = self.sp.encode(caption, out_type=int)
        token_ids = [self.bos_id] + token_ids + [self.eos_id]

        # Truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length - 1] + [self.eos_id]

        # Pad
        attention_mask = [1] * len(token_ids)
        pad_len = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        caption_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return pixel_values, caption_ids, attention_mask


class CachedCaptionDataset(Dataset):
    """Dataset using pre-cached CLIP features instead of raw images.

    Expects a cache file (dict: image_id -> tensor(50, 768) in float16)
    created by cache_features.py.
    """

    def __init__(
        self,
        caption_json: str,
        cache_path: str,
        sp_model_path: str,
        max_length: int = 64,
    ):
        with open(caption_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.annotations = data["annotations"]
        self.max_length = max_length

        # SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        # Load cached CLIP features into memory
        self.cache = torch.load(cache_path, map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        caption = ann["caption"]

        # Retrieve cached CLIP features (float16 -> float32)
        image_features = self.cache[image_id].float()

        # Tokenize caption: <bos> tokens <eos>
        token_ids = self.sp.encode(caption, out_type=int)
        token_ids = [self.bos_id] + token_ids + [self.eos_id]

        # Truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[: self.max_length - 1] + [self.eos_id]

        # Pad
        attention_mask = [1] * len(token_ids)
        pad_len = self.max_length - len(token_ids)
        token_ids = token_ids + [self.pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

        caption_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return image_features, caption_ids, attention_mask
