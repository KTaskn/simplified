from dataclasses import dataclass, field
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- Paths ---
    coco_image_dir: str = str(BASE_DIR / "coco_images")
    stair_train_json: str = str(BASE_DIR / "stair_captions_v1.2" / "stair_captions_v1.2_train.json")
    stair_val_json: str = str(BASE_DIR / "stair_captions_v1.2" / "stair_captions_v1.2_val.json")
    snow_train_json: str = str(BASE_DIR / "stair_captions_v1.2_snow_simplified" / "train.json")
    snow_val_json: str = str(BASE_DIR / "stair_captions_v1.2_snow_simplified" / "val.json")
    output_dir: str = str(BASE_DIR / "outputs")

    # --- Dataset variant: "stair" or "snow" ---
    dataset_variant: str = "stair"

    # --- CLIP ---
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embed_dim: int = 768  # ViT-B/32 output dim

    # --- Tokenizer ---
    sp_vocab_size: int = 8000
    sp_model_type: str = "unigram"
    max_caption_length: int = 64

    # Special token IDs (set after tokenizer training)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # --- Model ---
    decoder_dim: int = 512
    decoder_layers: int = 6
    decoder_heads: int = 8
    decoder_ff_dim: int = 2048
    decoder_dropout: float = 0.1

    # --- Training ---
    batch_size: int = 1024
    num_workers: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 30
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    use_amp: bool = True

    # --- Generation ---
    beam_size: int = 5
    max_gen_length: int = 64

    @property
    def train_json(self) -> str:
        if self.dataset_variant == "snow":
            return self.snow_train_json
        return self.stair_train_json

    @property
    def val_json(self) -> str:
        if self.dataset_variant == "snow":
            return self.snow_val_json
        return self.stair_val_json

    @property
    def tokenizer_prefix(self) -> str:
        return f"sp_{self.dataset_variant}_{self.sp_vocab_size}"

    @property
    def tokenizer_model_path(self) -> str:
        return str(Path(self.output_dir) / "tokenizer" / f"{self.tokenizer_prefix}.model")
