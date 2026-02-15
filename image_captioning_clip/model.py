"""Image Captioning Model: CLIP encoder (frozen) + Projection + Transformer Decoder."""

import math

import torch
import torch.nn as nn
from transformers import CLIPModel


class ImageCaptionModel(nn.Module):
    def __init__(
        self,
        clip_model_name: str,
        vocab_size: int,
        decoder_dim: int = 512,
        decoder_layers: int = 6,
        decoder_heads: int = 8,
        decoder_ff_dim: int = 2048,
        decoder_dropout: float = 0.1,
        clip_embed_dim: int = 768,
        max_length: int = 64,
        pad_token_id: int = 0,
        skip_clip: bool = False,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.decoder_dim = decoder_dim
        self.max_length = max_length
        self.skip_clip = skip_clip

        # --- CLIP Vision Encoder (frozen) ---
        if not skip_clip:
            clip_model = CLIPModel.from_pretrained(clip_model_name, use_safetensors=True)
            self.clip_vision = clip_model.vision_model
            for param in self.clip_vision.parameters():
                param.requires_grad = False
            self.clip_vision.eval()
        else:
            self.clip_vision = None

        # CLIP ViT outputs: (batch, num_patches+1, hidden_dim)
        # We use all patch tokens as memory for cross-attention
        self.projection = nn.Sequential(
            nn.Linear(clip_embed_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )

        # --- Token Embedding + Positional Encoding ---
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_length, decoder_dim)

        # --- Transformer Decoder ---
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ff_dim,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=decoder_layers
        )

        # --- Output projection ---
        self.output_proj = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @torch.no_grad()
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract image features using frozen CLIP.

        Returns: (batch, num_patches+1, decoder_dim)
        """
        if self.clip_vision is None:
            raise RuntimeError("CLIP vision encoder not loaded (skip_clip=True). Use image_features instead.")
        self.clip_vision.eval()
        clip_output = self.clip_vision(pixel_values=pixel_values)
        # last_hidden_state: (batch, seq_len, hidden_dim), seq_len = num_patches + 1 (CLS)
        image_features = clip_output.last_hidden_state
        return self.projection(image_features)

    def decode(
        self,
        caption_ids: torch.Tensor,
        memory: torch.Tensor,
        caption_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode captions given image memory.

        Args:
            caption_ids: (batch, seq_len) token IDs
            memory: (batch, num_patches+1, decoder_dim) image features
            caption_mask: (batch, seq_len) attention mask (1=attend, 0=ignore)

        Returns: (batch, seq_len, vocab_size) logits
        """
        batch_size, seq_len = caption_ids.shape
        device = caption_ids.device

        # Token + positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(caption_ids) + self.pos_embedding(positions)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

        # Padding mask for decoder (True = ignore)
        tgt_key_padding_mask = None
        if caption_mask is not None:
            tgt_key_padding_mask = caption_mask == 0

        # Decode
        output = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        logits = self.output_proj(output)
        return logits

    def forward(
        self,
        pixel_values: torch.Tensor | None = None,
        caption_ids: torch.Tensor = None,
        caption_mask: torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full forward pass: image -> caption logits.

        Args:
            pixel_values: (batch, 3, H, W) CLIP-preprocessed images (used if image_features is None)
            caption_ids: (batch, seq_len) input token IDs (teacher forcing)
            caption_mask: (batch, seq_len) attention mask
            image_features: (batch, seq_len_img, clip_dim) pre-cached CLIP features

        Returns: (batch, seq_len, vocab_size) logits
        """
        if image_features is not None:
            memory = self.projection(image_features)
        else:
            memory = self.encode_image(pixel_values)
        logits = self.decode(caption_ids, memory, caption_mask)
        return logits

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[list[int]]:
        """Autoregressive decoding. Greedy if temperature<=0, else sampling."""
        self.eval()
        memory = self.encode_image(pixel_values)
        batch_size = pixel_values.size(0)
        device = pixel_values.device

        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.decode(generated, memory)
            next_logits = logits[:, -1, :]

            if temperature <= 0:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    topk_vals, _ = next_logits.topk(top_k, dim=-1)
                    threshold = topk_vals[:, -1].unsqueeze(-1)
                    next_logits[next_logits < threshold] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        results = []
        for seq in generated.tolist():
            # Trim after EOS
            if eos_token_id in seq:
                seq = seq[: seq.index(eos_token_id) + 1]
            results.append(seq)
        return results
