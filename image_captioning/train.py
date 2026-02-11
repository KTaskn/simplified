"""Training loop for Japanese Image Captioning model."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import CocoCaptionDataset
from model import ImageCaptionModel


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_token_id: int) -> float:
    """Compute token-level accuracy, ignoring padding tokens."""
    preds = logits.argmax(dim=-1).reshape(-1)
    targets_flat = targets.reshape(-1)
    mask = targets_flat != pad_token_id
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == targets_flat[mask]).sum().item()
    return correct / mask.sum().item()


def train_one_epoch(
    model: ImageCaptionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple[float, float, int]:
    model.train()
    # Keep CLIP frozen
    model.clip_vision.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
    for pixel_values, caption_ids, caption_mask in pbar:
        pixel_values = pixel_values.to(device)
        caption_ids = caption_ids.to(device)
        caption_mask = caption_mask.to(device)

        # Input: all tokens except last; Target: all tokens except first
        input_ids = caption_ids[:, :-1]
        target_ids = caption_ids[:, 1:]
        input_mask = caption_mask[:, :-1]

        optimizer.zero_grad()

        with autocast(enabled=cfg.use_amp):
            logits = model(pixel_values, input_ids, input_mask)
            # logits: (batch, seq_len, vocab_size)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accuracy (computed without grad)
        with torch.no_grad():
            preds = logits.argmax(dim=-1).reshape(-1)
            targets_flat = target_ids.reshape(-1)
            mask = targets_flat != cfg.pad_token_id
            total_correct += (preds[mask] == targets_flat[mask]).sum().item()
            total_tokens += mask.sum().item()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        batch_acc = compute_accuracy(logits, target_ids, cfg.pad_token_id)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{batch_acc:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

        if global_step % 100 == 0:
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/accuracy", batch_acc, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

    avg_loss = total_loss / num_batches
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc, global_step


@torch.no_grad()
def validate(
    model: ImageCaptionModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    cfg: Config,
    epoch: int,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]")
    for pixel_values, caption_ids, caption_mask in pbar:
        pixel_values = pixel_values.to(device)
        caption_ids = caption_ids.to(device)
        caption_mask = caption_mask.to(device)

        input_ids = caption_ids[:, :-1]
        target_ids = caption_ids[:, 1:]
        input_mask = caption_mask[:, :-1]

        with autocast(enabled=cfg.use_amp):
            logits = model(pixel_values, input_ids, input_mask)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

        # Accuracy
        preds = logits.argmax(dim=-1).reshape(-1)
        targets_flat = target_ids.reshape(-1)
        mask = targets_flat != cfg.pad_token_id
        total_correct += (preds[mask] == targets_flat[mask]).sum().item()
        total_tokens += mask.sum().item()

        total_loss += loss.item()
        num_batches += 1

        batch_acc = compute_accuracy(logits, target_ids, cfg.pad_token_id)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}")

    avg_loss = total_loss / num_batches
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["stair", "snow"], default="stair")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = Config(dataset_variant=args.variant)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset variant: {cfg.dataset_variant}")

    # Output dirs
    run_dir = Path(cfg.output_dir) / f"run_{cfg.dataset_variant}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Datasets
    print("Loading datasets...")
    train_dataset = CocoCaptionDataset(
        caption_json=cfg.train_json,
        image_dir=cfg.coco_image_dir,
        sp_model_path=cfg.tokenizer_model_path,
        clip_model_name=cfg.clip_model_name,
        max_length=cfg.max_caption_length,
        split="train",
    )
    val_dataset = CocoCaptionDataset(
        caption_json=cfg.val_json,
        image_dir=cfg.coco_image_dir,
        sp_model_path=cfg.tokenizer_model_path,
        clip_model_name=cfg.clip_model_name,
        max_length=cfg.max_caption_length,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Model
    print("Building model...")
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

    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    total_steps = len(train_loader) * cfg.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        total_steps=total_steps,
        pct_start=cfg.warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)
    scaler = GradScaler(enabled=cfg.use_amp)

    writer = SummaryWriter(log_dir=str(run_dir / "logs"))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]

    # Training loop
    print(f"\nStarting training for {cfg.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {cfg.batch_size}, Total steps: {total_steps}\n")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, cfg, epoch, writer, global_step,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, cfg, epoch)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1}/{cfg.epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Time: {elapsed:.1f}s"
        )

        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/train_accuracy", train_acc, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/val_accuracy", val_acc, epoch)

        # Save checkpoint
        ckpt_data = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "config": vars(cfg),
        }

        # Save latest
        torch.save(ckpt_data, ckpt_dir / "latest.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, ckpt_dir / "best.pt")
            print(f"  -> New best val loss: {best_val_loss:.4f}")

    writer.close()
    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in {ckpt_dir}")


if __name__ == "__main__":
    main()
