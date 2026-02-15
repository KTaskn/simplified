"""Training loop for Japanese Image Captioning model."""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import CachedCaptionDataset, CocoCaptionDataset
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


def find_max_batch_size(
    model: ImageCaptionModel,
    device: torch.device,
    cfg: Config,
    use_cache: bool,
    min_bs: int = 256,
    max_bs: int = 32768,
    safety_factor: float = 0.85,
) -> int:
    """Binary search for the maximum batch size that fits in GPU memory.

    Uses a realistic loss (CrossEntropyLoss) to match actual training memory usage.
    Applies safety_factor to avoid borderline OOM during real training.
    """
    from torch.cuda.amp import autocast

    seq_len = cfg.max_caption_length - 1  # input length after shift
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    def try_batch(bs: int) -> bool:
        torch.cuda.empty_cache()
        try:
            model.train()
            if not use_cache and model.clip_vision is not None:
                model.clip_vision.eval()

            if use_cache:
                dummy_img = torch.randn(bs, 50, cfg.clip_embed_dim, device=device)
            else:
                dummy_img = torch.randn(bs, 3, 224, 224, device=device)
            dummy_ids = torch.randint(0, cfg.sp_vocab_size, (bs, seq_len), device=device)
            dummy_targets = torch.randint(0, cfg.sp_vocab_size, (bs, seq_len), device=device)
            dummy_mask = torch.ones(bs, seq_len, device=device)

            with autocast(enabled=cfg.use_amp):
                if use_cache:
                    logits = model(caption_ids=dummy_ids, caption_mask=dummy_mask,
                                   image_features=dummy_img)
                else:
                    logits = model(pixel_values=dummy_img, caption_ids=dummy_ids,
                                   caption_mask=dummy_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), dummy_targets.reshape(-1))
            loss.backward()

            # Clean up explicitly
            del dummy_img, dummy_ids, dummy_targets, dummy_mask, logits, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                return False
            raise

    # Binary search (in steps of 256)
    low, high = min_bs, max_bs
    best_bs = min_bs

    while low <= high:
        mid = (low + high) // 2
        mid = max(min_bs, (mid // 256) * 256)
        if mid <= best_bs and low > min_bs:
            break
        print(f"  Trying batch_size={mid}...", end=" ", flush=True)
        if try_batch(mid):
            print("OK")
            best_bs = mid
            low = mid + 256
        else:
            print("OOM")
            high = mid - 256

    # Apply safety margin
    safe_bs = max(min_bs, (int(best_bs * safety_factor) // 256) * 256)
    if safe_bs != best_bs:
        print(f"  Applying {int(safety_factor*100)}% safety margin: {best_bs} -> {safe_bs}")

    return safe_bs


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
    use_cache: bool = False,
) -> tuple[float, float, int]:
    model.train()
    if not use_cache:
        # Keep CLIP frozen
        model.clip_vision.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
    for batch_images, caption_ids, caption_mask in pbar:
        batch_images = batch_images.to(device)
        caption_ids = caption_ids.to(device)
        caption_mask = caption_mask.to(device)

        # Input: all tokens except last; Target: all tokens except first
        input_ids = caption_ids[:, :-1]
        target_ids = caption_ids[:, 1:]
        input_mask = caption_mask[:, :-1]

        optimizer.zero_grad()

        with autocast(enabled=cfg.use_amp):
            if use_cache:
                logits = model(caption_ids=input_ids, caption_mask=input_mask,
                               image_features=batch_images)
            else:
                logits = model(pixel_values=batch_images, caption_ids=input_ids,
                               caption_mask=input_mask)
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
    use_cache: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]")
    for batch_images, caption_ids, caption_mask in pbar:
        batch_images = batch_images.to(device)
        caption_ids = caption_ids.to(device)
        caption_mask = caption_mask.to(device)

        input_ids = caption_ids[:, :-1]
        target_ids = caption_ids[:, 1:]
        input_mask = caption_mask[:, :-1]

        with autocast(enabled=cfg.use_amp):
            if use_cache:
                logits = model(caption_ids=input_ids, caption_mask=input_mask,
                               image_features=batch_images)
            else:
                logits = model(pixel_values=batch_images, caption_ids=input_ids,
                               caption_mask=input_mask)
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
    parser.add_argument("--model_size", choices=["base", "small", "tiny", "micro"], default="base")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vocab_size", type=int, default=None, help="SentencePiece vocab size (default: 8000)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing (0=disabled)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (0=disabled)")
    parser.add_argument("--use_cache", action="store_true", help="Use pre-cached CLIP features")
    parser.add_argument("--auto_batch", action="store_true", help="Auto-find max batch size for GPU")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    kwargs = {"dataset_variant": args.variant, "model_size": args.model_size}
    if args.vocab_size is not None:
        kwargs["sp_vocab_size"] = args.vocab_size
    cfg = Config(**kwargs)
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.learning_rate = args.lr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset variant: {cfg.dataset_variant}, Model size: {cfg.model_size}")
    print(f"Decoder: dim={cfg.decoder_dim}, layers={cfg.decoder_layers}, heads={cfg.decoder_heads}, ff={cfg.decoder_ff_dim}")

    # Output dirs
    run_dir = Path(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Datasets
    use_cache = args.use_cache
    if use_cache:
        cache_dir = Path(cfg.output_dir) / "clip_cache"
        train_cache = cache_dir / "train.pt"
        val_cache = cache_dir / "val.pt"
        if not train_cache.exists() or not val_cache.exists():
            print("ERROR: Cache files not found. Run cache_features.py first.")
            print(f"  Expected: {train_cache}, {val_cache}")
            return
        print("Loading datasets (cached CLIP features)...")
        train_dataset = CachedCaptionDataset(
            caption_json=cfg.train_json,
            cache_path=str(train_cache),
            sp_model_path=cfg.tokenizer_model_path,
            max_length=cfg.max_caption_length,
        )
        val_dataset = CachedCaptionDataset(
            caption_json=cfg.val_json,
            cache_path=str(val_cache),
            sp_model_path=cfg.tokenizer_model_path,
            max_length=cfg.max_caption_length,
        )
        # Cached features are in-memory, fewer workers needed
        num_workers = min(cfg.num_workers, 4)
    else:
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
        num_workers = cfg.num_workers

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
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
        skip_clip=use_cache,
    ).to(device)

    # Only optimize non-frozen parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,}")

    # Auto batch size
    if args.auto_batch and device.type == "cuda":
        print("Auto-detecting max batch size...")
        optimal_bs = find_max_batch_size(model, device, cfg, use_cache)
        print(f"Auto batch size: {optimal_bs} (was {cfg.batch_size})")
        cfg.batch_size = optimal_bs
        # Rebuild DataLoaders with new batch size
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )

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

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id, label_smoothing=args.label_smoothing)
    scaler = GradScaler(enabled=cfg.use_amp)

    writer = SummaryWriter(log_dir=str(run_dir / "logs"))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt["best_val_loss"]

    # Training loop
    patience = args.patience
    patience_counter = 0
    print(f"\nStarting training for {cfg.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Batch size: {cfg.batch_size}, Total steps: {total_steps}")
    print(f"Label smoothing: {args.label_smoothing}, Early stopping patience: {patience}\n")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()

        train_loss, train_acc, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            criterion, device, cfg, epoch, writer, global_step, use_cache,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, cfg, epoch, use_cache)

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
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_loss": best_val_loss,
            "config": vars(cfg),
        }

        # Save latest
        torch.save(ckpt_data, ckpt_dir / "latest.pt")

        # Save best + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, ckpt_dir / "best.pt")
            print(f"  -> New best val loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience > 0 and patience_counter >= patience:
                print(f"  -> Early stopping: val_loss did not improve for {patience} epochs")
                break

    writer.close()
    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in {ckpt_dir}")

    # Append result to grid CSV
    results_csv = Path(cfg.output_dir) / f"grid_results_{cfg.dataset_variant}.csv"
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["variant", "model_size", "vocab_size", "trainable_params",
                         "best_val_loss", "best_val_acc", "best_epoch", "epochs"])
        best_ckpt_path = ckpt_dir / "best.pt"
        best_epoch = -1
        best_val_acc_value = 0.0
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_epoch = best_ckpt["epoch"] + 1
            best_val_acc_value = best_ckpt.get("val_acc", 0.0)
        w.writerow([cfg.dataset_variant, cfg.model_size, cfg.sp_vocab_size, total_params,
                     f"{best_val_loss:.4f}", f"{best_val_acc_value:.4f}", best_epoch, cfg.epochs])


if __name__ == "__main__":
    main()
