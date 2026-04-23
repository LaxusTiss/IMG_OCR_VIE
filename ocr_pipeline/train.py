from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from ocr_pipeline.dataset import OCRCropDataset, OCRVocab, collate_fn
from ocr_pipeline.model import CRNN


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def greedy_decode(log_probs: torch.Tensor, vocab: OCRVocab) -> list[str]:
    best = log_probs.argmax(dim=-1).permute(1, 0)
    decoded = []
    for sample in best:
        tokens = []
        prev = None
        for token in sample.tolist():
            if token != prev and token != vocab.blank_idx:
                tokens.append(token)
            prev = token
        decoded.append(vocab.decode(tokens))
    return decoded


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, vocab: OCRVocab) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    exact_matches = 0
    total_distance = 0
    total_chars = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            label_lengths = batch["label_lengths"].to(device)

            logits = model(images)
            log_probs = logits.log_softmax(dim=-1)
            input_lengths = torch.full(
                size=(images.shape[0],),
                fill_value=log_probs.shape[0],
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, labels, input_lengths, label_lengths)

            predictions = greedy_decode(log_probs.cpu(), vocab)
            for pred, target in zip(predictions, batch["texts"]):
                exact_matches += int(pred == target)
                total_distance += levenshtein(pred, target)
                total_chars += max(len(target), 1)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": exact_matches / max(total_samples, 1),
        "cer": total_distance / max(total_chars, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        log_probs = logits.log_softmax(dim=-1)
        input_lengths = torch.full(
            size=(images.shape[0],),
            fill_value=log_probs.shape[0],
            dtype=torch.long,
            device=device,
        )
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CRNN OCR recognizer on prepared crops.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("prepared_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/crnn_baseline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional limit for quick smoke tests.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional limit for quick smoke tests.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a previous checkpoint.")
    parser.add_argument("--lr-patience", type=int, default=4, help="Epochs to wait before reducing LR.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor when val CER plateaus.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    prepared_dir = args.prepared_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = OCRVocab.from_charset_file(prepared_dir / "charset.txt")
    train_set = OCRCropDataset(prepared_dir, "train", vocab, args.image_height, args.max_width)
    val_set = OCRCropDataset(prepared_dir, "val", vocab, args.image_height, args.max_width)
    if args.max_train_samples > 0:
        train_set = Subset(train_set, range(min(args.max_train_samples, len(train_set))))
    if args.max_val_samples > 0:
        val_set = Subset(val_set, range(min(args.max_val_samples, len(val_set))))

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = CRNN(num_classes=vocab.size).to(device)
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience)

    history = []
    best_score = math.inf
    start_epoch = 1

    config = {
        "prepared_dir": str(prepared_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "image_height": args.image_height,
        "max_width": args.max_width,
        "vocab_size": vocab.size,
        "device": str(device),
    }

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = checkpoint.get("history", [])
        best_score = checkpoint.get("best_score", math.inf)
        start_epoch = checkpoint.get("epoch", 0) + 1
        old_config = checkpoint.get("config", {})
        config.update({k: v for k, v in old_config.items() if k in config})
        print(json.dumps({"resumed_from": str(args.resume), "start_epoch": start_epoch}, ensure_ascii=False))

    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "charset.txt").write_text(vocab.chars, encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, val_loader, criterion, device, vocab)
        scheduler.step(metrics["cer"])
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
            "history": history,
            "best_score": best_score,
        }
        if metrics["cer"] < best_score:
            best_score = metrics["cer"]
            checkpoint["best_score"] = best_score
            torch.save(checkpoint, output_dir / "best.pt")
        checkpoint["best_score"] = best_score
        torch.save(checkpoint, output_dir / "last.pt")

        (output_dir / "history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
