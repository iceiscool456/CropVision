"""
Local fine-tuning script for plant classifiers — sized for an 8 GB Mac (MPS).

Strategy for low-memory machines:
  * default to a SMALL backbone (ViT-tiny)
  * freeze the backbone and train only the classifier head (a "linear probe").
    This is fast, memory-light, and surprisingly effective for transfer
    learning.  Pass --full-finetune to unfreeze everything (needs more RAM).

Usage:
    python -m src.training.finetune \
        --data-dir data/plants \
        --epochs 5 \
        --batch-size 16 \
        --output models/finetuned-plants

The output directory can then be loaded as a local HuggingFace model if you
later want an on-device classifier (e.g. for offline use).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.datasets import build_dataloaders

DEFAULT_MODEL = "WinKawaks/vit-tiny-patch16-224"  # ~22M params, fits 8 GB


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _freeze_backbone(model) -> None:
    """Freeze everything except the final classification head."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name


@torch.no_grad()
def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for pixel_values, labels in loader:
        pixel_values, labels = pixel_values.to(device), labels.to(device)
        preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train(args: argparse.Namespace) -> None:
    device = _pick_device()
    print(f"[train] device={device}  model={args.model}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    train_loader, val_loader, id2label, label2id = build_dataloaders(
        args.data_dir, processor, batch_size=args.batch_size
    )
    print(f"[train] {len(id2label)} classes: {list(id2label.values())}")

    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # replace the pretrained head
    ).to(device)

    if not args.full_finetune:
        _freeze_backbone(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] frozen backbone — training head only ({trainable:,} params)")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    out_dir = Path(args.output)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (pixel_values, labels) in enumerate(train_loader, 1):
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(pixel_values=pixel_values).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if step % 10 == 0:
                print(f"  epoch {epoch} step {step}  loss={running / step:.4f}")

        acc = _evaluate(model, val_loader, device)
        print(f"[train] epoch {epoch}: val_acc={acc:.2%}")
        if acc >= best_acc:
            best_acc = acc
            out_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            print(f"[train] saved best model ({acc:.2%}) → {out_dir}")

    print(f"[train] done. best val_acc={best_acc:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a plant classifier")
    parser.add_argument("--data-dir", required=True, help="ImageFolder dataset root")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="base model id")
    parser.add_argument("--output", default="models/finetuned", help="save dir")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="unfreeze the whole backbone (needs more RAM; off by default)",
    )
    train(parser.parse_args())


if __name__ == "__main__":
    main()
