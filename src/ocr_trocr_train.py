"""
=============================================================================
TrOCR Fine-Tuning & Evaluation on Product-Label Images
=============================================================================

This module fine-tunes Microsoft's TrOCR (Transformer-based Optical Character
Recognition) model on product label images extracted from the Amazon ML
Challenge dataset.  The goal is to read textual information (brand names,
quantities, units) directly from product images and produce a Character Error
Rate (CER) evaluation report.

Architecture
------------
    ┌────────────────┐       ┌────────────────────┐
    │ Product Image  │──────▶│ ViT Image Encoder   │
    └────────────────┘       └────────┬───────────┘
                                      │ visual features
                                      ▼
                             ┌────────────────────┐
                             │ RoBERTa Text Decoder│──▶ predicted text
                             └────────────────────┘

Model : microsoft/trocr-base-handwritten  (fine-tuned here)
Metric: Character Error Rate (CER) via `jiwer`

Usage
-----
    python src/ocr_trocr_train.py                          # defaults
    python src/ocr_trocr_train.py --epochs 15 --lr 3e-5    # custom

NOTE: This script is provided as a reference implementation.  Running it
requires a GPU with ≥8 GB VRAM and the OCR image dataset to be present
under `data/ocr/`.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

# Optional — graceful fallback if jiwer is not installed
try:
    import jiwer
    _HAS_JIWER = True
except ImportError:
    _HAS_JIWER = False

# ────────────────────────── Logging ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# 1.  DATASET
# ============================================================================
class OCRDataset(Dataset):
    """
    Custom PyTorch dataset for TrOCR training.

    Expected directory layout
    -------------------------
        data/ocr/
        ├── images/
        │   ├── 00001.jpg
        │   ├── 00002.jpg
        │   └── ...
        └── labels.csv          # columns: image_filename, text
    """

    def __init__(
        self,
        image_dir: str,
        labels_csv: str,
        processor: TrOCRProcessor,
        max_target_length: int = 128,
    ):
        super().__init__()
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.max_target_length = max_target_length

        # Load ground-truth labels
        self.df = pd.read_csv(labels_csv)
        self.df = self.df.dropna(subset=["text"]).reset_index(drop=True)
        log.info("Loaded %d labelled samples from %s", len(self.df), labels_csv)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["image_filename"]

        # Load image → RGB
        image = Image.open(img_path).convert("RGB")

        # Processor: image → pixel_values,  text → input_ids (for decoder)
        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        # Tokenise the target text
        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Replace padding token id with -100 so it's ignored by CrossEntropyLoss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================================
# 2.  METRICS
# ============================================================================
def compute_cer(predictions: list[str], references: list[str]) -> float:
    """
    Compute Character Error Rate (CER).

    CER = (Substitutions + Insertions + Deletions) / len(reference)

    Uses `jiwer` if available, otherwise falls back to a manual
    Levenshtein-based implementation.
    """
    if _HAS_JIWER:
        cer = jiwer.cer(references, predictions)
        return cer

    # Manual fallback — edit distance
    total_chars = 0
    total_errors = 0
    for pred, ref in zip(predictions, references):
        dist = _levenshtein(pred, ref)
        total_errors += dist
        total_chars += max(len(ref), 1)
    return total_errors / total_chars


def _levenshtein(s1: str, s2: str) -> int:
    """Classic dynamic-programming edit distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


# ============================================================================
# 3.  TRAINING LOOP
# ============================================================================
def train_one_epoch(
    model, dataloader, optimizer, scheduler, device, epoch, total_epochs
):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [TRAIN]")

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    avg_loss = running_loss / len(dataloader)
    return avg_loss


# ============================================================================
# 4.  EVALUATION LOOP
# ============================================================================
@torch.no_grad()
def evaluate(model, dataloader, processor, device, epoch, total_epochs):
    model.eval()
    all_preds = []
    all_refs = []
    running_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} [EVAL] ")

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass for loss
        outputs = model(pixel_values=pixel_values, labels=labels)
        running_loss += outputs.loss.item()

        # Generate predictions (greedy decoding)
        generated_ids = model.generate(pixel_values, max_new_tokens=128)
        pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        # Decode reference labels (replace -100 with pad_token_id)
        label_ids = labels.clone()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        ref_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

        all_preds.extend(pred_texts)
        all_refs.extend(ref_texts)

    avg_loss = running_loss / len(dataloader)
    cer = compute_cer(all_preds, all_refs)
    return avg_loss, cer, all_preds, all_refs


# ============================================================================
# 5.  RESULTS REPORTING
# ============================================================================
def save_results(
    output_dir: Path,
    history: list[dict],
    final_preds: list[str],
    final_refs: list[str],
):
    """Persist training history and sample predictions to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training history ──
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("Training history saved → %s", history_path)

    # ── Per-sample results ──
    results_df = pd.DataFrame({"reference": final_refs, "prediction": final_preds})
    results_df["char_error_rate"] = results_df.apply(
        lambda r: compute_cer([r["prediction"]], [r["reference"]]), axis=1
    )
    results_path = output_dir / "ocr_predictions.csv"
    results_df.to_csv(results_path, index=False)
    log.info("Per-sample results saved → %s  (%d samples)", results_path, len(results_df))

    # ── Summary report ──
    report_path = output_dir / "ocr_report.txt"
    final_epoch = history[-1]
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  TrOCR Fine-Tuning — Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp         : {datetime.now().isoformat()}\n")
        f.write(f"Total Epochs      : {len(history)}\n")
        f.write(f"Final Train Loss  : {final_epoch['train_loss']:.4f}\n")
        f.write(f"Final Val Loss    : {final_epoch['val_loss']:.4f}\n")
        f.write(f"Final Val CER     : {final_epoch['val_cer']:.4f}\n")
        f.write(f"Best Val CER      : {min(h['val_cer'] for h in history):.4f}\n\n")
        f.write("── Sample Predictions ─────────────────────────────────\n")
        for i in range(min(10, len(final_preds))):
            f.write(f"\n  REF : {final_refs[i]}\n")
            f.write(f"  PRED: {final_preds[i]}\n")
            sample_cer = compute_cer([final_preds[i]], [final_refs[i]])
            f.write(f"  CER : {sample_cer:.4f}\n")
        f.write("\n" + "=" * 60 + "\n")
    log.info("Summary report saved → %s", report_path)

    return results_df


# ============================================================================
# 6.  MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR on product-label images"
    )
    parser.add_argument("--image_dir",   default="data/ocr/images",         help="Path to OCR images")
    parser.add_argument("--labels_csv",  default="data/ocr/labels.csv",     help="Path to labels CSV")
    parser.add_argument("--output_dir",  default="models/ocr",              help="Where to save checkpoints & results")
    parser.add_argument("--model_name",  default="microsoft/trocr-base-handwritten", help="HuggingFace model ID")
    parser.add_argument("--epochs",      type=int,   default=10,            help="Training epochs")
    parser.add_argument("--batch_size",  type=int,   default=16,            help="Batch size")
    parser.add_argument("--lr",          type=float, default=5e-5,          help="Learning rate")
    parser.add_argument("--val_split",   type=float, default=0.15,          help="Validation split ratio")
    parser.add_argument("--seed",        type=int,   default=42,            help="Random seed")
    args = parser.parse_args()

    # ── Reproducibility ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Load processor & model ──
    log.info("Loading TrOCR processor and model: %s", args.model_name)
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    # Configure decoder for generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(device)

    # ── Dataset & dataloaders ──
    full_dataset = OCRDataset(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        processor=processor,
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=default_data_collator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=default_data_collator,
    )

    log.info("Train samples: %d  |  Val samples: %d", train_size, val_size)

    # ── Optimiser & scheduler ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # ── Training ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    history = []

    log.info("Starting TrOCR fine-tuning for %d epochs …", args.epochs)
    log.info("=" * 60)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, args.epochs
        )
        val_loss, val_cer, preds, refs = evaluate(
            model, val_loader, processor, device, epoch, args.epochs
        )

        log.info(
            "Epoch %02d/%02d  │  train_loss=%.4f  val_loss=%.4f  val_CER=%.4f",
            epoch + 1, args.epochs, train_loss, val_loss, val_cer,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_cer": round(val_cer, 4),
        })

        # Save best model checkpoint
        if val_cer < best_cer:
            best_cer = val_cer
            ckpt_path = output_dir / "best_model"
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            log.info("  ✅  New best CER=%.4f — checkpoint saved → %s", best_cer, ckpt_path)

    # ── Final evaluation & report ──
    log.info("=" * 60)
    log.info("Training complete.  Best CER: %.4f", best_cer)

    # Reload best checkpoint for final evaluation
    best_model = VisionEncoderDecoderModel.from_pretrained(output_dir / "best_model")
    best_model.to(device)
    _, final_cer, final_preds, final_refs = evaluate(
        best_model, val_loader, processor, device, args.epochs - 1, args.epochs
    )
    log.info("Final evaluation CER (best checkpoint): %.4f", final_cer)

    results_df = save_results(output_dir, history, final_preds, final_refs)

    # Save final model as well
    final_path = output_dir / "final_model"
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    log.info("Final model saved → %s", final_path)

    log.info("🎉  All done!")


if __name__ == "__main__":
    main()
