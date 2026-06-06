#!/usr/bin/env python3
"""
Evaluate a saved RoBERTa model on the DocILE validation set.

Usage:
    python evaluate.py --model_dir models/final_roberta/partial_4
    python evaluate.py --model_dir checkpoints/roberta/partial_3/checkpoint-best --batch_size 64
    python evaluate.py --model_dir models/final_roberta/lora --plot
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from roberta.config import MAX_LENGTH, VAL_CACHE
from roberta.data_utils import build_label_maps, build_hf_datasets, load_cache
from roberta.metrics import evaluate_loader
from roberta.plotting import plot_loss_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved RoBERTa model on DocILE val set")
    parser.add_argument("--model_dir",   required=True, help="Path to saved model / checkpoint")
    parser.add_argument("--val_cache",   default=str(VAL_CACHE))
    parser.add_argument("--train_cache", default=None,
                        help="Needed to reconstruct the full label map (recommended)")
    parser.add_argument("--label_maps",  default=None,
                        help="Path to label_maps.json saved alongside the model")
    parser.add_argument("--max_length",  type=int, default=MAX_LENGTH)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--plot",        action="store_true", help="Show loss curve from trainer log")
    return parser.parse_args()


def load_label_maps(model_dir: Path, train_cache_path, val_cache_path):
    """Try to load label maps from model_dir/label_maps.json, else reconstruct."""
    label_maps_file = model_dir / "label_maps.json"
    if label_maps_file.exists():
        with open(label_maps_file) as f:
            data = json.load(f)
        label2id = data["label2id"]
        id2label = {int(k): v for k, v in data["id2label"].items()}
        print(f"[eval] Loaded label maps from {label_maps_file} ({len(label2id)} classes)")
        return label2id, id2label

    # Fall back: reconstruct from caches
    print("[eval] label_maps.json not found — reconstructing from caches …")
    from roberta.data_utils import load_cache as _lc
    _, val_labels = _lc(Path(val_cache_path))
    if train_cache_path:
        _, train_labels = _lc(Path(train_cache_path))
    else:
        train_labels = val_labels
        print("[eval] Warning: train_cache not provided; label set may be incomplete")
    _, label2id, id2label = build_label_maps(train_labels, val_labels)
    return label2id, id2label


def main():
    args      = parse_args()
    model_dir = Path(args.model_dir)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")
    print(f"[eval] Model : {model_dir}")

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    label2id, id2label = load_label_maps(model_dir, args.train_cache, args.val_cache)
    num_labels = len(label2id)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    from roberta.data_utils import load_cache
    val_texts, val_labels = load_cache(Path(args.val_cache))
    val_ids = [label2id[l] for l in val_labels if l in label2id]
    # Filter texts whose label is in label2id (safety guard)
    paired = [(t, label2id[l]) for t, l in zip(val_texts, val_labels) if l in label2id]
    val_texts_f, val_ids_f = zip(*paired) if paired else ([], [])

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    from datasets import Dataset as HFDataset
    val_ds = HFDataset.from_dict({"text": list(val_texts_f), "label": list(val_ids_f)})

    def _tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    val_ds = val_ds.map(_tokenize, batched=True, remove_columns=["text"])
    val_ds.set_format(type="torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader   = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    try:
        # Try loading as a PEFT model first
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=num_labels
        )
        model = PeftModel.from_pretrained(base, str(model_dir))
        print("[eval] Loaded as PEFT/LoRA model")
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=num_labels,
            id2label=id2label, label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        print("[eval] Loaded as standard model")

    model = model.to(device)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    results = evaluate_loader(model, loader, device, id2label=id2label)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"F1 (micro): {results['f1_micro']:.4f}")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print("\nClassification Report:\n")
    print(results["report"])

    # Optionally save results
    out = model_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "report"}, f, indent=2
        )
    print(f"[eval] Results saved → {out}")


if __name__ == "__main__":
    main()
