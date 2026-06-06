#!/usr/bin/env python3
"""
Zero-shot baseline: evaluate the raw (un-finetuned) RoBERTa-base on the
DocILE validation set.

Usage:
    python zero_shot.py
    python zero_shot.py --val_json data/val.json --batch_size 16
"""

import argparse

import torch
from torch.utils.data import DataLoader

from roberta.config import (
    ANNOTATIONS_DIR,
    MAX_LENGTH,
    MODEL_NAME,
    VAL_JSON,
)
from roberta.data_utils import build_text_label_lists, build_label_maps
from roberta.metrics import evaluate_loader
from roberta.model_utils import build_model

# Inline dataset (no cache needed for zero-shot)
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer


class _SimpleDataset(TorchDataset):
    def __init__(self, texts, label_ids, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(label_ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot RoBERTa baseline")
    parser.add_argument("--val_json",        default=str(VAL_JSON))
    parser.add_argument("--annotations_dir", default=str(ANNOTATIONS_DIR))
    parser.add_argument("--model_name",      default=MODEL_NAME)
    parser.add_argument("--max_length",      type=int, default=MAX_LENGTH)
    parser.add_argument("--batch_size",      type=int, default=8)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[zero_shot] Device: {device}")

    from pathlib import Path
    val_texts, val_labels = build_text_label_lists(
        Path(args.val_json), Path(args.annotations_dir)
    )
    unique_labels, label2id, id2label = build_label_maps(val_labels, val_labels)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    val_ids   = [label2id[l] for l in val_labels]
    dataset   = _SimpleDataset(val_texts, val_ids, tokenizer, args.max_length)
    loader    = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(
        args.model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        freeze_emb=False,
        freeze_layers=0,
    ).to(device)

    results = evaluate_loader(model, loader, device, id2label=id2label)

    print("\n=== Zero-Shot Baseline Results ===")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"F1 (micro): {results['f1_micro']:.4f}")
    print(f"F1 (macro): {results['f1_macro']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print("\nClassification Report:\n")
    print(results["report"])


if __name__ == "__main__":
    main()
