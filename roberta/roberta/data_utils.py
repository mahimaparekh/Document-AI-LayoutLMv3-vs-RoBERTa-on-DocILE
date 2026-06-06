"""
Data utilities: loading annotation files, extracting text/labels,
caching to JSON, and building HuggingFace datasets ready for the Trainer.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Annotation file discovery
# ---------------------------------------------------------------------------

def find_annotation_file(annotations_dir: Path, name: str) -> Optional[Path]:
    """Return the Path for *name* inside *annotations_dir*, or None."""
    # Candidate 1: exact name
    cand = annotations_dir / name
    if cand.exists():
        return cand
    # Candidate 2: name + .json extension
    cand2 = annotations_dir / (name if name.endswith(".json") else f"{name}.json")
    if cand2.exists():
        return cand2
    # Candidate 3: recursive glob
    matches = list(annotations_dir.glob(f"**/{Path(name).name}"))
    if matches:
        return matches[0]
    return None


# ---------------------------------------------------------------------------
# Text / label extraction from a single annotation file
# ---------------------------------------------------------------------------

def extract_from_file(ann_path: Path) -> Tuple[List[str], List[str]]:
    """Return (texts, labels) parsed from one DocILE annotation JSON."""
    with open(ann_path, "r") as f:
        data = json.load(f)
    texts, labels = [], []
    for item in data.get("field_extractions", []):
        text  = item.get("text", "")
        label = item.get("fieldtype", None)
        if not text or label is None:
            continue
        texts.append(text.strip())
        labels.append(label)
    return texts, labels


# ---------------------------------------------------------------------------
# Build full text/label lists for a split
# ---------------------------------------------------------------------------

def build_text_label_lists(
    split_json_path: Path,
    annotations_dir: Path,
) -> Tuple[List[str], List[str]]:
    """
    Load a split file (train.json / val.json), find each annotation file,
    and return concatenated (texts, labels).
    """
    with open(split_json_path, "r") as f:
        names: List[str] = json.load(f)

    all_texts: List[str]  = []
    all_labels: List[str] = []
    missing: List[str]    = []

    for name in names:
        ann_path = find_annotation_file(annotations_dir, name)
        if ann_path is None:
            missing.append(name)
            continue
        texts, labels = extract_from_file(ann_path)
        all_texts.extend(texts)
        all_labels.extend(labels)

    if missing:
        print(
            f"[data] Warning: {len(missing)} file(s) not found "
            f"(first 10): {missing[:10]}"
        )
    print(
        f"[data] Loaded {len(all_texts)} samples from "
        f"{split_json_path.name} ({len(names) - len(missing)}/{len(names)} files)"
    )
    return all_texts, all_labels


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_cache(texts: List[str], labels: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"texts": texts, "labels": labels}, f)
    print(f"[data] Saved cache → {path}")


def load_cache(path: Path) -> Tuple[List[str], List[str]]:
    with open(path, "r") as f:
        data = json.load(f)
    print(f"[data] Loaded cache ← {path}  ({len(data['texts'])} samples)")
    return data["texts"], data["labels"]


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def build_label_maps(
    train_labels: List[str], val_labels: List[str]
) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(train_labels + val_labels))
    label2id = {lab: idx for idx, lab in enumerate(unique_labels)}
    id2label  = {idx: lab for lab, idx in label2id.items()}
    print(f"[data] {len(unique_labels)} unique label classes")
    return unique_labels, label2id, id2label


# ---------------------------------------------------------------------------
# HuggingFace Dataset construction
# ---------------------------------------------------------------------------

def build_hf_datasets(
    train_texts:  List[str],
    train_labels: List[str],
    val_texts:    List[str],
    val_labels:   List[str],
    label2id:     Dict[str, int],
    tokenizer_name: str,
    max_length:   int = 128,
):
    """
    Tokenise texts and return (train_ds, val_ds, tokenizer) ready for
    HuggingFace Trainer.
    """
    train_ids = [label2id[l] for l in train_labels]
    val_ids   = [label2id[l] for l in val_labels]

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_ids})
    val_ds   = Dataset.from_dict({"text": val_texts,   "label": val_ids})

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize_batch,   batched=True, remove_columns=["text"])
    train_ds.set_format(type="torch")
    val_ds.set_format(type="torch")

    print(f"[data] train_ds={len(train_ds)}, val_ds={len(val_ds)}")
    return train_ds, val_ds, tokenizer
