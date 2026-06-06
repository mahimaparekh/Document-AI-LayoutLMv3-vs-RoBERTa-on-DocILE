#!/usr/bin/env python3
"""
Prepare and cache the DocILE text/label data for RoBERTa training.

Usage:
    python prepare_data.py
    python prepare_data.py --data_dir /path/to/docile --cache_dir /path/to/cache
    python prepare_data.py --force   # re-build even if cache already exists
"""

import argparse
from pathlib import Path

from roberta.config import ANNOTATIONS_DIR, CACHE_DIR, TRAIN_JSON, VAL_JSON
from roberta.data_utils import build_text_label_lists, save_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Build DocILE text/label cache")
    parser.add_argument("--train_json",      default=str(TRAIN_JSON))
    parser.add_argument("--val_json",        default=str(VAL_JSON))
    parser.add_argument("--annotations_dir", default=str(ANNOTATIONS_DIR))
    parser.add_argument("--cache_dir",       default=str(CACHE_DIR))
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild cache even if it already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_json      = Path(args.train_json)
    val_json        = Path(args.val_json)
    annotations_dir = Path(args.annotations_dir)
    cache_dir       = Path(args.cache_dir)

    train_cache = cache_dir / "train_extracted.json"
    val_cache   = cache_dir / "val_extracted.json"

    if train_cache.exists() and val_cache.exists() and not args.force:
        print("[prepare_data] Cache already exists. Use --force to rebuild.")
        return

    print("[prepare_data] Extracting training data …")
    train_texts, train_labels = build_text_label_lists(train_json, annotations_dir)
    save_cache(train_texts, train_labels, train_cache)

    print("[prepare_data] Extracting validation data …")
    val_texts, val_labels = build_text_label_lists(val_json, annotations_dir)
    save_cache(val_texts, val_labels, val_cache)

    print("[prepare_data] Done.")


if __name__ == "__main__":
    main()
