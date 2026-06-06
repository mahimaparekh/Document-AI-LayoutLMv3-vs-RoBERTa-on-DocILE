#!/usr/bin/env python3
"""
Train RoBERTa on the DocILE field-type classification task.

Usage:
    # Full fine-tune (all experiments defined in config.py):
    python train_roberta.py --experiment full_finetune
    python train_roberta.py --experiment partial_1
    python train_roberta.py --experiment lora

    # Resume from a checkpoint:
    python train_roberta.py --experiment partial_4 --checkpoint checkpoints/roberta/checkpoint-500

    # Override any training arg on the command line:
    python train_roberta.py --experiment partial_2 --epochs 3 --batch_size 32

Available experiments:
    full_finetune, partial_1, partial_2, partial_3,
    partial_4, partial_5, lora
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, get_scheduler

from roberta.config import (
    EXPERIMENTS,
    FINAL_MODEL_DIR,
    MAX_LENGTH,
    MODEL_NAME,
    OUTPUT_DIR,
    TRAIN_CACHE,
    TRAIN_DEFAULTS,
    VAL_CACHE,
)
from roberta.data_utils import build_hf_datasets, build_label_maps, load_cache
from roberta.metrics import METRIC_FN
from roberta.model_utils import build_model, build_optimizer
from roberta.plotting import plot_from_trainer_log


def parse_args():
    parser = argparse.ArgumentParser(description="Train RoBERTa on DocILE")
    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Which experiment config to run",
    )
    parser.add_argument("--checkpoint", default=None, help="Resume from this checkpoint path")
    parser.add_argument("--output_dir", default=None, help="Override checkpoint output dir")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs",     type=int, default=None, help="Override num_train_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override per_device_train_batch_size")
    parser.add_argument("--no_fp16",    action="store_true", help="Disable mixed precision")
    parser.add_argument("--save_model", action="store_true", help="Save final model after training")
    parser.add_argument("--plot",       action="store_true", help="Save loss curve after training")
    return parser.parse_args()


def main():
    args = parse_args()
    exp  = EXPERIMENTS[args.experiment]

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    print(f"\n[train] Loading cached data …")
    train_texts, train_labels = load_cache(TRAIN_CACHE)
    val_texts,   val_labels   = load_cache(VAL_CACHE)
    unique_labels, label2id, id2label = build_label_maps(train_labels, val_labels)

    train_ds, val_ds, tokenizer = build_hf_datasets(
        train_texts, train_labels,
        val_texts,   val_labels,
        label2id,
        tokenizer_name=args.model_name,
        max_length=args.max_length,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    print(f"\n[train] Building model for experiment '{args.experiment}' …")
    model = build_model(
        model_name      = args.model_name,
        num_labels      = len(unique_labels),
        id2label        = id2label,
        label2id        = label2id,
        freeze_emb      = exp.get("freeze_embeddings", True),
        freeze_layers   = exp.get("freeze_layers", 0),
        use_lora        = exp.get("use_lora", False),
        lora_r          = exp.get("lora_r", 8),
        lora_alpha      = exp.get("lora_alpha", 32),
        lora_dropout    = exp.get("lora_dropout", 0.1),
        lora_target_modules = exp.get("lora_target_modules", ["query", "value"]),
        from_checkpoint = args.checkpoint,
    )

    # -----------------------------------------------------------------------
    # Optimiser & (optional) scheduler
    # -----------------------------------------------------------------------
    optimizer = build_optimizer(
        model,
        lr_encoder    = exp.get("lr_encoder", 2e-5),
        lr_classifier = exp.get("lr_classifier", 2e-5),
        weight_decay  = exp.get("weight_decay", TRAIN_DEFAULTS["weight_decay"]),
    )

    # Merge defaults ← experiment overrides ← CLI overrides
    train_kwargs = {**TRAIN_DEFAULTS, **exp.get("extra_train_args", {})}
    train_kwargs["metric_for_best_model"] = exp.get("metric_for_best", "f1_macro")
    if args.epochs     is not None: train_kwargs["num_train_epochs"]            = args.epochs
    if args.batch_size is not None: train_kwargs["per_device_train_batch_size"] = args.batch_size
    if args.no_fp16:                train_kwargs["fp16"] = False

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / args.experiment
    train_kwargs["output_dir"] = str(out_dir)

    # warmup_steps override for partial_1 (computed dynamically)
    if train_kwargs.get("warmup_steps") is None:
        train_kwargs.pop("warmup_steps", None)   # let warmup_ratio take over

    training_args = TrainingArguments(**train_kwargs)

    # Optional cosine scheduler (partial_4, lora)
    lr_scheduler = None
    if exp.get("scheduler") in ("cosine", "cosine_with_restarts"):
        bs    = training_args.per_device_train_batch_size
        ga    = training_args.gradient_accumulation_steps
        steps = len(train_ds) // (bs * ga) * int(training_args.num_train_epochs)
        lr_scheduler = get_scheduler(
            name              = exp["scheduler"],
            optimizer         = optimizer,
            num_warmup_steps  = int(training_args.warmup_ratio * steps),
            num_training_steps= steps,
        )

    # -----------------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------------
    metric_key  = exp.get("metric_for_best", "f1_macro")
    compute_fn  = METRIC_FN.get(metric_key, METRIC_FN["f1_macro"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        tokenizer     = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_fn,
        optimizers    = (optimizer, lr_scheduler),
    )

    print(f"\n[train] Starting training — experiment: {args.experiment}\n")
    trainer.train(resume_from_checkpoint=args.checkpoint)

    # -----------------------------------------------------------------------
    # Post-training
    # -----------------------------------------------------------------------
    if args.save_model:
        save_dir = FINAL_MODEL_DIR / args.experiment
        trainer.save_model(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))
        # Save label maps alongside the model
        with open(save_dir / "label_maps.json", "w") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
        print(f"[train] Model saved → {save_dir}")

    if args.plot:
        plot_path = out_dir / "loss_curve.png"
        plot_from_trainer_log(
            trainer.state.log_history,
            title     = f"Loss Curve — {args.experiment}",
            save_path = plot_path,
        )

    print("\n[train] Done.")


if __name__ == "__main__":
    main()
