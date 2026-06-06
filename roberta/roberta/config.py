"""
Centralised configuration for all RoBERTa experiments on the DocILE dataset.
Edit the paths and hyperparameters here; everything else reads from this file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (override these for your environment)
# ---------------------------------------------------------------------------
BASE_DATA_DIR   = Path("data")                          # root of your DocILE data
TRAIN_JSON      = BASE_DATA_DIR / "train.json"
VAL_JSON        = BASE_DATA_DIR / "val.json"
ANNOTATIONS_DIR = BASE_DATA_DIR / "annotations"

CACHE_DIR       = BASE_DATA_DIR / "cache"               # extracted text/label caches
TRAIN_CACHE     = CACHE_DIR / "train_extracted.json"
VAL_CACHE       = CACHE_DIR / "val_extracted.json"

OUTPUT_DIR      = Path("checkpoints") / "roberta"       # Trainer checkpoints
FINAL_MODEL_DIR = Path("models") / "final_roberta"      # saved after best run

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "roberta-base"
MAX_LENGTH = 128

# ---------------------------------------------------------------------------
# Training defaults (used by train_roberta.py; each experiment can override)
# ---------------------------------------------------------------------------
TRAIN_DEFAULTS = dict(
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    num_train_epochs            = 5,
    weight_decay                = 0.01,
    load_best_model_at_end      = True,
    greater_is_better           = True,
    fp16                        = True,
    logging_steps               = 100,
    save_total_limit            = 2,
    report_to                   = "none",
)

# ---------------------------------------------------------------------------
# Experiment registry
# Each entry defines the frozen-layer count, LR groups, and extra TrainingArguments.
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "full_finetune": dict(
        freeze_embeddings   = True,
        freeze_layers       = 0,           # freeze no encoder layers
        lr_encoder          = 2e-5,
        lr_classifier       = 2e-5,
        metric_for_best     = "f1_macro",
        extra_train_args    = dict(
            learning_rate               = 2e-5,
            num_train_epochs            = 5,
        ),
    ),
    "partial_1": dict(
        freeze_embeddings   = True,
        freeze_layers       = 8,           # freeze first 8 encoder layers
        lr_encoder          = 1e-5,
        lr_classifier       = 5e-5,
        metric_for_best     = "f1",
        extra_train_args    = dict(
            learning_rate               = 1e-5,
            num_train_epochs            = 3,
            warmup_steps                = None,   # computed dynamically
        ),
    ),
    "partial_2": dict(
        freeze_embeddings   = True,
        freeze_layers       = 6,
        lr_encoder          = 3e-5,
        lr_classifier       = 5e-5,
        metric_for_best     = "f1",
        extra_train_args    = dict(
            learning_rate               = 3e-5,
            num_train_epochs            = 5,
            warmup_ratio                = 0.1,
            max_grad_norm               = 1.0,
        ),
    ),
    "partial_3": dict(
        freeze_embeddings   = True,
        freeze_layers       = 6,
        lr_encoder          = 3e-5,
        lr_classifier       = 5e-5,
        metric_for_best     = "f1_macro",
        extra_train_args    = dict(
            learning_rate               = 3e-5,
            num_train_epochs            = 5,
            weight_decay                = 0.05,
            warmup_ratio                = 0.1,
            max_grad_norm               = 1.0,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size  = 64,
            gradient_accumulation_steps = 2,
            fp16_full_eval              = True,
        ),
    ),
    "partial_4": dict(
        freeze_embeddings   = True,
        freeze_layers       = 4,
        lr_encoder          = 4e-5,
        lr_classifier       = 8e-5,
        weight_decay        = 0.1,
        metric_for_best     = "f1",
        scheduler           = "cosine",
        extra_train_args    = dict(
            learning_rate               = 4e-5,
            num_train_epochs            = 3,
            weight_decay                = 0.1,
            warmup_ratio                = 0.06,
            max_grad_norm               = 1.0,
            per_device_train_batch_size = 24,
            per_device_eval_batch_size  = 64,
            fp16_full_eval              = True,
        ),
    ),
    "partial_5": dict(
        freeze_embeddings   = True,
        freeze_layers       = 3,
        lr_encoder          = 5e-5,
        lr_classifier       = 9e-5,
        weight_decay        = 0.15,
        metric_for_best     = "f1",
        scheduler           = "cosine_with_restarts",
        extra_train_args    = dict(
            learning_rate               = 5e-5,
            num_train_epochs            = 3,
            weight_decay                = 0.15,
            warmup_ratio                = 0.08,
            max_grad_norm               = 0.9,
            per_device_train_batch_size = 24,
            per_device_eval_batch_size  = 64,
            gradient_accumulation_steps = 2,
            fp16_full_eval              = True,
            lr_scheduler_type           = "cosine_with_restarts",
        ),
    ),
    "lora": dict(
        freeze_embeddings   = True,
        freeze_layers       = 4,
        lr_encoder          = 4e-5,
        lr_classifier       = 8e-5,
        weight_decay        = 0.1,
        metric_for_best     = "f1",
        scheduler           = "cosine",
        use_lora            = True,
        lora_r              = 8,
        lora_alpha          = 32,
        lora_dropout        = 0.1,
        lora_target_modules = ["query", "value"],
        extra_train_args    = dict(
            learning_rate               = 4e-5,
            num_train_epochs            = 5,
            weight_decay                = 0.1,
            warmup_ratio                = 0.1,
            max_grad_norm               = 1.0,
            per_device_train_batch_size = 24,
            per_device_eval_batch_size  = 64,
            fp16_full_eval              = True,
        ),
    ),
}
