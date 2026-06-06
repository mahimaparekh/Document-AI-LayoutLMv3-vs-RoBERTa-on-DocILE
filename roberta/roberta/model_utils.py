"""
Model utilities: building RoBERTa (optionally with LoRA), freezing layers,
and setting up per-group optimisers.
"""

from typing import Dict, List, Optional

import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification


# ---------------------------------------------------------------------------
# Freeze helpers
# ---------------------------------------------------------------------------

def freeze_embeddings(model) -> None:
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False


def freeze_encoder_layers(model, n_layers: int) -> None:
    """Freeze the first *n_layers* transformer encoder layers."""
    for layer in model.roberta.encoder.layer[:n_layers]:
        for param in layer.parameters():
            param.requires_grad = False


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    model_name:         str,
    num_labels:         int,
    id2label:           Dict[int, str],
    label2id:           Dict[str, int],
    freeze_emb:         bool = True,
    freeze_layers:      int  = 0,
    use_lora:           bool = False,
    lora_r:             int  = 8,
    lora_alpha:         int  = 32,
    lora_dropout:       float = 0.1,
    lora_target_modules: Optional[List[str]] = None,
    from_checkpoint:    Optional[str] = None,
):
    """
    Load (or resume) a RoBERTa sequence-classification model,
    optionally apply LoRA, and freeze requested layers.

    Returns the model (on CPU; move to device in the caller if needed).
    """
    load_path = from_checkpoint if from_checkpoint else model_name
    model = AutoModelForSequenceClassification.from_pretrained(
        load_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,   # safe when resuming from a checkpoint
    )

    if freeze_emb:
        freeze_embeddings(model)
    if freeze_layers > 0:
        freeze_encoder_layers(model, freeze_layers)

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "peft is required for LoRA experiments. "
                "Install with: pip install peft"
            )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules or ["query", "value"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[model] Trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")
    return model


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def build_optimizer(
    model,
    lr_encoder:    float,
    lr_classifier: float,
    weight_decay:  float = 0.01,
) -> AdamW:
    """
    Two parameter groups:
      - classifier head  → higher LR (lr_classifier)
      - everything else  → lower LR  (lr_encoder)
    Only includes parameters with requires_grad=True.
    """
    classifier_params = [
        p for n, p in model.named_parameters()
        if "classifier" in n and p.requires_grad
    ]
    encoder_params = [
        p for n, p in model.named_parameters()
        if "classifier" not in n and p.requires_grad
    ]
    optimizer = AdamW(
        [
            {"params": encoder_params,    "lr": lr_encoder},
            {"params": classifier_params, "lr": lr_classifier},
        ],
        weight_decay=weight_decay,
    )
    return optimizer
