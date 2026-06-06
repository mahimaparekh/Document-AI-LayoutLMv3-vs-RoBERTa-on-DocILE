"""
Metric helpers for HuggingFace Trainer and standalone evaluation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# compute_metrics callables for Trainer
# ---------------------------------------------------------------------------

def compute_metrics_micro(eval_pred):
    """Micro-averaged metrics (used in most partial-tuning experiments)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall":    recall_score(labels, preds,    average="micro", zero_division=0),
        "f1":        f1_score(labels, preds,        average="micro", zero_division=0),
    }


def compute_metrics_macro(eval_pred):
    """Macro-averaged metrics (used in full-finetune and partial_3)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    accuracy_score(labels, preds),
        "prec_macro":  precision_score(labels, preds, average="macro", zero_division=0),
        "prec_micro":  precision_score(labels, preds, average="micro", zero_division=0),
        "rec_macro":   recall_score(labels, preds,    average="macro", zero_division=0),
        "rec_micro":   recall_score(labels, preds,    average="micro", zero_division=0),
        "f1_macro":    f1_score(labels, preds,        average="macro", zero_division=0),
        "f1_micro":    f1_score(labels, preds,        average="micro", zero_division=0),
    }


METRIC_FN = {
    "f1":       compute_metrics_micro,
    "f1_macro": compute_metrics_macro,
}


# ---------------------------------------------------------------------------
# Standalone (non-Trainer) evaluation loop
# ---------------------------------------------------------------------------

def evaluate_loader(model, loader, device, id2label=None):
    """
    Run *model* over *loader* and return a dict with accuracy, f1, precision,
    recall plus a full classification_report string.
    """
    import torch

    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    target_names = (
        [id2label[i] for i in range(len(id2label))] if id2label else None
    )
    report = classification_report(
        all_true, all_preds, target_names=target_names, zero_division=0
    )
    return {
        "accuracy":  accuracy_score(all_true, all_preds),
        "f1_micro":  f1_score(all_true, all_preds,        average="micro", zero_division=0),
        "f1_macro":  f1_score(all_true, all_preds,        average="macro", zero_division=0),
        "precision": precision_score(all_true, all_preds, average="micro", zero_division=0),
        "recall":    recall_score(all_true, all_preds,    average="micro", zero_division=0),
        "report":    report,
    }
