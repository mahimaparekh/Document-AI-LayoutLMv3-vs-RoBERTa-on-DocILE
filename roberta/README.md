# RoBERTa on DocILE — Modular Codebase

Field-type classification on the [DocILE](https://github.com/rossumai/docile) dataset using RoBERTa-base. Part of the *Document AI: RoBERTa vs LayoutLMv3* project.

---

## Project structure

```
roberta_docile/
├── roberta/                  # Library package
│   ├── config.py             # All paths, hyperparameters, experiment configs
│   ├── data_utils.py         # Data loading, caching, HF dataset construction
│   ├── model_utils.py        # Model factory (standard + LoRA), optimiser builder
│   ├── metrics.py            # compute_metrics functions + standalone eval loop
│   └── plotting.py           # Loss-curve helpers
│
├── prepare_data.py           # Step 1 — extract & cache text/labels from annotations
├── zero_shot.py              # Step 2 (optional) — zero-shot baseline
├── train_roberta.py          # Step 3 — fine-tune any experiment
├── evaluate.py               # Step 4 — evaluate a saved model
└── plot_loss.py              # Utility — plot loss curves
```

---

## Setup

```bash
# 1. Clone the DocILE repo for dataset utilities
git clone https://github.com/rossumai/docile.git

# 2. Install dependencies
pip install -r requirements.txt

# For GPU (CUDA 12.1):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For PDF rasterisation
apt-get install -y poppler-utils   # Linux / Colab
```

---

## Data layout

Put your DocILE data files under `data/` (or edit the paths in `roberta/config.py`):

```
data/
├── train.json
├── val.json
└── annotations/
    ├── <doc_id>.json
    └── ...
```

---

## Quickstart

### 1 — Prepare / cache data
```bash
python prepare_data.py
# with custom paths:
python prepare_data.py --data_dir /path/to/docile --cache_dir data/cache
```

### 2 — Zero-shot baseline (optional)
```bash
python zero_shot.py
```

### 3 — Train
```bash
# Full fine-tune (5 epochs)
python train_roberta.py --experiment full_finetune --save_model --plot

# Partial fine-tune variants (freeze first N encoder layers)
python train_roberta.py --experiment partial_1
python train_roberta.py --experiment partial_2
python train_roberta.py --experiment partial_3
python train_roberta.py --experiment partial_4 --save_model
python train_roberta.py --experiment partial_5

# LoRA fine-tune
python train_roberta.py --experiment lora --save_model --plot

# Resume from a checkpoint
python train_roberta.py --experiment partial_4 --checkpoint checkpoints/roberta/partial_4/checkpoint-500

# Override epochs / batch size on the fly
python train_roberta.py --experiment full_finetune --epochs 3 --batch_size 32
```

### 4 — Evaluate a saved model
```bash
python evaluate.py --model_dir models/final_roberta/partial_4
python evaluate.py --model_dir models/final_roberta/lora --batch_size 64
```

### 5 — Plot loss curves
```bash
# From Trainer's trainer_state.json (automatic when --plot is passed to train_roberta.py)
python plot_loss.py \
    --log_file checkpoints/roberta/partial_4/trainer_state.json \
    --title "Partial Fine-Tune 4" \
    --save figures/partial4.png

# From manually recorded values
python plot_loss.py \
    --epochs 1 2 3 4 5 \
    --train_loss 0.577 0.524 0.469 0.483 0.469 \
    --val_loss   0.770 0.767 0.775 0.747 0.750 \
    --title "Full Fine-Tune" \
    --save figures/full_finetune.png
```

---

## Experiments

| ID | Frozen layers | LR (encoder / cls) | Scheduler | Epochs |
|----|--------------|---------------------|-----------|--------|
| `full_finetune` | embeddings only | 2e-5 / 2e-5 | linear | 5 |
| `partial_1` | emb + 8 enc | 1e-5 / 5e-5 | linear | 3 |
| `partial_2` | emb + 6 enc | 3e-5 / 5e-5 | linear | 5 |
| `partial_3` | emb + 6 enc | 3e-5 / 5e-5 | linear | 5 |
| `partial_4` | emb + 4 enc | 4e-5 / 8e-5 | cosine | 3 |
| `partial_5` | emb + 3 enc | 5e-5 / 9e-5 | cosine_with_restarts | 3 |
| `lora` | emb + 4 enc + LoRA (r=8) | 4e-5 / 8e-5 | cosine | 5 |

To add a new experiment, just add an entry to `EXPERIMENTS` in `roberta/config.py`.

---

## Known fixes applied during modularisation

- `os.path.join(p_alt)` → `os.path.exists(p_alt)` (typo in original notebook that silently skipped missing training files)
- Duplicate `Trainer(...)` instantiation in Partial Tuning 3 removed (second call overwrote the scheduler)
- `annotations_path` used as both `str` and `Path` in various cells — unified to `Path` throughout
- Loss-curve lists in Partial Tuning 1 had 5 values for 3 epochs — truncation now handled automatically in `plot_loss_curve`
- `latest_ckpt` used but never defined in full fine-tune cell — replaced with `MODEL_NAME` (load from HuggingFace Hub) with a `--checkpoint` flag for resuming
