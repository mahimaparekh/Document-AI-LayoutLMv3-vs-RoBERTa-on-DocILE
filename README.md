# Document-AI-LayoutLMv3-vs-RoBERTa-on-DocILE
A Comprehensive Analysis of LayoutLMv3 and RoBERTa for Classification and Key Information Extraction Fine-Tuned on the DocILE Dataset.

# RoBERTa on DocILE

## Project Structure

```
roberta/
  config.py        ← all paths, hyperparameters, experiment configs
  data_utils.py    ← data loading and preprocessing
  model_utils.py   ← model and optimiser setup
  metrics.py       ← evaluation metrics
  plotting.py      ← loss curve helpers

prepare_data.py    ← extract and cache data from annotations
zero_shot.py       ← run zero-shot baseline (no training)
train_roberta.py   ← train a model
evaluate.py        ← evaluate a saved model
plot_loss.py       ← plot loss curves
```

---

## Setup

**1. Clone this repo**
```bash
git clone https://github.com/mahimaparekh/Document-AI-LayoutLMv3-vs-RoBERTa-on-DocILE.git
cd Document-AI-LayoutLMv3-vs-RoBERTa-on-DocILE
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For GPU (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Getting the Dataset

**1.** Go to https://docile.rossum.ai/ and request a token. They will email it to you.

**2.** Clone the DocILE repo and run their download script:
```bash
git clone https://github.com/rossumai/docile.git
cd docile
./download_dataset.sh YOUR_TOKEN annotated-trainval data/docile --unzip
```

**3.** Open `roberta/config.py` and set `BASE_DATA_DIR` to wherever the data was downloaded:
```python
BASE_DATA_DIR = Path("/path/to/docile/data/docile")
```

---

## Running the Code

**Step 1 — Prepare the data** (only needs to be done once)
```bash
python prepare_data.py
```

**Step 2 — (Optional) Run zero-shot baseline**
```bash
python zero_shot.py
```

**Step 3 — Train**
```bash
python train_roberta.py --experiment full_finetune --save_model --plot
```

**Step 4 — Evaluate**
```bash
python evaluate.py --model_dir models/final_roberta/full_finetune
```

---

## Available Experiments

| Experiment | Frozen Layers | Epochs |
|---|---|---|
| `full_finetune` | embeddings only | 5 |
| `partial_1` | embeddings + first 8 encoder layers | 3 |
| `partial_2` | embeddings + first 6 encoder layers | 5 |
| `partial_3` | embeddings + first 6 encoder layers | 5 |
| `partial_4` | embeddings + first 4 encoder layers | 3 |
| `partial_5` | embeddings + first 3 encoder layers | 3 |
| `lora` | embeddings + first 4 encoder layers + LoRA | 5 |

Pass any experiment name to `--experiment`:
```bash
python train_roberta.py --experiment partial_4 --save_model --plot
```

You can also override epochs or batch size on the fly:
```bash
python train_roberta.py --experiment partial_4 --epochs 3 --batch_size 32
```

To resume from a checkpoint:
```bash
python train_roberta.py --experiment partial_4 --checkpoint checkpoints/roberta/partial_4/checkpoint-500
```

---

## Plotting Loss Curves

Loss curves are saved automatically if you pass `--plot` during training. To plot manually:
```bash
python plot_loss.py \
    --log_file checkpoints/roberta/partial_4/trainer_state.json \
    --title "Partial Fine-Tune 4" \
    --save figures/partial4.png
```
