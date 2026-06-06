#!/usr/bin/env python3
"""
Plot a training / validation loss curve from manually recorded values or from
a Trainer log JSON file.

Usage (manual values):
    python plot_loss.py \
        --epochs 1 2 3 4 5 \
        --train_loss 0.577 0.524 0.469 0.483 0.469 \
        --val_loss   0.770 0.767 0.775 0.747 0.750 \
        --title "Full Fine-Tune" \
        --save figures/full_finetune_loss.png

Usage (from Trainer log — trainer_state.json):
    python plot_loss.py \
        --log_file checkpoints/roberta/partial_4/trainer_state.json \
        --title "Partial Fine-Tune 4" \
        --save figures/partial4_loss.png
"""

import argparse
import json
from pathlib import Path

from roberta.plotting import plot_from_trainer_log, plot_loss_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Plot epoch vs loss curves")
    parser.add_argument("--epochs",      nargs="+", type=int)
    parser.add_argument("--train_loss",  nargs="+", type=float)
    parser.add_argument("--val_loss",    nargs="+", type=float)
    parser.add_argument("--log_file",    default=None, help="Path to trainer_state.json")
    parser.add_argument("--title",       default="Epoch vs Loss")
    parser.add_argument("--save",        default=None, help="Output image path (PNG)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.log_file:
        with open(args.log_file) as f:
            state = json.load(f)
        plot_from_trainer_log(
            state.get("log_history", []),
            title     = args.title,
            save_path = Path(args.save) if args.save else None,
        )
    elif args.epochs and args.train_loss and args.val_loss:
        plot_loss_curve(
            epochs          = args.epochs,
            training_loss   = args.train_loss,
            validation_loss = args.val_loss,
            title           = args.title,
            save_path       = Path(args.save) if args.save else None,
        )
    else:
        print(
            "Provide either --log_file OR all of --epochs, --train_loss, --val_loss.\n"
            "Run with --help for usage."
        )


if __name__ == "__main__":
    main()
