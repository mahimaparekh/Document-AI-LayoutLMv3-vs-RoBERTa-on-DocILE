"""
Plotting utilities for training / validation loss curves.
"""

from pathlib import Path
from typing import List, Optional


def plot_loss_curve(
    epochs:          List[int],
    training_loss:   List[float],
    validation_loss: List[float],
    title:           str = "Epoch vs Loss",
    save_path:       Optional[Path] = None,
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        epochs:          List of epoch numbers (e.g. [1, 2, 3, 4, 5]).
        training_loss:   Loss values for each epoch.
        validation_loss: Validation loss values for each epoch.
        title:           Plot title.
        save_path:       If given, saves the figure there instead of showing it.
    """
    import matplotlib.pyplot as plt

    # Truncate to the shorter of the two lists so mismatched lengths never crash
    n = min(len(epochs), len(training_loss), len(validation_loss))
    epochs, training_loss, validation_loss = (
        epochs[:n], training_loss[:n], validation_loss[:n]
    )

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, training_loss,   marker="o", label="Training Loss",   linewidth=2)
    plt.plot(epochs, validation_loss, marker="o", label="Validation Loss", linewidth=2)
    plt.title(title, fontsize=14, weight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss",  fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Saved figure → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_from_trainer_log(
    log_history: list,
    title:       str = "Epoch vs Loss",
    save_path:   Optional[Path] = None,
) -> None:
    """
    Build the loss curve directly from the Trainer's log_history list
    (trainer.state.log_history after training).
    """
    train_entries = [e for e in log_history if "loss" in e and "epoch" in e]
    eval_entries  = [e for e in log_history if "eval_loss" in e]

    if not train_entries or not eval_entries:
        print("[plot] Not enough log data to plot.")
        return

    # Average training loss per epoch
    from collections import defaultdict
    epoch_losses: dict = defaultdict(list)
    for e in train_entries:
        epoch_losses[round(e["epoch"])].append(e["loss"])
    sorted_epochs = sorted(epoch_losses)
    t_loss = [sum(epoch_losses[ep]) / len(epoch_losses[ep]) for ep in sorted_epochs]
    v_loss = [e["eval_loss"] for e in eval_entries[: len(sorted_epochs)]]

    plot_loss_curve(sorted_epochs, t_loss, v_loss, title=title, save_path=save_path)
