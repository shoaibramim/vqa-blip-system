"""
utils.py

Shared utility functions for the VQA system:
    - Random seed initialisation for reproducibility
    - Answer space helpers
    - Training curve visualisation
    - JSON serialisation helpers
"""

import json
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for servers and Kaggle
import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch to ensure reproducibility.

    Parameters
    ----------
    seed : Integer seed value (default 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Answer space helpers
# ---------------------------------------------------------------------------


def load_answer_space(
    path: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Load answer_space.txt and return bidirectional label-answer mappings.

    Parameters
    ----------
    path : Absolute or relative path to answer_space.txt.

    Returns
    -------
    answer2label : dict mapping answer string -> integer label.
    label2answer : dict mapping integer label -> answer string.
    """
    with open(path, "r", encoding="utf-8") as f:
        answers = [line.strip() for line in f if line.strip()]
    answer2label: Dict[str, int] = {a: i for i, a in enumerate(answers)}
    label2answer: Dict[int, str] = {i: a for i, a in enumerate(answers)}
    return answer2label, label2answer


# ---------------------------------------------------------------------------
# Class-weight helpers
# ---------------------------------------------------------------------------


def compute_class_weights(
    dataframe,
    answer2label: Dict[str, int],
    num_classes: int,
    cap: float = 3.0,
) -> "torch.Tensor":
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Parameters
    ----------
    dataframe   : pandas DataFrame with an 'answer' column.
    answer2label: Mapping from answer string to integer label.
    num_classes : Total number of classes.
    cap         : Maximum weight value (default 3.0). Limits the loss
                  contribution of singleton / near-singleton classes.

    Returns
    -------
    1-D FloatTensor of shape (num_classes,) with weights capped at `cap`.
    """
    labels = [
        answer2label.get(str(a).strip(), -1)
        for a in dataframe["answer"]
    ]
    valid = [l for l in labels if 0 <= l < num_classes]
    counts = Counter(valid)
    total = len(valid)
    weights = torch.ones(num_classes)
    for cls, cnt in counts.items():
        weights[cls] = total / (num_classes * cnt)
    return weights.clamp(max=cap)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_loss: List[float],
    val_loss: List[float],
    accuracy: List[float],
    output_dir: str = "outputs",
    model_name: str = "model",
) -> str:
    """
    Plot training loss, validation loss, and accuracy against epoch number
    and save the figure as a PNG file.

    Parameters
    ----------
    train_loss : List of training losses, one per epoch.
    val_loss   : List of validation losses, one per epoch.
    accuracy   : List of validation accuracies, one per epoch.
    output_dir : Directory where the PNG is saved.
    model_name : Used in the figure title and filename.

    Returns
    -------
    Absolute path of the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Training Curves - {model_name}", fontsize=14, fontweight="bold")

    # --- Training loss ---
    axes[0].plot(epochs, train_loss, marker="o", color="tab:blue", label="Train Loss")
    axes[0].set_title("Training Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # --- Validation loss ---
    axes[1].plot(epochs, val_loss, marker="s", color="tab:orange", label="Val Loss")
    axes[1].set_title("Validation Loss vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # --- Accuracy ---
    axes[2].plot(epochs, accuracy, marker="^", color="tab:green", label="Accuracy")
    axes[2].set_title("Accuracy vs Epoch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[utils] Training curves saved to '{save_path}'")
    return save_path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def save_json(data: dict, path: str) -> None:
    """Serialise a dictionary to a JSON file, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict:
    """Deserialise a JSON file and return the parsed dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
