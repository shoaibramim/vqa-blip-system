"""
experiment_logger.py

Manages experiment records in experiments/VQAExperiments.csv.

Each epoch of each experiment is written as its own row immediately when
log_epoch() is called, so the full learning curve is persisted even if
training is interrupted.  The epoch number is stored both in the row 'id'
(as '{uuid}_ep{epoch}') and inside the 'metrics' JSON dict.
"""

import csv
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DEFAULT_CSV_PATH = os.path.join("experiments", "VQAExperiments.csv")

CSV_COLUMNS = [
    "id",
    "model_name",
    "hyperparameters",
    "train_loss",
    "val_loss",
    "metrics",
    "timestamp",
]


class ExperimentLogger:
    """
    Append-only CSV logger for VQA training experiments.

    Usage
    -----
    logger = ExperimentLogger()
    exp_id = logger.start_experiment("BLIP", config)
    for epoch in range(epochs):
        ...
        logger.log_epoch(exp_id, epoch, train_loss, val_loss, metrics)
    logger.finish_experiment(exp_id)
    """

    def __init__(self, csv_path: str = DEFAULT_CSV_PATH) -> None:
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._ensure_header()
        # In-memory store keyed by experiment_id
        self._active: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_header(self) -> None:
        """Create the CSV with a header row if it does not already exist."""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_experiment(
        self, model_name: str, hyperparameters: Dict[str, Any]
    ) -> str:
        """
        Register a new experiment.

        Returns
        -------
        Unique experiment ID string (UUID4).
        """
        exp_id = str(uuid.uuid4())
        # Only store the fields needed to construct per-epoch rows.
        self._active[exp_id] = {
            "model_name"     : model_name,
            "hyperparameters": json.dumps(hyperparameters),
        }
        print(
            f"[ExperimentLogger] Started experiment {exp_id} "
            f"(model='{model_name}')"
        )
        return exp_id

    def log_epoch(
        self,
        experiment_id: str,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
    ) -> None:
        """
        Append one CSV row per epoch immediately.

        The row 'id' is '{experiment_id}_ep{epoch}' so every epoch is
        uniquely addressable.  The epoch number is also embedded inside
        the metrics JSON for easy filtering.
        """
        if experiment_id not in self._active:
            raise KeyError(
                f"Experiment '{experiment_id}' not found. "
                "Call start_experiment() first."
            )
        base = self._active[experiment_id]
        row = {
            "id"             : f"{experiment_id}_ep{epoch}",
            "model_name"     : base["model_name"],
            "hyperparameters": base["hyperparameters"],
            "train_loss"     : round(float(train_loss), 6),
            "val_loss"       : round(float(val_loss), 6),
            "metrics"        : json.dumps({**metrics, "epoch": epoch}),
            "timestamp"      : self._utc_now(),
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

        print(
            f"[ExperimentLogger] Epoch {epoch:>2} | "
            f"train={train_loss:.4f}  val={val_loss:.4f}  "
            f"acc={metrics.get('accuracy', 0):.4f}  "
            f"f1={metrics.get('f1', 0):.4f}  "
            f"bleu={metrics.get('bleu', 0):.4f}"
        )

    def finish_experiment(self, experiment_id: str) -> None:
        """
        Remove the experiment from the in-memory store.
        All epoch rows have already been written by log_epoch().
        """
        if experiment_id not in self._active:
            raise KeyError(f"Experiment '{experiment_id}' not found.")
        self._active.pop(experiment_id)
        print(
            f"[ExperimentLogger] Experiment {experiment_id} complete. "
            f"Log: '{self.csv_path}'"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()
