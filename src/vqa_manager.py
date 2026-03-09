"""
vqa_manager.py

Central orchestrator for training and validation of VQA models.

Responsibilities
----------------
- Model initialisation via a ModelStrategy object.
- Full training loop with optional gradient accumulation and mixed precision.
- Validation loop with metric computation via Evaluator.
- Per-epoch checkpoint saving to outputs/checkpoints/.
- Delegation of logging to ExperimentLogger.
- Inference pass to produce a predictions CSV.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.model_strategies import ModelStrategy
from src.experiment_logger import ExperimentLogger
from src.evaluator import Evaluator


class VQAManager:
    """
    Orchestrates the full training / validation / inference pipeline.

    Parameters
    ----------
    strategy      : A concrete ModelStrategy instance.
    logger        : ExperimentLogger for per-epoch CSV records.
    num_classes   : Number of answer classes.
    config        : Optional dict overriding default hyperparameters.
    class_weights : Optional inverse-frequency weight tensor (num_classes,).

    Default hyperparameters
    -----------------------
    learning_rate               : 3e-5
    batch_size                  : 8
    epochs                      : 10
    gradient_accumulation_steps : 8
    label_smoothing             : 0.1
    warmup_ratio                : 0.1
    weight_decay                : 0.01
    max_grad_norm               : 1.0
    freeze_vision_epochs        : 3
    classifier_dropout          : 0.15
    class_weight_cap            : 3.0
    checkpoint_dir              : outputs/checkpoints
    mixed_precision             : True
    """

    _DEFAULT_CONFIG: Dict[str, Any] = {
        "learning_rate"              : 3e-5,
        "batch_size"                 : 8,
        "epochs"                     : 10,
        "gradient_accumulation_steps": 8,
        "label_smoothing"            : 0.1,
        "warmup_ratio"               : 0.1,
        "weight_decay"               : 0.01,
        "max_grad_norm"              : 1.0,
        "freeze_vision_epochs"       : 3,
        "classifier_dropout"         : 0.15,
        "class_weight_cap"           : 3.0,
        "checkpoint_dir"             : "outputs/checkpoints",
        "mixed_precision"            : True,
    }

    def __init__(
        self,
        strategy: ModelStrategy,
        logger: ExperimentLogger,
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.strategy = strategy
        self.logger = logger
        self.num_classes = num_classes
        self.config: Dict[str, Any] = {**self._DEFAULT_CONFIG, **(config or {})}

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[VQAManager] Device: {self.device}")

        self.strategy.load_model(num_classes=num_classes, device=self.device)
        self.model: nn.Module = self.strategy.get_model()

        # Only optimise trainable parameters (frozen vision encoder is excluded)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=self.config["learning_rate"],
            weight_decay=self.config.get("weight_decay", 0.01),
        )

        cw = class_weights.to(self.device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=cw,
            label_smoothing=self.config.get("label_smoothing", 0.1),
        )

        use_amp = (
            self.config["mixed_precision"] and self.device.type == "cuda"
        )
        self.scaler = GradScaler(enabled=use_amp)
        self._use_amp = use_amp
        self._scheduler = None

        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)

        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
        self.accuracy_history: List[float] = []

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        """
        Run the full training loop over all configured epochs.

        Returns
        -------
        Dictionary with keys "train_loss", "val_loss", "accuracy".
        """
        epochs        = self.config["epochs"]
        accum         = self.config["gradient_accumulation_steps"]
        freeze_epochs = self.config.get("freeze_vision_epochs", 0)

        # Build cosine schedule over total optimizer steps
        steps_per_epoch = int(np.ceil(len(train_loader) / accum))
        total_opt_steps = steps_per_epoch * epochs
        warmup_steps    = int(self.config.get("warmup_ratio", 0.1) * total_opt_steps)
        self._scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_opt_steps
        )
        print(
            f"[VQAManager] Scheduler: {total_opt_steps} total steps, "
            f"{warmup_steps} warmup steps"
        )

        experiment_id = self.logger.start_experiment(
            model_name=self.strategy.get_name(),
            hyperparameters=self.config,
        )

        for epoch in range(1, epochs + 1):
            # Unfreeze vision encoder after freeze_vision_epochs
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                if hasattr(self.strategy, "unfreeze_vision"):
                    self.strategy.unfreeze_vision()
                    existing_ids = {
                        id(p)
                        for group in self.optimizer.param_groups
                        for p in group["params"]
                    }
                    new_params = [
                        p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in existing_ids
                    ]
                    if new_params:
                        self.optimizer.add_param_group({
                            "params": new_params,
                            "lr"    : self.config["learning_rate"] * 0.1,
                        })
                        print(
                            f"[VQAManager] Added {len(new_params)} vision params "
                            f"at lr={self.config['learning_rate'] * 0.1:.1e}"
                        )

            print(f"\n{'='*60}\nEpoch {epoch} / {epochs}\n{'='*60}")

            train_loss = self._train_one_epoch(train_loader, epoch)
            val_loss, metrics = self._validate_one_epoch(val_loader)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.accuracy_history.append(metrics.get("accuracy", 0.0))

            print(
                f"  [Summary] train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"accuracy={metrics.get('accuracy', 0):.4f}  "
                f"f1={metrics.get('f1', 0):.4f}  "
                f"bleu={metrics.get('bleu', 0):.4f}"
            )

            self._save_checkpoint(epoch)
            self.logger.log_epoch(
                experiment_id=experiment_id,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
            )

        self.logger.finish_experiment(experiment_id)

        return {
            "train_loss": self.train_loss_history,
            "val_loss": self.val_loss_history,
            "accuracy": self.accuracy_history,
        }

    # ------------------------------------------------------------------
    # Single epoch: training
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self, loader: DataLoader, epoch: int
    ) -> float:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        accum_steps = self.config["gradient_accumulation_steps"]

        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            labels = batch["answer_label"].to(self.device)

            with autocast(enabled=self._use_amp):
                logits = self.strategy.forward(batch)
                # Divide by accum_steps so gradients are averaged, not summed
                loss = self.criterion(logits, labels) / accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("max_grad_norm", 1.0),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self._scheduler is not None:
                    self._scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            total_steps += 1

            if step % 50 == 0:
                print(
                    f"  [Epoch {epoch}] step {step+1}/{len(loader)} "
                    f"| loss={loss.item() * accum_steps:.4f}"
                )

        return total_loss / max(total_steps, 1)

    # ------------------------------------------------------------------
    # Single epoch: validation
    # ------------------------------------------------------------------

    def _validate_one_epoch(
        self, loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_questions: List[str] = []

        with torch.no_grad():
            for batch in loader:
                labels = batch["answer_label"].to(self.device)

                with autocast(enabled=self._use_amp):
                    logits = self.strategy.forward(batch)
                    loss = self.criterion(logits, labels)

                preds = logits.argmax(dim=-1)
                total_loss += loss.item()
                total_steps += 1

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_questions.extend(batch["question"])

        evaluator = Evaluator()
        metrics = evaluator.compute(
            predictions=all_preds,
            references=all_labels,
            questions=all_questions,
        )
        avg_loss = total_loss / max(total_steps, 1)
        return avg_loss, metrics

    # ------------------------------------------------------------------
    # Inference: generate predictions CSV
    # ------------------------------------------------------------------

    def generate_predictions(
        self,
        loader: DataLoader,
        label2answer: Dict[int, str],
        experiment_id: str,
        output_path: str,
    ) -> None:
        """
        Run inference on a DataLoader and save predictions to a CSV file.

        Output CSV columns: id, experiment_id, image_id, question,
                            predicted_answer, timestamp.

        Parameters
        ----------
        loader         : DataLoader over the split to evaluate.
        label2answer   : Mapping from predicted class index to answer string.
        experiment_id  : UUID string of the parent experiment run.
        output_path    : Target path for the output CSV file.
        """
        self.model.eval()
        records: List[Dict[str, str]] = []

        with torch.no_grad():
            for batch in loader:
                preds = self.strategy.predict(batch)
                for i, pred_idx in enumerate(preds.cpu().tolist()):
                    records.append(
                        {
                            "id"           : str(uuid.uuid4()),
                            "experiment_id": experiment_id,
                            "image_url"    : batch["image_id"][i],
                            "question"     : batch["question"][i],
                            "answer"       : label2answer.get(
                                pred_idx, "unknown"
                            ),
                            "ground_truth" : batch["answer"][i],
                            "timestamp"    : datetime.now(
                                tz=timezone.utc
                            ).isoformat(),
                        }
                    )

        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[VQAManager] Predictions saved to '{output_path}' ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(
            self.config["checkpoint_dir"],
            f"{self.strategy.get_name()}_epoch_{epoch}.pt",
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss_history": self.train_loss_history,
                "val_loss_history": self.val_loss_history,
                "accuracy_history": self.accuracy_history,
            },
            path,
        )
        print(f"[VQAManager] Checkpoint saved: '{path}'")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Restore model and optimizer state from a checkpoint file.

        Returns
        -------
        The epoch number stored in the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss_history = checkpoint.get("train_loss_history", [])
        self.val_loss_history = checkpoint.get("val_loss_history", [])
        self.accuracy_history = checkpoint.get("accuracy_history", [])
        epoch = checkpoint.get("epoch", 0)
        print(f"[VQAManager] Loaded checkpoint from epoch {epoch}")
        return epoch
