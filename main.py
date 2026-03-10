"""
main.py

Entry point for the VQA system.

Parses command-line arguments, initialises all components, runs
training and evaluation, saves predictions, and plots training curves.

Usage
-----
    python main.py --model blip --epochs 3 --batch_size 8
    python main.py --model clip  --epochs 3 --batch_size 8

Run  python main.py --help  for a full list of options.
"""

import argparse
import os
import sys

from transformers import BlipProcessor, CLIPProcessor

from src.dataset_processor import DatasetProcessor
from src.model_strategies import BLIPStrategy, CLIPStrategy
from src.vqa_manager import VQAManager
from src.experiment_logger import ExperimentLogger
from src.utils import set_seed, plot_training_curves, compute_class_weights


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQA System -- BLIP / CLIP fine-tuning for multi-class "
        "answer classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join("data", "Dataset_VQA_CVNLP"),
        help="Root directory of the VQA dataset (contains images/, *.csv, "
        "answer_space.txt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["blip", "clip"],
        default="blip",
        help="Model strategy to use for training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Samples per batch."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Accumulate gradients over N steps before an optimizer update.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes. Use 0 on Windows.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="Maximum token length for question encoding.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=os.path.join("outputs", "checkpoints"),
        help="Directory for saving model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for saving predictions and plots.",
    )
    parser.add_argument(
        "--experiments_csv",
        type=str,
        default=os.path.join("experiments", "VQAExperiments.csv"),
        help="Path to the experiment log CSV file.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        default=False,
        help="Disable automatic mixed precision (AMP) training.",
    )
    parser.add_argument(
        "--classifier_dropout",
        type=float,
        default=0.2,
        help="Dropout probability in the MLP classification head (BLIP).",
    )
    parser.add_argument(
        "--class_weight_cap",
        type=float,
        default=5.0,
        help="Maximum class weight cap for CrossEntropyLoss (reduces loss spikes).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("VQA System -- Starting")
    print("=" * 60)
    print(f"  Model     : {args.model.upper()}")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print(f"  AMP       : {not args.no_amp}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Select strategy and matching processor
    # ------------------------------------------------------------------
    if args.model == "blip":
        strategy = BLIPStrategy(dropout=args.classifier_dropout)
        print("[main] Loading BlipProcessor ...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    else:
        strategy = CLIPStrategy()
        print("[main] Loading CLIPProcessor ...")
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("[main] Initialising DatasetProcessor ...")
    dataset_processor = DatasetProcessor(
        data_dir=args.data_dir,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )

    train_loader = dataset_processor.get_train_loader()
    eval_loader = dataset_processor.get_eval_loader()
    num_classes = dataset_processor.get_num_classes()
    label2answer = dataset_processor.label2answer

    print(f"[main] Classes      : {num_classes}")
    print(f"[main] Train batches: {len(train_loader)}")
    print(f"[main] Eval  batches: {len(eval_loader)}")

    # ------------------------------------------------------------------
    # Class weights (for imbalanced answer distribution)
    # ------------------------------------------------------------------
    print("[main] Computing class weights ...")
    class_weights = compute_class_weights(
        dataset_processor.train_df, dataset_processor.answer2label, num_classes,
        cap=args.class_weight_cap,
    )

    # ------------------------------------------------------------------
    # Logger
    # ------------------------------------------------------------------
    logger = ExperimentLogger(csv_path=args.experiments_csv)

    # ------------------------------------------------------------------
    # Manager
    # ------------------------------------------------------------------
    config = {
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "checkpoint_dir": args.checkpoint_dir,
        "mixed_precision": not args.no_amp,
        "classifier_dropout": args.classifier_dropout,
        "class_weight_cap": args.class_weight_cap,
    }

    manager = VQAManager(
        strategy=strategy,
        logger=logger,
        num_classes=num_classes,
        config=config,
        class_weights=class_weights,
    )

    # Optionally resume from a checkpoint
    if args.resume:
        manager.load_checkpoint(args.resume)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n[main] Starting training ...")
    history = manager.train(train_loader, eval_loader)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------
    plot_training_curves(
        train_loss=history["train_loss"],
        val_loss=history["val_loss"],
        accuracy=history["accuracy"],
        output_dir=args.output_dir,
        model_name=strategy.get_name(),
    )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    manager.generate_predictions(
        loader=eval_loader,
        label2answer=label2answer,
        experiment_id="final",
        output_path=predictions_path,
    )

    print("\n[main] All done.")
    print(f"  Predictions : {predictions_path}")
    print(f"  Experiments : {args.experiments_csv}")
    print(f"  Checkpoints : {args.checkpoint_dir}")
    print(f"  Plots       : {args.output_dir}")


if __name__ == "__main__":
    main()
