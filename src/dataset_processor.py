"""
dataset_processor.py

Handles loading, preprocessing, and batching of the VQA dataset.
Converts raw CSV + image data into PyTorch Dataset / DataLoader objects
suitable for multi-class classification with BLIP or CLIP.
"""

import os
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


# Applied only during training. With just 1,449 unique images shared across
# ~9,974 samples, augmentation provides substantial additional variety.
_TRAIN_AUGMENT = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    T.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
])


class VQADataset(Dataset):
    """
    PyTorch Dataset for VQA multi-class classification.

    Each sample returns a dictionary with:
        pixel_values  : preprocessed image tensor
        input_ids     : tokenised question token ids
        attention_mask: attention mask for the question
        answer_label  : integer class label
        image_id      : raw image identifier string
        question      : raw question string
        answer        : raw ground-truth answer string

    Edge-case handling
    ------------------
    Comma-separated multi-label answers (e.g. "picture, wall_decoration") exist
    in ~10% of samples.
      - Training  : _resolve_answer() randomly picks one valid sub-token
                    (label augmentation for multi-label samples).
      - Eval/Test : always picks the first valid sub-token (deterministic).
    Missing images are replaced with a blank black image.

    Image augmentation (training only)
    -----------------------------------
    RandomHorizontalFlip + ColorJitter + RandomResizedCrop applied before the
    HuggingFace processor. With only 1,449 unique images this provides
    significant extra variety.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir: str,
        answer2label: Dict[str, int],
        processor: Any,
        max_length: int = 32,
        is_training: bool = False,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.answer2label = answer2label
        self.processor = processor
        self.max_length = max_length
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.dataframe.iloc[idx]

        # Images are stored as {image_dir}/{image_id}.png
        image_path = os.path.join(self.image_dir, f"{row['image_id']}.png")

        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            # Fallback: blank black image so training can continue
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Apply augmentation to the PIL image BEFORE the processor
        if self.is_training:
            image = _TRAIN_AUGMENT(image)

        question = str(row["question"])
        answer = str(row["answer"])
        label = self._resolve_answer(answer, training=self.is_training)

        encoding = self.processor(
            images=image,
            text=question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        return {
            "pixel_values" : encoding["pixel_values"].squeeze(0),
            "input_ids"    : encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer_label" : torch.tensor(label, dtype=torch.long),
            "image_id"     : str(row["image_id"]),
            "question"     : question,
            "answer"       : answer,
        }

    def _resolve_answer(self, answer: str, training: bool = False) -> int:
        """
        Map an answer string to an integer label.

        Priority:
        1. Exact match in answer_space  (fast path, covers ~90% of data).
        2. Comma-split tokens:
             training=True  -> shuffle tokens, pick first valid (augmentation)
             training=False -> scan left-to-right, pick first valid (deterministic)
        3. Final fallback: label 0.
        """
        answer = answer.strip()
        if answer in self.answer2label:
            return self.answer2label[answer]
        parts = [p.strip() for p in answer.split(",") if p.strip()]
        if training:
            random.shuffle(parts)
        for part in parts:
            if part in self.answer2label:
                return self.answer2label[part]
        return 0


class DatasetProcessor:
    """
    Loads and preprocesses the VQA dataset from disk.

    Responsibilities
    ----------------
    - Parse answer_space.txt into integer label mappings.
    - Load data_train.csv and data_eval.csv.
    - Construct VQADataset and DataLoader objects.

    Parameters
    ----------
    data_dir   : Root directory of the dataset (contains images/, *.csv, etc.).
    processor  : A HuggingFace processor (BlipProcessor or CLIPProcessor).
    batch_size : Number of samples per batch.
    num_workers: DataLoader worker count. Use 0 on Windows if multiprocessing
                 causes issues.
    max_length : Maximum token length for question encoding.
    """

    def __init__(
        self,
        data_dir: str,
        processor: Any,
        batch_size: int = 8,
        num_workers: int = 0,
        max_length: int = 32,
    ) -> None:
        self.data_dir = data_dir
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

        self.image_dir = os.path.join(data_dir, "images")
        self.answer_space_path = os.path.join(data_dir, "answer_space.txt")
        self.train_csv = os.path.join(data_dir, "data_train.csv")
        self.eval_csv = os.path.join(data_dir, "data_eval.csv")

        # Populated by _load_answer_space
        self.answer2label: Dict[str, int] = {}
        self.label2answer: Dict[int, str] = {}
        self._load_answer_space()

        # Pre-load DataFrames; exposed for class-weight computation in VQAManager
        self.train_df = self._load_dataframe(self.train_csv)
        self.eval_df  = self._load_dataframe(self.eval_csv)

    # ------------------------------------------------------------------
    # Answer space
    # ------------------------------------------------------------------

    def _load_answer_space(self) -> None:
        """Parse answer_space.txt and build bidirectional label mappings."""
        with open(self.answer_space_path, "r", encoding="utf-8") as f:
            answers = [line.strip() for line in f if line.strip()]
        self.answer2label = {ans: idx for idx, ans in enumerate(answers)}
        self.label2answer = {idx: ans for idx, ans in enumerate(answers)}

    def get_num_classes(self) -> int:
        """Return the total number of distinct answer classes."""
        return len(self.answer2label)

    # ------------------------------------------------------------------
    # CSV loading with validation
    # ------------------------------------------------------------------

    def _load_dataframe(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = {"question", "answer", "image_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file '{csv_path}' is missing required columns: {missing}"
            )
        df["answer"] = df["answer"].astype(str)
        return df

    # ------------------------------------------------------------------
    # DataLoader factories
    # ------------------------------------------------------------------

    def get_train_loader(self) -> DataLoader:
        """Return a shuffled DataLoader for the training split (augmentation ON)."""
        dataset = VQADataset(
            dataframe=self.train_df,
            image_dir=self.image_dir,
            answer2label=self.answer2label,
            processor=self.processor,
            max_length=self.max_length,
            is_training=True,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_eval_loader(self) -> DataLoader:
        """Return a non-shuffled DataLoader for the evaluation split (no augmentation)."""
        dataset = VQADataset(
            dataframe=self.eval_df,
            image_dir=self.image_dir,
            answer2label=self.answer2label,
            processor=self.processor,
            max_length=self.max_length,
            is_training=False,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
