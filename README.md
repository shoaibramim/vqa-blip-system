# VQA BLIP System

A modular Visual Question Answering system built on `Salesforce/blip-vqa-base`
with an optional `openai/clip-vit-base-patch32` backend, framed as a
**582-class classification** problem over the
[Visual Question Answering — Computer Vision & NLP](https://www.kaggle.com/datasets/bhavikardeshna/visual-question-answering-computer-vision-nlp)
Kaggle dataset.

**Best result achieved:** accuracy **0.3929**, F1 **0.3895**, BLEU **0.0814**
(epoch 10, BLIP, Kaggle T4 GPU).

---

## Project Overview

Given an image and a natural-language question, the system predicts a single
answer from a fixed 582-entry answer vocabulary.

The architecture uses the **Strategy Pattern** so the underlying
vision-language backbone (BLIP or CLIP) can be swapped without touching the
training loop.

---

## Improvements Over Baseline

| Root Cause                                                        | Fix Applied                                                                                                           |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| First comma-token of multi-label answers always mapped to label 0 | `_resolve_answer()` splits on commas and walks sub-tokens; training randomly shuffles sub-tokens (label augmentation) |
| Only the last epoch logged in `VQAExperiments.csv`                | `ExperimentLogger.log_epoch()` appends immediately; row ID uses `{uuid}_ep{epoch}`                                    |
| Flat MLP head after CLS token                                     | 3-layer MLP: `Linear → LayerNorm → GELU → Dropout → Linear`                                                           |
| Constant learning rate                                            | Cosine schedule with 10 % linear warm-up (`get_cosine_schedule_with_warmup`)                                          |
| Class-weight cap of 10× caused loss spikes (33.8 observed)        | Cap reduced to **3×** — limits per-batch loss contribution of singleton classes                                       |
| Only 1,449 unique images reused ~7× → overfitting                 | `RandomHorizontalFlip + ColorJitter + RandomResizedCrop` (training split only)                                        |
| Per-epoch checkpoint (~400 MB × 6 = 2.4 GB) filled Kaggle disk    | **Best-only checkpointing** — single `BLIP_best.pt`; old file deleted before new write                                |
| High dropout (0.3) underfitting small head                        | `Dropout(0.15)` — reduced regularisation for a well-constrained classification head                                   |
| Gradient accumulation steps 4 → effective batch too small         | Accumulation steps **8** → effective batch of **64**                                                                  |

---

## Architecture

```
Image + Question
       │
       ▼
  Processor (BlipProcessor / CLIPProcessor)
       │
       ▼
  ModelStrategy
  ┌────────────────────┐        ┌──────────────────────┐
  │  BLIPStrategy      │   or   │  CLIPStrategy        │
  │  blip-vqa-base     │        │  clip-vit-base-patch32│
  └────────────────────┘        └──────────────────────┘
       │                                 │
  BLIP Vision + Text encoder        CLIP Image + Text encoder
  (cross-attention fusion)           (frozen, concat fusion)
       │                                 │
  [CLS] embedding (768-d)          Concatenated (1024-d)
       │                                 │
  Linear(768→512) → LayerNorm → GELU → Dropout(0.15) → Linear(512→C)
       │
       ▼  argmax
  Predicted Answer Class  ──►  VQAManager ──► ExperimentLogger
                                    │
                          outputs/predictions.csv
                          outputs/BLIP_training_curves.png
                          outputs/checkpoints/BLIP_best.pt
```

Training details:

- **Optimiser:** AdamW (`lr=3e-5`, `weight_decay=0.01`)
- **Schedule:** cosine with 10 % warm-up, stepped per optimizer update
- **Loss:** `CrossEntropyLoss` with label smoothing 0.1 and inverse-frequency
  class weights (capped at 3×)
- **AMP:** `torch.cuda.amp` mixed precision enabled by default
- **Gradient accumulation:** 8 steps → effective batch size 64
- **Vision freeze:** first 3 epochs; then unfrozen at 10× reduced LR

---

## Repository Structure

```
vqa-blip-system/
│
├── data/
│   └── Dataset_VQA_CVNLP.txt    
│
├── src/
│   ├── __init__.py
│   ├── dataset_processor.py      VQADataset (+ augmentation) + DatasetProcessor
│   ├── model_strategies.py       ModelStrategy, BLIPStrategy, CLIPStrategy
│   ├── vqa_manager.py            Training / validation / inference loop
│   ├── evaluator.py              Accuracy, F1, BLEU computation
│   ├── experiment_logger.py      Per-epoch CSV experiment log
│   └── utils.py                  Seed, class weights, plotting helpers
│
├── notebooks/
│   └── vqa-blip-system.ipynb     Self-contained Kaggle notebook
│
├── experiments/
│   └── VQAExperiments.csv
│
├── outputs/
│   ├── predictions.csv
│   └── checkpoints/
│       └── BLIP_best.pt
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Dataset

| File               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `data_train.csv`   | 9,974 training samples (question, answer, image) |
| `data_eval.csv`    | 2,494 evaluation samples                         |
| `answer_space.txt` | 582 possible answer classes, one per line        |
| `images/{id}.png`  | 1,449 PNG images keyed by image_id               |

The task is framed as **multi-class classification**: each answer string in
`answer_space.txt` is mapped to an integer label.

~10 % of answer strings are comma-separated multi-labels (e.g.
`"picture, wall_decoration"`). During training, the system randomly selects one
valid sub-token per sample (label augmentation); during evaluation it always
picks the first valid sub-token (deterministic).

---

## Training Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/vqa-blip-system.git
cd vqa-blip-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Copy or symlink the dataset into `data/`:

```
data/
  Dataset_VQA_CVNLP/
    images/
    data_train.csv
    data_eval.csv
    answer_space.txt
```

### 5. Run training with BLIP (default)

```bash
python main.py --model blip --epochs 10 --batch_size 8
```

### 6. Run training with CLIP

```bash
python main.py --model clip --epochs 10 --batch_size 8
```

### 7. Resume from the best checkpoint

```bash
python main.py --model blip --resume outputs/checkpoints/BLIP_best.pt
```

### Full argument reference

```
--data_dir              Path to dataset root           (default: data/Dataset_VQA_CVNLP)
--model                 blip | clip                    (default: blip)
--epochs                Training epochs                (default: 10)
--batch_size            Samples per batch              (default: 8)
--lr                    AdamW learning rate            (default: 3e-5)
--gradient_accum...     Grad accumulation steps        (default: 8)
--classifier_dropout    MLP head dropout probability   (default: 0.15)
--class_weight_cap      Max class weight for CE loss   (default: 3.0)
--num_workers           DataLoader workers             (default: 0)
--max_length            Question token max length      (default: 32)
--checkpoint_dir        Checkpoint save dir            (default: outputs/checkpoints)
--output_dir            Output dir for plots/preds     (default: outputs)
--experiments_csv       Experiment log CSV path        (default: experiments/VQAExperiments.csv)
--seed                  Random seed                    (default: 42)
--no_amp                Disable mixed precision        (flag)
--resume                Checkpoint path to resume      (optional)
```

---

## Kaggle Usage

1. Upload the dataset to Kaggle Datasets (or add the existing
   `bhavikardeshna/visual-question-answering-computer-vision-nlp` dataset).
2. Create a new Kaggle notebook and attach the dataset.
3. Upload `notebooks/vqa-blip-system.ipynb` to Kaggle.
4. Set the accelerator to **GPU T4 x1** in the notebook settings.
5. Update the `DATASET_DIR` variable in the CONFIG cell if your input path
   differs from the default.
6. Run all cells sequentially.

The notebook is fully self-contained — all class definitions are inlined so
no internet access beyond HuggingFace model downloads is required.

---

## Evaluation Metrics

| Metric   | Description                                                     |
| -------- | --------------------------------------------------------------- |
| Accuracy | Exact match between predicted and ground-truth answer class     |
| F1       | Weighted F1 score across all answer classes (handles imbalance) |
| BLEU     | Sentence BLEU treating answer strings as word sequences         |

Metrics are printed after each validation epoch and appended immediately to
`experiments/VQAExperiments.csv` (one row per epoch with a unique
`{uuid}_ep{epoch}` ID).

### Benchmark Results (BLIP, 10 epochs, Kaggle T4)

| Epoch  | Val Accuracy | Val F1     | Val BLEU   |
| ------ | ------------ | ---------- | ---------- |
| 1      | 0.2838       | 0.2462     | 0.0465     |
| 5      | 0.3624       | 0.3521     | 0.0698     |
| **10** | **0.3929**   | **0.3895** | **0.0814** |

---

## Example Predictions

After training, `outputs/predictions.csv` contains rows such as:

| image_url        | question                          | answer | ground_truth |
| ---------------- | --------------------------------- | ------ | ------------ |
| .../image100.png | what is the object on the shelves | cup    | cup          |
| .../image888.png | how many chairs are there         | 6      | 6            |
| .../image399.png | what is the colour of the bag     | pink   | pink         |

---

## Pushing to GitHub

```bash
# Initialise git in the project root
git init
git add .
git commit -m "Initial commit: VQA BLIP system"

# Push to a new GitHub repository
git remote add origin https://github.com/<your-username>/vqa-blip-system.git
git branch -M main
git push -u origin main
```

Add a `.gitignore` to exclude large files:

```gitignore
outputs/checkpoints/
data/
__pycache__/
*.pt
*.pyc
.env
venv/
```

---

## Hardware Requirements

| Setting       | Minimum (Kaggle T4) | Recommended       |
| ------------- | ------------------- | ----------------- |
| GPU VRAM      | 8 GB                | 16 GB (A100/V100) |
| RAM           | 8 GB                | 16 GB             |
| Training time | ~20 min / epoch     | ~8 min / epoch    |

Mixed precision (AMP) and gradient accumulation (steps=8, effective batch=64)
are enabled by default to maximise GPU utilisation on free-tier Kaggle T4 GPUs.
Only the best checkpoint is saved (`BLIP_best.pt`, ~400 MB) to stay within the
Kaggle 20 GB disk limit.
