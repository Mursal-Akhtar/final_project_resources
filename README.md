# Deep Learning — Multi‑label Retinal Disease Detection

This repository contains a complete PyTorch workflow for multi‑label classification of retinal fundus images, targeting three diseases:
- Diabetic Retinopathy (DR)
- Glaucoma (G)
- Age‑related Macular Degeneration (AMD)

The project progresses through four tasks:
1) Transfer learning strategies (baseline vs. fine‑tuning),
2) Loss function exploration under class imbalance,
3) Attention mechanisms (Squeeze‑and‑Excitation and Multi‑Head Attention),
4) Final ensembling and test‑time augmentation (TTA).

It also includes utilities for data checks, single‑backbone training, and generating CSV submissions (e.g., for Kaggle‑style onsite test sets).

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Data and Label Format](#data-and-label-format)
- [Environment and Installation](#environment-and-installation)
- [Quick Start](#quick-start)
- [Tasks and Scripts](#tasks-and-scripts)
  - [Task 1 — Transfer Learning Strategies](#task-1--transfer-learning-strategies)
  - [Task 2 — Loss Functions under Class Imbalance](#task-2--loss-functions-under-class-imbalance)
  - [Task 3 — Attention Mechanisms](#task-3--attention-mechanisms)
  - [Task 4 — Final Ensembling and TTA](#task-4--final-ensembling-and-tta)
- [Utilities](#utilities)
- [Checkpoints and Pretrained Weights](#checkpoints-and-pretrained-weights)
- [Results and Metrics](#results-and-metrics)
- [Tips and Troubleshooting](#tips-and-troubleshooting)
- [Reproducibility](#reproducibility)
- [Acknowledgments](#acknowledgments)

---

## Repository Structure

- [code_template.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/code_template.py) — Minimal end‑to‑end training/eval template for a single backbone.
- [task1.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task1.py) — Baseline vs. fine‑tuning strategies (head‑only vs. full).
- [task2.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task2.py) — Loss functions: BCE, Focal Loss, Class‑Balanced Loss.
- [task3.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task3.py) — Attention‑augmented models: SE blocks, Multi‑Head Attention.
- [task4_final.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task4_final.py) — Final ensembling, TTA, and summary.
- [generate_predictions.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/generate_predictions.py) — Produce onsite test CSV submissions from saved checkpoints.
- [debug_task1.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/debug_task1.py) — Sanity checks for data and pretrained checkpoints.

Folders and data files:
- checkpoints/ — Saved model weights from training runs.
- pretrained_backbone/ — Provided pretrained weights for backbones (e.g., ResNet18, EfficientNet‑B0).
- images/
  - images/train — Training images
  - images/val — Validation images
  - images/offsite_test — “Offsite” test images (used for evaluation in scripts)
  - images/onsite_test — “Onsite” test images (for final submissions)
- train.csv, val.csv — Labeled splits for training and validation.
- offsite_test.csv — Labeled test set used for offline evaluation.
- onsite_test_submission.csv — Unlabeled test set for producing a submission file.

---

## Data and Label Format

All CSVs are expected to have the following columns:
- id — image filename (relative to the corresponding images/* directory)
- D — DR label (0/1)
- G — Glaucoma label (0/1)
- A — AMD label (0/1)

Example (train.csv or val.csv):
```
id,D,G,A
0001.png,1,0,0
0002.png,0,1,1
...
```

- For offsite_test.csv: same structure with labels present, used to compute metrics.
- For onsite_test_submission.csv: typically only the id column is provided; the scripts generate D/G/A predictions and save them to a submission CSV.

Directory layout example:
- images/train/<image files referenced by train.csv>
- images/val/<image files referenced by val.csv>
- images/offsite_test/<image files referenced by offsite_test.csv>
- images/onsite_test/<image files referenced by onsite_test_submission.csv>

---

## Environment and Installation

Requirements (typical versions):
- Python 3.9+ (3.10+ recommended)
- PyTorch and torchvision (GPU recommended)
- numpy, pandas, scikit‑learn, Pillow

Install example (pip):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick the right CUDA/CPU wheel
pip install numpy pandas scikit-learn pillow
```

Install example (conda):
```bash
conda create -n retina python=3.10 -y
conda activate retina
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pandas scikit-learn pillow
```

---

## Quick Start

1) Put your images into the images/ folders and ensure the CSV files reference the correct filenames.
2) Place provided pretrained weights into pretrained_backbone/:
   - ./pretrained_backbone/ckpt_resnet18_ep50.pt
   - ./pretrained_backbone/ckpt_efficientnet_ep50.pt
3) Run a quick sanity check:
```bash
python debug_task1.py
```
4) Try a complete baseline + fine‑tuning experiment:
```bash
python task1.py
```

---

## Tasks and Scripts

### Task 1 — Transfer Learning Strategies
Script: [task1.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task1.py)

What it does:
- Compares three strategies on ResNet18 and EfficientNet‑B0:
  1) Baseline (no fine‑tuning) — load the pretrained model and evaluate only.
  2) Fine‑tune classifier head only — freeze the backbone, train only FC/classifier.
  3) Fine‑tune full model — unfreeze all layers, train with low LR.

Outputs:
- Saves best checkpoints under checkpoints/ (e.g., task1_2_resnet18.pt, task1_3_efficientnet.pt).
- Prints per‑class metrics and average F1.
- Saves a summary JSON: task1_results.json.

Run:
```bash
python task1.py
```

Key functions:
- build_model(...) — constructs backbone (resnet18 or efficientnet_b0).
- task1_baseline(...), task1_finetune_head(...), task1_finetune_full(...).
- evaluate_model(...) — computes Accuracy, Precision, Recall, F1, Cohen’s Kappa for each label and the average F1.

---

### Task 2 — Loss Functions under Class Imbalance
Script: [task2.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task2.py)

What it does:
- Compares three loss functions for multi‑label classification:
  - BCEWithLogitsLoss (baseline)
  - Focal Loss (focuses on hard examples)
  - Class‑Balanced Loss (reweights by effective number of samples)
- Trains both ResNet18 and EfficientNet‑B0 with each loss.
- Optionally generates onsite submission CSVs.

Outputs:
- Best checkpoints saved to checkpoints/ (e.g., task2_focal_resnet18.pt).
- Prints per‑class metrics and average F1.
- Writes task2_results.json with all results.
- If configured, writes CSV submissions to submissions/ (e.g., task2_focal_resnet18.csv).

Run:
```bash
python task2.py
```

Notes:
- Calculates samples_per_class from train.csv for class‑balanced weighting.
- Set onsite_csv and onsite_image_dir to generate a submission file automatically.

---

### Task 3 — Attention Mechanisms
Script: [task3.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task3.py)

What it does:
- Augments backbones with attention:
  - SEBlock (Squeeze‑and‑Excitation) at multiple stages.
  - MultiHeadAttention (MHA) at deeper stages (smaller spatial maps) for stability.
- Provides custom model definitions:
  - ResNet18WithAttention(attention_type="se"|"mha")
  - EfficientNetWithAttention(attention_type="se"|"mha")

Outputs:
- Best attention‑augmented checkpoints saved to checkpoints/ (e.g., task3_se_resnet18.pt).
- Prints per‑class metrics and average F1.
- Saves task3_results.json.

Run:
```bash
python task3.py
```

Notes:
- MHA can be more memory‑intensive; the script reduces batch size for MHA if needed.
- For ResNet18, full fine‑tuning works well; EfficientNet is often better with head‑only fine‑tuning.

---

### Task 4 — Final Ensembling and TTA
Script: [task4_final.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/task4_final.py)

What it does:
- Loads best Task 3 checkpoints for ResNet18 with SE and MHA:
  - Expected paths:
    - checkpoints/task3_se_resnet18.pt
    - checkpoints/task3_mha_resnet18.pt
- Evaluates several final strategies:
  - Weighted logit ensembling with TTA:
    - SE:MHA = 0.8:0.2
    - SE:MHA = 0.9:0.1
  - Pure SE with and without TTA
  - Hard voting ensemble
- Reports validation and offsite test macro‑F1 and prints a summary.
- Saves task4_final_results.json.

Run:
```bash
python task4_final.py
```

Notes:
- TTA uses a simple horizontal flip for inference.
- Threshold of 0.5 is applied to sigmoid probabilities (or equivalently, logits >= 0).

---

## Utilities

- [code_template.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/code_template.py)
  - Minimal, single‑backbone training script: dataset, transforms, training loop, and evaluation.
  - Useful as a starting point or for quick experiments.

- [generate_predictions.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/generate_predictions.py)
  - Loads a saved checkpoint and writes a submission CSV for an unlabeled onsite test set.
  - Example run (edit paths inside the script or adapt to your own):
    ```bash
    python generate_predictions.py
    ```

- [debug_task1.py](https://github.com/Mursal-Akhtar/Deep-Learning_Multi-retinal-disease-detection/blob/37f0fad096c643cc24538fb1298098d132d5e4e8/debug_task1.py)
  - Checks CSVs, pretrained checkpoints, and image directories exist and look sane.

---

## Checkpoints and Pretrained Weights

Place pretrained backbones here:
```
pretrained_backbone/
  ckpt_resnet18_ep50.pt
  ckpt_efficientnet_ep50.pt
```

Training scripts save best models to:
```
checkpoints/
  task1_2_resnet18.pt
  task1_3_resnet18.pt
  task2_focal_efficientnet.pt
  task3_se_resnet18.pt
  task3_mha_resnet18.pt
  ...
```

Submissions (when generated) are written to:
```
submissions/
  task2_focal_resnet18.csv
  resnet18_task1_submission.csv
  ...
```

---

## Results and Metrics

All evaluation functions compute, per disease and averaged:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1‑score (macro)
- Cohen’s kappa

Where to find results:
- The console prints per‑class and average metrics during/after training.
- JSON summaries:
  - task1_results.json
  - task2_results.json
  - task3_results.json
  - task4_final_results.json

Note: Results depend on your data, preprocessing, and training environment (hardware, seeds, etc.).

---

## Tips and Troubleshooting

- Mismatched state_dict keys:
  - The scripts try to load either full models or backbone‑only weights when needed. If you see key mismatches, ensure your checkpoint matches the model definition used.
- GPU memory:
  - MHA attention can be heavier; the code automatically clamps batch size for MHA. Reduce `batch_size` if you encounter OOM.
- Data paths:
  - Double‑check CSV `id` values correspond to files in images/* directories.
- Data normalization:
  - Transforms use ImageNet means/stds: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
- Workers:
  - num_workers is set to 0 in the scripts for portability; you can increase it for performance on Linux.

---

## Reproducibility

- Random seeds are not globally fixed across all scripts. If you need strict reproducibility:
  - Set seeds for Python, NumPy, and PyTorch.
  - Disable nondeterministic cuDNN features or set `torch.backends.cudnn.deterministic=True`.
  - Control data shuffling and any augmentations.

Example snippet:
```python
import random, numpy as np, torch
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Acknowledgments

- Backbones from torchvision (ResNet18, EfficientNet‑B0).
- SE Block: Hu et al., “Squeeze‑and‑Excitation Networks,” CVPR 2018.
- Multi‑Head Attention: Vaswani et al., “Attention Is All You Need,” NeurIPS 2017.
- Class‑Balanced Loss: Cui et al., “Class‑Balanced Loss Based on Effective Number of Samples,” CVPR 2019.
- Focal Loss: Lin et al., “Focal Loss for Dense Object Detection,” ICCV 2017.

---

If you want, I can open a pull request to add this README.md to the repository.
