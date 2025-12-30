"""
Task 4 Variant C: SE + MHA Ensemble + Finest Thresholds
Uses Task 3 SE + Task 3 MHA
With finest threshold grid: 0.20-0.80, step 0.01
"""
import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score
from PIL import Image

from task3 import ResNet18WithAttention


class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file: str, image_dir: str, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels


def infer_logits(model: nn.Module, loader: DataLoader, device, use_tta: bool = True) -> np.ndarray:
    model.eval()
    all_logits = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            if use_tta:
                logits_orig = model(imgs)
                imgs_flip = torch.flip(imgs, dims=[3])
                logits_flip = model(imgs_flip)
                logits = (logits_orig + logits_flip) / 2.0
            else:
                logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def infer_logits_ensemble(models, loader, device, use_tta: bool = True) -> np.ndarray:
    for m in models:
        m.eval()
    all_logits = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits_sum = 0
            for m in models:
                if use_tta:
                    l1 = m(imgs)
                    l2 = m(torch.flip(imgs, dims=[3]))
                    logits_sum += (l1 + l2) / 2.0
                else:
                    logits_sum += m(imgs)
            logits_sum /= len(models)
            all_logits.append(logits_sum.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def find_best_thresholds(logits: np.ndarray, labels: np.ndarray, grid=None) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.20, 0.81, 0.01)
    num_classes = logits.shape[1]
    best_thresholds = np.zeros(num_classes)
    for c in range(num_classes):
        best_f1, best_t = -1, 0.5
        y_true = labels[:, c]
        for t in grid:
            preds = (logits[:, c] > t).astype(int)
            f1 = f1_score(y_true, preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thresholds[c] = best_t
    return best_thresholds


def evaluate_with_thresholds(logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
    preds = (logits > thresholds[None, :]).astype(int)
    f1s = []
    for c in range(logits.shape[1]):
        f1s.append(f1_score(labels[:, c], preds[:, c], average="binary", zero_division=0))
    return float(np.mean(f1s))


def load_resnet(attention_type: str, ckpt_path: str, device) -> nn.Module:
    model = ResNet18WithAttention(attention_type=attention_type, num_classes=3, pretrained_weights=None)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_task4_combined():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Paths
    val_csv = "val.csv"
    test_csv = "offsite_test.csv"
    val_dir = "./images/val"
    test_dir = "./images/offsite_test"

    # Checkpoints
    ckpt_task3_se = "checkpoints/task3_se_resnet18.pt"
    ckpt_task3_mha = "checkpoints/task3_mha_resnet18.pt"

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Loaders
    val_ds = RetinaMultiLabelDataset(val_csv, val_dir, transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_dir, transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load models
    print("Loading Task 3 SE ResNet18...")
    model_se = load_resnet("se", ckpt_task3_se, device)
    print("Loading Task 3 MHA ResNet18...")
    model_mha = load_resnet("mha", ckpt_task3_mha, device)

    # Load labels
    val_labels = pd.read_csv(val_csv).iloc[:, 1:].values

    # Inference with TTA
    print("\nInferring validation set...")
    val_logits_se = infer_logits(model_se, val_loader, device, use_tta=True)
    val_logits_mha = infer_logits(model_mha, val_loader, device, use_tta=True)
    val_logits_ensemble = (val_logits_se + val_logits_mha) / 2.0

    print("Inferring test set...")
    test_logits_se = infer_logits(model_se, test_loader, device, use_tta=True)
    test_logits_mha = infer_logits(model_mha, test_loader, device, use_tta=True)
    test_logits_ensemble = (test_logits_se + test_logits_mha) / 2.0

    # Threshold search (finest grid: 0.20-0.80, step 0.01)
    print("\nSearching thresholds on validation (grid: 0.20-0.80, step 0.01)...")
    grid = np.arange(0.20, 0.81, 0.01)
    thresholds = find_best_thresholds(val_logits_ensemble, val_labels, grid)

    # Evaluate
    val_f1 = evaluate_with_thresholds(val_logits_ensemble, val_labels, thresholds)
    test_f1 = evaluate_with_thresholds(test_logits_ensemble, val_labels, thresholds)

    print(f"\n{'='*70}")
    print(f"TASK 4 VARIANT C: SE + MHA Ensemble + Finest Thresholds")
    print(f"{'='*70}")
    print(f"Models: Task3 SE + Task3 MHA")
    print(f"Threshold Grid: 0.20-0.80, step 0.01 (finest)")
    print(f"TTA: Yes (orig + hflip)")
    print(f"\nVal F1:  {val_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"Thresholds: {[round(x, 2) for x in thresholds]}")
    print(f"vs Reference (78.8%): {test_f1:.4f} vs 0.788")
    if test_f1 > 0.803:
        print(f"✓ EXCEEDS >1.5% threshold!")
    elif test_f1 > 0.788:
        print(f"✓ Better than reference")
    else:
        print(f"→ Below reference")
    print(f"{'='*70}\n")

    results = {
        "variant": "C: SE + MHA Ensemble + Finest Thresholds",
        "threshold_grid": "0.20-0.80, step 0.01",
        "tta": True,
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "thresholds": thresholds.tolist(),
    }
    with open("task4_variant_c_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to task4_variant_c_results.json\n")


if __name__ == "__main__":
    run_task4_combined()
