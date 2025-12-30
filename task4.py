import os
import json
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import f1_score

from task3 import ResNet18WithAttention


# ========================
# Dataset
# ========================
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
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels


# ========================
# Utility: TTA inference
# ========================
def infer_logits(model: nn.Module, loader: DataLoader, device, use_tta: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.numpy()

            if use_tta:
                # original
                logits_orig = model(imgs)
                # horizontal flip
                imgs_flip = torch.flip(imgs, dims=[3])
                logits_flip = model(imgs_flip)
                logits = (logits_orig + logits_flip) / 2.0
            else:
                logits = model(imgs)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels)

    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


def infer_logits_ensemble(models: List[nn.Module], loader: DataLoader, device, use_tta: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    for m in models:
        m.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.numpy()
            logits_sum = 0
            for m in models:
                if use_tta:
                    l1 = m(imgs)
                    l2 = m(torch.flip(imgs, dims=[3]))
                    logits_sum += (l1 + l2) / 2.0
                else:
                    logits_sum += m(imgs)
            logits = logits_sum / len(models)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels)
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


# ========================
# Threshold search per class
# ========================
def find_best_thresholds(logits: np.ndarray, labels: np.ndarray, grid=None) -> np.ndarray:
    if grid is None:
        grid = np.arange(0.30, 0.71, 0.05)
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


# ========================
# Model loading
# ========================
def load_resnet(attention_type: str, ckpt_path: str, device) -> nn.Module:
    model = ResNet18WithAttention(attention_type=attention_type, num_classes=3, pretrained_weights=None)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ========================
# Main evaluation flow
# ========================
def run_task4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Paths
    train_csv = "train.csv"
    val_csv = "val.csv"
    test_csv = "offsite_test.csv"
    train_dir = "./images/train"
    val_dir = "./images/val"
    test_dir = "./images/offsite_test"

    ckpt_se = "checkpoints/task3_resnet18_se.pt"
    ckpt_mha = "checkpoints/task3_resnet18_mha.pt"

    # Transforms
    base_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data loaders
    val_ds = RetinaMultiLabelDataset(val_csv, val_dir, base_transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_dir, base_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load models
    model_se = load_resnet("se", ckpt_se, device)
    model_mha = load_resnet("mha", ckpt_mha, device)

    results = {}

    # Candidate A: SE only
    val_logits, val_labels = infer_logits(model_se, val_loader, device, use_tta=True)
    th_se = find_best_thresholds(val_logits, val_labels)
    val_f1_se = evaluate_with_thresholds(val_logits, val_labels, th_se)
    test_logits, test_labels = infer_logits(model_se, test_loader, device, use_tta=True)
    test_f1_se = evaluate_with_thresholds(test_logits, test_labels, th_se)
    results["resnet18_se"] = {
        "val_f1": val_f1_se,
        "test_f1": test_f1_se,
        "thresholds": th_se.tolist(),
    }

    # Candidate B: MHA only
    val_logits, val_labels = infer_logits(model_mha, val_loader, device, use_tta=True)
    th_mha = find_best_thresholds(val_logits, val_labels)
    val_f1_mha = evaluate_with_thresholds(val_logits, val_labels, th_mha)
    test_logits, test_labels = infer_logits(model_mha, test_loader, device, use_tta=True)
    test_f1_mha = evaluate_with_thresholds(test_logits, test_labels, th_mha)
    results["resnet18_mha"] = {
        "val_f1": val_f1_mha,
        "test_f1": test_f1_mha,
        "thresholds": th_mha.tolist(),
    }

    # Candidate C: Ensemble (SE + MHA)
    val_logits, val_labels = infer_logits_ensemble([model_se, model_mha], val_loader, device, use_tta=True)
    th_ens = find_best_thresholds(val_logits, val_labels)
    val_f1_ens = evaluate_with_thresholds(val_logits, val_labels, th_ens)
    test_logits, test_labels = infer_logits_ensemble([model_se, model_mha], test_loader, device, use_tta=True)
    test_f1_ens = evaluate_with_thresholds(test_logits, test_labels, th_ens)
    results["resnet18_ensemble"] = {
        "val_f1": val_f1_ens,
        "test_f1": test_f1_ens,
        "thresholds": th_ens.tolist(),
    }

    # Print summary
    print("\nTASK 4 - Threshold + TTA + Ensemble (Offsite F1)")
    for k, v in results.items():
        print(f"{k:20s} | val_f1={v['val_f1']:.4f} | test_f1={v['test_f1']:.4f}")
        print(f"  thresholds: {[round(x,2) for x in v['thresholds']]}")

    # Save
    with open("task4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved task4_results.json")


if __name__ == "__main__":
    run_task4()
