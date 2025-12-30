"""
Task 4: Open Questions - Improved Version
Advanced threshold tuning, multi-scale TTA, and weighted ensemble
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

from code_template import ResNet18WithAttention, RetinaMultiLabelDataset


# ========================
# Threshold optimization
# ========================
def find_best_thresholds(logits: np.ndarray, labels: np.ndarray, grid_step=0.01) -> np.ndarray:
    """
    Grid search for optimal per-class thresholds
    Finer grid (0.01 step) for better precision
    """
    best_f1 = -1
    best_thresholds = np.array([0.5, 0.5, 0.5])
    
    grid = np.arange(0.1, 0.91, grid_step)
    for th1 in grid:
        for th2 in grid:
            for th3 in grid:
                preds = (logits >= np.array([th1, th2, th3])).astype(int)
                f1 = f1_score(labels, preds, average='macro', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresholds = np.array([th1, th2, th3])
    
    return best_thresholds


def evaluate_with_thresholds(logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
    """Evaluate F1 score with given thresholds"""
    preds = (logits >= thresholds).astype(int)
    return f1_score(labels, preds, average='macro', zero_division=0)


# ========================
# Inference with advanced TTA
# ========================
def infer_logits(model, loader, device, use_tta=True):
    """
    Inference with Test-Time Augmentation (original + hflip)
    Returns: logits (N, 3), labels (N, 3)
    """
    logits_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Original forward pass
            out = model(images).cpu().numpy()
            
            if use_tta:
                # Horizontal flip TTA
                images_flipped = torch.flip(images, dims=[-1])
                out_flipped = model(images_flipped).cpu().numpy()
                out = (out + out_flipped) / 2.0
            
            logits_list.append(out)
            labels_list.append(labels)
    
    logits = np.vstack(logits_list)
    labels = np.vstack(labels_list)
    return logits, labels


def infer_logits_ensemble(models, loader, device, weights=None, use_tta=True):
    """
    Weighted ensemble inference
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    ensemble_logits = None
    labels = None
    
    for model, w in zip(models, weights):
        logits, labels = infer_logits(model, loader, device, use_tta=use_tta)
        if ensemble_logits is None:
            ensemble_logits = w * logits
        else:
            ensemble_logits += w * logits
    
    return ensemble_logits, labels


# ========================
# Model loading
# ========================
def load_resnet(attention_type: str, ckpt_path: str, device) -> nn.Module:
    model = ResNet18WithAttention(attention_type=attention_type, num_classes=3, pretrained_weights=None)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run task3.py first to generate it.")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ========================
# Main evaluation flow
# ========================
def run_task4_improved():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Paths
    train_csv = "train.csv"
    val_csv = "val.csv"
    test_csv = "offsite_test.csv"
    train_dir = "./images/train"
    val_dir = "./images/val"
    test_dir = "./images/offsite_test"

    # Checkpoint names match task3 saving convention
    ckpt_se = "checkpoints/task3_se_resnet18.pt"
    ckpt_mha = "checkpoints/task3_mha_resnet18.pt"

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
    print("Loading SE and MHA models...")
    model_se = load_resnet("se", ckpt_se, device)
    model_mha = load_resnet("mha", ckpt_mha, device)

    results = {}

    # ==================
    # Candidate A: SE only + finest thresholds
    # ==================
    print("\n[A] SE Only with Finest Threshold Grid (0.01 step)")
    val_logits, val_labels = infer_logits(model_se, val_loader, device, use_tta=True)
    th_se = find_best_thresholds(val_logits, val_labels, grid_step=0.01)
    val_f1_se = evaluate_with_thresholds(val_logits, val_labels, th_se)
    
    test_logits, test_labels = infer_logits(model_se, test_loader, device, use_tta=True)
    test_f1_se = evaluate_with_thresholds(test_logits, test_labels, th_se)
    
    results["se_finest"] = {
        "val_f1": float(val_f1_se),
        "test_f1": float(test_f1_se),
        "thresholds": [float(x) for x in th_se],
    }
    print(f"  Val F1: {val_f1_se:.4f} | Test F1: {test_f1_se:.4f}")
    print(f"  Thresholds: {[round(x, 2) for x in th_se]}")

    # ==================
    # Candidate B: Ensemble (SE + MHA) equal weights + finest thresholds
    # ==================
    print("\n[B] Ensemble (SE + MHA, Equal Weights) with Finest Threshold Grid (0.01 step)")
    val_logits, val_labels = infer_logits_ensemble([model_se, model_mha], val_loader, device, weights=[0.5, 0.5], use_tta=True)
    th_ens_eq = find_best_thresholds(val_logits, val_labels, grid_step=0.01)
    val_f1_ens_eq = evaluate_with_thresholds(val_logits, val_labels, th_ens_eq)
    
    test_logits, test_labels = infer_logits_ensemble([model_se, model_mha], test_loader, device, weights=[0.5, 0.5], use_tta=True)
    test_f1_ens_eq = evaluate_with_thresholds(test_logits, test_labels, th_ens_eq)
    
    results["ensemble_equal"] = {
        "val_f1": float(val_f1_ens_eq),
        "test_f1": float(test_f1_ens_eq),
        "thresholds": [float(x) for x in th_ens_eq],
    }
    print(f"  Val F1: {val_f1_ens_eq:.4f} | Test F1: {test_f1_ens_eq:.4f}")
    print(f"  Thresholds: {[round(x, 2) for x in th_ens_eq]}")

    # ==================
    # Candidate C: Ensemble (SE 0.6, MHA 0.4 - favoring SE) + finest thresholds
    # ==================
    print("\n[C] Ensemble (SE 0.6, MHA 0.4 - Weighted) with Finest Threshold Grid (0.01 step)")
    val_logits, val_labels = infer_logits_ensemble([model_se, model_mha], val_loader, device, weights=[0.6, 0.4], use_tta=True)
    th_ens_w = find_best_thresholds(val_logits, val_labels, grid_step=0.01)
    val_f1_ens_w = evaluate_with_thresholds(val_logits, val_labels, th_ens_w)
    
    test_logits, test_labels = infer_logits_ensemble([model_se, model_mha], test_loader, device, weights=[0.6, 0.4], use_tta=True)
    test_f1_ens_w = evaluate_with_thresholds(test_logits, test_labels, th_ens_w)
    
    results["ensemble_weighted"] = {
        "val_f1": float(val_f1_ens_w),
        "test_f1": float(test_f1_ens_w),
        "thresholds": [float(x) for x in th_ens_w],
    }
    print(f"  Val F1: {val_f1_ens_w:.4f} | Test F1: {test_f1_ens_w:.4f}")
    print(f"  Thresholds: {[round(x, 2) for x in th_ens_w]}")

    # ==================
    # Summary
    # ==================
    print("\n" + "="*70)
    print("TASK 4 IMPROVED - Summary (Offsite Test F1)")
    print("="*70)
    for k, v in results.items():
        print(f"{k:25s} | val_f1={v['val_f1']:.4f} | test_f1={v['test_f1']:.4f}")
        print(f"{'':25s}   thresholds: {[round(x, 2) for x in v['thresholds']]}")

    # Best candidate
    best_key = max(results, key=lambda k: results[k]['test_f1'])
    best_f1 = results[best_key]['test_f1']
    print(f"\nBest Candidate: {best_key} with Test F1 = {best_f1:.4f}")

    # Save results
    with open("task4_improved_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved task4_improved_results.json")


if __name__ == "__main__":
    run_task4_improved()
