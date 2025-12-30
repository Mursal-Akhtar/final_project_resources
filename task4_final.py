"""
Task 4: Final Optimizations
Option 1: Aggressive SE weighting (0.8/0.2, 0.9/0.1)
Option 4: Hard voting ensemble
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

from code_template import RetinaMultiLabelDataset
from task3 import ResNet18WithAttention


# ========================
# Inference with TTA
# ========================
def infer_logits(model, loader, device, use_tta=True):
    """Inference with horizontal flip TTA"""
    logits_list = []
    labels_list = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            # Original
            out = model(images).cpu().numpy()
            
            if use_tta:
                # Horizontal flip
                images_flipped = torch.flip(images, dims=[-1])
                out_flipped = model(images_flipped).cpu().numpy()
                out = (out + out_flipped) / 2.0
            
            logits_list.append(out)
            labels_list.append(labels)
    
    logits = np.vstack(logits_list)
    labels = np.vstack(labels_list)
    return logits, labels


def infer_logits_ensemble(models, loader, device, weights=None, use_tta=True):
    """Weighted ensemble inference"""
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


def infer_hard_voting(models, loader, device, use_tta=True):
    """Hard voting: each model votes 0/1, take majority"""
    all_votes = []
    labels = None
    
    for model in models:
        logits, labels = infer_logits(model, loader, device, use_tta=use_tta)
        # Convert to hard predictions with 0.5 threshold
        votes = (logits >= 0.0).astype(int)  # logits >= 0 means prob >= 0.5
        all_votes.append(votes)
    
    # Stack and take majority vote
    votes_stacked = np.stack(all_votes, axis=0)  # (num_models, N, 3)
    majority_votes = (votes_stacked.sum(axis=0) > len(models) / 2).astype(int)
    
    return majority_votes, labels


# ========================
# Evaluation
# ========================
def evaluate_predictions(preds, labels):
    """Evaluate F1 from hard predictions"""
    return f1_score(labels, preds, average='macro', zero_division=0)


def evaluate_logits(logits, labels, threshold=0.5):
    """Evaluate F1 from logits with threshold"""
    preds = (logits >= 0.0).astype(int) if threshold == 0.5 else (torch.sigmoid(torch.tensor(logits)).numpy() >= threshold).astype(int)
    return f1_score(labels, preds, average='macro', zero_division=0)


# ========================
# Model loading
# ========================
def load_resnet(attention_type, ckpt_path, device):
    model = ResNet18WithAttention(attention_type=attention_type, num_classes=3, pretrained_weights=None)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ========================
# Main
# ========================
def run_task4_final():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Paths
    val_csv = "val.csv"
    test_csv = "offsite_test.csv"
    val_dir = "./images/val"
    test_dir = "./images/offsite_test"

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
    # Option 1A: Aggressive SE weighting (0.8/0.2)
    # ==================
    print("\n[Option 1A] Ensemble SE=0.8, MHA=0.2 (Aggressive SE)")
    val_logits, val_labels = infer_logits_ensemble([model_se, model_mha], val_loader, device, weights=[0.8, 0.2], use_tta=True)
    val_f1 = evaluate_logits(val_logits, val_labels, threshold=0.5)
    
    test_logits, test_labels = infer_logits_ensemble([model_se, model_mha], test_loader, device, weights=[0.8, 0.2], use_tta=True)
    test_f1 = evaluate_logits(test_logits, test_labels, threshold=0.5)
    
    results["ensemble_0.8_0.2"] = {
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "weights": [0.8, 0.2],
    }
    print(f"  Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    # ==================
    # Option 1B: Very aggressive SE weighting (0.9/0.1)
    # ==================
    print("\n[Option 1B] Ensemble SE=0.9, MHA=0.1 (Very Aggressive SE)")
    val_logits, val_labels = infer_logits_ensemble([model_se, model_mha], val_loader, device, weights=[0.9, 0.1], use_tta=True)
    val_f1 = evaluate_logits(val_logits, val_labels, threshold=0.5)
    
    test_logits, test_labels = infer_logits_ensemble([model_se, model_mha], test_loader, device, weights=[0.9, 0.1], use_tta=True)
    test_f1 = evaluate_logits(test_logits, test_labels, threshold=0.5)
    
    results["ensemble_0.9_0.1"] = {
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "weights": [0.9, 0.1],
    }
    print(f"  Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    # ==================
    # Option 1C: Pure SE (1.0/0.0) - baseline check
    # ==================
    print("\n[Option 1C] Pure SE (1.0/0.0) - Baseline Check")
    val_logits, val_labels = infer_logits(model_se, val_loader, device, use_tta=True)
    val_f1 = evaluate_logits(val_logits, val_labels, threshold=0.5)
    
    test_logits, test_labels = infer_logits(model_se, test_loader, device, use_tta=True)
    test_f1 = evaluate_logits(test_logits, test_labels, threshold=0.5)
    
    results["pure_se"] = {
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "weights": [1.0, 0.0],
    }
    print(f"  Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    # ==================
    # Option 4: Hard Voting Ensemble
    # ==================
    print("\n[Option 4] Hard Voting Ensemble (Majority Vote)")
    val_preds, val_labels = infer_hard_voting([model_se, model_mha], val_loader, device, use_tta=True)
    val_f1 = evaluate_predictions(val_preds, val_labels)
    
    test_preds, test_labels = infer_hard_voting([model_se, model_mha], test_loader, device, use_tta=True)
    test_f1 = evaluate_predictions(test_preds, test_labels)
    
    results["hard_voting"] = {
        "val_f1": float(val_f1),
        "test_f1": float(test_f1),
        "method": "majority_vote",
    }
    print(f"  Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    # ==================
    # Summary
    # ==================
    print("\n" + "="*70)
    print("TASK 4 FINAL - Summary (Offsite Test F1)")
    print("="*70)
    for k, v in results.items():
        print(f"{k:25s} | val_f1={v['val_f1']:.4f} | test_f1={v['test_f1']:.4f}")

    # Best candidate
    best_key = max(results, key=lambda k: results[k]['test_f1'])
    best_f1 = results[best_key]['test_f1']
    print(f"\nBest Candidate: {best_key} with Test F1 = {best_f1:.4f}")
    print(f"Task 3 SE Baseline: 0.8355 (for reference)")

    # Save results
    with open("task4_final_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved task4_final_results.json")


if __name__ == "__main__":
    run_task4_final()
