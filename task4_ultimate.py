"""
Task 4 ULTIMATE - Every optimization trick to maximize onsite test performance
Tries: TTA, threshold tuning, multiple ensemble strategies, all models
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from code_template import build_model
from task3 import ResNet18WithAttention


# ========================
# Dataset
# ========================
class RetinaDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, has_labels=True):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.has_labels = has_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row.iloc[0])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        if self.has_labels:
            labels = torch.tensor(row[1:].values.astype("float32"))
            return img, labels, row.iloc[0]
        else:
            return img, row.iloc[0]


# ========================
# TTA (Test-Time Augmentation)
# ========================
def tta_inference(model, img, device):
    """Apply TTA with multiple augmentations"""
    model.eval()
    
    augmentations = [
        lambda x: x,  # Original
        lambda x: TF.hflip(x),  # Horizontal flip
        lambda x: TF.vflip(x),  # Vertical flip
        lambda x: TF.rotate(x, 90),  # Rotate 90
        lambda x: TF.rotate(x, -90),  # Rotate -90
    ]
    
    predictions = []
    with torch.no_grad():
        for aug in augmentations:
            aug_img = aug(img).unsqueeze(0).to(device)
            output = torch.sigmoid(model(aug_img))
            predictions.append(output.cpu().numpy())
    
    # Average predictions
    return np.mean(predictions, axis=0)


def batch_tta_inference(model, loader, device):
    """TTA inference on entire dataset"""
    all_preds = []
    all_filenames = []
    
    print("Running TTA inference (5 augmentations per image)...")
    for batch in loader:
        if len(batch) == 2:
            imgs, filenames = batch
        else:
            imgs, _, filenames = batch
        
        for i, img in enumerate(imgs):
            pred = tta_inference(model, img, device)
            all_preds.append(pred[0])
            all_filenames.append(filenames[i])
    
    return np.array(all_preds), all_filenames


def batch_simple_inference(model, loader, device, has_labels=True):
    """Simple inference without TTA"""
    all_preds = []
    all_labels = [] if has_labels else None
    all_filenames = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if has_labels:
                imgs, labels, filenames = batch
                all_labels.append(labels.numpy())
            else:
                imgs, filenames = batch
            
            imgs = imgs.to(device)
            outputs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.extend(outputs)
            all_filenames.extend(filenames)
    
    if has_labels:
        return np.array(all_preds), np.array(all_labels), all_filenames
    else:
        return np.array(all_preds), all_filenames


# ========================
# Threshold optimization
# ========================
def optimize_thresholds(preds, labels, num_classes=3):
    """Find optimal threshold per class on validation set"""
    thresholds = []
    
    for i in range(num_classes):
        best_thresh = 0.5
        best_f1 = 0
        
        for thresh in np.arange(0.3, 0.7, 0.05):
            pred_binary = (preds[:, i] > thresh).astype(int)
            f1 = f1_score(labels[:, i], pred_binary, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        thresholds.append(best_thresh)
        print(f"  Class {i}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")
    
    return np.array(thresholds)


# ========================
# Main optimization pipeline
# ========================
def run_ultimate_optimization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("TASK 4 ULTIMATE OPTIMIZATION")
    print("="*70)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    val_ds = RetinaDataset("val.csv", "./images/val", transform, has_labels=True)
    onsite_ds = RetinaDataset("onsite_test_submission.csv", "./images/onsite_test", transform, has_labels=False)
    
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    onsite_loader = DataLoader(onsite_ds, batch_size=1, shuffle=False, num_workers=0)  # Batch=1 for TTA
    
    # Load all models
    print("\nLoading models...")
    se_model = ResNet18WithAttention("se", num_classes=3, pretrained_weights=None)
    se_model.load_state_dict(torch.load("checkpoints/task3_se_resnet18.pt", map_location=device))
    se_model.to(device).eval()
    
    mha_model = ResNet18WithAttention("mha", num_classes=3, pretrained_weights=None)
    mha_model.load_state_dict(torch.load("checkpoints/task3_mha_resnet18.pt", map_location=device))
    mha_model.to(device).eval()
    
    task2_model = build_model("resnet18", num_classes=3, pretrained=False)
    task2_model.load_state_dict(torch.load("checkpoints/task1_2_resnet18.pt", map_location=device))
    task2_model.to(device).eval()
    
    print("✓ All models loaded")
    
    # ========================
    # Strategy 1: SE with TTA + Threshold Optimization
    # ========================
    print("\n" + "="*70)
    print("STRATEGY 1: SE + TTA + Threshold Optimization")
    print("="*70)
    
    # Get validation predictions
    val_se_preds, val_se_labels, _ = batch_simple_inference(se_model, val_loader, device, has_labels=True)
    
    # Optimize thresholds
    print("\nOptimizing thresholds on validation set...")
    thresholds = optimize_thresholds(val_se_preds, val_se_labels)
    
    # TTA on onsite test
    onsite_se_tta, filenames = batch_tta_inference(se_model, onsite_loader, device)
    
    # Apply optimized thresholds
    strategy1_preds = (onsite_se_tta > thresholds).astype(int)
    
    save_submission(filenames, strategy1_preds, "strategy1_se_tta_thresh.csv")
    
    # ========================
    # Strategy 2: Weighted Ensemble (SE + MHA + Task2) with TTA
    # ========================
    print("\n" + "="*70)
    print("STRATEGY 2: Weighted Ensemble (All 3 Models) + TTA")
    print("="*70)
    
    # TTA for all models on onsite
    print("\nRunning TTA for MHA...")
    onsite_mha_tta, _ = batch_tta_inference(mha_model, onsite_loader, device)
    
    print("Running TTA for Task2...")
    onsite_task2_tta, _ = batch_tta_inference(task2_model, onsite_loader, device)
    
    # Weighted average (weights based on offsite performance)
    # SE: 82.05%, MHA: 79.29%, Task2: 80.23%
    # Normalized weights: SE=0.42, MHA=0.31, Task2=0.27
    weights = np.array([0.50, 0.20, 0.30])  # Favor SE more
    
    weighted_preds = (weights[0] * onsite_se_tta + 
                     weights[1] * onsite_mha_tta + 
                     weights[2] * onsite_task2_tta)
    
    strategy2_preds = (weighted_preds > 0.5).astype(int)
    save_submission(filenames, strategy2_preds, "strategy2_weighted_ensemble_tta.csv")
    
    # ========================
    # Strategy 3: Ensemble + Threshold Optimization
    # ========================
    print("\n" + "="*70)
    print("STRATEGY 3: Weighted Ensemble + Threshold Optimization")
    print("="*70)
    
    # Get validation predictions for all models
    val_mha_preds, _, _ = batch_simple_inference(mha_model, val_loader, device, has_labels=True)
    val_task2_preds, _, _ = batch_simple_inference(task2_model, val_loader, device, has_labels=True)
    
    # Weighted ensemble on validation
    val_weighted = (weights[0] * val_se_preds + 
                   weights[1] * val_mha_preds + 
                   weights[2] * val_task2_preds)
    
    # Optimize thresholds
    print("\nOptimizing thresholds for ensemble...")
    ensemble_thresholds = optimize_thresholds(val_weighted, val_se_labels)
    
    # Apply to test predictions
    strategy3_preds = (weighted_preds > ensemble_thresholds).astype(int)
    save_submission(filenames, strategy3_preds, "strategy3_ensemble_tta_thresh.csv")
    
    # ========================
    # Strategy 4: Best single model (SE) with aggressive TTA
    # ========================
    print("\n" + "="*70)
    print("STRATEGY 4: SE Model + Maximum TTA (10 augmentations)")
    print("="*70)
    
    # More aggressive TTA
    aggressive_tta_preds = []
    for batch in onsite_loader:
        if len(batch) == 2:
            imgs, _ = batch
        else:
            imgs, _, _ = batch
        
        img = imgs[0]
        
        # 10 augmentations
        augs = [
            lambda x: x,
            lambda x: TF.hflip(x),
            lambda x: TF.vflip(x),
            lambda x: TF.rotate(x, 90),
            lambda x: TF.rotate(x, -90),
            lambda x: TF.rotate(x, 180),
            lambda x: TF.adjust_brightness(x, 1.1),
            lambda x: TF.adjust_brightness(x, 0.9),
            lambda x: TF.adjust_contrast(x, 1.1),
            lambda x: TF.adjust_contrast(x, 0.9),
        ]
        
        preds = []
        with torch.no_grad():
            for aug in augs:
                aug_img = aug(img).unsqueeze(0).to(device)
                output = torch.sigmoid(se_model(aug_img))
                preds.append(output.cpu().numpy())
        
        aggressive_tta_preds.append(np.mean(preds, axis=0)[0])
    
    aggressive_tta_preds = np.array(aggressive_tta_preds)
    strategy4_preds = (aggressive_tta_preds > thresholds).astype(int)
    save_submission(filenames, strategy4_preds, "strategy4_se_aggressive_tta_thresh.csv")
    
    # Summary
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE - 4 STRATEGIES GENERATED")
    print("="*70)
    print("\nGenerated submissions:")
    print("1. strategy1_se_tta_thresh.csv - SE + TTA + Optimized Thresholds")
    print("2. strategy2_weighted_ensemble_tta.csv - 3-Model Weighted Ensemble + TTA")
    print("3. strategy3_ensemble_tta_thresh.csv - Ensemble + TTA + Optimized Thresholds")
    print("4. strategy4_se_aggressive_tta_thresh.csv - SE + 10x TTA + Thresholds")
    print("\nRECOMMENDATION: Try Strategy 3 first (most comprehensive)")
    print("="*70)


def save_submission(filenames, predictions, output_file):
    """Save predictions to CSV"""
    df = pd.DataFrame({
        'id': filenames,
        'D': predictions[:, 0],
        'G': predictions[:, 1],
        'A': predictions[:, 2]
    })
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")
    print(f"  DR: {df['D'].sum()}, Glaucoma: {df['G'].sum()}, AMD: {df['A'].sum()}")


if __name__ == "__main__":
    run_ultimate_optimization()
    
    print("\n\nDOWNLOAD ALL 4 FILES:")
    print("from google.colab import files")
    print("files.download('strategy3_ensemble_tta_thresh.csv')  # BEST BET")
    print("files.download('strategy1_se_tta_thresh.csv')")
    print("files.download('strategy2_weighted_ensemble_tta.csv')")
    print("files.download('strategy4_se_aggressive_tta_thresh.csv')")
