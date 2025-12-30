import os
import pandas as pd
import numpy as np
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


# ========================
# Dataset preparation
# ========================
class RetinaMultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
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


# ========================
# Build model
# ========================
def build_model(backbone="resnet18", num_classes=3, pretrained=True):
    if backbone == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported backbone")
    return model


# ========================
# Evaluation function
# ========================
def evaluate_model(model, data_loader, device, backbone="resnet18"):
    """Evaluate model and compute metrics"""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    disease_names = ["DR", "Glaucoma", "AMD"]
    results = {}

    print(f"\n{'='*60}")
    print(f"Results for {backbone.upper()}")
    print(f"{'='*60}")

    disease_f1_scores = []

    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        kappa = cohen_kappa_score(y_t, y_p)

        disease_f1_scores.append(f1)

        print(f"\n{disease} Metrics:")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1-score : {f1:.4f}")
        print(f"  Kappa    : {kappa:.4f}")

        results[disease] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "kappa": float(kappa)
        }

    avg_f1 = np.mean(disease_f1_scores)
    print(f"\n{'='*60}")
    print(f"Average F1-score: {avg_f1:.4f}")
    print(f"{'='*60}\n")

    results["average_f1"] = float(avg_f1)
    return results


# ========================
# Task 1-1: Baseline (No Fine-tuning)
# ========================
def task1_baseline(backbone, train_csv, val_csv, test_csv, train_image_dir, 
                   val_image_dir, test_image_dir, pretrained_backbone=None, 
                   batch_size=32, img_size=256, device=None):
    """
    Task 1-1: Test pretrained models WITHOUT fine-tuning
    Establishes baseline F1-scores
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*60}")
    print(f"# TASK 1-1: BASELINE (No Fine-tuning)")
    print(f"# Backbone: {backbone.upper()}")
    print(f"{'#'*60}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & DataLoaders
    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = build_model(backbone, num_classes=3, pretrained=True).to(device)

    # Load pretrained backbone if provided
    if pretrained_backbone is not None:
        print(f"Loading pretrained backbone from: {pretrained_backbone}")
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        # Only load backbone weights, not classifier
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc' not in k and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Freeze all parameters (no training)
    for param in model.parameters():
        param.requires_grad = False

    # Evaluate on test set
    print(f"\nEvaluating {backbone} baseline on test set...")
    results = evaluate_model(model, test_loader, device, backbone)

    return results


# ========================
# Task 1-2: Fine-tune Classifier Head Only
# ========================
def task1_finetune_head(backbone, train_csv, val_csv, test_csv, train_image_dir, 
                        val_image_dir, test_image_dir, pretrained_backbone=None,
                        epochs=20, batch_size=32, lr=1e-3, img_size=256, 
                        save_dir="checkpoints", device=None):
    """
    Task 1-2: Fine-tune only the classifier head
    Keep backbone frozen, train only FC/classifier layers
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*60}")
    print(f"# TASK 1-2: FINE-TUNE CLASSIFIER HEAD ONLY")
    print(f"# Backbone: {backbone.upper()}")
    print(f"{'#'*60}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & DataLoaders
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = build_model(backbone, num_classes=3, pretrained=True).to(device)

    # Load pretrained backbone if provided
    if pretrained_backbone is not None:
        print(f"Loading pretrained backbone from: {pretrained_backbone}")
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc' not in k and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Freeze backbone, unfreeze classifier
    if backbone == "resnet18":
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
    elif backbone == "efficientnet":
        for param in model.features.parameters():
            param.requires_grad = False

    # Loss & Optimizer (only train classifier parameters)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"task1_2_{backbone}.pt")

    print(f"\nTraining classifier head for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # Evaluate on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nEvaluating {backbone} (fine-tuned head) on test set...")
    results = evaluate_model(model, test_loader, device, backbone)

    return results, ckpt_path


# ========================
# Task 1-3: Fine-tune Full Model
# ========================
def task1_finetune_full(backbone, train_csv, val_csv, test_csv, train_image_dir, 
                        val_image_dir, test_image_dir, pretrained_backbone=None,
                        epochs=20, batch_size=32, lr=1e-5, img_size=256, 
                        save_dir="checkpoints", device=None):
    """
    Task 1-3: Fine-tune the entire model
    Train all layers with a low learning rate
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*60}")
    print(f"# TASK 1-3: FINE-TUNE FULL MODEL")
    print(f"# Backbone: {backbone.upper()}")
    print(f"{'#'*60}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & DataLoaders
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = build_model(backbone, num_classes=3, pretrained=True).to(device)

    # Load pretrained backbone if provided
    if pretrained_backbone is not None:
        print(f"Loading pretrained backbone from: {pretrained_backbone}")
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'fc' not in k and 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Loss & Optimizer (low learning rate for full model training)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"task1_3_{backbone}.pt")

    print(f"\nTraining full model for {epochs} epochs (lr={lr})...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # Evaluate on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nEvaluating {backbone} (fine-tuned full) on test set...")
    results = evaluate_model(model, test_loader, device, backbone)

    return results, ckpt_path


# ========================
# Main
# ========================
if __name__ == "__main__":
    # Data paths
    train_csv = "train.csv"
    val_csv = "val.csv"
    test_csv = "offsite_test.csv"
    train_image_dir = "./images/train"
    val_image_dir = "./images/val"
    test_image_dir = "./images/offsite_test"

    # Pretrained backbones
    pretrained_resnet18 = './pretrained_backbone/ckpt_resnet18_ep50.pt'
    pretrained_efficientnet = './pretrained_backbone/ckpt_efficientnet_ep50.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Store all results
    all_results = {}

    # ========================
    # ResNet18 Results
    # ========================
    print(f"\n\n{'='*70}")
    print(f"{'='*70}")
    print(f"RESNET18 EXPERIMENTS")
    print(f"{'='*70}")
    print(f"{'='*70}")

    # Task 1-1: Baseline ResNet18
    results_resnet_baseline = task1_baseline(
        backbone="resnet18",
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        pretrained_backbone=pretrained_resnet18,
        device=device
    )
    all_results["resnet18_baseline"] = results_resnet_baseline

    # Task 1-2: Fine-tune Head ResNet18
    results_resnet_head, _ = task1_finetune_head(
        backbone="resnet18",
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        pretrained_backbone=pretrained_resnet18,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        device=device
    )
    all_results["resnet18_finetune_head"] = results_resnet_head

    # Task 1-3: Fine-tune Full ResNet18
    results_resnet_full, _ = task1_finetune_full(
        backbone="resnet18",
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        pretrained_backbone=pretrained_resnet18,
        epochs=20,
        batch_size=32,
        lr=1e-5,
        device=device
    )
    all_results["resnet18_finetune_full"] = results_resnet_full

    # ========================
    # EfficientNet Results
    # ========================
    print(f"\n\n{'='*70}")
    print(f"{'='*70}")
    print(f"EFFICIENTNET EXPERIMENTS")
    print(f"{'='*70}")
    print(f"{'='*70}")

    # Task 1-1: Baseline EfficientNet
    results_eff_baseline = task1_baseline(
        backbone="efficientnet",
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        pretrained_backbone=pretrained_efficientnet,
        device=device
    )
    all_results["efficientnet_baseline"] = results_eff_baseline

    # Task 1-2: Fine-tune Head EfficientNet
    results_eff_head, _ = task1_finetune_head(
        backbone="efficientnet",
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        pretrained_backbone=pretrained_efficientnet,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        device=device
    )
    all_results["efficientnet_finetune_head"] = results_eff_head

    # Task 1-3: Fine-tune Full EfficientNet
    results_eff_full, _ = task1_finetune_full(
        backbone="efficientnet",
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        test_image_dir=test_image_dir,
        pretrained_backbone=pretrained_efficientnet,
        epochs=20,
        batch_size=32,
        lr=1e-5,
        device=device
    )
    all_results["efficientnet_finetune_full"] = results_eff_full

    # ========================
    # Summary
    # ========================
    print(f"\n\n{'='*70}")
    print(f"TASK 1 SUMMARY - Average F1-Scores")
    print(f"{'='*70}")
    print(f"ResNet18 Baseline:           {all_results['resnet18_baseline']['average_f1']:.4f}")
    print(f"ResNet18 Fine-tune Head:     {all_results['resnet18_finetune_head']['average_f1']:.4f}")
    print(f"ResNet18 Fine-tune Full:     {all_results['resnet18_finetune_full']['average_f1']:.4f}")
    print(f"\nEfficientNet Baseline:       {all_results['efficientnet_baseline']['average_f1']:.4f}")
    print(f"EfficientNet Fine-tune Head: {all_results['efficientnet_finetune_head']['average_f1']:.4f}")
    print(f"EfficientNet Fine-tune Full: {all_results['efficientnet_finetune_full']['average_f1']:.4f}")
    print(f"{'='*70}")

    # Save results to JSON
    with open("task1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to task1_results.json")
