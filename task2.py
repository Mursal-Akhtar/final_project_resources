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
# Loss Functions
# ========================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    Helps with class imbalance by focusing on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model logits [batch_size, num_classes]
            targets: Binary labels [batch_size, num_classes]
        """
        # Convert logits to probabilities
        p = torch.sigmoid(outputs)
        
        # Calculate binary cross entropy
        bce_loss = nn.BCELoss(reduction='none')(p, targets)
        
        # Calculate focal weight: (1 - p_t)^gamma
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal loss with alpha weighting
        focal_loss = self.alpha * focal_weight * bce_loss
        
        return focal_loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss for multi-label classification
    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples"
    
    Uses re-weighting based on the number of samples per class
    """
    def __init__(self, samples_per_class=None, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.samples_per_class = samples_per_class

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Model logits [batch_size, num_classes]
            targets: Binary labels [batch_size, num_classes]
        """
        p = torch.sigmoid(outputs)
        bce_loss = nn.BCELoss(reduction='none')(p, targets)
        
        # Calculate effective number and weights
        if self.samples_per_class is not None:
            effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
            weights = (1.0 - self.beta) / effective_num
            weights = weights / weights.sum() * len(weights)
            weights = torch.tensor(weights, dtype=torch.float32, device=bce_loss.device)
        else:
            # If not provided, use uniform weights
            weights = torch.ones(bce_loss.shape[1], device=bce_loss.device)
        
        # Apply weighted BCE loss
        weighted_loss = bce_loss * weights.unsqueeze(0)
        
        return weighted_loss.mean()


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

        results[disease] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "kappa": float(kappa)
        }

    avg_f1 = np.mean(disease_f1_scores)
    results["average_f1"] = float(avg_f1)
    return results


# ========================
# Prediction helper for Kaggle onsite submission
# ========================
def predict_onsite_and_save(model, onsite_csv, onsite_image_dir, output_csv, device, img_size=256, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    onsite_ds = RetinaMultiLabelDataset(onsite_csv, onsite_image_dir, transform)
    onsite_loader = DataLoader(onsite_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_preds, all_files = [], []

    with torch.no_grad():
        for imgs, filenames in onsite_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_files.extend(filenames)

    all_preds = np.array(all_preds)
    submission_df = pd.DataFrame({
        "id": all_files,
        "D": all_preds[:, 0],
        "G": all_preds[:, 1],
        "A": all_preds[:, 2],
    })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    submission_df.to_csv(output_csv, index=False)
    print(f"Saved onsite submission: {output_csv} | samples: {len(submission_df)}")
    return submission_df


# ========================
# Training function
# ========================
def train_with_loss(backbone, loss_name, train_csv, val_csv, test_csv, 
                    train_image_dir, val_image_dir, test_image_dir,
                    pretrained_backbone=None, epochs=20, batch_size=32, 
                    lr=1e-4, img_size=256, save_dir="checkpoints", 
                    device=None, onsite_csv=None, onsite_image_dir=None, submissions_dir="submissions", **loss_kwargs):
    """
    Train model with specified loss function
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*70}")
    print(f"# TASK 2: LOSS FUNCTION - {loss_name.upper()}")
    print(f"# Backbone: {backbone.upper()}")
    print(f"{'#'*70}")

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
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)

    # Load pretrained model
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu")
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and 'fc' not in k and 'classifier' not in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    # Unfreeze all parameters (full fine-tuning)
    for param in model.parameters():
        param.requires_grad = True

    # Select loss function
    if loss_name == "bce":
        criterion = nn.BCEWithLogitsLoss()
        print(f"Using BCEWithLogitsLoss")
    elif loss_name == "focal":
        criterion = FocalLoss(alpha=loss_kwargs.get('alpha', 0.25), 
                            gamma=loss_kwargs.get('gamma', 2.0))
        print(f"Using Focal Loss (alpha={loss_kwargs.get('alpha', 0.25)}, gamma={loss_kwargs.get('gamma', 2.0)})")
    elif loss_name == "class_balanced":
        samples_per_class = loss_kwargs.get('samples_per_class', None)
        criterion = ClassBalancedLoss(samples_per_class=samples_per_class,
                                     beta=loss_kwargs.get('beta', 0.9999))
        print(f"Using Class-Balanced Loss (beta={loss_kwargs.get('beta', 0.9999)})")
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"task2_{loss_name}_{backbone}.pt")

    print(f"\nTraining for {epochs} epochs with learning rate {lr}...")
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

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    # Evaluate on test set (offsite)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nEvaluating {backbone} with {loss_name} loss on offsite test set...")
    results = evaluate_model(model, test_loader, device, backbone)

    print(f"\n{'-'*70}")
    for disease, metrics in results.items():
        if disease != "average_f1":
            print(f"{disease}: F1 = {metrics['f1_score']:.4f}")
    print(f"Average F1-score (offsite): {results['average_f1']:.4f}")
    print(f"{'-'*70}")

    # Generate onsite submission CSV (for Kaggle)
    submission_path = None
    if onsite_csv and onsite_image_dir:
        os.makedirs(submissions_dir, exist_ok=True)
        submission_path = os.path.join(submissions_dir, f"task2_{loss_name}_{backbone}.csv")
        predict_onsite_and_save(model, onsite_csv, onsite_image_dir, submission_path, device=device, img_size=img_size, batch_size=batch_size)

    return results, ckpt_path, submission_path


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
    onsite_csv = "onsite_test_submission.csv"
    onsite_image_dir = "./images/onsite_test"

    # Pretrained backbones
    pretrained_resnet18 = './pretrained_backbone/ckpt_resnet18_ep50.pt'
    pretrained_efficientnet = './pretrained_backbone/ckpt_efficientnet_ep50.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Calculate samples per class for class-balanced loss
    train_data = pd.read_csv(train_csv)
    samples_per_class_resnet = train_data[['D', 'G', 'A']].sum().values
    print(f"\nSamples per class (DR, Glaucoma, AMD): {samples_per_class_resnet}")

    # Store all results
    all_results = {}

    # ========================
    # Loss Functions to compare
    # ========================
    loss_configs = [
        ("bce", {}),  # Baseline BCE
        ("focal", {"alpha": 0.25, "gamma": 2.0}),  # Focal Loss
        ("class_balanced", {"beta": 0.9999, "samples_per_class": samples_per_class_resnet}),  # Class-Balanced Loss
    ]

    # ========================
    # ResNet18 with different losses
    # ========================
    print(f"\n\n{'='*70}")
    print(f"{'='*70}")
    print(f"RESNET18 - LOSS FUNCTION COMPARISON")
    print(f"{'='*70}")
    print(f"{'='*70}")

    for loss_name, loss_kwargs in loss_configs:
        results, ckpt_path, submission_path = train_with_loss(
            backbone="resnet18",
            loss_name=loss_name,
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
            device=device,
            onsite_csv=onsite_csv,
            onsite_image_dir=onsite_image_dir,
            **loss_kwargs
        )
        all_results[f"resnet18_{loss_name}"] = results
        if submission_path:
            all_results[f"resnet18_{loss_name}"]["onsite_submission"] = submission_path

    # ========================
    # EfficientNet with different losses
    # ========================
    print(f"\n\n{'='*70}")
    print(f"{'='*70}")
    print(f"EFFICIENTNET - LOSS FUNCTION COMPARISON")
    print(f"{'='*70}")
    print(f"{'='*70}")

    for loss_name, loss_kwargs in loss_configs:
        results, ckpt_path, submission_path = train_with_loss(
            backbone="efficientnet",
            loss_name=loss_name,
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
            device=device,
            onsite_csv=onsite_csv,
            onsite_image_dir=onsite_image_dir,
            **loss_kwargs
        )
        all_results[f"efficientnet_{loss_name}"] = results
        if submission_path:
            all_results[f"efficientnet_{loss_name}"]["onsite_submission"] = submission_path

    # ========================
    # Summary
    # ========================
    print(f"\n\n{'='*70}")
    print(f"TASK 2 SUMMARY - Average F1-Scores")
    print(f"{'='*70}")

    print("\nResNet18:")
    for loss_name, _ in loss_configs:
        f1 = all_results[f"resnet18_{loss_name}"]["average_f1"]
        print(f"  {loss_name.upper():20s}: {f1:.4f}")

    print("\nEfficientNet:")
    for loss_name, _ in loss_configs:
        f1 = all_results[f"efficientnet_{loss_name}"]["average_f1"]
        print(f"  {loss_name.upper():20s}: {f1:.4f}")

    print(f"{'='*70}")

    # Save results to JSON
    with open("task2_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to task2_results.json")
