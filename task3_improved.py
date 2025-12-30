"""
Task 3 IMPROVED: Attention Mechanisms with optimizations
- Class-balanced loss (from Task 2 success)
- Extended training (40 epochs with early stopping)
- Learning rate scheduler
- Better data augmentation
- Optimized hyperparameters
"""

import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score


# ========================
# Dataset with augmentation
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
# Class-Balanced Loss (from Task 2)
# ========================
class ClassBalancedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weights):
        super().__init__()
        self.pos_weights = pos_weights
        
    def forward(self, logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weights
        )


# ========================
# Attention Modules
# ========================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head spatial self-attention"""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        self.head_dim = channels // num_heads
        
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        b, c, h, w = x.size()
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        out = self.norm(out)
        
        return x + out


# ========================
# ResNet18 with Attention
# ========================
class ResNet18WithAttention(nn.Module):
    def __init__(self, attention_type, num_classes=3, pretrained_weights=None):
        super().__init__()
        self.attention_type = attention_type
        
        # Base ResNet18
        resnet = models.resnet18(pretrained=False)
        if pretrained_weights:
            resnet.load_state_dict(torch.load(pretrained_weights, map_location='cpu'))
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Add attention modules
        if attention_type == "se":
            self.attn1 = SEBlock(64, reduction=16)
            self.attn2 = SEBlock(128, reduction=16)
            self.attn3 = SEBlock(256, reduction=16)
            self.attn4 = SEBlock(512, reduction=16)
        elif attention_type == "mha":
            self.attn1 = None
            self.attn2 = None
            self.attn3 = MultiHeadSelfAttention(256, num_heads=4)
            self.attn4 = MultiHeadSelfAttention(512, num_heads=4)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        if self.attn1:
            x = self.attn1(x)
        
        x = self.layer2(x)
        if self.attn2:
            x = self.attn2(x)
        
        x = self.layer3(x)
        if self.attn3:
            x = self.attn3(x)
        
        x = self.layer4(x)
        if self.attn4:
            x = self.attn4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


# ========================
# Training with improvements
# ========================
def train_attention_model(
    attention_type,
    train_csv="train.csv",
    val_csv="val.csv",
    test_csv="offsite_test.csv",
    train_dir="./images/train",
    val_dir="./images/val",
    test_dir="./images/offsite_test",
    epochs=40,
    batch_size=32,
    lr=5e-5,
    save_dir="checkpoints",
    pretrained_backbone="./pretrained_backbone/ckpt_resnet18_ep50.pt"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training ResNet18 + {attention_type.upper()} Attention (IMPROVED)")
    print(f"{'='*60}")
    
    # Enhanced data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Datasets
    train_ds = RetinaMultiLabelDataset(train_csv, train_dir, train_transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_dir, test_transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_dir, test_transform)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Calculate class weights for balanced loss
    train_labels = pd.read_csv(train_csv).iloc[:, 1:].values
    pos_counts = train_labels.sum(axis=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weights = torch.tensor(neg_counts / pos_counts, dtype=torch.float32).to(device)
    print(f"Class weights (DR, Glaucoma, AMD): {pos_weights.cpu().numpy()}")
    
    # Model
    model = ResNet18WithAttention(
        attention_type=attention_type,
        num_classes=3,
        pretrained_weights=pretrained_backbone
    ).to(device)
    
    # Class-balanced loss
    criterion = ClassBalancedBCEWithLogitsLoss(pos_weights)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Training
    best_val_f1 = -1
    patience = 10
    patience_counter = 0
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"task3_improved_{attention_type}_resnet18.pt")
    
    for epoch in range(epochs):
        # Train
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
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds)
        
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(np.array(y_true), np.array(y_pred), average='macro', zero_division=0)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"✓ Saved best model (Val F1: {val_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test
    print(f"\nLoading best model from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Per-disease metrics
    disease_names = ["DR", "Glaucoma", "AMD"]
    results = {"attention_type": attention_type, "diseases": {}}
    
    print(f"\n{'='*60}")
    print(f"Test Set Results - ResNet18 + {attention_type.upper()}")
    print(f"{'='*60}")
    
    disease_f1_scores = []
    for i, disease in enumerate(disease_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        acc = accuracy_score(y_t, y_p)
        precision = precision_score(y_t, y_p, average="macro", zero_division=0)
        recall = recall_score(y_t, y_p, average="macro", zero_division=0)
        f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
        
        disease_f1_scores.append(f1)
        results["diseases"][disease] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        print(f"{disease:10s} | Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
    
    # Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    results["macro_f1"] = float(macro_f1)
    results["best_val_f1"] = float(best_val_f1)
    
    print(f"{'='*60}")
    print(f"MACRO F1: {macro_f1:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    result_file = f"task3_improved_{attention_type}_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {result_file}")
    
    return macro_f1, results


# ========================
# Main
# ========================
if __name__ == "__main__":
    print("Task 3 IMPROVED - Attention Mechanisms")
    print("Training with class-balanced loss, augmentation, and optimized hyperparameters\n")
    
    # Train SE (Squeeze-and-Excitation)
    se_f1, se_results = train_attention_model(
        attention_type="se",
        epochs=40,
        batch_size=32,
        lr=5e-5  # Lower LR for stability
    )
    
    # Train MHA (Multi-head Attention)
    mha_f1, mha_results = train_attention_model(
        attention_type="mha",
        epochs=40,
        batch_size=16,  # Smaller batch for memory
        lr=1e-4  # Slightly higher for MHA
    )
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - TASK 3 IMPROVED")
    print("="*60)
    print(f"ResNet18 + SE:  {se_f1:.4f}")
    print(f"ResNet18 + MHA: {mha_f1:.4f}")
    print(f"\nReference: 78.8%")
    print(f"SE vs Ref: {(se_f1 - 0.788)*100:+.2f}%")
    print(f"MHA vs Ref: {(mha_f1 - 0.788)*100:+.2f}%")
    print("="*60)
