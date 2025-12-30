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
# Attention Mechanisms
# ========================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (SE-Net)
    Channel attention mechanism that recalibrates channel-wise feature responses
    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        b, c, h, w = x.size()
        se = x.view(b, c, -1).mean(dim=2)  # [b, c]
        
        # Squeeze-and-Excitation
        se = self.fc1(se)  # [b, c/16]
        se = self.relu(se)
        se = self.fc2(se)  # [b, c]
        se = self.sigmoid(se)  # [b, c]
        
        # Scale input
        se = se.view(b, c, 1, 1)
        return x * se


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA)
    Self-attention mechanism with multiple representation subspaces
    Reference: Vaswani et al., "Attention is All You Need" (NeurIPS 2017)
    """
    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.query = nn.Linear(in_channels, in_channels, bias=False)
        self.key = nn.Linear(in_channels, in_channels, bias=False)
        self.value = nn.Linear(in_channels, in_channels, bias=False)
        self.fc_out = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input: [b, c, h, w]
        b, c, h, w = x.size()
        
        # Reshape to [b, c, h*w]
        x_reshape = x.view(b, c, -1)  # [b, c, hw]
        x_reshape = x_reshape.permute(0, 2, 1)  # [b, hw, c]
        
        # Linear projections
        Q = self.query(x_reshape)  # [b, hw, c]
        K = self.key(x_reshape)    # [b, hw, c]
        V = self.value(x_reshape)  # [b, hw, c]
        
        # Split into multiple heads
        Q = Q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, hw, d]
        K = K.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, hw, d]
        V = V.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [b, h, hw, d]
        
        # Scaled dot-product attention
        # Use sqrt on CPU-free scalar to avoid casting issues
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, V)  # [b, h, hw, d]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous()  # [b, hw, h, d]
        out = out.view(b, -1, c)  # [b, hw, c]
        
        # Final linear projection
        out = self.fc_out(out)  # [b, hw, c]
        out = self.dropout(out)
        
        # Reshape back
        out = out.permute(0, 2, 1)  # [b, c, hw]
        out = out.view(b, c, h, w)  # [b, c, h, w]
        
        # Residual connection for stability
        return out + x


# ========================
# Modified Models with Attention
# ========================

class ResNet18WithAttention(nn.Module):
    """ResNet18 with SE attention blocks inserted"""
    def __init__(self, attention_type="se", num_classes=3, pretrained_weights=None):
        super(ResNet18WithAttention, self).__init__()
        self.attention_type = attention_type
        
        # Load base ResNet18
        base_model = models.resnet18(pretrained=False)
        
        # Copy layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)
        
        # Add attention blocks
        if attention_type == "se":
            self.attention1 = SEBlock(64)
            self.attention2 = SEBlock(128)
            self.attention3 = SEBlock(256)
            self.attention4 = SEBlock(512)
        elif attention_type == "mha":
            # Apply MHA at deeper stages (small spatial map) to control memory
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
            self.attention3 = MultiHeadAttention(256, num_heads=4)
            self.attention4 = MultiHeadAttention(512, num_heads=4)
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            # Load weights into base model layers
            for name, param in self.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.attention1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)
        
        x = self.layer3(x)
        x = self.attention3(x)
        
        x = self.layer4(x)
        x = self.attention4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class EfficientNetWithAttention(nn.Module):
    """EfficientNet-B0 with SE or MHA attention"""
    def __init__(self, attention_type="se", num_classes=3, pretrained_weights=None):
        super(EfficientNetWithAttention, self).__init__()
        self.attention_type = attention_type
        
        # Load base EfficientNet
        base_model = models.efficientnet_b0(pretrained=False)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.classifier = base_model.classifier
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)
        
        # Add attention block on top of features
        if attention_type == "se":
            self.attention = SEBlock(1280)  # EfficientNet-B0 final channels
        elif attention_type == "mha":
            self.attention = MultiHeadAttention(1280, num_heads=8)
        
        # Load pretrained weights if provided
        if pretrained_weights is not None:
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            # Load weights into base model
            for name, param in self.named_parameters():
                if name in state_dict:
                    param.data = state_dict[name]
    
    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ========================
# Evaluation function
# ========================
def evaluate_model(model, data_loader, device, backbone="resnet18"):
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
# Training function
# ========================
def train_with_attention(backbone, attention_type, train_csv, val_csv, test_csv,
                        train_image_dir, val_image_dir, test_image_dir,
                        pretrained_backbone=None, epochs=20, batch_size=32,
                        img_size=256, save_dir="checkpoints", device=None):
    """
    Train model with attention mechanism
    Uses head-only fine-tuning strategy (proven best for this task)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#'*70}")
    print(f"# TASK 3: ATTENTION MECHANISMS")
    print(f"# Backbone: {backbone.upper()}")
    print(f"# Attention: {attention_type.upper()}")
    print(f"{'#'*70}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets & DataLoaders
    effective_batch_size = batch_size if attention_type != "mha" else min(batch_size, 8)
    train_ds = RetinaMultiLabelDataset(train_csv, train_image_dir, transform)
    val_ds = RetinaMultiLabelDataset(val_csv, val_image_dir, transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_image_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=effective_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=effective_batch_size, shuffle=False, num_workers=0)

    # Build model with attention
    if backbone == "resnet18":
        model = ResNet18WithAttention(attention_type=attention_type, 
                                     num_classes=3,
                                     pretrained_weights=pretrained_backbone).to(device)
        # Fine-tune full model (ResNet18 works well with full FT)
        for param in model.parameters():
            param.requires_grad = True
        lr = 1.5e-4 if attention_type == "mha" else 1e-4
    elif backbone == "efficientnet":
        model = EfficientNetWithAttention(attention_type=attention_type,
                                         num_classes=3,
                                         pretrained_weights=pretrained_backbone).to(device)
        # Fine-tune head only (EfficientNet better with head FT)
        for param in model.features.parameters():
            param.requires_grad = False
        lr = 1e-3

    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training
    best_val_loss = float("inf")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"task3_{attention_type}_{backbone}.pt")

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

    # Evaluate on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\nEvaluating {backbone} with {attention_type} attention on offsite test set...")
    results = evaluate_model(model, test_loader, device, backbone)

    print(f"\n{'-'*70}")
    for disease, metrics in results.items():
        if disease != "average_f1":
            print(f"{disease}: F1 = {metrics['f1_score']:.4f}")
    print(f"Average F1-score (offsite): {results['average_f1']:.4f}")
    print(f"{'-'*70}")

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

    all_results = {}

    # Attention types
    attention_types = ["se", "mha"]

    # ========================
    # ResNet18 with attention
    # ========================
    print(f"\n\n{'='*70}")
    print(f"RESNET18 - ATTENTION MECHANISMS")
    print(f"{'='*70}\n")

    for attn_type in attention_types:
        results, ckpt_path = train_with_attention(
            backbone="resnet18",
            attention_type=attn_type,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
            test_image_dir=test_image_dir,
            pretrained_backbone=pretrained_resnet18,
            epochs=20,
            batch_size=32,
            device=device
        )
        all_results[f"resnet18_{attn_type}"] = results

    # ========================
    # EfficientNet with attention
    # ========================
    print(f"\n\n{'='*70}")
    print(f"EFFICIENTNET - ATTENTION MECHANISMS")
    print(f"{'='*70}\n")

    for attn_type in attention_types:
        results, ckpt_path = train_with_attention(
            backbone="efficientnet",
            attention_type=attn_type,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            train_image_dir=train_image_dir,
            val_image_dir=val_image_dir,
            test_image_dir=test_image_dir,
            pretrained_backbone=pretrained_efficientnet,
            epochs=20,
            batch_size=32,
            device=device
        )
        all_results[f"efficientnet_{attn_type}"] = results

    # ========================
    # Summary
    # ========================
    print(f"\n\n{'='*70}")
    print(f"TASK 3 SUMMARY - Attention Mechanisms (Offsite F1)")
    print(f"{'='*70}")

    print("\nResNet18:")
    for attn_type in attention_types:
        f1 = all_results[f"resnet18_{attn_type}"]["average_f1"]
        print(f"  {attn_type.upper():20s}: {f1:.4f}")

    print("\nEfficientNet:")
    for attn_type in attention_types:
        f1 = all_results[f"efficientnet_{attn_type}"]["average_f1"]
        print(f"  {attn_type.upper():20s}: {f1:.4f}")

    print(f"{'='*70}")

    # Save results to JSON
    with open("task3_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to task3_results.json")
