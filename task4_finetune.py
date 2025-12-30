import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score

from task3 import ResNet18WithAttention
from PIL import Image


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
        img = Image.open(img_path).convert("RGB")
        labels = torch.tensor(row[1:].values.astype("float32"))
        if self.transform:
            img = self.transform(img)
        return img, labels


# ========================
# Evaluation
# ========================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.numpy()
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels)
            y_pred.extend(preds)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_scores = [f1_score(y_true[:, i], y_pred[:, i], average="binary", zero_division=0) for i in range(3)]
    return float(np.mean(f1_scores))


# ========================
# Fine-tune
# ========================
def finetune_task4():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Paths
    train_csv = "train.csv"
    val_csv = "val.csv"
    test_csv = "offsite_test.csv"
    train_dir = "./images/train"
    val_dir = "./images/val"
    test_dir = "./images/offsite_test"
    ckpt_se = "checkpoints/task3_se_resnet18.pt"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load train+val combined, split into train (800) and tuning val (200)
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)
    combined = pd.concat([train_data, val_data], ignore_index=True)

    # Split: first 800 for fine-tuning, last 200 for early stopping
    train_split = combined.iloc[:800]
    tune_val_split = combined.iloc[800:]

    # Save temp CSVs
    train_split.to_csv("temp_train_combined.csv", index=False)
    tune_val_split.to_csv("temp_val_combined.csv", index=False)

    # Datasets
    train_ds = RetinaMultiLabelDataset("temp_train_combined.csv", train_dir, transform)
    tune_val_ds = RetinaMultiLabelDataset("temp_val_combined.csv", val_dir, transform)
    test_ds = RetinaMultiLabelDataset(test_csv, test_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    tune_val_loader = DataLoader(tune_val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load Task 3 SE checkpoint
    if not os.path.exists(ckpt_se):
        raise FileNotFoundError(f"{ckpt_se} not found. Run task3.py first.")

    model = ResNet18WithAttention(attention_type="se", num_classes=3, pretrained_weights=None)
    state = torch.load(ckpt_se, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    # Optimizer: Adam on all params (already good from Task 3)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    
    # Scheduler: Cosine annealing over 5 epochs
    num_epochs = 5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))

    criterion = nn.BCEWithLogitsLoss()
    best_val_f1 = -1
    patience = 2
    patience_counter = 0
    best_ckpt_path = "checkpoints/task4_finetune_se_resnet18.pt"

    print("=" * 70)
    print("TASK 4: Fine-tuning Task 3 SE checkpoint")
    print("=" * 70)
    print(f"Training on {len(train_ds)} images + tuning val {len(tune_val_ds)} images")
    print(f"LR=5e-5, Cosine annealing, Early stopping patience=2\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_ds)

        # Eval on tuning val
        val_f1 = evaluate(model, tune_val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Tuning Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ✓ Best model saved: {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⚠ Early stopping at epoch {epoch+1}")
                break

    # Load best checkpoint and eval on offsite test
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_f1 = evaluate(model, test_loader, device)

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Best Tuning Val F1: {best_val_f1:.4f}")
    print(f"Offsite Test F1:    {test_f1:.4f}")
    print(f"Improvement over Task 3 SE (0.8355): {(test_f1 - 0.8355) * 100:+.2f}%")
    print(f"Reference (78.8%):  {test_f1:.4f} vs 0.788")
    if test_f1 > 0.803:
        print(f"✓ EXCEEDS >1.5% threshold! Expected 100% on Task 4")
    else:
        print(f"→ Comparable/below reference")
    print(f"{'='*70}\n")

    # Save results
    results = {
        "model": "ResNet18 + SE attention (Task 3) + fine-tune",
        "best_tuning_val_f1": float(best_val_f1),
        "offsite_test_f1": float(test_f1),
        "checkpoint": best_ckpt_path,
        "improvement_over_task3_se": float(test_f1 - 0.8355),
    }
    with open("task4_finetune_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to task4_finetune_results.json\n")

    # Cleanup
    if os.path.exists("temp_train_combined.csv"):
        os.remove("temp_train_combined.csv")
    if os.path.exists("temp_val_combined.csv"):
        os.remove("temp_val_combined.csv")


if __name__ == "__main__":
    finetune_task4()
