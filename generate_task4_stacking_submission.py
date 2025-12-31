"""
Generate Kaggle Submission for Task 4 Stacking Ensemble
Combines Task 3 SE + MHA with meta-learner trained on validation set
Run this in Colab where the trained models exist
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

from task3 import ResNet18WithAttention


# ========================
# Dataset for prediction
# ========================
class RetinaMultiLabelDataset(Dataset):
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
# Load models
# ========================
def load_task3_models(device):
    """Load both Task 3 models"""
    print("Loading Task 3 models...")
    
    # SE model
    se_model = ResNet18WithAttention("se", num_classes=3, pretrained_weights=None)
    se_state = torch.load("checkpoints/task3_se_resnet18.pt", map_location=device)
    se_model.load_state_dict(se_state)
    se_model.to(device)
    se_model.eval()
    
    # MHA model
    mha_model = ResNet18WithAttention("mha", num_classes=3, pretrained_weights=None)
    mha_state = torch.load("checkpoints/task3_mha_resnet18.pt", map_location=device)
    mha_model.load_state_dict(mha_state)
    mha_model.to(device)
    mha_model.eval()
    
    print("✓ Models loaded successfully")
    return se_model, mha_model


# ========================
# Get predictions from base models
# ========================
def get_predictions(models, loader, device, has_labels=True):
    """Get predictions from both models"""
    se_model, mha_model = models
    
    se_preds = []
    mha_preds = []
    all_labels = [] if has_labels else None
    all_filenames = []
    
    with torch.no_grad():
        for batch in loader:
            if has_labels:
                imgs, labels, filenames = batch
                all_labels.append(labels.numpy())
            else:
                imgs, filenames = batch
            
            imgs = imgs.to(device)
            
            # SE predictions
            se_out = torch.sigmoid(se_model(imgs)).cpu().numpy()
            se_preds.append(se_out)
            
            # MHA predictions
            mha_out = torch.sigmoid(mha_model(imgs)).cpu().numpy()
            mha_preds.append(mha_out)
            
            all_filenames.extend(filenames)
    
    if has_labels:
        return (np.vstack(se_preds), np.vstack(mha_preds), 
                np.vstack(all_labels), all_filenames)
    else:
        return np.vstack(se_preds), np.vstack(mha_preds), all_filenames


# ========================
# Meta-learner
# ========================
class StackingMetaLearner:
    """Per-class logistic regression meta-learner"""
    
    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.meta_models = [LogisticRegression(max_iter=1000, random_state=42) 
                           for _ in range(num_classes)]
    
    def train(self, se_preds, mha_preds, labels):
        """Train on validation set"""
        print("\nTraining meta-learner on validation set...")
        X = np.hstack([se_preds, mha_preds])
        
        for i in range(self.num_classes):
            y = labels[:, i]
            self.meta_models[i].fit(X, y)
            coef = self.meta_models[i].coef_[0]
            print(f"  Class {i}: SE weight={coef[i]:.4f}, MHA weight={coef[i+3]:.4f}")
        
        print("✓ Meta-learner trained")
    
    def predict(self, se_preds, mha_preds):
        """Predict on test set"""
        X = np.hstack([se_preds, mha_preds])
        preds = np.zeros((X.shape[0], self.num_classes))
        
        for i in range(self.num_classes):
            preds[:, i] = self.meta_models[i].predict_proba(X)[:, 1]
        
        return preds


# ========================
# Generate stacking submission
# ========================
def generate_stacking_submission():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*70}")
    print(f"TASK 4 STACKING ENSEMBLE SUBMISSION")
    print(f"{'='*70}")
    print(f"Device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load validation set (has labels)
    val_ds = RetinaMultiLabelDataset("val.csv", "./images/val", transform, has_labels=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Load onsite test set (no labels - just image filenames)
    onsite_ds = RetinaMultiLabelDataset("onsite_test_submission.csv", "./images/onsite_test", transform, has_labels=False)
    onsite_loader = DataLoader(onsite_ds, batch_size=32, shuffle=False, num_workers=0)
    
    # Load models
    models = load_task3_models(device)
    
    # Get validation predictions (for training meta-learner)
    print(f"\nGenerating predictions on validation set ({len(val_ds)} samples)...")
    val_se, val_mha, val_labels, _ = get_predictions(models, val_loader, device, has_labels=True)
    print(f"✓ Validation predictions generated")
    
    # Train meta-learner
    meta_learner = StackingMetaLearner(num_classes=3)
    meta_learner.train(val_se, val_mha, val_labels)
    
    # Get onsite test predictions
    print(f"\nGenerating predictions on onsite test set ({len(onsite_ds)} samples)...")
    test_se, test_mha, filenames = get_predictions(models, onsite_loader, device, has_labels=False)
    print(f"✓ Onsite test predictions generated")
    
    # Get stacked predictions
    print("\nGenerating stacked ensemble predictions...")
    stack_probs = meta_learner.predict(test_se, test_mha)
    stack_preds = (stack_probs > 0.5).astype(int)
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': filenames,
        'D': stack_preds[:, 0],  # DR
        'G': stack_preds[:, 1],  # Glaucoma
        'A': stack_preds[:, 2]   # AMD
    })
    
    # Save submission
    output_csv = "task4_stacking_onsite_submission.csv"
    submission_df.to_csv(output_csv, index=False)
    
    print(f"\n✓ Submission saved to: {output_csv}")
    print(f"  Samples: {len(submission_df)}")
    
    # Statistics
    print(f"\nPrediction Distribution:")
    print(f"  DR (D):       {submission_df['D'].sum()} positives ({submission_df['D'].sum()/len(submission_df)*100:.1f}%)")
    print(f"  Glaucoma (G): {submission_df['G'].sum()} positives ({submission_df['G'].sum()/len(submission_df)*100:.1f}%)")
    print(f"  AMD (A):      {submission_df['A'].sum()} positives ({submission_df['A'].sum()/len(submission_df)*100:.1f}%)")
    
    print(f"\nFirst 5 predictions:")
    print(submission_df.head())
    
    print(f"\n{'='*70}")
    print(f"STACKING SUBMISSION READY")
    print(f"{'='*70}")
    print(f"\nMethod: Meta-learner (LogisticRegression) on Task 3 SE + MHA")
    print(f"Expected: Potentially better than single SE model (~82%)")
    print(f"\nDownload command:")
    print(f"  from google.colab import files")
    print(f"  files.download('{output_csv}')")
    print(f"\nSubmit to Kaggle and compare with Task 3 SE submission!")
    print(f"{'='*70}\n")
    
    return submission_df


# ========================
# Main
# ========================
if __name__ == "__main__":
    submission = generate_stacking_submission()
