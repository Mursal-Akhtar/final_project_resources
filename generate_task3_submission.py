"""
Generate Kaggle Submission for Task 3 SE Model (82.05% on offsite test)
Run this in Colab where the trained model exists
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from task3 import ResNet18WithAttention


# ========================
# Dataset for prediction
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
        if self.transform:
            img = self.transform(img)
        return img, row.iloc[0]  # Return image and filename


# ========================
# Generate Predictions
# ========================
def generate_task3_predictions(model_path, onsite_csv, onsite_image_dir, 
                               output_csv, attention_type="se", 
                               batch_size=32, img_size=256):
    """
    Generate predictions using Task 3 attention model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"TASK 3 SUBMISSION - {attention_type.upper()} Model")
    print(f"Model: {model_path}")
    print(f"{'='*70}")

    # Transforms (same as training)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    test_ds = RetinaMultiLabelDataset(onsite_csv, onsite_image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load model
    print(f"\nLoading {attention_type.upper()} model...")
    model = ResNet18WithAttention(attention_type, num_classes=3, pretrained_weights=None)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Model loaded from: {model_path}")
    else:
        print(f"ERROR: Model file not found: {model_path}")
        return None

    model.to(device)
    model.eval()

    # Generate predictions
    all_predictions = []
    all_filenames = []

    print(f"\nGenerating predictions on {len(test_ds)} onsite test samples...")
    with torch.no_grad():
        for imgs, filenames in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_predictions.extend(preds)
            all_filenames.extend(filenames)

    # Create submission dataframe
    all_predictions = np.array(all_predictions)
    
    submission_df = pd.DataFrame({
        'id': all_filenames,
        'D': all_predictions[:, 0],  # DR
        'G': all_predictions[:, 1],  # Glaucoma
        'A': all_predictions[:, 2]   # AMD
    })

    # Save to CSV
    submission_df.to_csv(output_csv, index=False)
    print(f"\n✓ Predictions saved to: {output_csv}")
    print(f"  Samples: {len(submission_df)}")
    
    # Summary statistics
    print(f"\nPrediction Distribution:")
    print(f"  DR (D):       {submission_df['D'].sum()} positives ({submission_df['D'].sum()/len(submission_df)*100:.1f}%)")
    print(f"  Glaucoma (G): {submission_df['G'].sum()} positives ({submission_df['G'].sum()/len(submission_df)*100:.1f}%)")
    print(f"  AMD (A):      {submission_df['A'].sum()} positives ({submission_df['A'].sum()/len(submission_df)*100:.1f}%)")
    
    print(f"\nFirst 5 predictions:")
    print(submission_df.head())
    
    print(f"\n{'='*70}")
    print(f"SUBMISSION FILE READY FOR KAGGLE")
    print(f"{'='*70}")
    print(f"\nDownload this file and submit to Kaggle:")
    print(f"  {output_csv}")
    print(f"\nExpected performance (based on offsite test):")
    if attention_type == "se":
        print(f"  Task 3 SE: ~82% F1 (our best model)")
    elif attention_type == "mha":
        print(f"  Task 3 MHA: ~79% F1")
    
    return submission_df


# ========================
# Main
# ========================
if __name__ == "__main__":
    # Data paths (adjust if needed for your Colab setup)
    onsite_csv = "onsite_test_submission.csv"
    onsite_image_dir = "./images/onsite_test"
    
    # Task 3 model paths
    se_model_path = "./checkpoints/task3_se_resnet18.pt"
    mha_model_path = "./checkpoints/task3_mha_resnet18.pt"
    
    # Output submission files
    se_submission = "task3_se_onsite_submission.csv"
    mha_submission = "task3_mha_onsite_submission.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ========================
    # Generate Task 3 SE Predictions (BEST MODEL - 82.05%)
    # ========================
    print(f"\n{'#'*70}")
    print(f"# GENERATING TASK 3 SE SUBMISSION (82.05% on offsite test)")
    print(f"{'#'*70}\n")

    se_preds = generate_task3_predictions(
        model_path=se_model_path,
        onsite_csv=onsite_csv,
        onsite_image_dir=onsite_image_dir,
        output_csv=se_submission,
        attention_type="se",
        batch_size=32,
        img_size=256
    )

    # ========================
    # Optional: Generate Task 3 MHA Predictions (79.29%)
    # ========================
    print(f"\n\n{'#'*70}")
    print(f"# GENERATING TASK 3 MHA SUBMISSION (79.29% on offsite test)")
    print(f"{'#'*70}\n")
    
    mha_preds = generate_task3_predictions(
        model_path=mha_model_path,
        onsite_csv=onsite_csv,
        onsite_image_dir=onsite_image_dir,
        output_csv=mha_submission,
        attention_type="mha",
        batch_size=32,
        img_size=256
    )

    # ========================
    # Download Instructions
    # ========================
    print(f"\n\n{'='*70}")
    print(f"DOWNLOAD AND SUBMIT TO KAGGLE")
    print(f"{'='*70}")
    print(f"\nIn Colab, run:")
    print(f"  from google.colab import files")
    print(f"  files.download('{se_submission}')")
    print(f"\nThen upload to Kaggle competition to see actual performance!")
    print(f"\nRecommendation: Submit SE model first (best offsite performance)")
    print(f"{'='*70}\n")
