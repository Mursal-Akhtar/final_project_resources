import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


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
# Generate Predictions
# ========================
def generate_predictions(backbone, model_path, onsite_csv, onsite_image_dir, 
                        output_csv, batch_size=32, img_size=256, device=None):
    """
    Generate predictions on onsite test set for Kaggle submission
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"Generating Predictions - {backbone.upper()}")
    print(f"Model: {model_path}")
    print(f"{'='*70}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    test_ds = RetinaMultiLabelDataset(onsite_csv, onsite_image_dir, transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build and load model
    model = build_model(backbone, num_classes=3, pretrained=False).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"ERROR: Model file not found: {model_path}")
        return None

    model.eval()

    # Generate predictions
    all_predictions = []
    all_filenames = []

    print(f"\nGenerating predictions on {len(test_ds)} samples...")
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
    print(f"\nFirst 5 predictions:")
    print(submission_df.head())

    return submission_df


# ========================
# Main
# ========================
if __name__ == "__main__":
    import torch.nn as nn
    
    # Data paths
    onsite_csv = "onsite_test_submission.csv"
    onsite_image_dir = "./images/onsite_test"
    
    # Model paths (from Task 1 best results)
    resnet18_model = "./checkpoints/task1_3_resnet18.pt"  # Full fine-tune: 79.87% F1
    efficientnet_model = "./checkpoints/task1_2_efficientnet.pt"  # Head fine-tune: 75.84% F1
    
    # Output submission files
    resnet18_submission = "submissions/resnet18_task1_submission.csv"
    efficientnet_submission = "submissions/efficientnet_task1_submission.csv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create submissions directory
    os.makedirs("submissions", exist_ok=True)

    # ========================
    # Generate ResNet18 Predictions
    # ========================
    print(f"\n\n{'#'*70}")
    print(f"# GENERATING ONSITE TEST PREDICTIONS - TASK 1")
    print(f"{'#'*70}\n")

    resnet18_preds = generate_predictions(
        backbone="resnet18",
        model_path=resnet18_model,
        onsite_csv=onsite_csv,
        onsite_image_dir=onsite_image_dir,
        output_csv=resnet18_submission,
        batch_size=32,
        device=device
    )

    # ========================
    # Generate EfficientNet Predictions
    # ========================
    efficientnet_preds = generate_predictions(
        backbone="efficientnet",
        model_path=efficientnet_model,
        onsite_csv=onsite_csv,
        onsite_image_dir=onsite_image_dir,
        output_csv=efficientnet_submission,
        batch_size=32,
        device=device
    )

    # ========================
    # Summary
    # ========================
    print(f"\n\n{'='*70}")
    print(f"PREDICTIONS GENERATED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"\nResNet18 Submission: {resnet18_submission}")
    print(f"  - Samples: {len(resnet18_preds)}")
    print(f"  - DR positives: {resnet18_preds['D'].sum()}")
    print(f"  - Glaucoma positives: {resnet18_preds['G'].sum()}")
    print(f"  - AMD positives: {resnet18_preds['A'].sum()}")

    print(f"\nEfficientNet Submission: {efficientnet_submission}")
    print(f"  - Samples: {len(efficientnet_preds)}")
    print(f"  - DR positives: {efficientnet_preds['D'].sum()}")
    print(f"  - Glaucoma positives: {efficientnet_preds['G'].sum()}")
    print(f"  - AMD positives: {efficientnet_preds['A'].sum()}")

    print(f"\n{'='*70}")
    print(f"Next: Upload these CSV files to Kaggle competition")
    print(f"{'='*70}\n")
