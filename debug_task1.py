import os
import torch
from torchvision import models
import pandas as pd

# Check data
print("="*70)
print("CHECKING DATA")
print("="*70)

csv_files = ["train.csv", "val.csv", "offsite_test.csv", "onsite_test_submission.csv"]
for csv in csv_files:
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        print(f"{csv}: {len(df)} samples, columns: {list(df.columns)}")
        print(f"  First row: {df.iloc[0].tolist()}")
    else:
        print(f"{csv}: NOT FOUND")

# Check pretrained models
print("\n" + "="*70)
print("CHECKING PRETRAINED MODELS")
print("="*70)

model_paths = ['./pretrained_backbone/ckpt_resnet18_ep50.pt', 
               './pretrained_backbone/ckpt_efficientnet_ep50.pt']

for model_path in model_paths:
    if os.path.exists(model_path):
        print(f"\n{model_path}: EXISTS")
        state_dict = torch.load(model_path, map_location="cpu")
        print(f"  Keys in checkpoint: {list(state_dict.keys())[:5]}...")  # First 5 keys
        print(f"  Total parameters: {len(state_dict)}")
    else:
        print(f"{model_path}: NOT FOUND")

# Check if we can build and load models
print("\n" + "="*70)
print("CHECKING MODEL BUILDING")
print("="*70)

# ResNet18
print("\nResNet18:")
model_resnet = models.resnet18(pretrained=True)
print(f"  Original model FC: {model_resnet.fc}")

# Try loading pretrained weights
resnet_state = torch.load('./pretrained_backbone/ckpt_resnet18_ep50.pt', map_location="cpu")
print(f"  Checkpoint has fc weights: {'fc.weight' in resnet_state and 'fc.bias' in resnet_state}")

# EfficientNet
print("\nEfficientNet:")
model_eff = models.efficientnet_b0(pretrained=True)
print(f"  Original model classifier: {model_eff.classifier}")

eff_state = torch.load('./pretrained_backbone/ckpt_efficientnet_ep50.pt', map_location="cpu")
print(f"  Checkpoint has classifier weights: {'classifier.1.weight' in eff_state or 'classifier.weight' in eff_state}")
print(f"  Keys in checkpoint: {list(eff_state.keys())[:10]}...")

# Check images
print("\n" + "="*70)
print("CHECKING IMAGE DIRECTORIES")
print("="*70)

dirs = ['./images/train', './images/val', './images/offsite_test', './images/onsite_test']
for d in dirs:
    if os.path.exists(d):
        files = os.listdir(d)
        print(f"{d}: {len(files)} images")
    else:
        print(f"{d}: NOT FOUND")
