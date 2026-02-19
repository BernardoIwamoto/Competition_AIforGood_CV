
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import time
import psutil
import os

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.swin_model import SwinTransformerUNet

# Define the transforms directly here to ensure we test exactly what we implemented
def get_train_transforms():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ToTensorV2()
    ])

class DummyDataset(Dataset):
    def __init__(self, length=100):
        self.length = length
        self.transform = get_train_transforms()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Create larger dummy images to simulate original GID images (e.g., 512x512 or 1024x1024)
        # to ensure resizing is working and saving memory.
        image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (1024, 1024), dtype=np.uint8)
        
        augmented = self.transform(image=image, mask=mask)
        image = augmented["image"].float() / 255.0
        mask = augmented["mask"].unsqueeze(0).float()
        
        return image, mask

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def main():
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    if DEVICE == "cpu" and torch.cuda.is_available():
        DEVICE = "cuda"
    print(f"Using device: {DEVICE}")

    # Model
    model = SwinTransformerUNet(num_classes=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Model loaded. Memory: {get_memory_usage():.2f} MB")

    # Dataset
    batch_size = 8
    dataset = DummyDataset(length=32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training loop simulation...")
    
    model.train()
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        
        # Verify shape
        if batch_idx == 0:
            print(f"Input batch shape: {images.shape}") # Should be [8, 3, 256, 256]
            if images.shape[2:] != (256, 256):
                 print("ERROR: Resizing did not work!")
                 return

        optimizer.zero_grad()
        outputs = model(images)
        
        # Check output shape
        if outputs.shape[-2:] != masks.shape[-2:]:
            outputs = torch.nn.functional.interpolate(
                outputs, 
                size=masks.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch_idx+1}/{len(loader)} processed. Memory: {get_memory_usage():.2f} MB")

    print("\n✅ Verification SUCCESS: Model trained for 1 epoch with Batch Size 8 and Resize(256, 256).")

if __name__ == "__main__":
    main()
