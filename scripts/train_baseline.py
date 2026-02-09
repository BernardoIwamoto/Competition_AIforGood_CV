import torch
import torch.nn as nn
from src.data_loader.loaders import create_dataloaders
from src.models.unet_baseline import UNetSmall

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader = create_dataloaders(
    "data/train.csv",
    "data/val.csv",
    batch_size=8,
    num_workers=0
)

model = UNetSmall().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch}: Train Loss {total_loss / len(train_loader):.4f}")