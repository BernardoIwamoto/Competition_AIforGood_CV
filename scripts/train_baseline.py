from pathlib import Path
import torch
import torch.nn as nn

from src.data_loader.loaders import create_dataloaders
from src.models.unet_baseline import UNetSmall
from src.utils.metrics import iou_score, dice_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DATALOADERS
train_loader, val_loader = create_dataloaders(
    "data/train.csv",
    "data/val.csv",
    batch_size=8,
    num_workers=0
)

# MODELO
model = UNetSmall().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# CHECKPOINTS
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

save_every = 5
num_epochs = 50


# TREINAMENTO
for epoch in range(num_epochs):
    #TREINO
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # VALIDAÇÃO
    model.eval()
    val_iou = 0.0
    val_dice = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            val_iou += iou_score(preds, masks).item()
            val_dice += dice_score(preds, masks).item()

    val_iou /= len(val_loader)
    val_dice /= len(val_loader)

    # LOG 
    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val IoU: {val_iou:.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # CHECKPOINT
    if (epoch + 1) % save_every == 0:
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pth"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_iou": val_iou,
                "val_dice": val_dice,
            },
            checkpoint_path,
        )
        print(f"Checkpoint salvo em {checkpoint_path}")

# SALVA MODELO FINAL
final_path = checkpoint_dir / "model_last.pth"
torch.save(
    {
        "epoch": num_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    },
    final_path,
)

print(f"Modelo final salvo em {final_path}")