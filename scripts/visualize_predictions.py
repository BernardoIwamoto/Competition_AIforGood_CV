import torch
import matplotlib.pyplot as plt

from src.data_loader.loaders import create_dataloaders
from src.models.unet_baseline import UNetSmall

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader = create_dataloaders(
    "train.csv",
    "val.csv",
    batch_size=1,
    num_workers=0
)

model = UNetSmall().to(DEVICE)
model.load_state_dict(torch.load("baseline.pth", map_location=DEVICE))
model.eval()

images, masks = next(iter(val_loader))
images = images.to(DEVICE)

with torch.no_grad():
    preds = torch.sigmoid(model(images))

img = images[0].permute(1, 2, 0).cpu().numpy()
gt = masks[0][0].cpu().numpy()
pred = preds[0][0].cpu().numpy()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title("Imagem")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,3,2)
plt.title("GT")
plt.imshow(gt, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Predição")
plt.imshow(pred > 0.5, cmap="gray")
plt.axis("off")

plt.show()