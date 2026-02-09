from torch.utils.data import DataLoader
from src.datasets.farmland_dataset import FarmlandDataset
from src.augmentations.transforms import get_train_transforms, get_val_transforms

def create_dataloaders(
    train_csv,
    val_csv,
    batch_size=8,
    num_workers=4
):
    train_dataset = FarmlandDataset(
        csv_path=train_csv,
        transform=get_train_transforms()
    )

    val_dataset = FarmlandDataset(
        csv_path=val_csv,
        transform=get_val_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        #pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        #pin_memory=True
    )

    return train_loader, val_loader