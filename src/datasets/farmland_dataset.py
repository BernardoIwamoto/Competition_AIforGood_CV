import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path

from src.utils.mask_utils import rgb_mask_to_binary


class FarmlandDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        """
        Dataset para segmentação binária de farmland. Cria, de fato, o dataset como uma classe.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = Path(row["image_path"])
        mask_path = Path(row["label_path"])

        # --- leitura da imagem ---
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Erro ao ler imagem {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- leitura da máscara ---
        mask_rgb = cv2.imread(str(mask_path))
        if mask_rgb is None:
            raise RuntimeError(f"Erro ao ler máscara {mask_path}")

        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)

        # --- conversão para binário ---
        mask = rgb_mask_to_binary(mask_rgb)

        # --- normalização ---
        image = image.astype("float32") / 255.0

        # --- HWC -> CHW ---
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
