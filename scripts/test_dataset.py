import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch

# permite importar src/
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from src.datasets.farmland_dataset import FarmlandDataset


def main():
    csv_path = repo_root / "data" / "dataset_map.csv"

    dataset = FarmlandDataset(csv_path)

    print(f"Dataset carregado com {len(dataset)} amostras")

    # testa alguns índices
    for idx in [0, 10, 100, 1000]:
        image, mask = dataset[idx]

        print(f"\nÍndice {idx}")
        print("Imagem shape:", image.shape)
        print("Máscara shape:", mask.shape)
        print("Valores únicos da máscara:", torch.unique(mask))

        # visualização
        img_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze(0).numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_np)
        axs[0].set_title("Imagem")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Máscara binária (Farmland)")
        axs[1].axis("off")

        plt.show()


if __name__ == "__main__":
    main()
