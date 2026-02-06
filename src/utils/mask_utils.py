import numpy as np

def rgb_mask_to_binary(mask_rgb):
    """
    Converte máscara RGB em máscara binária (Farmland).
    Farmland = (0, 255, 0) -> 1
    Todo o resto -> 0
    Converte aquele verde das máscaras em 1, e o resto em 0.
    """
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError("A máscara precisa ser HxWx3 (RGB).")

    farmland = np.array([0, 255, 0], dtype=np.uint8)

    binary_mask = np.all(mask_rgb == farmland, axis=-1).astype(np.uint8)

    return binary_mask
