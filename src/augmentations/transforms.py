import albumentations as A
import cv2

def get_train_transforms(image_size=512):
    return A.Compose([
        # Geometria
        # Flips e rotações de 90°
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Rotações arbitrárias até 45°. BORDER_REFLECT evita bordas pretas que confundiriam o modelo (bordas pretas não existem na natureza)
        A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

        # Simula diferentes altitudes/zooms. Crop aleatório e resize de volta
        # para image_size. scale=(0.5, 1.0) significa que o crop pode ser
        # de 50% a 100% da imagem original — forçando o modelo a aprender
        # features em múltiplas escalas.
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.5, 1.0),
            ratio=(0.9, 1.1),  # quase quadrado, como imagens de satélite
            p=0.5
        ),

        # Pequenas distorções geométricas — simula artefatos de projeção
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.2),

        # Cor e iluminação
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),

        # Simula mudanças sazonais mais drásticas (vegetação seca vs verde, por exemplo)
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.3
        ),

        # Ruído e blur
        # GaussianBlur simula haze atmosférico e limites de resolução do sensor
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),

        # Ruído de sensor — muito comum em imagens de satélite
        A.GaussNoise(p=0.2),

        # Regularização visual
        # CoarseDropout: remove patches aleatórios da imagem,
        # forçando o modelo a não depender de nenhuma região específica.
        # Útil para evitar overfitting em texturas repetitivas de farmland.
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.2
        ),
    ])

def get_val_transforms():
    return A.Compose([])