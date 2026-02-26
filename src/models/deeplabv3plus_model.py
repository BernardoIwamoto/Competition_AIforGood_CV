import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3PlusModel(nn.Module):
    def __init__(
        self,
        encoder_name: str = "resnet101", # maior acurácia, maior custo
        encoder_weights: str = "imagenet", # pré-treinado no ImageNet
        num_classes: int = 1,
    ):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None,  # logits — sigmoid aplicado na loss (BCEDiceLoss)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)