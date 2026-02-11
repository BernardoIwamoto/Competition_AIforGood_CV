import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


class SegFormerModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.logits