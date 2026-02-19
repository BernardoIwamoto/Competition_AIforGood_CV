import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SwinTransformerUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        
        # 1. Backbone: Swin Transformer v2 Tiny
        weights = Swin_V2_T_Weights.DEFAULT
        self.backbone = swin_v2_t(weights=weights)
        
        # Return nodes (stages output)
        # Swin-Tiny stages have channels: 96, 192, 384, 768
        # Stride: 4, 8, 16, 32
        return_nodes = {
            "features.1": "stage1", # Stride 4,  96 channels
            "features.3": "stage2", # Stride 8,  192 channels
            "features.5": "stage3", # Stride 16, 384 channels
            "features.7": "stage4", # Stride 32, 768 channels
        }
        self.encoder = create_feature_extractor(self.backbone, return_nodes=return_nodes)

        # 2. Decoder
        # Layer 4 (Deepest): Input 768 from stage4. Concats with stage3 (384).
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = ConvBlock(768 + 384, 384)
        
        # Layer 3: Input 384 from dec4. Concats with stage2 (192).
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = ConvBlock(384 + 192, 192)
        
        # Layer 2: Input 192 from dec3. Concats with stage1 (96).
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = ConvBlock(192 + 96, 96)
        
        # Layer 1: Input 96 from dec2. No more skip connections from backbone (stage1 is stride 4).
        # We need to upsample 4x to match input resolution.
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.dec1 = ConvBlock(96, 64) # Converting to a smaller channel size for final pred
        
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        # features is a dictionary: {"stage1": ..., "stage4": ...}
        features = self.encoder(x)
        
        # Permute channel-last outputs from Swin backbone to channel-first
        # Swin outputs: (B, H, W, C) -> (B, C, H, W)
        s1 = features["stage1"].permute(0, 3, 1, 2).contiguous()
        s2 = features["stage2"].permute(0, 3, 1, 2).contiguous()
        s3 = features["stage3"].permute(0, 3, 1, 2).contiguous()
        s4 = features["stage4"].permute(0, 3, 1, 2).contiguous()
        
        # Decoder
        
        # Block 4
        x = self.up4(s4)
        # Handle small size mismatches if any (though usually inputs are power of 2)
        if x.shape[-2:] != s3.shape[-2:]:
            x = F.interpolate(x, size=s3.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s3], dim=1)
        x = self.dec4(x)
        
        # Block 3
        x = self.up3(x)
        if x.shape[-2:] != s2.shape[-2:]:
            x = F.interpolate(x, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s2], dim=1)
        x = self.dec3(x)
        
        # Block 2
        x = self.up2(x)
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, s1], dim=1)
        x = self.dec2(x)
        
        # Block 1 (Final Upsample back to original resolution)
        x = self.up1(x)
        x = self.dec1(x)
        
        outputs = self.final_conv(x)
        return outputs
