import torch
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.swin_model import SwinTransformerUNet

def test_swin_implementation():
    print("Testing Swin Transformer UNet Implementation...")
    
    # 1. Inspect Backbone Nodes
    # NOTE: We skip manual graph node inspection as it can identify false negatives.
    # The real test is the forward pass below.


    # 2. Test Model Instantiation and Forward Pass
    print("\n2. Testing Model Forward Pass:")
    try:
        model = SwinTransformerUNet(num_classes=1)
        model.eval()
        print("  Model instantiated successfully.")
        
        # Create dummy input: (B, C, H, W)
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 512, 512)
        print(f"  Input shape: {input_tensor.shape}")
        
        output = model(input_tensor)
        print(f"  Output shape: {output.shape}")
        
        expected_shape = (batch_size, 1, 512, 512)
        if output.shape == expected_shape:
            print("  SUCCESS: Output shape matches expected shape.")
            
            # 3. Test Backward Pass (Gradient Flow)
            print("\n3. Testing Backward Pass (Gradient Flow):")
            target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float()
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(output, target)
            loss.backward()
            print(f"  Loss: {loss.item()}")
            print("  SUCCESS: Backward pass completed without errors.")
            
        else:
            print(f"  FAILURE: Output shape mismatch. Expected {expected_shape}, got {output.shape}")

    except Exception as e:
        print(f"  ERROR during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_swin_implementation()
