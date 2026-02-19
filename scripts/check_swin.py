
try:
    import torchvision.models as models
    print(f"torchvision.models.swin_t: {hasattr(models, 'swin_t')}")
except Exception as e:
    print(f"torchvision error: {e}")

try:
    from transformers import SwinModel
    print("transformers.SwinModel: Available")
except ImportError:
    print("transformers.SwinModel: Not Available")

try:
    from transformers import SwinForSemanticSegmentation
    print("transformers.SwinForSemanticSegmentation: Available")
except ImportError:
    print("transformers.SwinForSemanticSegmentation: Not Available")

try:
    from transformers import UperNetForSemanticSegmentation
    print("transformers.UperNetForSemanticSegmentation: Available")
except ImportError:
    print("transformers.UperNetForSemanticSegmentation: Not Available")
