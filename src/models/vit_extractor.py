import torch
import torch.nn as nn
import timm

class FrozenViTExtractor(nn.Module):
    """
    Loads pre-trained Vision Transformer (vit_base_patch16_224),
    removes classification head, freezes all parameters,
    and extracts global 768-D representation vectors.
    """
    def __init__(self, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.backbone(x)
