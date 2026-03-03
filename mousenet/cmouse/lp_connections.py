import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO ASK TRIPP: about additional slide "For other regions without data, I will modulate all layers (2/3, 4, and 5)."?
LP_PATHWAYS = [
    {
        'sources': [('sSC', None), ('VISp', '5')],
        'target': ('VISl', '4'),
    },
    {
        'sources': [('sSC', None), ('VISl', '5')],
        'target': ('VISal', '2/3'),
    },
    {
        'sources': [('sSC', None), ('VISl', '5')],
        'target': ('VISal', '4'),
    },
        {
        'sources': [('sSC', None), ('VISl', '5')],
        'target': ('VISal', '5'),
    },
]

# TODO ASK TRIPP: what kernel size for the conditioning network (I just defaulted to 3?) and if it should be sparse Conv2d?
# because how would we parameterize fan in connections of disynaptic LP pathways since we're using SFT?
class SFTLayer(nn.Module):
    """
    Implement Spatial Feature Transform (SFT) layer.

    Stride ensures that the conditioning feature maps are appropriately downsampled to match the target feature map size.
    TODO ASK TRIPP: I assumed fixed kernel size of 3 (just a random number...)
    """

    def __init__(self, target_area_name, in_channels, out_channels, stride, kernel_size=3):
        super(SFTLayer, self).__init__()
        self.name = target_area_name # helpful for debugging

        # Ensure the scale and bias produced are the same size as the input x
        self.scale_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)
        self.bias_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=stride)

    def forward(self, cond, x):
        print(f"SFTLayer targetting {self.name}: Conditioning input has dimension {cond.shape}, feature map input has dimension {x.shape}")
        
        scale = self.scale_conv(cond)
        bias = self.bias_conv(cond)
        
        assert scale.shape == x.shape, f"Scale shape {scale.shape} does not match input feature map shape {x.shape}"
        assert bias.shape == x.shape, f"Bias shape {bias.shape} does not match input feature map shape {x.shape}"

        return x * scale + bias