import torch
import torch.nn as nn
from .conv import Conv2dMask, ConvParam

# [20] A. E. Allen, C. A. Procyk, M. Howarth, L. Walmsley, and T. M. Brown, “Visual 
# input to the mouse lateral posterior and posterior thalamic nuclei: photoreceptive 
# origins and retinotopic order,” The Journal of Physiology, vol. 594, no. 7, pp. 
# 1911–1929, Apr. 2016, doi: 10.1113/JP271707.
KERNEL_SIZE = 17

# Modulatory pathways modelled (to layer 4 targets, consistent with feedforward and 
# data from [24] N. Zhou, S. P. Masterson, J. K. Damron, W. Guido, and M. E. Bickford, 
# “The Mouse Pulvinar Nucleus Links the Lateral Extrastriate Cortex, Striatum, and Amygdala.,” 
# J Neurosci, vol. 38, no. 2, pp. 347–362, Jan. 2018, doi: 10.1523/JNEUROSCI.1279-17.2017.)

# SC --> LP --> All HVAs
# VISl5 --> LP --> VISrl, VISal
# VISp5 --> LP --> VISl, VISal, VISrl
LP_PATHWAYS = [
    {
        'sources': [('sSC', None), ('VISp', '5')],
        'target': ('VISl', '4'),
    },
    {
        'sources': [('sSC', None), ('VISl', '5'), ('VISp', '5')],
        'target': ('VISrl', '4'),
    },
    {
        'sources': [('sSC', None), ('VISl', '5'), ('VISp', '5')],
        'target': ('VISal', '4'),
    },
    {
        'sources': [('sSC', None)],
        'target': ('VISli', '4'),
    },
    {
        'sources': [('sSC', None)],
        'target': ('VISpl', '4'),
    },
    {
        'sources': [('sSC', None)],
        'target': ('VISpor', '4'),
    },
]

class SFTLayer(nn.Module):
    """
    Implement Spatial Feature Transform (SFT) layer.

    Stride ensures that the conditioning feature maps are appropriately downsampled to match the target feature map size.
    """

    def __init__(self, target_area_name, in_channels, out_channels, out_sigma):
        super(SFTLayer, self).__init__()
        self.name = target_area_name # helpful for debugging

        # Ensure the scale and bias produced are the same size as the input x
        gsw = (KERNEL_SIZE - 1) // 2
        conv_params = ConvParam(in_channels=in_channels, out_channels=out_channels, gsh=1, gsw=gsw, out_sigma=out_sigma)
        
        self.scale_conv = Conv2dMask(conv_params.in_channels, conv_params.out_channels, conv_params.kernel_size, conv_params.gsh, conv_params.gsw, stride=conv_params.stride, padding=conv_params.padding)
        self.bias_conv = Conv2dMask(conv_params.in_channels, conv_params.out_channels, conv_params.kernel_size, conv_params.gsh, conv_params.gsw, stride=conv_params.stride, padding=conv_params.padding)


    def forward(self, cond, x):
        print(f"SFTLayer targetting {self.name}: Conditioning input has dimension {cond.shape}, feature map input has dimension {x.shape}")
        
        scale = self.scale_conv(cond)
        bias = self.bias_conv(cond)
        
        assert scale.shape == x.shape, f"Scale shape {scale.shape} does not match input feature map shape {x.shape}"
        assert bias.shape == x.shape, f"Bias shape {bias.shape} does not match input feature map shape {x.shape}"

        return x * scale + bias