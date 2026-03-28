import torch
import torch.nn as nn
from .conv import Conv2dMask, ConvParam

# Based on WF RF sizes in https://www.sciencedirect.com/science/article/pii/S0960982219313235
KERNEL_SIZES = [5, 5, 5]
DILATIONS = [6, 7, 9]
EFFECTIVE_RF_SIZE = [ kernel_size + (kernel_size - 1) * (dilation - 1) for kernel_size, dilation in zip(KERNEL_SIZES, DILATIONS) ]
assert EFFECTIVE_RF_SIZE == [25, 29, 37], f"Expected effective receptive field sizes of [25, 29, 37], but got {EFFECTIVE_RF_SIZE}"

class WideFieldCells(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WideFieldCells, self).__init__()
        """
        Models the WF cells which begin in sSC and project to the LP.
        3 parallel dilated convolutions to model the range of RF receptive field sizes.
    
        The input and output feature maps are the same size, 
        so we use a stride of 1 and appropriate padding.
        """
        if isinstance(out_channels, float):
            assert out_channels.is_integer(), "out_channels must be an integer or a float with no decimal part"
            out_channels = int(out_channels)

        # Divide output channels equally among three dilated convolutions
        out_channels_d1 = int(out_channels // 3)
        out_channels_d2 = int(out_channels // 3)
        out_channels_d3 = int(out_channels - out_channels_d1 - out_channels_d2)

        # Rearrange so largest number of out_channels is assigned to out_channels_d2 (this is the median RF size)
        max_out_channels = max(out_channels_d1, out_channels_d2, out_channels_d3)
        if out_channels_d1 == max_out_channels:
            out_channels_d1, out_channels_d2, out_channels_d3 = out_channels_d2, out_channels_d1, out_channels_d3
        elif out_channels_d3 == max_out_channels:
            out_channels_d1, out_channels_d2, out_channels_d3 = out_channels_d1, out_channels_d3, out_channels_d2

        print(f"sSC --> LP WF Cells initialized with in_channels: {in_channels}, out_channels: {out_channels}")
        
        gsw = (KERNEL_SIZES[0] - 1) // 2
        conv_d1_params = ConvParam(in_channels=in_channels, out_channels=out_channels_d1, gsh=1, gsw=gsw, out_sigma=1, dilation=DILATIONS[0])
        self.conv_d1 = Conv2dMask(conv_d1_params.in_channels, conv_d1_params.out_channels, conv_d1_params.kernel_size, conv_d1_params.gsh, conv_d1_params.gsw, stride=conv_d1_params.stride, padding=conv_d1_params.padding, dilation=DILATIONS[0])
        print(f"sSC --> LP WF Cells conv_d1 (effective RD size = {EFFECTIVE_RF_SIZE[0]}): initialized with out_channels: {out_channels_d1}, kernel_size: {conv_d1_params.kernel_size}, dilation: {DILATIONS[0]}")

        gsw = (KERNEL_SIZES[1] - 1) // 2
        conv_d2_params = ConvParam(in_channels=in_channels, out_channels=out_channels_d2, gsh=1, gsw=gsw, out_sigma=1, dilation=DILATIONS[1])
        self.conv_d2 = Conv2dMask(conv_d2_params.in_channels, conv_d2_params.out_channels, conv_d2_params.kernel_size, conv_d2_params.gsh, conv_d2_params.gsw, stride=conv_d2_params.stride, padding=conv_d2_params.padding, dilation=DILATIONS[1])
        print(f"sSC --> LP WF Cells conv_d2 (effective RD size = {EFFECTIVE_RF_SIZE[1]}): initialized with out_channels: {out_channels_d2}, kernel_size: {conv_d2_params.kernel_size}, dilation: {DILATIONS[1]}")
        
        gsw = (KERNEL_SIZES[2] - 1) // 2
        conv_d3_params = ConvParam(in_channels=in_channels, out_channels=out_channels_d3, gsh=1, gsw=gsw, out_sigma=1, dilation=DILATIONS[2])
        self.conv_d3 = Conv2dMask(conv_d3_params.in_channels, conv_d3_params.out_channels, conv_d3_params.kernel_size, conv_d3_params.gsh, conv_d3_params.gsw, stride=conv_d3_params.stride, padding=conv_d3_params.padding, dilation=DILATIONS[2])
        print(f"sSC --> LP WF Cells conv_d3 (effective RD size = {EFFECTIVE_RF_SIZE[2]}): initialized with out_channels: {out_channels_d3}, kernel_size: {conv_d3_params.kernel_size}, dilation: {DILATIONS[2]}")


    def forward(self, x):
        out_d1 = self.conv_d1(x)
        out_d2 = self.conv_d2(x)
        out_d3 = self.conv_d3(x)

        return torch.cat([out_d1, out_d2, out_d3], dim=1)