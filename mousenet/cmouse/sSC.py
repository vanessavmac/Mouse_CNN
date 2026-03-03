import torch
import torch.nn as nn

class MousesSCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(MousesSCLayer, self).__init__()
        """
        The superficial SC is modelled as 3 parallel convolutions with 
        different dilations (d=1, 2, 3) to represent wide-field cells.

        NOTE: The input and output feature maps of the sSC layer are the same size, so we use a stride of 1 and appropriate padding.

        Kernel size is 9 based on:

            A. E. Allen, C. A. Procyk, M. Howarth, L. Walmsley, and T. M. Brown, “Visual input 
            to the mouse lateral posterior and posterior thalamic nuclei: photoreceptive origins 
            and retinotopic order,” The Journal of Physiology, vol. 594, no. 7, pp. 1911–1929, 
            Apr. 2016, doi: 10.1113/JP271707.

        :param in_channels: number of input channels from the retina (RGCs)
        :param out_channels: each dilated convolution outputs out_channels // 3 channels, so total output channels is out_channels
        :param kernel_size
        """
        if isinstance(out_channels, float):
            assert out_channels.is_integer(), "out_channels must be an integer or a float with no decimal part"
            out_channels = int(out_channels)

        # TODO ASK TRIPP: note, i made this assumption, d1/d2/d3 represent median plus/minus std of RF sizes
        # Divide output channels equally among three dilated convolutions
        out_channels_d1 = int(out_channels // 3)
        out_channels_d2 = int(out_channels // 3)
        out_channels_d3 = int(out_channels - out_channels_d1 - out_channels_d2)
        
        print(f"sSC initialized with in_channels: {in_channels}, out_channels: {out_channels} "
              f"(dilation 1: {out_channels_d1}, dilation 2: {out_channels_d2}, dilation 3: {out_channels_d3}), "
              f"kernel_size: {kernel_size}")
        
        # TODO ASK TRIPP: Should I make this sparse Conv2d? Dilated convs model model varying receptive fields of wide-field cells
        # which take inputs from RGCs and project to the LP. Similar to dLGN, should I use hard coded values?
        # (hard coded values based on kernel size of 9)
        # INPUT_GSH = 1 #Gaussian height of input to LGNv 
        # INPUT_GSW = 4 #Gaussian width of input to LGNv
        # ACTUALLY YES DO THIS
        self.conv_d1 = nn.Conv2d(in_channels, out_channels_d1, kernel_size=kernel_size, padding=kernel_size//2, dilation=1)
        self.conv_d2 = nn.Conv2d(in_channels, out_channels_d2, kernel_size=kernel_size, padding=kernel_size//2*2, dilation=2)
        self.conv_d3 = nn.Conv2d(in_channels, out_channels_d3, kernel_size=kernel_size, padding=kernel_size//2*3, dilation=3)



    def forward(self, x):
        out_d1 = self.conv_d1(x)
        out_d2 = self.conv_d2(x)
        out_d3 = self.conv_d3(x)

        return torch.cat([out_d1, out_d2, out_d3], dim=1)