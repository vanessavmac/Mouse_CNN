import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import pathlib
from .exps.imagenet.config import EDGE_Z

class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw, out_sigma, dilation=1):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param gsh: Gaussian height for generating Gaussian mask 
        :param gsw: Gaussian width for generating Gaussian mask
        :param out_sigma: ratio between output size and input size, 1/2 means reduce output size to 1/2 of the input size
        """

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.gsh = gsh
        self.gsw = gsw
        self.kernel_size = 2*int(self.gsw * EDGE_Z) + 1

        KmS = int((self.kernel_size-1/out_sigma))
        if np.mod(KmS,2)==0:
            padding = int(KmS/2)
        else:
            padding = (int(KmS/2), int(KmS/2+1), int(KmS/2), int(KmS/2+1))

        if dilation > 1:
            assert int(1/out_sigma) == 1, "Currently only support dilation when out_sigma is 1."
            # Adjust padding for dilation
            effective_kernel_size = self.kernel_size + (self.kernel_size - 1) * (dilation - 1)
            KmS = int(effective_kernel_size - 1 / out_sigma)
            if np.mod(KmS, 2) == 0:
                padding = int(KmS / 2)
            else:
                padding = (int(KmS / 2), int(KmS / 2 + 1), int(KmS / 2), int(KmS / 2 + 1))

        self.padding = padding
        self.stride = int(1/out_sigma)
        
class ConvLayer:
    def __init__(self, source_name, target_name, params, out_size):
        """
        :param params: ConvParam containing the parameters of the layer
        :param source_name: name of the source area, e.g. VISp4, VISp2/3, VISp5
        :param target_name: name of the target area
        :param out_size: output size of the layer
        """
        self.params = params
        self.source_name = source_name
        self.target_name = target_name
        self.out_size = out_size

class NonConvParam:
    def __init__(self, out_channels):
        self.out_channels = out_channels

class NonConvLayer:
    def __init__(self, params, source_name, target_name, layer, out_size):
        """
        :param params: NonConvParam containing the parameters of the layer
        :param layer: layer object (make sure this is a torch layer nn.Module) that can be called in forward pass
        """
        self.params = params
        self.layer = layer
        self.source_name = source_name
        self.target_name = target_name
        self.out_size = out_size


def get_retinotopic_mask(layer, retinomap):
    region_name = ''.join(x for x in layer.lower() if x.isalpha())
    mask = torch.zeros(32, 32)
    if layer == "input":
        return
    if region_name == "visp":
        return 1

    for area in retinomap:
        area_name = area[0].lower()
        if area_name == region_name:
            normalized_polygon = area[1]
            x, y = normalized_polygon.exterior.coords.xy
            x, y = list(x), list(y)
            xshift= yshift = int(0)
            if area_name != "visp":
                xshift = int((max(x) - min(x))/4)
                yshift = int((max(y) - min(y))/4)
            x1, x2 = int(max(min(x)+xshift, 0)), int(min(max(x) - xshift, 32))
            y1, y2 = int(max(min(y) + yshift, 0)), int(min(max(y) - yshift, 32))
            mask[x1:x2, y1:y2] = 1
            mask_sum = mask.sum()
            project_root = pathlib.Path(__file__).parent.parent.resolve()
            file = os.path.join(project_root, "retinotopics", "mask_areas", f"{area_name}.pkl")
            pickle.dump(mask_sum, open(file,"wb"))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            mask.to(device)
            return mask

    # raise ValueError(f"Could not find area for layer {layer} in retinomap")


class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask 
    """
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0, dilation=1):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation)
        self.mypadding = nn.ConstantPad2d(padding, 0)
        if gsh == 0 or gsw == 0 or mask == 0:
            self.mask = None # special case, kernel size is 1, so no Gaussian mask applied
        elif mask==1:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)))
        elif mask ==2:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask(gsh, gsw)), requires_grad=False) 
        elif mask ==3:
            self.mask = nn.Parameter(torch.Tensor(self.make_gaussian_kernel_mask_vary_channel(gsh, gsw, kernel_size, out_channels, in_channels)), requires_grad=False)
        else:
            assert("mask should be 0, 1, 2, 3!")

    def forward(self, input):
        if self.mask is not None:
            return super(Conv2dMask, self)._conv_forward(self.mypadding(input), self.weight*self.mask, self.bias)
        else:
            return super(Conv2dMask, self)._conv_forward(self.mypadding(input), self.weight, self.bias)
            
    def make_gaussian_kernel_mask(self, peak, sigma):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma*EDGE_Z)        
        x = np.arange(-width, width+1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X**2 + Y**2)

        probability = peak * np.exp(-radius**2/2/sigma**2)

        re = np.random.rand(len(x), len(x)) < probability
        # plt.imshow(re, cmap='Greys')
        return re
    
    def make_gaussian_kernel_mask_vary_channel(self, peak, sigma, kernel_size, out_channels, in_channels):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :param kernel_size: kernel size of the conv2d 
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak, sigma)
        return re