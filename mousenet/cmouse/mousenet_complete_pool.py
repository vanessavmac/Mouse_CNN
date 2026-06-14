from copyreg import pickle
import torch
from torch import nn
import networkx as nx
import numpy as np
import pathlib, os
import pickle
from .exps.imagenet.config import  INPUT_SIZE, EDGE_Z, OUTPUT_AREAS, HIDDEN_LINEAR, NUM_CLASSES
import pdb

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
    def __init__(self, in_channels, out_channels, kernel_size, gsh, gsw, mask=3, stride=1, padding=0):
        super(Conv2dMask, self).__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.mypadding = nn.ConstantPad2d(padding, 0)
        if mask == 0:
            self.mask = None
        if mask==1:
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

class MouseNetCompletePool(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, retinomap=None):
        super(MouseNetCompletePool, self).__init__()
        self.Convs = nn.ModuleDict()
        self.BNs = nn.ModuleDict()
        self.network = network
        self.retinomap = retinomap
        
        G, _ = network.make_graph()
        self.top_sort = list(nx.topological_sort(G))

        for layer in network.layers:
            params = layer.params
            self.Convs[layer.source_name + layer.target_name] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                                    params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
            ## plotting Gaussian mask
            #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
            #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
            if layer.target_name not in self.BNs:
                self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)

        # calculate total size output to classifier
        total_size=0
        
        for area in OUTPUT_AREAS:
            layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
            total_size += int(16*layer.params.out_channels)
        #     if area =='VISp5':
        #         layer = network.find_conv_source_target('VISp2/3','VISp5')
        #         visp_out = layer.params.out_channels
        #         # create 1x1 Conv downsampler for VISp5
        #         visp_downsample_channels = visp_out
        #         ds_stride = 2
        #         self.visp5_downsampler = nn.Conv2d(visp_out, visp_downsample_channels, 1, stride=ds_stride)
        #         total_size += INPUT_SIZE[1]/ds_stride * INPUT_SIZE[2]/ds_stride * visp_downsample_channels
        #     else:
        #         layer = network.find_conv_source_target('%s2/3'%area[:-1],'%s'%area)
        #         total_size += int(layer.out_size*layer.out_size*layer.params.out_channels)
        
        # self.classifier = nn.Sequential(
            # nn.Linear(int(total_size), NUM_CLASSES),
            # nn.Linear(int(total_size), HIDDEN_LINEAR),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(HIDDEN_LINEAR, HIDDEN_LINEAR),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(HIDDEN_LINEAR, NUM_CLASSES),
        # )

    def get_img_feature(self, x, area_list, flatten=False, no_pooling=False, return_signals=False):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the (flatten/unflatten) activation of that area
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}

        signals = {
            'regions': {},
            'projections': {},
        }

        for area in self.top_sort:
            if area == 'input':
                continue
   
            if area == 'LGNd' or area == 'LGNv':
                layer = self.network.find_conv_source_target('input', area)
                layer_name = layer.source_name + layer.target_name
                conv_out = self.Convs[layer_name](x)
                calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](conv_out))

                if return_signals:
                    signals['regions'][area] = calc_graph[area]
                    signals['projections'][area] = {'input': conv_out}
                continue

            proj_conv_outs = {}
            for layer in self.network.layers:
                if layer.target_name == area:
                    layer_name = layer.source_name + layer.target_name
                    conv_out = self.Convs[layer_name](calc_graph[layer.source_name])
                    proj_conv_outs[layer.source_name] = conv_out

                    if area not in calc_graph:
                        calc_graph[area] = conv_out
                    else:
                        calc_graph[area] = calc_graph[area] + conv_out

            calc_graph[area] = nn.ReLU(inplace=True)(
                self.BNs[area](
                    calc_graph[area]
                )
            )

            if return_signals:
                signals['regions'][area] = calc_graph[area]
                signals['projections'][area] = {src: co for src, co in proj_conv_outs.items()}

        if len(area_list) == 0:
            area_list = self.top_sort[1:] # if no area specified, return all areas except input

        if return_signals:
            def _detach(d):
                if isinstance(d, torch.Tensor):
                    return d.detach()
                return {k: _detach(v) for k, v in d.items()}
            signals = _detach(signals)

        if len(area_list) == 1:
            result = torch.flatten(calc_graph[area_list[0]], 1) if flatten else calc_graph[area_list[0]]
            return (result, signals) if return_signals else result

        elif no_pooling:
            re = {area: calc_graph[area] for area in area_list}
            return (re, signals) if return_signals else re

        else:
            re = None
            for area in area_list:
                pooled = torch.nn.AdaptiveAvgPool2d(4)(calc_graph[area])
                re = pooled if re is None else torch.cat([pooled, re], axis=1)
            return (re, signals) if return_signals else re

    def forward(self, x):
        x = self.get_img_feature(x, OUTPUT_AREAS, flatten=False)
        # x = self.classifier(x)
        return x
