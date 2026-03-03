from copyreg import pickle
import torch
from torch import nn
import networkx as nx
import numpy as np
import pathlib, os
import pickle
from .exps.imagenet.config import  INPUT_SIZE, EDGE_Z, OUTPUT_AREAS, HIDDEN_LINEAR, NUM_CLASSES
import pdb
from .network import ConvLayer, NonConvLayer

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

class MouseNetCompletePool(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, retinomap=None):
        super(MouseNetCompletePool, self).__init__()
        self.Convs = nn.ModuleDict()
        self.BNs = nn.ModuleDict()
        self.Retina = nn.ModuleDict()
        self.LP_pathways = nn.ModuleDict()
        self.network = network
        self.retinomap = retinomap
        
        self.LP_pathways_sources = {} # keys: targets, values: list of sources

        G, _ = network.make_graph()
        self.top_sort = list(nx.topological_sort(G))

        for layer in network.layers:
            if layer.__class__.__name__ == ConvLayer.__name__:
                params = layer.params
                self.Convs[layer.source_name + layer.target_name] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                                        params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
                ## plotting Gaussian mask
                #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
                #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))
                if layer.target_name not in self.BNs:
                    self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)
            elif layer.__class__.__name__ == NonConvLayer.__name__:
                params = layer.params
                if layer.target_name == "RGCdLGN" or layer.target_name == "RGCsSC":
                    assert layer.layer.__class__.__name__ == "MouseRetinaLayer", "The only NonConvLayer that can target RGCdLGN or RGCsSC is the retina, but getting %s"%(layer.layer.__class__.__name__)
                    self.Retina[layer.source_name + layer.target_name] = layer.layer
                    
                    if layer.target_name not in self.BNs:
                        self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)
                elif layer.target_name == "sSC":
                    # NOTE technically, the sSC are conv layers but handle it as NonConv since it's a custom module with parallel convolutions
                    assert layer.layer.__class__.__name__ == "MousesSCLayer", "The only NonConvLayer that can target sSC is the sSC layer, but getting %s"%(layer.layer.__class__.__name__)
                    self.Retina[layer.source_name + layer.target_name] = layer.layer
                    
                    if layer.target_name not in self.BNs:
                        self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)
                else:
                    assert layer.layer.__class__.__name__ == "SFTLayer", "Any NonConvLayer that is not the retina must be an SFTLayer representing an LP pathway, but getting %s"%(layer.layer.__class__.__name__)
                    assert type(layer.source_name) == list, "The SFTLayer's source name must be a list"
                    assert layer.target_name not in self.LP_pathways, "Each target area can only have one LP pathway"
                    assert layer.target_name not in self.LP_pathways_sources, "Each target area can only have one LP pathway"
                    
                    self.LP_pathways[layer.target_name] = layer.layer
                    self.LP_pathways_sources[layer.target_name] = layer.source_name

                    # NOTE: self.BNs is not updated SFTLayer only modulate existing ConvLayers and NonConvLayers
            else:
                raise ValueError(f"Layer {layer} is not ConvLayer or NonConvLayer, cannot be added to model.")

    def get_img_feature(self, x, area_list, flatten=False):
        """
        function for get activations from a list of layers for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :param area_list: a list of area names
        :return: if list length is 1, return the (flatten/unflatten) activation of that area
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        calc_graph = {}

        for area in self.top_sort:
            if area == 'LP' or area == 'LPn':
                raise Exception("LP should not be included in area_list as it's a set of pathways.")
            
            if area == 'input':
                continue
            
            # RGC projections to the LGN and SC
            if area == 'RGCdLGN' or area == 'RGCsSC':
                layer = self.network.find_layer_by_source_target('input', area, layer_type=NonConvLayer.__name__)
                layer_name = layer.source_name + layer.target_name
                if area in calc_graph:
                    raise ValueError(f"Area {area} already exists in calc_graph, but this pathway only has 1 possible input.")
                calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](self.Retina[layer_name](x)))
                continue
            
            # Geniculate Pathway
            if area == 'LGNd' or area == 'LGNv':
                layer = self.network.find_layer_by_source_target('RGCdLGN', area, layer_type=ConvLayer.__name__)
                layer_name = layer.source_name + layer.target_name
                if area in calc_graph:
                    raise ValueError(f"Area {area} already exists in calc_graph, but this pathway only has 1 possible input.")
                calc_graph[area] =  nn.ReLU(inplace=True)(self.BNs[area](self.Convs[layer_name](calc_graph[layer.source_name])))
                continue
            
            # Extrageniculate Pathway
            if area == 'sSC':
                layer = self.network.find_layer_by_source_target('RGCsSC', area, layer_type=NonConvLayer.__name__)
                layer_name = layer.source_name + layer.target_name
                if area in calc_graph:
                    raise ValueError(f"Area {area} already exists in calc_graph, but this pathway only has 1 possible input.")
                calc_graph[area] =  nn.ReLU(inplace=True)(self.BNs[area](self.Retina[layer_name](calc_graph[layer.source_name])))
                continue

            # V1/HVAs
            for layer in self.network.layers:
                if layer.target_name == area and layer.__class__.__name__ == ConvLayer.__name__:
                    layer_name = layer.source_name + layer.target_name
                    if area not in calc_graph:
                        calc_graph[area] = self.Convs[layer_name](
                                calc_graph[layer.source_name]
                            )
                    else:
                        calc_graph[area] = calc_graph[area] + self.Convs[layer_name](calc_graph[layer.source_name])

            # LP pathways 
            input_maps = []
            if area in self.LP_pathways_sources.keys():
                source_areas = self.LP_pathways_sources[area]
                print(f"{area} receives LP input from {source_areas}")
                for source_area in source_areas:
                    if source_area in calc_graph:
                        input_maps.append(calc_graph[source_area])
                    else:
                        raise ValueError(f"Source area {source_area} for LP pathway to {area} has not been calculated yet. Check the topological sort order.")
            
            if len(input_maps) > 0:
                # TODO ASK TRIPP: is this reasonable to make feature maps same size using average pooling before modulating? this is because
                # HVAs take disynaptic input from the SC and/or V1 (feature map of size 64); but other HVAs (specifically VISl)
                # is only size 32, and SFT requires the input conditioning inputs to be same size to apply convolution
                print(f"\nApplying SFT modulation for {area} with input from {len(input_maps)} source areas.")
                min_input_map_size = min([input_map.shape[2] for input_map in input_maps]) if len(input_maps) > 0 else None
                max_input_mape_size = max([input_map.shape[2] for input_map in input_maps]) if len(input_maps) > 0 else None

                if min_input_map_size != max_input_mape_size:
                    print(f"Input maps for SFT modulation of {area} have different spatial sizes. Applying adaptive average pooling to match the smallest size {min_input_map_size}.")
                    input_maps = [torch.nn.AdaptiveAvgPool2d(min_input_map_size)(input_map) if input_map.shape[2] != min_input_map_size else input_map for input_map in input_maps]
                    print(f"After pooling, input maps for SFT modulation of {area} have sizes: {[input_map.shape for input_map in input_maps]}")

                # TODO ASK TRIPP, does it make sense to modulate feature maps before applying batch norm and relu? this provides greater control / affects both positive and negative values?
                sft_layer = self.LP_pathways[area]
                calc_graph[area] = sft_layer(torch.cat(input_maps, dim=1), calc_graph[area])
                
            calc_graph[area] = nn.ReLU(inplace=True)(
                self.BNs[area](
                    calc_graph[area]
                )
            )
            # if calc_graph[area].sum() == 0:
            #     pdb.set_trace()
        
        if len(area_list) == 1:
            if flatten:
                return torch.flatten(calc_graph['%s'%(area_list[0])], 1)
            else:
                return calc_graph['%s'%(area_list[0])]

        else:
            re = None
            for area in area_list:
                if re is None:
                    re = torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area])
                    # re = torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1)
                else:
                    re=torch.cat([torch.nn.AdaptiveAvgPool2d(4) (calc_graph[area]), re], axis=1)
                    # re=torch.cat([
                        # torch.flatten(
                        # nn.ReLU(inplace=True)(self.BNs['%s_downsample'%area](self.Convs['%s_downsample'%area](calc_graph[area]))), 
                        # 1), 
                        # re], axis=1)
                # if area == 'VISp5':
                #     re=torch.flatten(self.visp5_downsampler(calc_graph['VISp5']), 1)
                # else:
                #     if re is not None:
                #         re = torch.cat([torch.flatten(calc_graph[area], 1), re], axis=1)
                #     else:
                #         re = torch.flatten(calc_graph[area], 1)
        return re

    def forward(self, x):
        x = self.get_img_feature(x, OUTPUT_AREAS, flatten=False)
        # x = self.classifier(x)
        return x
