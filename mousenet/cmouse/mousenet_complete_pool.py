from copyreg import pickle
import torch
from torch import nn
import networkx as nx
from .exps.imagenet.config import  OUTPUT_AREAS
from .conv import Conv2dMask, ConvLayer, NonConvLayer

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
                raise Exception("LP should not be included in topological sort as it's a set of pathways.")
            
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
                # print(f"{area} receives LP input from {source_areas}")
                for source_area in source_areas:
                    if source_area in calc_graph:
                        input_maps.append(calc_graph[source_area])
                    else:
                        raise ValueError(f"Source area {source_area} for LP pathway to {area} has not been calculated yet. Check the topological sort order.")
            
            if len(input_maps) > 0:
                # Due to stride of 2 outbound from VISp, this makes feature maps a different size
                # Apply pooling to ensure feature maps are same size before being fed into the conditioning network
                # print(f"\nApplying SFT modulation for {area} with input from {len(input_maps)} source areas.")
                min_input_map_size = min([input_map.shape[2] for input_map in input_maps]) if len(input_maps) > 0 else None
                max_input_mape_size = max([input_map.shape[2] for input_map in input_maps]) if len(input_maps) > 0 else None

                if min_input_map_size != max_input_mape_size:
                    # print(f"Input maps for SFT modulation of {area} have different spatial sizes. Applying adaptive average pooling to match the smallest size {min_input_map_size}.")
                    input_maps = [torch.nn.AdaptiveAvgPool2d(min_input_map_size)(input_map) if input_map.shape[2] != min_input_map_size else input_map for input_map in input_maps]
                    # print(f"After pooling, input maps for SFT modulation of {area} have sizes: {[input_map.shape for input_map in input_maps]}")

                # Modulate feature maps before applying batch norm and relu
                sft_layer = self.LP_pathways[area]
                calc_graph[area] = sft_layer(torch.cat(input_maps, dim=1), calc_graph[area])
                # print(f"{area} was modulated via SFT.")
            else:
                # print(f"{area} does not receive modulatory inputs, skipping SFT modulation.")
                pass

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
