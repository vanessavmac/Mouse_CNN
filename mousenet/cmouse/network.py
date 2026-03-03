import numpy as np
import networkx as nx
from .anatomy import gen_anatomy
import torch
from torch import nn
from .exps.imagenet.config import INPUT_SIZE, EDGE_Z, INPUT_GSH, INPUT_GSW, get_out_sigma, DLGN_ON_RGCS_CHANNELS, DLGN_OFF_RGCS_CHANNELS, DLGN_ON_OFF_RGCS_CHANNELS, DLGN_OTHER_RGCS_CHANNELS, SC_ON_RGCS_CHANNELS, SC_OFF_RGCS_CHANNELS, SC_ON_OFF_RGCS_CHANNELS, SC_OTHER_RGCS_CHANNELS
import os
import pickle
import matplotlib.pyplot as plt
import pathlib
import pdb
from .retina import MouseRetinaLayer
from .sSC import MousesSCLayer
from .lp_connections import LP_PATHWAYS, SFTLayer

class ConvParam:
    def __init__(self, in_channels, out_channels, gsh, gsw, out_sigma):
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

class Network:
    """
    network class that contains all conv paramters needed to construct torch model.
    """
    def __init__(self, retinotopic=False):
        self.layers = []
        self.area_channels = {}
        self.area_size = {}
        self.retinotopic = retinotopic

    def find_layer_by_source_target(self, source_name, target_name, layer_type):
        for layer in self.layers:
            if layer.source_name == source_name and layer.target_name == target_name and layer.__class__.__name__ == layer_type:
                return layer
        raise ValueError(f"No layer found with source {source_name} and target {target_name} and type {layer_type}")
    
    def find_conv_target_area(self, target_name):
        for layer in self.layers:
            if layer.target_name == target_name and layer.__class__.__name__ == ConvLayer.__name__:
                return layer
        raise ValueError(f"No conv layer found with target {target_name}")
        
    def construct_from_anatomy(self, anet, architecture):
        """
        construct network from anatomy 
        :param anet: anatomy class which contains anatomical connections
        :param architecture: architecture class which calls set_num_channels for calculating connection strength
        """
        self.area_channels['input'] = INPUT_SIZE[0]
        self.area_size['input'] = INPUT_SIZE[1]


        ############################################
        # construct RGCs -> sSC
        sSC_projecting_rgcs_layer = MouseRetinaLayer(
                num_rgb_dog_output_channels=(SC_ON_RGCS_CHANNELS, SC_OFF_RGCS_CHANNELS, SC_ON_OFF_RGCS_CHANNELS, SC_OTHER_RGCS_CHANNELS),
                rgb_kernel_size=9
            )
        out_channels = sSC_projecting_rgcs_layer.out_channels
        
        sSC_projecting_rgcs = NonConvLayer(
            params=NonConvParam(out_channels=out_channels),
            layer=sSC_projecting_rgcs_layer, 
            source_name='input', target_name='RGCsSC', out_size=INPUT_SIZE[1]
        )
        architecture.set_num_channels('RGCsSC', '', out_channels)
        self.area_channels['RGCsSC'] = out_channels
        self.area_size['RGCsSC'] = INPUT_SIZE[1]
        self.layers.append(sSC_projecting_rgcs)

        # Wide-field cells take direct input from RGCs and project to the LP
        out_sigma = 1
        out_size = INPUT_SIZE[1] * out_sigma
        sSC_out_channels = int(np.floor(anet.find_layer('sSC','').num/out_sigma/INPUT_SIZE[1]/INPUT_SIZE[2]))
        sSC_layer = MousesSCLayer(in_channels=self.area_channels['RGCsSC'], out_channels=sSC_out_channels)
        sSC_nonconv_layer = NonConvLayer(
            params=NonConvParam(out_channels=sSC_out_channels),
            layer=sSC_layer,
            source_name='RGCsSC', target_name='sSC', out_size=out_size
        )
        architecture.set_num_channels('sSC', '', sSC_out_channels)
        self.area_channels['sSC'] = sSC_out_channels
        self.area_size['sSC'] = out_size
        self.layers.append(sSC_nonconv_layer)


        ############################################
        # construct RGCs → dLGN
        # NOTE: RGCs in the retina send their axons directly to synapse onto dLGN relay 
        # (thalamocortical) neurons, which then project to cortical layer 4
        # (SOURCE: https://pmc.ncbi.nlm.nih.gov/articles/PMC6380502/)     
        dLGN_projecting_rgcs_layer = MouseRetinaLayer(
                num_rgb_dog_output_channels=(DLGN_ON_RGCS_CHANNELS, DLGN_OFF_RGCS_CHANNELS, DLGN_ON_OFF_RGCS_CHANNELS, DLGN_OTHER_RGCS_CHANNELS), 
                rgb_kernel_size=9
            )   
        out_channels = dLGN_projecting_rgcs_layer.out_channels

        dLGN_projecting_rgcs = NonConvLayer(
            params=NonConvParam(out_channels=out_channels),
            layer=dLGN_projecting_rgcs_layer, 
            source_name='input', target_name='RGCdLGN', out_size=INPUT_SIZE[1]
        )
        
        architecture.set_num_channels('RGCdLGN', '', out_channels)
        self.area_channels['RGCdLGN'] = out_channels
        self.area_size['RGCdLGN'] = INPUT_SIZE[1]
        self.layers.append(dLGN_projecting_rgcs)

        # Use a 1x1 conv to model dLGN relay neurons (model only the excitatory neurons)
        out_sigma = 1
        out_size = INPUT_SIZE[1] * out_sigma
        dLGN_out_channels = int(np.floor(anet.find_layer('LGNd','').num/out_sigma/INPUT_SIZE[1]/INPUT_SIZE[2]))       
        convlayer = ConvLayer(
            params=ConvParam(
                in_channels=self.area_channels['RGCdLGN'], 
                out_channels=dLGN_out_channels,
                gsh=INPUT_GSH,
                gsw=0, out_sigma=out_sigma
            ), # gsw=0 means kernel size of 1 since RGCdLGN already account for complete receptive field size
            source_name='RGCdLGN', target_name='LGNd', out_size=out_size,
        )   
        architecture.set_num_channels('LGNd', '', dLGN_out_channels)
        self.area_channels['LGNd'] = dLGN_out_channels
        self.area_size['LGNd'] = out_size
        self.layers.append(convlayer)


        ############################################
        # construct conv layers for all other connections
        G, _ = anet.make_graph()
        Gtop = nx.topological_sort(G)
        root = next(Gtop) # get root of graph

        # DEBUGGING: NEATLY PRINT TOPOLOGICAL SORTING
        print("DEBUG: TOPOLOGICAL SORTING OF ANATOMICAL NET:")
        for i, node in enumerate(Gtop):
            print(f"{i}: {node.area} {node.depth}")

        for i, e in enumerate(nx.edge_bfs(G, root)):
            
            in_layer_name = e[0].area+e[0].depth
            out_layer_name = e[1].area+e[1].depth
            print('constructing layer %s: %s to %s'%(i, in_layer_name, out_layer_name))
            
            in_conv_layer = self.find_conv_target_area(in_layer_name)
            in_size = in_conv_layer.out_size
            in_channels = in_conv_layer.params.out_channels
            
            out_anat_layer = anet.find_layer(e[1].area, e[1].depth)
            
            out_sigma = get_out_sigma(e[0].area, e[0].depth, e[1].area, e[1].depth)
            out_size = in_size * out_sigma
            self.area_size[e[1].area+e[1].depth] = out_size
            out_channels = int(np.floor(out_anat_layer.num/out_size**2))
            if self.retinotopic:
                project_root = pathlib.Path(__file__).parent.parent.resolve()
                mask_pickle = ''.join(x for x in in_layer_name.lower() if x.isalpha())
                mask_path = os.path.join(project_root, "retinotopics", "mask_areas", f"{mask_pickle}.pkl")
                if os.path.exists(mask_path):
                    mask_size = pickle.load(open(mask_path, "rb"))
                    out_channels = out_channels*int((32*32)/mask_size)


            
            architecture.set_num_channels(e[1].area, e[1].depth, out_channels)
            self.area_channels[e[1].area+e[1].depth] = out_channels
            
            convlayer = ConvLayer(in_layer_name, out_layer_name, 
                                  ConvParam(in_channels=in_channels, 
                                            out_channels=out_channels,
                                        gsh=architecture.get_kernel_peak_probability(e[0].area, e[0].depth, e[1].area, e[1].depth),
                                        gsw=architecture.get_kernel_width_pixels(e[0].area, e[0].depth, e[1].area, e[1].depth), out_sigma=out_sigma),
                                    out_size)
            
            self.layers.append(convlayer)

        ############################################
        # construct all modulatory SFT connections through the LP
        # a single layer is created for each target area that receives modulatory input from the LP, 
        # and the conditioning input to this layer is the concatenated feature maps from all source areas in the pathway.
        for pathway in LP_PATHWAYS:
            print(f"Constructing SFT layer for LP pathway with target {pathway['target']} and conditioning sources {pathway['sources']}")
            conditioning_area_names = [area + depth if depth is not None else area for area, depth in pathway['sources']]
            target_area_name = pathway['target'][0] + pathway['target'][1]
            
            num_source_channels = sum([self.area_channels[source_area] for source_area in conditioning_area_names])
            assert all(source_area in self.area_channels for source_area in conditioning_area_names), \
                f"Some source areas not found in area_channels: {[s for s in conditioning_area_names if s not in self.area_channels]}"

            print(f"the area size of the target area {target_area_name} is {self.area_size[target_area_name]}")
            print(f"the area of the source areas {conditioning_area_names} are {[self.area_size[source_area] for source_area in conditioning_area_names]}")

            min_input_map_size = min([self.area_size[source_area] for source_area in conditioning_area_names])
            max_input_map_size = max([self.area_size[source_area] for source_area in conditioning_area_names])

            if min_input_map_size != max_input_map_size:
                print(f"Input maps for SFT modulation of {target_area_name} have different spatial sizes {min_input_map_size} vs {max_input_map_size}.")
                print(f"We assume that they will be adjusted to the smallest size {min_input_map_size} and then concatenated before being passed to the SFT layer.")

            # Calculate the out_sigma for the SFT layer based on the target area size and the input feature map size
            out_sigma = self.area_size[target_area_name] / min_input_map_size
            print(f"Calculated out_sigma for SFT layer targeting {target_area_name} is {out_sigma}")
            assert out_sigma in [0.5, 1], f"Calculated out_sigma {out_sigma} is not valid. Expected 0.5 or 1, corresponding to feature maps of size 32 or 64."

            sft_layer = NonConvLayer(
                params = None,
                out_size = None,
                source_name = conditioning_area_names,
                target_name = target_area_name,
                layer = SFTLayer(
                    target_area_name=target_area_name,
                    in_channels=num_source_channels, 
                    out_channels=self.area_channels[target_area_name],
                    # @TODO ASK TRIPP, to match mousenet, I used stride since i know feature maps either 64 or 32 (note that HVAs never modulate VISp/dLGN)
                    stride = int(1/out_sigma),
                ),
            )
            self.layers.append(sft_layer)
            
    def make_graph(self):
        """
        produce networkx graph
        """
        G = nx.DiGraph()
        edges = []
        for p in self.layers:
            if isinstance(p.source_name, list):
                # SFT layers have multiple sources
                for source in p.source_name:
                    edges.append((source, p.target_name))
            else:
                edges.append((p.source_name, p.target_name))
        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = { layer:'%s\n%s'%(layer, int(self.area_channels[layer])) for layer in G.nodes()}
        return G, node_label_dict

    def draw_graph(self, node_size=2000, node_color='yellow', edge_color='red'):
        """
        draw the network structure
        """
        # TODO: this only works for conv layers; need to add non-conv layers to drawing
        G, node_label_dict = self.make_graph()
        edge_label_dict = {(c.source_name, c.target_name):(c.params.kernel_size) for c in self.layers}
        plt.figure(figsize=(12,12))
        pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        nx.draw(G, pos, node_size=node_size, node_color=node_color, edge_color=edge_color,alpha=0.4)
        nx.draw_networkx_labels(G, pos, node_label_dict, font_size=10,font_weight=640, alpha=0.7, font_color='black')
        nx.draw_networkx_edge_labels(G, pos, edge_label_dict, font_size=20, font_weight=640,alpha=0.7, font_color='red')
        plt.show()  


def gen_network_from_anatomy(architecture):
    anet = gen_anatomy(architecture)
    net = Network()
    net.construct_from_anatomy(anet, architecture)
    return net

def save_network_to_pickle(net, file_path):
    f = open(file_path,'wb')
    pickle.dump(net, f)

def load_network_from_pickle(file_path):
    f = open(file_path,'rb')
    net = pickle.load(f)
    return net

def gen_network(net_name, architecture):
    file_path = './myresults/%s.pkl'%net_name
    if os.path.exists(file_path):
        net = load_network_from_pickle(file_path)
    else:
        net = gen_network_from_anatomy(architecture)
        if not os.path.exists('./myresults'):
            os.mkdir('./myresults')
        save_network_to_pickle(net, file_path)
    return net
