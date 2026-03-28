import numpy as np
import networkx as nx
from .anatomy import gen_anatomy
import os

from .exps.imagenet.config import (
    DLGN_GSW,
    SSC_GSW,
    DLGN_OFF_RGCS_CHANNELS,
    DLGN_ON_OFF_RGCS_CHANNELS,
    DLGN_ON_RGCS_CHANNELS,
    DLGN_OTHER_RGCS_CHANNELS,
    GSH_1,
    INPUT_SIZE,
    LP_OUTPUT_SIZE,
    LP_PATHWAYS,
    LP_RF_SIZE,
    SC_OFF_RGCS_CHANNELS,
    SC_ON_OFF_RGCS_CHANNELS,
    SC_ON_RGCS_CHANNELS,
    SC_OTHER_RGCS_CHANNELS,
    get_lp_target_areas,
    get_out_sigma,
    get_targets_of_source,
    get_unique_lp_sources,
)
import pickle
import matplotlib.pyplot as plt
import pathlib
from .retina import MouseRetinaLayer
from .wfcells import WideFieldCells
from .conv import ConvLayer, LPConvInputMultipleTargets, CustomConvLayer, ConvParam, LPCustomConvInputMultipleTargets, CustomConvParam

class Network:
    """
    network class that contains all conv paramters needed to construct torch model.
    """
    def __init__(self, retinotopic=False):
        self.layers = []
        self.area_channels = {}
        self.area_size = {}
        self.retinotopic = retinotopic

    def find_layer_by_source_target(self, source_name, target_name, layer_type=None):
        for layer in self.layers:
            if layer.source_name == source_name and layer.target_name == target_name and (layer_type is None or layer.__class__.__name__ == layer_type):
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
        print("Constructing RGCs to sSC pathway...")
        sSC_projecting_rgcs_layer = MouseRetinaLayer(
                num_rgb_dog_output_channels=(SC_ON_RGCS_CHANNELS, SC_OFF_RGCS_CHANNELS, SC_ON_OFF_RGCS_CHANNELS, SC_OTHER_RGCS_CHANNELS),
                rgb_kernel_size=int(SSC_GSW * 2 + 1) # NOTE: slide 22 in 499 presentation
            )
        out_channels = sSC_projecting_rgcs_layer.out_channels
        
        sSC_projecting_rgcs = CustomConvLayer(
            params=CustomConvParam(out_channels=out_channels),
            layer=sSC_projecting_rgcs_layer, 
            source_name='input', target_name='RGCsSC', out_size=INPUT_SIZE[1]
        )
        architecture.set_num_channels('RGCsSC', '', out_channels)
        self.area_channels['RGCsSC'] = out_channels
        self.area_size['RGCsSC'] = INPUT_SIZE[1]
        self.layers.append(sSC_projecting_rgcs)

        # Use a 1x1 conv to model sSC neurons (model only the excitatory neurons)
        out_sigma = 1/2 # NOTE: for simplicity apply stride of 2, since SC is modulated by VISp which is half the size of the input image
        out_size = INPUT_SIZE[1] * out_sigma
        sSC_out_channels = int(np.floor(anet.find_layer('sSC','').num/out_size/out_size))
        convlayer = ConvLayer(
            params=ConvParam(
                in_channels=self.area_channels['RGCsSC'], 
                out_channels=sSC_out_channels,
                gsh=GSH_1,
                gsw=0, out_sigma=out_sigma
            ), # gsw=0 means kernel size = 1 since RGCsSC already account for complete receptive field size
            source_name='RGCsSC', target_name='sSC', out_size=out_size,
        )   
        architecture.set_num_channels('sSC', '', sSC_out_channels)
        self.area_channels['sSC'] = sSC_out_channels
        self.area_size['sSC'] = out_size
        self.layers.append(convlayer)


        ############################################
        # construct RGCs → dLGN
        # NOTE: RGCs in the retina send their axons directly to synapse onto dLGN relay 
        # (thalamocortical) neurons, which then project to cortical layer 4
        # (SOURCE: https://pmc.ncbi.nlm.nih.gov/articles/PMC6380502/)    
        print("Constructing RGCs to dLGN pathway...") 
        dLGN_projecting_rgcs_layer = MouseRetinaLayer(
                num_rgb_dog_output_channels=(DLGN_ON_RGCS_CHANNELS, DLGN_OFF_RGCS_CHANNELS, DLGN_ON_OFF_RGCS_CHANNELS, DLGN_OTHER_RGCS_CHANNELS), 
                rgb_kernel_size=int(DLGN_GSW * 2 + 1)
            )   
        out_channels = dLGN_projecting_rgcs_layer.out_channels

        dLGN_projecting_rgcs = CustomConvLayer(
            params=CustomConvParam(out_channels=out_channels),
            layer=dLGN_projecting_rgcs_layer, 
            source_name='input', target_name='RGCdLGN', out_size=INPUT_SIZE[1]
        )
        
        architecture.set_num_channels('RGCdLGN', '', out_channels)
        self.area_channels['RGCdLGN'] = out_channels
        self.area_size['RGCdLGN'] = INPUT_SIZE[1]
        self.layers.append(dLGN_projecting_rgcs)

        # Use a 1x1 conv to model dLGN neurons (model only the excitatory neurons)
        out_sigma = 1 # NOTE: apply stride of 1 for simplicity
        out_size = INPUT_SIZE[1] * out_sigma
        dLGN_out_channels = int(np.floor(anet.find_layer('LGNd','').num/out_sigma/INPUT_SIZE[1]/INPUT_SIZE[2]))       
        convlayer = ConvLayer(
            params=ConvParam(
                in_channels=self.area_channels['RGCdLGN'], 
                out_channels=dLGN_out_channels,
                gsh=GSH_1,
                gsw=0, out_sigma=out_sigma
            ), # gsw=0 means kernel size = 1 since RGCdLGN already account for complete receptive field size
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
            
            assert e[1].area != 'LGNd' and e[1].area != 'sSC' and e[1].area != 'LP', "LGNd, sSC, and LP are modelled with custom layers and should not have conv layers constructed for them here."
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
        # All connections to LP should be modelled as separate Conv operators. 
        # To produce a valid topological ordering that preserves modulation of each 
        # feedforward LP pathway, the LP is modelled as multiple areas called
        # LP_{target area to modulate}. We use ConvLayerMultipleSourcesTargets 
        # and CustomConvLayerMultipleSourcesTargets to model how each target area 
        # receives mutliple modulatory inputs.
        lp_target_areas = get_lp_target_areas()
        lp_out_channels = int(np.floor(anet.find_layer('LP', '').num/LP_OUTPUT_SIZE**2))
        for target in lp_target_areas:
            architecture.set_num_channels(f'LP_{target}', '', lp_out_channels)
            self.area_channels[f'LP_{target}'] = lp_out_channels
            self.area_size[f'LP_{target}'] = LP_OUTPUT_SIZE

        # Create a CONV layer from each unique source area to the LP(s) it projects to
        # IMPORTANT NOTE: Not every pair of source and target LP area has a unique conv layer.
        unique_sources = get_unique_lp_sources()
        for source in unique_sources:
            targets = get_targets_of_source(source)
            target_names = [f"LP_{target}" for target in targets]

            if source != "sSC":
                convlayer = LPConvInputMultipleTargets(
                    params=ConvParam(
                        in_channels=self.area_channels[source], 
                        out_channels=lp_out_channels,
                        gsh=GSH_1,
                        gsw=(LP_RF_SIZE - 1) // 2, out_sigma=1/2 if "VISp" in source else 1
                    ), 
                    source_name=source, target_names=target_names, out_size=LP_OUTPUT_SIZE
                )
                self.layers.append(convlayer)
            else:
                # Model WF cells that project from sSC to LP
                wfcells_layer = WideFieldCells(
                    in_channels=self.area_channels['sSC'],
                    out_channels=lp_out_channels,
                )
                custom_convlayer = LPCustomConvInputMultipleTargets(
                    params=CustomConvParam(out_channels=lp_out_channels),
                    layer=wfcells_layer,
                    source_name='sSC', target_names=target_names, out_size=LP_OUTPUT_SIZE
                )
                self.layers.append(custom_convlayer)

        # LP produces 𝛾/ꞵ, 1x1 convolutions reduce the number of channels so that 𝛾/ꞵ matches the target area dimensions
        for pathway in LP_PATHWAYS:
            target_name = pathway['target'][0] + pathway['target'][1]
            # Since we only modulate HVAs, we know they are all modelled by a regular ConvLayer, find_conv_target_area is suitable
            target_conv_layer = self.find_conv_target_area(target_name)
            target_out_channels = target_conv_layer.params.out_channels
            
            print(f"Constructing LP pathway to {target_name} with {target_out_channels} output channels...")
            
            gamma_conditioning_layer = ConvLayer(
                params=ConvParam(
                    in_channels=lp_out_channels,
                    out_channels=target_out_channels,
                    gsh=GSH_1,
                    gsw=0, out_sigma=1
                ),
                source_name=f'LP_{target_name}', target_name=f'SFT_gamma_{target_name}', out_size=self.area_size[target_name]
            )
            self.layers.append(gamma_conditioning_layer)

            beta_conditioning_layer = ConvLayer(
                params=ConvParam(
                    in_channels=lp_out_channels,
                    out_channels=target_out_channels,
                    gsh=GSH_1,
                    gsw=0, out_sigma=1
                ),
                source_name=f'LP_{target_name}', target_name=f'SFT_beta_{target_name}', out_size=self.area_size[target_name]
            )
            self.layers.append(beta_conditioning_layer)

        ############################################
        # construct V1 → sSC
        gamma_conditioning_layer = ConvLayer(
            params=ConvParam(
                in_channels=self.area_channels['VISp5'], 
                out_channels=self.area_channels['sSC'],
                gsh=GSH_1,
                gsw=SSC_GSW, # NOTE: comes from slide 22 in 499 presentation
                out_sigma=1/2
            ),
            source_name='VISp5', target_name='SFT_gamma_sSC', out_size=self.area_size['sSC']
        )
        self.layers.append(gamma_conditioning_layer)
        
        beta_conditioning_layer = ConvLayer(
            params=ConvParam(
                in_channels=self.area_channels['VISp5'], 
                out_channels=self.area_channels['sSC'],
                gsh=GSH_1,
                gsw=SSC_GSW, # NOTE: comes from slide 22 in 499 presentation
                out_sigma=1/2
            ),
            source_name='VISp5', target_name='SFT_beta_sSC', out_size=self.area_size['sSC']
        )
        self.layers.append(beta_conditioning_layer)

    def make_graph(self):
        """
        produce networkx graph
        """
        G = nx.DiGraph()
        edges = []
        for p in self.layers:
            if p.__class__.__name__ == ConvLayer.__name__ or p.__class__.__name__ == CustomConvLayer.__name__:
                edges.append((p.source_name, p.target_name))
            elif p.__class__.__name__ == LPConvInputMultipleTargets.__name__ or p.__class__.__name__ == LPCustomConvInputMultipleTargets.__name__:
                for target in p.target_names:
                    edges.append((p.source_name, target))
            else:
                raise ValueError(f"Unexpected layer type {p.__class__.__name__} when making graph.")
            
            # Complete the SFT pathway by adding edges from the SFT layer to the target area
            if p.__class__.__name__ == ConvLayer.__name__ and ("SFT_gamma_" in p.target_name or "SFT_beta_" in p.target_name):
                target_name = p.target_name.split("_")[-1]
                edges.append((p.target_name, target_name))

        for edge in edges:
            G.add_edge(edge[0], edge[1])
        node_label_dict = { layer: '%s\n%s'%(layer, int(self.area_channels[layer])) if layer in self.area_channels else 'N/A' for layer in G.nodes()}

        return G, node_label_dict

    def draw_graph(self, node_size=2000, node_color='yellow', edge_color='red'):
        """
        generate mermaid diagram code for the network structure
        """
        G, node_label_dict = self.make_graph()
        mermaid_code = "graph TD\n"
        
        # Add node definitions with channel counts
        for node in G.nodes():
            if node in self.area_channels:
                channels = int(self.area_channels[node])
                label = f"{node}<br/>({channels})"
            else:
                label = node
            # Escape special characters for Mermaid
            safe_node = node.replace("-", "_").replace(" ", "_")
            mermaid_code += f'    {safe_node}["{label}"]\n'
        
        # Add edges
        for edge in G.edges():
            source, target = edge
            safe_source = source.replace("-", "_").replace(" ", "_")
            safe_target = target.replace("-", "_").replace(" ", "_")
            mermaid_code += f"    {safe_source} --> {safe_target}\n"
        
        # Save to file
        output_path = './network_graph.md'
        with open(output_path, 'w') as f:
            f.write("# MouseNet Network Architecture\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_code)
            f.write("```\n\n")
            f.write("You can render this diagram at: https://mermaid.live/\n")
        
        print(f"Network diagram saved to {output_path}")
        print(f"To visualize, copy the Mermaid code to https://mermaid.live/")  


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
