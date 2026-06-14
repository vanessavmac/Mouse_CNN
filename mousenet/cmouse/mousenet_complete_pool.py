from copyreg import pickle
import torch
from torch import nn
import networkx as nx
from .exps.imagenet.config import  OUTPUT_AREAS, get_lp_pathways_description, get_unique_lp_sources
from .conv import Conv2dMask, ConvLayer, CustomConvLayer, LPConvInputMultipleTargets, LPCustomConvInputMultipleTargets

class MouseNetCompletePool(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """
    def __init__(self, network, mask=3, retinomap=None, sft_settings=[], use_normal_init=True):
        super(MouseNetCompletePool, self).__init__()
        self.Convs = nn.ModuleDict()
        # NOTE: No batch norm applied to the generated gamma/beta values, since they are just raw outputs of the SFT generator layers          
        self.BNs = nn.ModuleDict()
        # keys are source areas that project to the LP, values are the convolutional 
        # layers that process the input from that source area before it is sent to the LP
        self.LP_input_layers = nn.ModuleDict()
        self.Retina = nn.ModuleDict()
        self.network = network
        self.retinomap = retinomap
        self.SFT_LNs = nn.ModuleDict()
        self.sft_settings = self._parse_sft_settings(sft_settings)
        print(f"SFT settings: {self.sft_settings}")
        
        if "layernorm" in self.sft_settings:
            self._initialize_sft_layernorms()
        
        G, _ = network.make_graph()
        self.top_sort = list(nx.topological_sort(G))

        for layer in network.layers:
            params = layer.params

            if layer.__class__.__name__ == ConvLayer.__name__:                
                layer_name = layer.source_name + layer.target_name         
                assert layer_name not in self.Convs, f"Layer {layer_name} already exists in Convs, but each layer should only be added once. Check network initialization for duplicate layers with name {layer_name}."

                self.Convs[layer_name] = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                                        params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
                
                if use_normal_init:
                    if "SFT_gamma_" in layer.target_name or "SFT_beta_" in layer.target_name:
                        print(f"Using normal initialization for layer {layer_name} with mean 0 and std 0.01 since use_normal_init is set to True.")

                        nn.init.normal_(self.Convs[layer_name].weight, mean=0.0, std=0.01)
                        if self.Convs[layer_name].bias is not None:
                            nn.init.zeros_(self.Convs[layer_name].bias)
                else:
                    print(f"Using default initialization for layer {layer_name} since use_normal_init is set to False.")
                
                ## plotting Gaussian mask
                #plt.title('%s_%s_%sx%s'%(e[0].replace('/',''), e[1].replace('/',''), params.kernel_size, params.kernel_size))
                #plt.savefig('%s_%s'%(e[0].replace('/',''), e[1].replace('/','')))

                # NOTE: this will skip creating a batch norm after the 1x1 convs (gamma and beta are just raw outputs of the SFT generator layers
                if layer.target_name not in self.BNs:
                    if "SFT_gamma_" in layer.target_name or "SFT_beta_" in layer.target_name:
                        pass
                    else:
                        self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)

            elif layer.__class__.__name__ == CustomConvLayer.__name__:
                assert layer.target_name == "RGCdLGN" or layer.target_name == "RGCsSC", f"CustomConvLayer with target {layer.target_name} and source {layer.source_name} is not supported."

                assert layer.layer.__class__.__name__ == "MouseRetinaLayer", "The only CustomConvLayer that can target RGCdLGN or RGCsSC is the retina, but getting %s"%(layer.layer.__class__.__name__)
                layer_name = layer.source_name + layer.target_name
                self.Retina[layer_name] = layer.layer

                assert layer.target_name not in self.BNs, f"Batch norm for area {layer.target_name} already exists in BNs."
                self.BNs[layer.target_name] = nn.BatchNorm2d(params.out_channels)

            elif layer.__class__.__name__ == LPConvInputMultipleTargets.__name__:
                assert all('LP_' in target_name for target_name in layer.target_names), "All target names for LPCustomConvInputMultipleTargets should contain 'LP_' since they are all targeting the LP, but got target names %s"%(layer.target_names)

                conv_layer = Conv2dMask(params.in_channels, params.out_channels, params.kernel_size,
                                                        params.gsh, params.gsw, stride=params.stride, mask=mask, padding=params.padding)
                
                layer_name = layer.source_name
                assert layer_name not in self.LP_input_layers, f"Layer {layer_name} already exists in LP_inputs, but each layer should only be added once. Check network initialization for duplicate layers with name {layer_name}."
                self.LP_input_layers[layer_name] = conv_layer

                # NOTE: no batch norm is created here; we create a separate batch norm for each LP pathway below

            elif layer.__class__.__name__ == LPCustomConvInputMultipleTargets.__name__:
                assert layer.layer.__class__.__name__ == "WideFieldCells" and layer.source_name == "sSC", "The only CustomConvLayer that can target the LP from sSC is the WideFieldCells, but getting %s and source_name %s"%(layer.layer.__class__.__name__, layer.source_name)
                assert all('LP_' in target_name for target_name in layer.target_names), "All target names for LPCustomConvInputMultipleTargets should contain 'LP_' since they are all targeting the LP, but got target names %s"%(layer.target_names)

                layer_name = layer.source_name
                assert layer_name not in self.LP_input_layers, f"Layer {layer_name} already exists in LP_inputs, but each layer should only be added once. Check network initialization for duplicate layers with name {layer_name}."
                self.LP_input_layers[layer_name] = layer.layer

                # NOTE: no batch norm is created here; we create a separate batch norm for each LP pathway below
                
            else:
                raise ValueError(f"Layer {layer} is not ConvLayer or CustomConvLayer, cannot be added to model.")
        
        # For each LP target_area (which corresponds to a unique LP pathway), create a batch norm layer that acts on the summed source inputs
        lp_out_channels = None
        for area in network.area_channels:
            if 'LP_' in area:
                if lp_out_channels is not None:
                    assert lp_out_channels == network.area_channels[area], f"Expected all LP pathways to have the same number of output channels. Check network initialization and area_channels for areas with 'LP' in their name."
                else:
                    lp_out_channels = network.area_channels[area]

                assert area not in self.BNs
                self.BNs[area] = nn.BatchNorm2d(lp_out_channels) # batch norm for the summed inputs to the LP

        assert len(self.LP_input_layers) == len(get_unique_lp_sources()) and self.LP_input_layers.keys() == set(get_unique_lp_sources()), "Expected 3 unique sources that project to the LP, but got %d. Check get_unique_lp_sources function and network initialization."%len(self.LP_input_layers)

        assert all("SFT" not in bn_key for bn_key in self.BNs.keys()), "Expected no SFT modulation batch norms to be created in BNs."

    def _parse_sft_settings(self, sft_settings):
        # Handle None or empty input
        if sft_settings is None:
            sft_settings = []
        settings = set(sft_settings)
        valid_settings = {"tanh_clamp", "layernorm", "only_beta", "gamma_not_1"}
        unknown_settings = settings - valid_settings
        if len(unknown_settings) != 0:
            raise ValueError(
                f"Unknown sft_settings flags: {sorted(unknown_settings)}. "
                f"Expected subset of {sorted(valid_settings)}."
            )
        return settings

    def _get_or_create_sft_ln(self, signal_name, tensor):
        c, h, w = tensor.shape[1], tensor.shape[2], tensor.shape[3]
        ln_key = f"{signal_name}__{c}_{h}_{w}"
        if ln_key not in self.SFT_LNs:
            self.SFT_LNs[ln_key] = nn.LayerNorm([c, h, w]).to(device=tensor.device, dtype=tensor.dtype)
        return self.SFT_LNs[ln_key]

    def _initialize_sft_layernorms(self):
        for layer in self.network.layers:
            if layer.__class__.__name__ != ConvLayer.__name__:
                continue
            if not ("SFT_gamma_" in layer.target_name or "SFT_beta_" in layer.target_name):
                continue

            target_area = layer.target_name.split("_")[-1]
            c = int(self.network.area_channels[target_area])
            h = int(self.network.area_size[target_area])
            w = int(self.network.area_size[target_area])

            signal_kind = "gamma" if "SFT_gamma_" in layer.target_name else "beta"
            ln_key = f"{target_area}_{signal_kind}__{c}_{h}_{w}"
            if ln_key not in self.SFT_LNs:
                self.SFT_LNs[ln_key] = nn.LayerNorm([c, h, w])

    def _apply_sft_transforms(self, gamma_raw, beta_raw, sft_settings, signal_prefix):
        gamma = gamma_raw
        beta = beta_raw

        if "layernorm" in sft_settings:
            beta = self._get_or_create_sft_ln(f"{signal_prefix}_beta", beta)(beta)
            if "only_beta" not in sft_settings:
                gamma = self._get_or_create_sft_ln(f"{signal_prefix}_gamma", gamma)(gamma)

        if "tanh_clamp" in sft_settings:
            beta = torch.tanh(beta)
            if "only_beta" not in sft_settings:
                gamma = torch.tanh(gamma)

        if "only_beta" in sft_settings:
            gamma = torch.ones_like(gamma)
        else:
            if not "gamma_not_1" in sft_settings:
                gamma = 1.0 + gamma
            else:
                gamma = gamma

        return gamma, beta


    def get_img_feature(self, x, area_list, flatten=False, return_gamma_beta=False, turn_on_modulation=True, no_pooling=False, return_signals=False):
        calc_graph = {}
        gamma_calc_graph = {}
        beta_calc_graph = {}

        # Stores all signals for tuning map analysis.
        # Structure:
        #   signals['regions'][area]             -> post-BN, post-ReLU activation [B, C, H, W]
        #   signals['projections'][area][source] -> raw conv_out                  [B, C, H, W]
        #   signals['sft_gamma'][area]           -> post-transform (gamma - 1)    [B, C, H, W]
        #   signals['sft_beta'][area]            -> post-transform beta            [B, C, H, W]

        signals = {
            'regions': {},
            'projections': {},
            'sft_gamma': {},
            'sft_beta': {},
        }

        for area in self.top_sort:
            if area == 'input':
                continue
            
            # RGC projections to the LGN and SC
            if area == 'RGCdLGN' or area == 'RGCsSC':
                layer = self.network.find_layer_by_source_target('input', area, layer_type=CustomConvLayer.__name__)
                layer_name = layer.source_name + layer.target_name
                if area in calc_graph:
                    raise ValueError(f"Area {area} already exists in calc_graph, but this pathway only has 1 possible input.")
                retina_out = self.Retina[layer_name](x)
                calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](retina_out))

                if return_signals:
                    signals['regions'][area] = calc_graph[area]
                    signals['projections'][area] = {'input': retina_out}
                continue
            
            # Geniculate Pathway
            if area == 'LGNd' or area == 'LGNv':
                layer = self.network.find_layer_by_source_target('RGCdLGN', area, layer_type=ConvLayer.__name__)
                layer_name = layer.source_name + layer.target_name
                if area in calc_graph:
                    raise ValueError(f"Area {area} already exists in calc_graph, but this pathway only has 1 possible input.")
                conv_out = self.Convs[layer_name](calc_graph[layer.source_name])
                calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](conv_out))

                if return_signals:
                    signals['regions'][area] = calc_graph[area]
                    signals['projections'][area] = {layer.source_name: conv_out}
                continue
            
            # Extrageniculate Pathway
            if area == 'sSC':
                layer = self.network.find_layer_by_source_target('RGCsSC', area, layer_type=ConvLayer.__name__)
                layer_name = layer.source_name + layer.target_name
                conv_out = self.Convs[layer_name](calc_graph[layer.source_name]) # NOTE: BN and ReLU applied after modulation
                calc_graph[area] = conv_out

                # NOTE: no batch norm applied to the generated gamma/beta values, since they are just raw outputs of the SFT generator layers
                # NOTE: VISp5 is the only source that modulates sSC, hardcoded the name      
                gamma_raw = self.Convs['VISp5SFT_gamma_sSC'](calc_graph["VISp5"])
                beta_raw = self.Convs['VISp5SFT_beta_sSC'](calc_graph["VISp5"])
                gamma, beta = self._apply_sft_transforms(gamma_raw, beta_raw, self.sft_settings, "sSC")

                gamma_calc_graph[area] = gamma
                beta_calc_graph[area] = beta

                if turn_on_modulation:
                    modulated = gamma * calc_graph[area] + beta
                else:
                    modulated = calc_graph[area]

                calc_graph[area] = nn.ReLU(inplace=True)(self.BNs[area](modulated))

                if return_signals:
                    signals['regions'][area] = calc_graph[area]
                    signals['projections'][area] = {layer.source_name: conv_out}
                    signals['sft_gamma'][area] = gamma - 1.0    # post-transform gamma
                    signals['sft_beta'][area]  = beta           # post-transform beta
                continue
            
            if "SFT_gamma_" in area or "SFT_beta_" in area:
                modulatory_target_area = area.split('_')[-1]
                if modulatory_target_area != "sSC":
                    calc_graph[area] = self.Convs[f"LP_{modulatory_target_area}{area}"](calc_graph[f"LP_{modulatory_target_area}"])
                    # NOTE: no batch norm applied to the generated gamma/beta values, since they are just raw outputs of the SFT generator layers
                else:
                    pass # We handle sSC and it's modulation in the if-block above "if area == 'sSC'"
                continue

            # V1/HVAs and inputs to the LP
            found_layer_for_area = False
            proj_conv_outs = {}   # source -> raw conv output, collected for BN rescaling

            for layer in self.network.layers:
                if layer.__class__.__name__ == ConvLayer.__name__ and layer.target_name == area:
                    assert "LP_" not in area, f"Expected area {area} to not contain 'LP_' since it's a target of a ConvLayer, but got {area}"

                    layer_name = layer.source_name + layer.target_name
                    conv_out = self.Convs[layer_name](calc_graph[layer.source_name])
                    proj_conv_outs[layer.source_name] = conv_out

                    calc_graph[area] = conv_out if area not in calc_graph \
                        else calc_graph[area] + conv_out
                    found_layer_for_area = True

                elif (layer.__class__.__name__ == LPConvInputMultipleTargets.__name__ or layer.__class__.__name__ == LPCustomConvInputMultipleTargets.__name__) and area in layer.target_names:
                    assert "LP_" in area, f"Expected area {area} to contain 'LP_' since it's a target of an LPConvInputMultipleTargets or LPCustomConvInputMultipleTargets layer."

                    conv_out = self.LP_input_layers[layer.source_name](calc_graph[layer.source_name])
                    proj_conv_outs[layer.source_name] = conv_out

                    calc_graph[area] = conv_out if area not in calc_graph \
                        else calc_graph[area] + conv_out
                    found_layer_for_area = True
            
            if not found_layer_for_area:
                raise ValueError(f"Did not find any layer in the network that targets area {area}, but expected to find at least one based on the topological sort of the graph. Check network initialization to ensure that all areas have at least one incoming layer, and check the forward pass to ensure that all layers are being iterated through correctly.")
            
            # Perform modulation if required
            lp_pathway_description = get_lp_pathways_description(area)
            if len(lp_pathway_description) > 0:
                assert len(lp_pathway_description) == 1, f"Expected exactly one LP pathway description for area {area}, but got {len(lp_pathway_description)}. Check get_lp_pathways_description function."
                gamma_raw = calc_graph[f"SFT_gamma_{area.split('_')[-1]}"]
                beta_raw = calc_graph[f"SFT_beta_{area.split('_')[-1]}"]
                gamma, beta = self._apply_sft_transforms(gamma_raw, beta_raw, self.sft_settings, area)

                gamma_calc_graph[area] = gamma
                beta_calc_graph[area] = beta

                if turn_on_modulation:
                    calc_graph[area] = gamma * calc_graph[area] + beta
                
                if return_signals:
                    signals['sft_gamma'][area] = gamma - 1.0 # subtract 1 constant gamma = 1 doesn't affect tuning map
                    signals['sft_beta'][area]  = beta

            # Apply batch norm and relu after modulation (if applicable) or after summing inputs (if no modulation)
            calc_graph[area] = nn.ReLU(inplace=True)(
                self.BNs[area](
                    calc_graph[area]
                )
            )
            # if calc_graph[area].sum() == 0:
            #     pdb.set_trace()
        
            # Store signals after BN is finalised (running stats updated)
            if return_signals:
                signals['regions'][area] = calc_graph[area]
                signals['projections'][area] = {src: co for src, co in proj_conv_outs.items()}


        if len(area_list) == 0:
            area_list = list(calc_graph.keys())

        if return_signals:
            # Detach everything so analysis code doesn't hold the graph
            def _detach(d):
                if isinstance(d, torch.Tensor):
                    return d.detach()
                return {k: _detach(v) for k, v in d.items()}
            signals = _detach(signals)

        if len(area_list) == 1:
            area = area_list[0]
            result = torch.flatten(calc_graph[area], 1) if flatten else calc_graph[area]
            if return_gamma_beta:
                return (result, gamma_calc_graph, beta_calc_graph,
                        signals) if return_signals else (result, gamma_calc_graph, beta_calc_graph)
            return (result, signals) if return_signals else result

        elif no_pooling:
            re = {area: calc_graph[area] for area in area_list}
            if return_gamma_beta:
                return (re, gamma_calc_graph, beta_calc_graph,
                        signals) if return_signals else (re, gamma_calc_graph, beta_calc_graph)
            return (re, signals) if return_signals else re

        else:
            re = None
            for area in area_list:
                pooled = torch.nn.AdaptiveAvgPool2d(4)(calc_graph[area])
                re = pooled if re is None else torch.cat([pooled, re], axis=1)
        if return_gamma_beta:
            return (re, gamma_calc_graph, beta_calc_graph,
                    signals) if return_signals else (re, gamma_calc_graph, beta_calc_graph)
        return (re, signals) if return_signals else re

    def forward(self, x):
        x = self.get_img_feature(x, OUTPUT_AREAS, flatten=False)
        # x = self.classifier(x)
        return x
