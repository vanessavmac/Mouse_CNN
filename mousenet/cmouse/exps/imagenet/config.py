import os
import math

DATA_DIR = '../data'  #os.environ['DATA_DIR']
RESULT_DIR = 'myresults'  #os.environ['RESULT_DIR']

INPUT_SIZE=(4,64,64) # NOTE: Major change to add the luminance channel as input to the model
LUM_CHANNEL = 3 # index of the luminance channel in the input tensor
NUM_CLASSES = 1000
HIDDEN_LINEAR = 2048 #4096

EDGE_Z = 1 #Z-score (# standard deviations) of edge of kernel
GSH_1 = 1 #Gaussian height of input to LGNd 
DLGN_GSW = 4 #Gaussian width of input to LGNd (corresponds to 9x9 kernel)
SSC_GSW = 5 #Gaussian width of input to sSC (corresponds to 11x11 kernel)

#OUTPUT_AREAS = ['VISp5', 'VISl5', 'VISpor5'] # for SimpleNet VISl
OUTPUT_AREAS = ['VISp5','VISl5', 'VISrl5', 'VISli5', 'VISpl5', 'VISal5', 'VISpor5']

# TODO VANESSA also model RGCs which project to both?
TOTAL_RGCS = 46000 # C57BL/6J mice have on average 46K ± 2K RBPMS+ RGCs per retina [SOURCE: https://pubmed.ncbi.nlm.nih.gov/36078097/], which matches the mouse type in https://pmc.ncbi.nlm.nih.gov/articles/PMC4982907/#sec2

DLGN_PROJECTING_RGCS = int(TOTAL_RGCS * 0.40)  # ~40% of all RGCs project to the dLGN
SC_PROJECTING_RGCS = int(TOTAL_RGCS * 0.88) # ~85–90% of all RGCs project to SC

# Determine number of RGCs of each type ON, OFF, ON-OFF, and other
# Based on proportions described in https://journals.physiology.org/doi/full/10.1152/jn.00227.2016
# 89 dLGN and 103 SC projecting RGCs are sampled to perform full characterization; the proportions of ON/OFF/ON-OFF are treated as representative of the entire RGC inputs to SC/dLGN
# This is a reasonable assumption since ">80% of dLGN- and SC-projecting RGCs fell into one of the six groups we used"
prop_dLGN_ON = 0.50
prop_dLGN_OFF = 0.21
prop_dLGN_ON_OFF = 0.11

DLGN_ON_RGCS_CHANNELS = math.ceil(DLGN_PROJECTING_RGCS * prop_dLGN_ON / INPUT_SIZE[1] / INPUT_SIZE[2])
DLGN_OFF_RGCS_CHANNELS = math.ceil(DLGN_PROJECTING_RGCS * prop_dLGN_OFF / INPUT_SIZE[1] / INPUT_SIZE[2])
DLGN_ON_OFF_RGCS_CHANNELS = math.ceil(DLGN_PROJECTING_RGCS * prop_dLGN_ON_OFF / INPUT_SIZE[1] / INPUT_SIZE[2])
DLGN_OTHER_RGCS_CHANNELS = math.ceil(DLGN_PROJECTING_RGCS * (1 - prop_dLGN_ON - prop_dLGN_OFF - prop_dLGN_ON_OFF) / INPUT_SIZE[1] / INPUT_SIZE[2])

prop_SC_ON = 0.35
prop_SC_OFF = 0.26
prop_SC_ON_OFF = 0.19

SC_ON_RGCS_CHANNELS = math.ceil(SC_PROJECTING_RGCS * prop_SC_ON / INPUT_SIZE[1] / INPUT_SIZE[2])
SC_OFF_RGCS_CHANNELS = math.ceil(SC_PROJECTING_RGCS * prop_SC_OFF / INPUT_SIZE[1] / INPUT_SIZE[2])
SC_ON_OFF_RGCS_CHANNELS = math.ceil(SC_PROJECTING_RGCS * prop_SC_ON_OFF / INPUT_SIZE[1] / INPUT_SIZE[2])
SC_OTHER_RGCS_CHANNELS = math.ceil(SC_PROJECTING_RGCS * (1 - prop_SC_ON - prop_SC_OFF - prop_SC_ON_OFF) / INPUT_SIZE[1] / INPUT_SIZE[2])


def get_out_sigma(source_area, source_depth, target_area, target_depth):
    if target_depth == '4':
        if target_area != 'VISp' and target_area != 'VISpor':
            return 1/2
        if target_area == 'VISpor':
            if source_area == 'VISp':
                return 1/2    
    return 1


# [20] A. E. Allen, C. A. Procyk, M. Howarth, L. Walmsley, and T. M. Brown, “Visual 
# input to the mouse lateral posterior and posterior thalamic nuclei: photoreceptive 
# origins and retinotopic order,” The Journal of Physiology, vol. 594, no. 7, pp. 
# 1911–1929, Apr. 2016, doi: 10.1113/JP271707.
LP_RF_SIZE = 17

# LP modulates HVAs which all have feature maps of size 32x32
LP_OUTPUT_SIZE = 32

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

def get_lp_pathways_description(target_name):
    return [pathway for pathway in LP_PATHWAYS if pathway['target'][0] + pathway['target'][1] == target_name]

def get_unique_lp_sources():
    unique_sources = set()
    for pathway in LP_PATHWAYS:
        for source_area, source_depth in pathway['sources']:
            unique_sources.add(source_area + (source_depth if source_depth is not None else ''))

    return unique_sources

def get_lp_target_areas():
    all_lp_targets = [pathway['target'][0] + pathway['target'][1] for pathway in LP_PATHWAYS]
    assert len(set(all_lp_targets)) == len(LP_PATHWAYS), f"Expected all LP targets to be unique, but got duplicate targets in {all_lp_targets}"

    return all_lp_targets

def get_targets_of_source(source_name):
    target_areas = []
    for pathway in LP_PATHWAYS:
        for source_area, source_depth in pathway['sources']:
            if source_area + (source_depth if source_depth is not None else '') == source_name:
                target_areas.append(pathway['target'][0] + pathway['target'][1])
    assert len(set(target_areas)) == len(target_areas), f"Expected each source to have unique targets, but got duplicate targets for source {source_name} in {target_areas}"

    return target_areas
