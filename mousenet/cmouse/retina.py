import torch
import torch.nn as nn
import torch.nn.functional as F
from .exps.imagenet.config import INPUT_SIZE
from .exps.imagenet.config import LUM_CHANNEL
class MouseRetinaLayer(nn.Module):
    def __init__(self, num_rgb_dog_output_channels, rgb_kernel_size, sigma_c=5.61, sigma_s=16.98):
        """"
        Build the RGC feature maps.

        num_rgb_dog_output_channels: tuple of (on_channels, off_channels, on_off_channels, other) for the RGCs that project to the dLGN and SC/LP.
        - the on_channels, off_channels, and on_off_channels represent DoG channels, which already account for receptive field reduction.
        - "other" represents other RGCs that project to the target area which are not modelled by DoGs. 
        The receptive field of SC/dLGN neurons are applied to these channels instead.

        rgb_kernel_size: receptive field size of SC/dLGN neurons applied to the "other" RGC channels, which are not modelled by DoGs.
        sigma_c and sigma_s: parameters for the center and surround Gaussian kernels of the DoG.
        """
        super().__init__()
        # Fixed Gaussian Kernels
        self.k_c = self._get_gaussian_kernel(sigma_c)
        self.k_s = self._get_gaussian_kernel(sigma_s)
        
        # Learnable mixing of ON/OFF channels (Per-pixel)
        self.pixel_mix_weights = nn.Parameter(torch.randn(1, 2, INPUT_SIZE[1], INPUT_SIZE[2]))  # Shape: [1, 2, H, W]

        # Proportional inputs of different RGC types to SC/LP
        on_channels, off_channels, on_off_channels, other_channels = num_rgb_dog_output_channels
        print(f"MouseRetinaLayer initialized with ON channels: {on_channels}, OFF channels: {off_channels}, ON-OFF channels: {on_off_channels}, OTHER channels: {other_channels}")

        # NOTE: Kernel size should be based on the receptive field size of RGCs
        # that is already determined by the fixed DoG filtering
        # so we use a kernel size of 1 to generate the appropriate number of channels
        self.conv_on = nn.Conv2d(1, on_channels, kernel_size=1)
        self.conv_off = nn.Conv2d(1, off_channels, kernel_size=1)
        self.conv_on_off = nn.Conv2d(1, on_off_channels, kernel_size=1)

        # TODO: Make this a sparse Conv2d?
        self.conv_other = nn.Conv2d(3, other_channels, kernel_size=rgb_kernel_size, padding=rgb_kernel_size // 2)

        self.out_channels = on_channels + off_channels + on_off_channels + other_channels

    def _get_gaussian_kernel(self, sigma):
        """
        Meant to match ndi.gaussian_filter()
        which is used in "A deep neural network model of the primate superior 
        colliculus for emotion recognition" (Méndez et al, 2022)
        to model the center-surround receptive fields of RGCs.
        """
        # Ensure kernel size is odd and we capture enough of the Gaussian
        k_size = int(4 * sigma + 1) | 1
        x = torch.arange(k_size).float() - k_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma**2))
        gauss /= gauss.sum()
        kernel = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).view(1, 1, k_size, k_size)
        return kernel

    def forward(self, x):
        # x shape: [Batch, 4, 64, 64] (RGB + LUV-L)
        lum = x[:, LUM_CHANNEL, :, :] 
        rgb = x[:, :LUM_CHANNEL, :, :]
        
        # Fixed DoG Filtering
        center = F.conv2d(lum, self.k_c.to(lum.device), padding=self.k_c.shape[-1]//2)
        surround = F.conv2d(lum, self.k_s.to(lum.device), padding=self.k_s.shape[-1]//2)
        raw_dog = center - surround
        
        # ON/OFF RGCs
        pure_on = F.relu(raw_dog)
        pure_off = F.relu(-raw_dog)
        
        # Learnable ON-OFF Mixing (Per-pixel)
        mixed_input = torch.cat([pure_on, pure_off], dim=1)
        mix_on_off = F.relu(torch.sum(mixed_input * self.pixel_mix_weights, dim=1, keepdim=True))
        
        # Proportional RGC outputs to SC/LP
        feat_on = self.conv_on(pure_on)
        feat_off = self.conv_off(pure_off)
        feat_on_off = self.conv_on_off(mix_on_off)
        feat_other = F.relu(self.conv_other(rgb))

        # Final Features to the model are [ON...OFF...ON-OFF...OTHER...]
        return torch.cat([feat_on, feat_off, feat_on_off, feat_other], dim=1)