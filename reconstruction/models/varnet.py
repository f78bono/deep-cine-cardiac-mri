from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import reconstruction as rec
from reconstruction.data import transforms
from .denoisers.norm_unet import NormUnet, NormUnet3D



class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / rec.utils.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    
    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Get low frequency line locations to mask them out
        cent = mask.shape[-3] // 2
        left = torch.nonzero(mask[:,0,:].squeeze()[:cent] == 0)[-1]
        right = torch.nonzero(mask[:,0,:].squeeze()[cent:] == 0)[0] + cent
        num_low_freqs = right - left
        pad = (mask.shape[-3] - num_low_freqs + 1) // 2

        # Time-averaged k-space
        x = transforms.mask_center(torch.mean(masked_kspace, 1), pad, pad + num_low_freqs)

        # Convert to image space
        x = rec.utils.ifft2c(x)
        
        # Since batch=1, change batch dim to coil dim
        # to deal with each coil independently
        x, b = self.chans_to_batch_dim(x)

        # Estimate sensitivities
        x = self.norm_unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        
        x = x.unsqueeze(1)
        return x



        
class VarNet(nn.Module):
    """
    An adaptation of the end-to-end variational network model for dynamic
    MRI reconstruction. Reference paper:
    
    `A. Sriram et al. "End-to-end variational networks for accelerated MRI
    reconstruction". In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020`.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        dynamic_type: str = 'XF',
        weight_sharing: bool = False,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade U-Net.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                U-Net to share the same parameters in both x-f and y-f planes.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        
        if dynamic_type in ['XF', 'XT']:
            if weight_sharing:
                self.model = NormUnet(chans, pools)
            else:
                self.model = nn.ModuleList([NormUnet(chans, pools), NormUnet(chans, pools)])
        elif dynamic_type == '3D':
            self.model = NormUnet3D(chans, pools)
        else:
            self.model = NormUnet(chans, pools)
            
        self.cascades = nn.ModuleList(
            [VarNetBlock(self.model, dynamic_type, weight_sharing) for _ in range(num_cascades)]
        )


    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return rec.utils.complex_abs(rec.utils.complex_mul(rec.utils.ifft2c(kspace_pred),
                    rec.utils.complex_conj(sens_maps)).sum(dim=2, keepdim=False))


class VarNetBlock(nn.Module):
    """
    Model block for time dynamics-adjusted end-to-end variational network.
    A series of these blocks can be stacked to form the full variational network.
    """

    def __init__(self, model: nn.Module, dynamic_type: str, weight_sharing: bool,):
        """
        Args:
            model: Module for "regularization" component of variational
                network. Its architecture depends on the specfic dynamics mode.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                U-Net to share the same parameters in both x-f and y-f planes.
        """
        super().__init__()

        self.model = model
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing

        # Regularisation parameter is learned during training
        self.Softplus = nn.Softplus(1.) 
        lambda_init = np.log(np.exp(1)-1.)/1.
        self.lambda_reg = nn.Parameter(torch.tensor(lambda_init*torch.ones(1),dtype=torch.float),
                                         requires_grad=True)

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Forward operator: from coil-combined image-space to k-space.
        """
        return rec.utils.fft2c(rec.utils.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Backward operator: from k-space to coil-combined image-space.
        """
        x = rec.utils.ifft2c(x)
        return rec.utils.complex_mul(x, rec.utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )
 
    def xfyf_transform(self, image_combined: torch.Tensor) -> torch.Tensor:
        """
        Separate input into two volumes in the rotated planes x-f and y-f
        (or x-t, y-t if in 'XT' dynamics mode). After being processed by
        their respective U-Nets, the volumes are then combined back into one.
        """
        b, t, h, w, ch = image_combined.shape
        
        # Subtract the image temporal average for numerical stability
        image_temp = image_combined.clone()
        image_mean = torch.stack(t * [torch.mean(image_temp, dim=1)], dim=1)
        x = image_combined - image_mean
        
        if self.dynamic_type == 'XF':
            # Apply temporal FFT
            x = x.permute(0,2,3,1,4) # b,h,w,t,2
            x = rec.utils.fft1c(x)
            x = x.permute(0,3,1,2,4) # b,t,h,w,2
        
        # Reshape to xf, yf planes
        xf = x.clone().permute(0,2,3,1,4).view(b*h, 1, w, t, 2)   
        yf = x.clone().permute(0,3,2,1,4).view(b*w, 1, h, t, 2)
        
        # UNet opearting on temporal transformed xf, yf-domain
        if self.weight_sharing:
            xf = self.model(xf)
            yf = self.model(yf)
        else:
            model_xf, model_yf = self.model
            xf = model_xf(xf)
            yf = model_yf(yf)
        
        # Reshape from xf, yf
        xf_r = xf.view(b,h,1,w,t,2).permute(0,4,2,1,3,5) # b,t,1,h,w,2
        yf_r = yf.view(b,w,1,h,t,2).permute(0,4,2,3,1,5) # b,t,1,h,w,2
        
        out = 0.5 * (xf_r + yf_r)
        
        if self.dynamic_type == 'XF':
            # Apply temporal IFFT
            out = out.permute(0,2,3,4,1,5) # b,1,h,w,t,2
            out = rec.utils.ifft1c(out)
            out = out.permute(0,4,1,2,3,5) # b,t,1,h,w,2
        
        # Residual connection
        return out + image_mean.unsqueeze(2)
        

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        
        # current_kspace: 6d tensor of shape (b, t, c, h, w, ch)
        image_combined = self.sens_reduce(current_kspace, sens_maps)
        
        if self.dynamic_type in ['XF', 'XT']:
            model_out = self.xfyf_transform(image_combined.squeeze(2))
            model_term = self.sens_expand(model_out, sens_maps)
            
        if self.dynamic_type == '2D':
            # Batch dimension b=1. Make first dimension time so
            # that each slice is trained independently. This is
            # similar to static MRI reconstruction.
            
            # Input to model has shape (t, 1, h, w, ch)
            model_out = self.model(image_combined.squeeze(0))
            model_term = self.sens_expand(
                model_out.unsqueeze(0), sens_maps  # Add back batch dimension
            )
            
        if self.dynamic_type == '3D':
            # In this mode the whole spatio-temporal volume is
            # processed by a 3D U-Net at once.
            
            # Input to model has shape (b, 1, t, h, w, ch)
            model_out = self.model(image_combined.permute(0,2,1,3,4,5))
            model_term = self.sens_expand(
                model_out.permute(0,2,1,3,4,5), sens_maps
            )
        
        # Data consistency step
        v = self.Softplus(self.lambda_reg)
        return (1 - mask) * model_term + mask * (model_term + v * ref_kspace) / (1 + v)
        