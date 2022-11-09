from typing import List, Tuple, Dict, Union
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft, fftshift, ifftshift

import reconstruction as rec
from reconstruction.data import transforms
from .denoisers.unet import Unet
from .denoisers.mwcnn import MWCNN
from .denoisers.kspace_net import KSpaceCNN



class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    XPDNet network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        res_connection: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            res_connection: Adds a residual connection between input and
                output of the U-Net model.
        """
        super().__init__()
        
        self.res_connection = res_connection
        self.unet_model = Unet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape
        return b, x.view(b * c, h, w, comp).permute(
                0, 3, 1, 2,
            )

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, comp, h, w = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, comp, h, w).permute(
                0, 1, 3, 4, 2,
            )

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
        b, x = self.chans_to_batch_dim(x)

        # Estimate sensitivities
        x_temp = x.clone()
        x = self.unet_model(x)
        if self.res_connection:
            x += x_temp
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        
        x = x.unsqueeze(1)
        return x



class ForwardOperator(nn.Module):
    """
    The forward operator in the encoding matrix. It transforms a coil-combined
    input image-space into the corresponding k-space through the use of coil
    sensitivity maps and a Fourier transform. The resulting k-space is optionally
    masked if used in a data-consistency layer.
    """
    def __init__(self, masked: bool = False):
        """
        Args:
            masked: Whether to apply subsampling mask to output k-space.
        """
        super().__init__()
        self.masked = masked
        
    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        buffer_size: int,
    ) -> torch.Tensor:
        
        # Operator only acts on the first image in the buffer
        image = torch.stack([image[..., 0], image[..., buffer_size]], dim=-1)
        kspace = rec.utils.fft2c(rec.utils.complex_mul(image, sens_maps))
        if self.masked:
            kspace = kspace * mask + 0.0
            
        return kspace
   
        
        
class BackwardOperator(nn.Module):
    """
    The backward/adjoint operator in the encoding matrix. It transforms an input
    k-space into the corresponding coil-combined image-space through the use of
    coil sensitivity maps and a Fourier transform. The input k-space is optionally
    masked if used in a data-consistency layer.
    """
    def __init__(self, masked: bool = False):
        """
        Args:
            masked: Whether to apply subsampling mask to input k-space.
        """
        super().__init__()
        self.masked = masked
        
    def forward(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        buffer_size: int,
    ) -> torch.Tensor:
        
        # Operator only acts on the first kspace in the buffer
        kspace = torch.stack([kspace[..., 0], kspace[..., buffer_size]], dim=-1)
        if self.masked:
            kspace = kspace * mask + 0.0
        image = rec.utils.ifft2c(kspace)
        return rec.utils.complex_mul(image, rec.utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )


        
class XPDNet(nn.Module):
    """
    An adaptation of the XPDNet model for dynamic MRI reconstruction.
    
    Reference paper:
    `Z. Ramzi et al. "XPDNet for MRI Reconstruction: an application to the 2020 fastMRI
    challenge". arXiv: 2010.07290, 2021`.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        n_scales: int = 3,
        n_filters_per_scale: List[int] = [16, 32, 64],
        n_convs_per_scale: List[int] = [2, 2, 2],
        n_first_convs: int = 1,
        first_conv_n_filters: int = 16,
        res: bool = False,
        primal_only: bool = True,
        n_primal: int = 5,
        n_dual: int = 1,
        dynamic_type: str = 'XF',
        weight_sharing: bool = False,
    ):
        """
        Args:
            num_cascades: Number of unrolled iterations for XPDNet.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            n_scales: Number of scales, i.e. number of pooling layers, in
                image denoiser MWCNN.
            n_filters_per_scale: Number of filters used by the convolutional
                layers at each scale in image denoiser MWCNN.
            n_convs_per_scale: Number of convolutional layers per scale in
                image denoiser MWCNN.
            n_first_convs: Number of convolutional layers at the start of
                the architecture, i.e. before pooling layers, in image denoiser
                MWCNN.
            first_conv_n_filters: Number of filters used by the inital
                convolutional layers in image denoiser MWCNN.
            res: Whether to use a residual connection between input and output in
                image denoiser MWCNN.
            primal_only: Whether to generate a buffer in k-space or only in image
                space.
            n_primal: The size of the buffer in image-space.
            n_dual: The size of the buffer in k-space.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                image net to share the same parameters in both x-f and y-f planes.
        """
        super().__init__()
        
        self.domain_sequence = 'KI' * num_cascades
        self.i_buffer_mode = True
        self.k_buffer_mode = not primal_only
        self.i_buffer_size = n_primal
        self.k_buffer_size = 1 if primal_only else n_dual
        self.n_scales = n_scales
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.backward_op = BackwardOperator(masked=False)
        
         # ----- k-space Net -----
         
        if not primal_only:
            self.kspace_net = nn.ModuleList([KSpaceCNN(
                    in_chans = 2 * (n_dual+2),
                    out_chans = 2 * n_dual,
                    n_convs = 3,
                    n_filters = 16,
                ) for _ in range(num_cascades)]
            )
        else:
            self.kspace_net = [self.measurements_residual for _ in range(num_cascades)]

        
        # ----- Image Net -----
        denoiser_kwargs = {
            'in_chans': 2 * (n_primal+1),
            'out_chans': 2 * n_primal,
            'dims': 2,
            'n_scales': n_scales,
            'n_filters_per_scale': n_filters_per_scale,
            'n_convs_per_scale': n_convs_per_scale,
            'n_first_convs': n_first_convs,
            'first_conv_n_filters': first_conv_n_filters,
            'res': res,
        }
        
        if dynamic_type in ['XF', 'XT']:
            if weight_sharing:
                self.image_net = nn.ModuleList([MWCNN(**denoiser_kwargs) for _ in range(num_cascades)])
            else:
                self.image_net = nn.ModuleList(
                    [nn.ModuleList([MWCNN(**denoiser_kwargs), MWCNN(**denoiser_kwargs)]) for _ in range(num_cascades)]
                )
        else:
            self.image_net = nn.ModuleList([MWCNN(**denoiser_kwargs) for _ in range(num_cascades)])
            
        
        # ----- Cascade Blocks -----
        buffer_kwargs = {
            'i_buffer_mode': self.i_buffer_mode,
            'k_buffer_mode': self.k_buffer_mode,
            'i_buffer_size': self.i_buffer_size,
            'k_buffer_size': self.k_buffer_size,
        }
        self.cascades = nn.ModuleList(
            [XPDNetBlock(
                self.kspace_net,
                self.image_net,
                self.n_scales,
                self.dynamic_type,
                self.weight_sharing,
                buffer_kwargs,
            ) for _ in range(len(self.domain_sequence))]
        )

    
    def measurements_residual(self, concat_kspace: torch.Tensor) -> torch.Tensor:
        current_kspace = torch.stack([concat_kspace[..., 0], concat_kspace[..., 2]], dim=-1)
        ref_kspace = torch.stack([concat_kspace[..., 1], concat_kspace[..., 3]], dim=-1)
        return current_kspace - ref_kspace


    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask)
        image = self.backward_op(masked_kspace, mask, sens_maps, 1)
        
        # Generate buffers in k-space and image-space
        kspace_buffer = torch.repeat_interleave(masked_kspace, self.k_buffer_size, dim=-1)
        image_buffer = torch.repeat_interleave(image, self.i_buffer_size, dim=-1)
        
        # Unrolled iterations
        for i_domain, domain in enumerate(self.domain_sequence):
            image_buffer, kspace_buffer = self.cascades[i_domain](
                domain,
                i_domain,
                image_buffer,
                kspace_buffer,
                masked_kspace,
                mask,
                sens_maps,
            )
        
        out_image = torch.stack(
            [image_buffer[..., 0], image_buffer[..., self.i_buffer_size]],
            dim=-1,
        )
        
        return rec.utils.complex_abs(out_image.squeeze(2))
        


class XPDNetBlock(nn.Module):
    """
    Model block for time dynamics-adjusted XPDNet, consisting of
    alternating k-space and image-space correction modules. 
    A series of these blocks can be stacked to form the full XPDNet.
    """

    def __init__(
        self,
        kspace_net: nn.Module,
        image_net: nn.Module,
        n_scales: int,
        dynamic_type: str,
        weight_sharing: bool,
        buffer_kwargs: Dict[str, Union[bool, int]],
    ):
        """
        Args:
            kspace_net: k-space interpolation network module.
            image_net: Image denoising network module.
            n_scales: Number of scales in image denoiser MWCNN.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                image net to share the same parameters in both x-f and y-f planes.
            buffer_kwargs: A dictionary containing buffer-related variables.
        """
        super().__init__()
        
        self.kspace_net = kspace_net
        self.image_net = image_net
        self.n_scales = n_scales
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing
        self.i_buffer_mode = buffer_kwargs['i_buffer_mode']
        self.k_buffer_mode = buffer_kwargs['k_buffer_mode']
        self.i_buffer_size = buffer_kwargs['i_buffer_size']
        self.k_buffer_size = buffer_kwargs['k_buffer_size']
        
        self.forward_op = ForwardOperator(masked=True)
        self.backward_op = BackwardOperator(masked=True)
        
        
    def k_domain_correction(
        self,
        i_domain: int,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        ref_kspace: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates the kspace buffer and feeds it to the kspace net
        corresponding to the current unrolled iteration.
        """
        
        forward_op_res = rec.utils.real_to_complex_multi_ch(
            self.forward_op(image_buffer, mask, sens_maps, self.i_buffer_size), 1,
        )
        
        if self.k_buffer_mode:
            kspace_buffer = rec.utils.real_to_complex_multi_ch(kspace_buffer, self.k_buffer_size)
            kspace_buffer = torch.cat([kspace_buffer, forward_op_res], dim=-1)
        else:
            kspace_buffer = forward_op_res
            
        kspace_buffer = torch.cat(
            [kspace_buffer,
            rec.utils.real_to_complex_multi_ch(ref_kspace, 1)],
            dim=-1,
        )
        
        kspace_buffer = rec.utils.complex_to_real_multi_ch(kspace_buffer)
        return self.kspace_net[i_domain//2](kspace_buffer)


    def i_domain_correction(
        self,
        i_domain: int,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates the image buffer and feeds it to the image net
        corresponding to the current unrolled iteration.
        """
        
        backward_op_res = rec.utils.real_to_complex_multi_ch(
            self.backward_op(kspace_buffer, mask, sens_maps, self.k_buffer_size), 1,
        )
        
        if self.i_buffer_mode:
            image_buffer = rec.utils.real_to_complex_multi_ch(image_buffer, self.i_buffer_size)
            image_buffer = torch.cat([image_buffer, backward_op_res], dim=-1)
        else:
            image_buffer = backward_op_res 
        
        image_buffer = rec.utils.complex_to_real_multi_ch(image_buffer)    
        b, t, c, h, w, ch = image_buffer.shape  # c=1 (coil-combined image)
        ch_out = 2 * self.i_buffer_size
        
        if self.dynamic_type in ['XF', 'XT']:
            model_out = self.xfyf_transform(image_buffer.squeeze(2), i_domain)
            
        if self.dynamic_type == '2D':
            # Batch dimension b=1. Make first dimension time so
            # that each slice is trained independently. This is
            # similar to static MRI reconstruction.
            
            # Input to model has shape (t, ch, h, w)
            image_in = image_buffer.permute(0,1,2,5,3,4).reshape(b*t, c*ch, h, w)
            model_out = self.image_net[i_domain//2](image_in).reshape(
                b, t, c, ch_out, h, w).permute(0,1,2,4,5,3)
                            
        return model_out

 
    def xfyf_transform(self, image_buffer: torch.Tensor, i_domain: int) -> torch.Tensor:
        """
        Separate input into two volumes in the rotated planes x-f and y-f
        (or x-t, y-t if in 'XT' dynamics mode). After being processed by
        their respective image nets, the volumes are then combined back into one.
        """
        b, t, h, w, ch = image_buffer.shape  #here ch is 2*(buffer size+1)
        ch_out = 2 * self.i_buffer_size
        
        # Subtract the image temporal average for numerical stability
        image_temp = image_buffer.clone()
        image_mean = torch.stack(t * [torch.mean(image_temp, dim=1)], dim=1)
        x = image_buffer - image_mean
        
        if self.dynamic_type == 'XF':
            # Apply temporal FFT
            x = rec.utils.real_to_complex_multi_ch(x, self.i_buffer_size + 1)
            x = ifftshift(fft(fftshift(x, 1), t, 1, 'ortho'), 1)
            x = rec.utils.complex_to_real_multi_ch(x) 
        
        # Reshape to xf, yf planes
        xf = x.clone().permute(0,2,4,3,1).view(b*h, ch, w, t)   
        yf = x.clone().permute(0,3,4,2,1).view(b*w, ch, h, t)
        
        # Padding in preparation for MWCNN
        xf, paddings_xf = rec.utils.pad_for_mwcnn(xf, self.n_scales)
        yf, paddings_yf = rec.utils.pad_for_mwcnn(yf, self.n_scales)
        
        # MWCNN opearting on temporal transformed xf, yf-domain
        if self.weight_sharing:
            model = self.image_net[i_domain//2]
            xf = model(xf)
            yf = model(yf)
        else:
            model_xf, model_yf = self.image_net[i_domain//2]
            xf = model_xf(xf)
            yf = model_yf(yf)
            
        # Unpad model output
        xf = rec.utils.unpad_from_mwcnn(xf, paddings_xf)
        yf = rec.utils.unpad_from_mwcnn(yf, paddings_yf)
        
        # Reshape from xf, yf
        xf_r = xf.view(b,h,1,ch_out,w,t).permute(0,5,2,1,4,3) # b,t,1,h,w,ch_out
        yf_r = yf.view(b,w,1,ch_out,h,t).permute(0,5,2,4,1,3) # b,t,1,h,w,ch_out
        
        out = 0.5 * (xf_r + yf_r)
        
        if self.dynamic_type == 'XF':
            # Apply temporal IFFT
            out = rec.utils.real_to_complex_multi_ch(out, self.i_buffer_size)
            out = fftshift(ifft(ifftshift(out, 1), t, 1, 'ortho'), 1)
            out = rec.utils.complex_to_real_multi_ch(out) 
        
        # Residual connection with the first n_primal elements of input
        in_res = torch.cat(
            [image_mean.unsqueeze(2)[..., :self.i_buffer_size],
             image_mean.unsqueeze(2)[..., self.i_buffer_size+1: -1]],
            dim = -1,
        )
        return out + in_res
        

    def forward(
        self,
        domain: str,
        i_domain: int,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if domain == 'K':
            kspace_buffer = self.k_domain_correction(
                i_domain,
                image_buffer,
                kspace_buffer,
                mask,
                sens_maps,
                ref_kspace,
            )
            
        if domain == 'I':
            image_buffer = self.i_domain_correction(
                i_domain,
                image_buffer,
                kspace_buffer,
                mask,
                sens_maps,
            )
        
        return image_buffer, kspace_buffer
        