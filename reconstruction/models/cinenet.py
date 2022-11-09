from typing import List, Tuple, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import reconstruction as rec
from reconstruction.data import transforms
from .denoisers.unet import Unet



class CineNet(nn.Module):
    """
    An adaptation of the CineNet model for dynamic MRI reconstruction,
    which consists of alternating U-Net and Conjugate Gradient (CG) blocks.
    Reference paper:
    
    A. Kofler et al. `An end-to-end-trainable iterative network architecture 
    for accelerated radial multi-coil 2D cine MR image reconstruction.`
    In Medical Physics, 2021.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        CG_iters: int = 4,
        chans: int = 18,
        pools: int = 4,
        dynamic_type: str = 'XF',
        weight_sharing: bool = False,
    ):
        """
        Args:
            num_cascades: Number of alternations between CG and U-Net modules.
            CG_iters: Number of  CG iterations in the CG module.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade U-Net.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                U-Net to share the same parameters in both x-f and y-f planes.
        """
        super().__init__()
        
        if dynamic_type in ['XF', 'XT']:
            if weight_sharing:
                self.model = Unet(chans, pools, dims=2)
            else:
                self.model = nn.ModuleList([Unet(chans, pools, dims=2), Unet(chans, pools, dims=2)])
        elif dynamic_type == '3D':
            self.model = Unet(chans, pools, dims=3)
        else:
            self.model = Unet(chans, pools, dims=2)
            
        self.cascades = nn.ModuleList(
            [CineNetBlock(self.model, CG_iters, dynamic_type, weight_sharing) for _ in range(num_cascades)]
        )


    def forward(self, masked_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        
        # Coil-combined image, shape (b, t, 1, h, w, ch)
        image_pred = rec.utils.complex_mul(
                        rec.utils.ifft2c(masked_kspace), rec.utils.complex_conj(sens_maps)
                    ).sum(dim=2, keepdim=True)
                    
        image_ref = image_pred.clone()
        
        for cascade in self.cascades:
            image_pred = cascade(image_pred, image_ref, mask, sens_maps)

        return rec.utils.complex_abs(image_pred.squeeze(2))
        
        
        
class CineNetBlock(nn.Module):
    """
    Model block for CineNet with several temporal dynamics adjustments.
    A series of these blocks can be stacked to form the full network.
    """

    def __init__(self, model: nn.Module, CG_iters: int, dynamic_type: str, weight_sharing: bool):
        """
        Args:
            model: Module for UNet-type image denoiser component of CineNet.
                Its architecture depends on the specfic dynamics mode.
            CG_iters: Number of  CG iterations in the CG module.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                U-Net to share the same parameters in both x-f and y-f planes.
        """
        super().__init__()

        self.model = model
        self.CG_iters = CG_iters
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
    
    def HOperator(self, x: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        The operator H = A^H \circ A + \lambda_Reg * \Id, where A is the encoding matrix.
        This ensures data consistency.
        """
        # Forward operator
        k_coils = self.sens_expand(x, sens_maps)
        # Apply sampling mask
        k_masked = k_coils * mask + 0.0
        # Backward operator
        x_combined = self.sens_reduce(k_masked, sens_maps)
        # Result of H(x)
        return x_combined + self.Softplus(self.lambda_reg) * x 
      
      
    def ConjGrad(self, x:torch.Tensor, b:torch.Tensor, mask:torch.Tensor, sens_maps:torch.Tensor, CG_iters:int)-> torch.Tensor:
        """
        Conjugate Gradient method for solving the system Hx = b
        """
        # x is the starting value, b the rhs
        r = self.HOperator(x, mask, sens_maps)
        r = b-r
        
        # Initialize p
        p = r.clone()
        
        # Old squared norm of residual
        sqnorm_r_old = torch.dot(r.flatten(), r.flatten())
      
        for kiter in range(CG_iters):
            # Calculate H(p)
            d = self.HOperator(p, mask, sens_maps)
            
            # Calculate step size alpha;
            inner_p_d = torch.dot(p.flatten(), d.flatten())
            alpha = sqnorm_r_old / inner_p_d
            
            # Perform step and calculate new residual
            x = torch.add(x, p, alpha = alpha.item())
            r = torch.add(r, d, alpha = -alpha.item())
            
            # New residual norm
            sqnorm_r_new = torch.dot(r.flatten(), r.flatten())
            
            # Calculate beta and update the norm
            beta = sqnorm_r_new / sqnorm_r_old
            sqnorm_r_old = sqnorm_r_new
            
            p = torch.add(r, p, alpha = beta.item())
    
        return x

 
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
        xf = x.clone().permute(0,2,4,3,1).view(b*h, 2, w, t)   
        yf = x.clone().permute(0,3,4,2,1).view(b*w, 2, h, t)
        
        # UNet opearting on temporal transformed xf, yf-domain
        if self.weight_sharing:
            xf = self.model(xf)
            yf = self.model(yf)
        else:
            model_xf, model_yf = self.model
            xf = model_xf(xf)
            yf = model_yf(yf)
        
        # Reshape from xf, yf
        xf_r = xf.view(b,h,1,2,w,t).permute(0,5,2,1,4,3) # b,t,1,h,w,2
        yf_r = yf.view(b,w,1,2,h,t).permute(0,5,2,4,1,3) # b,t,1,h,w,2
        
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
        image_pred: torch.Tensor,
        image_ref: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        
        # Prepare image for input to U-Net
        b, t, c, h, w, ch = image_pred.shape  # c=1 (coil-combined image)
        
        if self.dynamic_type in ['XF', 'XT']:
            model_out = self.xfyf_transform(image_pred.squeeze(2))
            
        if self.dynamic_type == '2D':
            # Batch dimension b=1. Make first dimension time so
            # that each slice is trained independently. This is
            # similar to static MRI reconstruction.
            
            # Input to model has shape (t, ch, h, w)
            image_in = image_pred.permute(0,1,2,5,3,4).reshape(b*t, c*ch, h, w)
            model_out = self.model(image_in).reshape(b, t, c, ch, h, w,
                            ).permute(0,1,2,4,5,3)
            
        if self.dynamic_type == '3D':
            # In this mode the whole spatio-temporal volume is
            # processed by a 3D U-Net at once.
            
            # Input to model has shape (b, ch, t, h, w)
            image_in = image_pred.permute(0,5,2,1,3,4).reshape(b, ch*c, t, h, w)
            model_out = self.model(image_in).reshape(b, ch, c, t, h, w,
                            ).permute(0,3,2,4,5,1)
            
        return self.ConjGrad(
                    model_out, image_ref + self.Softplus(self.lambda_reg) * model_out, mask, sens_maps, self.CG_iters
                )
        
