from typing import Optional
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

import reconstruction as rec
from reconstruction.data.transforms import center_crop_to_smallest


class InferenceTransform(nn.Module):
    """
    Data saving module for reconstruction output of the inference
    dataset. This is generally a subset of the test set and it is
    used for visualisation purposes.
    """
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        save_path: Path,
    ): 
        """
        Args:
            model: Trained model used for dynamic reconstruction of
                MRI data.
            model_type: One of 'varnet', 'cinenet', 'xpdnet'.
            save_path: Path to directory where saving data will be
                stored.
        """
        super(InferenceTransform, self).__init__()
    
        assert model_type in ['varnet', 'cinenet', 'xpdnet'], \
            'Wrong model_type arg.'
        
        self.model_type = model_type
        self.save_path = save_path
        self.device = 'cuda'
        self.model = model.to(self.device).eval()
        
    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        target: torch.Tensor,
        fname: str,
        sens_maps: Optional[torch.Tensor]=None,
    ) -> float:
        
        # Image reconstruction of inference dataset using trained model
        model_time_start = time.time()
        masked_k = masked_kspace.to(self.device)
        mask = mask.to(self.device)
        if self.model_type == 'cinenet':
            sens_maps = sens_maps.to(self.device) 
            output = self.model(masked_k, mask, sens_maps)
        else:
            output = self.model(masked_k, mask)
        model_time_end = time.time()
        output = output.cpu()
        
        # Generate zero-filled reconstruction for qualitative comparison
        scaling_factor = torch.sqrt(torch.prod(torch.as_tensor(masked_kspace.shape[-3:-1])))
        images = rec.utils.ifft2c(masked_kspace, norm=None) * scaling_factor
        zero_filled = rec.utils.rss_complex(images, dim=2)
        
        # Crop all tensors to the same size (for visualisation)
        target, output = center_crop_to_smallest(target, output)
        target, zero_filled = center_crop_to_smallest(target, zero_filled)
        
        # Store ndarray-converted tensors to save_path
        target = target.numpy().astype('float32')
        output = output.numpy().astype('float32')
        zero_filled = zero_filled.numpy().astype('float32')
        
        np.save(str(self.save_path) + f'/target_{fname[0]}.npy', target[0])
        np.save(str(self.save_path) + f'/output_{self.model_type}_{fname[0]}.npy', output[0])
        np.save(str(self.save_path) + f'/zero_filled_{fname[0]}.npy', zero_filled[0])
        
        return model_time_end - model_time_start
    