from typing import Tuple, List
import torch
import torch.nn.functional as F



def pad_for_mwcnn(x: torch.Tensor, n_scales: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Pads a tensor for input to a multi-scale wavelet CNN.
    Padding is applied to the last two dimensions.
    
    Source:
    https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/fastmri_recon/models/utils/pad_for_pool.py

    Args:
        x: A PyTorch tensor with at least 2 dimensions.
        n_scales: Number of scales in multi-scale wavelet CNN.
        
    Returns:
        The padded tensor and the corresponding padding values.
    """
    if x.dim() < 2:
        raise ValueError("Number of dimensions cannot be less than 2")
    problematic_dims = torch.tensor(x.shape[-2:])

    k = torch.div(problematic_dims, 2**n_scales, rounding_mode='floor' )
    n_pad = torch.where(
        torch.eq(torch.remainder(problematic_dims, 2**n_scales), 0),
        0,
        (k+1) * 2**n_scales - problematic_dims
    )

    padding_left = torch.where(
        torch.logical_or(
            torch.eq(torch.remainder(problematic_dims, 2), 0),
            torch.eq(n_pad, 0),
        ),
        torch.div(n_pad, 2, rounding_mode='floor'),
        1 + torch.div(n_pad, 2, rounding_mode='floor'),
    )
    padding_right = torch.div(n_pad, 2, rounding_mode='floor')

    paddings = []
    for i in range(2):
        paddings += [padding_left[-1-i], padding_right[-1-i]]

    x_padded = F.pad(x, paddings)

    return x_padded, paddings
    

    
def unpad_from_mwcnn(x: torch.Tensor, pad: List[torch.Tensor]) -> torch.Tensor:
    """
    Unpads the output tensor from a multi-scale wavelet CNN.
    
    Args:
        x: A padded PyTorch tensor with at least 2 dimensions.
        pad: The corresponding left and right padding values,
            ordered from the last to second last dimensions in x.
            
    Returns:
        The unpadded tensor.
    """
    if pad[1] == 0:
        return x[..., pad[2]:, pad[0]:] if pad[3] == 0 else x[..., pad[2]:-pad[3], pad[0]:]
    elif pad[3] == 0:
        return x[..., pad[2]:, pad[0]:] if pad[1] == 0 else x[..., pad[2]:, pad[0]:-pad[1]]
    else:
        return x[..., pad[2]:-pad[3], pad[0]:-pad[1]]