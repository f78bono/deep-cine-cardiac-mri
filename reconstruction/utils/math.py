import numpy as np
import torch


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data ** 2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def real_to_complex_multi_ch(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Real to complex tensor conversion.
    
    Converts a stack of n complex tensors, stored as a torch.float array
    with last dimension (channel dimension) of size 2n, into a single
    torch.complex tensor with n channels.
    
    Args:
        x: A torch.float-type tensor where the first n>=2 elements of the
            last dimension correspond to the real part and the last n>=2
            elements of the last dimension correspond to the imaginary
            part of the stacked complex tensors.
        n: The number of stacked complex tensors.
        
    Returns:
        A torch.complex-type tensor with the last dimension of size n.
    """
    if not x.shape[-1] == 2*n:
        raise ValueError("Real and imaginary parts do not have the same size")
        
    return torch.complex(x[..., :n], x[..., n:])
    
    
def complex_to_real_multi_ch(x: torch.Tensor) -> torch.Tensor:
    """
    Complex to real tensor conversion.
    
    Converts a torch.complex tensor with the last dimension >= 1
    into a torch.float tensor with stacked real and imaginary parts.
    
    Args:
        x: A torch.complex-type tensor with the last dimension >= 1.
    
    Returns:
        A torch.float-type tensor with last dimension double the size
        of that of x.
    """
    return torch.cat([x.real, x.imag], dim=-1)