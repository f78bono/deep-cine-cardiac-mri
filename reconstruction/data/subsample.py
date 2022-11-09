"""
This source code is based on the fastMRI repository from Facebook AI
Research and is used as a general framework to handle MRI data. Link:

https://github.com/facebookresearch/fastMRI
"""

import contextlib
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@contextlib.contextmanager
def temp_seed(rng: np.random, seed: Optional[Union[int, Tuple[int, ...]]]):
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.
    """

    def __init__(self, center_fractions: Sequence[float], accelerations: Sequence[int]):
        """
        Args:
            center_fractions: When using a random mask, number of low-frequency
                lines to retain. When using an equispaced masked, fraction of
                low-frequency lines to retain.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
        """
        if not len(center_fractions) == len(accelerations):
            raise ValueError(
                "Number of center fractions should match number of accelerations"
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        choice = self.rng.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        return center_fraction, acceleration
        
        
        
class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a Cartesian sub-sampling mask of a given shape,
    as implemented in
    "A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image
    Reconstruction" by J. Schlemper et al.

    The mask selects a subset of rows from the input k-space data. If the
    k-space data has N rows, the mask picks out:
        1. center_fraction rows in the center corresponding to low-frequencies.
        2. The remaining rows are selected according to a tail-adjusted 
           Gaussian probability density function. This ensures that the
           expected number of rows selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the RandomMaskFunc object is called.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Create the mask.

        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the third
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")
            
        with temp_seed(self.rng, seed):
            sample_n, acc = self.choose_acceleration()
        
        N, Nc, Nx, Ny, Nch = shape
        
        # generate normal distribution
        normal_pdf = lambda length, sensitivity: np.exp(-sensitivity * (np.arange(length) - length / 2)**2)
        pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
        lmda = Nx / (2.*acc)
        n_lines = int(Nx / acc)
    
        # add uniform distribution so that probability of sampling
        # high-frequency lines is non-zero
        pdf_x += lmda * 1./Nx
    
        if sample_n:
            # lines are never randomly sampled from the already
            # sampled center
            pdf_x[Nx//2 - sample_n//2 : Nx//2 + sample_n//2] = 0
            pdf_x /= np.sum(pdf_x)  # normalise distribution
            n_lines -= sample_n
    
        mask = np.zeros((N, Nx))
        for i in range(N):
            # select low-frequency lines according to pdf
            idx = np.random.choice(Nx, n_lines, False, pdf_x)
            mask[i, idx] = 1
    
        if sample_n:
            # central lines are always sampled
            mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
        
        # reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-3] = Nx
        mask_shape[0] = N
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask    


class EquispacedMaskFunc(MaskFunc):
    """
    EquispacedMaskFunc creates a sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N rows, the mask picks out:
        1. N_low_freqs = (N * center_fraction) rows in the center
           corresponding to low-frequencies.
        2. The other rows are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of rows selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.
    """

    def __call__(
        self, shape: Sequence[int], seed: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> torch.Tensor:
        """
        Args:
            shape: The shape of the mask to be created. The shape should have
                at least 3 dimensions. Samples are drawn along the third last
                dimension.
            seed: Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same
                shape. The random state is reset afterwards.

        Returns:
            A mask of the specified shape.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_fraction, acceleration = self.choose_acceleration()
            num_rows = shape[-3]
            num_low_freqs = int(round(num_rows * center_fraction))

            # create the mask
            mask = np.zeros(num_rows, dtype=np.float32)
            pad = (num_rows - num_low_freqs + 1) // 2
            mask[pad : pad + num_low_freqs] = True

            # determine acceleration rate by adjusting for the number of low frequencies
            adjusted_accel = (acceleration * (num_low_freqs - num_rows)) / (
                num_low_freqs * acceleration - num_rows
            )
            offset = self.rng.randint(0, round(adjusted_accel))

            accel_samples = np.arange(offset, num_rows - 1, adjusted_accel)
            accel_samples = np.around(accel_samples).astype(np.uint)
            mask[accel_samples] = True

            # reshape the mask
            mask_shape = [1 for _ in shape]
            mask_shape[-3] = num_rows
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask


def create_mask_for_mask_type(
    mask_type_str: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
) -> MaskFunc:
    """
    Creates a mask of the specified type.

    Args:
        center_fractions: What fraction of the center of k-space to include.
        accelerations: What accelerations to apply.
    """
    if mask_type_str == "random":
        return RandomMaskFunc(center_fractions, accelerations)
    elif mask_type_str == "equispaced":
        return EquispacedMaskFunc(center_fractions, accelerations)
    else:
        raise Exception(f"{mask_type_str} not supported")
