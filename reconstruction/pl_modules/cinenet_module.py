from argparse import ArgumentParser
import torch

from reconstruction.data import transforms
from reconstruction.utils import SSIMLoss
from reconstruction.models import CineNet, CineNet_RNN
from .mri_module import MriModule


class CineNetModule(MriModule):
    """
    Pytorch Lightning module for training CineNet.
    
    The architecture variations for dynamic MRI reconstruction are
    inspired by the deep learning network introduced in the following paper:

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
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
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
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        
        self.num_cascades = num_cascades
        self.CG_iters = CG_iters
        self.pools = pools
        self.chans = chans
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        
        assert self.dynamic_type in ['XF', 'XT', '2D', '3D', 'CRNN'], \
        "dynamic_type argument must be one of 'XF', 'XT', '2D', '3D' or 'CRNN'"
        
        if self.dynamic_type == 'CRNN':
            self.cinenet = CineNet_RNN(
                num_cascades=self.num_cascades,
                CG_iters=self.CG_iters,
                chans=self.chans,
            )
        else:
            self.cinenet = CineNet(
                num_cascades=self.num_cascades,
                CG_iters=self.CG_iters,
                chans=self.chans,
                pools=self.pools,
                dynamic_type=self.dynamic_type,
                weight_sharing = self.weight_sharing,
            )

        self.loss = SSIMLoss()

    def forward(self, masked_kspace, mask, coils_maps):
        return self.cinenet(masked_kspace, mask, coils_maps)
    
    def training_step(self, batch, batch_idx):
        masked_kspace, mask, coils_maps, target, fname, slice_num, max_value, _ = batch

        output = self(masked_kspace, mask, coils_maps)
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def validation_step(self, batch, batch_idx):
        masked_kspace, mask, coils_maps, target, fname, slice_num, max_value, _ = batch

        output = self.forward(masked_kspace, mask, coils_maps)
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "val_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }
        
    def test_step(self, batch, batch_idx):
        masked_kspace, mask, coils_maps, target, fname, slice_num, max_value, _ = batch

        output = self(masked_kspace, mask, coils_maps)
        target, output = transforms.center_crop_to_smallest(target, output)

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output,
            "target": target,
            "test_loss": self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=max_value
            ),
        }

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # param overwrites

        # network params
        parser.add_argument(
            "--num_cascades",
            default=12,
            type=int,
            help="Number of alternations between CG and U-Net modules",
        )
        parser.add_argument(
            "--CG_iters",
            default=4,
            type=int,
            help="Number of Conjugate Gradient iterations",
        )
        parser.add_argument(
            "--pools",
            default=4,
            type=int,
            help="Number of U-Net pooling layers in CineNet blocks",
        )
        parser.add_argument(
            "--chans",
            default=18,
            type=int,
            help="Number of channels for U-Net in CineNet blocks",
        )
        parser.add_argument(
            "--dynamic_type",
            default='XF',
            type=str,
            help="""Architectural variation for dynamic reconstruction.
                    Options are ['XF', 'XT', '2D', '3D', 'CRNN']""",
        )
        parser.add_argument(
            "--weight_sharing",
            default=False,
            type=bool,
            help="Allows parameter sharing of U-Nets in x-f, y-f planes",
        )
        
        # training params (opt)
        parser.add_argument(
            "--lr", default=0.0003, type=float, help="Adam learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.1,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
