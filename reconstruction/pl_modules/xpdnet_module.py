from typing import List
from argparse import ArgumentParser
import torch

from reconstruction.data import transforms
from reconstruction.utils import SSIMLoss
from reconstruction.models import XPDNet, XPDNet_RNN
from .mri_module import MriModule


class XPDNetModule(MriModule):
    """
    Pytorch Lightning module for training XPDNet. 
    
    The architecture variations for dynamic MRI reconstruction are
    inspired by the XPDNet for static MRI reconstruction, introduced in
    the following paper:

    Z. Ramzi et al. "XPDNet for MRI Reconstruction: an application to the
    2020 fastMRI challenge". arXiv: 2010.07290, 2021.
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        crnn_chans: int = 18,
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
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of unrolled iterations for XPDNet.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            crnn_chans: Hidden state size in CRNN XPDNet.
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
                convolutional layer in image denoiser MWCNN.
            res: Whether to use a residual connection between input and output in
                image denoiser MWCNN.
            primal_only: Whether to generate a buffer in k-space or only in image
                space.
            n_primal: The size of the buffer in image-space.
            n_dual: The size of the buffer in k-space.
            dynamic_type: Type of architecture adjustment for dynamic setting.
            weight_sharing: Optional setting in 'XF' or 'XT' dynamics mode, allowing
                image net to share the same parameters in both x-f and y-f planes.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.num_cascades = num_cascades
        self.sens_chans = sens_chans
        self.sens_pools = sens_pools
        self.crnn_chans = crnn_chans
        self.n_scales = n_scales
        self.n_filters_per_scale = n_filters_per_scale
        self.n_convs_per_scale = n_convs_per_scale
        self.n_first_convs = n_first_convs
        self.first_conv_n_filters = first_conv_n_filters
        self.res = res
        self.primal_only = primal_only
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.dynamic_type = dynamic_type
        self.weight_sharing = weight_sharing
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        
        assert self.dynamic_type in ['XF', 'XT', '2D', 'CRNN'], \
        "dynamic_type argument must be one of 'XF', 'XT', '2D' or 'CRNN'"
        
        if self.dynamic_type == 'CRNN':
            self.xpdnet = XPDNet_RNN(
                num_cascades=self.num_cascades,
                sens_chans=self.sens_chans,
                sens_pools=self.sens_pools,
                chans=self.crnn_chans,
                primal_only=self.primal_only,
                n_primal=self.n_primal,
                n_dual=self.n_dual,
            )
        else:
            self.xpdnet = XPDNet(
                num_cascades=self.num_cascades,
                sens_chans=self.sens_chans,
                sens_pools=self.sens_pools,
                n_scales=self.n_scales,
                n_filters_per_scale=self.n_filters_per_scale,
                n_convs_per_scale=self.n_convs_per_scale,
                n_first_convs=self.n_first_convs,
                first_conv_n_filters=self.first_conv_n_filters,
                res=self.res,
                primal_only=self.primal_only,
                n_primal=self.n_primal,
                n_dual=self.n_dual,
                dynamic_type=self.dynamic_type,
                weight_sharing = self.weight_sharing,
            )

        self.loss = SSIMLoss()

    def forward(self, masked_kspace, mask):
        return self.xpdnet(masked_kspace, mask)
    
    def training_step(self, batch, batch_idx):
        masked_kspace, mask, target, fname, slice_num, max_value, _ = batch

        output = self(masked_kspace, mask)
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
        masked_kspace, mask, target, fname, slice_num, max_value, _ = batch

        output = self.forward(masked_kspace, mask)
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
        masked_kspace, mask, target, fname, slice_num, max_value, _ = batch

        output = self(masked_kspace, mask)
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
            help="Number of XPDNet cascades",
        )
        parser.add_argument(
            "--sens_chans",
            default=8,
            type=int,
            help="Number of channels for sense map estimation U-Net in XPDNet",
        )
        parser.add_argument(
            "--sens_pools",
            default=4,
            type=int,
            help="Number of pooling layers for sense map estimation U-Net in XPDNet",
        )
        parser.add_argument(
            "--crnn_chans",
            default=18,
            type=int,
            help="Hidden state size in CRNN XPDNet",
        )
        parser.add_argument(
            "--n_scales",
            default=3,
            type=int,
            help="Number of scales, i.e. number of pooling layers, in image denoiser module",
        )
        parser.add_argument(
            "--n_filters_per_scale",
            nargs="+",
            default=[16, 32, 64],
            type=int,
            help="""Number of filters used by the convolutional layers
                    at each scale in image denoiser module""",
        )
        parser.add_argument(
            "--n_convs_per_scale",
            nargs="+",
            default=[2, 2, 2],
            type=int,
            help="""Number of convolutional layers per scale in
                    image denoiser module""",
        )
        parser.add_argument(
            "--n_first_convs",
            default=1,
            type=int,
            help="""Number of convolutional layers at the start of the architecture,
                    i.e. before pooling layers, in image denoiser module""",
        )
        parser.add_argument(
            "--first_conv_n_filters",
            default=16,
            type=int,
            help="Number of filters in the inital convolutional layers",
        )
        parser.add_argument(
            "--res",
            default=False,
            type=bool,
            help="Whether to use a residual connection in image denoising module",
        )
        parser.add_argument(
            "--primal_only",
            default=True,
            type=bool,
            help="Whether to generate a buffer in k-space or only in image-space",
        )
        parser.add_argument(
            "--n_primal",
            default=5,
            type=int,
            help="The size of the buffer in image-space",
        )
        parser.add_argument(
            "--n_dual",
            default=1,
            type=int,
            help="The size of the buffer in k-space",
        )
        parser.add_argument(
            "--dynamic_type",
            default='XF',
            type=str,
            help="""Architectural variation for dynamic reconstruction.
                    Options are ['XF', 'XT', '2D', 'CRNN']""",
        )
        parser.add_argument(
            "--weight_sharing",
            default=False,
            type=bool,
            help="Allows parameter sharing of MWCNN nets in x-f, y-f planes",
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
