from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F



class MWCNN(nn.Module):
    """
    PyTorch implementation of a Multi-scale Wavelet CNN model, based on the
    standard U-Net architecture where pooling operations have been replaced
    by discrete wavelet transforms.
    
    Source:
    https://github.com/zaccharieramzi/fastmri-reproducible-benchmark/blob/master/fastmri_recon/models/subclassed_models/denoisers/mwcnn.py#L134
    """
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            dims: int = 2,
            n_scales: int = 3,
            n_filters_per_scale: List[int] = [16, 32, 64],
            n_convs_per_scale: List[int] = [2, 2, 2],
            n_first_convs: int = 1,
            first_conv_n_filters: int = 16,
            res: bool = False,
        ):
        """
        Args:
            in_chans: Number of channels in the input to the MWCNN model.
            out_chans: Number of channels in the output of the MWCNN model.
            dims: number of dimensions for convolutional operations (2 or 3).
            n_scales: Number of scales, i.e. number of pooling layers.
            n_filters_per_scale: Number of filters used by the convolutional
                layers at each scale.
            n_convs_per_scale: Number of convolutional layers per scale.
            n_first_convs: Number of convolutional layers at the start of
                the architecture, i.e. before pooling layers.
            first_conv_n_filters: Number of filters used by the inital
                convolutional layers.
            res: Whether to use a residual connection between input and output.
        """            
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dims = dims
        self.n_scales = n_scales
        self.n_filters_per_scale = n_filters_per_scale
        self.n_convs_per_scale = n_convs_per_scale
        self.n_first_convs = n_first_convs
        self.first_conv_n_filters = first_conv_n_filters
        self.res = res   
        
        assert self.dims in [2, 3], \
        "Dimensions must be either 2 or 3"

        if self.dims == 2:
            conv_op = nn.Conv2d
        if self.dims == 3:
            conv_op = nn.Conv3d

        # First and last convolutions block without pooling
        if self.n_first_convs > 0:
            self.first_convs = nn.ModuleList([ConvBlock(
                  in_chans = self.in_chans,
                  n_filters = self.first_conv_n_filters,
                  dims = self.dims,
            )])
            for _ in range(1, 2 * self.n_first_convs - 1):
                self.first_convs.append(ConvBlock(
                    in_chans = self.first_conv_n_filters,
                    n_filters = self.first_conv_n_filters,
                    dims = self.dims,
                ))
            self.first_convs.append(conv_op(
                self.first_conv_n_filters,
                self.out_chans,
                kernel_size=3,
                padding='same',
                bias=True,
            ))

        # All convolution blocks during pooling/unpooling
        self.conv_blocks_per_scale = nn.ModuleList([
            nn.ModuleList([ConvBlock(
                in_chans = self.chans_for_conv_for_scale(i_scale, i_conv)[0],
                n_filters = self.chans_for_conv_for_scale(i_scale, i_conv)[1],
                dims = self.dims,
            ) for i_conv in range(self.n_convs_per_scale[i_scale] * 2)])
            for i_scale in range(self.n_scales)
        ])

        if self.n_first_convs < 1:
            # Adjust last convolution of the last convolution block
            self.conv_blocks_per_scale[0][-1] = conv_op(
                self.n_filters_per_scale[0],                                        
                4 * self.out_chans,
                kernel_size=3,
                padding='same',
                bias=True,
            )

        # Pooling operations
        self.pooling = DWT()
        self.unpooling = IWT()


    def chans_for_conv_for_scale(self, i_scale: int, i_conv: int) -> Tuple[int, int]:
        """
        Returns input channels and number of filters for each convolution at each
        scale for both downsampling and upsampling sections of the network.
        """
        in_chans = self.n_filters_per_scale[i_scale]
        n_filters = self.n_filters_per_scale[i_scale]

        # Convolutions in downsampling section
        if i_conv == 0:
            if i_scale == 0:
                in_chans = 4 * self.first_conv_n_filters
            else:
                in_chans = 4 * self.n_filters_per_scale[i_scale-1]

        # Convolutions in upsampling section
        if i_conv == self.n_convs_per_scale[i_scale] * 2 - 1:
            if i_scale == 0:
                n_filters = max(4 * self.first_conv_n_filters, 4 * self.out_chans)
            else:
                n_filters = 4 * self.n_filters_per_scale[i_scale-1]

        return in_chans, n_filters


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Input tensor of shape `(N,in_chans,H,W)`
                        - 
        Returns:
            Output tensor of shape `(N,out_chans,H,W)`
        """
        last_feature_for_scale = []
        current_feature = inputs
        
        # First convolutions
        if self.n_first_convs > 0:
            for conv in self.first_convs[:self.n_first_convs]:
                current_feature = conv(current_feature)
            first_conv_feature = current_feature

        # Downsampling section
        for i_scale in range(self.n_scales):
            current_feature = self.pooling(current_feature)
            n_convs = self.n_convs_per_scale[i_scale]
            for conv in self.conv_blocks_per_scale[i_scale][:n_convs]:
                current_feature = conv(current_feature)
            last_feature_for_scale.append(current_feature)

        # Upsampling section
        for i_scale in range(self.n_scales - 1, -1, -1):
            if i_scale != self.n_scales - 1:
                current_feature = self.unpooling(current_feature)
                current_feature = current_feature + last_feature_for_scale[i_scale]
            n_convs = self.n_convs_per_scale[i_scale]
            for conv in self.conv_blocks_per_scale[i_scale][n_convs:]:
                current_feature = conv(current_feature)
        current_feature = self.unpooling(current_feature)

        # Last convolution
        if self.n_first_convs > 0:
            current_feature = current_feature + first_conv_feature
            for conv in self.first_convs[self.n_first_convs:]:
                current_feature = conv(current_feature)
        if self.res:
            outputs = inputs + current_feature
        else:
            outputs = current_feature
        return outputs



class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of one convolution layer, followed by
    instance normalization and LeakyReLU activation.
    """
    def __init__(self, in_chans: int, n_filters: int, dims: int):
        """
        Args:
            in_chans: Number of channels in the input.
            n_filters: Number of convolutional filters.
            dims: Number of dimensions for convolutional operations (2 or 3).
        """
        super().__init__()

        if dims == 2:
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d
            
        if dims == 3:
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d

        self.layers = nn.Sequential(
            conv_op(in_chans, n_filters, kernel_size=3, padding='same', bias=False),
            norm_op(n_filters),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)



class DWT(nn.Module):
    """
    A discrete wavelet transform used in the down-pooling operations of the
    MWCNN network.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x01 = inputs[:, :, 0::2] / 2
        x02 = inputs[:, :, 1::2] / 2
        x1 = x01[..., 0::2]
        x2 = x02[..., 0::2]
        x3 = x01[..., 1::2]
        x4 = x02[..., 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)



class IWT(nn.Module):
    """
    A discrete inverse wavelet transform used in the up-pooling operations of the
    MWCNN network.
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, ch, h, w = inputs.shape
        new_ch = ch // 4

        x1 = inputs[:, 0:new_ch] / 2
        x2 = inputs[:, new_ch:2*new_ch] / 2
        x3 = inputs[:, 2*new_ch:3*new_ch] / 2
        x4 = inputs[:, 3*new_ch:4*new_ch] / 2

        outputs = torch.zeros([b, new_ch, 2*h, 2*w], dtype=inputs.dtype).cuda()
        outputs[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        outputs[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        outputs[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        outputs[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return outputs

        