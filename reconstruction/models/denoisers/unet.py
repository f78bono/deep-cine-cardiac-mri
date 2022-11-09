import torch
from torch import nn
from torch.nn import functional as F


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234-241.
    Springer, 2015.
    """

    def __init__(
        self,
        chans: int = 32,
        num_pool_layers: int = 4,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        dims: int = 2,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output of the U-Net model.
            drop_prob: Dropout probability.
            dims: number of dimensions for convolutional operations (2 or 3).
        """
        super().__init__()
        
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.dims = dims
        
        assert dims in [2, 3], \
        "Dimensions must be either 2 or 3"
        
        if dims == 2:
            conv_op = nn.Conv2d
        if dims == 3:
            conv_op = nn.Conv3d

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, dims)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob, dims))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob, dims)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, dims))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob, dims))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch, dims))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob, dims),
                conv_op(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape
                        - `(N, in_chans, H, W)` if dims = 2
                        - `(N, in_chans, T, H, W)` if dims = 3

        Returns:
            Output tensor of shape
                        - `(N, out_chans, H, W)` if dims = 2
                        - `(N, out_chans, T, H, W)` if dims = 3
        """
        if self.dims == 2:
            pool_op = F.avg_pool2d
        if self.dims == 3:
            pool_op = F.avg_pool3d
        
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = pool_op(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad if needed to handle odd input dimensions
            if self.dims == 2:
                padding = [0, 0, 0, 0]
            if self.dims == 3:
                padding = [0, 0, 0, 0, 0, 0]
            
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if self.dims == 3:
                if output.shape[-3] != downsample_layer.shape[-3]:
                    padding[5] = 1  # padding temporal end
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding)

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float, dims: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
            dims: number of dimensions for convolutional operations (2 or 3).
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        self.dims = dims
        
        if self.dims == 2:
            conv_op = nn.Conv2d
            norm_op = nn.InstanceNorm2d
            drop_op = nn.Dropout2d
            
        if self.dims == 3:
            conv_op = nn.Conv3d
            norm_op = nn.InstanceNorm3d
            drop_op = nn.Dropout3d      

        self.layers = nn.Sequential(
            conv_op(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            norm_op(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            drop_op(drop_prob),
            conv_op(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            norm_op(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            drop_op(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape
                        - `(N, in_chans, H, W)` if dims = 2
                        - `(N, in_chans, T, H, W)` if dims = 3

        Returns:
            Output tensor of shape
                        - `(N, out_chans, H, W)` if dims = 2
                        - `(N, out_chans, T, H, W)` if dims = 3
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int, dims:int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            dims: number of dimensions for convolutional operations (2 or 3).
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dims = dims
        
        if self.dims == 2:
            up_conv_op = nn.ConvTranspose2d
            norm_op = nn.InstanceNorm2d
            
        if self.dims == 3:
            up_conv_op = nn.ConvTranspose3d
            norm_op = nn.InstanceNorm3d

        self.layers = nn.Sequential(
            up_conv_op(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            norm_op(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input tensor of shape
                        - `(N, in_chans, H, W)` if dims = 2
                        - `(N, in_chans, T, H, W)` if dims = 3

        Returns:
            Output tensor of shape
                        - `(N, out_chans, H*2, W*2)` if dims = 2
                        - `(N, out_chans, T*2, H*2, W*2)` if dims = 3
        """
        return self.layers(image)
