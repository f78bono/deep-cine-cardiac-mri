import torch
from torch import nn
from torch.nn import functional as F


class KSpaceCNN(nn.Module):
    """
    A simple CNN model performing k-space interpolation of a buffer in the
    k-space correction module of XPDNet. The model architecture consists of
    consecutive convolutional layers, each followed by an activation function.
    """
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            n_convs: int = 3,
            n_filters: int = 16,
        ):
        """
        Args:
            in_chans: Number of channels in the input to the CNN model.
            out_chans: Number of channels in the output of the CNN model.
            n_convs: Number of consecutive convolutional layers.
            n_filters: Number of convolutional filters.
        """  
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.n_convs = n_convs
        self.n_filters = n_filters

        convs = nn.ModuleList([
            nn.Conv3d(self.in_chans, self.n_filters, 3, padding='same'),
            nn.ReLU(inplace=True),
        ])
        for _ in range(1, self.n_convs-1):
            convs.append(
                nn.Conv3d(self.n_filters, self.n_filters, 3, padding='same'),
            )
            convs.append(nn.ReLU(inplace=True))
        convs.append(nn.Conv3d(self.n_filters, self.out_chans, 3, padding='same'))

        self.layers = nn.Sequential(*convs)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Input tensor of shape `(N, T, N_coils, H, W, in_chans)`
        Returns:
            Output tensor of shape `(N, T, N_coils, H, W, out_chans)`
        """

        b, t, c, h, w, ch = inputs.shape

        outputs = inputs.permute(0,2,5,1,3,4).reshape(b*c, ch, t, h, w)
        outputs = self.layers(outputs)
        outputs = outputs.reshape(b, c, self.out_chans, t, h, w).permute(0,3,1,4,5,2)

        return outputs