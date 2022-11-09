from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import reconstruction as rec
from .varnet import SensitivityModel


class VarNet_RNN(nn.Module):
    """
    A hybrid model for Dynamic MRI Reconstruction, inspired by combining
    the End-to-End Variational Network [1] and Recurrent Convolutional
    Neural Networks (RCNN) [2].
    
    Reference papers:
    [1] A. Sriram et al. `End-to-end variational networks for accelerated MRI
        reconstruction`. In International Conference on Medical Image Computing and
        Computer-Assisted Intervention, 2020.
    [2] C. Qin et al. `Convolutional Recurrent Neural Networks for Dynamic MR
        Image Reconstruction`. In IEEE Transactions on Medical Imaging 38.1,
        pp. 280–290, 2019.
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for convolutional layers of the RCNN.
        """
        super(VarNet_RNN, self).__init__()
        
        self.num_cascades = num_cascades
        self.chans = chans
        
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.bcrnn = BCRNNlayer(input_size=2, hidden_size=self.chans, kernel_size=3)
        
        self.conv1_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv1_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv2_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv2_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv3_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv3_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv4_x = nn.Conv2d(self.chans, 2, 3, padding = 3//2)
        self.relu = nn.ReLU(inplace=True)
        
        self.Softplus = nn.Softplus(1.) 
        lambda_init = np.log(np.exp(1)-1.)/1.
        self.lambda_reg = nn.Parameter(torch.tensor(lambda_init*torch.ones(1),dtype=torch.float),
                                         requires_grad=True)

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Forward operator: from coil-combined image-space to k-space.
        """
        return rec.utils.fft2c(rec.utils.complex_mul(x.permute(0,4,2,3,1).unsqueeze(2), sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Backward operator: from k-space to coil-combined image-space.
        """
        x = rec.utils.ifft2c(x)
        return rec.utils.complex_mul(x, rec.utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=False
        ).permute(0,4,2,3,1)   # b, ch, h, w, t
        
    def data_consistency(self,
        x: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        
        current_kspace = self.sens_expand(x, sens_maps)
        v = self.Softplus(self.lambda_reg)
        dc = (1 - mask) * current_kspace + mask * (current_kspace + v * ref_kspace) / (1 + v)   # b,t,c,h,w,ch
        return self.sens_reduce(dc, sens_maps)
        

    def forward(self, ref_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_kspace, mask: Input 6D tensors of shape `(b, t, c, h, w, ch)`.

        Returns:
            Output tensor of shape `(b, t, h, w)`.
        """
        sens_maps = self.sens_net(ref_kspace, mask)
        current_kspace = ref_kspace.clone()
        x = self.sens_reduce(current_kspace, sens_maps)
        
        b, ch, h, w, t = x.size()
        size_h = [t*b, self.chans, h, w]
        
        # Initialise parameters of rcnn layers at the first iteration to zero
        net = {}
        rcnn_layers = 5
        for j in range(rcnn_layers-1):
            net['t0_x%d'%j] = Variable(torch.zeros(size_h)).cuda()

        # Recurrence through iterations
        for i in range(1, self.num_cascades + 1):

            x = x.permute(4,0,1,2,3)
            x = x.contiguous()
            
            net['t%d_x0' % (i-1)] = net['t%d_x0' % (i-1)].view(t, b, self.chans, h, w)
            net['t%d_x0'%i] = self.bcrnn(x, net['t%d_x0' % (i-1)])
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1, self.chans, h, w)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i] + net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i] + net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i] + net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1, ch, h, w)
            net['t%d_out'%i] = x + net['t%d_x4'%i]

            net['t%d_out'%i] = net['t%d_out'%i].view(-1, b, ch, h, w)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,3,4,0)
            net['t%d_out'%i].contiguous()
            
            net['t%d_out'%i] = self.data_consistency(net['t%d_out'%i], ref_kspace, mask, sens_maps)
            
            x = net['t%d_out'%i]

        out = net['t%d_out'%i]
        return rec.utils.complex_abs(out.permute(0,4,2,3,1))


class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(CRNNcell, self).__init__()
        
        # Convolution for input
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # Convolution for hidden states in temporal dimension
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        # Convolution for hidden states in iteration dimension
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        input: torch.Tensor,
        hidden_iteration: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input: Input 4D tensor of shape `(b, ch, h, w)`
            hidden_iteration: hidden states in iteration dimension, 4d tensor of shape (b, hidden_size, h, w)
            hidden: hidden states in temporal dimension, 4d tensor of shape (b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(b, hidden_size, h, w)`.
        """
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(BCRNNlayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.CRNN_model = CRNNcell(input_size, self.hidden_size, kernel_size)

    def forward(self, input: torch.Tensor, hidden_iteration: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input 5D tensor of shape `(t, b, ch, h, w)`
            hidden_iteration: hidden states (output of BCRNNlayer) from previous
                    iteration, 5d tensor of shape (t, b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(t, b, hidden_size, h, w)`.
        """
        t, b, ch, h, w = input.shape
        size_h = [b, self.hidden_size, h, w]
        
        hid_init = Variable(torch.zeros(size_h)).cuda()
        output_f = []
        output_b = []
        
        # forward
        hidden = hid_init
        for i in range(t):
            hidden = self.CRNN_model(input[i], hidden_iteration[i], hidden)
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(t):
            hidden = self.CRNN_model(input[t - i - 1], hidden_iteration[t - i -1], hidden)
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if b == 1:
            output = output.view(t, 1, self.hidden_size, h, w)

        return output
