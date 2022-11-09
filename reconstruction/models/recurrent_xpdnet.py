from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import reconstruction as rec
from .xpdnet import SensitivityModel, ForwardOperator, BackwardOperator
from .denoisers.kspace_net import KSpaceCNN


class XPDNet_RNN(nn.Module):
    """
    A hybrid model for Dynamic MRI Reconstruction, inspired by combining
    XPDNet [1] and Recurrent Convolutional Neural Networks (RCNN) [2].
    
    Reference papers:
    [1] Z. Ramzi et al. `XPDNet for MRI Reconstruction: an application to the 2020 fastMRI
        challenge`. arXiv: 2010.07290, 2021.
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
        primal_only: bool = True,
        n_primal: int = 5,
        n_dual: int = 1,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for convolutional layers of the RCNN.
            primal_only: Whether to generate a buffer in k-space or only in image
                space.
            n_primal: The size of the buffer in image-space.
            n_dual: The size of the buffer in k-space.
        """
        super(XPDNet_RNN, self).__init__()
        
        self.num_cascades = num_cascades
        self.chans = chans
        self.domain_sequence = 'KI' * num_cascades
        self.i_buffer_mode = True
        self.k_buffer_mode = not primal_only
        self.i_buffer_size = n_primal
        self.k_buffer_size = 1 if primal_only else n_dual
        
        self.backward_op = BackwardOperator(masked=True)
        self.forward_op = ForwardOperator(masked=True)
        
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.bcrnn = BCRNNlayer(input_size=2*(n_primal+1), hidden_size=self.chans, kernel_size=3)
        
        if not primal_only:
            self.kspace_net = nn.ModuleList([KSpaceCNN(
                    in_chans = 2 * (n_dual+2),
                    out_chans = 2 * n_dual,
                    n_convs = 3,
                    n_filters = 16,
                ) for _ in range(num_cascades)]
            )
        else:
            self.kspace_net = [self.measurements_residual for _ in range(num_cascades)]
        
        
        self.conv1_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv1_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv2_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv2_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv3_x = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv3_h = nn.Conv2d(self.chans, self.chans, 3, padding = 3//2)
        self.conv4_x = nn.Conv2d(self.chans, 2*n_primal, 3, padding = 3//2)
        self.relu = nn.ReLU(inplace=True)
                                         
                                         
    def measurements_residual(self, concat_kspace: torch.Tensor) -> torch.Tensor:
        current_kspace = torch.stack([concat_kspace[..., 0], concat_kspace[..., 2]], dim=-1)
        ref_kspace = torch.stack([concat_kspace[..., 1], concat_kspace[..., 3]], dim=-1)
        return current_kspace - ref_kspace
        
        
    def k_domain_correction(
        self,
        i_cascade: int,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        ref_kspace: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates the kspace buffer and feeds it to the kspace net
        corresponding to the current unrolled iteration.
        """
        
        forward_op_res = rec.utils.real_to_complex_multi_ch(
            self.forward_op(image_buffer, mask, sens_maps, self.i_buffer_size), 1,
        )
        
        if self.k_buffer_mode:
            kspace_buffer = rec.utils.real_to_complex_multi_ch(kspace_buffer, self.k_buffer_size)
            kspace_buffer = torch.cat([kspace_buffer, forward_op_res], dim=-1)
        else:
            kspace_buffer = forward_op_res
            
        kspace_buffer = torch.cat(
            [kspace_buffer,
            rec.utils.real_to_complex_multi_ch(ref_kspace, 1)],
            dim=-1,
        )
        
        kspace_buffer = rec.utils.complex_to_real_multi_ch(kspace_buffer)
        return self.kspace_net[i_cascade](kspace_buffer)


    def update_image_buffer(
        self,
        image_buffer: torch.Tensor,
        kspace_buffer: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Updates the image buffer from the kspace buffer at the
        current unrolled iteration.
        """
        
        backward_op_res = rec.utils.real_to_complex_multi_ch(
            self.backward_op(kspace_buffer, mask, sens_maps, self.k_buffer_size), 1,
        )
        
        if self.i_buffer_mode:
            image_buffer = rec.utils.real_to_complex_multi_ch(image_buffer, self.i_buffer_size)
            image_buffer = torch.cat([image_buffer, backward_op_res], dim=-1)
        else:
            image_buffer = backward_op_res 
        
        image_buffer = rec.utils.complex_to_real_multi_ch(image_buffer)    
        return image_buffer
        

    def forward(self, ref_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_kspace, mask: Input 6D tensors of shape `(b, t, c, h, w, 2)`.

        Returns:
            Output tensor of shape `(b, t, h, w)`.
        """
        
        sens_maps = self.sens_net(ref_kspace, mask)
        image = self.backward_op(ref_kspace, mask, sens_maps, 1)
        
        # Generate buffers in k-space and image-space
        kspace_buffer = torch.repeat_interleave(ref_kspace, self.k_buffer_size, dim=-1)
        image_buffer = torch.repeat_interleave(image, self.i_buffer_size, dim=-1)

        b, t, h, w, ch_primal = image_buffer.squeeze(2).size()
        ch = 2 * (self.i_buffer_size + 1)
        size_h = [t*b, self.chans, h, w]
        
        # Initialise parameters of rcnn layers at the first iteration to zero
        net = {}
        rcnn_layers = 5
        for j in range(rcnn_layers-1):
            net['t0_x%d'%j] = Variable(torch.zeros(size_h)).cuda()

        # Recurrence through iterations
        for i in range(1, self.num_cascades + 1):
            
            kspace_buffer = self.k_domain_correction(
                i-1,
                image_buffer,
                kspace_buffer,
                mask,
                sens_maps,
                ref_kspace,
            )
            
            image_buffer = self.update_image_buffer(
                image_buffer,
                kspace_buffer,
                mask,
                sens_maps,
            )

            x = image_buffer.squeeze(2).permute(1,0,4,2,3)  # (t,b,ch,h,w)
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

            # Residual connection
            x_res = torch.cat(
                [x.view(-1, ch, h, w)[:, :self.i_buffer_size],
                 x.view(-1, ch, h, w)[:, self.i_buffer_size+1: -1]],
                dim = 1,
            )
            net['t%d_out'%i] = x_res + net['t%d_x4'%i]

            net['t%d_out'%i] = net['t%d_out'%i].view(t, b, 1, ch_primal, h, w)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,0,2,4,5,3)
            net['t%d_out'%i].contiguous()
            
            image_buffer = net['t%d_out'%i]


        out_image = torch.stack(
            [image_buffer[..., 0], image_buffer[..., self.i_buffer_size]],
            dim=-1,
        )
        
        return rec.utils.complex_abs(out_image.squeeze(2))


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
