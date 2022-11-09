from typing import List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import reconstruction as rec


class CineNet_RNN(nn.Module):
    """
    A hybrid model for Dynamic MRI Reconstruction, inspired by combining
    CineNet [1] and Recurrent Convolutional Neural Networks (RCNN) [2].
    
    Reference papers:
    [1] A. Kofler et al. `An end-to-end-trainable iterative network architecture 
        for accelerated radial multi-coil 2D cine MR image reconstruction.`
        In Medical Physics, 2021.
    [2] C. Qin et al. `Convolutional Recurrent Neural Networks for Dynamic MR
        Image Reconstruction`. In IEEE Transactions on Medical Imaging 38.1,
        pp. 280–290, 2019.
    """
    def __init__(
        self,
        num_cascades: int = 10,
        CG_iters: int = 4,
        chans: int = 64,
    ):
        """
        Args:
            num_cascades: Number of alternations between CG and RCNN modules.
            CG_iters: Number of  CG iterations in the CG module.
            chans: Number of channels for convolutional layers of the RCNN.
        """
        super(CineNet_RNN, self).__init__()
        
        self.num_cascades = num_cascades
        self.CG_iters = CG_iters
        self.chans = chans
        
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
        return rec.utils.fft2c(rec.utils.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Backward operator: from k-space to coil-combined image-space.
        """
        x = rec.utils.ifft2c(x)
        return rec.utils.complex_mul(x, rec.utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )
        
    def HOperator(self, x: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        The operator H = A^H \circ A + \lambda_Reg * \Id, where A is the encoding matrix.
        This ensures data consistency.
        """
        # Forward operator
        k_coils = self.sens_expand(x, sens_maps)
        # Apply sampling mask
        k_masked = k_coils * mask + 0.0
        # Backward operator
        x_combined = self.sens_reduce(k_masked, sens_maps)
        # Result of H(x)
        return x_combined + self.Softplus(self.lambda_reg) * x 
        

    def ConjGrad(self, x:torch.Tensor, b:torch.Tensor, mask:torch.Tensor, sens_maps:torch.Tensor, CG_iters:int)-> torch.Tensor:
        """
        Conjugate Gradient method for solving the system Hx = b
        """
        # x is the starting value, b the rhs
        r = self.HOperator(x, mask, sens_maps)
        r = b-r
        
        # Initialize p
        p = r.clone()
        
        # Old squared norm of residual
        sqnorm_r_old = torch.dot(r.flatten(), r.flatten())
      
        for kiter in range(CG_iters):
            # Calculate H(p)
            d = self.HOperator(p, mask, sens_maps)
            
            # Calculate step size alpha;
            inner_p_d = torch.dot(p.flatten(), d.flatten())
            alpha = sqnorm_r_old / inner_p_d
            
            # Perform step and calculate new residual
            x = torch.add(x, p, alpha = alpha.item())
            r = torch.add(r, d, alpha = -alpha.item())
            
            # New residual norm
            sqnorm_r_new = torch.dot(r.flatten(), r.flatten())
            
            # Calculate beta and update the norm
            beta = sqnorm_r_new / sqnorm_r_old
            sqnorm_r_old = sqnorm_r_new
            
            p = torch.add(r, p, alpha = beta.item())
    
        return x


    def forward(self, ref_kspace: torch.Tensor, mask: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ref_kspace, mask, sens_maps: tensors of shape `(b, t, c, h, w, ch)`.

        Returns:
            Output tensor of shape `(b, t, h, w)`.
        """
        
        x_ref = self.sens_reduce(ref_kspace, sens_maps)
        
        x = x_ref.clone().squeeze(2).permute(0,4,2,3,1)
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
            net['t%d_x0'%i] = self.bcrnn(x, net['t%d_x0'%(i-1)])
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
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,0,3,4,2).unsqueeze(2)
            net['t%d_out'%i].contiguous()
            
            net['t%d_out'%i] = self.ConjGrad(
                net['t%d_out'%i], x_ref + self.Softplus(self.lambda_reg) * net['t%d_out'%i], mask, sens_maps, self.CG_iters
            )
            net['t%d_out'%i] = net['t%d_out'%i].squeeze(2).permute(0,4,2,3,1)
            
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
