import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np
import enum
from typing import Tuple

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs):
        output = F.group_norm(
            inputs.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(inputs)

def norm_block(norm_type, dim, affine=True):
    
    if norm_type == 'in':
        mod = Fp32GroupNorm(dim, dim, affine=False) # instance norm
    elif norm_type == 'ln':
        mod = Fp32GroupNorm(1, dim, affine=affine)  # layer norm
    elif norm_type == 'bn':
        mod = nn.BatchNorm1d(dim, affine=affine)

    return mod

class DSTCNBlock(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel_size, 
                 stride=1, skip=False, dilation=1):
        '''
        TDNNBlock
        '''
        super(DSTCNBlock, self).__init__()
        
        # normalization
        self.skip = skip
        
        # conv
        self.dconv = nn.Conv1d(input_dim, input_dim, kernel_size,
                               dilation=dilation, stride=stride, 
                               groups=input_dim)
        self.nonlinearity = nn.PReLU()
        self.normalization = nn.GroupNorm(1, input_dim, eps=1e-08)
        self.sconv = nn.Conv1d(input_dim, output_dim, 1, bias=False)

        if self.skip:
            self.skip_out = nn.MaxPool1d(kernel_size=kernel_size, 
                                         stride=stride,
                                         dilation=dilation)

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        out = x
        out = self.normalization(self.nonlinearity(self.dconv(out)))
        out = self.sconv(out)

        if self.skip:
            skip = self.skip_out(x)
            return out + skip
        else:
            return out

def convblock(n_in, n_out, k, 
              stride, dropout,
              norm_type, activation,
              norm_affine=True):
    return nn.Sequential(nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                         nn.Dropout(p=dropout),
                         norm_block(norm_type, dim=n_out, affine=norm_affine),
                         activation)

class ConvFilter(nn.Module):
    def __init__(self, conv_filters,
                 dropout=0.0):
        
        super().__init__()
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_filters:
            self.conv_layers.append(convblock(in_d, dim, k, stride, dropout,
                                              'in', activation=nn.ReLU()))
            in_d = dim
    
    def forward(self, x):
        # BxT -> BxCxT
        # x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        
        return x

class ConvDS(nn.Module):
    def __init__(self, ds_layers,
                 in_dim, dropout=0.0):
        super().__init__()

        in_d = in_dim
        skip = False
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in ds_layers:
            if in_d == dim and dim == 512:
                skip = True
            self.conv_layers.append(DSTCNBlock(input_dim=in_d, output_dim=dim, 
                                               kernel_size=k, stride=stride,
                                               skip=skip))
            in_d = dim
            skip = False

    def forward(self, x):
        # BxT -> BxCxT
        
        for conv in self.conv_layers:
            x = conv(x)

        return x