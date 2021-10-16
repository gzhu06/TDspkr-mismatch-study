import torch.nn.functional as F
from torch import nn
import torch
from backbones.aggregator.tdnn_utils import TDNNLayer
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math

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
    
def norm_block(is_layer_norm, dim, affine=True, is_instance_norm=False):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        if is_instance_norm:
            mod = Fp32GroupNorm(dim, dim, affine=False) # instance norm
        else:
            mod = Fp32GroupNorm(1, dim, affine=affine)  # layer norm

    return mod
    
class TDNN_Block(nn.Module):
    def __init__(self, input_dim, output_dim=512, context_size=5, 
                 dilation=1, norm='bn', affine=True):
        super(TDNN_Block, self).__init__()
        if norm == 'bn':
            norm_layer = nn.BatchNorm1d(output_dim, affine=affine)
        elif norm == 'ln':
#             norm_layer = nn.GroupNorm(1, output_dim, affine=affine)
            norm_layer = Fp32GroupNorm(1, output_dim, affine=affine)
        elif norm == 'in':
            norm_layer = nn.GroupNorm(output_dim, output_dim, affine=False)
        else:
            raise ValueError('Norm should be {bn, ln, in}.')
        self.tdnn_layer = nn.Sequential(
            TDNNLayer(input_dim, output_dim, context_size, dilation),
            norm_layer,
            nn.ReLU())
        
    def forward(self, x):
        return self.tdnn_layer(x)

class xvecTDNN(nn.Module):
    def __init__(self, feature_dim=512, embed_dim=512, norm='bn', p_dropout=0.0):
        super(xvecTDNN, self).__init__()
        self.tdnn = nn.Sequential(
            TDNN_Block(feature_dim, 512, 5, 1, norm=norm),
            TDNN_Block(512, 512, 3, 2, norm=norm),
            TDNN_Block(512, 512, 3, 3, norm=norm),
            TDNN_Block(512, 512, 1, 1, norm=norm),
            TDNN_Block(512, 1500, 1, 1, norm=norm),
        )
        
        self.fc1 = nn.Linear(3000, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, embed_dim)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)
#         self.relu2 = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Note: x must be (batch_size, feat_dim, chunk_len)
        x = self.tdnn(x)
        
        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        
        x = self.fc1(stats)                                     # embedding a
        x = self.dropout_fc1(self.relu1(self.bn1(x)))
#         x = self.dropout_fc2(self.relu2(self.bn2(self.fc2(x)))) # embedding b
        x = self.dropout_fc2(self.bn2(self.fc2(x))) # embedding b
        
        return x

class architecture(nn.Module):
    def __init__(self, embed_dim=512):
        super(architecture, self).__init__()
        self.tdnn_aggregator = xvecTDNN(feature_dim=512, embed_dim=512, norm='bn')

    def forward(self, x):
        out = self.tdnn_aggregator(x)
        return out