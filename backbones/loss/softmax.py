import torch.nn.functional as F
from torch import nn

class loss(nn.Module):
    
    def __init__(self, embed_dim, num_classes):
        super(loss, self).__init__()
        self.linear = nn.Linear(embed_dim, num_classes, bias=False)

    def forward(self, inputs, labels):
        """Inputs have to have dimension (N, C_in, L_in)"""
        o = self.linear(inputs)
        softmax_output = F.log_softmax(o, dim=1)
        loss = F.nll_loss(softmax_output, labels)

        return loss, softmax_output