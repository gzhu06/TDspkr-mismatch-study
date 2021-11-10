import torch
import torch.nn as nn
import torch.nn.functional as F

class loss(nn.Module):

    def __init__(self, in_features, out_features, scaler=None, margin=None):
        '''
        Angular Penalty Softmax Loss
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        
        num_features: number of classes

        '''
        super(loss, self).__init__()

        self.scaler = 30.0 if not scaler else scaler
        self.margin = 0.35 if not margin else margin # 0.35 by default
        self.in_features = in_features
        self.out_features = out_features
        
        self.W = nn.Parameter(torch.randn(in_features, out_features), requires_grad=True)
        nn.init.kaiming_uniform_(self.W, 0.25)

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''

        
        W = F.normalize(self.W, p=2, dim=0)  # weight normalization
        x = F.normalize(x, p=2, dim=1)       # feature normalization
        logits = torch.mm(x, W)
        outputs = logits.clone()
        
        # AMsoftmax
        y_view = labels.view(-1, 1)
        if y_view.is_cuda: y_view = y_view.cpu()
        m = torch.zeros(x.shape[0], W.shape[1]).scatter_(1, y_view, self.margin)
        if x.is_cuda: m = m.cuda()
        logits = logits - m
        softmax_output = F.log_softmax(self.scaler * logits, dim=1)
        loss = F.nll_loss(softmax_output, labels)

        return loss, outputs