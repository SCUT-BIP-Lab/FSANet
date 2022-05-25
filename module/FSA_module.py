# Demo Code for Paper:
# [Title]  - "Hand Gesture Authentication by Discovering Fine-grained Spatiotemporal Identity Characteristics"
# [Author] - Wenwei Song, Wenxiong Kang, and Liang Lin
# [Github] - https://github.com/SCUT-BIP-Lab/FSANet.git

from torch import nn
import torch
import torch.nn.functional as F

class FSAModule(nn.Module):
    def __init__(self,  in_channels: int, reduction=2, reconstruct=True):

        super(FSAModule, self).__init__()
        self.reduction = reduction
        self.in_channels = in_channels
        self.reconstruct = reconstruct
        # convD is used to learn selected bases
        self.convD = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        # convV is used to adapt and compress the input feature
        self.convV = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        # convA is used to learn attentional queries
        self.convA = nn.Conv3d(in_channels, in_channels // self.reduction, kernel_size=1)
        self.tanh = nn.Tanh()
        # global_adapt is used to adapt and reconstruct the global enhancement feature
        self.global_adapt = nn.Sequential(
                nn.Conv3d(in_channels // self.reduction, in_channels, kernel_size=1),
                nn.BatchNorm3d(in_channels)
            )

        nn.init.constant_(self.global_adapt[1].weight, 0)
        nn.init.constant_(self.global_adapt[1].bias, 0)

    def forward(self, x: torch.Tensor):

        batch_size, c, t, h, w = x.size()
        n = t * h * w
        assert c == self.in_channels, 'input channel not equal!'
        V = self.convV(x)  # (b, c, t, h, w); adapt and compress the input feature;
        D = self.convD(x)  # (b, k, t, h, w); learn k selected bases;
        A = self.convA(x)  # (b, c, t, h, w); learn attentional queries;

        value = V.view(batch_size, c // self.reduction, n) # compressed input feature
        dct_base = D.view(batch_size, c // self.reduction, n) # bases without normalization
        attention_vectors = A.view(batch_size, c // self.reduction, n) # attentional queries
        # we perform tanh activation and vector normalization successively on the convolution result
        # to obtain the selected bases
        dct_base = self.tanh(dct_base)
        dct_base_l2 = torch.norm(dct_base, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        dct_base = torch.div(dct_base, dct_base_l2)
        # acquire K = c//2 frequency domain features
        spectrum = torch.bmm(value, dct_base.permute(0, 2, 1))  # (b, c, k)
        global_descriptors = spectrum.permute(0, 2, 1) # (b, k, c)

        attention_vectors = F.softmax(attention_vectors, dim=1)  # (b, c, n)
        # calculate the global enhancement feature for each local feature
        global_info = global_descriptors.matmul(attention_vectors)  # (b, k, n)
        global_info = global_info.view(batch_size, c // self.reduction, t, h, w) # (b, k, t, h, w)
        global_info = self.global_adapt(global_info) # (b, c, t, h, w)
        y = global_info + x

        return y, dct_base