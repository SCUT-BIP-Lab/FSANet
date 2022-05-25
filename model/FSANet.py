# Demo Code for Paper:
# [Title]  - "Hand Gesture Authentication by Discovering Fine-grained Spatiotemporal Identity Characteristics"
# [Author] - Wenwei Song, Wenxiong Kang, and Liang Lin
# [Github] - https://github.com/SCUT-BIP-Lab/FSANet.git

import torch
import torch.nn as nn
import torchvision
from module.FSA_module import FSAModule


class Model_FSANet(torch.nn.Module):
    def __init__(self, frame_length, feature_dim, out_dim, tdflow_stride=2):
        super(Model_FSANet, self).__init__()
        self.out_dim = out_dim  # the identity feature dim
        # load the pretrained ResNet18
        self.model = torchvision.models.resnet18(pretrained=True)
        # change the last fc with the shape of 512×512
        self.model.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        # there are 64 frames in each dynamic hand gesture video
        self.frame_length = frame_length
        temporal_diff_length = frame_length - 1
        # tdflow lenth = (temporal_diff_frame_length - temporal_kernel_size) // tdflow_stride + 1
        self.tdflow_frame_length = (temporal_diff_length - 3) // tdflow_stride + 1

        # build TD-Flow module from Conv1 of ResNet18
        conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(tdflow_stride, 2, 2), padding=(0, 3, 3), bias=False)
        # reshape 2D conv parameters to 3D
        pretrained_conv1_params = self.model.conv1.weight.data.unsqueeze(1)
        conv1.weight.data = pretrained_conv1_params
        self.model.conv1 = conv1
        # build FSA_Module
        self.fsa_module = FSAModule(in_channels=self.model.layer3[0].conv2.out_channels)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # calculate the temporal difference map
    def getTemporalDifferenceMap(self, physical_feature):
        physical_feature = physical_feature.view((-1, self.frame_length) + physical_feature.shape[-3:])
        temporal_diff = physical_feature[:, :self.frame_length - 1, :, :, :] - physical_feature[:, 1:self.frame_length, :, :, :]
        temporal_diff = torch.sum(temporal_diff, 2)
        return temporal_diff

    def forward(self, data, label=None):
        # get 3D temporal difference map
        data_diff = self.getTemporalDifferenceMap(data).unsqueeze(dim=1)
        # get the TD-Flow
        x = self.model.conv1(data_diff)
        x = x.permute(0, 2, 1, 3, 4)
        Bt, T, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # local behavioral feature extracting
        for i in range(2):
            layer_name = "layer" + str(i + 1)
            layer = getattr(self.model, layer_name)
            x = layer(x)

        # global enhancement feature extracting and feature fusion
        x = self.model.layer3[0](x)
        bn, c, h, w = x.size()
        x = x.view(-1, self.tdflow_frame_length, c, h, w).transpose(1, 2).contiguous()
        x, dct_base = self.fsa_module(x)
        x = x.transpose(1, 2).contiguous().view(bn, c, h, w)
        # global behavioral information summarization
        x = self.model.layer3[1](x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        x = x.view(-1, self.tdflow_frame_length, self.out_dim)

        id_feature = torch.mean(x, dim=1, keepdim=False)
        id_feature = torch.div(id_feature, torch.norm(id_feature, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization for AMSoftmax

        return id_feature, dct_base


