# Demo Code for Paper:
# [Title]  - "Hand Gesture Authentication by Discovering Fine-grained Spatiotemporal Identity Characteristics"
# [Author] - Wenwei Song, Wenxiong Kang, and Liang Lin
# [Github] - https://github.com/SCUT-BIP-Lab/FSANet.git

import torch
from model.FSANet import Model_FSANet
# from loss.loss import AMSoftmax

def feedforward_demo(frame_length, feature_dim, out_dim):
    model = Model_FSANet(frame_length=frame_length, feature_dim=feature_dim, out_dim=out_dim)
    # AMSoftmax loss function
    # criterian = AMSoftmax(in_feats=out_dim, n_classes=143)
    # there are 143 identities in the training set
    data = torch.randn(2, 64, 3, 224, 224) #batch, frame, channel, h, w
    data = data.view(-1, 3, 224, 224) #regard the frame as batch
    id_feature, dct_base = model(data) # feedforward
    # Use the id_feature to calculate the EER when testing or to calculate the loss when training
    # when training
    # loss_backbone, _ = self.criterian(id_feature, label)
    # th = math.cos(30 / 180 * math.pi)  # cos(30)
    # eye = torch.eye(128, requires_grad=False)
    # relu = nn.ReLU(inplace=True)
    # cos_matrix = torch.bmm(dct_base, dct_base.permute(0, 2, 1)) - eye
    # cos_matrix = relu(cos_matrix.abs() - th)
    # dct_loss = cos_matrix.sum(dim=-1)
    # dct_loss = dct_loss.mean()
    # loss = loss_backbone + 0.01 * dct_loss
    return id_feature

if __name__ == '__main__':
    # there are 64 frames in each dynamic hand gesture video
    frame_length = 64
    # the feature dim of last feature map (layer4) from ResNet18 is 512
    feature_dim = 512
    # the identity feature dim
    out_dim = 512

    # feedforward process
    id_feature = feedforward_demo(frame_length, feature_dim, out_dim)
    print("Demo is finished!")

