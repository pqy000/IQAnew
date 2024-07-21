import torch as torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from efficientnet import efficientnet_b0
from efficientnet import efficientnet_b1
from efficientnet import efficientnet_b2
from efficientnet import efficientnet_b3
from efficientnet import efficientnet_b4
from efficientnet import efficientnet_b5
from efficientnet import efficientnet_b6
from efficientnet import efficientnet_b7
# from resnet18and34 import resnet18
# from resnet18and34 import resnet34
# from squeeze_net import _squeezenet
# from use_resnet50 import resnet50
import matplotlib.pyplot as plt


# model_urls = {
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
# }
# resnet50rebackbonebyenffect
class LBPLayer(nn.Module):
    def __init__(self, radius=1, neighbors=8):
        super(LBPLayer, self).__init__()
        self.radius = radius
        self.neighbors = neighbors

    def forward(self, x):
        # pad image for 3x3 mask size
        x = F.pad(input=x, pad=[1, 1, 1, 1], mode='constant')
        # print(x.shape)

        b = x.shape
        M = b[2]
        N = b[3]

        y = x
        y00 = y[:, :, 0:M - 2, 0:N - 2]
        y01 = y[:, :, 0:M - 2, 1:N - 1]
        y02 = y[:, :, 0:M - 2, 2:N]
        y10 = y[:, :, 1:M - 1, 0:N - 2]
        y11 = y[:, :, 1:M - 1, 1:N - 1]
        y12 = y[:, :, 1:M - 1, 2:N]
        y20 = y[:, :, 2:M, 0:N - 2]
        y21 = y[:, :, 2:M, 1:N - 1]
        y22 = y[:, :, 2:M, 2:N]

        bit = torch.ge(y01, y11)

        tmp = torch.mul(bit, torch.tensor(1))

        bit = torch.ge(y02, y11)
        val = torch.mul(bit, torch.tensor(2))
        val = torch.add(val, tmp)

        bit = torch.ge(y12, y11)
        tmp = torch.mul(bit, torch.tensor(4))
        val = torch.add(val, tmp)

        bit = torch.ge(y22, y11)
        tmp = torch.mul(bit, torch.tensor(8))
        val = torch.add(val, tmp)

        bit = torch.ge(y21, y11)
        tmp = torch.mul(bit, torch.tensor(16))
        val = torch.add(val, tmp)

        bit = torch.ge(y20, y11)
        tmp = torch.mul(bit, torch.tensor(32))
        val = torch.add(val, tmp)

        bit = torch.ge(y10, y11)
        tmp = torch.mul(bit, torch.tensor(64))
        val = torch.add(val, tmp)

        bit = torch.ge(y00, y11)
        tmp = torch.mul(bit, torch.tensor(128))
        val = torch.add(val, tmp)
        # print(val.shape)

        return val


class BaseModel(nn.Module):
    def __init__(self, pretrained=False):
        super(BaseModel, self).__init__()
        self.efficientnet = efficientnet_b0(pretrained=True)  # efficientnet
        self.efficientnet1 = efficientnet_b0(pretrained=True)
        # self.efficientnet = resnet18(pretrained=True)#resnet18
        # self.efficientnet = resnet34(pretrained=True)
        # self.efficientnet1 = resnet34(pretrained=True)
        # self.efficientnet1 = resnet18(pretrained=True)
        # self.efficientnet1 = _squeezenet('1_1',pretrained=True)
        # self.efficientnet1 = _squeezenet('1_1',pretrained=True)
        # self.efficientnet = resnet50(pretrained=True)
        # self.efficientnet1 = resnet50(pretrained=True)

        self.lbp = LBPLayer()
        # plt.imshow(self.lbp[:,:,0])
        # plt.show()
        self.gated_fusion = GatedFusionNetwork(input_size1=1000, input_size2=1000, hidden_size=100)
        # self.lbp = nn.Sequential(
        #     LBPLayer(),
        #     # nn.Conv2d(24, 3, 1, 1, 0)
        # )
        self.score = nn.Sequential(
            # nn.Linear(100352, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        lbp = self.lbp(x)
        efficientnet_output = self.efficientnet(x)
        lbp_output = self.efficientnet1(lbp.to(torch.float32))
        fused_output = self.gated_fusion(efficientnet_output, lbp_output)
        # x = fused_output

        x = self.score(fused_output)

        # x = self.efficientnet(x)
        # lbp = self.efficientnet1(lbp.to(torch.float32))
        # x = self.score(torch.cat([x, lbp], dim=1))
        # x = self.score(x)
        # x = self.score(lbp_output)
        return x


class GatedFusionNetwork(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(GatedFusionNetwork, self).__init__()
        self.pathway1 = nn.Linear(input_size1, hidden_size)
        self.pathway2 = nn.Linear(input_size2, hidden_size)
        self.gating = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()  # gate 的计算是基于输入数据的特征表示的。在门控融合网络中，out1 和 out2 表示了不同路径的特征表示，
        )
        self.output = nn.Linear(hidden_size, 100)

    def forward(self, input1, input2):
        out1 = torch.relu(self.pathway1(input1))
        out2 = torch.relu(self.pathway2(input2))
        fused = torch.cat((out1, out2), dim=1)
        gate = self.gating(fused)

        # gated_output = gate * out1 + (1 - gate) * out2
        gated_output = gate * out1 + (1 - gate) * out2
        output = self.output(gated_output)
        # output =gated_output
        return output



