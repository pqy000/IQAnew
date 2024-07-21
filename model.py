import torchvision.models as models
import torch as torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class LBPLayer(nn.Module):
    def __init__(self, radius=1, neighbors=8):
        super(LBPLayer, self).__init__()
        self.radius = radius
        self.neighbors = neighbors

    def forward(self, x):
        # pad image for 3x3 mask size
        x = F.pad(input=x, pad=[1, 1, 1, 1], mode='constant')
        b = x.shape
        M = b[2]
        N = b[3]

        y = x
        # select elements within 3x3 mask
        y00 = y[:, :, 0:M - 2, 0:N - 2]
        y01 = y[:, :, 0:M - 2, 1:N - 1]
        y02 = y[:, :, 0:M - 2, 2:N]
        y10 = y[:, :, 1:M - 1, 0:N - 2]
        y11 = y[:, :, 1:M - 1, 1:N - 1]
        y12 = y[:, :, 1:M - 1, 2:N]
        y20 = y[:, :, 2:M, 0:N - 2]
        y21 = y[:, :, 2:M, 1:N - 1]
        y22 = y[:, :, 2:M, 2:N]

        # Apply comparisons and multiplications
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
        return val


class BaseModel(nn.Module):
    def __init__(self, pretrained=False, dropout_prob=0.5) -> None:
        super(BaseModel, self).__init__()
        # self.res = nn.Sequential(
        #     *list(models.resnext50_32x4d(pretrained=True).children())[:-1], nn.Flatten())
        # self.lbp_res = nn.Sequential(
        #     *list(models.resnext50_32x4d(pretrained=True).children())[:-1], nn.Flatten())
        self.res = resnet50_backbone(lda_out_channels=16, in_chn=128, pretrained=False)
        self.lbp_res = resnet50_backbone(lda_out_channels=16, in_chn=128, pretrained=False)
        self.lbp = nn.Sequential(
            LBPLayer(),
            # nn.Conv2d(24, 3, 1, 1, 0)
        )

        self.score = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(32, 1)

        )

    def forward(self, x):
        lbp = self.lbp(x)
        x = self.res(x)
        lbp = self.lbp_res(lbp.to(torch.float32))
        # print(x.shape)
        # print(lbp.shape)
        # print((torch.cat([x, lbp], dim=1)).shape)
        # concatenated = torch.cat([x, lbp], dim=1)
        # flattened = concatenated.view(concatenated.size(0), -1)
        # print(flattened.shape)
        x = self.score(torch.cat([x, lbp], dim=1))  # x = self.score(torch.cat([x, lbp], dim=1))

        return x


class ResNetBackbone(nn.Module):
    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)  # 归一化
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #        self.flat = nn.Sequential(
        #          nn.AvgPool2d(7, stride=7),
        #          nn.Flatten()
        #       )
        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, in_chn - lda_out_channels * 3)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

        # nn.init.kaiming_normal_(self.lda4_fc.weight.data)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # the same effect as lda operation in the paper, but save much more memory
        lda_1 = self.lda1_fc(self.lda1_pool(x).view(x.size(0), -1))
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))

        x = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items()
                      if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model


def resnet152_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet152'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    return model
