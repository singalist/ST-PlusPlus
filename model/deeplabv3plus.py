from model.resnet import resnet50, resnet101
from model.conv import conv3x3

import torch
from torch import nn
import torch.nn.functional as F


class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone, nclass, lightweight=True):
        super(DeepLabV3Plus, self).__init__()

        if backbone == 'resnet50':
            self.backbone = resnet50(True)

        low_level_channels = self.backbone.channels[0]
        high_level_channels = self.backbone.channels[-1]

        self.head = ASPPModule(high_level_channels, (12, 24, 36), lightweight)

        self.reduce = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(conv3x3(high_level_channels // 8 + 48, 256, 1, lightweight),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),

                                  conv3x3(256, 256, 1, lightweight),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Dropout(0.1, False))

        self.classifier = nn.Conv2d(256, nclass, 1, bias=True)

    def base_forward(self, x):
        h, w = x.shape[-2:]

        c1, _, _, c4 = self.backbone.base_forward(x)

        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        out = torch.cat([c1, c4], dim=1)
        out = self.fuse(out)

        out = self.classifier(out)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return out

    def forward(self, x, tta=False):
        if not tta:
            return self.base_forward(x)

        else:
            out = F.softmax(self.base_forward(x), dim=1)

            out += F.softmax(self.base_forward(x.flip(3)), dim=1).flip(3)

            return out


def ASPPConv(in_channels, out_channels, atrous_rate, lightweight):
    block = nn.Sequential(conv3x3(in_channels, out_channels, atrous_rate, lightweight),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, lightweight):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, lightweight)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, lightweight)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, lightweight)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True),
                                     nn.Dropout2d(0.5, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)