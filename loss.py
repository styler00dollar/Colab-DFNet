from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from utils import resize_like


def gram_matrix(y):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def total_variation_loss(y):
    loss = (
        torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
        torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    )
    return loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.add_module('vgg', VGG16())
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        for x_feat, y_feat in zip(x_vgg, y_vgg):
            content_loss += self.criterion(x_feat, y_feat)

        return content_loss


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.add_module('vgg', VGG16())
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        style_loss = 0.0
        for x_feat, y_feat in zip(x_vgg, y_vgg):
            style_loss += self.criterion(gram_matrix(x_feat), gram_matrix(y_feat))

        return style_loss


class InpaintingLoss(nn.Module):
    def __init__(self, p=[0, 1, 2, 3, 4, 5], q=[0, 1, 2],
                 w=[6., 0.1, 240., 0.1]):
        super().__init__()

        self.l1 = nn.L1Loss()
        self.content = PerceptualLoss()
        self.style = StyleLoss()
        self.p = p
        self.q = q
        self.w = w

    def forward(self, input, gt):
        loss_struct = 0.0
        loss_text = 0.0

        for i in self.p:
            out = input[i]
            gt_res = resize_like(gt, out)

            (b, ch, h, w) = out.size()
            loss_rec = self.l1(out, gt_res) / (ch * h * w)

            loss_struct += (self.w[0] * loss_rec)

        for i in self.q:
            out = input[i]
            gt_res = resize_like(gt, out)

            loss_prc = self.content(out, gt_res)

            loss_style = self.style(out, gt_res)

            loss_tv = total_variation_loss(out)

            loss_text += (self.w[1] * loss_prc) + (self.w[2] * loss_style) + (self.w[3] * loss_tv)
        
        return loss_struct + loss_text


class VGG16(torch.nn.Module):
    def __init__(self):
        super().__init__()

        features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        for x in range(4):
            self.slice1.add_module(str(x), features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
