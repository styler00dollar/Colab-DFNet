from vic.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss
from vic.filters import *
from vic.colors import *
from vic.discriminators import *


from tensorboardX import SummaryWriter
writer = SummaryWriter()

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from utils import resize_like
from metrics import *

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
    def __init__(self, p=[0, 1], q=[0, 1],
                 w=[6., 0.1, 240., 0.1]):
        super().__init__()

        self.l1 = nn.L1Loss()
        self.content = PerceptualLoss()
        self.style = StyleLoss()

        # new loss
        """
        if self.config.HFEN_TYPE == 'L1':
          l_hfen_type = nn.L1Loss()
        if self.config.HFEN_TYPE == 'MSE':
          l_hfen_type = nn.MSELoss()
        if self.config.HFEN_TYPE == 'Charbonnier':
          l_hfen_type = CharbonnierLoss()
        if self.config.HFEN_TYPE == 'ElasticLoss':
          l_hfen_type = ElasticLoss()
        if self.config.HFEN_TYPE == 'RelativeL1':
          l_hfen_type = RelativeL1()
        if self.config.HFEN_TYPE == 'L1CosineSim':
          l_hfen_type = L1CosineSim()
        """

        l_hfen_type = L1CosineSim()
        self.HFENLoss = HFENLoss(loss_f=l_hfen_type, kernel='log', kernel_size=15, sigma = 2.5, norm = False)

        self.ElasticLoss = ElasticLoss(a=0.2, reduction='mean')

        self.RelativeL1 = RelativeL1(eps=.01, reduction='mean')

        self.L1CosineSim = L1CosineSim(loss_lambda=5, reduction='mean')

        self.ClipL1 = ClipL1(clip_min=0.0, clip_max=10.0)

        self.FFTloss = FFTloss(loss_f = torch.nn.L1Loss, reduction='mean')

        self.OFLoss = OFLoss()

        self.GPLoss = GPLoss(trace=False, spl_denorm=False)

        self.CPLoss = CPLoss(rgb=True, yuv=True, yuvgrad=True, trace=False, spl_denorm=False, yuv_denorm=False)

        layers_weights = {'conv_1_1': 1.0, 'conv_3_2': 1.0}
        self.Contextual_Loss = Contextual_Loss(layers_weights, crop_quarter=False, max_1d_size=100,
            distance_type = 'cosine', b=1.0, band_width=0.5,
            use_vgg = True, net = 'vgg19', calc_type = 'regular')

        self.psnr_metric = PSNR()
        self.ssim_metric = SSIM()
        #self.ae_metric = AE()
        self.mse_metric = MSE()


        self.p = p
        self.q = q
        self.w = w

    def forward(self, input, gt, iteration):

        # just one loop
        total_loss = 0.0
        #loss_text = 0.0

        for i in self.p:
          out = input[i]
          gt_res = resize_like(gt, out)
          """
          (b, ch, h, w) = out.size()
          loss_rec = self.l1(out, gt_res) / (ch * h * w)

          l1_forward = (self.w[0] * loss_rec)
          writer.add_scalar('l1', l1_forward, iteration)
          total_loss += l1_forward
          """

          #for i in self.q:
          #out = input[i]
          #gt_res = resize_like(gt, out)

          PerceptualLoss_forward = self.content(out, gt_res) # loss_PerceptualLoss
          writer.add_scalar('loss/Perceptual', PerceptualLoss_forward, iteration)
          total_loss += PerceptualLoss_forward

          #total_loss += self.style(out, gt_res)

          total_variation_loss_forward = total_variation_loss(out) #tv
          writer.add_scalar('loss/TV', total_variation_loss_forward, iteration)
          total_loss += total_variation_loss_forward

          # new loss
          """
          HFENLoss_forward = self.HFENLoss(out, gt_res)
          writer.add_scalar('loss/HFEN', HFENLoss_forward, iteration)
          total_loss += HFENLoss_forward

          ElasticLoss_forward = self.ElasticLoss(out, gt_res)
          writer.add_scalar('loss/Elastic', HFENLoss_forward, iteration)
          total_loss += ElasticLoss_forward

          RelativeL1_forward = self.RelativeL1(out, gt_res)
          writer.add_scalar('loss/RelativeL1', HFENLoss_forward, iteration)
          total_loss += RelativeL1_forward
          """
          L1CosineSim_forward = self.L1CosineSim(out, gt_res)
          writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, iteration)
          total_loss += L1CosineSim_forward
          """
          ClipL1_forward = self.ClipL1(out, gt_res)
          writer.add_scalar('loss/ClipL1', ClipL1_forward, iteration)
          total_loss += ClipL1_forward

          FFTloss_forward = self.FFTloss(out, gt_res)
          writer.add_scalar('loss/FFTloss', FFTloss_forward, iteration)
          total_loss += FFTloss_forward

          OFLoss_forward = self.OFLoss(out)
          writer.add_scalar('loss/OFLoss', OFLoss_forward, iteration)
          total_loss += OFLoss_forward

          GPLoss_forward = self.GPLoss(out, gt_res)
          writer.add_scalar('loss/GPLoss', GPLoss_forward, iteration)
          total_loss += GPLoss_forward

          CPLoss_forward = self.CPLoss(out, gt_res)
          writer.add_scalar('loss/CPLoss', CPLoss_forward, iteration)
          total_loss += CPLoss_forward

          Contextual_Loss_forward = self.Contextual_Loss(out, gt_res)
          writer.add_scalar('loss/Contextual', Contextual_Loss_forward, iteration)
          total_loss += Contextual_Loss_forward
          """
          #total_loss += loss_rec + loss_PerceptualLoss + loss_style
        #loss_text += (self.w[1] * loss_prc) + (self.w[2] * loss_style) + (self.w[3] * loss_tv)


          writer.add_scalar('Total', total_loss, iteration)

          # PSNR (Peak Signal-to-Noise Ratio)
          writer.add_scalar('metrics/PSNR', self.psnr_metric(gt_res, out), iteration)

          # SSIM (Structural Similarity)
          writer.add_scalar('metrics/SSIM', self.ssim_metric(gt_res, out), iteration)

          # AE (Average Angular Error)
          #writer.add_scalar('metrics/SSIM', ae_metric(gt_res, out), iteration)

          # MSE (Mean Square Error)
          writer.add_scalar('metrics/MSE', self.mse_metric(gt_res, out), iteration)

          # LPIPS (Learned Perceptual Image Patch Similarity)
          #writer.add_scalar('metrics/SSIM', lpips_metric(gt_res, out), iteration)


        """
        # two loops, like the original code here: https://github.com/Yukariin/DFNet/blob/master/loss.py
        total_loss = 0.0
        for i in self.p:
            out = input[i]
            gt_res = resize_like(gt, out)

            (b, ch, h, w) = out.size()
            #loss_rec = self.l1(out, gt_res) / (ch * h * w)
            #total_loss += (self.w[0] * loss_rec)

            total_loss += self.L1CosineSim(out, gt_res)

            #total_loss += self.RelativeL1(out, gt_res)

            #total_loss += self.ClipL1(out, gt_res)


        for i in self.q:
            out = input[i]
            gt_res = resize_like(gt, out)

            total_loss += self.content(out, gt_res) # loss_PerceptualLoss

            total_loss += self.style(out, gt_res)

            total_loss += total_variation_loss(out) #tv

            # new loss

            total_loss += self.HFENLoss(out, gt_res)

            total_loss += self.ElasticLoss(out, gt_res)

            #total_loss += self.RelativeL1(out, gt_res)

            #total_loss += self.L1CosineSim(out, gt_res)

            #total_loss += self.ClipL1(out, gt_res)

            total_loss += self.FFTloss(out, gt_res)

            total_loss += self.OFLoss(out)

            total_loss += self.GPLoss(out, gt_res)

            total_loss += self.CPLoss(out, gt_res)
        """

        #return loss_struct + loss_text
        return total_loss
