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

def gram_matrix(y):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

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
    def __init__(self, p=[0, 1, 2,3,4,5], q=[0, 1, 2,3,4,5],
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

        PerceptualLoss_forward = 0
        style_loss_forward = 0
        total_variation_loss_forward = 0
        L1CosineSim_forward = 0

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

          PerceptualLoss_forward += 0.1*self.content(out, gt_res) # loss_PerceptualLoss
          
          #total_loss += PerceptualLoss_forward

          style_loss_forward += 240*self.style(out, gt_res)
          #writer.add_scalar('loss/Style', style_loss, iteration)
          #total_loss += style_loss
          

          total_variation_loss_forward = 0.1*total_variation_loss(out) #tv
          
          #writer.add_scalar('loss/TV', total_variation_loss_forward, iteration)
          #total_loss += total_variation_loss_forward

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
          L1CosineSim_forward += 6*self.L1CosineSim(out, gt_res)
          
          #writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, iteration)
          #total_loss += L1CosineSim_forward
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

          CPLoss_forward = 0.1*self.CPLoss(out, gt_res)
          writer.add_scalar('loss/CPLoss', CPLoss_forward, iteration)
          total_loss += CPLoss_forward

          Contextual_Loss_forward = self.Contextual_Loss(out, gt_res)
          writer.add_scalar('loss/Contextual', Contextual_Loss_forward, iteration)
          total_loss += Contextual_Loss_forward
          """






        writer.add_scalar('loss/Perceptual', PerceptualLoss_forward, iteration)
        writer.add_scalar('loss/Style', style_loss_forward, iteration)
        writer.add_scalar('loss/TV', total_variation_loss_forward, iteration)
        writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, iteration)

        total_loss = PerceptualLoss_forward + style_loss_forward + total_variation_loss_forward + L1CosineSim_forward

        #total_loss += loss_rec + loss_PerceptualLoss + loss_style
        #loss_text += (self.w[1] * loss_prc) + (self.w[2] * loss_style) + (self.w[3] * loss_tv)


        writer.add_scalar('Total', total_loss, iteration)

        # PSNR (Peak Signal-to-Noise Ratio)
        writer.add_scalar('metrics/PSNR', self.psnr_metric(gt_res, out), iteration)

        # SSIM (Structural Similarity)
        #writer.add_scalar('metrics/SSIM', self.ssim_metric(gt_res, out), iteration)

        # AE (Average Angular Error)
        #writer.add_scalar('metrics/SSIM', ae_metric(gt_res, out), iteration)

        # MSE (Mean Square Error)
        writer.add_scalar('metrics/MSE', self.mse_metric(gt_res, out), iteration)

        # LPIPS (Learned Perceptual Image Patch Similarity)
        #writer.add_scalar('metrics/SSIM', lpips_metric(gt_res, out), iteration)

        """

        # two loops, like the original code here: https://github.com/Yukariin/DFNet/blob/master/loss.py
        total_loss = 0.0
        loss_rec_total = 0.0
        perceptual_forward_total = 0.0
        style_forward_total = 0.0
        tv_forward_total = 0.0
        
        for i in self.p:
            out = input[i]
            gt_res = resize_like(gt, out)

            (b, ch, h, w) = out.size()
            loss_rec = self.l1(out, gt_res) / (ch * h * w)
            loss_rec_total += (6 * loss_rec)
            #writer.add_scalar('loss/loss_rec', 6 * loss_rec, iteration)
            
            #total_loss += self.L1CosineSim(out, gt_res)

            #total_loss += self.RelativeL1(out, gt_res)

            #total_loss += self.ClipL1(out, gt_res)


        for i in self.q:
            out = input[i]
            gt_res = resize_like(gt, out)

            perceptual_forward = 0.1*self.content(out, gt_res) # loss_PerceptualLoss
            perceptual_forward_total += perceptual_forward
            #writer.add_scalar('loss/perceptual', perceptual_forward, iteration)

            style_forward = 240*self.style(out, gt_res)
            style_forward_total += style_forward
            #writer.add_scalar('loss/style', style_forward, iteration)

            tv_forward = 0.1*total_variation_loss(out) #tv
            tv_forward_total += tv_forward
            #writer.add_scalar('loss/tv', tv_forward, iteration)

            # new loss

            #total_loss += self.HFENLoss(out, gt_res)

            #total_loss += self.ElasticLoss(out, gt_res)

            #total_loss += self.RelativeL1(out, gt_res)

            #total_loss += self.L1CosineSim(out, gt_res)

            #total_loss += self.ClipL1(out, gt_res)

            #total_loss += self.FFTloss(out, gt_res)

            #total_loss += self.OFLoss(out)

            #total_loss += self.GPLoss(out, gt_res)

            #total_loss += self.CPLoss(out, gt_res)


        writer.add_scalar('loss/loss_rec', loss_rec_total, iteration)
        writer.add_scalar('loss/perceptual', perceptual_forward_total, iteration)
        writer.add_scalar('loss/style', style_forward_total, iteration)
        writer.add_scalar('loss/tv', tv_forward_total, iteration)

        total_loss = loss_rec_total + perceptual_forward_total + style_forward_total + tv_forward_total
        writer.add_scalar('Total', total_loss, iteration)


		# PSNR (Peak Signal-to-Noise Ratio)
        writer.add_scalar('metrics/PSNR', self.psnr_metric(gt_res, out), iteration)

		# SSIM (Structural Similarity)
		#writer.add_scalar('metrics/SSIM', self.ssim_metric(gt_res, out), iteration)

		# AE (Average Angular Error)
		#writer.add_scalar('metrics/SSIM', ae_metric(gt_res, out), iteration)

		# MSE (Mean Square Error)
        writer.add_scalar('metrics/MSE', self.mse_metric(gt_res, out), iteration)

		# LPIPS (Learned Perceptual Image Patch Similarity)
		#writer.add_scalar('metrics/SSIM', lpips_metric(gt_res, out), iteration)
        """

        #return loss_struct + loss_text
        return total_loss
