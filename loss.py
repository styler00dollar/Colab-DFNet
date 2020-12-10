resume_iteration = 0

from vic.loss import CharbonnierLoss, GANLoss, GradientPenaltyLoss, HFENLoss, TVLoss, GradientLoss, ElasticLoss, RelativeL1, L1CosineSim, ClipL1, MaskedL1Loss, MultiscalePixelLoss, FFTloss, OFLoss, L1_regularization, ColorLoss, AverageLoss, GPLoss, CPLoss, SPL_ComputeWithTrace, SPLoss, Contextual_Loss, StyleLoss
from vic.perceptual_loss import PerceptualLoss
from vic.filters import *
from vic.colors import *
from vic.discriminators import *


from tensorboardX import SummaryWriter

logdir='/path/'

writer = SummaryWriter(logdir=logdir)

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models

from utils import resize_like
from metrics import *

from torchvision.utils import save_image


class InpaintingLoss(nn.Module):
    def __init__(self, p=[0, 1, 2,3,4,5], q=[0, 1, 2,3,4,5],
                 w=[6., 0.1, 240., 0.1]):
        super().__init__()

        #self.l1 = nn.L1Loss()
        #self.perceptual = PerceptualLoss()
        #self.style = StyleLoss()

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

        self.StyleLoss = StyleLoss()

        self.TVLoss = TVLoss(tv_type='tv', p = 1)

        self.PerceptualLoss = PerceptualLoss(model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True, gpu_ids=[0], model_path=None)

        layers_weights = {'conv_1_1': 1.0, 'conv_3_2': 1.0}
        self.Contextual_Loss = Contextual_Loss(layers_weights, crop_quarter=False, max_1d_size=100,
            distance_type = 'cosine', b=1.0, band_width=0.5,
            use_vgg = True, net = 'vgg19', calc_type = 'regular')

        self.psnr_metric = PSNR()
        self.ssim_metric = SSIM()
        self.ae_metric = AE()
        self.mse_metric = MSE()


    def forward(self, input, gt, iteration):

        # just one loop
        total_loss = 0.0

        L1CosineSim_forward = 0.0
        perceptual_forward = 0.0
        style_forward = 0.0
        tv_forward = 0.0
        PSNR_value = 0.0


        # Input batchsize here
        for i in range(24):
          out = input[0][i]
          gt_res = gt[i]

          #gt_res = resize_like(gt, out)

          out = out.unsqueeze(0)
          gt_res = gt_res.unsqueeze(0)

          # new loss
          """
          HFENLoss_forward = self.HFENLoss(out, gt_res)
          total_loss += HFENLoss_forward

          ElasticLoss_forward = self.ElasticLoss(out, gt_res)
          total_loss += ElasticLoss_forward

          RelativeL1_forward = self.RelativeL1(out, gt_res)
          total_loss += RelativeL1_forward
          """
          L1CosineSim_forward += 5*self.L1CosineSim(out, gt_res)
          #total_loss += L1CosineSim_forward

          #writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, iteration)
          #total_loss += L1CosineSim_forward
          """
          ClipL1_forward = self.ClipL1(out, gt_res)
          total_loss += ClipL1_forward

          FFTloss_forward = self.FFTloss(out, gt_res)
          total_loss += FFTloss_forward

          OFLoss_forward = self.OFLoss(out)
          total_loss += OFLoss_forward

          GPLoss_forward = self.GPLoss(out, gt_res)
          total_loss += GPLoss_forward

          CPLoss_forward = 0.1*self.CPLoss(out, gt_res)
          total_loss += CPLoss_forward

          Contextual_Loss_forward = self.Contextual_Loss(out, gt_res)
          total_loss += Contextual_Loss_forward
          """

          style_forward += 240*self.StyleLoss(out, gt_res)
          #total_loss += style_forward

          tv_forward += 0.0000005*self.TVLoss(out)
          #total_loss += tv_forward

          perceptual_forward += 2*self.PerceptualLoss(out, gt_res)
          #total_loss += perceptual_forward

          PSNR_value += self.psnr_metric(gt_res, out)


        writer.add_scalar('loss/Perceptual', perceptual_forward, iteration + resume_iteration)
        writer.add_scalar('loss/Style', style_forward, iteration + resume_iteration)
        writer.add_scalar('loss/TV', tv_forward, iteration + resume_iteration)
        writer.add_scalar('loss/L1CosineSim', L1CosineSim_forward, iteration + resume_iteration)

        total_loss = perceptual_forward + style_forward + tv_forward + L1CosineSim_forward

        #total_loss += loss_rec + loss_PerceptualLoss + loss_style
        #loss_text += (self.w[1] * loss_prc) + (self.w[2] * loss_style) + (self.w[3] * loss_tv)


        writer.add_scalar('Total', total_loss, iteration + resume_iteration)
        writer.add_scalar('metrics/PSNR', PSNR_value/24, iteration + resume_iteration)


        # PSNR (Peak Signal-to-Noise Ratio)
        #writer.add_scalar('metrics/PSNR', self.psnr_metric(gt_res, out), iteration)
        #writer.add_scalar('metrics/PSNR', PSNR_value, iteration+resume_iteration)

        # SSIM (Structural Similarity)
        #writer.add_scalar('metrics/SSIM', self.ssim_metric(gt_res, out), iteration)

        # AE (Average Angular Error)
        #writer.add_scalar('metrics/AE', self.ae_metric(gt_res, out), iteration)

        # MSE (Mean Square Error)
        #writer.add_scalar('metrics/MSE', self.mse_metric(gt_res, out), iteration)

        # LPIPS (Learned Perceptual Image Patch Similarity)
        # pip install LPIPS
        #writer.add_scalar('metrics/SSIM', lpips_metric(gt_res, out), iteration)

        return total_loss
