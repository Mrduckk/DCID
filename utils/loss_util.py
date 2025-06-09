import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils.common import *
from torchvision import models as tv
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
import os
import time
from PIL import Image
from .losses import VGG, ResNet, Inception


class multi_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(multi_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1, out2, out3,img, gt1, feature_layers=[2]):
        gt2 = F.interpolate(gt1, scale_factor=0.5, mode='bilinear', align_corners=False)
        gt3 = F.interpolate(gt1, scale_factor=0.25, mode='bilinear', align_corners=False)
        
        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        loss2 = self.lam_p*self.loss_fn(out2, gt2, feature_layers=feature_layers) + self.lam*F.l1_loss(out2, gt2)
        loss3 = self.lam_p*self.loss_fn(out3, gt3, feature_layers=feature_layers) + self.lam*F.l1_loss(out3, gt3)
        
        return loss1+loss2+loss3     

class one_VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, lam=1, lam_p=1):
        super(one_VGGPerceptualLoss, self).__init__()
        self.loss_fn = VGGPerceptualLoss()
        self.lam = lam
        self.lam_p = lam_p
    def forward(self, out1,gt1, feature_layers=[2]):

        loss1 = self.lam_p*self.loss_fn(out1, gt1, feature_layers=feature_layers) + self.lam*F.l1_loss(out1, gt1)
        
        return loss1            





class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target,swd=False, feature_layers=[0, 1, 2, 3], style_layers=[]):
        # print(input.shape)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if isinstance(swd, bool):
            criterion = torch.nn.L1Loss()
        else:
            criterion = SWD()
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += criterion(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += criterion(gram_x, gram_y)
        # print('loss',loss.shape)
        return loss


class SWD(nn.Module):
	def __init__(self):
		super(SWD, self).__init__()
		self.l1loss = torch.nn.L1Loss() # torch.nn.MSELoss()#
		self.patchmatch = PatchMatchLoss()

	def forward(self, fake_samples, true_samples, k=0):
		N, C, H, W = true_samples.shape

		# s = H / 48
		# if s >= 0.5:
		# 	_, true_samples = self.patchmatch(fake_samples, true_samples, int(32*s//4))

		num_projections = C//2
		# if C==3:
		# 	num_projections = 1

		true_samples = true_samples.view(N, C, -1)
		fake_samples = fake_samples.view(N, C, -1)

		projections = torch.from_numpy(np.random.normal(size=(num_projections, C)).astype(np.float32))
		projections = torch.FloatTensor(projections).to(true_samples.device)
		projections = F.normalize(projections, p=2, dim=1)

		projected_true = projections @ true_samples
		projected_fake = projections @ fake_samples

		sorted_true, true_index = torch.sort(projected_true, dim=2)
		sorted_fake, fake_index = torch.sort(projected_fake, dim=2)

		return self.l1loss(sorted_true, sorted_fake).mean() 
		
		# sort_diff = torch.abs(sorted_true - sorted_fake)

		# soft_att = self.index_i_j(true_index, fake_index, H, W)
		# sort_diff = sort_diff * soft_att

		# return sort_diff.mean()


	def index_i_j(self, index_a, index_b, H, W):
		index_ai, index_aj = index_a // H, index_a % W
		index_bi, index_bj = index_b // H, index_b % W

		# dist = torch.abs(index_ai - index_bi) + torch.abs(index_aj - index_bj)
		# dist = 1 - dist.type(torch.float) / (H + W)

		di = torch.stack([torch.abs(index_ai - index_bi), torch.abs(index_aj - index_bj)], 3)
		# print(di.shape)
		dist = torch.max(di, dim=3)[0]
		dist = 1 - dist.type(torch.float) / H
		# print(dist, dist.shape, torch.min(dist), torch.max(dist), torch.mean(dist))	
		return dist

		# dist = torch.pow(index_ai - index_bi, 2) + torch.pow(index_aj - index_bj, 2)
		# dist = dist.type(torch.float) / (H*H + W*W)

		# dist = torch.exp((1.0 - dist) / 0.3) / exp(1/0.3) 
		# print(dist.shape)
		# dist_sum = torch.sum(dist, dim=1, keepdim=True)
		# dist = torch.div(dist, dist_sum)

		# print(dist, torch.min(dist), torch.max(dist), torch.mean(dist))		
		
		# return dist
		# return 1 - torch.sqrt(dist)

	def blur_1d(self, x):
		N, C, _ = x.shape
		k = get_gaussian_kernel1d(15, 15)
		k = k.to(x.device).unsqueeze(0).unsqueeze(0)
		k = k.repeat([C, 1, 1])
		out = torch.nn.functional.conv1d(x, k, stride=1, padding=0, groups=C)
		return out


class PatchMatchLoss(nn.Module):
	def __init__(self):
		super(PatchMatchLoss, self).__init__()
		self.l1loss = torch.nn.L1Loss()

	def bis(self, input, dim, index):
		# batch index select
		# input: [N, ?, ?, ...]
		# dim: scalar > 0
		# index: [N, idx]
		views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
		expanse = list(input.size())
		expanse[0] = -1
		expanse[dim] = -1
		index = index.view(views).expand(expanse)
		return torch.gather(input, dim, index)

	def search_org(self, lr_unfold, reflr_unfold, ks=3, pd=1, stride=1):
		# lr_unfold: [N, H*W, C*k*k]
		# reflr_unfold: [N, C*k*k, Hr*Wr]
		batch, _, _ = lr_unfold.size()

		lr_unfold = F.normalize(lr_unfold, dim=2)
		reflr_unfold = F.normalize(reflr_unfold, dim=1)
		
		# lr_unfold, _ = torch.sort(lr_unfold, dim=2)
		# reflr_unfold, _ = torch.sort(reflr_unfold, dim=1)
		# print(lr_unfold.shape, reflr_unfold.shape)
		corr = torch.bmm(lr_unfold, reflr_unfold)  # [N, H*W, Hr*Wr]
		p = int(sqrt(corr.shape[1]))
		corr = corr.view(batch, p, p, corr.shape[-1])
		
		sorted_corr, ind_l = torch.topk(corr, 1, dim=-1, largest=True, sorted=True)  # [N, H, W, num_nbr]
		return sorted_corr, ind_l

	def transfer(self, fea_unfold, index, soft_att, ks=3, pd=1, stride=1):
		# fea_unfold: [N, C*k*k, Hr*Wr]
		# index: [N, Hi, Wi]
		# soft_att: [N, 1, Hi, Wi]
		scale = self.sr_kernel
		out_unfold = self.bis(fea_unfold, 2, index)  # [N, C*k*k, Hi*Wi]
		_, Hi, Wi = index.size()
		out_fold = F.fold(out_unfold, output_size=(Hi*scale, Wi*scale), kernel_size=(ks, ks), padding=pd, stride=self.sr_kernel)
		return out_fold

	def forward(self, data_sr, data_hr, sr_kernel=16):
		# print(N, C, H, W) 16-20 > 12-16 > 16-28 12-20
		self.sr_kernel = sr_kernel
		N, C, H, W = data_sr.shape # [N, 3, 192, 192]
		hr_kernel = sr_kernel + sr_kernel # // 2
		padding = (hr_kernel-sr_kernel)//2

		data_sr_unfold = F.unfold(data_sr, kernel_size=(sr_kernel, sr_kernel), padding=0, stride=sr_kernel) # [N, 27, 4096] [N, C*k*k, H*W]
		data_hr_unfold = F.unfold(data_hr, kernel_size=(hr_kernel, hr_kernel), padding=padding, stride=sr_kernel) # [N, 363, 4096] [N, C*k*k, H*W]
		p = data_hr_unfold.shape[-1]
		# [N, 3, 3, 3, 4096] -> [N, 3, 4096, 3, 3]
		data_sr_unfold = data_sr_unfold.view(N, C, sr_kernel, sr_kernel, p).permute(0, 4, 1, 2, 3).contiguous()
		data_sr_unfold = data_sr_unfold.view(N*p, C*sr_kernel*sr_kernel).unsqueeze(dim=1)
		
		data_hr_unfold = data_hr_unfold.view(N, C, hr_kernel, hr_kernel, p).permute(0, 4, 1, 2, 3).contiguous()
		data_hr_unfold = data_hr_unfold.view(N*p, C, hr_kernel, hr_kernel)

		search_k = sr_kernel #sr_kernel//2
		padding = 0

		reflr_unfold = F.unfold(data_hr_unfold, kernel_size=(sr_kernel, sr_kernel), padding=0, stride=1) 
		corr_all_l, index_all_l = self.search_org(data_sr_unfold, reflr_unfold, ks=search_k, pd=padding, stride=1)

		index_all = index_all_l[:, :, :, 0]  # [N*p*p, k_y, k_x]
		soft_att_all = corr_all_l[:, :, :, 0:1].permute(0, 3, 1, 2)  # [N*p*p, 1, k_y, k_x]

		warp_hr = self.transfer(reflr_unfold, index_all, soft_att_all, search_k, pd=padding, stride=1)  # [N*py*px, C, k_y, k_x]
		
		warp_hr = warp_hr.view(N, H//search_k, W//search_k, C, search_k, search_k).permute(0, 3, 1, 4, 2, 5).contiguous()  # [N, C, py, H//py, px, W//px]
		warp_hr = warp_hr.view(N, C, H, W)

		return corr_all_l, warp_hr


    