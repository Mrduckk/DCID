# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp, sqrt
from torch.nn import L1Loss, MSELoss
from torchvision import models
import kornia
from torchvision import models as tv
import os

from collections import OrderedDict
# from utils import grid_positions, warp

def gaussian(window_size, sigma):
	gauss = torch.Tensor([exp(
			-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) \
		for x in range(window_size)])
	return gauss / gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = Variable(_2D_window.expand(
		channel, 1, window_size, window_size).contiguous())
	return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
	mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1 * mu2

	sigma1_sq = F.conv2d(
		img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(
		img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(
		img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

	C1 = 0.01 ** 2
	C2 = 0.03 ** 2

	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
			   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size=11, size_average=True):
	(_, channel, _, _) = img1.size()
	window = create_window(window_size, channel)

	if img1.is_cuda:
		window = window.cuda(img1.get_device())
	window = window.type_as(img1)

	return _ssim(img1, img2, window, window_size, channel, size_average)

class SSIMLoss(nn.Module):
	def __init__(self, window_size=11, size_average=True):
		super(SSIMLoss, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		(_, channel, _, _) = img1.size()

		if channel == self.channel and \
				self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size,
					 channel, self.size_average)

def normalize_batch(batch):
	mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
	std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
	return (batch - mean) / std
    
class VGG19(torch.nn.Module):
	def __init__(self):
		super(VGG19, self).__init__()
		features = models.vgg19(pretrained=True).features
		self.relu1_1 = torch.nn.Sequential()
		self.relu1_2 = torch.nn.Sequential()

		self.relu2_1 = torch.nn.Sequential()
		self.relu2_2 = torch.nn.Sequential()

		self.relu3_1 = torch.nn.Sequential()
		self.relu3_2 = torch.nn.Sequential()
		self.relu3_3 = torch.nn.Sequential()
		self.relu3_4 = torch.nn.Sequential()

		self.relu4_1 = torch.nn.Sequential()
		self.relu4_2 = torch.nn.Sequential()
		self.relu4_3 = torch.nn.Sequential()
		self.relu4_4 = torch.nn.Sequential()

		self.relu5_1 = torch.nn.Sequential()
		self.relu5_2 = torch.nn.Sequential()
		self.relu5_3 = torch.nn.Sequential()
		self.relu5_4 = torch.nn.Sequential()

		for x in range(2):
			self.relu1_1.add_module(str(x), features[x])

		for x in range(2, 4):
			self.relu1_2.add_module(str(x), features[x])

		for x in range(4, 7):
			self.relu2_1.add_module(str(x), features[x])

		for x in range(7, 9):
			self.relu2_2.add_module(str(x), features[x])

		for x in range(9, 12):
			self.relu3_1.add_module(str(x), features[x])

		for x in range(12, 14):
			self.relu3_2.add_module(str(x), features[x])

		for x in range(14, 16):
			self.relu3_3.add_module(str(x), features[x])

		for x in range(16, 18):
			self.relu3_4.add_module(str(x), features[x])

		for x in range(18, 21):
			self.relu4_1.add_module(str(x), features[x])

		for x in range(21, 23):
			self.relu4_2.add_module(str(x), features[x])

		for x in range(23, 25):
			self.relu4_3.add_module(str(x), features[x])

		for x in range(25, 27):
			self.relu4_4.add_module(str(x), features[x])

		for x in range(27, 30):
			self.relu5_1.add_module(str(x), features[x])

		for x in range(30, 32):
			self.relu5_2.add_module(str(x), features[x])

		for x in range(32, 34):
			self.relu5_3.add_module(str(x), features[x])

		for x in range(34, 36):
			self.relu5_4.add_module(str(x), features[x])

		# don't need the gradients, just want the features
		for param in self.parameters():
			param.requires_grad = False

	def forward(self, x):
		relu1_1 = self.relu1_1(x)
		relu1_2 = self.relu1_2(relu1_1)

		relu2_1 = self.relu2_1(relu1_2)
		relu2_2 = self.relu2_2(relu2_1)

		relu3_1 = self.relu3_1(relu2_2)
		relu3_2 = self.relu3_2(relu3_1)
		relu3_3 = self.relu3_3(relu3_2)
		relu3_4 = self.relu3_4(relu3_3)

		relu4_1 = self.relu4_1(relu3_4)
		relu4_2 = self.relu4_2(relu4_1)
		relu4_3 = self.relu4_3(relu4_2)
		relu4_4 = self.relu4_4(relu4_3)

		relu5_1 = self.relu5_1(relu4_4)
		relu5_2 = self.relu5_2(relu5_1)
		relu5_3 = self.relu5_3(relu5_2)
		relu5_4 = self.relu5_4(relu5_3)

		out = {
			'relu1_1': relu1_1,
			'relu1_2': relu1_2,

			'relu2_1': relu2_1,
			'relu2_2': relu2_2,

			'relu3_1': relu3_1,
			'relu3_2': relu3_2,
			'relu3_3': relu3_3,
			'relu3_4': relu3_4,

			'relu4_1': relu4_1,
			'relu4_2': relu4_2,
			'relu4_3': relu4_3,
			'relu4_4': relu4_4,

			'relu5_1': relu5_1,
			'relu5_2': relu5_2,
			'relu5_3': relu5_3,
			'relu5_4': relu5_4,
		}
		return out


class SWDLoss(nn.Module):
	def __init__(self):
		super(SWDLoss, self).__init__()
		self.add_module('vgg', VGG19())
		self.criterion = SWD()
		# self.SWD = SWDLocal()

	def forward(self, img1, img2, p=6):
		x = normalize_batch(img1)
		y = normalize_batch(img2)
		N, C, H, W = x.shape  # 192*192
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)

		swd_loss = 0.0
		swd_loss += self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'], k=H//4//p) * 1  # H//4=48
		swd_loss += self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'], k=H//8//p) * 1  # H//4=24
		swd_loss += self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'], k=H//16//p) * 2  # H//4=12

		return swd_loss * 8 / 100.0

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
		k = kornia.get_gaussian_kernel1d(15, 15)
		k = k.to(x.device).unsqueeze(0).unsqueeze(0)
		k = k.repeat([C, 1, 1])
		out = torch.nn.functional.conv1d(x, k, stride=1, padding=0, groups=C)
		return out

class VGG(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(VGG, self).__init__()
        vgg_pretrained_features = tv.vgg19(pretrained=pretrained).features
            
        # print(vgg_pretrained_features)
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()


        # vgg19
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])                
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,128,256,512,512]

  
    def get_features(self, x):
        # normalize the data
        h = (x-self.mean)/self.std

        
        h = self.stage1(h)
        h_relu1_2 = h
        
        h = self.stage2(h)
        h_relu2_2 = h
        
        h = self.stage3(h)
        h_relu3_3 = h
        
        h = self.stage4(h)
        h_relu4_3 = h

        h = self.stage5(h)
        h_relu5_3 = h

        # get the features of each layer
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

        return outs
       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x


class ResNet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(ResNet, self).__init__()

        model = tv.resnet101(pretrained=pretrained)
        model.eval()
        # print(model)

        self.stage1 =nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.stage2 = nn.Sequential(
            model.maxpool,
            model.layer1,
        )
        self.stage3 = nn.Sequential(
            model.layer2,
        )
        self.stage4 = nn.Sequential(
            model.layer3,
        )
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64,256,512,1024]#

    def get_features(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]#
        return outs

       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x
    
class Inception(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(Inception, self).__init__()
        inception = tv.inception_v3(pretrained=pretrained, aux_logits=False)
            
        # print(inception)
        self.stage1 = torch.nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,      
        )
        self.stage2 = torch.nn.Sequential(
            inception.maxpool1,
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3, 
        )
        self.stage3 = torch.nn.Sequential(
            inception.maxpool2, 
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
        )
        self.stage4 = torch.nn.Sequential(
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
        )
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
        self.chns = [64,192,288,768]
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

  
    def get_features(self, x):
        h = (x-self.mean)/self.std
        # h = (x-0.5)*2
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3,h_relu4_3]
        return outs
       
    def forward(self, x):
        feats_x = self.get_features(x)
        return feats_x
    

