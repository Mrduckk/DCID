
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch._jit_internal import Optional
from torch.nn import init
import math
import torchvision.models as models
import numpy as np
import torchvision.ops as ops
from einops import rearrange

from mmcv.cnn import ConvModule

class my_model_two_encoder(nn.Module):
    def __init__(self,args,
                 en_feature_num,
                 en_inter_num,
                 de_feature_num,
                 de_inter_num,
                 sam_number=1,
                 ):
        super(my_model_two_encoder, self).__init__()
        self.data_type = args.DATA_TYPE
        self.encoder_m = Encoder_mw(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=sam_number,)
        self.encoder_w = Encoder_UW(feature_num=en_feature_num, inter_num=en_inter_num, sam_number=0)
        
        self.decoder_pre = Decoder_KPA(en_num=en_feature_num,)
        self.decoder = Decoder(en_num=en_feature_num, feature_num=de_feature_num, inter_num=de_inter_num,
                            sam_number=sam_number)

        self.add_wideangle = args.ADD_WIDEANGLE


    def forward(self, x,w):
        y_m_1, y_m_2, y_m_3 = self.encoder_m(x,w,add_wideangle=False)
        y_w_1, y_w_2, y_w_3,ww = self.encoder_w(w)
        x_1, x_2, x_3 = self.decoder_pre(y_m_1, y_m_2, y_m_3, y_w_1, y_w_2, y_w_3)
        out_1, out_2, out_3 = self.decoder(x_1, x_2, x_3)
        return out_1,out_2,out_3

    
    def model_ema(self, decay=0.999):
        net_encoder = self.encoder
        net_decoder = self.decoder
        net_encoder_params = dict(net_encoder.named_parameters())
        net_encoder_ema_params = dict(self.net_encoder_ema.named_parameters())
        for k in net_encoder_ema_params.keys():
            net_encoder_ema_params[k].data.mul_(decay).add_(net_encoder_params[k].data, alpha=1 - decay)
        net_decoder_params = dict(net_decoder.named_parameters())
        net_decoder_ema_params = dict(self.net_decoder_ema.named_parameters())
        for k in net_decoder_ema_params.keys():
            net_decoder_ema_params[k].data.mul_(decay).add_(net_decoder_params[k].data, alpha=1 - decay)

 
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)


class Decoder(nn.Module):
    def __init__(self, en_num, feature_num, inter_num, sam_number):
        super(Decoder, self).__init__()
        self.preconv_3 = conv_relu(4 * en_num, feature_num, 3, padding=1)
        self.decoder_3 = Decoder_Level(feature_num, inter_num, sam_number)

        self.decoder_1 = Decoder_Level(feature_num, inter_num, sam_number)
        self.decoder_2 = Decoder_Level(feature_num, inter_num, sam_number)

        self.preconv_2 = conv_relu(2 * en_num + feature_num, feature_num, 3, padding=1)

        self.preconv_1 = conv_relu(en_num + feature_num, feature_num, 3, padding=1)

        

    def forward(self, y_1, y_2, y_3,):
        x_3 = y_3
        x_3 = self.preconv_3(x_3)
        out_3, feat_3 = self.decoder_3(x_3)

        x_2 = torch.cat([y_2, feat_3], dim=1)
        x_2 = self.preconv_2(x_2)
        out_2, feat_2 = self.decoder_2(x_2)

        x_1 = torch.cat([y_1, feat_2], dim=1)
        x_1 = self.preconv_1(x_1)
        out_1 = self.decoder_1(x_1, feat=False)

        return out_1, out_2, out_3



class Decoder_KPA(nn.Module):
    def __init__(self, en_num,warp_UW=False):
        super(Decoder_KPA, self).__init__()

        self.a_3 = nn.Parameter(torch.tensor(0.0))
        self.a_2 = nn.Parameter(torch.tensor(0.0))
        self.a_1 = nn.Parameter(torch.tensor(0.0))
        self.upconv_1 = nn.ConvTranspose2d(2 * en_num, 3, kernel_size=4, stride=2, padding=1)
        self.FusionNetwork_3 = FusionNetwork_KPA_warp(4 * en_num,)
        self.FusionNetwork_2 = FusionNetwork_KPA_warp(2 * en_num,)
        self.FusionNetwork_1 = FusionNetwork_KPA_warp(en_num,)


    def forward(self, y_m_1, y_m_2, y_m_3, y_w_1, y_w_2, y_w_3):
        
        # 对 y_w_* 进行 warp 操作
        y_m_3_new, y_w_3_new  = self.FusionNetwork_3(y_m_3, y_w_3)
        y_m_2_new, y_w_2_new  = self.FusionNetwork_2(y_m_2, y_w_2)
        y_m_1_new, y_w_1_new  = self.FusionNetwork_1(y_m_1, y_w_1)

        x_3 = y_m_3_new+self.a_3*y_w_3_new
        x_2 = y_m_2_new+self.a_2*y_w_2_new
        x_1 = y_m_1_new+self.a_1*y_w_1_new

        return x_1, x_2, x_3

class FusionNetwork_KPA_warp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.KPA = KPA(channels)

    def forward(self, moire, uw):
        KPA_uw = self.KPA(uw,moire)
        
        return moire, KPA_uw

class KPA(nn.Module): ### KPA
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=2,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim*2, 
                       dim//reduction_ratio,
                       kernel_size=1,
                    #    norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'),),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,  bias=True),
            # nn.Conv2d(dim, dim, kernel_size = 5, stride = 1, padding = 2, groups = dim, bias = False)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x,moire):
        B, C, H, W = x.shape 
        xx = self.pool(x)
        mm = self.pool(moire)
        feat_cat = torch.cat((mm,xx), dim=1)
        scale = self.proj(feat_cat).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(feat_cat, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None
        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K//2,
                     groups=B*C,
                     bias=bias)
        
        return x.reshape(B, C, H, W)



class Encoder_UW(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder_UW, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level_uw(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level_uw(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level_uw(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x):
        # print('pix',x.shape)
        downscale=2
        x = F.pixel_unshuffle(x, downscale_factor=downscale)
        x = self.conv_first(x)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)

        return out_feature_1, out_feature_2, out_feature_3,x

class Encoder_mw(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Encoder_mw, self).__init__()
        self.conv_first = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_first_w = nn.Sequential(
            nn.Conv2d(12, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True)
        )
        self.conv_first_con = nn.Sequential(
            nn.Conv2d(feature_num*2, feature_num, kernel_size=5, stride=1, padding=2, bias=True),
            # nn.ReLU(inplace=True)
        )
        self.encoder_1 = Encoder_Level(feature_num, inter_num, level=1, sam_number=sam_number)
        self.encoder_2 = Encoder_Level(2 * feature_num, inter_num, level=2, sam_number=sam_number)
        self.encoder_3 = Encoder_Level(4 * feature_num, inter_num, level=3, sam_number=sam_number)

    def forward(self, x,w,add_wideangle=False):
        # print('pix',x.shape)
        downscale=2
        x = F.pixel_unshuffle(x, downscale_factor=downscale)
        x = self.conv_first(x)
        # print('here')
        if add_wideangle!=False:
            w1 = F.pixel_unshuffle(w, downscale_factor=downscale)
            w1 = self.conv_first_w(w1)
            x_concat = torch.cat((x, w1), dim=1)
            x = self.conv_first_con(x_concat)

        out_feature_1, down_feature_1 = self.encoder_1(x)
        out_feature_2, down_feature_2 = self.encoder_2(down_feature_1)
        out_feature_3 = self.encoder_3(down_feature_2)
        
        return out_feature_1, out_feature_2, out_feature_3

class Encoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number):
        super(Encoder_Level, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1, 2, 1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        for sam_block in self.sam_blocks:
            out_feature = sam_block(out_feature)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature
    
class Encoder_Level_uw(nn.Module):
    def __init__(self, feature_num, inter_num, level, sam_number=0):
        super(Encoder_Level_uw, self).__init__()
        self.rdb = RDB(in_channel=feature_num, d_list=(1,1), inter_num=inter_num)
        self.sam_blocks = nn.ModuleList()

        if level < 3:
            self.down = nn.Sequential(
                nn.Conv2d(feature_num, 2 * feature_num, kernel_size=3, stride=2, padding=1, bias=True),
                nn.ReLU(inplace=True)
            )
        self.level = level

    def forward(self, x):
        out_feature = self.rdb(x)
        if self.level < 3:
            down_feature = self.down(out_feature)
            return out_feature, down_feature
        return out_feature


class Decoder_Level(nn.Module):
    def __init__(self, feature_num, inter_num, sam_number):
        super(Decoder_Level, self).__init__()
        self.rdb = RDB(feature_num, (1, 2, 1), inter_num)
        self.sam_blocks = nn.ModuleList()
        for _ in range(sam_number):
            sam_block = SAM(in_channel=feature_num, d_list=(1, 2, 3, 2, 1), inter_num=inter_num)
            self.sam_blocks.append(sam_block)
        self.conv = conv(in_channel=feature_num, out_channel=12, kernel_size=3, padding=1)

    def forward(self, x, feat=True):
        x = self.rdb(x)
        for sam_block in self.sam_blocks:
            x = sam_block(x)
        out = self.conv(x)
        out = F.pixel_shuffle(out, 2)

        if feat:
            feature = F.interpolate(x, scale_factor=2, mode='bilinear')
            return out, feature
        else:
            return out


class DB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(DB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t

class SAM(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(SAM, self).__init__()
        self.basic_block = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_2 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.basic_block_4 = DB(in_channel=in_channel, d_list=d_list, inter_num=inter_num)
        self.fusion = CSAF(3 * in_channel)

    def forward(self, x):
        x_0 = x
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        
        y_0 = self.basic_block(x_0)
        y_2 = self.basic_block_2(x_2)
        y_4 = self.basic_block_4(x_4)
        y_2 = F.interpolate(y_2, scale_factor=2, mode='bilinear')
        y_4 = F.interpolate(y_4, scale_factor=4, mode='bilinear')
        
        y = self.fusion(y_0, y_2, y_4)
        y = x + y

        return y


class CSAF(nn.Module):
    def __init__(self, in_chnls, ratio=4):
        super(CSAF, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.compress2 = nn.Conv2d(in_chnls // ratio, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x0, x2, x4):
        out0 = self.squeeze(x0)
        out2 = self.squeeze(x2)
        out4 = self.squeeze(x4)
        out = torch.cat([out0, out2, out4], dim=1)
        out = self.compress1(out)
        out = F.relu(out)
        out = self.compress2(out)
        out = F.relu(out)
        out = self.excitation(out)
        out = F.sigmoid(out)
        w0, w2, w4 = torch.chunk(out, 3, dim=1)
        x = x0 * w0 + x2 * w2 + x4 * w4

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)

        t = self.conv_post(t)
        return t + x


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out



