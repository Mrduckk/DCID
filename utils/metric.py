from .common import SSIM, PSNR, tensor2img
import lpips
import torch
import numpy as np
from math import log10
import skimage.color as skcolor
class create_metrics():
    """
       We note that for different benchmarks, previous works calculate metrics in different ways, which might
       lead to inconsistent SSIM results (and slightly different PSNR), and thus we follow their individual
       ways to compute metrics on each individual dataset for fair comparisons.
       For our 4K dataset, calculating metrics for 4k image is much time-consuming,
       thus we benchmark evaluations for all methods with a fast pytorch SSIM implementation referred from
       "https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py".
    """
    def __init__(self, args, device):
        self.data_type = args.DATA_TYPE
        self.lpips_fn = lpips.LPIPS(net='alex').to(device=device)
        self.fast_ssim = SSIM()
        self.fast_psnr = PSNR()

    def compute(self, out_img, gt):
        res_psnr, res_ssim = self.fast_psnr_ssim(out_img, gt)
        
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)

        # calculate LPIPS
        res_lpips = self.lpips_fn.forward(pre, tar, normalize=True).item()
        res_deltaE = self.deltaE(out_img, gt)
        return res_lpips, res_psnr, res_ssim , res_deltaE


    def fast_psnr_ssim(self, out_img, gt):
        pre = torch.clamp(out_img, min=0, max=1)
        tar = torch.clamp(gt, min=0, max=1)
        # print(pre.shape)
        # print(tar.shape)
        psnr = self.fast_psnr(pre, tar)
        ssim = self.fast_ssim(pre, tar)
        return psnr, ssim


    def deltaE(self, out, gt):
        return np.mean(skcolor.deltaE_ciede2000(skcolor.rgb2lab(out.permute(0,2,3,1).cpu()), skcolor.rgb2lab(gt.permute(0,2,3,1).cpu())))

