import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils.loss_util import *
from utils.common import *
from torch.nn.parameter import Parameter
from functools import partial
import time
# from model.pwcnet import get_backwarp
from lightglue.utils import load_image, rbd
import kornia
# from utils.utils import InputPadder, forward_interpolate


def model_fn_decorator(args,loss_fn,logger, device,blur=False, mode='train'):


    def test_model_fn_mw(args, data, model, save_path, compute_metrics,batch_idx,epoch):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0
        cur_deltaE = 0.0


        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        w_img = data['w_img'].to(device)
        b, c, h, w = in_img.size()

        moire = in_img
        w_img_= w_img
        gt=label

        b, c, h_w, w_w = w_img.size()
        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1
        in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
        w_img = img_pad(w_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)

        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            st = time.time()
            
            out_1, out_2, out_3 = model(in_img,w_img)

            torch.cuda.synchronize()
            cur_time = time.time()-st
            if h_odd_pad != 0:
               out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_odd_pad != 0:
               out_1 = out_1[:, :, :, w_pad:-w_odd_pad]
            

        if args.EVALUATION_METRIC:
            cur_lpips, cur_psnr, cur_ssim, cur_deltaE = compute_metrics.compute(out_1, label)

        
        if batch_idx % args.SAVE_ITER == 0:
            logger.add_image('test/input', in_img[0], epoch)
            logger.add_image('test/out', out_1[0], epoch)
            logger.add_image('test/gt', label[0], epoch)
        # save images
        if args.SAVE_IMG:
            out_1 = out_1.detach().cpu()
            w_img = w_img_.detach().cpu()
            label = label.detach().cpu()
            gt = gt.detach().cpu()
            moire = moire.detach().cpu()
            torchvision.utils.save_image(gt, save_path + '/' + 'test_%s_gt' % number[0] + '.%s' % args.SAVE_IMG)
            torchvision.utils.save_image(moire, save_path + '/' + 'test_%s_moire' % number[0] + '.%s' % args.SAVE_IMG)
            torchvision.utils.save_image(w_img, save_path + '/' + 'test_%s_uw' % number[0] + '.%s' % args.SAVE_IMG)
            torchvision.utils.save_image(out_1, save_path + '/' + 'test_%s_g_out' % number[0] + '.%s' % args.SAVE_IMG)
            
        return cur_psnr, cur_ssim, cur_lpips, cur_deltaE, cur_time

    def test_model_fn_mw_align(args, data, model, save_path, compute_metrics,batch_idx,epoch,extractor,matcher):
        # prepare input and forward
        number = data['number']
        cur_psnr = 0.0
        cur_ssim = 0.0
        cur_lpips = 0.0
        cur_deltaE = 0.0


        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        w_img = data['w_img'].to(device)
        b, c, h, w = in_img.size()
        moire = in_img
        w_img_=w_img
        gt=label

        # pad image such that the resolution is a multiple of 32
        w_pad = (math.ceil(w/32)*32 - w) // 2
        h_pad = (math.ceil(h/32)*32 - h) // 2
        w_odd_pad = w_pad
        h_odd_pad = h_pad
        if w % 2 == 1:
            w_odd_pad += 1
        if h % 2 == 1:
            h_odd_pad += 1

        
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            st = time.time()

            w_img = lightglue_align(extractor,matcher,in_img,w_img)
            
            in_img = img_pad(in_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
            w_img = img_pad(w_img, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
            # w_img =torch.randn(1, 3, 2304, 3840).to(device)

            out_1, out_2, out_3 = model(in_img,w_img)
            torch.cuda.synchronize()
            cur_time = time.time()-st
            if h_odd_pad != 0:
               out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_odd_pad != 0:
               out_1 = out_1[:, :, :, w_pad:-w_odd_pad]

        if args.EVALUATION_METRIC:
            cur_lpips, cur_psnr, cur_ssim, cur_deltaE = compute_metrics.compute(out_1, label)

        
        if batch_idx % args.SAVE_ITER == 0:
            logger.add_image('test/input', in_img[0], epoch)
            logger.add_image('test/out', out_1[0], epoch)
            logger.add_image('test/gt', label[0], epoch)
        # save images
        if args.SAVE_IMG:
            out_1 = out_1.detach().cpu()
            w_img = w_img_.detach().cpu()
            label = label.detach().cpu()
            gt = gt.detach().cpu()
            moire = moire.detach().cpu()
            out_save = torch.cat((gt,moire,w_img,out_1),3)
            torchvision.utils.save_image(out_save, save_path + '/' + 'test_%s' % number[0] + '.%s' % args.SAVE_IMG)
            
        return cur_psnr, cur_ssim, cur_lpips, cur_deltaE, cur_time


    def lightglue_align(extractor,matcher,image0,image1):

        img_src = image0
        img_dst = image1
        image0 = F.interpolate(image0, scale_factor=0.25, mode='bilinear') 
        image1 = F.interpolate(image1, scale_factor=0.25, mode='bilinear')
        
        feats0 = extractor.extract(image0.squeeze(0).to(device),resize=None)
        feats1 = extractor.extract(image1.squeeze(0).to(device),resize=None)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]  # remove batch dimension
        # print(feats0)
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
        # print(m_kpts0[:5])

        pts_src = np.float32([list(pt) for pt in m_kpts0.cpu()])
        pts_dst = np.float32([list(pt) for pt in m_kpts1.cpu()])
        
        flag = 0
        if len(pts_dst) < 4 or len(pts_src) < 4:
            H = torch.eye(3, dtype=torch.float32).to(device)
        else:
            H = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC)[0]
            if H is None:
                H = torch.eye(3, dtype=torch.float32).to(device)
            else:
                H = torch.tensor(H, dtype=torch.float32).to(device)
                flag = 1

        
        scale_factor = 4

        
        S = torch.tensor([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=torch.float32).to(device)
        S_inv = torch.tensor([[1/scale_factor, 0, 0], [0, 1/scale_factor, 0], [0, 0, 1]], dtype=torch.float32).to(device)
        
        
        H_original = S @ H @ S_inv
        warped_img = kornia.geometry.warp_perspective(img_dst, H_original.unsqueeze(0), (img_src.shape[2], img_src.shape[3]))
        
        if flag == 0:
            return None
        return warped_img


    def model_fn_mw(args, data, model, iters,epoch,cube=False):
        model.train()
        # prepare input and forward
        in_img = data['in_img'].to(device)
        label = data['label'].to(device)
        w_img = data['w_img'].to(device)
        b,c,h,w = in_img.shape

        
        out_1,out_2,out_3= model(in_img,w_img)
        loss = loss_fn(out_1, out_2, out_3,in_img, label)


        # save images
        if epoch % args.SAVE_EPOCH == (args.SAVE_EPOCH-1) and iters % args.SAVE_ITER == (args.SAVE_ITER - 1):
            in_save = in_img.detach().cpu()
            w_save = w_img.detach().cpu()
            out_save = out_1.detach().cpu()
            gt_save = label.detach().cpu()
            res_save = torch.cat((in_save, w_save, out_save, gt_save), 3)
            save_number = (iters + 1) // args.SAVE_ITER
            save_dir = os.path.join(args.VISUALS_DIR, f"epoch_{epoch:03d}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = f"{args.VISUALS_DIR}/epoch_{epoch:03d}/visual_x{args.SAVE_ITER:04d}_{save_number:05d}.jpg"
            torchvision.utils.save_image(res_save, save_path)
            # torchvision.utils.save_image(res_save,
            #                              args.VISUALS_DIR + '/epoch_%03d_' % epoch + '/visual_x%04d_' % args.SAVE_ITER + '%05d' % save_number + '.jpg')


        return loss
      


    if mode == 'test' and args.DATA_TYPE == 'DCID' and args.EVALUATION_TIME == True and args.ADD_ALIGN_TIME == True:
        fn = test_model_fn_mw_align
    elif mode == 'test' and args.DATA_TYPE == 'DCID':
        fn = test_model_fn_mw
    elif args.DATA_TYPE == 'DCID':
        fn = model_fn_mw
    return fn