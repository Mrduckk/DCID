import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import datetime
import logging
import lpips
import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision
import random
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
import os
from model.model import model_fn_decorator
from model.nets import my_model_two_encoder

from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
from config.config import args
import logging
from collections import OrderedDict

from lightglue import LightGlue, SuperPoint, DISK


def test(args, TestImgLoader, model, model_fn_test, save_path, compute_metrics, device, epoch, logger, extractor=None, matcher=None):
    tbar = tqdm(TestImgLoader)
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_deltaE = 0
    total_time = 0
    avg_val_time = 0

    if args.EVALUATION_TIME == True and args.ADD_ALIGN_TIME:
        extractor = SuperPoint(max_num_keypoints=256).eval().to(device)  # load the extractor
        matcher = LightGlue(features="superpoint").eval().to(device)
    torch.cuda.synchronize()
    
    for batch_idx, data in enumerate(tbar):
        model.eval()

        number = data['number']
        
        if args.EVALUATION_TIME == True and args.ADD_ALIGN_TIME:
            cur_psnr, cur_ssim, cur_lpips, cur_deltaE, cur_time = model_fn_test(args, data, model, save_path, compute_metrics,batch_idx,epoch,extractor,matcher)
        else:
            cur_psnr, cur_ssim, cur_lpips, cur_deltaE, cur_time = model_fn_test(args, data, model, save_path, compute_metrics,batch_idx,epoch)
        if cur_psnr == -1:
            continue
        if args.EVALUATION_METRIC:
            logging.info('%s: LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f, and deltaE is %.4f' % (number[0], cur_lpips, cur_psnr, cur_ssim, cur_deltaE))
        if args.EVALUATION_TIME:
            logging.info('%s: TIME is %.4f' % (number[0], cur_time))
        total_psnr += cur_psnr
        avg_val_psnr = total_psnr / (batch_idx+1)
        total_ssim += cur_ssim
        avg_val_ssim = total_ssim / (batch_idx+1)
        total_deltaE += cur_deltaE
        avg_val_deltaE = total_deltaE / (batch_idx+1)
        total_lpips += cur_lpips
        avg_val_lpips = total_lpips / (batch_idx+1)
        # skip calculation for first five samples to avoid warming-up cost
        if batch_idx > 5:
            total_time += cur_time
            avg_val_time = total_time / (batch_idx-5)
        if args.EVALUATION_METRIC:
            desc = 'Test: Avg. LPIPS = %.4f, Avg. PSNR = %.4f and SSIM = %.4f , and deltaE is %.4f' % (avg_val_lpips, avg_val_psnr, avg_val_ssim,avg_val_deltaE)
        elif args.EVALUATION_TIME:
            desc = 'Avg. TIME is %.4f' % avg_val_time
        else:
            desc = 'Test without any evaluation'
        tbar.set_description(desc)
        tbar.update()

    torch.cuda.synchronize()
    if args.EVALUATION_METRIC:
        logger.add_scalar('Test/psnr', cur_psnr, epoch)
        logger.add_scalar('Test/ssim', cur_ssim, epoch)
        logger.add_scalar('Test/cur_deltaE',cur_deltaE, epoch)
        logger.add_scalar('Test/lpips', cur_lpips, epoch)
        logging.warning('Avg. LPIPS is %.4f, PSNR is %.4f and SSIM is %.4f , and deltaE is %.4f' % (avg_val_lpips, avg_val_psnr, avg_val_ssim,avg_val_deltaE))
    if args.EVALUATION_TIME:
        logging.warning('Avg. TIME is %.4f' % avg_val_time)

def init():
    
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_logs')
    mkdir(args.LOGS_DIR)
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    mkdir(args.TEST_RESULT_DIR)
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.GPU_ID
    
    torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    
    logger = SummaryWriter(args.LOGS_DIR)
    return logger,device

def load_checkpoint(model):
    if args.LOAD_PATH:
        load_path = args.LOAD_PATH
        save_path = args.TEST_RESULT_DIR + '/customer'
        log_path = args.TEST_RESULT_DIR + '/customer_result.log'
        epoch = 0

    else:
        load_epoch = args.TEST_EPOCH
        if load_epoch == 'auto':
            load_path = args.NETS_DIR + '/checkpoint_latest.tar'
            save_path = args.TEST_RESULT_DIR + '/latest'
            log_path = args.TEST_RESULT_DIR + '/latest_result.log'
            epoch = 0
        else:
            load_path = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
            save_path = args.TEST_RESULT_DIR + '/' + '%04d' % load_epoch
            log_path = args.TEST_RESULT_DIR + '/%04d_' % load_epoch + 'result.log'
            epoch = load_epoch
    mkdir(save_path)
    if load_path.endswith('.pth'):
        model_state_dict = torch.load(load_path)
    else:
        model_state_dict = torch.load(load_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if 'module.' in k:
            name = k[7:]  
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
    model_state_dict = new_state_dict
    model.load_state_dict(model_state_dict)

    return load_path, save_path, log_path,epoch

def set_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def main():
    logger,device = init()
    
    model = my_model_two_encoder(args,en_feature_num=args.EN_FEATURE_NUM,
                en_inter_num=args.EN_INTER_NUM,
                de_feature_num=args.DE_FEATURE_NUM,
                de_inter_num=args.DE_INTER_NUM,
                sam_number=args.SAM_NUMBER,
                ).to(device)
    model._initialize_weights()
        
    load_path, save_path, log_path,epoch = load_checkpoint(model)

    
    set_logging(log_path)
    logging.warning(datetime.now())

    
    if args.EVALUATION_COST:
        if args.ADD_WIDEANGLE:
            calculate_cost(model, input_size=(1, 3, 2176, 3840),input_num=2)
        else:
            calculate_cost(model, input_size=(1, 3, 2176, 3840))

    logging.warning('load model from %s' % load_path)
    logging.warning('save image results to %s' % save_path)
    logging.warning('save logger to %s' % log_path)

    compute_metrics = None
    if args.EVALUATION_TIME:
        args.EVALUATION_METRIC = False
    if args.EVALUATION_METRIC:
        from utils.metric import create_metrics
        compute_metrics = create_metrics(args, device=device)

    
    loss_fn = None
    model_fn_test = model_fn_decorator(args = args,loss_fn=loss_fn,logger=logger, device=device, mode='test')

    
    test_path = args.TEST_DATASET
    args.BATCH_SIZE = 1
    TestImgLoader = create_dataset(args, data_path=test_path, mode='test',align=args.ALIGN, device = device)

    
    test(args, TestImgLoader, model, model_fn_test, save_path, compute_metrics, device = device, epoch = epoch, logger=logger)

if __name__ == '__main__':
    main()
    

