import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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
from model.model import model_fn_decorator
from model.nets import my_model_two_encoder
from dataset.load_data import *
from tqdm import tqdm
from utils.loss_util import *
from utils.common import *
from config.config import args
from test import test
from torch.optim import lr_scheduler as lr_scheduler_way
from collections import OrderedDict
from lightglue import LightGlue, SuperPoint, DISK

def train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch, iters, lr_scheduler,):
    """
    Training Loop for each epoch
    """
    tbar = tqdm(TrainImgLoader)
    total_loss = 0
    total_vggloss = 0
    total_l1loss = 0
    total_fftloss = 0
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    for batch_idx, data in enumerate(tbar):
        loss= model_fn(args, data, model, iters,epoch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iters += 1
        total_loss += loss.item()
        avg_train_loss = total_loss / (batch_idx+1)

        desc = 'Training  : Epoch %d, lr %.7f, Avg. Loss = %.5f' % (epoch, lr, avg_train_loss)
        
        tbar.set_description(desc)
        tbar.update()
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    
    lr_scheduler.step()

    return lr, avg_train_loss, iters

def init():
    # Make dirs
    args.LOGS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'logs')
    args.NETS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'net_checkpoints')
    args.VISUALS_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'train_visual')
    args.TEST_RESULT_DIR = os.path.join(args.SAVE_PREFIX, args.EXP_NAME, 'test_result')
    mkdir(args.TEST_RESULT_DIR)
    mkdir(args.LOGS_DIR)
    mkdir(args.NETS_DIR)
    mkdir(args.VISUALS_DIR)

    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.GPU_ID
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    random.seed(args.SEED)
    np.random.seed(args.SEED)
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed_all(args.SEED)
    if args.SEED == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark =  True

    
    logger = SummaryWriter(args.LOGS_DIR)
    
    return logger, device

def load_checkpoint(model, optimizer, load_epoch):
    if load_epoch %10 != 0:
            load_dir = args.NETS_DIR + '/checkpoint_latest.tar'
    else:
        load_dir = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % load_epoch + '.tar'
    print('Loading pre-trained checkpoint %s' % load_dir)
    model_state_dict = torch.load(load_dir)['state_dict']
    # print(model_state_dict.keys())
    
    if 'module.' not in list(model_state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = 'module.'+k 
            new_state_dict[name] = v
        model_state_dict = new_state_dict
    model.load_state_dict(model_state_dict)
    optimizer_dict = torch.load(load_dir)['optimizer']
    optimizer.load_state_dict(optimizer_dict)
    learning_rate = torch.load(load_dir)['learning_rate']
    iters = torch.load(load_dir)['iters']
    print('Learning rate recorded from the checkpoint: %s' % str(learning_rate))

    return learning_rate, iters

def main():
    
    # torch.multiprocessing.set_start_method('spawn')
    logger, device = init()
    
    
    model = my_model_two_encoder(args,en_feature_num=args.EN_FEATURE_NUM,
                en_inter_num=args.EN_INTER_NUM,
                de_feature_num=args.DE_FEATURE_NUM,
                de_inter_num=args.DE_INTER_NUM,
                sam_number=args.SAM_NUMBER,
                )
    model._initialize_weights()

    model = torch.nn.DataParallel(model)

    model.to(device)

    
    optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr': args.BASE_LR}], betas=(0.9, 0.999))
    learning_rate = args.BASE_LR
    iters = 0
    
    if args.LOAD_EPOCH!=False:
        learning_rate, iters = load_checkpoint(model, optimizer, args.LOAD_EPOCH)
    
    
    loss_fn = multi_VGGPerceptualLoss(lam=args.LAM,lam_p=args.LAM_P).to(device)
    
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_MULT, eta_min=args.ETA_MIN,
                                               last_epoch=args.LOAD_EPOCH - 1)
    
    model_fn = model_fn_decorator(args = args,loss_fn=loss_fn,logger=logger, device=device)
    loss_fn_test = None
    model_fn_test = model_fn_decorator(args = args,loss_fn=loss_fn_test,logger=logger, device=device, mode='test')

    # create dataset
    train_path = args.TRAIN_DATASET
    TrainImgLoader = create_dataset(args, data_path=train_path, mode='train',align=args.ALIGN, device = device)


    if args.TRAIN_TEST:
        # Create dataset
        test_path = args.VAL_DATASET
        args.BATCH_SIZE = 1
        TestImgLoader = create_dataset(args, data_path=test_path, mode='test',align=args.ALIGN, device = device)

    # start training
    print("****start traininig!!!****")
    avg_train_loss = 0
    for epoch in range(args.LOAD_EPOCH + 1, args.EPOCHS + 1):
        train_epoch(args, TrainImgLoader, model, model_fn, optimizer, epoch,iters, lr_scheduler)

        logger.add_scalar('Train/avg_loss', avg_train_loss, epoch)
        logger.add_scalar('Train/learning_rate', learning_rate, epoch)

        if args.TRAIN_TEST:
            if epoch % args.TRAIN_TEST == 0:
                if args.EVALUATION_METRIC:
                    # load LPIPS metric
                    from utils.metric import create_metrics
                    compute_metrics = create_metrics(args, device=device)
                # test
                save_path = args.TEST_RESULT_DIR + '/customer/' + '%04d' % epoch
                print("test!")
                test(args, TestImgLoader, model, model_fn_test, save_path, compute_metrics,device,epoch,logger = logger, )

        # Save the network per ten epoch
        if epoch % 10 == 0:
            savefilename = args.NETS_DIR + '/checkpoint' + '_' + '%06d' % epoch + '.tar'
            torch.save({
                'learning_rate': learning_rate,
                'iters': iters,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.module.state_dict()
            }, savefilename)

        # Save the latest model
        savefilename = args.NETS_DIR + '/checkpoint' + '_' + 'latest.tar'
        torch.save({
            'learning_rate': learning_rate,
            'iters': iters,
            'optimizer': optimizer.state_dict(),
            'state_dict': model.module.state_dict()
        }, savefilename)
        

if __name__ == '__main__':
    main()
    
