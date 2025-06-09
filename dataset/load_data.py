import numpy as np
import torch
import argparse
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile
import torchvision
import os
from multiprocessing.dummy import Pool
import torch.nn.functional as F
from tqdm import tqdm
from lightglue.utils import load_image, rbd
from lightglue import LightGlue, SuperPoint, DISK
import kornia
# import multiprocessing

def create_dataset(
    args,
    data_path,
    mode='train',
    align=True,
    device = 'cpu',
):
    def _list_image_files_recursively(data_dir):
        file_list = []
        for home, dirs, files in os.walk(data_dir):
            
            if args.TRAIN_WORD == False:
                pass
            elif args.TRAIN_WORD not in home:
                continue
            for filename in files:
                if filename.endswith('gt.jpg'):
                    file_list.append(os.path.join(home, filename))
        file_list.sort()
        return file_list

    
    MW_files = _list_image_files_recursively(data_dir = data_path)
    dataset = MW_data_loader(args, MW_files, mode = mode, align = align, device = device)
    
    data_loader = data.DataLoader(
        dataset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=args.WORKER, drop_last=True,
        multiprocessing_context=torch.multiprocessing.get_context('spawn'),
    )

    return data_loader


class MW_data_loader(data.Dataset):

    def __init__(self, args, image_list, mode = 'train', align = True, device = 'cpu'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.device = device
        self.loader = args.LOADER
        self.align = align
        if args.EVALUATION_TIME == True and args.ADD_ALIGN_TIME == True:
            self.align == True
        if self.align == False :
            self.extractor = SuperPoint(max_num_keypoints=256).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)


    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        retry=10
        while retry>0:
            labels, moire_imgs, w_imgs, number=self.process_imgs(index)
            if w_imgs == None:
                # print(f'img_{index} Alignment unsuccessful - skipping')
                index = random.randint(0, len(self.image_list))
                retry -= 1
            else:
                break

        if self.mode == 'train':
            if self.loader == 'crop':
                labels, moire_imgs,w_imgs = crop_loader_MW(self.args.CROP_SIZE, [labels, moire_imgs,w_imgs])

            elif self.loader == 'resize':
                labels, moire_imgs,w_imgs = resize_loader(self.args.RESIZE_SIZE, [labels, moire_imgs,w_imgs])

            elif self.loader == 'default':
                labels, moire_imgs,w_imgs = labels, moire_imgs,w_imgs

        elif self.mode == 'test':
            if self.loader == 'resize':
                data['origin_label'] = labels
                labels, moire_imgs,w_imgs = resize_loader(self.args.RESIZE_SIZE, [labels, moire_imgs,w_imgs])
            else:
                labels, moire_imgs,w_imgs = labels, moire_imgs,w_imgs

        else:
            print('Unrecognized mode! Please select either "train" or "test"')
            raise NotImplementedError



        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number
        data['w_img'] = w_imgs
                
        return data

    def __len__(self):
        return len(self.image_list)

    
    def lightglue_align(self,extractor,matcher,image0,image1):
        img_src = image0.unsqueeze(0)
        img_dst = image1.unsqueeze(0)
        image0 = F.interpolate(image0.unsqueeze(0), scale_factor=0.25, mode='bilinear') 
        image1 = F.interpolate(image1.unsqueeze(0), scale_factor=0.25, mode='bilinear')
        
        feats0 = extractor.extract(image0.squeeze(0),resize=None)
        feats1 = extractor.extract(image1.squeeze(0),resize=None)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [
            rbd(x) for x in [feats0, feats1, matches01]
        ]   
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        pts_src = np.float32([list(pt) for pt in m_kpts0.cpu()])
        pts_dst = np.float32([list(pt) for pt in m_kpts1.cpu()])
        
        flag = 0
        if len(pts_dst) < 4 or len(pts_src) < 4:
            H = torch.eye(3, dtype=torch.float32).to(self.device)
        else:
            H = cv2.findHomography(pts_dst, pts_src, cv2.RANSAC)[0]
            if H is None:
                H = torch.eye(3, dtype=torch.float32).to(self.device)
            else:
                H = torch.tensor(H, dtype=torch.float32).to(self.device)
                flag = 1

        
        scale_factor = 4

        
        S = torch.tensor([[scale_factor, 0, 0], [0, scale_factor, 0], [0, 0, 1]], dtype=torch.float32).to(self.device)
        S_inv = torch.tensor([[1/scale_factor, 0, 0], [0, 1/scale_factor, 0], [0, 0, 1]], dtype=torch.float32).to(self.device)
        
        
        H_original = S @ H @ S_inv
       
        warped_img = kornia.geometry.warp_perspective(img_dst, H_original.unsqueeze(0), (img_src.shape[2], img_src.shape[3])).squeeze(0)
        
        if flag == 0:
            return None
        return warped_img


    def process_imgs(self,index):
        data = {}
        path_tar = self.image_list[index]
        number = os.path.split(os.path.split(path_tar)[-2])[-1]+os.path.split(path_tar)[-1][0:4]
        path_src_m = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_m.jpg'
        if self.align:   
            path_src_w = os.path.split(path_tar)[0] + '/w_align/' + os.path.split(path_tar)[-1][0:4] + '_w_align.jpg'
        else:
            path_src_w = os.path.split(path_tar)[0] + '/' + os.path.split(path_tar)[-1][0:4] + '_w.jpg'
        
        
        w_imgs = Image.open(path_src_w).convert('RGB')
        w_imgs = default_toTensor(w_imgs).to(self.device)
        moire_imgs = Image.open(path_src_m).convert('RGB')
        moire_imgs = default_toTensor(moire_imgs).to(self.device)
        labels = Image.open(path_tar).convert('RGB')
        labels = default_toTensor(labels).to(self.device)
        if self.align == False:
            w_imgs = self.lightglue_align(self.extractor, self.matcher, moire_imgs, w_imgs)

        return labels, moire_imgs, w_imgs, number
        


def crop_loader_MW(crop_size, tensor_set=[]):
    cropped_tensors = []
    
    _, height, width = tensor_set[0].shape
    x_max = max(0, width - crop_size)
    y_max = max(0, height - crop_size)
    x = random.randint(0, x_max) if x_max > 0 else 0
    y = random.randint(0, y_max) if y_max > 0 else 0
    
    for tensor in tensor_set:
        cropped = tensor[:, y:y+crop_size, x:x+crop_size]
        cropped_tensors.append(cropped)
    
    return cropped_tensors
    

def resize_loader(resize_size, tensor_set=[]):
    resized_tensors = []
    
    for tensor in tensor_set:
        tensor = tensor.unsqueeze(0)
        resized = F.interpolate(
            tensor, 
            size=(resize_size, resize_size), 
            mode='bicubic', 
            align_corners=False
        )
        resized = resized.squeeze(0)
        resized_tensors.append(resized)
    
    return resized_tensors

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)
    return composed_transform(img)


