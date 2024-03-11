import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import os
import random
from torchvision import transforms
from torchvision.transforms import functional as F
from pathlib import Path

class MyDataset(Dataset):
    def __init__(self, cfg, mode='train', inference=0):
        if mode == 'train':
            with open(cfg.train_file, 'r') as file:
                self.file = file.readlines()
            if inference == 0:
                self.transform = self.train_transform
            else:
                self.transform = self.val_transform
        elif mode == 'val':
            with open(cfg.val_file, 'r') as file:
                self.file = file.readlines()
            self.transform = self.val_transform
        self.file_prefix = cfg.file_prefix
        self.num_classes = cfg.num_classes
        self.img_h = cfg.img_h
        self.img_w = cfg.img_w
        self.mode = mode
        self.mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        # self.loss = cfg.loss

    def train_transform(self, img):
        """
        :param img:
        :param seed:
        :return:  tensor of img in range 0-1
        """
        # _img = ImageOps.autocontrast(img)
        img_ = transforms.RandomCrop(32, padding=4)(img)
        img_ = transforms.RandomHorizontalFlip()(img)
        img_ = transforms.RandomRotation(15)(img_)
        # img_ = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)(img_)
        # img_ = transforms.RandomPerspective(distortion_scale=0.1)(img_)
        # img_ = transforms.RandomAffine(degrees=0, shear=5)(img_)
        # img_ = transforms.RandomResizedCrop((self.img_h, self.img_w), scale=(0.8, 1.0), ratio=(0.8, 1.2))(img_)
        # img_ = transforms.Resize((self.img_h, self.img_w))(img)
        img_ = transforms.ToTensor()(img_)
        img_ = transforms.Normalize(self.mean, self.std)(img_)
        return img_

    def val_transform(self, img):
        # w, h = img.size
        # min_size = min(w, h)
        # _img = ImageOps.autocontrast(img)
        img_ = transforms.Resize((self.img_h, self.img_w))(img)
        img_ = transforms.ToTensor()(img_)
        img_ = transforms.Normalize(self.mean, self.std)(img_)
        return img_

    def __getitem__(self, index):
        entry = self.file[index % len(self.file)].rstrip()
        img_path, label = entry.split(',')
        img = Image.open(os.path.join(self.file_prefix, img_path)).convert('RGB')
        img_tensor = self.transform(img)
        # if self.loss == 'CELoss':
        label = int(label)
        return {'x': img_tensor, 'y': label, 'file': img_path}
        # else:
        #     assert self.loss == 'KDLoss'
        #     label = label.split('_')
        #     label = list(map(float, label))
        #     label = torch.tensor(label)
        #     return {'x': img_tensor, 'y': label, 'file': img_path}


    def __len__(self):
        return len(self.file)