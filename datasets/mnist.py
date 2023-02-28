import sys

from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from common.util import read_split_data


class MNIST(data.Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.is_train = mode == "train"
        self.data = torchvision.datasets.MNIST(cfg.GLOBAL.DATA_DIR, train=self.is_train)

    def __getitem__(self, index):
        img, label = self.data[index]

        x = transforms.ToTensor()(img)
        # x = (x - x.min()) / (x.max() - x.min())
        # x = self.preprocess(x)
        label = F.one_hot(torch.tensor(label), 10)

        return x, label

    def __len__(self):
        return len(self.data)

    def preprocess(self, data):
        # Here are just some of the operations, for more operations, please visit:
        # https://pytorch.org/vision/stable/transforms.html#compositions-of-transforms

        if self.mode == "train":
            transform_list = [
                transforms.Resize((self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)),
                transforms.RandomCrop((self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=self.cfg.TRAIN.TRANSFORMS_BRIGHTNESS,
                                       contrast=self.cfg.TRAIN.TRANSFORMS_CONTRAST,
                                       saturation=self.cfg.TRAIN.TRANSFORMS_SATURATION,
                                       hue=self.cfg.TRAIN.TRANSFORMS_HUE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        elif self.mode == "val" or self.mode == "test" or self.mode == "inference":
            transform_list = [
                transforms.Resize((self.cfg.TRAIN.IMG_SIZE, self.cfg.TRAIN.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ]
        else:
            print("mode only support [train, val, test, inference]!")
            sys.exit(0)

        transform = transforms.Compose(transform_list)
        data = transform(data)
        return data

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
        return images, labels
