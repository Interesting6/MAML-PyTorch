#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os, glob
import numpy as np
import pandas as pd
import torch
from torch.utils import data as utdata
from PIL import Image
from torchvision import transforms


rootPath = ''
splitPath = '/home/cheny/DataSet/miniImagenet/splits'
miniImagePath = 'C:/Users/Tream/Desktop/DataSets/Few-shot/miniImagenet/{}/'





resz = transforms.Resize(84)
centcrop = transforms.CenterCrop(84)
totensor = transforms.ToTensor()
norml = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

cps = transforms.Compose([
            resz,
            centcrop,
            totensor,
            norml
        ])

def load_image(path):
    img = Image.open(path)
    img = cps(img)
    return img


class miniImageNetDS(utdata.Dataset):
    def __init__(self, req_st='train', num_tasks=32, n_way=5, n_sq=(5, 15)):
        super(miniImageNetDS, self).__init__()
        self.req_st = req_st
        self.num_all_calss = len(glob.glob(miniImagePath.format(req_st)+'*'))  # require里面的类的个数
        self.images_path = glob.glob(miniImagePath.format(req_st)+'*')  # require类里每类的路径
        self.num_each_class = 600
        self.num_tasks = num_tasks
        self.n_way = n_way
        self.n_sq = sum(n_sq)

    def __len__(self):
        return self.num_all_calss

    def __getitem__(self, idx):
        idxclass_path = self.images_path[idx] + '/*'
        idxclass_all_img_path = glob.glob(idxclass_path)
        sq_idx = torch.randperm(self.num_each_class)[:self.n_sq]
        idxclass_img_path = [idxclass_all_img_path[i] for i in sq_idx]
        idxclass_img = torch.stack(list(map(load_image, idxclass_img_path)), 0)
        y = torch.tensor(idx).expand(self.n_sq)
        return idxclass_img, y


class tasksSampler(object):
    def __init__(self, num_all_class, n_tasks=32, n_way=5, n_sq=(5, 15)):
        self.num_all_class = num_all_class
        self.n_tasks = n_tasks  # n_episode
        self.n_way = n_way
        self.n_sq = sum(n_sq)  # num of support and query

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for i in range(self.n_tasks):
            select_class = np.random.permutation(self.num_all_class)[:self.n_way]  # 随机选择n_way个类
            yield select_class
            # 返回n_way个类的索引


def loadDL(req_st='train', num_tasks=32, n_way=5, n_sq=(5, 15)):
    ds = miniImageNetDS(req_st, num_tasks, n_way, n_sq)
    num_all_class = ds.num_all_calss
    sampler = tasksSampler(num_all_class, num_tasks, n_way, n_sq)
    ld = utdata.DataLoader(ds, batch_sampler=sampler)
    return ld



def rangeto0_1withMinMax(imgts):
    min = imgts.view(3, -1).min(-1)[0]
    range_ = imgts.view(3, -1).max(-1)[0] - min
    imgts.sub_(min[:, None, None]).div_(range_[:, None, None])
    return imgts


if __name__ == '__main__':
    ld = loadDL(num_tasks=2, n_way=3, n_sq=(1, 2))
    print(len(ld))
    # batch0X, batch0y = next(iter(ld))
    # print(batch0X.shape, batch0y.shape)
    # from torchvision.utils import make_grid
    # from matplotlib import pyplot as plt
    #
    # # print(batch0[0].view(3, -1).max(-1)[0], batch0[0].view(3, -1).min(-1)[0])
    # # batch0[0] = rangeto0_1withMeanStd(batch0[0])
    # # print(batch0[0].view(3, -1).max(-1)[0], batch0[0].view(3, -1).min(-1)[0])
    # batch0X = batch0X.view(-1, 3, 84, 84)
    # print(batch0y)
    # for i in range(len(batch0X)):
    #     batch0X[i] = rangeto0_1withMinMax(batch0X[i])
    #
    # grids = make_grid(batch0X)
    # grids = grids.numpy().transpose(1,2,0)
    # plt.imshow(grids)
    # plt.axis('off')
    # plt.show()

    batch = list(iter(ld))
    batchX, batchY = list(zip(*batch))
    batchX, batchY = [torch.stack(i) for i in (batchX,batchY)]
    print(batchX.shape)
    print(batchY.shape)

