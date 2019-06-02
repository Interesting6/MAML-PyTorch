#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import os
import numpy as np
import pandas as pd
import torch
from torch.utils import data as utdata
from functools import partial

rootPath = ''
splitPath = '/home/cheny/DataSet/miniImagenet/splits'
miniImageNpPath = '/home/cheny/DataSet/miniImagenet/npdata'



def load_images_path(req_st):
    csv_path = os.path.join(splitPath, req_st + '.csv')
    df = pd.read_csv(csv_path)
    images_path = df['filename'].values
    groups = df.groupby('label')
    n_all_calss = len(groups)
    _, group = next(iter(groups))
    n_each_class = len(group)
    return images_path, n_all_calss, n_each_class


def load_image(req_st, img_path):
    img_name = img_path.split('.')[0]
    img_path = os.path.join(miniImageNpPath, req_st, img_name+'.npy')
    img = np.load(img_path)
    img = torch.from_numpy(img)
    return img


class miniImageNetDS(utdata.Dataset):
    def __init__(self, req_st='train', ):
        super(miniImageNetDS, self).__init__()
        self.req_st = req_st
        self.images_path, self.n_all_calss, self.n_each_class = load_images_path(req_st)


    def __len__(self):
        return len(self.images_path)


    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        load_images = partial(load_image, self.req_st)
        img = torch.stack(list(map(load_images, img_path)), 0)
        y = idx // self.n_each_class
        return img, y


class tasksSampler(object):
    def __init__(self, n_all_class, n_each_class, n_tasks, n_way, n_sq):
        self.n_totalClass = n_all_class
        self.n_eachClass = n_each_class
        self.n_tasks = n_tasks  # n_episode
        self.n_way = n_way
        self.n_sq = n_sq  # num of support and query

    def __len__(self):
        return self.n_tasks

    def __iter__(self):
        for i in range(self.n_tasks):
            select_class = np.random.permutation(self.n_totalClass)[:self.n_way]  # 随机选择n_way个类
            select_example = [ np.random.choice(self.n_eachClass, size=self.n_sq, replace=False)
                            + self.n_eachClass * i for i in select_class ]
            # select_example = torch.tensor(select_example).view(-1)
            yield select_example
            # 返回n_way*(n_spt+n_qry)个样本的索引


def loadDL(req_st='train', n_tasks=32, n_way=5, n_sq=16):
    ds = miniImageNetDS(req_st, )
    n_all_class, n_each_class = ds.n_all_calss, ds.n_each_class
    sampler = tasksSampler(n_all_class, n_each_class, n_tasks, n_way, n_sq)
    ld = utdata.DataLoader(ds, batch_sampler=sampler)
    return ld



def rangeto0_1withMinMax(imgts):
    min = imgts.view(3, -1).min(-1)[0]
    range_ = imgts.view(3, -1).max(-1)[0] - min
    imgts.sub_(min[:, None, None]).div_(range_[:, None, None])
    return imgts


if __name__ == '__main__':
    ld = loadDL(n_sq=6)
    print(len(ld))
    batch0 = next(iter(ld))
    print(batch0.shape)
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt

    # print(batch0[0].view(3, -1).max(-1)[0], batch0[0].view(3, -1).min(-1)[0])
    # batch0[0] = rangeto0_1withMeanStd(batch0[0])
    # print(batch0[0].view(3, -1).max(-1)[0], batch0[0].view(3, -1).min(-1)[0])

    for i in range(len(batch0)):
        batch0[i] = rangeto0_1withMinMax(batch0[i])

    grids = make_grid(batch0)
    grids = grids.numpy().transpose(1,2,0)
    plt.imshow(grids)
    plt.axis('off')
    plt.show()
