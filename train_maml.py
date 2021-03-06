#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import torch
from meta_learner import MAML
import argparse
from loadData import loadDL
import numpy as np



def run(args):
    use_cuda = args.use_cuda
    tasks_num = args.tasks_num
    n_way = args.n_way
    k_spt = args.k_spt
    k_qry = args.k_qry


    torch.manual_seed(1)
    torch.cuda.manual_seed(1) if torch.cuda.is_available() else 0
    np.random.seed(1)

    # config = [
    #     ('conv2d', [1, 64, 3, 3, 2, 0]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('conv2d', [64, 64, 3, 3, 2, 0]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('conv2d', [64, 64, 3, 3, 2, 0]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('conv2d', [64, 64, 2, 2, 1, 0]),
    #     ('relu', [True]),
    #     ('bn', [64]),
    #     ('flatten', []),
    #     ('linear', [64, args.n_way])
    # ]
    config = [
        ('conv2d', [3, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d',[2]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d',[2]),
        ('conv2d', [32, 32, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d',[2]),
        ('conv2d', [32, 32, 2, 2, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('max_pool2d',[2]),
        ('flatten', []),
        ('linear', [800, args.n_way])
    ]

    maml = MAML(args, config)
    maml = maml.cuda() if use_cuda else maml

    ds_dl = loadDL('train', args.tasks_num, n_way, (k_spt,k_qry) )
    tsds_dl = loadDL('test', args.tasks_num, n_way, (k_spt,k_qry) )
    imgsize = [3, 84, 84]
    eval_i = 0

    maml.train()
    for epoch in range(1, args.max_epoch):
        all_tasks_batch = list(iter(ds_dl))
        batchX, batchY = list(zip(*all_tasks_batch))
        batchX, batchY = [torch.stack(i) for i in (batchX, batchY)]
        batchX = batchX.cuda() if use_cuda else batchX
        # batchY = batchY.cuda() if use_cuda else batchY

        tsk_xs = batchX[:,:,:k_spt,].contiguous().view(tasks_num, n_way*k_spt, *imgsize)
        tsk_xq = batchX[:,:,k_spt:,].contiguous().view(tasks_num, n_way*k_qry, *imgsize)
        tsk_ys = torch.arange(n_way).view(1,n_way,1).expand(tasks_num, n_way, k_spt).reshape(tasks_num, -1)
        tsk_yq = torch.arange(n_way).view(1,n_way,1).expand(tasks_num, n_way, k_qry).reshape(tasks_num, -1)
        tsk_ys = tsk_ys.cuda() if use_cuda else tsk_ys
        tsk_yq = tsk_yq.cuda() if use_cuda else tsk_yq
        accs = maml(tsk_xs,tsk_ys, tsk_xq,tsk_yq)
        print('step:', epoch, ' training accuracy on query set:', accs)
        with open('C:/Users/Tream/Desktop/GitHub/MAMLs/MAML-PyTorch/results/outs.txt', 'a', encoding='utf-8') as f:
            f.write(str(accs)+'\n')
        # if epoch%50==0:
        #     print('step:',epoch,' training accuracy on query set:', accs)


        if epoch%100==0:
            eval_i += 1
            maml.eval()
            accs = []
            for _ in range(1000//args.tasks_num):
                for bx, by in tsds_dl:
                    xs = bx[:, :k_spt].contiguous().view(n_way*k_spt, *imgsize)
                    xq = bx[:, k_spt:].contiguous().view(n_way*k_qry, *imgsize)
                    ys = torch.arange(n_way).view(n_way, 1).expand(n_way, k_spt).reshape(-1)
                    yq = torch.arange(n_way).view(n_way, 1).expand(n_way, k_qry).reshape(-1)

                    xs = xs.cuda() if use_cuda else xs; ys = ys.cuda() if use_cuda else ys
                    xq = xq.cuda() if use_cuda else xq; yq = yq.cuda() if use_cuda else yq
                    acc = maml.fine_tuning(xs, ys, xq, yq)
                    accs.append(acc)

            accs = np.array(accs).mean(axis=0).astype('float')
            print('test accuracy on query set:', accs)
            torch.save(maml.state_dict(), 'C:/Users/Tream/Desktop/GitHub/MAMLs/MAML-PyTorch/saves/mamlImagenet_{}.pt'.format(eval_i))
            with open('C:/Users/Tream/Desktop/GitHub/MAMLs/MAML-PyTorch/results/outs.txt', 'a', encoding='utf-8') as f:
                f.write('\n'+str(accs)+'\n')
            maml.train()







if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_epoch', type=int, help='epoch number', default=4000)
    argparser.add_argument('--tasks_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--update_order', type=int, help='meta update 1st or 2nd derivatives', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--num_update', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--test_num_update', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--use_cuda', type=int, help='use cuda or not', default=1)
    argparser.add_argument('--req_ds', type=str, help='use which dataset', default='train')

    args = argparser.parse_args()

    run(args)

