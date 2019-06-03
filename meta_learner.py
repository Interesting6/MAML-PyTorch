#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from copy import deepcopy
from learner import Learner





class MAML(nn.Module):
    """
    The Meta-Learner
    """
    def __init__(self, args, config):
        super(MAML, self).__init__()
        # learning rate
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr

        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.tasks_num = args.tasks_num

        self.num_update = args.num_update
        self.test_num_update = args.test_num_update

        self.learner = Learner(config)
        self.meta_optim = optim.Adam(self.learner.parameters(), lr=self.meta_lr)
        self.order = args.update_order  # 二阶导或一阶近似
        self.num_learnable_param = len(self.learner.parameters())


    def forward(self, tsk_xs, tsk_ys, tsk_xq, tsk_yq):
        """

        :param tsk_xs: [tsk, n_way*k_spt, c,h,w]
        :param tsk_ys: [tsk, n_way*k_spt,]
        :param tsk_xq: [tsk, n_way*k_qry, c,h,w]
        :param tsk_yq: [tsk, n_way*k_qry,]
        :return:
        """

        tasks_num, xs_sz, ch, h, w = tsk_xs.shape
        xq_sz = tsk_xq.size(1)

        corr_q_list = [0 for _ in range(self.num_update)]  # 用来判断在支持集上更新次数对模型准确率的影响
        task_loss = [None for _ in range(self.tasks_num)]
        tasks_grads = [None for _ in range(self.tasks_num)]
        cum_loss_task = 0
        cum_grads_task = [0. for _ in range(self.num_learnable_param)] # 每个参数的累计梯度

        for i in range(tasks_num): # 对于每一个任务
            # 支持集上第一次更新
            logits_s = self.learner(tsk_xs[i])
            loss_s = F.cross_entropy(logits_s, tsk_ys[i])
            grads = torch.autograd.grad(loss_s, self.learner.parameters())
            fast_weights = list(map(lambda p,g: p - self.update_lr*g,
                                    self.learner.parameters(), grads))

            with torch.no_grad():
                logits_q = self.learner(tsk_xq[i], fast_weights)
                pred_q = logits_q.argmax(dim=1)
                corr_q = (pred_q == tsk_yq[i]).sum().item()
                corr_q_list[0] += corr_q

            # 支持集上第2到第内层循环数次的更新
            for k in range(1, self.num_update):
                logits_s = self.learner(tsk_xs[i], fast_weights)
                loss_s = F.cross_entropy(logits_s, tsk_ys[i])
                grads = torch.autograd.grad(loss_s, fast_weights)
                fast_weights = list( map(lambda p,g: p - self.update_lr*g,
                                    self.learner.parameters(), grads) )

                with torch.no_grad():
                    logits_q = self.learner(tsk_xq[i], fast_weights)
                    pred_q = logits_q.argmax(dim=1)
                    corr_q =  (pred_q == tsk_yq[i]).sum().item()
                    corr_q_list[k] += corr_q

            # 子模型在查询集上的损失
            logits_q = self.learner(tsk_xq[i], fast_weights)
            loss_q = F.cross_entropy(logits_q, tsk_yq[i])
            if self.order == 2:
                cum_loss_task += loss_q
            else: # order == 1
                grads_task = torch.autograd.grad(loss_q, fast_weights)
                for j in range(self.num_learnable_param):
                    cum_grads_task[j] += grads_task[j]

            with torch.no_grad():
                logits_q = self.learner(tsk_xq[i], fast_weights)
                pred_q = logits_q.argmax(dim=1)
                corr_q = (pred_q == tsk_yq[i]).sum().item()
                corr_q_list[self.num_update-1] += corr_q


        # 所有任务结束
        if self.order == 2:
            loss_ = cum_loss_task / tasks_num  # 所有任务上查询集的平均损失
            self.meta_optim.zero_grad()
            loss_.backward()
            self.meta_optim.step()
        else:
            grads = [cum_grads / tasks_num for cum_grads in cum_grads_task] # 所有任务上查询集的平均梯度
            self.meta_optim.zero_grad()

            # with torch.no_grad(): # 手动更新所有参数，但只能普通更新
            #     for p,g in zip(self.learner.parameters(), grads):
            #         p.data.add_(-self.meta_lr, g.data)

            for group in self.meta_optim.param_groups:
                for p,g in zip(group['params'], grads):
                    if p.grad is not None:
                        p.grad.data = g.data

            self.meta_optim.step()

        accs = np.array(corr_q_list) / (tasks_num*xq_sz)

        return accs


    def fine_tuning(self, xs,ys, xq,yq):
        """
        在一个任务上进行
        :param xs: [n_way, k_spt, c,h,w]
        :param ys: [n_way, k_spt]
        :param xq: [n_way, k_qry, c,h,w]
        :param yq: [n_way, k_qry]
        :return:
        """
        assert len(xs.shape) == 4
        xq_sz = xq.size(0)

        corr_q_list = [0 for _ in range(self.test_num_update)]
        model = deepcopy(self.learner)

        fast_weights = None
        for k in range(self.test_num_update):
            logits_s = model(xs, fast_weights)
            loss_s = F.cross_entropy(logits_s, ys)

            grads = torch.autograd.grad(loss_s, model.parameters())
            fast_weights = list(map(lambda p, g: p - self.update_lr * g,
                                    model.parameters(), grads))

            with torch.no_grad():
                logits_q = model(xq, fast_weights)
                pred_q = logits_q.argmax(dim=1)
                corr_q = (pred_q==yq).sum().item()
                corr_q_list[k] += corr_q

        del model

        accs = np.array(corr_q) / xq_sz

        return accs

















