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
        loss_task = 0

        for i in range(tasks_num): # 对于每一个任务
            fast_weights = None
            for k in range(0, self.num_update):
                logits_s = self.learner(tsk_xs[i], fast_weights)
                loss_s = F.cross_entropy(logits_s, tsk_ys[i])
                grads = torch.autograd.grad(loss_s, self.learner.parameters())
                fast_weights = list( map(lambda p,g: p - self.update_lr*g,
                                    self.learner.parameters(), grads) )

                with torch.no_grad():
                    logits_q = self.learner(tsk_xq[i], fast_weights)
                    pred_q = logits_q.argmax(dim=1)
                    corr_q =  (pred_q == tsk_yq[i]).sum().item()
                    corr_q_list[k] += corr_q

            logits_q = self.learner(tsk_xq[i], fast_weights)
            loss_q = F.cross_entropy(logits_q, tsk_yq[i])
            loss_task += loss_q

            with torch.no_grad():
                logits_q = self.learner(tsk_xq[i], fast_weights)
                pred_q = logits_q.argmax(dim=1)
                corr_q = (pred_q == tsk_yq[i]).sum().item()
                corr_q_list[self.num_update-1] += corr_q


        # 所有任务结束
        loss_task = loss_task / tasks_num
        self.meta_optim.zero_grad()
        loss_task.backward()
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

















