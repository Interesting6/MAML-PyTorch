#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Treamy


import torch
from torch import nn
from torch.nn import functional as F


class Learner(nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()
        self.config = config
        self.params_wb = nn.ParameterList()  # parameters need to train
        self.params_bn = nn.ParameterList()

        for name, param in config:
            if name == 'conv2d':
                # param=[ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.empty(*param[:4]).permute(1, 0, 2, 3)) # ch_out, ch_in, k_h, k_w
                nn.init.kaiming_normal_(w)
                b = nn.Parameter(torch.zeros(param[1])) # ch_out
                self.params_wb.extend([w, b])
                del w, b

            elif name == 'linear':
                # param=[ch_in, ch_out]
                w = nn.Parameter(torch.empty(*param).t()) # ch_out, ch_in
                nn.init.kaiming_normal_(w)
                b = nn.Parameter(torch.zeros(param[1])) # ch_out
                self.params_wb.extend([w, b])
                del w, b

            elif name == 'bn':
                # param = [in_out, ]
                w = nn.Parameter(torch.ones(param[0]))
                b = nn.Parameter(torch.zeros(param[0]))
                mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.params_wb.extend([w, b,])
                self.params_bn.extend([mean, var])
                del w, b, mean, var

            elif name in [
                'tanh', 'relu', 'avg_pool2d', 'max_pool2d', 'flatten', 'sigmoid'
            ]:
                continue
            else:
                raise NotImplementedError


    def __repr__(self):
        info = ''
        for name, param in self.config:
            if name == 'conv2d':
                info += 'conv2d: (ch_in:{0}, ch_out:{1}, ker_sz:{2}*{3}, stride:{4}, ' \
                        'padding:{5}) \n'.format(*param)

            elif name == 'linear':
                info += 'linear: (ch_in:{}, ch_out{}) \n'.format(*param)

            elif name == 'max_pool2d':
                info += 'max_pool2d: (ker_sz:{}*{}, stride:{}, padding:{}) \n'.format(*param)
            elif name == 'avg_pool2d':
                info += 'avg_pool2d: (ker_sz:{}*{}, stride:{}, padding:{}) \n'.format(*param)

            elif name in [
                'tanh', 'relu', 'avg_pool2d', 'max_pool2d', 'flatten', 'sigmoid', 'bn'
            ]:
                info += (name + ': ' + str(tuple(param)) + ' \n')

            else:
                raise NotImplementedError

        return info

    # def __str__(self):
    #     return self.__repr__()

    def forward(self, x, params=None, bn_training=True):

        if params is None:
            params = self.params_wb

        idx = 0
        bn_idx = 0
        for name, param in self.config:
            if name == 'conv2d':
                w, b = params[idx], params[idx+1]
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2

            elif name == 'linear':
                w, b = params[idx], params[idx+1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'bn':
                w, b, mean, var = params[idx], params[idx+1], self.params_bn[bn_idx], self.params_bn[bn_idx+1]
                x = F.batch_norm(x, mean, var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'relu':
                x = F.relu(x)

            elif name == 'tanh':
                x = F.tanh(x)

            elif name == 'sigmoid':
                x = F.sigmoid(x)

            elif name == 'max_pool2d':
                x = F.max_pool2d(x, *param)

            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, *param)

            else:
                raise NotImplementedError

        assert idx == len(params)

        return x

    def zero_grad(self, params=None):
        with torch.no_grad():
            if params is None:
                for p in self.params_wb:
                    if p.grad is not None:
                        p.grad.zero_()

            else:
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.params_wb




if __name__ == '__main__':
    n_way = 5
    config = [
        ('conv2d', [1, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [64, n_way])
    ]

    model = Learner(config)

    print(model)

    batch = torch.rand(5, 1, 28, 28)
    logits = model(batch)
    print(logits)
    print(model.parameters())
