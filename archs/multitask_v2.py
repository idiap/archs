"""
multitask_v2.py

Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import torch
import torch.nn as nn
from .base import SerializableModule, ResidualBlock, _act_funcs, POOL_MAX, \
                  POOL_AVG

_CONV_ARG_IN_CHS  = 'in_channels'
_CONV_ARG_OUT_CHS = 'out_channels'
_CONV_ARG_KERNEL  = 'kernel_size'
_CONV_ARG_STRIDE  = 'stride'
_CONV_ARG_PAD     = 'padding'
_CONV_ARG_DIAL    = 'dialation'

class DoaMultiTaskResnet(SerializableModule):
    def __init__(self, n_freq, n_doa, ds_convs, n_rblk, task_layers,
                 roll_padding=True):
        """
        Args:
            n_freq : number of frequency bins
            n_doa  : number of directions-of-arrival
            ds_convs : downsampling layers at the beginning of the network,
                       list of arguments (in dict) for creating nn.Conv2d.
            n_rblk : number of residual blocks in the shared layers
            task_layers : list of task-specific layers, each branch is
                          represented by convolution layers, the output activation
                          function, and the pooling (along time) method
        """
        super(DoaMultiTaskResnet, self).__init__({'n_freq'       : n_freq,
                                                  'n_doa'        : n_doa,
                                                  'ds_convs'     : ds_convs,
                                                  'n_rblk'       : n_rblk,
                                                  'task_layers'  : task_layers,
                                                  'roll_padding' : roll_padding})

        seq = []
        n_ch = None
        for l in ds_convs:
            assert _CONV_ARG_DIAL not in l
            assert _CONV_ARG_PAD not in l
            assert n_ch is None or l[_CONV_ARG_IN_CHS] == n_ch
            n_ch = l[_CONV_ARG_OUT_CHS]

            seq.append(nn.Conv2d(**l))
            seq.append(nn.BatchNorm2d(n_ch))
            seq.append(nn.ReLU(inplace=True))

            # compute map size along frequency
            if _CONV_ARG_STRIDE in l:
                if isinstance(l[_CONV_ARG_STRIDE], tuple):
                    stride_freq = l[_CONV_ARG_STRIDE][1]
                else:
                    stride_freq = l[_CONV_ARG_STRIDE]
            else:
                stride_freq = 1
            if _CONV_ARG_KERNEL in l:
                if isinstance(l[_CONV_ARG_KERNEL], tuple):
                    kernel_freq = l[_CONV_ARG_KERNEL][1]
                else:
                    kernel_freq = l[_CONV_ARG_KERNEL]
            else:
                kernel_freq = 1
            n_freq = (n_freq - kernel_freq + stride_freq) // stride_freq

        # residual layers
        for _ in xrange(n_rblk):
            seq.append(ResidualBlock(n_ch, n_ch))

        # shared layers as a module
        self.shared = nn.Sequential(*seq)
        n_ch_shared = n_ch

        # task-specific layers
        to_doas = []
        branches = []
        self.pad_size = []
        self.pool_mthd = []
        for layers, act, pool_mthd in task_layers:
            # pooling method
            self.pool_mthd.append(pool_mthd)

            # compute doa padding size 
            s_doa = n_doa
            for l in layers[::-1]:
                assert _CONV_ARG_DIAL not in l
                assert _CONV_ARG_PAD not in l

                if _CONV_ARG_STRIDE in l:
                    if isinstance(l[_CONV_ARG_STRIDE], tuple):
                        stride_doa = l[_CONV_ARG_STRIDE][1]
                    else:
                        stride_doa = l[_CONV_ARG_STRIDE]
                else:
                    stride_doa = 1
                if _CONV_ARG_KERNEL in l:
                    if isinstance(l[_CONV_ARG_KERNEL], tuple):
                        kernel_doa = l[_CONV_ARG_KERNEL][1]
                    else:
                        kernel_doa = l[_CONV_ARG_KERNEL]
                else:
                    kernel_doa = 1
                assert stride_doa == 1
                s_doa = s_doa + kernel_doa - 1

            if roll_padding:
                to_doas.append(nn.Sequential(
                    nn.Conv2d(n_ch_shared, n_doa, kernel_size=1),
                    nn.BatchNorm2d(n_doa),
                    nn.ReLU(inplace=True)))
                self.pad_size.append(s_doa - n_doa)
            else:
                to_doas.append(nn.Sequential(
                    nn.Conv2d(n_ch_shared, s_doa, kernel_size=1),
                    nn.BatchNorm2d(s_doa),
                    nn.ReLU(inplace=True)))
                self.pad_size.append(0)

            # convolution layers
            seq = []
            n_ch = n_freq
            for i, l in enumerate(layers):
                assert l[_CONV_ARG_IN_CHS] == n_ch
                n_ch = l[_CONV_ARG_OUT_CHS]

                seq.append(nn.Conv2d(**l))
                if i < len(layers) - 1:
                    seq.append(nn.BatchNorm2d(n_ch))
                    seq.append(nn.ReLU(inplace=True))
                else:
                    seq.append(_act_funcs[act]())
            branches.append(nn.Sequential(*seq))
        self.to_doas = nn.ModuleList(to_doas)
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        # input : data, ch, time, freq
        x = self.shared(x)
        # now   : data, ch, time, freq
        # for each branch
        res = []
        for t, b, p, m in zip(self.to_doas, self.branches, self.pad_size,
                              self.pool_mthd):
            # make doa as feature
            y = t(x)
            # now  : data, doa, time, freq
            # swap : freq <-> doa
            y = y.permute(0, 3, 2, 1)
            # now  : data, freq, time, doa
            # padding
            if p > 0:
                y = torch.cat((y, y.narrow(3, 0, p)), dim=3)
            # now  : data, freq, time, doa
            # more convolutions
            y = b(y)
            # now  : data, feature, time, doa
            # pooling : along time
            if m == POOL_AVG:
                # average pooling
                y = torch.mean(y, 2)
            elif m == POOL_MAX:
                # max pooling
                y = torch.max(y, 2)[0]
            else:
                assert False
            # now  : data, feature, doa
            if y.size(1) == 1:
                # squeeze
                y = y.squeeze(1)
                # result : data, doa
            else:
                # swap : feature <-> doa
                y = y.permute(0, 2, 1)
                # result : data, doa, feature
            res.append(y)
        return res

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

