"""
base.py

Base classes

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Weipeng He <weipeng.he@idiap.ch>
"""

import cPickle as pickle

import torch
import torch.nn as nn

import numpy as np

_MODEL_SUFFIX = '.model.pkl'
_ARCH_SUFFIX = '.arch.pkl'

# activation function types
ACT_NONE = 0
ACT_SIGMOID = 1
ACT_TANH = 2
ACT_RELU = 3
ACT_SIG_NOSAT = 4

# instructions
ACT_INSTRUCTION = '{0:None, 1:Sigmoid, 2:Tanh, 3:ReLU, 4:Sigmoid(no saturation)}'

# pooling methods
POOL_MAX = 1
POOL_AVG = 2

class Identity(nn.Module):
    def forward(self, x):
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

class SigmoidNoSaturation(nn.Module):
    """
    Sigmoid function bewtween -0.05 and 1.05, so that it doesn't saturate
    between 0 and 1
    """
    def forward(self, x):
        return -0.05 + nn.functional.sigmoid(x) * 1.1

    def __repr__(self):
        return self.__class__.__name__ + ' ()'

# activation functions
_act_funcs = [Identity, nn.Sigmoid, nn.Tanh, nn.ReLU, SigmoidNoSaturation]

def num_params(net):
    return np.sum([np.prod(x.size()) for x in net.parameters()])

class SerializableModule(nn.Module):
    """Serializable (model + architecture) NN module
    """
    def __init__(self, args):
        super(SerializableModule, self).__init__()
        self.arch_args = args

    def save(self, name_prefix):
        torch.save(self.state_dict(), name_prefix + _MODEL_SUFFIX)
        arch = (self.__class__, self.arch_args)
        with open(name_prefix + _ARCH_SUFFIX, 'w') as f:
            pickle.dump(arch, f)

def load_module(name_prefix):
    with open(name_prefix + _ARCH_SUFFIX, 'r') as f:
        arch_class, arch_args = pickle.load(f)
    net = arch_class(**arch_args)
    net.load_state_dict(torch.load(name_prefix + _MODEL_SUFFIX))
    return net

def _conv3x3(in_channels, out_channels, stride=1):
    # 3x3 Convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        seq = [nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         padding=0, bias=False),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_channels, out_channels, kernel_size=3,
                         padding=1, bias=False),
               nn.BatchNorm2d(out_channels),
               nn.ReLU(inplace=True),
               nn.Conv2d(out_channels, out_channels, kernel_size=1,
                         padding=0, bias=False),
               nn.BatchNorm2d(out_channels),
               ]
        self.mseq = nn.Sequential(*seq)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.mseq(x)
        if self.downsample:
            residual = self.downsample(x)
        else:
            residual = x
        out += residual
        out = self.relu(out)
        return out

# -*- Mode: Python -*-
# vi:si:et:sw=4:sts=4:ts=4

