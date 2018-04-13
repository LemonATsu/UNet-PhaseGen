import torch
import torch.nn as nn
from torch.nn import init
import torch.optim.lr_scheduler
from torch.autograd import Variable
import itertools, functools
from utils import Pool, GANLoss, View, EnergyLoss, Transpose, Flatten
from collections import OrderedDict
import os
import numpy as np

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

class UNetModel(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UNetModel, self).__init__()
        self.gpu_ids = gpu_ids

        unet_block = UNetBlock(input_nc * 2, input_nc * 4, 4, 2, 1,
                                pos="innermost", norm_layer=norm_layer)
        unet_block = UNetBlock(input_nc * 2, input_nc * 2, 8, 2, 1, cat_nc=input_nc * 4,
                                submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(input_nc * 2, input_nc * 2, 8, 1, 2, cat_nc=input_nc * 4,
                                submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetBlock(output_nc, input_nc * 2, 32, 2, 16, input_nc=input_nc, cat_nc=input_nc * 4,
                                submodule=unet_block, pos="outermost", norm_layer=norm_layer)

        self.model = unet_block


    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def save(self, path):
        torch.save(self.model.cpu().state_dict(), path)
        if self.gpu_ids:
            self.model.cuda(self.gpu_ids[0])

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        if self.gpu_ids:
            self.model.cuda(self.gpu_ids[0])


class UNetBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                k_size, stride, padding, input_nc=None, cat_nc=None, submodule=None, pos=None,
                norm_layer=nn.BatchNorm2d,
                transpose=None):
        super(UNetBlock, self).__init__()
        self.pos = pos
        self.outermost = False
        if type(norm_layer) == functools.partial:
            # args are partially set.
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc == None:
            input_nc = outer_nc
        if transpose == None:
            transpose = padding
        if cat_nc == None:
            cat_nc = inner_nc * 2

        downconv = nn.Conv1d(input_nc, inner_nc, kernel_size=k_size,
                             stride=stride, padding=padding, bias=use_bias)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if pos == "outermost":
            # inner_nc * 2: it takes as input cat([x, submodule(x)])
            self.outermost = True
            upconv = nn.ConvTranspose1d(cat_nc, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=transpose, bias=use_bias)
            down = [downconv]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        elif pos == "innermost":
            upconv = nn.ConvTranspose1d(inner_nc, outer_nc, kernel_size=k_size+1,
                                        stride=stride, padding=transpose, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            # inner_nc * 2: it takes as input cat([x, submodule(x)])
            upconv = nn.ConvTranspose1d(cat_nc, outer_nc, kernel_size=k_size,
                                        stride=stride, padding=transpose, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        f = self.model(x)
        return torch.cat([x, f], 1)

