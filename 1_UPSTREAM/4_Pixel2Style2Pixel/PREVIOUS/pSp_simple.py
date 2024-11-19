from collections import namedtuple
import torch
from torch.nn import Conv2d, BatchNorm2d, Conv3d, BatchNorm3d, PReLU, ReLU, Sigmoid, MaxPool2d, MaxPool3d, AdaptiveAvgPool3d, Sequential, Module

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
	""" A named tuple describing a ResNet block. """

def get_block(in_channel, depth, num_units, stride=2):
	return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
	if num_layers == 50:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=4),
			get_block(in_channel=128, depth=256, num_units=14),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 100:
		blocks = [
			get_block(in_channel=64, depth=64, num_units=3),
			get_block(in_channel=64, depth=128, num_units=13),
			get_block(in_channel=128, depth=256, num_units=30),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	elif num_layers == 152:
		blocks = [
			get_block(in_channel=128, depth=128, num_units=11),
			get_block(in_channel=128, depth=256, num_units=36),
			get_block(in_channel=256, depth=512, num_units=3)
		]
	else:
		raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
	return blocks


class SEModule_3D(Module):
	def __init__(self, channels, reduction):
		super(SEModule_3D, self).__init__()
		self.avg_pool = AdaptiveAvgPool3d(1)
		self.fc1 = Conv3d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
		self.relu = ReLU(inplace=True)
		self.fc2 = Conv3d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
		self.sigmoid = Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x

class bottleneck_IR_SE_3D(Module):
	def __init__(self, in_channel, depth, stride):
		super(bottleneck_IR_SE_3D, self).__init__()
		if in_channel == depth:
			self.shortcut_layer = MaxPool3d(1, stride)
		else:
			self.shortcut_layer = Sequential(
				Conv3d(in_channel, depth, (1, 1, 1), stride, bias=False),
				BatchNorm3d(depth)
			)
		self.res_layer = Sequential(
			BatchNorm3d(in_channel),
			Conv3d(in_channel, depth, (3, 3, 3), (1, 1, 1), 1, bias=False),
			PReLU(depth),
			Conv3d(depth, depth, (3, 3, 3), stride, 1, bias=False),
			BatchNorm3d(depth),
			SEModule_3D(depth, 16)
		)

	def forward(self, x):
		shortcut = self.shortcut_layer(x)
		res = self.res_layer(x)
		return res + shortcut

from torch import nn
import numpy as np
from torch.nn import functional as F
import math

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class GradualStyleBlock_3D(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock_3D, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial*(2/3)))
        modules = []
        modules += [Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv3d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        modules += [
            Conv3d(out_c, out_c, kernel_size=3, stride=(2,2,1), padding=1),
            nn.LeakyReLU()
        ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class GradualStyleEncoder_3D(Module):
    def __init__(self, num_layers = 50):
        super(GradualStyleEncoder_3D, self).__init__()
        blocks = get_blocks(num_layers)

        unit_module = bottleneck_IR_SE_3D

        self.input_layer = Sequential(Conv3d(1, 64, (3, 3, 3), 1, 1, bias=False),
                                      BatchNorm3d(64),
                                      PReLU(64))
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 12
        self.coarse_ind = 4
        self.middle_ind = 8

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock_3D(512, 512, 12)
            elif i < self.middle_ind:
                style = GradualStyleBlock_3D(512, 512, 24)
            else:
                style = GradualStyleBlock_3D(512, 512, 48)
            self.styles.append(style)

        self.latlayer1 = nn.Conv3d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv3d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W, D = y.size()
        return F.interpolate(x, size=(H, W, D), mode='trilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out

class pSp(nn.Module):

	def __init__(self):
		super(pSp, self).__init__()
		self.n_styles = 12

		self.encoder = self.set_encoder()

	def set_encoder(self):

		encoder = GradualStyleEncoder_3D()

		return encoder

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		
		codes = self.encoder(x) 

		return codes
