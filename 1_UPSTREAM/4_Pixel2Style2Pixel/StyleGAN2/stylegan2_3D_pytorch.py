import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter3d

import torchvision
from torchvision import transforms

from PIL import Image
from pathlib import Path

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'
import sys
import os
import SimpleITK as sitk
import numpy as np
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel/StyleGAN2'))

from version import __version__
from diff_augment import DiffAugment
from data_utils import *
# constants

# helper classes

class NanException(Exception):
    pass

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class ChanNorm(nn.Module): #3D
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

class PermuteToFrom(nn.Module): #3D
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 4, 1, 2, 3)
        return out, loss

class Blur(nn.Module): #3D
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None] * f[:, None, None]
        f = f[None,:,:,:]
        return filter3d(x, f, normalized=True)

def save_image(tensor, path):
    array = tensor.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    array = np.fliplr(array)
    img = sitk.GetImageFromArray(array)
    sitk.WriteImage(img, path)

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

def default(value, d):
    return value if exists(value) else d

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def cast_list(el):
    return el if isinstance(el, list) else [el]

def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images): #3D
    device = images.device
    num_pixels = images.shape[2] * images.shape[3] * images.shape[4]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return (noise(n, latent_dim, device).unsqueeze(1).repeat([1, layers, 1]))

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers) 
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def latent_to_w(style_vectorizer, z, num_conv):
    return (style_vectorizer(z).unsqueeze(1).repeat([1,num_conv,1]))

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, int(im_size/2), 1).uniform_(0., 1.).cuda(device)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0] 
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# losses

def gen_hinge_loss(fake, real):
    return fake.mean()

def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

# augmentations

def random_hflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(3,))

def random_vflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(4,))

def random_zflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(-1,))

class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0.2, types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.2)
            images = random_vflip(images, prob=0.2)
            images = random_zflip(images, prob=0.2)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# stylegan2 classes

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)

        return self.net(x)

class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 1
        self.conv = Conv3DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='trilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, d, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x

class Conv3DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, d, h, w = x.shape 

        w1 = y[:, None, :, None, None, None] 
        w2 = self.weight[None, :, :, :, :, :] 
        weights = w2 * (w1 + 1) 

        if self.demod:
            dem = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            weights = weights * dem 

        x = x.reshape(1, -1, d, h, w) 

        _, _, *ws = weights.shape 
        weights = weights.reshape(b * self.filters, *ws) 

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv3d(x, weights, padding=padding, groups=b) 

        x = x.reshape(-1, self.filters, d, h, w) 
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False, ind=0):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False) if upsample else None
        self.ind = ind

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv3DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv3DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle_1, istyle_2, inoise_1, inoise_2):
        if exists(self.upsample):
            x = self.upsample(x)
        inoise_1 = inoise_1[:, :x.shape[2], :x.shape[3], :x.shape[4], :] 
        inoise_2 = inoise_2[:, :x.shape[2], :x.shape[3], :x.shape[4], :] 
        noise1 = self.to_noise1(inoise_1).permute((0, 4, 1, 2, 3))
        noise2 = self.to_noise2(inoise_2).permute((0, 4, 1, 2, 3))
        style1 = self.to_style1(istyle_1)
        x = self.conv1(x, style1) 
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle_2)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)
        rgb = self.to_rgb(x, prev_rgb, istyle_2)

        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True, ind=0):
        super().__init__()
        self.conv_res = nn.Conv3d(input_channels, filters, 1, stride = (2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv3d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv3d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class Generator(nn.Module):
    def __init__(self, 
                image_size, 
                latent_dim, 
                network_capacity = 16, 
                transparent = False, 
                attn_layers = [], 
                no_const = False, 
                fmap_max = 512,
                ):

        super().__init__()

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size*2/3) - 1) * 2 #48:4 -> 96:5 -> 192:6

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose3d(latent_dim, 512, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, 512, 6, 6, 3)))

        self.initial_conv = nn.Conv3d(512, 512, 3, padding=1)#conv3d(512 -> 512)
        self.blocks = nn.ModuleList([])

        block_0 = GeneratorBlock(
            latent_dim,
            input_channels=512,
            filters=512,
            upsample = False,
            upsample_rgb = True,
            rgba = transparent,
            ind = 0
        )

        block_1 = GeneratorBlock(
            latent_dim,
            input_channels=512,
            filters=512,
            upsample = True,
            upsample_rgb = True,
            rgba = transparent,
            ind = 1
        )

        #V3
        block_2 = GeneratorBlock(
            latent_dim,
            input_channels=512,
            filters=512,
            upsample = True,
            upsample_rgb = True,
            rgba = transparent,
            ind = 2
        )

        #V3
        if self.image_size == 48:

            block_3 = GeneratorBlock(
                latent_dim,
                input_channels=512,
                filters=256,
                upsample = True,
                upsample_rgb = False,
                rgba = transparent,
                ind = 3
            )

        #V3
        # elif self.image_size == 64:
        elif self.image_size == 96:

            block_3 = GeneratorBlock(
                latent_dim,
                input_channels=512,
                filters=256,
                upsample = True,
                upsample_rgb = True,
                rgba = transparent,
                ind = 3
            )

            block_4 = GeneratorBlock(
                latent_dim,
                input_channels=256,
                filters=128,
                upsample = True,
                upsample_rgb = False,
                rgba = transparent,
                ind = 4
            )

        elif self.image_size == 192:
            
            #V3
            block_3 = GeneratorBlock(
                latent_dim,
                input_channels=512,
                filters=256,
                upsample = True,
                upsample_rgb = True,
                rgba = transparent,
                ind = 3
            )

            block_4 = GeneratorBlock(
                latent_dim,
                input_channels=256,
                filters=128,
                upsample = True,
                upsample_rgb = True,
                rgba = transparent,
                ind = 4
            )

            block_5 = GeneratorBlock(
                latent_dim,
                input_channels=128,
                filters=16,
                upsample = True,
                upsample_rgb = False,
                rgba = transparent,
                ind = 5
            )

        self.blocks.append(block_0)
        self.blocks.append(block_1)
        self.blocks.append(block_2)

        if self.image_size == 48:
           self.blocks.append(block_3)

        # elif self.image_size == 64:
        elif self.image_size == 96:
            self.blocks.append(block_3)
            self.blocks.append(block_4)

        elif self.image_size == 192:
            self.blocks.append(block_3)
            self.blocks.append(block_4)
            self.blocks.append(block_5)

    def forward(self, styles, input_noise_1, input_noise_2):
        batch_size = styles.shape[0]
        image_size = self.image_size

        x = self.initial_block.expand(batch_size, -1, -1, -1, -1)

        rgb = None
        x = self.initial_conv(x)

        idx = 0
        for block in self.blocks:
            x, rgb = block(x, rgb, styles[:,idx,:], styles[:,idx+1,:], input_noise_1, input_noise_2)
            idx += 2

        return rgb

class Discriminator(nn.Module):
    def __init__(self, 
                 image_size, 
                 network_capacity = 16, 
                 fq_layers = [], 
                 fq_dict_size = 256, 
                 attn_layers = [], 
                 transparent = False, 
                 fmap_max = 512,
                 ):
        super().__init__()
        num_layers = int(log2(image_size*2/3) - 1) * 2 #48 -> 8, 96 -> 10 , 192 -> 12
        num_init_filters = 1
        self.image_size = image_size

        blocks = []

        block_0 = DiscriminatorBlock(input_channels=1, filters=64, downsample = True, ind=0) #64*64*64 -> 32*32*32     | 96*96*48 -> 48*48*24 | 48*48*24 -> 24*24*12
        block_1 = DiscriminatorBlock(input_channels=64, filters=128, downsample = True, ind=1) #32*32*32 -> 16*16*16   | 48*48*24 -> 24*24*12 | 24*24*12 -> 12*12*6
        block_2 = DiscriminatorBlock(input_channels=128, filters=256, downsample = True, ind=2) #16*16*16 -> 8*8*8     | 24*24*12 -> 12*12*6  | 12*12*6 -> 6*6*3
        
        if self.image_size == 48:
            block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = False, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3

        elif self.image_size == 96:
            block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = True, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3
            block_4 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=4) #4*4*4 -> 2*2*2

        elif self.image_size == 192:
            block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = True, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3
            block_4 = DiscriminatorBlock(input_channels=512, filters=512, downsample = True, ind=4)
            block_5 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=5)
        
        blocks.append(block_0)
        blocks.append(block_1)
        blocks.append(block_2)
        blocks.append(block_3)
        
        if self.image_size == 48:
            pass

        elif self.image_size == 96:
            blocks.append(block_4)

        elif self.image_size == 192:
            blocks.append(block_4)
            blocks.append(block_5)

        self.blocks = nn.ModuleList(blocks)

        chan_last = 512 #filters[-1] #4096
        latent_dim = 6 * 6 * 3 * chan_last #4096

        self.final_conv = nn.Conv3d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape
        index = 0
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze()

class StyleGAN2(nn.Module):
    def __init__(self, 
                image_size, 
                latent_dim = 512, 
                fmap_max = 512, 
                style_depth = 8, 
                network_capacity = 16, 
                transparent = False, 
                fp16 = False, 
                cl_reg = False, 
                steps = 1, 
                lr = 1e-4, 
                ttur_mult = 2, 
                fq_layers = [], 
                fq_dict_size = 256, 
                attn_layers = [], 
                no_const = False, 
                lr_mlp = 0.1, 
                rank = 0):
        
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim,
                                 style_depth,
                                 lr_mul = lr_mlp)

        self.G = Generator(image_size, 
                           latent_dim, 
                           network_capacity, 
                           transparent = transparent, 
                           attn_layers = attn_layers, 
                           no_const = no_const, 
                           fmap_max = fmap_max,
                           )

        self.D = Discriminator(image_size, 
                               network_capacity, 
                               fq_layers = fq_layers, 
                               fq_dict_size = fq_dict_size, 
                               attn_layers = attn_layers, 
                               transparent = transparent,
                               fmap_max = fmap_max,
                               )

        self.SE = StyleVectorizer(latent_dim, 
                                  style_depth, 
                                  lr_mul = lr_mlp)

        self.GE = Generator(image_size, 
                            latent_dim, 
                            network_capacity, 
                            transparent = transparent, 
                            attn_layers = attn_layers, 
                            no_const = no_const)

        self.D_cl = None

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        # turn off grad for exponential moving averages
        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        # init optimizers
        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv3d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x

class Trainer():
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        image_size = None,
        network_capacity = 8,
        fmap_max = 256,
        transparent = False,
        batch_size = 4,
        mixed_prob = 0.9,
        gradient_accumulate_every=1,
        lr = 2e-4,
        lr_mlp = 0.1,
        ttur_mult = 2,
        rel_disc_loss = False,
        num_workers = None,
        save_every = 1000,
        evaluate_every = 100,
        num_image_tiles = 8,
        trunc_psi = 0.6,
        fp16 = False,
        cl_reg = False,
        no_pl_reg = False,
        fq_layers = [],
        fq_dict_size = 256,
        attn_layers = [],
        no_const = False,
        aug_prob = 0.,
        aug_types = ['brightness', 'lightbrightness', 'contrast', 'lightcontrast'],
        top_k_training = False,
        generator_top_k_gamma = 0.99,
        generator_top_k_frac = 0.5,
        dual_contrast_loss = False,
        dataset_aug_prob = 0.,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        max_clamp = 2,
        len_dataset = 2670,
        *args,
        **kwargs
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = Path('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel/StyleGAN2/result')
        self.models_dir = Path('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel/StyleGAN2/model')
        self.fid_dir = Path('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel/StyleGAN2/fid')
        self.config_fold = Path('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel/StyleGAN2/config')
        self.config_path = Path(f'/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel/StyleGAN2/config/config_{name}.json')

        # assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.fmap_max = fmap_max
        self.transparent = transparent

        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size
        self.has_fq = len(self.fq_layers) > 0

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const

        self.aug_prob = aug_prob
        self.aug_types = aug_types

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.rel_disc_loss = rel_disc_loss
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.num_image_tiles = num_image_tiles
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.no_pl_reg = no_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.q_loss = None
        self.last_gp_loss = None
        self.last_cr_loss = None
        self.last_fid = None

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.top_k_training = top_k_training
        self.generator_top_k_gamma = generator_top_k_gamma
        self.generator_top_k_frac = generator_top_k_frac

        self.dual_contrast_loss = dual_contrast_loss

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        self.logger = aim.Session(experiment=name) if log else None

        self.max_clamp = max_clamp
        self.len_dataset = len_dataset

    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}
        
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, 
                            lr_mlp = self.lr_mlp, 
                            ttur_mult = self.ttur_mult, 
                            image_size = self.image_size, 
                            network_capacity = self.network_capacity, 
                            fmap_max = self.fmap_max, 
                            transparent = self.transparent, 
                            fq_layers = self.fq_layers, 
                            fq_dict_size = self.fq_dict_size, 
                            attn_layers = self.attn_layers, 
                            fp16 = self.fp16, 
                            cl_reg = self.cl_reg, 
                            no_const = self.no_const, 
                            rank = self.rank, 
                            *args, 
                            **kwargs)

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        if exists(self.logger):
            self.logger.set_params(self.hparams)

    def write_config(self):
        os.makedirs(self.config_fold, exist_ok=True)
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() #if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    def list_sort_nicely(l):
        def tryint(s):
            try:
                return int(s)
            except:
                return s
        
        def alphanum_key(s):
            return [ tryint(c) for c in re.split('([0-9]+)', s)]
        l.sort(key=alphanum_key)
        return l

    def set_data_src(self, folder):
        dataloader, self.dataset = get_loader(self.batch_size, self.image_size, self.max_clamp, self.len_dataset)
        
        self.loader = cycle(dataloader)
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    def train(self):
        assert exists(self.loader), 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        if not exists(self.GAN):
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda(self.rank)
        total_gen_loss = torch.tensor(0.).cuda(self.rank)

        # batch_size = math.ceil(self.batch_size / self.world_size)
        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob   = self.aug_prob
        aug_types  = self.aug_types
        aug_kwargs = {'prob': aug_prob, 'types': aug_types}

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = not self.no_pl_reg and self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        S = self.GAN.S if not self.is_ddp else self.S_ddp
        G = self.GAN.G if not self.is_ddp else self.G_ddp
        D = self.GAN.D if not self.is_ddp else self.D_ddp
        D_aug = self.GAN.D_aug if not self.is_ddp else self.D_aug_ddp

        backwards = partial(loss_backwards, self.fp16)

        # setup losses

        if not self.dual_contrast_loss:
            D_loss_fn = hinge_loss
            G_loss_fn = gen_hinge_loss
            G_requires_reals = False
        else:
            D_loss_fn = dual_contrastive_loss
            G_loss_fn = dual_contrastive_loss
            G_requires_reals = True

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[D_aug, S, G]):
            noise_1 = image_noise(batch_size, image_size, device=self.rank)
            noise_2 = image_noise(batch_size, image_size, device=self.rank)

            z = noise(batch_size, latent_dim, device=self.rank)
            w_space = latent_to_w(self.GAN.S, z, num_layers) 

            generated_images = G(w_space, noise_1, noise_2)
            fake_output = D_aug(generated_images.clone().detach(), detach = True, **aug_kwargs)

            image_batch = next(self.loader).cuda(self.rank)
            image_batch.requires_grad_()
            real_output = D_aug(image_batch, **aug_kwargs)

            real_output_loss = real_output
            fake_output_loss = fake_output

            divergence = D_loss_fn(real_output_loss, fake_output_loss)
            disc_loss = divergence

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                self.track(self.last_gp_loss, 'GP')
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

            total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.track(self.d_loss, 'D')

        self.GAN.D_opt.step()

        self.GAN.G_opt.zero_grad()

        for i in gradient_accumulate_contexts(self.gradient_accumulate_every, self.is_ddp, ddps=[S, G, D_aug]):
            noise_1 = image_noise(batch_size, image_size, device=self.rank)
            noise_2 = image_noise(batch_size, image_size, device=self.rank)

            z = noise(batch_size, latent_dim, device=self.rank)
            w_space = latent_to_w(self.GAN.S, z, num_layers) # batch * 12(=num_layer) * 512

            generated_images = G(w_space, noise_1, noise_2)
            fake_output = D_aug(generated_images, **aug_kwargs)
            fake_output_loss = fake_output

            real_output = None
            if G_requires_reals:
                image_batch = next(self.loader).cuda(self.rank)
                real_output = D_aug(image_batch, detach = True, **aug_kwargs)
                real_output = real_output.detach()

            if self.top_k_training:
                epochs = (self.steps * batch_size * self.gradient_accumulate_every) / len(self.dataset)
                k_frac = max(self.generator_top_k_gamma ** epochs, self.generator_top_k_frac)
                k = math.ceil(batch_size * k_frac)

                if k != batch_size:
                    fake_output_loss, _ = fake_output_loss.topk(k=k, largest=False)

            loss = G_loss_fn(fake_output_loss, real_output)
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_space, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, loss_id = 2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.track(self.g_loss, 'G')

        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)
            self.track(self.pl_mean, 'PL')

        if self.is_main and self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.is_main and self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{self.checkpoint_num}')
            self.load(self.checkpoint_num)
            raise NanException

        # periodically save results

        if self.is_main:
            if self.steps % self.save_every == 0:
                self.save(self.checkpoint_num)

            if self.steps % self.evaluate_every == 0 or (self.steps % 100 == 0 and self.steps < 2500):
                self.evaluate(floor(self.steps / self.evaluate_every))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, trunc = 1.0):
        self.GAN.eval()
        num_rows = self.num_image_tiles
    
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        # latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        latents = noise(self.batch_size, latent_dim, device=self.rank)
        n_1 = image_noise(num_rows ** 2, image_size, device=self.rank)
        n_2 = image_noise(num_rows ** 2, image_size, device=self.rank)

        # regular

        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n_1, n_2, trunc_psi = self.trunc_psi) #generate_truncated image에서 output size 확인 후 chunk하기
        save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.nii.gz'))
        
    @torch.no_grad()
    def truncate_style(self, tensor, trunc_psi = 0.75):
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy() #only 1*1*512
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        idx = 0
        w = self.truncate_style(w, trunc_psi = trunc_psi)
        w = w.unsqueeze(1).repeat([1, self.GAN.G.num_layers, 1])
        return w

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi_1, noi_2, trunc_psi = 0.75, num_image_tiles = 8): #style -> 1*512
        w = S(style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)

        generated_images = G(w_truncated, noi_1, noi_2)[0:1]
        return generated_images.clamp_(0., 1.0)

    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))

    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        try:
            # self.GAN.load_state_dict(load_data['GAN'], strict=True)
            self.GAN.load_state_dict(load_data['GAN'], strict=False)
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

class ModelLoader:
    def __init__(self, *, base_dir, name = 'default', load_from = -1):
        self.model = Trainer(name = name, base_dir = base_dir)
        self.model.load(load_from)

    def noise_to_styles(self, noise, trunc_psi = None):
        noise = noise.cuda()
        w = self.model.GAN.SE(noise)
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    def styles_to_images(self, w):
        batch_size, *_ = w.shape
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        w_def = [(w, num_layers)]

        w_tensors = styles_def_to_tensor(w_def)
        noise = image_noise(batch_size, image_size, device = 0)

        images = self.model.GAN.GE(w_tensors, noise)
        images.clamp_(0., 1.)
        return images
