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

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'
import sys
import os
import SimpleITK as sitk
import numpy as np


NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

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
    # array = tensor.cpu().detach()
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
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
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
# def noise_list(n, layers, latent_dim, device):
#     return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers) # 0<= tt <= 5
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)
    #nosie_list -> shape : [12, 512], 3 + [12, 512], 5

def latent_to_w(style_vectorizer, z, num_conv):
    return (style_vectorizer(z).unsqueeze(1).repeat([1,num_conv,1]))
# def latent_to_w(style_vectorizer, latent_descr):
#     return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

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
# def evaluate_in_chunks(max_batch_size, model, *args):
#     split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
#     chunked_outputs = [model(*i) for i in split_args]
#     if len(chunked_outputs) == 1:
#         return chunked_outputs[0] 
#     return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def): #이거 건드려야함...
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)
    #for t, n in style:
    # print(t)
    # print(n)
    # t_1 -> torch.size([12, 512]), n_1 -> 3
    # t_2 -> torch.size([12, 512]), n_2 -> 5
    # t_1[:, None, :] -> torch.size([12, 1, 512])
    # t_2[:, None, :] -> torch.size([12, 1, 512])
    # t_1[:, None, :].expand(-1, 3, -1) -> torch.size([12, 3, 512])
    # t_2[:, None, :].expand(-1, 5, -1) -> torch.size([12, 5, 512])
    # torch.cat -> torch.size([12, 8, 512]) [batch_size, layer, image_size]

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

# dataset

def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

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

    def forward(self, x, y):#x-> 12, 512, 4, 4, 4, y -> 12, 512
        b, c, d, h, w = x.shape #12, 512, 4, 4, 4

        w1 = y[:, None, :, None, None, None] #torch.size([12, 1, 512, 1, 1, 1])
        w2 = self.weight[None, :, :, :, :, :] #torch.size([1, 512, 512, 3, 3, 3])
        weights = w2 * (w1 + 1) #torch.size([12, 512, 512, 3, 3, 3])

        if self.demod:
            dem = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
            #torch.size([12, 512, 1, 1, 1, 1])
            weights = weights * dem #torch.size([12, 512, 512, 3, 3, 3])

        x = x.reshape(1, -1, d, h, w) #torch.size([1, 12*512, 4, 4, 4])

        _, _, *ws = weights.shape #*ws = 512, 3, 3, 3
        weights = weights.reshape(b * self.filters, *ws) #torch.size([12 * 512, 512, 3, 3, 3])

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv3d(x, weights, padding=padding, groups=b) #torch.size([1,12*512, 4, 4, 4])

        x = x.reshape(-1, self.filters, d, h, w) #torch.size([12, 512, 4, 4, 4])
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
        inoise_1 = inoise_1[:, :x.shape[2], :x.shape[3], :x.shape[4], :] #torch.size([12, 4, 4, 4, 1])
        inoise_2 = inoise_2[:, :x.shape[2], :x.shape[3], :x.shape[4], :] #torch.size([12, 4, 4, 4, 1])
        noise1 = self.to_noise1(inoise_1).permute((0, 4, 1, 2, 3))#torch.size([12, 4, 4, 512(filter)]).permute((0,3,2,1))
        noise2 = self.to_noise2(inoise_2).permute((0, 4, 1, 2, 3))#torch.size([12, 4, 4, 512(filter)]).permute((0,3,2,1))
        # noise1 = self.to_noise1(inoise).permute((0, 4, 3, 2, 1))#torch.size([12, 4, 4, 512(filter)]).permute((0,3,2,1))
        # noise2 = self.to_noise2(inoise).permute((0, 4, 3, 2, 1))#torch.size([12, 4, 4, 512(filter)]).permute((0,3,2,1))
        style1 = self.to_style1(istyle_1)#torch.size([12, 512(filters)])
        x = self.conv1(x, style1) #conv1((12, 512, 4, 4), (12, 512))
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
        self.num_layers = int(log2(image_size*2/3) - 1) * 2 #32:4 -> 64:5 -> 128:6

        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose3d(latent_dim, 512, 4, 1, 0, bias=False)
        else:
            # self.initial_block = nn.Parameter(torch.randn((1, 512, 4, 4, 4)))
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
        # styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        idx = 0
        # for style, block in zip(styles, self.blocks):
        for block in self.blocks:
            # x, rgb = block(x, rgb, styles[:,idx:idx+1,:], styles[:,idx+1:idx+2,:], input_noise_1, input_noise_2)
            x, rgb = block(x, rgb, styles[:,idx,:], styles[:,idx+1,:], input_noise_1, input_noise_2)
            idx += 2
            # x, rgb = block(x, rgb, style, input_noise)
        # print('rgb_shape :',rgb.shape) 

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
        # block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = True, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3
        
        if self.image_size == 48:
            block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = False, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3
            # block_4 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=4) #4*4*4
        # elif self.image_size == 64:
        elif self.image_size == 96:
            block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = True, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3
            block_4 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=4) #4*4*4 -> 2*2*2
            # block_5 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=5) #2*2*2 -> 2*2*2
            # block_4 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=4) #2*2*2 -> 2*2*2
        elif self.image_size == 192:
            block_3 = DiscriminatorBlock(input_channels=256, filters=512, downsample = True, ind=3) #8*8*8 -> 4*4*4        | 12*12*6 -> 6*6*3
            block_4 = DiscriminatorBlock(input_channels=512, filters=512, downsample = True, ind=4)
            block_5 = DiscriminatorBlock(input_channels=512, filters=512, downsample = False, ind=5)
        
        blocks.append(block_0)
        blocks.append(block_1)
        blocks.append(block_2)
        blocks.append(block_3)
        
        # if self.image_size == 32:
        if self.image_size == 48:
            pass
            # blocks.append(block_4)
        # elif self.image_size == 64:
        elif self.image_size == 96:
            blocks.append(block_4)
            # blocks.append(block_5)
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
        # self.G_opt = Adam(generator_params, lr = self.lr * (1/ttur_mult), betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))

        # init weights
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda(rank)

        # startup apex mixed precision
        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

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