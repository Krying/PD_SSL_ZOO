import os
import sys
sys.path.append(os.path.abspath('/workspace/DIF_HWAE_BRAIN'))

import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from data_utils import get_loader
from trainer_simple_diffusion import Trainer, GaussianDiffusion
from functools import partial
import torch.nn as nn
import SimpleITK as sitk
import argparse
import monai
from monai.utils import misc
from model import create_model
import ast

def args_as_list(s):
    v = ast.literal_eval(s)
    return v

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline')
# parser.add_argument('--dec_checkpoint', default='', help='start training from saved checkpoint')
# parser.add_argument('--enc_checkpoint', default='', help='start training from saved checkpoint')
parser.add_argument('--img_save_dir', default='/workspace/DIF_DDAE_PET/results', help='start training from saved checkpoint')
parser.add_argument('--log_dir', default='/workspace/DIF_DDAE_PET/results', help='start training from saved checkpoint')
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument('--dim', default='3D', type=str)
parser.add_argument('--model', default='ddpm', type=str, help='ddpm, wddpm')
parser.add_argument('--ddpm', default='ddpm_ansio_8', type=str, help='ddpm, ddpm_aniso, controlnet, ldm, ddm')
parser.add_argument('--image_size', default=256, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--optim_lr', default=2e-5, type=float)
parser.add_argument('--max_grad_norm', default=1., type=float)
parser.add_argument('--max_epochs', default=20, type=int)
parser.add_argument('--max', default=1.0, type=float)
parser.add_argument('--val_interval', default=2, type=int)
parser.add_argument('--depth', default=6, type=int)
parser.add_argument('--channels', default=[128,128,256,512], type=args_as_list)
parser.add_argument('--attn', default=[False, False, False, False, False, True], type=args_as_list)
parser.add_argument('--attn_head', default=[0,0,0,0,0,256], type=args_as_list)
parser.add_argument('--temp', default=0, type=int)
parser.add_argument('--norm_group', default=32, type=int)
parser.add_argument('--train_num_steps', default=600000, type=int)
parser.add_argument('--augmentation', default=0, type=int)
parser.add_argument('--scheduler', default='ddpm_linear', type=str, help='ddpm_linear, ddpm_cos, pndm_linear, pndm_cos')
parser.add_argument('--residual', default=2, type=int)
parser.add_argument('--flash_attn', default=True, type=str2bool)
parser.add_argument('--gradient_accumulate_every', default=1, type=int)
parser.add_argument('--save_and_sample_every', default=5000, type=int)
parser.add_argument('--amp', default=0, type=int)
parser.add_argument('--split_batches', default=True, type=str2bool)
parser.add_argument('--resume', default=0, type=int)
parser.add_argument('--num_fold', default=0, type=int)

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.logdir = args.log_dir
    misc.set_determinism(seed=2024)
    
    main_worker_enc(args=args)

def main_worker_enc(args):
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    loader = get_loader(args)

    model = create_model(args)
    model.to(device)

    model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('MODEL Number of Learnable Params:', model_n_parameters)   
    print(f"learnig_rate : {args.optim_lr}")

    diffusion = GaussianDiffusion(model = model,
                                  image_size = 128,
                                  noise_d = 64,
                                  num_sample_steps = 1000)

    trainer = Trainer(diffusion,
                      loader,
                      args)


    # for i in range(100, 106):
    #     trainer.load_for_generation(i)
    #     trainer.generation(200, i)
    # i = args.num_fold
    # print("hello1")
    # trainer.load_for_generation(79)
    # trainer.generation(100, 79)
    print("hello")
    trainer.load_for_generation(80)
    trainer.generation(100, 80)
    # for i in range(106, 111):
    #     trainer.load_for_generation(i)
    #     trainer.generation(200, i)

if __name__ == '__main__':
    main()
