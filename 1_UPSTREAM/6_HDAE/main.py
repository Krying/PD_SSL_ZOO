import os
import sys
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/UPSTREAM/6_HDAE'))

import ast
import torch
import argparse
import torch.nn as nn
import SimpleITK as sitk
from monai.utils import misc
from model import create_model
from data_utils import get_loader
from lr_scheduler import CosineAnnealingWarmUpRestarts
from trainer_ddpm_hdae import run_training_hdae

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

parser = argparse.ArgumentParser(description='HDAE upstream')
parser.add_argument('--model', default='HDAE', type=str)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--max_epochs', default=500, type=int)
parser.add_argument('--val_interval', default=2, type=int)
parser.add_argument('--eta_max', default=4e-5, type=float)
parser.add_argument('--optim_lr', default=2e-5, type=float)
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument('--log_dir', default='/workspace/PD_SSL_ZOO/UPSTREAM/6_HDAE/results')
parser.add_argument('--img_save_dir', default='/workspace/PD_SSL_ZOO/UPSTREAM/6_HDAE/results')

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.logdir = args.log_dir
    misc.set_determinism(seed=2024)
    
    main_worker_enc(args=args)

def main_worker_enc(args):
    args.log_dir = args.log_dir + '/' + str(args.optim_lr) + f'_model_{args.model}'
    args.img_save_dir = args.log_dir
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    args.test_mode = False

    loader = get_loader(args)

    model = create_model(args)

    model.to(device)

    enc_n_parameters = sum(p.numel() for p in model.semantic_encoder.parameters() if p.requires_grad)
    unet_n_parameters = sum(q.numel() for q in model.unet.parameters() if q.requires_grad)

    print('enc Number of Learnable Params:', enc_n_parameters)   
    print('unet Number of Learnable Params:', unet_n_parameters)   
    print(f"learnig_rate : {args.optim_lr}")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.optim_lr, weight_decay=0.05)
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=500, T_mult=1, eta_max=args.eta_max, T_up=2)

    accuracy = run_training_hdae(model,
                                 train_loader=loader[0],
                                 val_loader=loader[1],
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 args=args,
                                 )

    return accuracy

if __name__ == '__main__':
    main()

