import os
import sys
sys.path.append(os.path.abspath('/workspace/DIF_HWDAE_PET'))

import ast
import torch
import argparse
from monai.utils import misc
from model import create_model
from data_utils import get_loader
from trainer_HWDAE import run_training_hwdae
from lr_scheduler import CosineAnnealingWarmUpRestarts

parser = argparse.ArgumentParser(description='HWDAE upstream')
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--model', default='HWDAE', type=str)
parser.add_argument('--max_epochs', default=500, type=int)
parser.add_argument('--val_interval', default=2, type=int)
parser.add_argument('--optim_lr', default=2e-5, type=float)
parser.add_argument('--eta_max', default=4e-5, type=float)
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument('--log_dir', default='/workspace/PD_SSL_ZOO/UPSTREAM/1_HWDAE/results')
parser.add_argument('--img_save_dir', default='/workspace/PD_SSL_ZOO/UPSTREAM/1_HWDAE/results')

def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.logdir = args.log_dir
    misc.set_determinism(seed=2024)
    
    main_worker_enc(args=args)

def main_worker_enc(args):
    os.makedirs(args.log_dir + '/' + f'model_{args.model}_'+ str(args.optim_lr), exist_ok=True)
    os.makedirs(args.img_save_dir + '/' + f'model_{args.model}_'+ str(args.optim_lr), exist_ok=True)
    
    args.log_dir = args.log_dir + '/' + f'model_{args.model}_'+ str(args.optim_lr)
    args.img_save_dir = args.img_save_dir + '/' + f'model_{args.model}_'+ str(args.optim_lr)
    
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    
    loader = get_loader(args)

    model = create_model(args)
    
    model.to(device)

    enc_n_parameters = sum(p.numel() for p in model.semantic_encoder.parameters() if p.requires_grad)
    unet_n_parameters = sum(q.numel() for q in model.unet.parameters() if q.requires_grad)

    print('enc Number of Learnable Params:', enc_n_parameters)   
    print('unet Number of Learnable Params:', unet_n_parameters)   
    print(f"learnig_rate : {args.eta_max}")

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.optim_lr, weight_decay=0.05)
    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=500, T_mult=1, eta_max=args.eta_max, T_up=1)

    accuracy = run_training_hwdae(model,
                                  train_loader=loader[0],
                                  val_loader=loader[1],
                                  optimizer=optimizer,
                                  lr_scheduler=lr_scheduler,
                                  args=args,
                                  )

    return accuracy


if __name__ == '__main__':
    main()

