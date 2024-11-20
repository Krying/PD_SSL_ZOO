import os
import sys
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/1_UPSTREAM/2_WDDAE'))

import torch
import argparse
from monai.utils import misc
from model import create_model
from data_utils import get_loader
from trainer_simple_diffusion_wavelet import Trainer, GaussianDiffusion

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='WDDAE upstream')
parser.add_argument('--amp', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--image_size', default=192, type=int)
parser.add_argument('--optim_lr', default=2e-5, type=float)
parser.add_argument('--max_grad_norm', default=1., type=float)
parser.add_argument('--train_num_steps', default=480000, type=int)
parser.add_argument('--split_batches', default=True, type=str2bool)
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument('--save_and_sample_every', default=5000, type=int)
parser.add_argument('--gradient_accumulate_every', default=1, type=int)
parser.add_argument('--model', default='wddpm', type=str, help='ddpm, wddpm')
parser.add_argument('--log_dir', default='/workspace/PD_SSL_ZOO/1_UPSTREAM/2_WDDAE/results')
parser.add_argument('--img_save_dir', default='/workspace/PD_SSL_ZOO/1_UPSTREAM/2_WDDAE/results')

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

    model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('MODEL Number of Learnable Params:', model_n_parameters)   
    print(f"learnig_rate : {args.optim_lr}")

    diffusion = GaussianDiffusion(model = model,
                                  image_size = 192,
                                  noise_d = 96,
                                  num_sample_steps = 1000)

    trainer = Trainer(diffusion,
                      loader,
                      args)
    
    trainer.train(args)

if __name__ == '__main__':
    main()
