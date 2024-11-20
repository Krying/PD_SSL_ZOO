import os
import sys
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel'))

import torch
import argparse
import numpy as np
from monai.utils import misc
from metric import LPIPS, psnr
from model import create_model
from trainer import run_training
from data_utils import get_loader

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Pixel2Style2Pixel upstream')
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_layer', default=12, type=int)
parser.add_argument('--image_size', default=192, type=int)
parser.add_argument('--max_epochs', default=300, type=int)
parser.add_argument('--optim_lr', default=1e-6, type=float)
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument('--img_save_dir', default='/workspace/wjj910/Task1_clf/Task1_inversion/StyleGAN2_3d_PRO/INVERSION_PY/reconstruct/', help='start training from saved checkpoint')
parser.add_argument('--log_dir', default='/workspace/Ablation/ABLATION_PD/GAN_INV_PSP/outputs/', help='start training from saved checkpoint')


def main():
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    args.logdir = args.log_dir
    misc.set_determinism(seed=2023)
    
    main_worker_enc(args=args)

def main_worker_enc(args):

    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    train_loader, val_loader = get_loader(args)

    encoder, decoder, _ = create_model(args.enc_version, args)

    PSNR_metric = psnr()    

    start_epoch = 0
    encoder.to(device)
    decoder.to(device)

    enc_n_parameters = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('ENC Number of Learnable Params:', enc_n_parameters)   
    
    dec_n_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print('DEC Number of Learnable Params:', dec_n_parameters)   
    
    model_prior_dict = decoder.GE.state_dict()
    ckpt_dec = torch.load('/workspace/Ablation/ABLATION_PD/GAN_INV_PSP/model_60.pt', map_location='cpu')
    decoder.load_state_dict(ckpt_dec["GAN"])
    model_final_loaded_dict = decoder.GE.state_dict()

    layer_counter = 0

    for k, _v in model_final_loaded_dict.items():
        if k in model_prior_dict:
            layer_counter = layer_counter + 1

            old_wts = model_prior_dict[k]
            new_wts = model_final_loaded_dict[k]

            old_wts = old_wts.to("cpu").numpy()
            new_wts = new_wts.to("cpu").numpy()
            diff = np.mean(np.abs(old_wts, new_wts))
            if diff == 0.0:
                print("Warning: No difference found for layer {}".format(k))

    decoder.requires_grad_(False)

    dec_n_parameters = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print('DEC Number of Learnable Params:', dec_n_parameters)   

    optimizer = torch.optim.AdamW(params=encoder.parameters(), lr=args.optim_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4, amsgrad=False)
    
    scheduler = None

    accuracy = run_training(enc=encoder,
                            dec=decoder.GE,
                            criterion=None,
                            eval_psnr=PSNR_metric,
                            eval_lpips=None,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            args=args,
                            scheduler=scheduler,
                            start_epoch=start_epoch)
    

    return accuracy

if __name__ == '__main__':
    main()
