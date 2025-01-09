import os
import sys
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/2_DOWNSTREAM/1_EP/'))

import json
import torch
import random
import argparse
import numpy as np
from model import *
from utils import *
from pathlib import Path
from trainer_ep import *
from data_utils import get_loader
from lr_scheduler import create_scheduler

parser = argparse.ArgumentParser(description="EP")
parser.add_argument('--test', default=0, type=int)
parser.add_argument('--fold', default=None, type=int)
parser.add_argument("--num_class", default=2, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--max_epochs", default=50, type=int)
parser.add_argument("--min_lr", default=1e-8, type=float)
parser.add_argument("--optim_lr", default=1e-3, type=float)
parser.add_argument("--warm_up_epoch", default=10, type=int)
parser.add_argument("--start_decay_epoch", default=45, type=int)
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument("--down_type", default=None, type=str, choices=["1_EP", "2_PMP", "3_SOY"])
parser.add_argument("--log_dir", default="/workspace/PD_SSL_ZOO/2_DOWNSTREAM/1_EP/", type=str)
parser.add_argument("--log_dir_png", default="/workspace/PD_SSL_ZOO/2_DOWNSTREAM/1_EP/", type=str)
parser.add_argument("--linear_mode", default='linear', type=str, choices=["scratch", "linear", "fine_tuning"])
parser.add_argument("--name", default=None, type=str, choices=["HWDAE", "WDDAE", "DDAE", "P2S2P", "DisAE", "HDAE", "SimMIM"])

def main():
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices
    
    args.down_type = '1_EP'
    args.num_class = 2
    
###########################LINEAR_PROBING###########################
    args.linear_mode = 'linear'
    
    print("ET/PD LINEAR TRAIN START")
    args.max_epochs = 50
    args.warm_up_epoch = 10
    
    for i in range(5):
        args.fold = i+1
        print(f"FOLD_{args.fold} TRAIN PROCESS START")
        loaders = get_loader(args)
        
        args.test = 0
        main_worker(args=args, loader=loaders)

        args.log_dir = "/workspace/PD_SSL_ZOO/2_DOWNSTREAM/" + f"{args.down_type}/results/{args.name}/{args.name}_output_{args.fold}_{args.linear_mode}/test/"
        args.log_dir_png = args.log_dir
        os.makedirs(args.log_dir, mode=0o777, exist_ok=True)
        
        args.test = 1
        main_worker(args=args, loader=loaders)

def main_worker(args, loader):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda')

    model = create_model(args)
    model.to(device)

    if args.test == 0:
        if args.name in ['SimMIM', 'DisAE']:
            n_parameters = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
            model.head.requires_grad_(True)
        else:
            n_parameters = sum(p.numel() for p in model.linear.parameters() if p.requires_grad)
            model.linear.requires_grad_(True)
            
        print('Number of Learnable Params:', n_parameters)
        
        if args.name in ['SimMIM', 'DisAE']:
            optimizer = torch.optim.AdamW(params=model.head.parameters(), lr=args.optim_lr, weight_decay=0.05)
        else:
            optimizer = torch.optim.AdamW(params=model.linear.parameters(), lr=args.optim_lr, weight_decay=0.05)
        
        lr_scheduler = create_scheduler(name='poly_lr', optimizer=optimizer, args=args)
    
    else:
        pass
        
    clf_loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if args.test == 0: #Train, Valid phase
        args.log_dir = "/workspace/PD_SSL_ZOO/2_DOWNSTREAM/" + f"{args.down_type}/results/{args.name}/{args.name}_output_{args.fold}_{args.linear_mode}/"
        args.log_dir_png = args.log_dir + "png/"
        os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 
        os.makedirs(args.log_dir_png, mode=0o777, exist_ok=True)
        
        val_loss_min = 1

        for epoch in range(args.max_epochs):
            #Train phase
            train_stats = train_epoch(args, 
                                      epoch, 
                                      loader[0], 
                                      model,
                                      clf_loss_func, 
                                      optimizer, 
                                      device)
        
            lr = optimizer.param_groups[0]["lr"]
            print(f"lr : {lr}")
            train_stats['lr'] = lr

            if epoch > args.warm_up_epoch:
                #Valid phase
                valid_stats = val_epoch(args, 
                                        epoch, 
                                        loader[1], 
                                        model, 
                                        clf_loss_func, 
                                        device)

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'valid_{k}': v for k, v in valid_stats.items()},
                             'epoch': epoch}

                if valid_stats['loss'] < val_loss_min:
                    save_checkpoint(args.log_dir + f"model_best_loss.pt", model)
                    print('new best auc ({:.5f} --> {:.5f}).'.format(val_loss_min, valid_stats['loss']))
                    val_loss_min = valid_stats['loss']
                
                with (Path(f'{args.log_dir}log.txt')).open('a') as f:
                    f.write(json.dumps(log_stats) + "\n")

            lr_scheduler.step()

    else: #Test phase
        epoch_add = '_loss'

        epoch=f'ps_03{epoch_add}'
        test_stats_03 = test_epoch(args, 
                                    epoch, 
                                    loader[2],
                                    model,
                                    clf_loss_func, 
                                    device)

        epoch=f'ps_04{epoch_add}'
        test_stats_04 = test_epoch(args, 
                                    epoch, 
                                    loader[3],
                                    model,
                                    clf_loss_func, 
                                    device)

        epoch=f'sch{epoch_add}'
        test_stats_sch = test_epoch(args, 
                                   epoch, 
                                   loader[4], 
                                   model, 
                                   clf_loss_func, 
                                   device)
        
        log_stats = {**{f'test_stats_03{k}': v for k, v in test_stats_03.items()},
                    **{f'test_stats_04{k}': v for k, v in test_stats_04.items()},
                    **{f'test_stats_sch_{k}': v for k, v in test_stats_sch.items()}
                    }
        
        with (Path(f'{args.log_dir}final{epoch_add}_test_log.txt')).open('a') as f:
            f.write(json.dumps(log_stats) + "\n")
    
if __name__ == '__main__':
    main()