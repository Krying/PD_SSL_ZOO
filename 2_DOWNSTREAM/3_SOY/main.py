import os
import sys
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/DOWNSTREAM/3_SOY/'))

import json
import torch
import random
import argparse
import numpy as np
from model import *
from utils import *
from pathlib import Path
from trainer_soy import *
from data_utils import get_loader_reg
from lr_scheduler import create_scheduler
 
parser = argparse.ArgumentParser(description="SOY")
parser.add_argument('--test', default=0, type=int)
parser.add_argument('--fold', default=None, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--min_lr", default=1e-8, type=float)
parser.add_argument("--max_epochs", default=150, type=int)
parser.add_argument("--eta_max", default=1e-4, type=float)
parser.add_argument("--optim_lr", default=5e-5, type=float)
parser.add_argument("--warm_up_epoch", default=10, type=int)
parser.add_argument("--start_decay_epoch", default=150, type=int)
parser.add_argument('--cuda_visible_devices', default='0', type=str)
parser.add_argument("--down_type", default=None, type=str, choices=["1_EP", "2_PMP", "3_SOY"])
parser.add_argument("--log_dir", default="/workspace/PD_SSL_ZOO/2_DOWNSTREAM/3_SOY/", type=str)
parser.add_argument("--log_dir_png", default="/workspace/PD_SSL_ZOO/2_DOWNSTREAM/3_SOY/", type=str)
parser.add_argument('--data_per', default=None, type=int, choices=[100, 25]) #for data stress test
parser.add_argument("--linear_mode", default='linear', type=str, choices=["scratch", "linear", "fine_tuning"])
parser.add_argument("--name", default=None, type=str, choices=["HWDAE", "WDDAE", "DDAE", "P2S2P", "DisAE", "HDAE", "SimMIM"])

def main():
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_visible_devices

    args.down_type = '3_SOY'
    args.num_class = 1
    
###########################LINEAR_PROBING###########################
    if args.linear_mode == 'linear':
        args.optim_lr = 1e-3
        args.batch_size = 6

        args.max_epochs = 30
        args.warm_up_epoch = 10

        print(f"REG LINEAR TRAIN START")
        print(f"{args.name}_{args.data_per} TRAIN PROCESS START")

        if args.data_per == 100:
            for i in range(5):
                args.fold = i+1
                print(f"FOLD_{args.fold} TRAIN PROCESS START")
                loaders = get_loader_reg(args)

                args.test = 0
                main_worker(args=args, loader=loaders)

                args.log_dir = args.log_dir + f'test/'
                args.log_dir_png = args.log_dir
                os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 

                args.test = 1
                main_worker(args=args, loader=loaders)
                
        elif args.data_per == 25:
            for i in range(10):
                args.fold = (2*i)+1
                print(f"FOLD_{args.fold} TRAIN PROCESS START")
                loaders = get_loader_reg(args)

                args.test = 0
                main_worker(args=args, loader=loaders)

                args.log_dir = args.log_dir + f'test/'
                args.log_dir_png = args.log_dir
                os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 

                args.test = 1
                main_worker(args=args, loader=loaders)

##############################SCRATCH###############################
    elif args.linear_mode == 'scratch':
        args.optim_lr = 1e-4
        if args.name in ['SimMIM', 'DisAE', 'P2S2P']:
            args.batch_size = 3
        elif args.name in [ 'HWDAE', 'WDDPM', 'DDAE']:
            args.batch_size = 6
            
        print("REG SCRATCH TRAIN START")
        print(f"{args.name}_{args.data_per} TRAIN PROCESS START")
        
        if args.data_per == 100:
            args.max_epochs = 30
            args.warm_up_epoch = 10
                
            for i in range(5):
                args.fold = i+1
                print(f"FOLD_{args.fold} TRAIN PROCESS START")
                loaders = get_loader_reg(args)
                
                args.test = 0
                main_worker(args=args, loader=loaders)                
                
                args.log_dir = args.log_dir + f'test/'
                args.log_dir_png = args.log_dir
                os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 
                
                args.test = 1
                main_worker(args=args, loader=loaders)
            
        elif args.data_per == 25:
            args.max_epochs = 120
            args.warm_up_epoch = 60
            
            for i in range(10):
                args.fold = (i*2)+1
                print(f"FOLD_{args.fold} TRAIN PROCESS START")
                loaders = get_loader_reg(args)
                
                args.test = 0
                main_worker(args=args, loader=loaders)
                
                args.log_dir = args.log_dir + f'test/'
                args.log_dir_png = args.log_dir
                os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 

                args.test = 1
                main_worker(args=args, loader=loaders)

############################FINE_TUNING#############################
    elif args.linear_mode == 'fine_tuning':
        args.optim_lr = 1e-4
        if args.name in ['SimMIM', 'DisAE', 'P2S2P']:
            args.batch_size = 3
        elif args.name in [ 'HWDAE', 'WDDPM', 'DDAE']:
            args.batch_size = 6
            
        print("REG FINE_TUNING TRAIN START")
        print(f"{args.name}_{args.data_per} TRAIN PROCESS START")
        
        if args.data_per == 100:
            args.max_epochs = 30
            args.warm_up_epoch = 10

            for i in range(5):
                args.fold = i+1
                print(f"FOLD_{args.fold} TRAIN PROCESS START")
                loaders = get_loader_reg(args)

                args.test = 0
                main_worker(args=args, loader=loaders)
                
                args.log_dir = args.log_dir + f'test/'
                args.log_dir_png = args.log_dir
                os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 

                args.test = 1
                main_worker(args=args, loader=loaders)

        elif args.data_per == 25:
            args.max_epochs = 120
            args.warm_up_epoch = 60
            
            for i in range(10):
                args.fold = (i*2)+1
                print(f"FOLD_{args.fold} TRAIN PROCESS START")
                loaders = get_loader_reg(args)
                
                args.test = 0
                main_worker(args=args, loader=loaders)
    
                args.log_dir = args.log_dir + f'test/'
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
        if args.linear_mode == 'linear':
            if args.name in ['SimMIM', 'DisAE']:
                n_parameters = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
                model.head.requires_grad_(True)
            else:
                n_parameters = sum(p.numel() for p in model.linear.parameters() if p.requires_grad)
                model.linear.requires_grad_(True)

            print('Number of Learnable Params:', n_parameters)

            if args.name in ['SimMIM', 'DisAE']:
                optimizer = torch.optim.AdamW(params=model.head.parameters(), lr=args.min_lr, weight_decay=0.05)
            else:
                optimizer = torch.optim.AdamW(params=model.linear.parameters(), lr=args.min_lr, weight_decay=0.05)

        elif args.linear_mode == 'fine_tuning':
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of Learnable Params:', n_parameters)
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.min_lr, weight_decay=0.05)

        elif args.linear_mode == 'scratch':
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('Number of Learnable Params:', n_parameters)
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.min_lr, weight_decay=0.05)

    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.min_lr, weight_decay=0.05)

    criterion = nn.HuberLoss()

    if args.linear_mode == 'linear':
        lr_scheduler = create_scheduler(name='poly_lr', optimizer=optimizer, args=args)
    elif  args.linear_mode == 'scratch':
        lr_scheduler = create_scheduler(name='cosine_annealing_warm_restart', optimizer=optimizer, args=args)
    elif  args.linear_mode == 'fine_tuning':
        lr_scheduler = create_scheduler(name='cosine_annealing_warm_restart', optimizer=optimizer, args=args)

    if args.test == 0:
        if args.linear_mode == 'linear':
            args.log_dir = "/workspace/PD_SSL_ZOO/2_DOWNSTREAM/" + f"{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/"
        elif args.linear_mode == 'fine_tuning':
            args.log_dir = "/workspace/PD_SSL_ZOO/2_DOWNSTREAM/" + f"{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/"
        elif args.linear_mode == 'scratch':
            args.log_dir = "/workspace/PD_SSL_ZOO/2_DOWNSTREAM/" + f"{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/"
            
        args.log_dir_png = args.log_dir + "png/"
        os.makedirs(args.log_dir, mode=0o777, exist_ok=True) 
        os.makedirs(args.log_dir_png, mode=0o777, exist_ok=True)

        val_loss_max = 5

        for epoch in range(args.max_epochs): #Train phase / Valid phase
            train_stats = train_epoch(args, 
                                      epoch, 
                                      loader[0], 
                                      model, 
                                      criterion, 
                                      optimizer, 
                                      device)
            
            lr = optimizer.param_groups[0]["lr"]
            print(f"lr : {lr}")
            train_stats['lr'] = lr

            if epoch > args.warm_up_epoch:
                loss, val_stats = val_epoch(args, 
                                            epoch, 
                                            loader[1], 
                                            model, 
                                            criterion, 
                                            device)

                val_stats['loss'] = loss
                
                #loss best weight
                if val_stats['loss'] < val_loss_max:
                    save_checkpoint(args.log_dir + f"model_best_loss.pt", model)
                    print('new best loss ({:.5f} --> {:.5f}).'.format(val_loss_max, val_stats['loss']))
                    val_loss_max = val_stats['loss']

                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'valid_{k}': v for k, v in val_stats.items()},
                             'epoch': epoch}
                
                with (Path(f'{args.log_dir}log.txt')).open('a') as f:
                    f.write(json.dumps(log_stats) + "\n")

            lr_scheduler.step()

    else:
        test_03_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS03_onset_test.json"
        test_04_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS04_onset_test.json"
        
        with open(test_03_list, 'r') as test_03_file:
            test_03_files = json.load(test_03_file)

        with open(test_04_list, 'r') as test_04_file:
            test_04_files= json.load(test_04_file)

        files_ts_03 = []
        files_ts_04 = []

        for file_name, label in test_03_files['test'].items():
            files_ts_03.append({"image_ts": file_name, "label_ts": label})

        for file_name, label in test_04_files['test_04'].items():
            files_ts_04.append({"image_ts": file_name, "label_ts": label})
            
        epoch_add = '_loss'

        epoch=f'ps_03{epoch_add}'
        ps03_json_list = []

        loss, test_stats, target_list, output_list = test_epoch(args, 
                                                                epoch, 
                                                                loader[2], 
                                                                model, 
                                                                criterion, 
                                                                device)

        test_stats['loss'] = loss

        ps03_json_list.append({**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch})
        ps03_json_list.append({"target" : target_list, "pred" : output_list, "path" : files_ts_03})

        with (Path(f'{args.log_dir}final_03_log{epoch_add}_stats.txt')).open('a') as f:
            f.write(json.dumps(ps03_json_list[0]) + "\n")
        with (Path(f'{args.log_dir}final_03_log{epoch_add}_outcomes.txt')).open('a') as f:
            f.write(json.dumps(ps03_json_list[1]) + "\n")

        epoch=f'ps_04{epoch_add}'
        ps04_json_list = []
        loss, test_stats, target_list, output_list  = test_epoch(args, 
                                                                 epoch, 
                                                                 loader[3], 
                                                                 model, 
                                                                 criterion, 
                                                                 device)
        
        test_stats['loss'] = loss

        ps04_json_list.append({**{f'test_{k}': v for k, v in test_stats.items()}, 'epoch': epoch})
        ps04_json_list.append({"target" : target_list, "pred" : output_list, "path" : files_ts_04})

        with (Path(f'{args.log_dir}final_04_log{epoch_add}_stats.txt')).open('a') as f:
            f.write(json.dumps(ps04_json_list[0]) + "\n")
        with (Path(f'{args.log_dir}final_04_log{epoch_add}_outcomes.txt')).open('a') as f:
            f.write(json.dumps(ps04_json_list[1]) + "\n")
        
if __name__ == '__main__':
    main()