# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import shutil
import numpy as np
import torch
import torch.nn.parallel
import SimpleITK as sitk
from metric import *
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

# from utils import *
# import cv2


from tqdm import tqdm

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, int(im_size/2), 1).uniform_(0., 1.).cuda(device)

def train_epoch(encoder,
                decoder,
                criterion,
                train_loader,
                optimizer,
                epoch,
                args):
    
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    encoder.train()
    batch_loss = 0
    batch_dict = {}
    for idx, batch_data in enumerate(epoch_iterator):

        input_noise_1 = image_noise(1, 192, 'cuda')
        input_noise_2 = image_noise(1, 192, 'cuda')
        
        x = batch_data['image'].to('cuda').float()
        codes = encoder(x)
        pred = decoder(codes, input_noise_1, input_noise_2)
        optimizer.zero_grad()

        loss = F.mse_loss(pred, x)

        loss.backward()
        optimizer.step()
        epoch_iterator.set_description(f"Training ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
        batch_loss += loss.item()
    
    print(batch_loss / (idx+1))
    
    return batch_loss / (idx+1)


#get PSNR and LPIPS
def val_epoch(encoder,
              decoder,
              val_loader, 
              epoch, 
              eval_psnr, 
              eval_lpips,
              args):
    
    encoder.eval()
    psnr_list = []
    epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)

    with torch.no_grad():
        for idx, batch_data in enumerate(epoch_iterator_val):
            input_noise_1 = image_noise(1, 192, 'cuda')
            input_noise_2 = image_noise(1, 192, 'cuda')
            x = batch_data['image'].to('cuda').float()
            codes = encoder(x)
            pred = decoder(codes, input_noise_1, input_noise_2)
            
            img1 = pred.cpu().detach().numpy().squeeze()
            img2 = x.cpu().detach().numpy().squeeze()
            psnr_val = eval_psnr(img1, img2)
            epoch_iterator_val.set_description(f"Validate ({epoch} / {args.max_epochs} Steps) (loss={psnr_val:2.5f})")

            psnr_list.append(psnr_val)
        
        save_image(x, pred, 'val', args, epoch)

        psnr_mean = np.mean(psnr_list)


    return psnr_mean#, lpips_mean

def save_checkpoint(model,
                    epoch,
                    ckpt_logdir,
                    filename='model.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(ckpt_logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def save_image(x, pred, phase, args, epoch):
    pred_img = pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    pred_img = np.fliplr(pred_img)
    ori_img = x.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    ori_img = np.fliplr(ori_img)
    save_pred = sitk.GetImageFromArray(pred_img)
    save_ori = sitk.GetImageFromArray(ori_img)
    sitk.WriteImage(save_pred, f'/workspace/PD_SSL_ZOO/1_UPSTREAM/4_Pixel2Style2Pixel/outputs/pred_{epoch}.nii.gz')
    sitk.WriteImage(save_ori, f'/workspace/PD_SSL_ZOO/1_UPSTREAM/4_Pixel2Style2Pixel/outputs/ori_{epoch}.nii.gz')
        
import json
def run_training(enc,
                 dec,
                 criterion,
                 eval_psnr,
                 eval_lpips,
                 train_loader,
                 val_loader,
                 optimizer,
                 args,
                 scheduler=None,
                 start_epoch=0
                 ):

    val_psnr_max = 0.

    # if args.tsne == False:
    for epoch in range(start_epoch, args.max_epochs):
        tr_stat = train_epoch(enc,
                              dec,
                              criterion,
                              train_loader,
                              optimizer,
                              epoch=epoch,
                              args=args)

        print("Averaged train_stats: ", tr_stat)
        log_stats = {f'train_{epoch}': tr_stat}
        with (Path('/workspace/PD_SSL_ZOO/1_UPSTREAM/4_Pixel2Style2Pixel/outputs/log.txt')).open('a') as f:
            f.write(json.dumps(log_stats) + "\n")

        b_new_best = False
        if (epoch+1) % 1 == 0:
            val_psnr = val_epoch(enc,
                                 dec,
                                 val_loader,
                                 epoch,
                                 eval_psnr=eval_psnr,
                                 eval_lpips=eval_lpips,
                                 args=args
                                 )

            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), 'psnr', val_psnr)
            
            val_log_stats = {f'val_{epoch}': val_psnr}
            with (Path('/workspace/PD_SSL_ZOO/1_UPSTREAM/4_Pixel2Style2Pixel/outputs/log.txt')).open('a') as f:
                f.write(json.dumps(val_log_stats) + "\n")

            if val_psnr > val_psnr_max:
                save_checkpoint(enc,
                                epoch,
                                args.log_dir,
                                best_acc=val_psnr_max,
                                filename=f'model_{epoch}_final.pt',
                                optimizer=optimizer)

                print('new best ({:.6f} --> {:.6f}). '.format(val_psnr_max, val_psnr))
                val_psnr_max = val_psnr
                b_new_best = True
            
            if b_new_best:
                print('Copying to model.pt new best model!!!!')
                shutil.copyfile(os.path.join(args.log_dir, f'model_{epoch}_final.pt'), os.path.join(args.log_dir, 'model_best.pt'))

        if scheduler is not None:
            scheduler.step()
    print('Training Finished !, Best Accuracy: ', val_psnr_max)

    return val_psnr_max
