import ptwt
import pywt
import torch
import torch.nn.parallel
import SimpleITK as sitk
from pathlib import Path
from utils import *
import torch.nn.functional as F
import SimpleITK as sitk
from tqdm import tqdm

def train_epoch(args,
                epoch,
                data_loader,
                model,
                reg_criterion,
                optimizer,
                device):

    model.train()
    reg_epoch_loss = 0
    epoch_iterator = tqdm(data_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for i, batch in enumerate(epoch_iterator):
        inputs = batch['image_train'].to(device).float()
        reg_labels = batch['label_train'].unsqueeze(1).to(device).float()

        if args.name in ['WDDAE', 'HWDAE']:
            coeffs3 = ptwt.wavedec3(inputs, pywt.Wavelet('haar'), level=1, mode='zero')
            inputs = torch.cat((coeffs3[0], 
                                coeffs3[1]['aad'], 
                                coeffs3[1]['ada'], 
                                coeffs3[1]['add'], 
                                coeffs3[1]['daa'], 
                                coeffs3[1]['dad'], 
                                coeffs3[1]['dda'], 
                                coeffs3[1]['ddd']), dim=1)

        optimizer.zero_grad()

        if args.name in ['WDDAE', 'DDAE']:
            times = torch.zeros((inputs.shape[0],), device = 'cuda').float().uniform_(0.002, 0.002)
            reg_outputs = model(inputs, times)
        else:
            reg_outputs = model(inputs)

        reg_loss = reg_criterion(reg_outputs, reg_labels)
        reg_loss.backward()

        optimizer.step()
        loss = reg_loss.item()
        reg_epoch_loss += reg_loss.item()
        epoch_iterator.set_description(f"Training ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")

    reg_epoch_loss /= (i+1)

    return {'loss': round(loss, 4)}

def val_epoch(args,
              epoch, 
              data_loader,
              model, 
              reg_criterion,
              device):
    
    print(f"validation at epoch {epoch}")
    model.eval()
    val_reg_epoch_loss = 0

    with torch.no_grad():
        model.eval()
        epoch_iterator = tqdm(data_loader, desc="Val (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        target_list = []
        output_list = []
        
        for j, val_batch in enumerate(epoch_iterator):
            val_inputs = val_batch['image_val'].to(device).float()
            reg_labels = val_batch['label_val'].unsqueeze(1).to(device).float()
            
            if args.name in ['WDDAE', 'HWDAE']:
                coeffs3 = ptwt.wavedec3(val_inputs, pywt.Wavelet('haar'), level=1, mode='zero')
                val_inputs = torch.cat((coeffs3[0], 
                                        coeffs3[1]['aad'], 
                                        coeffs3[1]['ada'], 
                                        coeffs3[1]['add'], 
                                        coeffs3[1]['daa'], 
                                        coeffs3[1]['dad'], 
                                        coeffs3[1]['dda'], 
                                        coeffs3[1]['ddd']), dim=1)

            if args.name in ['WDDAE', 'DDAE']:
                times = torch.zeros((val_inputs.shape[0],), device = 'cuda').float().uniform_(0.002, 0.002)
                reg_outputs = model(val_inputs, times)
            else:
                reg_outputs = model(val_inputs)

            reg_loss = reg_criterion(reg_outputs, reg_labels)
            val_reg_epoch_loss += reg_loss.item()
            
            target_list.append(reg_labels.item())
            reg_outputs = round(reg_outputs.item(), 4)
            
            output_list.append(reg_outputs)
            epoch_iterator.set_description(f"Val ({epoch} / {args.max_epochs} Steps) (loss={reg_loss.item():2.5f})")
            
        loss = val_reg_epoch_loss / (j+1)
        valid_stats = get_evaluation_metrics(args, epoch, target_list, output_list)
        print(f"Val ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
            
        return loss, valid_stats


def test_epoch(args,
              epoch, 
              data_loader,
              model, 
              reg_criterion,
              device):
    
    epoch = epoch
    print(f"test at epoch {epoch}")
    model.eval()
    ts_reg_epoch_loss = 0

    with torch.no_grad():
        model.eval()
        epoch_iterator = tqdm(data_loader, desc="test (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        target_list = []
        output_list = []
        
        for j, val_batch in enumerate(epoch_iterator):
            val_inputs = val_batch['image_ts'].to(device).float()
            reg_labels = val_batch['label_ts'].unsqueeze(1).to(device).float()
                
            if args.name in ['WDDAE', 'HWDAE']:
                coeffs3 = ptwt.wavedec3(val_inputs, pywt.Wavelet('haar'), level=1, mode='zero')
                val_inputs = torch.cat((coeffs3[0], 
                                        coeffs3[1]['aad'], 
                                        coeffs3[1]['ada'], 
                                        coeffs3[1]['add'], 
                                        coeffs3[1]['daa'], 
                                        coeffs3[1]['dad'], 
                                        coeffs3[1]['dda'], 
                                        coeffs3[1]['ddd']), dim=1)

            if args.name in ['WDDAE', 'DDAE']:
                times = torch.zeros((val_inputs.shape[0],), device = 'cuda').float().uniform_(0.002, 0.002)
                reg_outputs = model(val_inputs, times)
            else:
                reg_outputs = model(val_inputs)

            reg_loss = reg_criterion(reg_outputs, reg_labels)
            ts_reg_epoch_loss += reg_loss.item()
            
            target_list.append(reg_labels.item())
            reg_outputs = round(reg_outputs.item(), 4)
            
            output_list.append(reg_outputs)
            epoch_iterator.set_description(f"Ts ({epoch} / {args.max_epochs} Steps) (loss={reg_loss.item():2.5f})")
        
        loss = ts_reg_epoch_loss / (j+1)
        test_stats = get_evaluation_metrics(args, epoch, target_list, output_list)
        return loss, test_stats, target_list, output_list