import ptwt
import pywt
import torch
from utils import *
from tqdm import tqdm
import torch.nn.parallel
import torch.nn.functional as F

def train_epoch(args,
                epoch, 
                data_loader,
                model, 
                clf_criterion,
                optimizer, 
                device
                ):

    model.train()
    clf_epoch_loss = 0
    epoch_iterator = tqdm(data_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)

    for i, batch in enumerate(epoch_iterator):
        inputs = batch['image_train'].to(device).float()
        clf_labels = batch['label_train'].to(device).float()

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
            clf_outputs = model(inputs, times)
        else:
            clf_outputs = model(inputs)
        
        clf_loss = clf_criterion(clf_outputs, clf_labels)
        clf_loss.backward()

        optimizer.step()
        loss = clf_loss.item()
        clf_epoch_loss += clf_loss.item()
        epoch_iterator.set_description(f"Training ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
        
    clf_epoch_loss /= (i+1)

    return {'loss': round(loss, 4)}

def val_epoch(args,
              epoch, 
              data_loader,
              model, 
              clf_criterion,
              device
            ):
    
    print(f"validation at epoch {epoch}")
    model.eval()
    val_clf_epoch_loss = 0
    mode = 'val'
    
    with torch.no_grad():
        model.eval()

        epoch_iterator = tqdm(data_loader, desc="Val (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        acc_targets = []
        y_pred_prob = []
        
        for j, val_batch in enumerate(epoch_iterator):
            val_inputs = val_batch['image_val'].to(device).float()
            clf_labels = val_batch['label_val'].to(device).float()
            
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
                clf_outputs = model(val_inputs, times)
            else:
                clf_outputs = model(val_inputs)

            pred_prob = (F.softmax(clf_outputs, dim=1))
            y_pred_prob.append(pred_prob[:,1].detach().cpu().numpy().astype(float))
            
            clf_loss = clf_criterion(clf_outputs, clf_labels)
            val_clf_epoch_loss += clf_loss.item()

            acc_targets.append(clf_labels.argmax(dim=-1).item()) 
            
            epoch_iterator.set_description(f"Val ({epoch} / {args.max_epochs} Steps) (loss={clf_loss.item():2.5f})")

        loss = val_clf_epoch_loss / (j+1)
        
        auroc = AUROC(args, acc_targets, y_pred_prob, epoch, mode)
        auprc = AUPRC(args, acc_targets, y_pred_prob, epoch, mode)
        
        print(f"Val ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
        print(f"(auroc={auroc:2.5f}) (auprc={auprc:2.5f})")

        return {'loss': round(loss, 4), 
                'auroc': round(auroc, 4), 
                'auprc': round(auprc, 4)}

def test_epoch(args,
               epoch, 
               data_loader,
               model, 
               clf_criterion,
               device
               ):
    
    epoch = epoch
    print(f"test at epoch {epoch}")
    model.eval()
    val_clf_epoch_loss = 0
    mode = 'test'
    
    with torch.no_grad():
        model.eval()

        epoch_iterator = tqdm(data_loader, desc="test (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        acc_targets = []
        y_pred_prob = []
        
        for j, val_batch in enumerate(epoch_iterator):
            val_inputs = val_batch['image_ts'].to(device).float()
            clf_labels = val_batch['label_ts'].to(device).float()
            
            if args.name in ['WDDPM', 'HWDAE']:
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
                clf_outputs = model(val_inputs, times)
            else:
                clf_outputs = model(val_inputs)

            pred_prob = (F.softmax(clf_outputs, dim=1))
            y_pred_prob.append(pred_prob[:,1].detach().cpu().numpy().astype(float))
            clf_loss = clf_criterion(clf_outputs, clf_labels)
            val_clf_epoch_loss += clf_loss.item()

            acc_targets.append(clf_labels.argmax(dim=-1).item()) 
            
            epoch_iterator.set_description(f"test ({epoch} / {args.max_epochs} Steps) (loss={clf_loss.item():2.5f})")

        loss = val_clf_epoch_loss / (j+1)

        auroc = AUROC(args, acc_targets, y_pred_prob, epoch, mode)
        auprc = AUPRC(args, acc_targets, y_pred_prob, epoch, mode)

        print(f"Test ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
        print(f"(auroc1={auroc:2.5f}) (auprc={auprc:2.5f})")

        return {'loss': round(loss, 4), 
                'auroc': round(auroc, 4), 
                'auprc': round(auprc, 4)}