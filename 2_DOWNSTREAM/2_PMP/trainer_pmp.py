import ptwt
import pywt
import torch
from utils import *
import numpy as np
from tqdm import tqdm
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

        acc_targets = []
        acc_outputs = []
        epoch_iterator = tqdm(data_loader, desc="Val (X / X Steps) (loss=X.X)", dynamic_ncols=True)
        y_true_one_hot = []
        y_pred_prob = []
        y_pred_prob_0 = []
        y_pred_prob_1 = []
        y_pred_prob_2 = []
        
        y_true_one_hot_0 = []
        y_true_one_hot_1 = []
        y_true_one_hot_2 = []
        
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

            y_pred_prob.append(pred_prob.detach().cpu().numpy().astype(float))
            y_pred_prob_0.append(pred_prob[:,0].detach().cpu().numpy().astype(float))
            y_pred_prob_1.append(pred_prob[:,1].detach().cpu().numpy().astype(float))
            y_pred_prob_2.append(pred_prob[:,2].detach().cpu().numpy().astype(float))                
            
            clf_loss = clf_criterion(clf_outputs, clf_labels)
            val_clf_epoch_loss += clf_loss.item()

            acc_targets.append(clf_labels.argmax(dim=-1).item()) 
            acc_outputs.append(clf_outputs.argmax(dim=1).item())

            y_true_one_hot.append(clf_labels.detach().cpu().numpy())
            y_true_one_hot_0.append(clf_labels[:,0].detach().cpu().numpy())
            y_true_one_hot_1.append(clf_labels[:,1].detach().cpu().numpy())
            y_true_one_hot_2.append(clf_labels[:,2].detach().cpu().numpy())
            
            epoch_iterator.set_description(f"Val ({epoch} / {args.max_epochs} Steps) (loss={clf_loss.item():2.5f})")

        loss = val_clf_epoch_loss / (j+1)
        Y_test = np.concatenate(y_true_one_hot) #concatenate true label align as batch_axis for calculate PRC
        y_score = np.concatenate(y_pred_prob) #concatenate pred value align as batch_axis for calculate PRC
        
        auroc = AUROC(args, y_true_one_hot_0, y_true_one_hot_1, y_true_one_hot_2, y_pred_prob_0, y_pred_prob_1, y_pred_prob_2, Y_test, y_score, epoch, mode)
        auprc = AUPRC(args, Y_test, y_score, epoch, mode)
        
        print(f"Val ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
        print("")
        print(f"(auroc_pd={auroc[0]:2.5f}) (auroc_msa={auroc[1]:2.5f}) (auroc_psp={auroc[2]:2.5f})")
        print(f"(auroc_avg_MACRO={auroc[3]:2.5f}) (auroc_avg_micro={auroc[4]:2.5f})")
        print("")
        print(f"(auprc_pd={auprc[0]:2.5f}) (auprc_msa={auprc[1]:2.5f}) (auprc_psp={auprc[2]:2.5f})")
        print(f"(auprc_avg_MACRO={auprc[3]:2.5f}) (auprc_avg_micro={auprc[4]:2.5f})")

        return {'loss': round(loss, 4),
                'auroc_pd': round(auroc[0], 4),
                'auroc_msa': round(auroc[1], 4),
                'auroc_psp': round(auroc[2], 4),
                'auroc_avg_macro': round(auroc[3], 4),
                'auroc_avg_micro': round(auroc[4], 4),
                'auprc_pd': round(auprc[0], 4),
                'auprc_msa': round(auprc[1], 4),
                'auprc_psp': round(auprc[2], 4),
                'auprc_avg_macro': round(auprc[3], 4),
                'auprc_avg_micro': round(auprc[4], 4)}

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
    ts_clf_epoch_loss = 0
    mode = 'test'
    with torch.no_grad():
        model.eval()

        acc_targets = []
        acc_outputs = []
        epoch_iterator = tqdm(data_loader, desc="test (X / X Steps) (loss=X.X)", dynamic_ncols=True)

        y_true_one_hot = []
        y_pred_prob = []
        
        y_pred_prob_0 = []
        y_pred_prob_1 = []
        y_pred_prob_2 = []
        
        y_true_one_hot_0 = []
        y_true_one_hot_1 = []
        y_true_one_hot_2 = []
        
        for j, val_batch in enumerate(epoch_iterator):
            val_inputs = val_batch['image_ts'].to(device).float()
            clf_labels = val_batch['label_ts'].to(device).float()
            
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

            y_pred_prob.append(pred_prob.detach().cpu().numpy().astype(float))
            y_pred_prob_0.append(pred_prob[:,0].detach().cpu().numpy().astype(float))
            y_pred_prob_1.append(pred_prob[:,1].detach().cpu().numpy().astype(float))
            y_pred_prob_2.append(pred_prob[:,2].detach().cpu().numpy().astype(float))                
            
            clf_loss = clf_criterion(clf_outputs, clf_labels)
            ts_clf_epoch_loss += clf_loss.item()

            acc_targets.append(clf_labels.argmax(dim=-1).item()) 
            acc_outputs.append(clf_outputs.argmax(dim=1).item())
            
            y_true_one_hot.append(clf_labels.detach().cpu().numpy())
            y_true_one_hot_0.append(clf_labels[:,0].detach().cpu().numpy())
            y_true_one_hot_1.append(clf_labels[:,1].detach().cpu().numpy())
            y_true_one_hot_2.append(clf_labels[:,2].detach().cpu().numpy())
            
            epoch_iterator.set_description(f"Test ({epoch} / {args.max_epochs} Steps) (loss={clf_loss.item():2.5f})")

        loss = ts_clf_epoch_loss / (j+1)

        Y_test = np.concatenate(y_true_one_hot) #concatenate true label align as batch_axis for calculate PRC
        y_score = np.concatenate(y_pred_prob) #concatenate pred value align as batch_axis for calculate PRC

        auroc = AUROC(args, y_true_one_hot_0, y_true_one_hot_1, y_true_one_hot_2, y_pred_prob_0, y_pred_prob_1, y_pred_prob_2, Y_test, y_score, epoch, mode)
        auprc = AUPRC(args, Y_test, y_score, epoch, mode)
        
        print(f"Test ({epoch} / {args.max_epochs} Steps) (loss={loss:2.5f})")
        print("")
        print(f"(auroc_pd={auroc[0]:2.5f}) (auroc_msa={auroc[1]:2.5f}) (auroc_psp={auroc[2]:2.5f})")
        print(f"(auroc_avg_MACRO={auroc[3]:2.5f}) (auroc_avg_micro={auroc[4]:2.5f})")
        print("")
        print(f"(auprc_pd={auprc[0]:2.5f}) (auprc_msa={auprc[1]:2.5f}) (auprc_psp={auprc[2]:2.5f})")
        print(f"(auprc_avg_MACRO={auprc[3]:2.5f}) (auprc_avg_micro={auprc[4]:2.5f})")


        return {'loss': round(loss, 4),
                'auroc_pd': round(auroc[0], 4),
                'auroc_msa': round(auroc[1], 4),
                'auroc_psp': round(auroc[2], 4),
                'auroc_avg_macro': round(auroc[3], 4),
                'auroc_avg_micro': round(auroc[4], 4),
                'auprc_pd': round(auprc[0], 4),
                'auprc_msa': round(auprc[1], 4),
                'auprc_psp': round(auprc[2], 4),
                'auprc_avg_macro': round(auprc[3], 4),
                'auprc_avg_micro': round(auprc[4], 4)}
