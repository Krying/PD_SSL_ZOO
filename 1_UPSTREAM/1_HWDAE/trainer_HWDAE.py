import os
import json
import pywt
import ptwt
import torch
from tqdm import tqdm
import SimpleITK as sitk
from pathlib import Path
from ema_pytorch import EMA
import torch.nn.functional as F
from generative.inferers import DiffusionInferer_ae
from generative.networks.schedulers import DDPMScheduler

def train_HWDAE_epoch(model,
                      ema_model,
                      inferer,
                      train_loader,
                      optimizer,
                      epoch):

    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    
    for idx, batch_data in progress_bar:
        images = batch_data.to('cuda')
        
        coeffs3 = ptwt.wavedec3(images, pywt.Wavelet('haar'), level=1, mode='zero')
        images = torch.cat((coeffs3[0], 
                            coeffs3[1]['aad'], 
                            coeffs3[1]['ada'], 
                            coeffs3[1]['add'], 
                            coeffs3[1]['daa'], 
                            coeffs3[1]['dad'], 
                            coeffs3[1]['dda'], 
                            coeffs3[1]['ddd']), dim=1)
            
        optimizer.zero_grad(set_to_none=True)
        
        noise = torch.randn_like(images).to('cuda')

        timesteps = torch.randint(0, 1000, (images.shape[0],), device=images.device).long()

        latent = model.semantic_encoder(images)

        noise_pred = inferer(inputs=images, 
                             diffusion_model=model.unet, 
                             noise=noise, 
                             timesteps=timesteps, 
                             cond = latent)
        
        loss = F.mse_loss(noise_pred.float(), noise.float())
        
        loss.backward()

        optimizer.step()
        ema_model.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": epoch_loss / (idx + 1)})
        
    return epoch_loss / (idx + 1)

def val_HWDAE_epoch(model,
                    ema_model,
                    scheduler,
                    inferer,
                    val_loader, 
                    epoch,
                    args):
    
    model.eval()
    ema_model.ema_model.eval()
    val_epoch_loss = 0
    val_epoch_loss_ema = 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            images_ori = batch_data.to('cuda')

            coeffs3 = ptwt.wavedec3(images_ori, pywt.Wavelet('haar'), level=1, mode='zero')
            images = torch.cat((coeffs3[0], 
                                coeffs3[1]['aad'], 
                                coeffs3[1]['ada'], 
                                coeffs3[1]['add'], 
                                coeffs3[1]['daa'], 
                                coeffs3[1]['dad'], 
                                coeffs3[1]['dda'], 
                                coeffs3[1]['ddd']), dim=1)

            timesteps = torch.randint(0, 1000, (images.shape[0],), device=images.device).long()
            
            noise = torch.randn_like(images).to('cuda')

            latent = model.semantic_encoder(images)

            noise_pred = inferer(inputs=images, 
                                 diffusion_model=model.unet, 
                                 noise=noise, 
                                 timesteps=timesteps, 
                                 cond = latent)
            
            val_loss = F.mse_loss(noise_pred.float(), noise.float())

            noise_pred_ema = inferer(inputs=images, 
                                     diffusion_model=ema_model.ema_model.unet, 
                                     noise=noise, 
                                     timesteps=timesteps, 
                                     cond = latent)
            
            val_loss_ema = F.mse_loss(noise_pred_ema.float(), noise.float())

        val_epoch_loss += val_loss.item()
        val_epoch_loss_ema += val_loss_ema.item()
        latent_ema = ema_model.ema_model.semantic_encoder(images)
    
        print({"val_loss": val_epoch_loss / (idx + 1)})
        print({"val_loss_ema": val_epoch_loss_ema / (idx + 1)})
        save_image(images_ori, 'ori', args, epoch)

        _, _, H, W, D = images.shape
        image = torch.randn((1, 8, H, W, D))
        image = image.to("cuda")
        scheduler.set_timesteps(num_inference_steps=1000)
        image_pred = inferer.sample(input_noise=image, 
                                    diffusion_model=model.unet, 
                                    scheduler=scheduler, 
                                    save_intermediates=False, 
                                    cond=latent)

        coeffs3[0] = image_pred[:,0:1,:,:,:]
        coeffs3[1]['aad'] = image_pred[:,1:2,:,:,:]
        coeffs3[1]['ada'] = image_pred[:,2:3,:,:,:]
        coeffs3[1]['add'] = image_pred[:,3:4,:,:,:]
        coeffs3[1]['daa'] = image_pred[:,4:5,:,:,:]
        coeffs3[1]['dad'] = image_pred[:,5:6,:,:,:]
        coeffs3[1]['dda'] = image_pred[:,6:7,:,:,:]
        coeffs3[1]['ddd'] = image_pred[:,7:8,:,:,:]

        reconstruction = ptwt.waverec3(coeffs3, pywt.Wavelet("haar"))            
        
        save_image(reconstruction, 'val', args, epoch)
        
        if epoch>100 and epoch%args.val_interval==0:
            image_pred = inferer.sample(input_noise=image, 
                                        diffusion_model=ema_model.ema_model.unet, 
                                        scheduler=scheduler, 
                                        save_intermediates=False, 
                                        cond=latent_ema)

            coeffs3[0] = image_pred[:,0:1,:,:,:]
            coeffs3[1]['aad'] = image_pred[:,1:2,:,:,:]
            coeffs3[1]['ada'] = image_pred[:,2:3,:,:,:]
            coeffs3[1]['add'] = image_pred[:,3:4,:,:,:]
            coeffs3[1]['daa'] = image_pred[:,4:5,:,:,:]
            coeffs3[1]['dad'] = image_pred[:,5:6,:,:,:]
            coeffs3[1]['dda'] = image_pred[:,6:7,:,:,:]
            coeffs3[1]['ddd'] = image_pred[:,7:8,:,:,:]
            
            reconstruction_ema = ptwt.waverec3(coeffs3, pywt.Wavelet("haar"))       

            save_image(reconstruction_ema, 'val_ema', args, epoch)
    
    return val_epoch_loss / (idx + 1)

def save_checkpoint(model,
                    ema_model,
                    epoch,
                    ckpt_logdir,
                    filename='model.pt',
                    optimizer=None,
                    scheduler=None):
    
    state_dict = model.state_dict()
    state_dict_ema = ema_model.state_dict()
    save_dict = {
            'epoch': epoch,
            'state_dict': state_dict,
            }
    if epoch > 100:
        save_dict['ema'] = state_dict_ema
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(ckpt_logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def save_image(pred, phase, args, epoch):
    pred_img = pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    save_pred = sitk.GetImageFromArray(pred_img)
    sitk.WriteImage(save_pred, f"{args.img_save_dir}/{phase}_{epoch}_pred.nii.gz")

def run_training_hwdae(model,
                       train_loader,
                       val_loader,
                       optimizer,
                       lr_scheduler,
                       args,
                       ):

    scheduler = DDPMScheduler(num_train_timesteps=1000, 
                              schedule="linear_beta", 
                              beta_start=0.0005, 
                              beta_end=0.0195)

    inferer = DiffusionInferer_ae(scheduler)

    ema_model = EMA(model, 
                    beta=0.995, 
                    update_after_step=15000,
                    update_every=10)
    
    for epoch in range(0, args.max_epochs):
        epoch_tr_loss = train_HWDAE_epoch(model,
                                          ema_model,
                                          inferer,
                                          train_loader,
                                          optimizer,
                                          epoch)

        log_stats = {f'train_{epoch}': epoch_tr_loss}
        with (Path(args.log_dir) / 'log.txt').open('a') as f:
            f.write(json.dumps(log_stats) + "\n")
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"lr : {lr}")
        lr_scheduler.step()

        if (epoch) % args.val_interval == 0:
            save_checkpoint(model,
                            ema_model,
                            epoch,
                            args.log_dir,
                            filename=f'model_{epoch}.pt',
                            optimizer=optimizer,
                            scheduler=lr_scheduler)
            
            epoch_val_loss = val_HWDAE_epoch(model,
                                             ema_model,
                                             scheduler,
                                             inferer,
                                             val_loader,
                                             epoch,
                                             args=args
                                             )


    return epoch_val_loss

