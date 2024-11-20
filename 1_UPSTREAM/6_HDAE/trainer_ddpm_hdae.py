import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
import SimpleITK as sitk
from ema_pytorch import EMA
import torch.nn.functional as F
from generative.inferers import DiffusionInferer_ae
from generative.networks.schedulers import DDPMScheduler

def train_ddpm_hdae_epoch(model,
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

def val_ddpm_hdae_epoch(model,
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

            timesteps = torch.randint(0, 1000, (images_ori.shape[0],), device=images_ori.device).long()

            noise = torch.randn_like(images_ori).to('cuda')

            latent = model.semantic_encoder(images_ori)
            
            latent_ema = ema_model.ema_model.semantic_encoder(images_ori)

            noise_pred = inferer(inputs=images_ori, 
                                 diffusion_model=model.unet, 
                                 noise=noise, 
                                 timesteps=timesteps, 
                                 cond = latent)
            
            val_loss = F.mse_loss(noise_pred.float(), noise.float())

            noise_pred_ema = inferer(inputs=images_ori, 
                                    diffusion_model=ema_model.ema_model.unet, 
                                    noise=noise, 
                                    timesteps=timesteps, 
                                    cond = latent_ema)
            
            val_loss_ema = F.mse_loss(noise_pred_ema.float(), noise.float())

        val_epoch_loss += val_loss.item()
        val_epoch_loss_ema += val_loss_ema.item()

        print({"val_loss": val_epoch_loss / (idx + 1)})
        print({"val_loss_ema": val_epoch_loss_ema / (idx + 1)})
        save_image(images_ori, 'ori', args, epoch)

        _, _, H, W, D = images_ori.shape
        image = torch.randn((1, 1, H, W, D))
        image = image.to("cuda")
        scheduler.set_timesteps(num_inference_steps=1000)
        image_pred = inferer.sample(input_noise=image, 
                                    diffusion_model=model.unet, 
                                    scheduler=scheduler, 
                                    save_intermediates=False, 
                                    cond=latent)

        save_image(image_pred, 'val', args, epoch)

        if epoch > 100:
            image_pred = inferer.sample(input_noise=image, 
                                        diffusion_model=ema_model.ema_model.unet, 
                                        scheduler=scheduler, 
                                        save_intermediates=False, 
                                        cond=latent_ema)

            save_image(image_pred, 'val_ema', args, epoch)
        
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


def run_training_hdae(model,
                      train_loader,
                      val_loader,
                      optimizer,
                      lr_scheduler,
                      args,
                      ):

    scheduler = DDPMScheduler(num_train_timesteps=1000, 
                              schedule="scaled_linear_beta", 
                              beta_start=0.0005, 
                              beta_end=0.0195)

    inferer = DiffusionInferer_ae(scheduler)

    n_epochs = args.max_epochs
    start_epoch = 0
    ema_model = EMA(model, 
                    beta=0.99,
                    update_after_step=15000,
                    update_every=10)
    
    for epoch in range(start_epoch, n_epochs):
        epoch_tr_loss = train_ddpm_hdae_epoch(model,
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

        if (epoch) % 2 == 0 and epoch>10:
            save_checkpoint(model,
                            ema_model,
                            epoch,
                            args.log_dir,
                            filename=f'model_{epoch}_final.pt',
                            optimizer=optimizer,
                            scheduler=lr_scheduler)

            epoch_val_loss = val_ddpm_hdae_epoch(model,
                                                 ema_model,
                                                 scheduler,
                                                 inferer,
                                                 val_loader,
                                                 epoch,
                                                 args=args
                                                 )

    return epoch_val_loss

