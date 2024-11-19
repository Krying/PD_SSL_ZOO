import os
import sys
import torch 
import torch.nn as nn
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/2_DOWNSTREAM/3_SOY/'))

def create_model(args):
    if args.name == "HWDAE":
        from PREVIOUS.HWDAE.generative.networks.nets.diffusion_model_aniso_unet_AE_official import DiffusionModelUNet_aniso_AE, DiffusionModelEncoder_ansio
        class HWDAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet =  DiffusionModelUNet_aniso_AE(spatial_dims=3,
                                                        in_channels=8, #wavelet
                                                        out_channels=8, #wavelet
                                                        num_channels=[128,128,256,256,512],
                                                        attention_levels=[False,False,False,False,True],
                                                        num_head_channels=[0,0,0,0,64],
                                                        norm_num_groups=32,
                                                        use_flash_attention=True,
                                                        iso_conv_down=(False, True, True, True, None),
                                                        iso_conv_up=(True, True, True, False, None),
                                                        num_res_blocks=2)

                self.semantic_encoder = DiffusionModelEncoder_ansio(spatial_dims=3,
                                                                    in_channels=8,
                                                                    out_channels=8,
                                                                    num_channels=[128,256,256,512],
                                                                    attention_levels=[False,False,False,False],
                                                                    num_head_channels=[0,0,0,0],
                                                                    norm_num_groups=32,
                                                                    iso_conv_down=(False, True, True, True),
                                                                    num_res_blocks=(2,2,2,2))

        model = HWDAE()

        def filter_ema_keys(checkpoint):
            ema_model_state_dict = {key.replace('ema_model.', ''): value 
                                    for key, value in checkpoint.items() 
                                    if 'online_model' not in key}
            del ema_model_state_dict['initted']
            del ema_model_state_dict['step']
            
            return ema_model_state_dict
        
        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/1_HWDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                checkpoint = ckpt['ema']
                new_ckpt = filter_ema_keys(checkpoint)
                model.load_state_dict(new_ckpt, strict=False)
                model = model.semantic_encoder
                model.requires_grad_(False)
                model.linear = nn.Linear(512*5, args.num_class)
                model.linear.requires_grad_(True)

            elif args.linear_mode == 'fine_tuning': #from linear
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/1_HWDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                checkpoint = ckpt['ema']
                new_ckpt = filter_ema_keys(checkpoint)
                model.load_state_dict(new_ckpt, strict=False)
                model = model.semantic_encoder
                
                model.linear = nn.Linear(512*5, args.num_class)
                
            elif args.linear_mode == 'scratch':
                model = model.semantic_encoder
                
                model.linear = nn.Linear(512*5, args.num_class)
                
                print("scratch")
        
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.semantic_encoder

                model.linear = nn.Linear(512*5, args.num_class)

                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'

                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.semantic_encoder

                model.linear = nn.Linear(512*5, args.num_class)

                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'scratch':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.semantic_encoder
                
                model.linear = nn.Linear(512*5, args.num_class)
               
                model.load_state_dict(ckpt['state_dict'])
                
    elif args.name == "WDDAE":
        from PREVIOUS.DDPM.generative.networks.nets.diffusion_model_aniso_unet import DiffusionModelUNet_aniso_enc
        
        def remove_ema_prefix(checkpoint):
            ema_model_state_dict = {key.replace('ema_model.model.', ''): value 
                                    for key, value in checkpoint.items() 
                                    if 'online_model' not in key}
            
            checkpoint['step'] = torch.Tensor(1)
            checkpoint['initted'] = torch.Tensor(1)
            
            return ema_model_state_dict
        
        model =  DiffusionModelUNet_aniso_enc(spatial_dims=3,
                                              in_channels=8, #wavelet
                                              out_channels=8, #wavelet
                                              num_channels=[128,128,256,512],
                                              attention_levels=[False,False,False,True],
                                              num_head_channels=[0,0,0,32],
                                              norm_num_groups=32,
                                              use_flash_attention=True,
                                              iso_conv_down=(False, True, True, None),
                                              iso_conv_up=(True, True, False, None),
                                              num_res_blocks=2,)
        
        model.linear = nn.Linear(512, args.num_class)
        
        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/DOWNSTREAM/WEIGHTS/2_WDDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                new_ckpt = remove_ema_prefix(ckpt['ema'])
                model.load_state_dict(new_ckpt, strict=False)
                model.requires_grad_(False)
                model.linear.requires_grad_(True)

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = '/workspace/PD_SSL_ZOO/DOWNSTREAM/WEIGHTS/2_WDDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                new_ckpt = remove_ema_prefix(ckpt['ema'])
                model.load_state_dict(new_ckpt, strict=False)

            elif args.linear_mode == 'scratch':
                print("scratch")
                pass
        
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'scratch':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

    elif args.name == "DDAE":
        from PREVIOUS.DDPM.generative.networks.nets.diffusion_model_aniso_unet import DiffusionModelUNet_aniso_enc
        
        def remove_ema_prefix(checkpoint):
            ema_model_state_dict = {key.replace('ema_model.model.', ''): value 
                                    for key, value in checkpoint.items() 
                                    if 'online_model' not in key}
            
            checkpoint['step'] = torch.Tensor(1)
            checkpoint['initted'] = torch.Tensor(1)
            
            return ema_model_state_dict
                
        model =  DiffusionModelUNet_aniso_enc(spatial_dims=3,
                                            in_channels=1, #wavelet
                                            out_channels=1, #wavelet
                                            num_channels=[16,32,64,128,256],
                                            attention_levels=[False,False,False,False,True],
                                            num_head_channels=[0,0,0,0,32],
                                            norm_num_groups=16,
                                            use_flash_attention=True,
                                            iso_conv_down=(False, True, True, True, None),
                                            iso_conv_up=(True, True, True, False, None),
                                            num_res_blocks=2,)
        
        model.linear = nn.Linear(256, args.num_class)
        
        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/DOWNSTREAM/WEIGHTS/3_DDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                new_ckpt = remove_ema_prefix(ckpt['ema'])
                model.load_state_dict(new_ckpt, strict=False)
                model.requires_grad_(False)
                model.linear.requires_grad_(True)

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = '/workspace/PD_SSL_ZOO/DOWNSTREAM/WEIGHTS/3_DDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                new_ckpt = remove_ema_prefix(ckpt['ema'])
                model.load_state_dict(new_ckpt, strict=False)

            elif args.linear_mode == 'scratch':
                print("scratch")
        
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'

                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'scratch':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'])


    elif args.name == 'P2S2P':
        from PREVIOUS.PSP.networks import GradualStyleEncoder_3D

        model = GradualStyleEncoder_3D()

        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/4_P2S2P.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                checkpoint = ckpt['state_dict']
                model.load_state_dict(checkpoint, strict=False)
                # model.load_state_dict(checkpoint, strict=False)
                model.requires_grad_(False)
                model.linear = nn.Linear(512*12, args.num_class)
                model.linear.requires_grad_(True)

            elif args.linear_mode == 'fine_tuning': #from up-stream
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/4_P2S2P.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                checkpoint = ckpt['state_dict']
                # model.load_state_dict(checkpoint, strict=False)
                model.load_state_dict(checkpoint, strict=False)
                model.linear = nn.Linear(512*12, args.num_class)

            elif args.linear_mode == 'scratch':
                model.linear = nn.Linear(512*12, args.num_class)
                print("scratch")
        
        # elif args.test == 1:
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.linear = nn.Linear(512*12, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'

                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.linear = nn.Linear(512*12, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'scratch':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.linear = nn.Linear(512*12, args.num_class)
                model.load_state_dict(ckpt['state_dict'])
                
    elif args.name == "DAE":
        from PREVIOUS.DAE import SwinTransformerForSimMIM_fine_tune, SimMIMSkip
        
        encoder = SwinTransformerForSimMIM_fine_tune(
            num_classes=0,
            img_size=192,
            patch_size=(2, 2, 2),
            in_chans=1,
            embed_dim=48,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=True,
        )
        encoder_stride = 32
        
        model = SimMIMSkip(
            encoder=encoder,
            encoder_stride=encoder_stride,
            loss="all_img",
            img_size=(192, 192, 96),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            choice='all',
            temperature=0.07,
        )
        encoder_stride = 32

        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/5_DisAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                model.requires_grad_(False)
                model.encoder.head.requires_grad_(True)
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/5_DisAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)

            elif args.linear_mode == 'scratch':
                model.encoder.head = nn.Linear(768, args.num_class)
                model = model.encoder
        
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                model.encoder.head = nn.Linear(768, args.num_class)
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.encoder
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'scratch':
                model.encoder.head = nn.Linear(768, args.num_class)
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

    elif args.name == "HDAE":
        from PREVIOUS.HDAE.generative.networks.nets.diffusion_model_aniso_unet_AE_no_wavelet import DiffusionModelUNet_aniso_AE_no_wavelet, DiffusionModelEncoder_ansio_no_wavelet
        class HDAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet =  DiffusionModelUNet_aniso_AE_no_wavelet(spatial_dims=3,
                                                                    in_channels=1, 
                                                                    out_channels=1, 
                                                                    num_channels=[8,32,64,128,256,512],
                                                                    attention_levels=[False,False,False,False,True,True],
                                                                    num_head_channels=[0,0,0,0,16,32],
                                                                    norm_num_groups=8,
                                                                    use_flash_attention=True,
                                                                    iso_conv_down=(False, True, True, True, True, None),
                                                                    iso_conv_up=(True, True, True, True, False, None),
                                                                    num_res_blocks=2,)

                self.semantic_encoder = DiffusionModelEncoder_ansio_no_wavelet(spatial_dims=3,
                                                                                in_channels=1,
                                                                                out_channels=1,
                                                                                num_channels=[16,32,128,256,512],
                                                                                attention_levels=(False,False,False,False,False),
                                                                                num_head_channels=[0,0,0,0,0],
                                                                                norm_num_groups=16,
                                                                                iso_conv_down=(False, True, True, True, True),
                                                                                resblock_updown=False,
                                                                                num_res_blocks=(2,2,2,2,2))
                                                            
        model = HDAE()

        def filter_ema_keys(checkpoint):
            ema_model_state_dict = {key.replace('ema_model.', ''): value 
                                    for key, value in checkpoint.items() 
                                    if 'online_model' not in key}
            del ema_model_state_dict['initted']
            del ema_model_state_dict['step']
            
            return ema_model_state_dict
        
        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/6_HDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                checkpoint = ckpt['ema']
                new_ckpt = filter_ema_keys(checkpoint)
                model.load_state_dict(new_ckpt, strict=False)
                model = model.semantic_encoder
                model.requires_grad_(False)
                model.linear = nn.Linear(512*6, args.num_class)
                model.linear.requires_grad_(True)

            elif args.linear_mode == 'fine_tuning': #from linear_probing
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/6_HDAE.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                checkpoint = ckpt['ema']
                new_ckpt = filter_ema_keys(checkpoint)
                model.load_state_dict(new_ckpt, strict=False)
                model = model.semantic_encoder
                model.linear = nn.Linear(512*6, args.num_class)
                
            elif args.linear_mode == 'scratch':
                model = model.semantic_encoder
                model.linear = nn.Linear(512*6, args.num_class)
                print("scratch")
        
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.semantic_encoder
                model.linear = nn.Linear(512*6, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.semantic_encoder
                model.linear = nn.Linear(512*6, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'scratch':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.semantic_encoder
                model.linear = nn.Linear(512*6, args.num_class)
                model.load_state_dict(ckpt['state_dict'])


        
    elif args.name == "SIMMIM":
        from PREVIOUS.SIMMIM.simmim import SwinTransformerForSimMIM_fine_tune, SimMIM
        
        encoder_fine_tuning = SwinTransformerForSimMIM_fine_tune(
            num_classes=1,
            img_size=192,
            patch_size=(2, 2, 2),
            in_chans=1,
            embed_dim=48,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=(7, 7, 7),
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=True,
        )
        encoder_stride = 32
        model = SimMIM(encoder=encoder_fine_tuning, encoder_stride=encoder_stride)

        if args.test == 0:
            if args.linear_mode == 'linear':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/7_SimMIM.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                model.requires_grad_(False)
                model.encoder.head.requires_grad_(True)
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = '/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/7_SimMIM.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model.load_state_dict(ckpt['model'])
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)

            elif args.linear_mode == 'scratch':
                model.encoder.head = nn.Linear(768, args.num_class)
                model = model.encoder
        
        else:
            if args.linear_mode == 'linear':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)
                model.load_state_dict(ckpt['state_dict'])

            elif args.linear_mode == 'fine_tuning':
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)
                model.load_state_dict(ckpt['state_dict'])
                
            elif args.linear_mode == 'scratch':
                model.encoder.head = nn.Linear(768, args.num_class)
                ckpt_path = f'/workspace/PD_SSL_ZOO/2_DOWNSTREAM/{args.down_type}/results/{args.data_per}/{args.name}/{args.name}_output_{args.fold}_{args.data_per}_{args.linear_mode}/model_best_loss.pt'
                print(f"ckpt_path : {ckpt_path}")
                ckpt = torch.load(ckpt_path, map_location='cpu')
                model = model.encoder
                model.head = nn.Linear(768, args.num_class)
                model.load_state_dict(ckpt['state_dict'])
                
    return model
