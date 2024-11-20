from generative.networks.nets.diffusion_model_aniso_unet import DiffusionModelUNet_aniso

def create_model(args): 
    if args.model == 'ddpm':
        model =  DiffusionModelUNet_aniso(spatial_dims=3,
                                          in_channels=1, 
                                          out_channels=1, 
                                          num_channels=[16,32,64,128,256],
                                          attention_levels=[False,False,False,False,True],
                                          num_head_channels=[0,0,0,0,32],
                                          norm_num_groups=16,
                                          use_flash_attention=True,
                                          iso_conv_down=(False, True, True, True, None),
                                          iso_conv_up=(True, True, True, False, None),
                                          num_res_blocks=2,)

    elif args.model == 'wddpm':
        model =  DiffusionModelUNet_aniso(spatial_dims=3,
                                          in_channels=8, 
                                          out_channels=8,
                                          num_channels=[128,128,256,512],
                                          attention_levels=[False,False,False,True],
                                          num_head_channels=[0,0,0,32],
                                          norm_num_groups=32,
                                          use_flash_attention=True,
                                          iso_conv_down=(False, True, True, None),
                                          iso_conv_up=(True, True, False, None),
                                          num_res_blocks=2,)
        
    return model
