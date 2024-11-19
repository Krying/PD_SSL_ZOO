
import torch

def create_model(args): 
    if args.model == 'HWDAE':
        from generative.networks.nets.diffusion_model_aniso_unet_AE_official import DiffusionModelUNet_aniso_AE, DiffusionModelEncoder_ansio
        class HWDAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unet =  DiffusionModelUNet_aniso_AE(spatial_dims=3,
                                                        in_channels=8, #wavelet
                                                        out_channels=8, #wavelet
                                                        num_channels=[128,128,256,256,512],
                                                        attention_levels=[False,False,False,True,True],
                                                        num_head_channels=[0,0,0,32,32],
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

    return model
