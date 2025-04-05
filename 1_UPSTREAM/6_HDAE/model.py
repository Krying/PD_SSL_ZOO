from generative.networks.nets.diffusion_model_aniso_unet_AE_no_wavelet import DiffusionModelUNet_aniso_AE_no_wavelet, DiffusionModelEncoder_ansio_no_wavelet
import torch

# Reference:

# title: Generative AI for Medical Imaging: extending the MONAI Framework
# url: https://arxiv.org/abs/2307.15208
# source code: https://github.com/Project-MONAI/GenerativeModels/tree/main/generative

def create_model(args): 
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
        
    return model
