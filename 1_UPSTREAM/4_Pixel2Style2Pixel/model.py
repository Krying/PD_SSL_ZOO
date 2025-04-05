import torch 
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/UPSTREAM/4_Pixel2Style2Pixel'))

from PREVIOUS.decoder import StyleGAN2
from PREVIOUS.pSp_simple import GradualStyleEncoder_3D
# Reference:

#title: Analyzing and Improving the Image Quality of StyleGAN
#url: https://github.com/lucidrains/stylegan2-pytorch
#source code: https://github.com/Project-MONAI/GenerativeModels/tree/main/generative

#title: Encoding in Style: a StyleGAN Encoder for Image-to-Image Translation
#url: https://arxiv.org/abs/2008.00951
#source code: https://github.com/eladrich/pixel2style2pixel

def create_model(enc_version, args):
    encoder = GradualStyleEncoder_3D()

    ckpt_dec = torch.load('/workspace/Ablation/ABLATION_PD/GAN_INV_PSP/model_60.pt', map_location='cpu')
    dec_model = StyleGAN2(image_size=args.image_size)
    dec_model.load_state_dict(ckpt_dec["GAN"])

    mapping = dec_model.S
    
    return encoder, dec_model, mapping
