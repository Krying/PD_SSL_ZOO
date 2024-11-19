import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetResBlock, get_conv_layer

from .swin_transformer_3d import SwinTransformer3D


class PixelShuffle3D(nn.Module):
    """
    https://github.com/assassint2017/PixelShuffle3D/blob/master/PixelShuffle3D.py
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        batch_size, channels, in_depth, in_height, in_width = inputs.size()
        channels //= self.upscale_factor**3
        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor
        input_view = inputs.contiguous().view(
            batch_size,
            channels,
            self.upscale_factor,
            self.upscale_factor,
            self.upscale_factor,
            in_depth,
            in_height,
            in_width,
        )
        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


class SwinTransformerForSimMIM(SwinTransformer3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([self.layers1, self.layers2, self.layers3, self.layers4])

    def forward(self, x, mask):
        _, _, D, H, W = x.size()
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        assert mask is not None
        B, L, _ = x.shape
        mask_tokens = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_tokens)
        x = x * (1.0 - w) + mask_tokens * w
        x = self.pos_drop(x)
        x = x.view(-1, self.embed_dim, D // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2])

        for layer in self.layers:
            x = layer[0](x)

        reduction = self.patch_size[0] * 16
        x = x.reshape(-1, (D // reduction) * (H // reduction) * (W // reduction), 2 * self.num_features)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = x.view(-1, 2 * self.num_features, D // 32, H // 32, W // 32)

        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return super().no_weight_decay() | {"mask_token"}

class SwinTransformerForSimMIM_fine_tune(SwinTransformer3D):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.mask_token, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([self.layers1, self.layers2, self.layers3, self.layers4])

    def forward(self, x):
        _, _, D, H, W = x.size()
        x = self.patch_embed(x)
        x0 = self.pos_drop(x)
        x1 = self.layers1[0](x0.contiguous())
        x2 = self.layers2[0](x1.contiguous())
        x3 = self.layers3[0](x2.contiguous())
        x4 = self.layers4[0](x3.contiguous())
        
        x = x4.reshape(-1, (D // 32) * (H // 32) * (W // 32), 2 * self.num_features)
        x = self.norm(x)
        x_cls = self.avgpool(x_cls.transpose(1, 2))  # B C 1
        x_cls = torch.flatten(x_cls, 1)
        x_cls = self.head(x_cls)
        return x_cls

class SimMIM(nn.Module):
    def __init__(self, encoder, encoder_stride, decoder="pixel_shuffle", loss="mask_only"):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride
        self.decoder = decoder
        self.loss = loss

        self.conv1 = nn.Conv3d(
            in_channels=2 * self.encoder.num_features, out_channels=self.encoder_stride**3 * 1, kernel_size=1
        )
        self.pixel_shuffle = PixelShuffle3D(self.encoder_stride)

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.pixel_shuffle(self.conv1(z))

        mask = (
            mask.repeat_interleave(self.patch_size[0], 1)
            .repeat_interleave(self.patch_size[1], 2)
            .repeat_interleave(self.patch_size[2], 3)
            .unsqueeze(1)
            .contiguous()
        )
        loss_recon = F.l1_loss(x, x_rec, reduction="none")
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans

        return loss, x_rec, mask

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, "no_weight_decay"):
            return {"encoder." + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, "no_weight_decay_keywords"):
            return {"encoder." + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_simmim(args):
    model_type = args.model_type
    if model_type == "swin":
        encoder = SwinTransformerForSimMIM(
            num_classes=1,#fine tuning시 바꿀 것
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
            drop_path_rate=args.drop_path_rate,
            # use_checkpoint=args.use_grad_checkpoint,
            patch_norm=True,
        )
        encoder_stride = 32
        model = SimMIM(encoder=encoder, encoder_stride=encoder_stride, decoder=args.decoder, loss=args.loss_type)

    return model
