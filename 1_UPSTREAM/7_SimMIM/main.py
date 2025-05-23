import argparse
import datetime
import os
import pdb
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
from logger import create_logger
from lr_scheduler import CosineAnnealingWarmUpRestarts
from data_utils import build_loader_simmim
from pathlib import Path
import json
# from config import get_config
from models import build_model
from timm.utils import AverageMeter
from utils import TensorboardLogger, auto_resume_helper, get_grad_norm, load_checkpoint, reduce_tensor, save_checkpoint

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser("SimMIM pre-training script", add_help=False)

    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--num_classes", default=0, type=int, help="number of input channels")
    parser.add_argument("--window_size", default=(7, 7, 7), type=tuple, help="window size")
    parser.add_argument("--patch_size", default=(2, 2, 2), type=tuple, help="window size")
    parser.add_argument("--mask_patch_size", default=16, type=int, help="window size")
    parser.add_argument("--img_size", default=96, type=int, help="image size")
    parser.add_argument("--num_heads", default=[3, 6, 12, 24], type=list, help="number of heads")
    parser.add_argument("--depths", default=[2, 2, 2, 2], type=list, help="number of depths")
    parser.add_argument("--embed_dim", default=48, type=int, help="embedding dimention")
    parser.add_argument("--mlp_ratio", default=4.0, type=float, help="MLP ratio")
    parser.add_argument("--drop_rate", default=0.0, type=float, help="drop rate")
    parser.add_argument("--attn_drop_rate", default=0.0, type=float, help="attention drop rate")
    parser.add_argument("--drop_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--layer_decay", default=1.0, type=float, help="layer decay")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--mask_ratio", default=0.6, type=float, help="drop path rate")

    parser.add_argument("--optimizer_name", type=str, default="adamw", help="optimizer name")
    parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
    parser.add_argument("--base_lr", default=1e-4, type=float, help="base learning rate")
    parser.add_argument("--weight_decay", default=0.05, type=float, help="weight decay")
    parser.add_argument("--betas", default=(0.9, 0.999), type=tuple, help="optimizer betas")
    parser.add_argument("--eps", default=1e-8, type=float, help="eps")
    parser.add_argument("--decoder", type=str, default="upsample", help="decoder type")
    parser.add_argument("--loss_type", type=str, default="mask_only", help="decoder type")

    parser.add_argument("--amp_opt_level", type=str, default="O1", help="amp opt level")
    parser.add_argument("--epoch", default=500, type=int, help="number of epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="number of epochs")
    parser.add_argument("--warmpup_epoch", default=20, type=int, help="warmup epoch")
    parser.add_argument("--decay_epoch", default=30, type=int, help="warmup epoch")
    parser.add_argument("--save_freq", default=1, type=int, help="saving frequency")
    parser.add_argument("--print_freq", default=900, type=int, help="print frequency")
    parser.add_argument("--accumulate_step", default=1, type=int, help="accumulation step")
    parser.add_argument("--clip_grad", default=1, type=int, help="saving frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed")

    parser.add_argument("--lr_scheduler_name", type=str, default="cosine", help="learning rate scheduler name")
    parser.add_argument("--min_lr", default=1e-5, type=float, help="min learning rate")
    parser.add_argument("--warmup_lr", default=5e-7, type=float, help="warmup lr")
    parser.add_argument("--lr_decay_rate", default=0.1, type=float, help="lr decay rate")
    parser.add_argument("--lr_gamma", default=0.1, type=float, help="lr gamma")
    parser.add_argument("--auto_resume", default=True, type=bool)
    parser.add_argument("--iso_spacing", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--model_type", type=str, default="swin", help="model type")
    parser.add_argument("--cache_dataset", default=True, action="store_true")
    parser.add_argument("--thread_loader", default=True, action="store_true")
    parser.add_argument("--onlycovid", default=False, action="store_true")

    parser.add_argument("--cache_rate", default=0.5, type=float, help="drop path rate")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for single GPU")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--log_dir", default="/workspace/PD_SSL_ZOO/UPSTREAM/7_SimMIM/output", help="path where to tensorboard log")

    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--deterministic", help="set seed for deterministic training", action="store_true")
    parser.add_argument(
        "--use_grad_checkpoint", action="store_true", help="whether to use gradient checkpointing to save memory"
    )

    parser.add_argument(
        "--output",
        default="/workspace/PD_SSL_ZOO/1_UPSTREAM/7_SimMIM/output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--decoder_off", action="store_true")
    parser.add_argument("--encoder_off", action="store_true")
    parser.add_argument(
        "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
    )

    # parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--roi_x", default=192, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=192, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    # parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    # parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--choice", default="mae", type=str, help="choice")
    parser.add_argument("--inf", default="notsim", type=str, help="choice")

    parser.add_argument("--variance", default=0.1, type=float, help="")
    parser.add_argument("--interpolate", default=4, type=float, help="")
    parser.add_argument("--temperature", default=0.07, type=float, help="drop path rate")
    parser.add_argument("--mm_con", default=0.02, type=float, help="drop path rate")

    args = parser.parse_args()

    return args


def main(args):
    data_loader_train, data_loader_val = build_loader_simmim(args)
    model = build_model(args, is_pretrain=True)
    model.cuda()
    logger.info(str(model))
    pretrained_dir = args.pretrained_dir
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.min_lr, weight_decay=0.05)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=500, T_mult=1, eta_max=args.base_lr, T_up=2)

    logger.info("Start training")
    if args.resume:
        model_dict = torch.load(os.path.join(args.pretrained_dir, "ckpt_ori.pth"))["model"]
        model.load_state_dict(model_dict)
        print("Use pretrained weights")
    start_time = time.time()
    val_loss_best = 1.

    for epoch in range(args.start_epoch, args.epoch):
        if not args.thread_loader:
            data_loader_train.sampler.set_epoch(epoch)
        train_loss_avg = train_one_epoch(args, model, data_loader_train, optimizer, epoch, lr_scheduler)
                
        log_stats = {f'train_{epoch}': train_loss_avg}
        with (Path(args.log_dir) / 'log.txt').open('a') as f:
            f.write(json.dumps(log_stats) + "\n")

        if epoch % 4 == 0:
            val_loss_avg = validate(data_loader_val, model, epoch)
            
            log_stats = {f'val_{epoch}': val_loss_avg}
            with (Path(args.log_dir) / 'log.txt').open('a') as f:
                f.write(json.dumps(log_stats) + "\n")

            if val_loss_avg <= val_loss_best:
                val_loss_best = val_loss_avg
                save_checkpoint(args, epoch, model, val_loss_avg, optimizer, lr_scheduler, logger, best_model=True)

            save_checkpoint(args, epoch, model, val_loss_avg, optimizer, lr_scheduler, logger, best_model=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

import SimpleITK as sitk
def save_image(image_pred, description):
    pred_img = image_pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
    save_pred = sitk.GetImageFromArray(pred_img)
    sitk.WriteImage(save_pred, f"{args.log_dir}/{description}_pred.nii.gz")

def train_one_epoch(args, model, data_loader, optimizer, epoch, lr_scheduler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx, (img, mask) in enumerate(data_loader):
        img = img["image"].cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)

        loss, _, _ = model(img, mask)

        loss = loss / args.accumulate_step
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        if (idx + 1) % args.accumulate_step == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        if (idx + 1) % 967 == 0:
            lr_scheduler.step()
        
        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{args.epoch}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg


@torch.no_grad()
def validate(data_loader, model, epoch):
    model.eval()
    loss_meter = AverageMeter()

    for idx, (img, mask) in enumerate(data_loader):
        img = img["image"].cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        loss, img_recon, mask_out = model(img, mask)
        loss_meter.update(loss.item(), img.size(0))

    logger.info(f" * Val Loss {loss_meter.avg:.3f}")
    save_image(img, f"img_{epoch}")
    save_image(img_recon, f"rec_{epoch}")
    save_image(mask_out.float(), f"mask_{epoch}")
    return loss_meter.avg


if __name__ == "__main__":
    args = parse_option()
    
    seed = args.seed

    if args.deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)

    cudnn.benchmark = True

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    log_writer = TensorboardLogger(log_dir=args.log_dir)

    logger = create_logger(output_dir=args.output, name=f"{args.model_type}")

    main(args)
