import os
import fire
import random
from retry.api import retry_call
from tqdm import tqdm
from datetime import datetime
from functools import wraps
import sys
import os

sys.path.append(os.path.abspath('/workspace/PD_SSL_ZOO/3_RECONSTRUCTION/0_GENERATION/4_StyleGAN2'))

from stylegan2_3D_pytorch import Trainer, NanException #UNET

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

def cast_list(el):
    return el if isinstance(el, list) else [el]

def timestamped_filename(prefix = 'generated-'):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
    return f'{prefix}{timestamp}'

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def run_training(rank, world_size, model_args, data, load_from, new, num_train_steps, name, seed):
    is_main = rank == 0
    is_ddp = world_size > 1

    if is_ddp:
        set_seed(seed)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        print(f"{rank + 1}/{world_size} process initialized.")

    model_args.update(
        is_ddp = is_ddp,
        rank = rank,
        world_size = world_size
    )

    model = Trainer(**model_args)

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data)

    progress_bar = tqdm(initial = model.steps, total = num_train_steps, mininterval=10., desc=f'{name}<{data}>')
    while model.steps < num_train_steps:
        retry_call(model.train, tries=3, exceptions=NanException)
        progress_bar.n = model.steps
        progress_bar.refresh()
        if is_main and model.steps % 100 == 0:
            model.print_log()

    model.save(model.checkpoint_num)

    if is_ddp:
        dist.destroy_process_group()

def train_from_folder(
    data = './data',
    results_dir = '/workspace/PD_SSL_ZOO/3_RECONSTRUCTION/0_GENERATION/4_StyleGAN2/synth_img',
    models_dir = './models',
    name = 'default',
    new = False,
    load_from = -1,
    image_size = 192,
    network_capacity = 8,
    fmap_max = 512,
    transparent = False,
    batch_size = 1,
    gradient_accumulate_every = 6,
    num_train_steps = 250000,
    learning_rate = 2e-5,
    lr_mlp = 0.1,
    ttur_mult = 1.5,
    rel_disc_loss = False,
    num_workers =  None,
    save_every = 1000,
    evaluate_every = 1000,
    generate = False,
    num_generate = 50,
    generate_interpolation = False,
    interpolation_num_steps = 100,
    save_frames = False,
    num_image_tiles = 1,
    trunc_psi = 0.75,
    mixed_prob = 0.9,
    fp16 = False,
    no_pl_reg = False,
    cl_reg = False,
    fq_layers = [],
    fq_dict_size = 256,
    attn_layers = [],
    no_const = False,
    aug_prob = 0.,
    aug_types = ['brightness', 'lightbrightness', 'contrast', 'lightcontrast'],
    top_k_training = True,
    generator_top_k_gamma = 0.99,
    generator_top_k_frac = 0.5,
    dual_contrast_loss = False,
    dataset_aug_prob = 0.,
    multi_gpus = False,
    seed = 42,
    log = False,
    max_clamp = 2,
    len_dataset = 1934
):
    model_args = dict(
        name = name,
        results_dir = results_dir,
        models_dir = models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        image_size = image_size,
        network_capacity = network_capacity,
        fmap_max = fmap_max,
        transparent = transparent,
        lr = learning_rate,
        lr_mlp = lr_mlp,
        ttur_mult = ttur_mult,
        rel_disc_loss = rel_disc_loss,
        num_workers = num_workers,
        save_every = save_every,
        evaluate_every = evaluate_every,
        num_image_tiles = num_image_tiles,
        trunc_psi = trunc_psi,
        fp16 = fp16,
        no_pl_reg = no_pl_reg,
        cl_reg = cl_reg,
        fq_layers = fq_layers,
        fq_dict_size = fq_dict_size,
        attn_layers = attn_layers,
        no_const = no_const,
        aug_prob = aug_prob,
        aug_types = cast_list(aug_types),
        top_k_training = top_k_training,
        generator_top_k_gamma = generator_top_k_gamma,
        generator_top_k_frac = generator_top_k_frac,
        dual_contrast_loss = dual_contrast_loss,
        dataset_aug_prob = dataset_aug_prob,
        mixed_prob = mixed_prob,
        log = log,
        max_clamp = max_clamp,
        len_dataset = len_dataset
    )

    model = Trainer(**model_args)
    model.load(load_from)
    samples_name = timestamped_filename()
    for num in tqdm(range(num_generate)):
        model.evaluate(f'{samples_name}-{num}', num_image_tiles)
    print(f'sample images generated at {results_dir}/{name}/{samples_name}')
    return


def main():
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    fire.Fire(train_from_folder)

if __name__ == '__main__':
    main()
