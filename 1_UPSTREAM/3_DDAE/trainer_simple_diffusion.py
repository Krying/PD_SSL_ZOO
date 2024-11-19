import math
from functools import partial, wraps

import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.special import expm1
from torch.cuda.amp import autocast
import SimpleITK as sitk

from tqdm import tqdm
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from accelerate import Accelerator
from ema_pytorch import EMA
from pathlib import Path
import os

# helpers

def divisible_by(numer, denom):
    return (numer % denom) == 0

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def exists(val):
    return val is not None

def identity(t):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

def l2norm(t):
    return F.normalize(t, dim = -1)

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 1,
        pred_objective = 'v',
        noise_schedule = logsnr_schedule_cosine,
        noise_d = None,
        noise_d_low = None,
        noise_d_high = None,
        num_sample_steps = 1000,
        clip_sample_denoised = True,
        min_snr_loss_weight = True,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert pred_objective in {'v', 'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # training objective

        self.pred_objective = pred_objective

        # noise schedule

        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # determine shifting or interpolated schedules

        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred = self.model(x, batch_log_snr)

        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred

        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # sampling related functions

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 1):
        return self.p_sample_loop((1, 1, 192, 192, 96))

    # training related functions - noise prediction

    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    def p_losses(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr)

        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start

        elif self.pred_objective == 'eps':
            target = noise

        loss = F.mse_loss(model_out, target, reduction = 'none')

        loss = reduce(loss, 'b ... -> b', 'mean')

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max = self.min_snr_gamma)

        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)

        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr

        return (loss * loss_weight).mean()

    def forward(self, img, *args, **kwargs):
        # b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        img = normalize_to_neg_one_to_one(img)
        times = torch.zeros((img.shape[0],), device = self.device).float().uniform_(0, 1)

        return self.p_losses(img, times, *args, **kwargs)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        loader,
        args
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = args.split_batches,
            mixed_precision = 'fp16' if args.amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        # is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters

        self.save_and_sample_every = args.save_and_sample_every

        self.batch_size = args.batch_size
        self.gradient_accumulate_every = args.gradient_accumulate_every
        # assert (args.train_batch_size * args.gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = args.train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = args.max_grad_norm

        dl = loader

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        self.opt = torch.optim.AdamW(diffusion_model.parameters(), lr = args.optim_lr)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, 
                           beta = 0.99, 
                           update_after_step=15000,
                           update_every = 10)
            self.ema.to(self.device)

        self.results_folder = Path(args.log_dir)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)


    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            # 'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
            
    def load_for_generation(self, num):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load('/workspace/PD_SSL_ZOO/2_DOWNSTREAM/WEIGHTS/3_DDAE.pt', map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        data['ema']['step'] = torch.Tensor(1)
        data['ema']['initted'] = torch.Tensor(1)
        self.ema.load_state_dict(data["ema"], strict=False)

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self, args):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            image_pred = self.ema.ema_model.sample(batch_size=1)
                            pred_img = image_pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
                            save_pred = sitk.GetImageFromArray(pred_img)
                            sitk.WriteImage(save_pred, f"{args.log_dir}/{milestone}_pred.nii.gz")
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')

    def generation(self, num_iter, num):
        self.ema.ema_model.eval()

        for i in range(num_iter):
            os.makedirs(f"/workspace/DIF_DDAE_PET/infer_samples_{num}", exist_ok=True)
            with torch.inference_mode():
                image_pred = self.ema.ema_model.sample(batch_size=1)
                image_pred.clamp_(0.,1.)
                pred_img = image_pred.cpu().detach().numpy().transpose(0,4,3,2,1).squeeze()
                save_pred = sitk.GetImageFromArray(pred_img)
                sitk.WriteImage(save_pred, f"/workspace/DIF_DDAE_PET/infer_samples_{num}/{i}.nii.gz")