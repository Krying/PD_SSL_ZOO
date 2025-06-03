<h1 align="center"><b> PD_SSL_ZOO</b></h1>

This is the codebase 
for the paper "Enhancing 3D Dopamine Transporter Imaging as a Biomarker for Parkinson's Disease via Self-Supervised Learning with Diffusion Models".

<p align="center"><img width=75% alt="FrontCover" src="./assets/Overview.png"></p>

## Publication
<b>Enhancing 3D Dopamine Transporter Imaging as a Biomarker for Parkinson's Disease via Self-Supervised Learning with Diffusion Models </b> <br/>

Jongjun Won<sup>1</sup>, Grace Yoojin Lee<sup>1</sup>, Sungyang Jo<sup>1</sup>, Jihyun Lee<sup>1</sup>, Sangjin Lee<sup>1</sup>, Jae Seung Kim<sup>1</sup>, Changhwan Sung<sup>1</sup>, Jungsu S. Oh<sup>1</sup>, Kyum-Yil Kwon<sup>2</sup>, Soo Bin Park<sup>2</sup>, Joonsang Lee<sup>1</sup>, Jieun Yum<sup>1</sup>, Sun Ju Chung<sup>1</sup>, and Namkug Kim<sup>1</sup><br/>

<sup>1 </sup>Asan Medical Center, <sup>2 </sup>Soonchunhyang University Seoul Hospital<br/>
<b>*Cell Reports Medicine* (Acceptance, to appear in 2025)</b>


## Contents

This repository is composed of

1_UPSTREAM   
2_DOWNSTREAM  
3_RECONSTRUCTION   
4_LATENT_MANIPULATION   

Our overall workflow code parts are mainly in "1_UPSTREAM" and "2_DOWNSTREAM," illustrated below: 


<br/>

This repository is based on other repositories of MONAI, lucidrains, and eladrich.

[Monai Generative Models](https://github.com/Project-MONAI/GenerativeModels/tree/main/generative) : HWDAE, WDDAE, DDAE, HDAE

[Monai/research-contribution (DisAE)](https://github.com/Project-MONAI/research-contributions/tree/main/DAE) : DisAE, SimMIM

[lucidrains/StyleGAN2-pytorch](https://github.com/lucidrains/stylegan2-pytorch) : StyleGAN2

[lucidrains/denoising_diffusion_pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch) : WDDAE, DDAE

[eladrich/Pixel2Style2Pixel](https://github.com/eladrich/pixel2style2pixel) : P2S2P
<br/>

## requirements
```bash
pip install -r requirements.txt
```
<br/>

## Pretrained upstream model weights & synthetic samlpe scans
[One Drive/Weight](https://liveuou-my.sharepoint.com/:f:/g/personal/krying_mail_ulsan_ac_kr/Ek8Gv600i5ZDiZi1b2wg_awBaipjjVZeqdWf5DRLhOHu9w?e=wdIlM3)

[One Drive/Synthetic FP-CIT-PET Samples](https://liveuou-my.sharepoint.com/:f:/g/personal/krying_mail_ulsan_ac_kr/EqKDcxeuwQNLr4UsUI15KeEBe5CgYPQyaCqI6615jhev5A?e=GvAPjr)

<br/>

# Train & Test

There are directories for each upstream model and downstream task. 

Models : 1_HWDAE, 2_WDDAE, 3_DDAE, 4_P2S2P, 5_DisAE, 6_HDAE, 7_SimMIM

Tasks : 1_EP, 2_PMP, 3_SOY 
<br/>

### Upstream: 
For the pre-training stage of SSL models. 

-> Please refer the "main.py" in the each folders of "/1_UPSTREAM/Models/."

```bash
python main.py --batch_size 2 --log_dir <log_dir>
```
<br/>

### Downstream: 
For the linear probing, training from scratch, or fine-tuning stages of downstream tasks from the upstream models.

-> Please refer the "main.py" in the each folders of "/2_DOWNSTREAM/Tasks/."

```bash
python main.py --batch_size <batch_size> --name <model_name> --log_dir <log_dir> --data_per <data_percentage> --linear_mode <linear | scratch | fine_tuning> 
```
<br/>

### Generation, Reconstruction and Latent Manipulation:

Unconditional image generation of "Models" in [WDDAE, DDAE, StyleGAN2]. 

-> Please refer the "generate.py" in the each folders of "/3_RECONSTRUCTION/0_GENERATION/Models/."

```bash
python generate.py
```
<br/>

Image Reconstruction of "Models" in [HWDAE, HDAE, P2S2P, DisAE, SimMIM]

-> Please refer the "RECONSTRUCTION.ipynb" in the each folders of "/3_RECONSTRUCTION/Models/."

<br/>

Latent Manipulation of HWDAE.   

-> Please refer the "HWDAE_MANIPULATION.ipynb" in the folder of "/4_HWDAE_MANIPULATION/."
