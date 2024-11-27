
from monai import data, transforms
import glob
import numpy as np
import os
import re
import natsort
import SimpleITK as sitk
import math

def get_loader(args):
    train_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[:] #ALL -> 2125 or 2130
    valid_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[-100:] #ALL -> 2125 or 2130
    
    files_tr = [{"image": tr_img} for tr_img in zip(train_real)]
    files_val = [{"image": val_img} for val_img in zip(valid_real)]

    train_transforms = transforms.Compose(
            [
                transforms.LoadImage(image_only=True),
                transforms.EnsureChannelFirst(),
                transforms.Orientation(axcodes="RAI"),
                # transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=22.0, b_min=0.0, b_max=1.0, clip=True),
                transforms.RandAffine(translate_range=(5,0,5), padding_mode='zeros', prob=0.1),
                transforms.RandZoom(min_zoom=0.9, max_zoom=1.2, padding_mode='minimum', prob=0.1),
                transforms.RandRotate(range_x=math.radians(15), range_y=math.radians(15), range_z=math.radians(15), prob=0.1),
                transforms.EnsureType(),
                transforms.ToTensor(track_meta=False)
            ]
        )
            
    val_transforms = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="RAI"),
            transforms.EnsureType(),
            transforms.ToTensor(track_meta=False)
        ]
    )

    dataset_train = data.Dataset(data = files_tr, transform = train_transforms)
    dataset_val = data.Dataset(data = files_val, transform = val_transforms)

    dataloader_train = data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_val = data.DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader_train, dataloader_val