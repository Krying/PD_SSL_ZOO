from monai import data, transforms
import glob
import numpy as np
import os
import re
import natsort
import SimpleITK as sitk
import math

def get_loader(args):
    train_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[:]
    
    print("Train [Total]  number = ", len(train_real))

    files_tr = [img_tr for img_tr in zip(train_real)]
    
    tr_transforms = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="LPS"),
            transforms.RandAffine(translate_range=(5,0,5), padding_mode='zeros', prob=0.1),
            transforms.RandZoom(min_zoom=0.9, max_zoom=1.2, padding_mode='minimum', prob=0.1),
            transforms.RandRotate(range_x=math.radians(15), range_y=math.radians(15), range_z=math.radians(15), prob=0.1),
            transforms.EnsureType(),
            transforms.ToTensor(track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    train_ds = data.Dataset(data = files_tr, transform = tr_transforms)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False
    )
    print("loader is ver (train)")

    loader = train_loader

    return loader
