from monai import data, transforms
import glob
import numpy as np
import os
import re
import natsort

def get_loader(batch_size, image_size, max_clamp, len_dataset):
    train_real = natsort.natsorted(glob.glob('/workspace/GAN_inversion_dataset_192/*.nii.gz'))
    print("Train [Total]  number = ", len(train_real))

    files_tr = [img_tr for img_tr in zip(train_real)]

    tr_transforms = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="RAI"),
            transforms.ScaleIntensityRange(a_min=0.0, a_max=22.0, b_min=0.0, b_max=1.0, clip=True),#fit96
            # transforms.Resize(spatial_size=(24, 32, 32)),
            # transforms.Resize(spatial_size=(96, 96, 48)),
            # transforms.Resize(spatial_size=(48, 48, 24)), 
            transforms.EnsureType(),
            transforms.ToTensor(track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    train_ds = data.Dataset(data = files_tr, transform = tr_transforms)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
        # persistent_workers=True,
    )


    loader = train_loader

    return loader, train_real
