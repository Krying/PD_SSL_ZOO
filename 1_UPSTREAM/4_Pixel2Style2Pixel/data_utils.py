
from monai import data, transforms
import glob
import numpy as np
import os
import re
import natsort
import SimpleITK as sitk

def get_loader(args):
    train_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[:] #ALL -> 2125 or 2130
    valid_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[-100:] #ALL -> 2125 or 2130
    
    files_tr = [{"image": tr_img} for tr_img in zip(train_real)]
    files_val = [{"image": val_img} for val_img in zip(valid_real)]

    train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], image_only=True),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAI"),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=22.0, b_min=0.0, b_max=1.0, clip=True),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                # RandSpatialCropd(roi_size=[96, 96, 96], keys=["image"], random_size=False, random_center=True),
                transforms.ToTensord(keys=["image"], track_meta=False),
            ]
        )
            
    val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"], image_only=True),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.Orientationd(keys=["image"], axcodes="RAI"),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=22.0, b_min=0.0, b_max=1.0, clip=True),
                # SpatialPadd(keys="image", spatial_size=[96, 96, 96]),
                # RandSpatialCropd(roi_size=[96, 96, 96], keys=["image"], random_size=False, random_center=True),
                transforms.ToTensord(keys=["image"], track_meta=False),
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
'''    
    elif args.enc_version == 'restyle':
    
        files_avg = [avg_tr for avg_tr in zip(avg_pd)]
        
        avg_transforms = transforms.Compose(
            [
                transforms.LoadImage(image_only=True),
                transforms.EnsureChannelFirst(),
                transforms.Orientation(axcodes="LPS"),
                transforms.CenterSpatialCrop(roi_size=(192, 192, 96)),
                transforms.ScaleIntensityRange(a_min=0.0, a_max=20.0, b_min=0.0, b_max=2.0, clip=True),
                # transforms.Resize(spatial_size=(args.image_size, args.image_size, args.image_size / 2)), 
                transforms.Resize(spatial_size=(96, 96, 48)),
                transforms.EnsureType(),
                transforms.ToTensor(track_meta=False)
            ]
        )
        
        files_tr = [{"train_x": tr_x, "train_y": tr_y} for tr_x, tr_y in zip(train_real, train_real)]            

        tr_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["train_x", "train_y"], image_only=True),
                transforms.EnsureChannelFirstd(keys=["train_x", "train_y"]),
                transforms.Lambdad(keys=["train_x"],func=Fit_Into_Template),
                transforms.Lambdad(keys=["train_y"],func=Fit_Into_Template),
                transforms.Orientationd(keys=["train_x", "train_y"],axcodes="LAS"),
                transforms.CenterSpatialCropd(keys=["train_x", "train_y"],roi_size=(192, 192, 96)),
                transforms.ScaleIntensityRanged(keys=["train_x", "train_y"],a_min=0.0, a_max=20.0, b_min=0.0, b_max=2.0, clip=True),
                # transforms.Resize(spatial_size=(args.image_size, args.image_size, args.image_size / 2)), 
                transforms.Resized(keys=["train_x", "train_y"], spatial_size=(96, 96, 48)),
                transforms.EnsureTyped(keys=["train_x", "train_y"]),
                transforms.ToTensord(keys=["train_x", "train_y"],track_meta=False)
            ]
        )

        files_val = [{"valid_x": val_x, "valid_y": val_y} for val_x, val_y in zip(valid_real, valid_real)]            

        val_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["valid_x", "valid_y"],image_only=True),
                transforms.EnsureChannelFirstd(keys=["valid_x", "valid_y"],),
                transforms.Lambdad(keys=["valid_x"],func=Fit_Into_Template),
                transforms.Lambdad(keys=["valid_y"],func=Fit_Into_Template),
                transforms.Orientationd(keys=["valid_x", "valid_y"],axcodes="LAS"),
                transforms.CenterSpatialCropd(keys=["valid_x", "valid_y"],roi_size=(192, 192, 96)),
                transforms.ScaleIntensityRanged(keys=["valid_x", "valid_y"],a_min=0.0, a_max=20.0, b_min=0.0, b_max=2.0, clip=True),
                # transforms.Resize(spatial_size=(args.image_size, args.image_size, args.image_size / 2)), 
                transforms.Resized(keys=["valid_x", "valid_y"],spatial_size=(96, 96, 48)), 
                transforms.EnsureTyped(keys=["valid_x", "valid_y"],),
                transforms.ToTensord(keys=["valid_x", "valid_y"],track_meta=False)
            ]
        )

        files_ts = [{"test_x": ts_x, "test_y": ts_y} for ts_x, ts_y in zip(test_real, test_real)]            

        #test phase need to label dict
        ts_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["test_x","test_y"], image_only=True),
                transforms.EnsureChannelFirstd(keys=["test_x","test_y"]),
                transforms.Lambdad(keys=["test_x"], func=Fit_Into_Template),
                transforms.Lambdad(keys=["test_y"], func=Fit_Into_Template),
                transforms.Orientationd(keys=["test_x","test_y"], axcodes="LAS"),
                transforms.CenterSpatialCropd(keys=["test_x","test_y"], roi_size=(192, 192, 96)),
                transforms.ScaleIntensityRanged(keys=["test_x","test_y"], a_min=0.0, a_max=20.0, b_min=0.0, b_max=2.0, clip=True),
                transforms.Resized(keys=["test_x","test_y"], spatial_size=(96, 96, 48)), 
                transforms.EnsureTyped(keys=["test_x","test_y"]),
                transforms.ToTensord(keys=["test_x","test_y"], track_meta=False)
            ]
        )

        # new_dataset -> Cachenew_dataset
        train_ds = data.CacheDataset(data = files_tr, transform = tr_transforms, cache_rate = 1.0, num_workers = 32)
        val_ds = data.CacheDataset(data = files_val, transform = val_transforms, cache_rate = 1.0, num_workers = 4)
        test_ds = data.CacheDataset(data = files_ts, transform = ts_transforms, cache_rate = 1.0, num_workers = 4)

        avg_ds = data.CacheDataset(data = files_avg, transform = avg_transforms, cache_rate = 1.0, num_workers = 1)

        test_mode = args.tsne
        
        avg_loader = data.DataLoader(
            avg_ds,
            batch_size=1,
            shuffle=False,
            num_workers=16,
            pin_memory=False
            # persistent_workers=True,
        )

        if test_mode == True:
            test_loader = data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                pin_memory=True,
                persistent_workers=True,
            )
            loader = [test_loader, test_real]
            print("loader is ver (tsne)")

        else:
            train_loader = data.DataLoader(
                train_ds,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                pin_memory=False
                # persistent_workers=True,
            )

            val_loader = data.DataLoader(
                val_ds,
                batch_size=1,
                shuffle=False,
                num_workers=16,
                pin_memory=False
                # persistent_workers=True,
            )
            print("loader is ver (train)")

            loader = [train_loader, val_loader]

        return loader, train_real, avg_loader

    elif args.enc_version == 'hyper':
            pass
        '''