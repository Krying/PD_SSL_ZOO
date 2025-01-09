import pandas as pd
from monai import data, transforms
import natsort
import glob
import numpy as np
import torch
import os
from collections import defaultdict
#Early PD / Normal
import pandas as pd
import math
import json

def get_loader(args):

    train_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS03_clf_ET_Early_PD_train_revised_fold{args.fold}.json"
    test_03_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS03_clf_ET_Early_PD_test_revised.json"
    test_04_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS04_clf_ET_Early_PD_test_revised.json"
    test_SCH_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/SCH_clf_ET_Early_PD_test.json"
    num_class = 2
        
    with open(train_list, 'r') as train_file:
        train_files = json.load(train_file)

    with open(test_03_list, 'r') as test_03_file:
        test_03_files = json.load(test_03_file)

    with open(test_04_list, 'r') as test_04_file:
        test_04_files= json.load(test_04_file)
        
    with open(test_SCH_list, 'r') as test_SCH_file:
        test_SCH_files= json.load(test_SCH_file)
        
    train_idx = 0
    valid_idx = 0
    test_03_idx = 0
    test_04_idx = 0
    test_SCH_idx = 0

    files_tr = []
    files_val = []
    files_ts_03 = []
    files_ts_04 = []
    files_ts_SCH = []

    for file_name, label in train_files['train'].items():
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=num_class)
        files_tr.append({"image_train": file_name, "label_train": label})
        train_idx += 1
                
    for file_name, label in train_files['valid'].items():
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=num_class)
        files_val.append({"image_val": file_name, "label_val": label})
        valid_idx += 1

    for file_name, label in test_03_files['test'].items():
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=num_class)
        files_ts_03.append({"image_ts": file_name, "label_ts": label})
        test_03_idx += 1

    for file_name, label in test_04_files['test_04'].items():
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=num_class)
        files_ts_04.append({"image_ts": file_name, "label_ts": label})
        test_04_idx += 1

    for file_name, label in test_SCH_files['test'].items():
        label = torch.nn.functional.one_hot(torch.as_tensor(label), num_classes=num_class)
        files_ts_SCH.append({"image_ts": file_name, "label_ts": label})
        test_SCH_idx += 1

    print("Train [Total]  number = ", train_idx)
    print("Valid [Total]  number = ", valid_idx)
    print("Test_03 [Total]  number = ", test_03_idx)
    print("Test_04 [Total]  number = ", test_04_idx)
    print("Test_SCH [Total]  number = ", test_SCH_idx)

    tr_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_train"]),
            transforms.EnsureChannelFirstd(keys=["image_train"]),
            transforms.Orientationd(keys=["image_train"], axcodes="LPS"),
            transforms.RandAffined(keys=["image_train"], translate_range=(5,0,5), padding_mode='zeros', prob=0.1),
            transforms.RandZoomd(keys=["image_train"], min_zoom=0.9, max_zoom=1.2, padding_mode='minimum', prob=0.1),
            transforms.RandRotated(keys=["image_train"], range_x=math.radians(15), range_y=math.radians(15), range_z=math.radians(15), prob=0.1),
            transforms.RandGaussianNoised(keys=["image_train"], mean=0.0, std=0.1, prob=0.2), #
            transforms.EnsureTyped(keys=["image_train", "label_train"]),
            transforms.ToTensord(keys=["image_train", "label_train"], track_meta=False)
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_val"]),
            transforms.EnsureChannelFirstd(keys=["image_val"]),
            transforms.Orientationd(keys=["image_val"], axcodes="LPS"),
            transforms.EnsureTyped(keys=["image_val", "label_val"]),
            transforms.ToTensord(keys=["image_val", "label_val"], track_meta=False)
        ]
    )

    ts_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_ts"]),
            transforms.EnsureChannelFirstd(keys=["image_ts"]),
            transforms.Orientationd(keys=["image_ts"], axcodes="LPS"),
            transforms.ScaleIntensityRanged(keys=["image_ts"], a_min=0.0, a_max=22.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.EnsureTyped(keys=["image_ts", "label_ts"]),
            transforms.ToTensord(keys=["image_ts", "label_ts"], track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    train_ds = data.Dataset(data = files_tr, transform = tr_transforms)
    val_ds = data.Dataset(data = files_val, transform = val_transforms)
    ts_03_ds = data.Dataset(data = files_ts_03, transform = ts_transforms)
    ts_04_ds = data.Dataset(data = files_ts_04, transform = ts_transforms)
    ts_SCH_ds = data.Dataset(data = files_ts_SCH, transform = ts_transforms)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    ts_03_loader = data.DataLoader(
        ts_03_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    ts_04_loader = data.DataLoader(
        ts_04_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )
    
    ts_SCH_loader = data.DataLoader(
        ts_SCH_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )
    print("loader is ver(train, val)")

    return [train_loader, val_loader, ts_03_loader, ts_04_loader, ts_SCH_loader]