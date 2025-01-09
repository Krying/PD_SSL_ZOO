import math
import json
from monai import data, transforms

def get_loader_reg(args):

    train_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS03_onset_train_fold{args.fold}.json"
    test_03_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS03_onset_test.json"
    test_04_list = f"/workspace/PD_SSL_ZOO/2_DOWNSTREAM/JSON/{args.down_type}/PS04_onset_test.json"
    
    with open(train_list, 'r') as train_file:
        train_files = json.load(train_file)

    with open(test_03_list, 'r') as test_03_file:
        test_03_files = json.load(test_03_file)

    with open(test_04_list, 'r') as test_04_file:
        test_04_files= json.load(test_04_file)

    train_idx = 0
    valid_idx = 0
    test_03_idx = 0
    test_04_idx = 0
    
    files_tr = []
    files_val = []
    files_ts_03 = []
    files_ts_04 = []
    
    for file_name, label in train_files['train'].items():
        files_tr.append({"image_train": file_name, "label_train": label})
        train_idx += 1
                
    for file_name, label in train_files['valid'].items():
        files_val.append({"image_val": file_name, "label_val": label})
        valid_idx += 1

    for file_name, label in test_03_files['test'].items():
        files_ts_03.append({"image_ts": file_name, "label_ts": label})
        test_03_idx += 1

    for file_name, label in test_04_files['test_04'].items():
        files_ts_04.append({"image_ts": file_name, "label_ts": label})
        test_04_idx += 1

    print("Train [Total]  number = ", train_idx)
    print("Valid [Total]  number = ", valid_idx)
    print("Test_03 [Total]  number = ", test_03_idx)
    print("Test_04 [Total]  number = ", test_04_idx)

    tr_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image_train"]),
            transforms.EnsureChannelFirstd(keys=["image_train"]),
            transforms.Orientationd(keys=["image_train"], axcodes="LPS"),
            transforms.RandAffined(keys=["image_train"], translate_range=(5,0,5), padding_mode='zeros', prob=0.1),
            transforms.RandZoomd(keys=["image_train"], min_zoom=0.9, max_zoom=1.2, padding_mode='minimum', prob=0.1),
            transforms.RandRotated(keys=["image_train"], range_x=math.radians(15), range_y=math.radians(15), range_z=math.radians(15), prob=0.1),
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

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    ts_03_loader = data.DataLoader(
        ts_03_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    ts_04_loader = data.DataLoader(
        ts_04_ds,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("loader is ver(train, val)")

    return [train_loader, val_loader, ts_03_loader, ts_04_loader]