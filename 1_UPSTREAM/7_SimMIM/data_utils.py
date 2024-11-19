import json
import os
import pdb

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data._utils.collate import default_collate

from monai.data import (
    Dataset,
    Dataset,
)
import math

from monai.transforms import (
    EnsureChannelFirstd,
    RandSpatialCropd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    ToTensord,
    RandAffined,
    RandZoomd,
    RandRotated
)


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)
    # pdb.set_trace()
    return tr, val


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=(2, 2, 2), mask_ratio=0.6):
        self.input_size = input_size #192
        self.mask_patch_size = mask_patch_size #32
        self.model_patch_size = model_patch_size[0] #2
        self.mask_ratio = mask_ratio #0.6
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        self.rand_size = self.input_size // self.mask_patch_size #6
        self.scale = self.mask_patch_size // self.model_patch_size #8
        self.token_count = int(self.rand_size*self.rand_size*self.rand_size/2) #6*6*3
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size, int(self.rand_size/2)))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1).repeat(self.scale, axis=2)
        return mask


class Transform:
    def __init__(self, args):
        self.transform_pet = Compose(
            [
                LoadImaged(keys=["image"], image_only=True),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="LPS"),
                RandAffined(keys=["image"], translate_range=(5,0,5), padding_mode='zeros', prob=0.1),
                RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.2, padding_mode='minimum', prob=0.1),
                RandRotated(keys=["image"], range_x=math.radians(15), range_y=math.radians(15), range_z=math.radians(15), prob=0.1),
                ToTensord(keys=["image"], track_meta=False),
            ]
        )
            
        self.mask_generator = MaskGenerator()

    def __call__(self, img):
        img = self.transform_pet(img)
        mask = self.mask_generator()
        return img, mask

def collate_fn(batch):
    if not isinstance(batch[0][0], tuple):
        return default_collate(batch)
    else:
        batch_num = len(batch)
        ret = []
        for item_idx in range(len(batch[0][0])):
            if batch[0][0][item_idx] is None:
                ret.append(None)
            else:
                ret.append(default_collate([batch[i][0][item_idx] for i in range(batch_num)]))
        ret.append(default_collate([batch[i][1] for i in range(batch_num)]))
        return ret

import natsort
import glob

def build_loader_simmim(args):
    train_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[:] #ALL -> 2125 or 2130
    valid_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[-100:] #ALL -> 2125 or 2130

    print("Train [Total]  number = ", len(train_real))

    files_tr = [{"image": tr_img} for tr_img in zip(train_real)]
    files_val = [{"image": val_img} for val_img in zip(valid_real)]

    transform = Transform(args)
    dataset_train = Dataset(data = files_tr, transform = transform)
    dataset_val = Dataset(data = files_val, transform = transform)

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return dataloader_train, dataloader_val
