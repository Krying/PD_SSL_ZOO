import glob
import math
import natsort
from monai import data, transforms

def get_loader(args):
    train_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[:] #ALL -> 2125 or 2130
    valid_real = natsort.natsorted(glob.glob(f'/workspace/dataset/PET/*.nii.gz'))[-100:] #ALL -> 2125 or 2130

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

    files_val = [img_val for img_val in zip(valid_real)]

    val_transforms = transforms.Compose(
        [
            transforms.LoadImage(image_only=True),
            transforms.EnsureChannelFirst(),
            transforms.Orientation(axcodes="LPS"),
            transforms.EnsureType(),
            transforms.ToTensor(track_meta=False)
        ]
    )

    # new_dataset -> Cachenew_dataset
    train_ds = data.Dataset(data = files_tr, transform = tr_transforms)
    val_ds = data.Dataset(data = files_val, transform = val_transforms)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False
        # persistent_workers=True,
    )

    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=False
        # persistent_workers=True,
    )
    print("loader is ver (train)")

    loader = [train_loader, val_loader]

    return loader
