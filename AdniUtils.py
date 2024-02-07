import os
import tempfile
from pathlib import Path

from datetime import datetime
from collections import namedtuple

import torch

from monai.data.dataloader import DataLoader

import torchshow
from colorama import Fore

from AdniDataset import AdniDataset

def getDevice(args):
    if args.device == 'mps':
        return torch.device("mps")
    elif args.device == 'cuda':
        return torch.device("cuda")
    else:
        return torch.device("cpu")
    
def moveToGpu(t, args):
    if args.device == 'mps':
        return t.to('mps')
    elif args.device == 'cuda':
        return t.to('cuda')
    else:
        return t.to('cpu')
    
def get_imageid_from_path(path):
    image_path = Path(path)
    image_fname = image_path.stem.split('.')[0]
    image_id = image_fname.split('_')[-1]
    return image_id

def save_model(model, model_dir_path:Path, args, fname):
    make_symlink=args.make_symlinks if args.make_symlinks is not None else False

    model_fname = f'{fname}'

    model_path = model_dir_path.joinpath(model_fname)
    torch.save(model.state_dict(), model_path)

    #make softlink in parent dir
    if make_symlink:
        parent_dir = model_dir_path.parent.absolute()
        symlink_path = parent_dir.joinpath(model_fname)

        src_rel_path = model_path.relative_to(model_path.parent.parent)
        if symlink_path.exists():
            os.remove(symlink_path)
        os.symlink(src_rel_path, symlink_path)

def display_3dvol_slices(vol_cdhw, slice_fractions, figsize=(4,8)):
    display_slices = []
    volShape = vol_cdhw.shape
    for sf in slice_fractions:
        sliceIds = torch.tensor(volShape).apply_(lambda x: int(x * sf))
        print(f'sliceIds: {sliceIds}')
        display_slices.append([
            vol_cdhw[0][sliceIds[1], :, :], vol_cdhw[0][:, sliceIds[2], :], vol_cdhw[0][:, :, sliceIds[3]]
        ])
    torchshow.show(display_slices, figsize=figsize)

def display_2dimg(img_2d):
    torchshow.show(img_2d)

LoadersNamedTuple = namedtuple('LoadersNamedTuple', ['train_loader', 'val_loader', 'test_loader'])
def get_adni_dataloaders(args, batch_sz, adniTrnsrfms, val_frac=0.0) -> LoadersNamedTuple:
    if args.datasetDir:
        directory = args.datasetDir
    else:
        directory = os.environ.get("MONAI_DATA_DIRECTORY")
    rootDir = tempfile.mkdtemp() if directory is None else directory
    print(f'rootDir={rootDir}')

    train_ds = AdniDataset(root_dir=rootDir, task='data', transform=adniTrnsrfms, section="training",
                           seed=42, val_frac=val_frac)
    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, 
                              num_workers=8, persistent_workers=True)
    print(f'TrainLoader: Vol Shape: {train_ds[0]["image"].shape}')

    val_loader = None
    if val_frac > 0.0:
        val_ds = AdniDataset(root_dir=rootDir, task="data", transform=adniTrnsrfms, section="validation",
                          seed=42,  val_frac=val_frac)
        print(f'ValLoader: len={len(val_ds)}, shape: {val_ds[0]["image"].shape}')
        val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=True, 
                              num_workers=8, persistent_workers=True)
        print(f'ValLoader: Vol Shape: {val_ds[0]["image"].shape}')

    if args.debug:
        yn='y'
        while yn=='y' or yn=='Y' or yn=='':
            sampleVol = train_ds[torch.randint(low=0, high=len(train_ds), size=(1,))] #get a random sample from train_ds
            print(f'Vol Size: {Fore.BLUE}{sampleVol["image"].shape}{Fore.WHITE}')
            # display_3dvol_slices(sampleVol['image'], slice_fractions=[0.3, 0.4, 0.5, 0.6, 0.7])
            display_2dimg(sampleVol['image'])
            yn = input('Sample another volume? y/n: ')
    
    return LoadersNamedTuple(train_loader=train_loader, val_loader=val_loader, test_loader=None)