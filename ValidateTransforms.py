import json
import argparse
from pathlib import Path

from monai import transforms

from AdniDataset import AdniDataset
from CustomTransforms import Random2DSliceTransformd, UnsqueezeTransformd
from colorama import Fore

expected_spatial_size = (128, 128, 128)

def _getAdniTransforms():
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=['image']),
        transforms.Orientationd(keys=['image'], axcodes="RAS"),
        transforms.CropForegroundd(keys=['image'], source_key="image"),
        transforms.Resized(keys=['image'], spatial_size=expected_spatial_size),
        # transforms.Resized(keys=['image'], spatial_size=(128, 128, 128)),
        # transforms.Resized(keys=['image'], spatial_size=(512, 512, 512)),
        Random2DSliceTransformd(keys=['image'], lo=0.35, hi=0.75),
        transforms.ScaleIntensityRangePercentilesd(keys=['image'], lower=0, upper=99.5, b_min=0, b_max=1),
        UnsqueezeTransformd(keys=["image"], axis=0),
    ])
def _validateTransforms(rootDir, lastProcessedIndex):
    txs = _getAdniTransforms()
    data_list_file_path = Path(f'{rootDir}/dataset.json')
    if not data_list_file_path.is_file():
        raise ValueError(f"Data list file {data_list_file_path} does not exist.")
    with open(data_list_file_path) as json_file:
        json_data = json.load(json_file)
    
    processedIndex = 0
    for item in json_data:
        if processedIndex < lastProcessedIndex:
            processedIndex += 1
            print(f'{Fore.GREEN}skipping: {Fore.WHITE}{processedIndex}: {item["image"]}')
            continue

        processedIndex += 1
        print(f'{Fore.MAGENTA}{processedIndex}: {Fore.BLUE}{item["image"]}{Fore.WHITE}')
        res = txs(item)
        #loadTx = transforms.LoadImage()
        #img = loadTx(f'{rootDir}/data/{item["image"]}')
        #for tx in txs:
        #    img = tx(img)
        #print(f'{item["image"]} : {img.shape}')
        img = res['image']
        if img.shape[0] != 1 or img.shape[1] != expected_spatial_size[1] or img.shape[2] != expected_spatial_size[2]:
            print(f'{Fore.RED}FAILED Index: {lastProcessedIndex} : {processedIndex} : {item["image"]}{Fore.WHITE}')
            assert("Problem!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootDir", type=str, default=".")
    parser.add_argument("--lastProcessedIndex", type=int, default="0")
    args = parser.parse_args()
    _validateTransforms(args.rootDir, args.lastProcessedIndex)
