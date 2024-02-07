from typing import Any
import random

from monai import transforms
from monai.config import KeysCollection

from colorama import Fore

class UnsqueezeTransformd(transforms.MapTransform):
    def __init__(
            self, keys: KeysCollection,
            axis: int = 0,
            allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.axis = axis

    def __call__(self, imgDict: Any):
        d = dict(imgDict)
        for k in self.key_iterator(d):
            d[k] = d[k].unsqueeze(self.axis)
        return d

class Random2DSliceTransformd(transforms.MapTransform):
    def __init__(
            self, 
            keys: KeysCollection, 
            lo:float = 0.3, 
            hi:float=0.75, 
            allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.lo = lo
        self.hi = hi
        self.curr_slice_str_rep = ""

    def __call__(self, imgDict: Any):
        d = dict(imgDict)
        for k in self.key_iterator(d):
            img = d[k]
            img = img[0] #just the image without channel dim
            slice_frac = random.uniform(self.lo, self.hi)
            slice_ch = random.randint(0, 2)
            img_shape = img.shape
            if slice_ch == 0:
                slice_idx = int(img_shape[0]*slice_frac)
                d[k] = img[slice_idx, :, :]
                self.curr_slice_str_rep = f"[{slice_idx}, {img_shape[1]}, {img_shape[2]}]"
                print(f'  (slice_ch, slice_idx)={Fore.RED}({slice_ch}, {slice_idx}) : {Fore.BLUE}[{slice_idx}, {img_shape[1]}, {img_shape[2]}]{Fore.WHITE}')
            elif slice_ch == 1:
                slice_idx = int(img_shape[1]*slice_frac)
                d[k] = img[:, slice_idx, :]
                self.curr_slice_str_rep = f"[{img_shape[0]}, {slice_idx}, {img_shape[2]}]"
                print(f'  (slice_ch, slice_idx): {Fore.RED}({slice_ch}, {slice_idx}) : {Fore.BLUE}[{img_shape[0]}, {slice_idx}, {img_shape[2]}]{Fore.WHITE}')
            elif slice_ch == 2:
                slice_idx = int(img_shape[2]*slice_frac)
                d[k] = img[:, :, slice_idx]
                self.curr_slice_str_rep = f"[{img_shape[0]}, {img_shape[1]}, {slice_idx}]"
                print(f'  (slice_ch, slice_idx): {Fore.RED}({slice_ch}, {slice_idx}) : {Fore.BLUE}[{img_shape[0]}, {img_shape[1]}, {slice_idx}]{Fore.WHITE}')
            else:
                print(f'ERR: Unexpected slice_ch: {slice_ch}"')
                assert(False)
        return d
    
class Sequential2DSliceTransformd(transforms.MapTransform):
    def __init__(
            self, 
            keys: KeysCollection, 
            lo:float = 0.3, 
            hi:float=0.75, 
            allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.lo = lo
        self.hi = hi
        self.curr_index = 0
        self.curr_ch = 0
        self.curr_slice_str_rep = ""
        self.is_next_slice_available = True

    def __call__(self, imgDict: Any):
        d = dict(imgDict)
        for k in self.key_iterator(d):
            img = d[k]
            img = img[0] #just the image without channel dim
            img_shape = img.shape
            slice_low_offsets = [int(dim * self.lo) for dim in img_shape]
            slice_hi_offsets = [int(dim * self.hi) for dim in img_shape]

            curr_slice_id_in_curr_channel = slice_low_offsets[self.curr_ch] + self.curr_index
            if curr_slice_id_in_curr_channel >= slice_hi_offsets[self.curr_ch]:
                self.curr_index = 0
                self.curr_ch = self.curr_ch + 1
                curr_slice_id_in_curr_channel = slice_low_offsets[self.curr_ch] + self.curr_index
            self.curr_index += 1

            # Check if next-slice(curr_index+1) is OOB on the last-channel
            if (self.curr_ch >= len(img_shape)-1) and ((self.curr_index + slice_low_offsets[self.curr_ch]) >= slice_hi_offsets[self.curr_ch]):
                self.is_next_slice_available = False
                
            if self.curr_ch == 0:
                d[k] = img[curr_slice_id_in_curr_channel, :, :]
                self.curr_slice_str_rep = f"[{curr_slice_id_in_curr_channel}, {img_shape[1]}, {img_shape[2]}]"
                print(f'  (slice_ch, slice_idx)={Fore.RED}({self.curr_ch}, {curr_slice_id_in_curr_channel}) : {Fore.BLUE}[{curr_slice_id_in_curr_channel}, {img_shape[1]}, {img_shape[2]}]{Fore.WHITE}')
            elif self.curr_ch == 1:
                d[k] = img[:, curr_slice_id_in_curr_channel, :]
                self.curr_slice_str_rep = f"[{img_shape[0]}, {curr_slice_id_in_curr_channel}, {img_shape[2]}]"
                print(f'  (slice_ch, slice_idx): {Fore.RED}({self.curr_ch}, {curr_slice_id_in_curr_channel}) : {Fore.BLUE}[{img_shape[0]}, {curr_slice_id_in_curr_channel}, {img_shape[2]}]{Fore.WHITE}')
            elif self.curr_ch == 2:
                d[k] = img[:, :, curr_slice_id_in_curr_channel]
                self.curr_slice_str_rep = f"[{img_shape[0]}, {img_shape[1]}, {curr_slice_id_in_curr_channel}]"
                print(f'  (slice_ch, slice_idx): {Fore.RED}({self.curr_ch}, {curr_slice_id_in_curr_channel}) : {Fore.BLUE}[{img_shape[0]}, {img_shape[1]}, {curr_slice_id_in_curr_channel}]{Fore.WHITE}')
            else:
                print(f'ERR: Unexpected slice_ch: {self.curr_ch}"')
                assert(False)
        return d

class SliceExtractorTransformd(transforms.MapTransform):
    def __init__(
            self, 
            keys: KeysCollection, 
            channel_id:int = 0,
            slice_id:int = 0,
            allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.channel_id = channel_id
        self.slice_id = slice_id
        self.curr_slice_str_rep = ""

    def __call__(self, imgDict: Any):
        d = dict(imgDict)
        for k in self.key_iterator(d):
            img = d[k]
            img = img[0] #just the image without channel dim
            img_shape = img.shape
            if self.channel_id == 0:
                d[k] = img[self.slice_id, :, :]
                self.curr_slice_str_rep = f"[{self.slice_id}, {img_shape[1]}, {img_shape[2]}]"
            elif self.channel_id == 1:
                d[k] = img[:, self.slice_id, :]
                self.curr_slice_str_rep = f"[{img_shape[0]}, {self.slice_id}, {img_shape[2]}]"
            elif self.channel_id == 2:
                d[k] = img[:, :, self.slice_id]
                self.curr_slice_str_rep = f"[{img_shape[0]}, {img_shape[1]}, {self.slice_id}]"
            else:
                print(f'ERR: Unexpected slice_ch: {self.channel_id}"')
                assert(False)
        return d
        