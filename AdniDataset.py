from typing import Callable
from pathlib import Path
import numpy as np
import sys
from collections.abc import Callable, Sequence
import json

from monai.config.type_definitions import PathLike
from monai.data.dataset import( CacheDataset)
from monai.transforms.transform import Randomizable
from monai.transforms.io.dictionary import LoadImaged

class AdniDataset(Randomizable, CacheDataset):
    """
    The Dataset to load and generate items for training, validation or test.
    It will also load these properties from the JSON config file of dataset. user can call `get_properties()`
    to get specified properties or all the properties loaded.
    It's based on :py:class:`monai.data.CacheDataset` to accelerate the training process.

    Args:
        root_dir: user's local directory for caching and loading the ADNI datasets.
        task: which task to download and execute: one of list ("adni_go_mci_mri", "adni_go_cn_mri").
        transform: transforms to execute operations on input data.
            for further usage, use `EnsureChannelFirstd` to convert the shape to [C, H, W, D].
        val_frac: percentage of validation fraction in the whole dataset, default is 0.2.
        seed: random seed to randomly shuffle the datalist before splitting into training and validation, default is 0.
            note to set same seed for `training` and `validation` sections.
        cache_num: number of items to be cached. Default is `sys.maxsize`.
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        cache_rate: percentage of cached data in total, default is 1.0 (cache all).
            will take the minimum of (cache_num, data_length x cache_rate, data_length).
        num_workers: the number of worker threads if computing cache in the initialization.
            If num_workers is None then the number returned by os.cpu_count() is used.
            If a value less than 1 is specified, 1 will be used instead.
        progress: whether to display a progress bar computing the transform cache content.
        copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
            default to `True`. if the random transforms don't modify the cached content
            (for example, randomly crop from the cached image and deepcopy the crop region)
            or if every cache item is only used once in a `multi-processing` environment,
            may set `copy=False` for better performance.
        as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
            it may help improve the performance of following logic.
        runtime_cache: whether to compute cache at the runtime, default to `False` to prepare
            the cache content at initialization. See: :py:class:`monai.data.CacheDataset`.

    Raises:
        ValueError: When ``root_dir`` is not a directory.
        ValueError: When ``task`` is not one of ["adni_go_mci_mri", "adni_go_cn_mri"].
        RuntimeError: When ``dataset_dir`` doesn't exist and downloading is not selected (``download=False``).

    Example::

        transform = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_data = AdniDataset( root_dir="./", task="adni_go_cn_mri", transform=transform, seed=12345)

        print(val_data[0]["image"], val_data[0]["label"])

    """

    def __init__(
        self,
        root_dir: PathLike,
        task: str,
        section: str,
        transform: Sequence[Callable] | Callable = (),
        seed: int = 0,
        val_frac: float = 0.2,
        cache_num: int = sys.maxsize,
        cache_rate: float = 1.0,
        num_workers: int = 1,
        progress: bool = True,
        copy_cache: bool = True,
        as_contiguous: bool = True,
        runtime_cache: bool = False
    ) -> None:
        root_dir = Path(root_dir)
        if not root_dir.is_dir():
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.set_random_state(seed=seed)
        dataset_dir = root_dir / task

        if not dataset_dir.exists():
            raise RuntimeError( f"Cannot find dataset directory: {dataset_dir}")

        self.indices: np.ndarray = np.array([])
        data = self._generate_data_list(dataset_dir)
        if transform == ():
            transform = LoadImaged(["image"])
        CacheDataset.__init__(
            self,
            data=data,
            transform=transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
            progress=progress,
            copy_cache=copy_cache,
            as_contiguous=as_contiguous,
            runtime_cache=runtime_cache,
        )

    def _generate_data_list(self, dataset_dir: PathLike) -> list[PathLike]:
        dataset_dir = Path(dataset_dir)
        # section = "training" if self.section in ["training", "validation"] else "test"
        datalist = self._load_adni_datalist(dataset_dir / "dataset.json", dataset_dir)
        return self._split_datalist(datalist)

    def _load_adni_datalist(self, data_list_file_path, base_dir: PathLike) -> list[PathLike]:
        data_list_file_path = Path(data_list_file_path)
        if not data_list_file_path.is_file():
            raise ValueError(f"Data list file {data_list_file_path} does not exist.")
        with open(data_list_file_path) as json_file:
            json_data = json.load(json_file)

        if base_dir is None:
            base_dir = data_list_file_path.parent

        #prefix basedir
        for item in json_data:
            item['image'] = f'{base_dir}/{item["image"]}'

        return json_data
        # ret = [base_dir/f for f in json_data]
        # return ret

    def _split_datalist(self, datalist: list[PathLike]) -> list[PathLike]:
        # if self.section == "test":
        #     return datalist
        length = len(datalist)
        indices = np.arange(length)
        self.randomize(indices)

        val_length = int(length * self.val_frac)
        if self.section == "training":
            self.indices = indices[val_length:]
        else:
            self.indices = indices[:val_length]

        return [datalist[i] for i in self.indices]
    
    def get_indices(self)->np.ndarray:
        return self.indices

    def randomize(self, data: np.ndarray) -> None:
        self.R.shuffle(data)

