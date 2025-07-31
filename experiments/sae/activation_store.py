import torch
from torch.utils.data import IterableDataset, DataLoader
import h5py
import glob
import os
import random
import tqdm
import functools
import numpy as np
import pandas as pd
from omegaconf import DictConfig

class FileFeatureDataset(IterableDataset):
    def __init__(
        self,
        base_feature_dir: str,
        file_type: str,
        buffer_size: int,
        h5_dataset_name: str = None,
        split_csv_path: str = None,
        split_name: str = None,
        feature_dim: int = None,
        dtype: torch.dtype = torch.float32,
        cache_maxsize: int = 512,
    ):
        super().__init__()
        self.file_type = file_type.lower()
        if self.file_type not in ["h5", "pt"]:
            raise ValueError(f"Unsupported file_type: {file_type}")
        if self.file_type == "h5" and not h5_dataset_name:
            raise ValueError("h5_dataset_name must be provided for 'h5' files")

        self.h5_dataset_name = h5_dataset_name
        self.buffer_size = buffer_size
        self.dtype = dtype
        self.cache_maxsize = cache_maxsize
        self.split_csv_path = split_csv_path
        self.split_name = split_name

        self.actual_feature_dir = os.path.join(base_feature_dir, f"{self.file_type}_files")
        if not os.path.isdir(self.actual_feature_dir):
            self.actual_feature_dir = base_feature_dir

        file_extension = f"*.{self.file_type}"
        all_files = sorted(glob.glob(os.path.join(self.actual_feature_dir, file_extension)))
        if not all_files:
            raise FileNotFoundError(f"No {file_extension} files found in {self.actual_feature_dir}")

        if split_csv_path and split_name:
            split_df = pd.read_csv(split_csv_path)
            if split_name not in split_df.columns:
                raise ValueError(f"Column '{split_name}' not in {split_csv_path}")
            slide_ids = set(split_df[split_name].dropna().astype(str))
            self.data_files = [fp for fp in all_files if os.path.splitext(os.path.basename(fp))[0] in slide_ids]
            if not self.data_files:
                raise FileNotFoundError(f"No {file_extension} files for split '{split_name}'")
        else:
            self.data_files = all_files

        self.file_patch_counts = []
        self.cumulative_patch_counts = [0]
        self.total_patches = 0
        self.feature_dim = feature_dim
        valid_files = []

        for fp in tqdm.tqdm(self.data_files, desc=f"Scanning {self.file_type.upper()} files"):
            try:
                if self.file_type == "h5":
                    with h5py.File(fp, "r") as f:
                        if self.h5_dataset_name not in f:
                            continue
                        dset = f[self.h5_dataset_name]
                        if dset.ndim != 2 or dset.shape[0] == 0:
                            continue
                        num_patches = dset.shape[0]
                        dim = dset.shape[1]
                else:
                    tensor_data = torch.load(fp, map_location='cpu')
                    if not isinstance(tensor_data, torch.Tensor) or tensor_data.ndim != 2 or tensor_data.shape[0] == 0:
                        continue
                    num_patches = tensor_data.shape[0]
                    dim = tensor_data.shape[1]
                    del tensor_data

                if self.feature_dim is None:
                    self.feature_dim = dim
                elif self.feature_dim != dim:
                    continue

                self.file_patch_counts.append(num_patches)
                self.cumulative_patch_counts.append(self.cumulative_patch_counts[-1] + num_patches)
                valid_files.append(fp)
            except:
                continue

        self.data_files = valid_files
        self.total_patches = self.cumulative_patch_counts[-1]
        if self.total_patches == 0:
            raise ValueError(f"No valid patches{' for split ' + split_name if split_name else ''} in {self.actual_feature_dir}")
        if self.feature_dim is None:
            raise ValueError(f"Could not determine feature dimension from {self.actual_feature_dir}")

        self.buffer = None
        self.buffer_pos = 0
        self._load_data_func = (
            self._get_h5_loader_cached() if self.file_type == "h5"
            else self._get_pt_loader_cached()
        )

    @functools.lru_cache(maxsize=None)
    def _get_h5_loader_cached(self):
        @functools.lru_cache(maxsize=self.cache_maxsize)
        def _load(fp: str) -> np.ndarray:
            try:
                with h5py.File(fp, "r") as f:
                    data = f[self.h5_dataset_name][:]
                    if data.ndim == 2 and data.shape[1] == self.feature_dim:
                        return data
            except:
                pass
            return None
        return _load

    @functools.lru_cache(maxsize=None)
    def _get_pt_loader_cached(self):
        @functools.lru_cache(maxsize=self.cache_maxsize)
        def _load(fp: str) -> torch.Tensor:
            try:
                data = torch.load(fp, map_location='cpu')
                if isinstance(data, torch.Tensor) and data.ndim == 2 and data.shape[1] == self.feature_dim:
                    return data.to(dtype=self.dtype)
            except:
                pass
            return None
        return _load

    def _fill_buffer(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        all_indices = list(range(len(self.data_files)))
        random.shuffle(all_indices)
        file_indices = all_indices[worker_id::num_workers]

        buffer_list = []
        filled = 0
        while filled < self.buffer_size:
            if not file_indices:
                break
            idx = file_indices.pop(0)
            fp = self.data_files[idx]
            data = self._load_data_func(fp)
            if data is None or data.shape[0] == 0:
                continue
            needed = self.buffer_size - filled
            if data.shape[0] > needed:
                start = random.randint(0, data.shape[0] - needed)
                raw = data[start : start + needed]
            else:
                raw = data
            if isinstance(raw, np.ndarray):
                tensor_slice = torch.from_numpy(raw).to(dtype=self.dtype)
            else:
                tensor_slice = raw.to(dtype=self.dtype)
            buffer_list.append(tensor_slice)
            filled += tensor_slice.shape[0]
        if not buffer_list:
            self.buffer = None
            self.buffer_pos = 0
            return
        self.buffer = torch.cat(buffer_list, dim=0)
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0])]
        self.buffer_pos = 0

    def __iter__(self):
        while True:
            if self.buffer is None or self.buffer_pos >= len(self.buffer):
                try:
                    self._fill_buffer()
                except RuntimeError:
                    return
                if self.buffer is None or len(self.buffer) == 0:
                    return
            yield self.buffer[self.buffer_pos]
            self.buffer_pos += 1

class FeatureActivationStore:
    def __init__(self, cfg: DictConfig, split_name: str):
        self.target_device = cfg.exp.sae.device
        dtype_str = cfg.exp.sae.get("dtype", "float32")
        self.torch_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(dtype_str)
        if self.torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")

        buffer_size = cfg.exp.data.num_batches_in_buffer * cfg.exp.training.batch_size
        cache_maxsize = cfg.exp.data.get("cache_maxsize", 512)
        split_csv_path = cfg.exp.data.get("split_csv_path")
        file_type = cfg.exp.data.get("file_type", "h5")
        h5_dataset_name = cfg.exp.data.get("h5_dataset_name")
        base_dir = cfg.exp.data.feature_dir

        self.dataset = FileFeatureDataset(
            base_feature_dir=base_dir,
            file_type=file_type,
            h5_dataset_name=h5_dataset_name,
            buffer_size=buffer_size,
            split_csv_path=split_csv_path,
            split_name=split_name,
            feature_dim=cfg.exp.sae.get("act_size"),
            dtype=self.torch_dtype,
            cache_maxsize=cache_maxsize,
        )
        self.act_size = self.dataset.feature_dim

        num_workers = cfg.exp.data.get("num_workers", 0)
        pin_memory = ("cuda" in str(self.target_device) and num_workers > 0)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.exp.training.batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=(cfg.exp.data.get("prefetch_factor", 2) if num_workers > 0 else None),
        )
        self.dataloader_iter = iter(self.dataloader)
        print(f"DataLoader for split '{split_name}' initialized, device={self.target_device}")

    def next_batch(self) -> torch.Tensor:
        try:
            batch = next(self.dataloader_iter)
            return batch.to(self.target_device)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            try:
                batch = next(self.dataloader_iter)
                return batch.to(self.target_device)
            except StopIteration:
                raise RuntimeError(f"Failed to get batch for split '{self.dataset.split_name}'")

    def get_feature_dim(self) -> int:
        if self.act_size is None:
            raise ValueError("Feature dimension not determined.")
        return self.act_size

    def get_total_patches(self) -> int:
        return self.dataset.total_patches

    def get_cache_info(self):
        if self.dataset.file_type == "h5":
            loader = self.dataset._get_h5_loader_cached()
        else:
            loader = self.dataset._get_pt_loader_cached()
        return loader.cache_info() if hasattr(loader, "cache_info") else "No cache info"
