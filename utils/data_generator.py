import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio

def minmax_per_band(img):
    band_min = img.min(axis=(1, 2), keepdims=True)
    band_max = img.max(axis=(1, 2), keepdims=True)
    return (img - band_min) / (band_max - band_min + 1e-8)


def read_mask_binary(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1, out_dtype=np.float32)
    return (mask > 0).astype(np.float32)


def get_bounds(path):
    with rasterio.open(path) as src:
        b = src.bounds
    return (b.left, b.bottom, b.right, b.top)

class DataGeneratorDual(Dataset):
    def __init__(self, images1_dir, images2_dir, masks_dir):
        self.images1_dir = images1_dir
        self.images2_dir = images2_dir
        self.masks_dir = masks_dir

        self.image_s1_list = sorted(f for f in os.listdir(images1_dir) if not f.startswith("."))
        self.image_s2_list = sorted(f for f in os.listdir(images2_dir) if not f.startswith("."))
        self.mask_list = sorted(f for f in os.listdir(masks_dir) if not f.startswith("."))

        if not (len(self.image_s1_list) == len(self.image_s2_list) == len(self.mask_list)):
            raise ValueError("S1, S2, and mask counts do not match.")

        self.filenames = self.image_s1_list
        self.extents = [get_bounds(os.path.join(images1_dir, f)) for f in self.image_s1_list]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_s1_path = os.path.join(self.images1_dir, self.image_s1_list[idx])
        img_s2_path = os.path.join(self.images2_dir, self.image_s2_list[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_list[idx])

        with rasterio.open(img_s1_path) as src1:
            img_array_s1 = src1.read(out_dtype=np.float32)

        with rasterio.open(img_s2_path) as src2:
            img_array_s2 = src2.read(out_dtype=np.float32)

        mask_array = read_mask_binary(mask_path)

        img_array_s1 = minmax_per_band(img_array_s1)
        img_array_s2 = minmax_per_band(img_array_s2)

        s1_tensor = torch.from_numpy(img_array_s1).float()
        s2_tensor = torch.from_numpy(img_array_s2).float()
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

        return [s1_tensor, s2_tensor], mask_tensor, self.filenames[idx], self.extents[idx]


class DataGeneratorSingle(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.image_list = sorted(f for f in os.listdir(images_dir) if not f.startswith("."))
        self.mask_list = sorted(f for f in os.listdir(masks_dir) if not f.startswith("."))

        if len(self.image_list) != len(self.mask_list):
            raise ValueError("Image and mask counts do not match.")

        self.filenames = self.image_list
        self.extents = [get_bounds(os.path.join(images_dir, f)) for f in self.image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_list[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_list[idx])

        with rasterio.open(img_path) as src:
            img_array = src.read(out_dtype=np.float32)

        mask_array = read_mask_binary(mask_path)
        img_array = minmax_per_band(img_array)

        if img_array.shape[1:] != mask_array.shape:
            raise ValueError(f"Image and mask shapes do not match for {self.image_list[idx]}")

        img_tensor = torch.from_numpy(img_array).float()
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

        return img_tensor, mask_tensor, self.filenames[idx], self.extents[idx]