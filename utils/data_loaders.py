import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_generator import DataGeneratorSingle, DataGeneratorDual

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_single_loader(base_dir, split, img_source, batch_size, shuffle=True, generator=None, num_workers=0, pin_memory=True, worker_init_fn=None):
    split_root = os.path.join(base_dir, f"{split}_{img_source}")

    imgs_dir = os.path.join(split_root, "patches", "images")
    masks_dir = os.path.join(split_root, "patches", "masks")

    dataset = DataGeneratorSingle(images_dir=imgs_dir, masks_dir=masks_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, worker_init_fn=worker_init_fn, generator=generator)
    return loader

def make_dual_loader(base_dir, split, batch_size, shuffle=True, generator=None, num_workers=0, pin_memory=True, worker_init_fn=None):
    s1_root = os.path.join(base_dir, f"{split}_S1")
    s2_root = os.path.join(base_dir, f"{split}_S2")

    imgs1_dir = os.path.join(s1_root, "patches", "images")
    imgs2_dir = os.path.join(s2_root, "patches", "images")
    masks_dir = os.path.join(s1_root, "patches", "masks")

    dataset = DataGeneratorDual(images1_dir=imgs1_dir, images2_dir=imgs2_dir, masks_dir=masks_dir)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers, worker_init_fn=worker_init_fn, generator=generator)
    return loader