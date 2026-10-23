import os
import sys
import glob
import time
import argparse
import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dl_model.models import FeatureLevelSCSEUNet
from losses.losses import BCETverskySobelLoss

class SpatialDataset(Dataset):
    """
    Dataset loader for 100% Spatially Disjoint Multimodal Sentinel-1 + Sentinel-2 Patches.
    """
    def __init__(self, split_dir, prefix):
        s1_dir = os.path.join(split_dir, f"{prefix}_S1", "patches", "images")
        s2_dir = os.path.join(split_dir, f"{prefix}_S2", "patches", "images")
        mask_dir = os.path.join(split_dir, f"{prefix}_mask_thin", "patches", "images")
        weight_dir = os.path.join(split_dir, f"{prefix}_weight_thin", "patches", "images")
        
        self.s1_files = sorted(glob.glob(os.path.join(s1_dir, "*.tif")))
        self.s2_files = sorted(glob.glob(os.path.join(s2_dir, "*.tif")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.weight_files = sorted(glob.glob(os.path.join(weight_dir, "*.tif")))
        
    def __len__(self):
        return len(self.s1_files)
        
    def __getitem__(self, idx):
        with rasterio.open(self.s1_files[idx]) as s: s1 = np.nan_to_num(s.read().astype(np.float32), nan=0.0)
        with rasterio.open(self.s2_files[idx]) as s: s2 = np.nan_to_num(s.read().astype(np.float32), nan=0.0)
        with rasterio.open(self.mask_files[idx]) as s: m = np.nan_to_num(s.read().astype(np.float32), nan=0.0)
        with rasterio.open(self.weight_files[idx]) as s: w = np.nan_to_num(s.read().astype(np.float32), nan=0.0)
        
        # Min-max normalization per patch
        s1 = (s1 - s1.min(axis=(1, 2), keepdims=True)) / (s1.max(axis=(1, 2), keepdims=True) - s1.min(axis=(1, 2), keepdims=True) + 1e-8)
        s2 = (s2 - s2.min(axis=(1, 2), keepdims=True)) / (s2.max(axis=(1, 2), keepdims=True) - s2.min(axis=(1, 2), keepdims=True) + 1e-8)
        
        m = (m > 0.5).astype(np.float32)
        return torch.from_numpy(s1), torch.from_numpy(s2), torch.from_numpy(m), torch.from_numpy(w)

def train_pipeline(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device} | Stage: {args.stage.upper()}")
    
    # Configure architecture flags based on stage
    if args.stage == 'baseline':
        use_scse = True
        use_sd = False
    elif args.stage == 'residual':
        use_scse = True
        use_sd = False
    elif args.stage == 'sd_scse':
        use_scse = True
        use_sd = True
    else:
        raise ValueError(f"Unknown stage: {args.stage}")
        
    model = FeatureLevelSCSEUNet(
        n_classes=1, 
        in_ch_s1=3, 
        in_ch_s2=4, 
        use_scse=use_scse, 
        use_sd=use_sd
    ).to(device)
    
    train_ds = SpatialDataset(os.path.join(args.data_dir, 'train'), 'train')
    val_ds = SpatialDataset(os.path.join(args.data_dir, 'val'), 'val')
    test_ds = SpatialDataset(os.path.join(args.data_dir, 'test'), 'test')
    
    if len(train_ds) == 0:
        print(f"\n[!] Notice: No dataset patches found in '{args.data_dir}/train'.")
        print("    To train, please place your multimodal dataset patches in 'data_split/' or pass '--data_dir <path>'.")
        print("    To verify the model architecture without a dataset, run:\n        python test_architecture.py\n")
        return

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"[*] Data loaded -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = BCETverskySobelLoss(bce_weight=1.0, tversky_weight=1.0, sobel_weight=0.25)
    
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float('inf')
    save_path = os.path.join(args.output_dir, f"best_model_{args.stage}.pth")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for s1, s2, m, w in train_loader:
            s1, s2, m, w = s1.to(device), s2.to(device), m.to(device), w.to(device)
            optimizer.zero_grad()
            logits = model(s1, s2)
            loss = criterion(logits, m, w)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * s1.size(0)
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s1, s2, m, w in val_loader:
                s1, s2, m, w = s1.to(device), s2.to(device), m.to(device), w.to(device)
                logits = model(s1, s2)
                loss = criterion(logits, m, w)
                val_loss += loss.item() * s1.size(0)
        val_loss /= len(val_ds)
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best checkpoint: {save_path}")
            
    print(f"\n[DONE] Training complete. Best model saved at: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Multimodal Edge-Aware Dual-Stream U-Net")
    parser.add_argument('--stage', type=str, default='sd_scse', choices=['baseline', 'residual', 'sd_scse'],
                        help="Model architecture stage to train")
    parser.add_argument('--data_dir', type=str, default='data_split', help="Path to data split directory")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Directory to save model checkpoints")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_pipeline(args)
