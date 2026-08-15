"""
Standalone training script for KLA Track 1 — AI-Based Restoration of Degraded Images.

Trains a single-channel U-Net with a PixelShuffle super-resolution head to
jointly denoise and 2x-upscale degraded semiconductor inspection images
(128x128 NoisyLR -> 256x256 GT), using a combined L1 + SSIM loss.

Usage:
    python train.py --train_noisy_dir data/train/NoisyLR \
                     --train_gt_dir data/train/GT \
                     --output_dir weights \
                     --epochs 60
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from pytorch_msssim import ssim as ssim_fn
except ImportError as e:
    raise ImportError(
        "pytorch-msssim is required. Install with: pip install pytorch-msssim"
    ) from e


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RestorationDataset(Dataset):
    """Pairs degraded (.npy) images with their ground-truth (.npy) counterpart
    by matching filename between the noisy and GT directories."""

    def __init__(self, noisy_dir: str, gt_dir: str):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.filenames = sorted(os.listdir(noisy_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        noisy_arr = np.load(os.path.join(self.noisy_dir, fname)).astype(np.float32)
        gt_arr = np.load(os.path.join(self.gt_dir, fname)).astype(np.float32)
        noisy_tensor = torch.from_numpy(noisy_arr).unsqueeze(0)  # (1, H, W)
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0)
        return noisy_tensor, gt_tensor, fname


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class RestorationUNet(nn.Module):
    """Single-channel U-Net with a PixelShuffle super-resolution head.
    Maps 128x128 degraded inputs to 256x256 restored outputs."""

    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.sr_head = nn.Sequential(
            nn.Conv2d(base, base * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, out_ch, 3, padding=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.sr_head(d1)


# ---------------------------------------------------------------------------
# Loss & metrics
# ---------------------------------------------------------------------------
def combined_loss(pred, target, ssim_weight=0.3):
    l1 = nn.functional.l1_loss(pred, target)
    ssim_val = ssim_fn(pred.clamp(0, 1), target, data_range=1.0, size_average=True)
    ssim_loss = 1 - ssim_val
    return l1 + ssim_weight * ssim_loss, l1.item(), ssim_val.item()


def compute_psnr(pred, target, data_range=1.0):
    mse = nn.functional.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float("inf"))
    return 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    full_dataset = RestorationDataset(args.train_noisy_dir, args.train_gt_dir)
    val_len = int(len(full_dataset) * args.val_frac)
    train_len = len(full_dataset) - val_len
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    print(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, "
        f"device: {device}"
    )

    model = RestorationUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        start = time.time()
        for noisy, gt, _ in train_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            optimizer.zero_grad()
            pred = model(noisy)
            loss, _, _ = combined_loss(pred, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0
        with torch.no_grad():
            for noisy, gt, _ in val_loader:
                noisy, gt = noisy.to(device), gt.to(device)
                pred = model(noisy).clamp(0, 1)
                loss, _, ssim_val = combined_loss(pred, gt)
                val_loss += loss.item()
                val_psnr += compute_psnr(pred, gt).item()
                val_ssim += ssim_val
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val PSNR: {val_psnr:.2f} | "
            f"Val SSIM: {val_ssim:.4f} | {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved new best model (val_loss: {val_loss:.4f})")

    print(f"Training complete. Best model saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RestorationUNet on degraded/GT image pairs."
    )
    parser.add_argument("--train_noisy_dir", type=str, default="data/train/NoisyLR")
    parser.add_argument("--train_gt_dir", type=str, default="data/train/GT")
    parser.add_argument("--output_dir", type=str, default="weights")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
