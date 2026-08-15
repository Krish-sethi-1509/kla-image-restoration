"""
Standalone inference script for KLA Track 1 — AI-Based Restoration of Degraded Images.

Loads a trained RestorationUNet checkpoint and runs it on a directory of
degraded (.npy) test images, saving restored (.npy) predictions to an
output directory.

This script is self-contained: it does not depend on any other project
file or notebook state, and runs with no manual edits required.

Usage:
    python inference.py --model_path weights/best_model.pth \
                         --input_dir Test_NoisyLR/NoisyLR \
                         --output_dir predictions
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition
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
    """
    Single-channel U-Net with a PixelShuffle super-resolution head.
    Maps 128x128 degraded (noisy/downsampled) inputs to 256x256 restored
    outputs (joint denoising + 2x super-resolution).
    """

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
# Inference
# ---------------------------------------------------------------------------
def run_inference(model_path: str, input_dir: str, output_dir: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(output_dir, exist_ok=True)

    model = RestorationUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path} on device: {device}")

    test_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npy"))
    if not test_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")
    print(f"Found {len(test_files)} test images. Running inference...")

    start = time.time()
    with torch.no_grad():
        for fname in test_files:
            noisy_arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
            noisy_tensor = (
                torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0).to(device)
            )  # (1, 1, H, W)

            pred = model(noisy_tensor).clamp(0, 1)
            pred_arr = pred.squeeze(0).squeeze(0).cpu().numpy()

            np.save(os.path.join(output_dir, fname), pred_arr)
    elapsed = time.time() - start

    avg_ms = (elapsed / len(test_files)) * 1000
    print(
        f"Inference complete: {len(test_files)} images in {elapsed:.2f}s "
        f"({avg_ms:.1f} ms/image)"
    )
    print(f"Predictions saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RestorationUNet inference on degraded test images."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="weights/best_model.pth",
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="Test_NoisyLR/NoisyLR",
        help="Directory containing degraded input images (.npy files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to save restored output images (.npy files).",
    )
    args = parser.parse_args()

    run_inference(args.model_path, args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
