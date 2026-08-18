"""
run.py — KLA Track 1 submission entry point.
Usage: python run.py <input-dir> <output-dir>
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

# ---- Model definition ----
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class RestorationUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base*2); self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base*2, base*4); self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2); self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2); self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2); self.dec1 = ConvBlock(base*2, base)
        self.sr_head = nn.Sequential(
            nn.Conv2d(base, base*4, 3, padding=1), nn.PixelShuffle(2),
            nn.Conv2d(base, base, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(base, out_ch, 3, padding=1),
        )
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sr_head(d1)


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>")
        sys.exit(1)

    input_dir, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    # Model path is fixed relative to this script — no manual config needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", "best_model.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RestorationUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    input_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".npy"))
    if not input_files:
        print(f"No .npy files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} input files. Running on {device}...")

    with torch.no_grad():
        for fname in input_files:
            arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)

            # normalize shape defensively: accept (H,W) or (H,W,1)
            if arr.ndim == 3:
                arr = arr[:, :, 0]

            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(tensor).clamp(0.0, 1.0)
            pred_arr = pred.squeeze(0).squeeze(0).cpu().numpy()

            # safety: guarantee no NaN/Inf and valid range before saving
            pred_arr = np.nan_to_num(pred_arr, nan=0.0, posinf=1.0, neginf=0.0)
            pred_arr = np.clip(pred_arr, 0.0, 1.0).astype(np.float32)

            np.save(os.path.join(output_dir, fname), pred_arr)

    print(f"Done. Restored {len(input_files)} images to {output_dir}")


if __name__ == "__main__":
    main()
