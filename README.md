# KLA Track 1 — AI-Based Restoration of Degraded Images

SEMICON India Hackathon 2026 submission for the KLA problem statement:
restore semiconductor inspection images degraded by speckle noise,
Gaussian noise, and downsampling — all at once — while generalizing to
unseen data and running fast at inference.

## Approach

A single-channel U-Net (~7.9M params) with a PixelShuffle super-resolution
head performs joint denoising and 2x upscaling in a single forward pass:
128x128 degraded input -> 256x256 restored output.

- **Loss:** combined L1 + SSIM (`L1 + 0.3 * (1 - SSIM)`), balancing pixel
  accuracy with structural fidelity — chosen since SSIM is a scored metric.
- **Optimizer:** Adam, lr=1e-4, with `ReduceLROnPlateau` scheduling on
  validation loss.
- **Data:** paired `.npy` arrays (float32, single channel), NoisyLR
  (128x128) and GT (256x256), matched by filename. 90/10 train/val split.

## Results

| Metric     | Value   |
|------------|---------|
| Val PSNR   | 25.86 dB |
| Val SSIM   | 0.7607   |
| Inference speed | ~15.5 ms/image (Tesla T4) |
| Trained    | 60 epochs |

## Repo structure

```
.
├── train.py            # standalone training script
├── inference.py         # standalone inference script (no manual edits needed)
├── requirements.txt
├── weights/
│   └── best_model.pth   # trained checkpoint
└── predictions/          # inference output on Test_NoisyLR (generated)
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train from scratch:

```bash
python train.py --train_noisy_dir data/train/NoisyLR \
                 --train_gt_dir data/train/GT \
                 --output_dir weights \
                 --epochs 60
```

Run inference on the provided test set:

```bash
python inference.py --model_path weights/best_model.pth \
                     --input_dir Test_NoisyLR/NoisyLR \
                     --output_dir predictions
```

## Tech stack

Python, PyTorch, pytorch-msssim, NumPy, trained on Google Colab (Tesla T4 GPU).

## References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
  Segmentation" (2015)
- Shi et al., "Real-Time Single Image and Video Super-Resolution Using an
  Efficient Sub-Pixel Convolutional Neural Network" (PixelShuffle, 2016)
- Wang et al., "Image Quality Assessment: From Error Visibility to
  Structural Similarity" (SSIM, 2004)
