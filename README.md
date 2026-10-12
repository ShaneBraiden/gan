# Simple StyleGAN

A small, readable StyleGAN implementation in PyTorch. The whole model, training loop and sampling code is in one file, [`stylegan.py`](stylegan.py), at about 500 lines including comments.

It comes with a built-in toy dataset of coloured shapes, so you can train it from scratch on a laptop CPU in about 20 minutes and see results without downloading anything. You can also train it on a folder of your own images.

## What's implemented

| StyleGAN idea | Where in `stylegan.py` |
|---|---|
| **Mapping network**: an 8-layer MLP that turns random noise `z` into a style vector `w` | `MappingNetwork` |
| **Learned constant input**: the generator starts from a trained 4×4 tensor, not from `z` | `Generator.const` |
| **AdaIN**: `w` sets the scale and shift of every feature map, in every layer | `AdaIN`, `StyleLayer` |
| **Noise injection**: per-pixel random noise adds fine, stochastic detail | `NoiseInjection` |
| **Style mixing**: during training, two `w`s are swapped at a random layer | `Generator.forward(mix_prob=...)` |
| **Truncation trick**: pulls `w` toward the average for cleaner but less varied samples | `Generator.z_to_w(truncation=...)` |
| Equalized learning rate | `EqualLinear`, `EqualConv2d` |
| Minibatch standard deviation, in the discriminator | `minibatch_stddev` |
| Non-saturating logistic loss with lazy R1 regularization | `train()` |
| EMA copy of the generator, used for all sampling | `train()` → `G_ema` |

## Architecture

```
            z (128)                                Generator (32×32 output)
               │
   ┌───────────▼───────────┐            const 4×4 ──► noise ─► AdaIN ◄── w[0]
   │   Mapping network     │                          conv  ─► noise ─► AdaIN ◄── w[1]
   │  PixelNorm + 8 × FC   │                                    │
   └───────────┬───────────┘            upsample ─► conv ─► noise ─► AdaIN ◄── w[2]   8×8
               │                                    conv ─► noise ─► AdaIN ◄── w[3]
               w (128)                                          │
               │                        upsample ─► conv ─► noise ─► AdaIN ◄── w[4]   16×16
   broadcast to every layer ──────►                 conv ─► noise ─► AdaIN ◄── w[5]
   (or mix two w's: style mixing)                               │
                                        upsample ─► conv ─► noise ─► AdaIN ◄── w[6]   32×32
                                                    conv ─► noise ─► AdaIN ◄── w[7]
                                                                │
                                                          1×1 conv ─► RGB image
```

The discriminator is an ordinary CNN. It takes the image through a 1×1 conv, then repeated [conv, conv, 2× downsample] blocks, then the minibatch-stddev layer, a final conv and fully connected layers, and outputs one real/fake score.

Early `w`s (4×4, 8×8) control **coarse** features such as shape, size and position. Later ones (16×16, 32×32) control **fine** features such as colour and texture. The `mix` command makes this visible.

## Installation

You need Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

The only dependencies are `torch`, `numpy` and `Pillow`; torchvision is not needed. A GPU (CUDA, or Apple Silicon via MPS) is used automatically if one is available, and everything also runs on a CPU.

## Quick start

```bash
# 1. Train on the built-in toy dataset (random circles, squares and triangles)
python stylegan.py train

# 2. Sample 16 new images
python stylegan.py generate --ckpt runs/stylegan/checkpoint.pt

# 3. Make a style-mixing grid
python stylegan.py mix --ckpt runs/stylegan/checkpoint.pt
```

During training, a sample grid is saved every 500 steps to `runs/stylegan/samples/step_XXXXXX.png`. Every grid uses the same 64 latent vectors, so you can watch the same images take shape.

With the defaults (32×32 images, 64 channels, 3000 steps), a run takes about **20–25 minutes on a laptop CPU** at roughly 0.45 s per step. On a GPU it takes a minute or two.

## Train on your own images

Point `--data` at a folder. It is searched recursively for `.png`, `.jpg`, `.jpeg`, `.bmp` and `.webp` files. Each image is centre-cropped to a square, resized, and randomly flipped left-to-right.

```bash
python stylegan.py train --data path/to/faces --img-size 64 --channels 128 --steps 20000 --out runs/faces
```

Some guidelines:

* **Dataset size**: GANs need many images. A few thousand is a sensible minimum; with only a few hundred, the discriminator memorises them.
* **Resolution**: 32 or 64 is a good place to start. 128 and above works, but needs a GPU and much longer training.
* **Capacity**: raise `--channels` (128, 256) for real photos. The toy dataset only needs 64.
* **Resuming**: continue an interrupted run with `--resume runs/faces/checkpoint.pt --steps <new total>`. The architecture settings are read from the checkpoint.

## Generate images

```bash
python stylegan.py generate --ckpt runs/stylegan/checkpoint.pt --n 64 --truncation 0.7 --seed 42 --out samples.png
```

* `--truncation` (ψ) sets the trade-off between quality and diversity. `1.0` uses raw samples (most varied, most artefacts). `0.5`–`0.7` gives cleaner, more "average" images. `0` returns the average image every time.
* `--seed` makes the output reproducible.

## Style mixing

```bash
python stylegan.py mix --ckpt runs/stylegan/checkpoint.pt --n 5 --cut 4
```

This saves an `(n+1) × (n+1)` grid:

* The **top row** and **left column** are ordinary generated images.
* Each inner cell takes its **coarse** styles (layers `0 … cut-1`) from the image at the top of its column, and its **fine** styles (the remaining layers) from the image at the left of its row.

With the 32×32 model there are 8 style inputs. `--cut 2` borrows only the 4×4 styles (overall layout). `--cut 6` borrows everything except the final 32×32 styles, so only fine colour detail comes from the left image.

## Command-line options

### `train`

| Option | Default | Description |
|---|---|---|
| `--data` | `shapes` | `shapes` for the built-in toy dataset, or a path to an image folder |
| `--img-size` | `32` | Output resolution; a power of two, at least 8 |
| `--z-dim` / `--w-dim` | `128` | Size of the latent `z` and the style vector `w` |
| `--channels` | `64` | Feature maps per conv layer, in both G and D |
| `--mapping-layers` | `8` | Depth of the mapping network |
| `--batch` | `32` | Batch size |
| `--steps` | `3000` | Total training steps |
| `--lr` | `0.002` | Adam learning rate, with betas = (0, 0.99) |
| `--r1` | `0.1` | R1 regularization weight γ (higher means a smoother D and more stable but blurrier results) |
| `--r1-every` | `4` | Apply R1 every N steps (lazy regularization) |
| `--mix-prob` | `0.9` | Probability of style mixing for each batch |
| `--ema` | `0.999` | Decay rate for the EMA generator |
| `--out` | `runs/stylegan` | Output folder for samples and the checkpoint |
| `--log-every` / `--sample-every` / `--save-every` | `50` / `500` / `1000` | Logging, sample-grid and checkpoint intervals, in steps |
| `--resume` | – | Checkpoint to continue training from |
| `--workers` | `0` | DataLoader worker processes (helps with large image folders) |
| `--device` | `auto` | `auto`, `cpu`, `cuda` or `mps` |
| `--seed` | `0` | Random seed |

### `generate`

| Option | Default | Description |
|---|---|---|
| `--ckpt` | required | Checkpoint to load (its EMA generator is used) |
| `--n` | `16` | Number of images |
| `--truncation` | `0.7` | Truncation ψ (`1.0` turns it off) |
| `--out` | `generated.png` | Output image grid |

### `mix`

| Option | Default | Description |
|---|---|---|
| `--ckpt` | required | Checkpoint to load |
| `--n` | `5` | Number of source images on each axis |
| `--cut` | half of the layers | How many coarse layers to take from the top row |
| `--truncation` | `0.7` | Truncation ψ |
| `--out` | `mixing.png` | Output image grid |

`generate` and `mix` also accept `--device` and `--seed`.

## Output layout

```
runs/stylegan/
├── checkpoint.pt          # G, D, EMA generator, optimizers, step, settings
└── samples/
    ├── step_000500.png    # 8×8 grid from fixed latents
    ├── step_001000.png
    └── ...
```

Small images are enlarged with nearest-neighbour scaling in saved grids so they are easy to see. The underlying samples are still `--img-size` pixels.

## Reading the training log

```
step   1000 | D 0.912 | G 1.734 | R1 0.052 | 450s
```

* **D** is the discriminator loss and **G** the generator loss. In a healthy GAN they move up and down without settling. If D stays near 0 and G keeps growing, the discriminator has won. Try a larger `--r1`, a lower `--lr` or more data.
* **R1** is the gradient penalty on real images. A steadily rising R1 means D is becoming sharp and overconfident.
* The sample grids are a better guide than the loss numbers. Look at them.

## How this differs from the official StyleGAN

This project is for learning, not for matching the paper's results. To keep the code short:

* **No progressive growing.** The networks train at full resolution from the start, as StyleGAN2 does.
* **AdaIN is kept** (StyleGAN 1). StyleGAN2 replaces it with weight demodulation, which removes the "droplet" artefacts.
* **The same channel count is used at every resolution**, and the upsampling is plain bilinear interpolation, with no blur filter.
* There is no path-length regularization, no mixed precision, no adaptive augmentation (ADA), and no FID evaluation.

## References

* T. Karras, S. Laine, T. Aila. *A Style-Based Generator Architecture for Generative Adversarial Networks.* CVPR 2019. [arXiv:1812.04948](https://arxiv.org/abs/1812.04948)
* T. Karras et al. *Analyzing and Improving the Image Quality of StyleGAN* (StyleGAN2). CVPR 2020. [arXiv:1912.04958](https://arxiv.org/abs/1912.04958)
* T. Karras et al. *Progressive Growing of GANs for Improved Quality, Stability, and Variation.* ICLR 2018 (equalized learning rate, minibatch stddev). [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
* L. Mescheder, A. Geiger, S. Nowozin. *Which Training Methods for GANs do actually Converge?* ICML 2018 (R1 regularization). [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)
