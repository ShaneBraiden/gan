"""
Simple StyleGAN in PyTorch (single file).

Implements the core ideas of StyleGAN (Karras et al., 2019):
  * Mapping network      z -> w  (MLP, 8 layers by default)
  * Synthesis network    starts from a learned constant 4x4 tensor
  * AdaIN                the style vector w modulates every conv layer
  * Noise injection      per-pixel noise adds stochastic detail
  * Style mixing         two latents are mixed during training
  * Truncation trick     trade diversity for quality at sampling time

plus a few training tricks from the StyleGAN papers: equalized learning rate,
minibatch standard deviation, non-saturating logistic loss, lazy R1
regularization and an EMA copy of the generator.
Progressive growing is left out to keep the code short.

Usage:
  python stylegan.py train                       # train on built-in toy shapes
  python stylegan.py train --data path/to/imgs   # train on your own images
  python stylegan.py generate --ckpt runs/stylegan/checkpoint.pt
  python stylegan.py mix --ckpt runs/stylegan/checkpoint.pt
"""
import argparse
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

ARCH_KEYS = ("img_size", "z_dim", "w_dim", "channels", "mapping_layers")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def lrelu(x):
    # Leaky ReLU with gain sqrt(2) so activations keep roughly unit variance.
    return F.leaky_relu(x, 0.2) * math.sqrt(2)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate (weights scaled at runtime)."""

    def __init__(self, in_dim, out_dim, bias_init=0.0, lr_mul=1.0, activation=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_dim,), float(bias_init)))
        self.scale = lr_mul / math.sqrt(in_dim)
        self.lr_mul = lr_mul
        self.activation = activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)
        return lrelu(out) if self.activation else out


class EqualConv2d(nn.Module):
    """Conv layer with equalized learning rate."""

    def __init__(self, in_ch, out_ch, kernel, padding=0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.scale = 1 / math.sqrt(in_ch * kernel * kernel)
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Module):
    """z -> w. A lower learning rate (lr_mul=0.01) keeps the mapping stable."""

    def __init__(self, z_dim, w_dim, n_layers):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(EqualLinear(z_dim if i == 0 else w_dim, w_dim, lr_mul=0.01, activation=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class NoiseInjection(nn.Module):
    """Adds per-pixel Gaussian noise, scaled by a learned per-channel weight."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        return x + self.weight * noise


class AdaIN(nn.Module):
    """Adaptive instance norm: normalize each feature map, then scale and shift it with the style."""

    def __init__(self, channels, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style = EqualLinear(w_dim, channels * 2)
        self.style.bias.data[:channels] = 1.0  # start with scale = 1, shift = 0

    def forward(self, x, w):
        scale, shift = self.style(w)[:, :, None, None].chunk(2, dim=1)
        return scale * self.norm(x) + shift


class StyleLayer(nn.Module):
    """[upsample] -> conv 3x3 -> noise -> activation -> AdaIN(w)"""

    def __init__(self, in_ch, out_ch, w_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv = EqualConv2d(in_ch, out_ch, 3, padding=1)
        self.noise = NoiseInjection(out_ch)
        self.adain = AdaIN(out_ch, w_dim)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = lrelu(self.noise(self.conv(x)))
        return self.adain(x, w)


# ---------------------------------------------------------------------------
# Generator and discriminator
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, img_size=32, z_dim=128, w_dim=128, channels=64, mapping_layers=8):
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)

        # 4x4 block: learned constant -> noise -> AdaIN -> conv -> noise -> AdaIN
        self.const = nn.Parameter(torch.randn(1, channels, 4, 4))
        self.const_noise = NoiseInjection(channels)
        self.const_adain = AdaIN(channels, w_dim)
        self.const_conv = StyleLayer(channels, channels, w_dim)

        # One block per doubling of resolution: 8x8, 16x16, ... img_size
        self.layers = nn.ModuleList()
        for _ in range(int(math.log2(img_size)) - 2):
            self.layers.append(StyleLayer(channels, channels, w_dim, upsample=True))
            self.layers.append(StyleLayer(channels, channels, w_dim))
        self.to_rgb = EqualConv2d(channels, 3, 1)

        self.num_ws = 2 + len(self.layers)  # one style vector per AdaIN
        self.register_buffer("w_avg", torch.zeros(w_dim))  # running mean of w, for truncation

    def z_to_w(self, z, truncation=1.0):
        w = self.mapping(z)
        if self.training:
            self.w_avg.lerp_(w.detach().mean(0), 0.005)
        if truncation < 1.0:
            w = self.w_avg + truncation * (w - self.w_avg)
        return w

    def synthesis(self, ws):
        """ws: [batch, num_ws, w_dim] -- one style vector per layer."""
        x = self.const.expand(ws.shape[0], -1, -1, -1)
        x = self.const_adain(lrelu(self.const_noise(x)), ws[:, 0])
        x = self.const_conv(x, ws[:, 1])
        for i, layer in enumerate(self.layers):
            x = layer(x, ws[:, i + 2])
        return self.to_rgb(x)

    def forward(self, z, truncation=1.0, mix_prob=0.0):
        w = self.z_to_w(z, truncation)
        ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)
        if mix_prob > 0 and random.random() < mix_prob:
            # Style mixing: layers after a random crossover point use a second latent.
            w2 = self.z_to_w(torch.randn_like(z), truncation)
            cut = random.randint(1, self.num_ws - 1)
            ws[:, cut:] = w2.unsqueeze(1)
        return self.synthesis(ws)


def minibatch_stddev(x):
    # Append the average std-dev across the batch as an extra feature map,
    # letting D notice when G produces too little variety.
    std = torch.sqrt(x.var(0, unbiased=False) + 1e-8).mean()
    return torch.cat([x, std.expand(x.shape[0], 1, x.shape[2], x.shape[3])], dim=1)


class DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = EqualConv2d(in_ch, in_ch, 3, padding=1)
        self.conv2 = EqualConv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = lrelu(self.conv2(lrelu(self.conv1(x))))
        return F.avg_pool2d(x, 2)


class Discriminator(nn.Module):
    def __init__(self, img_size=32, channels=64):
        super().__init__()
        self.from_rgb = EqualConv2d(3, channels, 1)
        self.blocks = nn.Sequential(*[DBlock(channels, channels) for _ in range(int(math.log2(img_size)) - 2)])
        self.final_conv = EqualConv2d(channels + 1, channels, 3, padding=1)
        self.fc = EqualLinear(channels * 4 * 4, channels, activation=True)
        self.out = EqualLinear(channels, 1)

    def forward(self, x):
        x = self.blocks(lrelu(self.from_rgb(x)))
        x = lrelu(self.final_conv(minibatch_stddev(x)))
        return self.out(self.fc(x.flatten(1)))


def build_generator(a):
    return Generator(a["img_size"], a["z_dim"], a["w_dim"], a["channels"], a["mapping_layers"])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def to_tensor(img):
    """PIL RGB image -> float tensor [3, H, W] in [-1, 1]."""
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


class ShapesDataset(Dataset):
    """Toy dataset: random coloured circles, squares and triangles on a dark background.
    Small and fast to learn, so you can watch StyleGAN train on a CPU."""

    def __init__(self, size, n=10000, seed=0):
        rng = random.Random(seed)
        self.images = torch.stack([self._make(size, rng) for _ in range(n)])

    @staticmethod
    def _make(size, rng):
        s = size * 4  # draw at 4x, then downsample for anti-aliased edges
        img = Image.new("RGB", (s, s), tuple(rng.randint(0, 80) for _ in range(3)))
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randint(1, 2)):
            r = rng.uniform(0.15, 0.3) * s
            cx, cy = rng.uniform(r, s - r), rng.uniform(r, s - r)
            color = tuple(rng.randint(100, 255) for _ in range(3))
            kind = rng.choice(["circle", "square", "triangle"])
            if kind == "circle":
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            elif kind == "square":
                draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
            else:
                draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(img)).permute(2, 0, 1).contiguous()  # uint8 saves memory

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.images[i].float() / 127.5 - 1.0


class ImageFolder(Dataset):
    """Every image under `root` (searched recursively), centre-cropped to a square and resized."""

    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(self, root, size):
        self.files = sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in self.EXTS)
        if not self.files:
            raise SystemExit(f"No images found in {root}")
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        x = to_tensor(img.resize((self.size, self.size), Image.Resampling.LANCZOS))
        return x.flip(2) if random.random() < 0.5 else x  # random horizontal flip


def infinite(loader):
    while True:
        yield from loader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_grid(images, path, ncols=None):
    """Save a batch of images in [-1, 1] as one PNG grid. Small images are upscaled for viewing."""
    images = ((images.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    n, h, w, _ = images.shape
    ncols = ncols or math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    grid = np.zeros((nrows * h, ncols * w, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, ncols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    out = Image.fromarray(grid)
    zoom = max(1, 128 // h)
    if zoom > 1:
        out = out.resize((out.width * zoom, out.height * zoom), Image.Resampling.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def load_generator(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    G = build_generator(ckpt["args"]).to(device)
    G.load_state_dict(ckpt["G_ema"])
    return G.eval()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        for k in ARCH_KEYS:  # architecture must match the checkpoint
            setattr(args, k, ckpt["args"][k])

    if args.img_size < 8 or args.img_size & (args.img_size - 1):
        raise SystemExit("--img-size must be a power of two >= 8 (8, 16, 32, 64, 128, ...)")

    if args.data == "shapes":
        print("Generating toy shapes dataset...")
        dataset = ShapesDataset(args.img_size)
    else:
        dataset = ImageFolder(args.data, args.img_size)
    if len(dataset) < args.batch:
        raise SystemExit(f"Dataset has {len(dataset)} images but --batch is {args.batch}")
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers)
    data = infinite(loader)

    G = build_generator(vars(args)).to(device)
    D = Discriminator(args.img_size, args.channels).to(device)
    G_ema = copy.deepcopy(G).eval()
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.99))

    start = 1
    if ckpt:
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        G_ema.load_state_dict(ckpt["G_ema"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
        start = ckpt["step"] + 1

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    fixed_z = torch.randn(64, args.z_dim, generator=torch.Generator().manual_seed(0)).to(device)
    n_params = sum(p.numel() for p in G.parameters()) / 1e6
    print(f"Training on {device} | {len(dataset)} images | G has {n_params:.2f}M params | output -> {out}")

    def save(step):
        torch.save({
            "step": step, "args": vars(args),
            "G": G.state_dict(), "D": D.state_dict(), "G_ema": G_ema.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
        }, out / "checkpoint.pt")

    t0 = time.time()
    r1_val = 0.0
    for step in range(start, args.steps + 1):
        real = next(data).to(device)

        # ---- Discriminator: real -> high score, fake -> low score ----
        D.requires_grad_(True)
        with torch.no_grad():
            fake = G(torch.randn(args.batch, args.z_dim, device=device), mix_prob=args.mix_prob)
        d_loss = F.softplus(D(fake)).mean() + F.softplus(-D(real)).mean()
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        # Lazy R1: penalize D's gradient on real images every few steps.
        if args.r1 > 0 and step % args.r1_every == 0:
            real_r1 = real.detach().requires_grad_(True)
            grad, = torch.autograd.grad(D(real_r1).sum(), real_r1, create_graph=True)
            r1 = grad.pow(2).flatten(1).sum(1).mean()
            opt_d.zero_grad(set_to_none=True)
            (args.r1 / 2 * r1 * args.r1_every).backward()
            opt_d.step()
            r1_val = r1.item()

        # ---- Generator: make D score fakes as real ----
        D.requires_grad_(False)
        fake = G(torch.randn(args.batch, args.z_dim, device=device), mix_prob=args.mix_prob)
        g_loss = F.softplus(-D(fake)).mean()
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        # ---- EMA of generator weights (smoother, better-looking samples) ----
        with torch.no_grad():
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.lerp_(p, 1 - args.ema)
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        if step % args.log_every == 0:
            print(f"step {step:6d} | D {d_loss.item():.3f} | G {g_loss.item():.3f} | "
                  f"R1 {r1_val:.3f} | {time.time() - t0:.0f}s", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            with torch.no_grad():
                save_grid(G_ema(fixed_z), out / "samples" / f"step_{step:06d}.png")
        if step % args.save_every == 0 or step == args.steps:
            save(step)

    print(f"Done. Checkpoint: {out / 'checkpoint.pt'}")


@torch.no_grad()
def generate(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    G = load_generator(args.ckpt, device)
    z = torch.randn(args.n, G.z_dim, device=device)
    save_grid(G(z, truncation=args.truncation), args.out)
    print(f"Saved {args.n} images to {args.out}")


@torch.no_grad()
def mix(args):
    """Style-mixing grid: the top row gives coarse styles (pose, shape),
    the left column gives fine styles (colour, texture)."""
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    G = load_generator(args.ckpt, device)
    n = args.n
    cut = args.cut if args.cut is not None else G.num_ws // 2
    if not 0 < cut < G.num_ws:
        raise SystemExit(f"--cut must be between 1 and {G.num_ws - 1}")

    w_top = G.z_to_w(torch.randn(n, G.z_dim, device=device), args.truncation)
    w_left = G.z_to_w(torch.randn(n, G.z_dim, device=device), args.truncation)
    top = G.synthesis(w_top.unsqueeze(1).repeat(1, G.num_ws, 1))
    left = G.synthesis(w_left.unsqueeze(1).repeat(1, G.num_ws, 1))

    cells = [torch.ones_like(top[:1])]  # blank top-left corner
    cells += list(top.split(1))
    for i in range(n):
        cells.append(left[i:i + 1])
        ws = w_left[i].repeat(n, G.num_ws, 1)  # fine layers from the left image
        ws[:, :cut] = w_top.unsqueeze(1)       # coarse layers from the top images
        cells += list(G.synthesis(ws).split(1))
    save_grid(torch.cat(cells), args.out, ncols=n + 1)
    print(f"Saved style-mixing grid to {args.out} (layers 0-{cut - 1} of {G.num_ws} from top row)")


def main():
    p = argparse.ArgumentParser(description="Simple StyleGAN in PyTorch")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a StyleGAN")
    t.add_argument("--data", default="shapes", help="'shapes' (built-in toy dataset) or a folder of images")
    t.add_argument("--img-size", type=int, default=32, help="output resolution, power of two")
    t.add_argument("--z-dim", type=int, default=128)
    t.add_argument("--w-dim", type=int, default=128)
    t.add_argument("--channels", type=int, default=64, help="feature maps per conv layer")
    t.add_argument("--mapping-layers", type=int, default=8)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--lr", type=float, default=0.002)
    t.add_argument("--r1", type=float, default=0.1, help="R1 regularization weight (gamma)")
    t.add_argument("--r1-every", type=int, default=4)
    t.add_argument("--mix-prob", type=float, default=0.9, help="style-mixing probability")
    t.add_argument("--ema", type=float, default=0.999, help="EMA decay for the generator copy")
    t.add_argument("--out", default="runs/stylegan")
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--sample-every", type=int, default=500)
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--resume", default=None, help="checkpoint to continue from")
    t.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")

    g = sub.add_parser("generate", help="sample images from a trained model")
    g.add_argument("--ckpt", required=True)
    g.add_argument("--n", type=int, default=16)
    g.add_argument("--truncation", type=float, default=0.7, help="1.0 = no truncation")
    g.add_argument("--out", default="generated.png")

    m = sub.add_parser("mix", help="make a style-mixing grid")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--n", type=int, default=5)
    m.add_argument("--cut", type=int, default=None, help="layers taken from the top row (default: half)")
    m.add_argument("--truncation", type=float, default=0.7)
    m.add_argument("--out", default="mixing.png")

    for sp in (t, g, m):
        sp.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
        sp.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    {"train": train, "generate": generate, "mix": mix}[args.cmd](args)


if __name__ == "__main__":
    main()
