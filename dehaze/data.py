from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def images(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.iterdir() if p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f"No images in {root}")
    return files


def pairs(root):
    root = Path(root)
    targets = {}
    for p in images(root / "gt"):
        if p.stem in targets:
            raise ValueError(f"Ambiguous ground truth: {p.stem}")
        targets[p.stem] = p
    result = []
    for p in images(root / "hazy"):
        key = p.stem if p.stem in targets else p.stem.rsplit("_", 1)[0]
        if key not in targets:
            raise ValueError(f"Missing ground truth for {p}")
        result.append((p, targets[key]))
    return result


def open_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def tensor(im):
    return torch.from_numpy(np.array(im, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1


def pil(x):
    a = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(a.permute(1, 2, 0).numpy())


class PairedHaze(Dataset):
    def __init__(self, root, resolution):
        self.files = pairs(root)
        self.resolution = resolution

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        hp, gp = self.files[index]
        h, g = open_rgb(hp), open_rgb(gp)
        if h.size != g.size:
            raise ValueError(f"Unaligned pair: {hp} {h.size} / {gp} {g.size}")
        r = self.resolution
        if min(h.size) < r:
            scale = r / min(h.size)
            size = tuple(max(r, round(s * scale)) for s in h.size)
            h, g = h.resize(size, Image.Resampling.BICUBIC), g.resize(size, Image.Resampling.BICUBIC)
        left, top = random.randint(0, h.width - r), random.randint(0, h.height - r)
        box = (left, top, left + r, top + r)
        h, g = h.crop(box), g.crop(box)
        if random.random() < 0.5:
            h, g = ImageOps.mirror(h), ImageOps.mirror(g)
        return {"hazy": tensor(h), "gt": tensor(g), "name": hp.name}
