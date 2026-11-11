"""Controllable image/folder inference with one LoRA and GDCM."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('strength must be finite and nonnegative')
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Folder with config.json, adapters.safetensors and gdcm.safetensors')
    parser.add_argument('--input', type=Path, required=True, help='Image or image directory')
    parser.add_argument('--output', type=Path, required=True, help='PNG file or output directory')
    parser.add_argument('--alpha', type=nonnegative, default=1.0, help='Dehazing strength')
    parser.add_argument('--beta', type=nonnegative, default=1.0, help='Detail strength; 0 disables GDCM')
    parser.add_argument('--mask', type=Path, help='Grayscale mask; white edits, black suppresses corrections')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--model', help='Override the local SD2.1-base directory')
    parser.add_argument('--ram-weights', help='Override the local RAM checkpoint')
    parser.add_argument('--ram-tokenizer', help='Override the local BERT tokenizer directory')
    return parser.parse_args()


def input_paths(path):
    if path.is_file() and path.suffix.lower() in EXTENSIONS:
        return [path]
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if paths:
            names = [p.stem.casefold() for p in paths]
            if len(names) != len(set(names)):
                raise ValueError('Input filenames must have unique stems to save PNG outputs')
            return paths
    raise ValueError(f'No supported input images: {path}')


def main():
    args = parse_args()
    paths = input_paths(args.input)
    output_file = args.output.suffix.lower() == '.png' and not args.output.is_dir()
    if output_file and args.input.is_dir():
        raise ValueError('Folder input requires an output directory')
    destinations = [args.output if output_file else args.output/(p.stem+'.png') for p in paths]
    if set(p.resolve() for p in paths) & set(p.resolve() for p in destinations):
        raise ValueError('Choose an output path that does not overwrite an input image')
    if args.mask and (not args.mask.is_file() or args.input.is_dir()):
        raise ValueError('--mask requires a mask file and a single input image')

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps
    from safetensors.torch import load_file
    from dehaze.data import open_rgb, tensor, pil
    from dehaze.gdcm import DetailBypassConfig, DetailBypassRefiner
    from dehaze.pisa_single import PisaSingle
    from dehaze.train_utils import sha

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = json.loads((args.checkpoint/'config.json').read_text(encoding='utf-8-sig'))
    base_config = dict(config['base_config'])
    for key, value in [('model', args.model), ('ram_weights', args.ram_weights),
                       ('ram_tokenizer', args.ram_tokenizer)]:
        if value is not None:
            base_config[key] = value
    complete_path = args.checkpoint/'complete.json'
    if complete_path.exists():
        complete = json.loads(complete_path.read_text(encoding='utf-8-sig'))
        for name, key in [('adapters.safetensors', 'base_sha256'),
                          ('gdcm.safetensors', 'gdcm_sha256'), ('config.json', 'config_sha256')]:
            if complete.get(key) and sha(args.checkpoint/name) != complete[key]:
                raise ValueError(f'Checkpoint checksum mismatch: {name}')

    model = PisaSingle(base_config, training_resources=False).eval()
    model.load_adapters(args.checkpoint/'adapters.safetensors')
    model.requires_grad_(False)
    gdcm = DetailBypassRefiner(DetailBypassConfig.from_dict(config['gdcm_config'])).cuda().eval()
    gdcm.load_state_dict(load_file(str(args.checkpoint/'gdcm.safetensors')), strict=True)
    gdcm.requires_grad_(False)

    for index, (path, destination) in enumerate(zip(paths, destinations)):
        torch.manual_seed(args.seed+index)
        torch.cuda.manual_seed_all(args.seed+index)
        source = open_rgb(path)
        width, height = source.size
        padding = (0, max(32, math.ceil(width/8)*8)-width,
                   0, max(32, math.ceil(height/8)*8)-height)
        x = F.pad(tensor(source)[None].cuda(), padding, mode='replicate')
        mask = None
        if args.mask:
            with Image.open(args.mask) as raw:
                mask_image = ImageOps.exif_transpose(raw).convert('L')
            if mask_image.size != source.size:
                raise ValueError('Mask and input image dimensions must match')
            mask = torch.from_numpy(np.array(mask_image, dtype=np.float32)/255)[None, None].cuda()
            mask = F.pad(mask, padding, mode='replicate')
        with torch.no_grad():
            coarse, _ = model(x, alpha=args.alpha, mask=mask)
            final = coarse if args.beta == 0 else gdcm(
                x.float(), coarse.float(), mask=mask, strength=args.beta)[0]
            final = final[0, :, :height, :width]
            if not torch.isfinite(final).all():
                raise FloatingPointError(f'Nonfinite output: {path}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            pil(final).save(destination)
        print(f'{path} -> {destination}', flush=True)


if __name__ == '__main__':
    main()
