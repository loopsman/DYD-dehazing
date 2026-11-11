"""Audited paired sources with dataset-balanced sampling and shared augmentations."""
from pathlib import Path
import random
from PIL import Image
from .data import PairedHaze, images, pairs


def source_pairs(source):
    if source['name'] == 'dehaze1k':
        return pairs(source['root'])
    targets = {p.stem: p for p in images(source['gt'])}
    result = []
    for p in images(source['hazy']):
        keys = [p.stem, p.stem.removesuffix('_hazy') + '_GT']
        matches = {targets[k] for k in keys if k in targets}
        if len(matches) != 1:
            raise ValueError(f'Ambiguous or missing paired GT: {p}')
        result.append((p, matches.pop()))
    return result


class MixedHaze:
    def __init__(self, sources, resolution):
        self.sources = sources
        self.datasets = []
        for source in sources:
            data = PairedHaze.__new__(PairedHaze)
            data.files = source_pairs(source)
            data.resolution = resolution
            self.datasets.append(data)

    def sample(self):
        index = random.randrange(len(self.datasets))
        data = self.datasets[index]
        item = data[random.randrange(len(data))]
        item['source'] = self.sources[index]['name']
        return item

    def audit(self):
        records = []
        for source, data in zip(self.sources, self.datasets):
            sizes = {}
            entries = []
            for h, g in data.files:
                if g not in sizes:
                    with Image.open(g) as im:
                        sizes[g] = im.size
                with Image.open(h) as im:
                    if im.size != sizes[g]:
                        raise ValueError(f'Unaligned pair: {h} / {g}')
                    size = im.size
                entries.append(dict(hazy=str(h), gt=str(g), size=list(size)))
            records.append(dict(name=source['name'], pairs=len(entries), unique_gt=len(sizes), entries=entries))
        return records
