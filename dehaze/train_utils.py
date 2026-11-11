"""Checkpoint, random-state and optimizer helpers for training."""


import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def finish_update(optimizer, scaler, parameters):
    """Never step nonfinite gradients. GradScaler lowers its scale on a rejected update."""
    scaler.unscale_(optimizer)
    finite = all((p.grad is None or torch.isfinite(p.grad).all().item() for p in parameters))
    if not finite:
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scaler.get_scale() >= old_scale:
            raise RuntimeError('GradScaler failed to reject overflow')
        return (False, None)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    if float(norm) == 0:
        raise RuntimeError('Zero adapter gradient')
    scaler.step(optimizer)
    scaler.update()
    return (True, float(norm))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)
