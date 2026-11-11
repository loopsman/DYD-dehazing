"""Attach a fresh paper GDCM to an immutable ASD single-LoRA student."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import json
import random
import shutil
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dehaze.train_utils import write_json
from dehaze.gdcm import DetailBypassConfig, DetailBypassLoss, DetailBypassRefiner, deterministic_vae_reconstruction, vae_detail_targets
from dehaze.pisa_data import ShuffledPairs
from dehaze.pisa_single import PisaSingle
from dehaze.train_utils import finish_update, restore_rng, rng_state
from dehaze.train_utils import atomic_torch, sha

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

class Run:

    def __init__(self, config, preflight=False):
        self.c = dict(config)
        self.preflight = preflight
        if preflight:
            self.c['output'] += '-preflight'
            self.c['steps'] = 2
        self.out = Path(self.c['output'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.writer = None

    def status(self, state, **fields):
        write_json(self.out / 'status.json', dict(state=state, gdcm_updates=self.step, target_updates=self.c['steps'], base_step=self.c['base_step'], combined_step=self.c['base_step'] + self.step, pid=os.getpid(), time_unix=time.time(), **fields))

    def frozen_check(self):
        for (name, p) in self.model.named_parameters():
            if p.requires_grad or p.grad is not None or p._version != self.frozen[name]:
                raise RuntimeError('Frozen dehazing parameter changed: ' + name)

    def save(self):
        self.frozen_check()
        folder = self.out / 'checkpoints' / f'gdcm-{self.step:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for (k, v) in self.gdcm.state_dict().items()}, str(folder / 'gdcm.safetensors'))
        shutil.copy2(self.base / 'adapters.safetensors', folder / 'adapters.safetensors')
        assert sha(folder / 'adapters.safetensors') == self.c['base_sha256']
        write_json(folder / 'config.json', dict(base_config=self.bc, training=self.c, gdcm_config=self.gdcm.config.to_dict(), student_inference='original ASD RAM/sample/t=1', detail_target='P(J)-P(D(E_mu(J))); raw unclipped VAE reconstruction'))
        atomic_torch(folder / 'training.pt', dict(gdcm=self.gdcm.state_dict(), optimizer=self.opt.state_dict(), scaler=self.scaler.state_dict(), rng=rng_state(), data=self.data.state_dict(), step=self.step, config=self.c))
        write_json(folder / 'complete.json', dict(base_step=self.c['base_step'], gdcm_updates=self.step, combined_step=self.c['base_step'] + self.step, base_sha256=self.c['base_sha256'], gdcm_sha256=sha(folder / 'gdcm.safetensors'), config_sha256=sha(folder / 'config.json'), training_sha256=sha(folder / 'training.pt')))
        write_json(self.out / 'latest.json', dict(gdcm_updates=self.step, checkpoint=str(folder.resolve())))
        return folder

    @torch.no_grad()
    def identity_check(self):
        state = rng_state()
        data_state = self.data.state_dict()
        try:
            item = self.data.sample()
            x = item['hazy'][None].cuda()
            (coarse, _) = self.model(x)
            (refined, details) = self.gdcm(x, coarse, strength=1.0)
            assert torch.equal(coarse, refined)
            assert torch.count_nonzero(details['correction']) == 0
            self.frozen_check()
            write_json(self.out / 'initial_identity.json', dict(passed=True, zero_correction=True, bit_exact_coarse_preserved=True, base_sha256=self.c['base_sha256']))
        finally:
            restore_rng(state)
            self.data.load_state_dict(data_state)

    def main(self):
        c = self.c
        if (self.out / 'STOP').exists():
            self.status('stopped_by_user')
            return
        if (self.out / 'config.json').exists() and read(self.out / 'config.json') != c:
            raise ValueError('Existing experiment configuration differs')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required')
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(c['seed'])
        np.random.seed(c['seed'])
        torch.manual_seed(c['seed'])
        torch.cuda.manual_seed_all(c['seed'])
        self.base = Path(c['base_checkpoint'])
        assert read(self.base / 'complete.json')['step'] == c['base_step']
        assert sha(self.base / 'adapters.safetensors') == c['base_sha256']
        self.bc = read(c['base_config']) if c.get('base_config') else read(self.base / 'config.json')
        write_json(self.out / 'config.json', c)
        self.status('loading_model')
        self.model = PisaSingle(self.bc, training_resources=False).eval()
        self.model.load_adapters(self.base / 'adapters.safetensors')
        self.model.requires_grad_(False)
        self.frozen = {n: p._version for (n, p) in self.model.named_parameters()}
        self.gdcm = DetailBypassRefiner(DetailBypassConfig.from_dict(c['gdcm'])).cuda()
        self.opt = torch.optim.AdamW(self.gdcm.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=c['precision'] == 'fp16', init_scale=512.0)
        self.data = ShuffledPairs(self.bc['sources'], c['resolution'], c['seed'])
        self.loss_fn = DetailBypassLoss(**c['loss']).cuda()
        assert {id(p) for g in self.opt.param_groups for p in g['params']} == {id(p) for p in self.gdcm.parameters()}
        write_json(self.out / 'protocol.json', dict(base_checkpoint=str(self.base.resolve()), base_sha256=c['base_sha256'], base_step=c['base_step'], new_gdcm_updates=c['steps'], trainable_parameters=sum((p.numel() for p in self.gdcm.parameters())), frozen='All SD2.1, student single LoRA, VAE, RAM, text encoder; no joint phase', training_sources=self.bc['sources'], training_pairs=len(self.data.lookup), coarse='unchanged ASD: hazy RAM text, posterior.sample(), t=1, alpha=1, no alignment', target='unclipped D(E_mu(GT)); GT used only for supervision', output='clip(coarse+GDCM_correction,-1,1); strength=1; mask=1', loss=c['loss'], code_hashes={p: sha(p) for p in [__file__, 'dehaze/gdcm.py', 'dehaze/pisa_single.py']}))
        latest = self.out / 'latest.json'
        if latest.exists():
            checkpoint = Path(read(latest)['checkpoint'])
            complete = read(checkpoint / 'complete.json')
            assert sha(checkpoint / 'training.pt') == complete['training_sha256']
            saved = torch.load(checkpoint / 'training.pt', map_location='cpu', weights_only=False)
            assert saved['config'] == c
            self.gdcm.load_state_dict(saved['gdcm'])
            self.opt.load_state_dict(saved['optimizer'])
            self.scaler.load_state_dict(saved['scaler'])
            self.data.load_state_dict(saved['data'])
            self.step = saved['step']
            restore_rng(saved['rng'])
            del saved
        else:
            self.identity_check()
        self.writer = SummaryWriter(str(self.out / 'tensorboard'), purge_step=self.step + 1 if self.step else None, flush_secs=10)
        if not latest.exists():
            self.save()
        while self.step < c['steps']:
            if (self.out / 'STOP').exists():
                self.save()
                self.status('stopped_by_user')
                return
            tick = time.perf_counter()
            state = rng_state()
            data_state = self.data.state_dict()
            for attempt in range(12):
                if attempt:
                    restore_rng(state)
                    self.data.load_state_dict(data_state)
                self.opt.zero_grad(set_to_none=True)
                totals = {}
                diagnostics = {}
                for _ in range(c['accumulation']):
                    item = self.data.sample()
                    x = item['hazy'][None].cuda()
                    gt = item['gt'][None].cuda()
                    with torch.no_grad():
                        (coarse, _) = self.model(x)
                        recon = deterministic_vae_reconstruction(self.model.vae, gt, clamp_output=False)
                        targets = vae_detail_targets(gt, recon, self.gdcm.pyramid)
                    with torch.autocast('cuda', dtype=torch.float16, enabled=c['precision'] == 'fp16'):
                        (refined, details) = self.gdcm(x, coarse, strength=1.0)
                    losses = self.loss_fn(refined.float(), gt, [b.float() for b in details['bands']], targets, details['correction'].float())
                    assert all((torch.isfinite(v).all() for v in losses.values()))
                    self.scaler.scale(losses['loss'] / c['accumulation']).backward()
                    for (k, v) in losses.items():
                        totals[k] = totals.get(k, 0) + float(v.detach()) / c['accumulation']
                    for (k, v) in dict(correction_abs=details['correction'].abs().float().mean(), gate_mean=torch.stack([g.float().mean() for g in details['gates']]).mean()).items():
                        diagnostics[k] = diagnostics.get(k, 0) + float(v.detach()) / c['accumulation']
                (accepted, norm) = finish_update(self.opt, self.scaler, list(self.gdcm.parameters()))
                if accepted:
                    break
            else:
                raise FloatingPointError('Repeated GDCM gradient overflow')
            self.step += 1
            self.frozen_check()
            row = dict(gdcm_updates=self.step, base_step=c['base_step'], combined_step=c['base_step'] + self.step, seconds=time.perf_counter() - tick, grad_norm=norm, scaler=self.scaler.get_scale(), **totals, **diagnostics)
            with (self.out / 'train.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            for (k, v) in totals.items():
                self.writer.add_scalar('loss_raw/' + k, v, self.step)
            for (k, weight) in [('final', 'final_weight'), ('compensation', 'compensation_weight'), ('gradient', 'gradient_weight'), ('fft', 'fft_weight'), ('correction', 'correction_weight')]:
                self.writer.add_scalar('loss_weighted/' + k, totals[k] * c['loss'][weight], self.step)
            for (k, v) in dict(**diagnostics, seconds=row['seconds'], grad_norm=norm).items():
                self.writer.add_scalar('train/' + k, v, self.step)
            self.writer.flush()
            self.status('training', loss=totals['loss'], seconds_per_update=row['seconds'])
            print(json.dumps(row), flush=True)
            if self.step % c['save_every'] == 0 or self.step == c['steps'] or self.preflight:
                self.save()
        assert sha(self.base / 'adapters.safetensors') == c['base_sha256']
        if self.preflight:
            weights = load_file(str(self.out / 'checkpoints' / f'gdcm-{self.step:06d}' / 'gdcm.safetensors'))
            assert all((torch.equal(weights[n], v.detach().cpu()) for (n, v) in self.gdcm.state_dict().items()))
            assert all((float(s['step']) == self.step for s in self.opt.state.values()))
            write_json(self.out / 'preflight.json', dict(passed=True, cuda=torch.cuda.get_device_name(), updates=self.step, accumulation=c['accumulation'], frozen_asd_unchanged=True, zero_identity=read(self.out / 'initial_identity.json'), optimizer_counters_verified=True, checkpoint_roundtrip=True, finite_losses_and_nonzero_gradients=True))
        self.status('complete')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/train_gdcm.json')
    p.add_argument('--preflight', action='store_true')
    args = p.parse_args()
    run = Run(read(args.config), args.preflight)
    try:
        run.main()
    except BaseException:
        run.status('failed', error=traceback.format_exc())
        raise
    finally:
        if run.writer:
            run.writer.close()
if __name__ == '__main__':
    main()
