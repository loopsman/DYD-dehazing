"""Alternate student/online-auxiliary updates from the common step-3000 state."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dehaze.pisa_auxiliary import AuxiliaryPisa
from dehaze.pisa_data import ShuffledPairs
from dehaze.train_utils import write_json
from dehaze.train_utils import rng_state, restore_rng
from dehaze.train_utils import atomic_torch, sha

def paired_update(optimizers, scalers, groups):
    """Reject the complete alternating window if either gradient overflows."""
    for (optimizer, scaler) in zip(optimizers, scalers):
        scaler.unscale_(optimizer)
    finite = all((p.grad is None or torch.isfinite(p.grad).all().item() for group in groups for p in group))
    if not finite:
        for (optimizer, scaler) in zip(optimizers, scalers):
            scaler.update(new_scale=scaler.get_scale() * 0.5)
            optimizer.zero_grad(set_to_none=True)
        return (False, None)
    norms = [float(torch.nn.utils.clip_grad_norm_(group, 1.0, error_if_nonfinite=True)) for group in groups]
    if min(norms) <= 0:
        raise RuntimeError(f'An optimizer received zero gradients: {norms}')
    for (optimizer, scaler) in zip(optimizers, scalers):
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return (True, norms)

def save_checkpoint(out, model, optimizers, scalers, data, step, config):
    folder = out / 'checkpoints' / f'step-{step:06d}'
    folder.mkdir(parents=True, exist_ok=True)
    model.save_adapters(folder / 'adapters.safetensors')
    model.save_auxiliary(folder / 'auxiliary.safetensors')
    state = dict(step=step, weights={n: p.detach().cpu().clone() for (n, p) in model.unet.named_parameters() if p.requires_grad}, auxiliary_weights=model.auxiliary_weights(), optimizer=optimizers[0].state_dict(), auxiliary_optimizer=optimizers[1].state_dict(), scaler=scalers[0].state_dict(), auxiliary_scaler=scalers[1].state_dict(), sampling=model.sampling_state(), rng=rng_state(), data=data.state_dict(), config=config)
    atomic_torch(folder / 'training.pt', state)
    write_json(folder / 'config.json', config)
    write_json(folder / 'complete.json', dict(step=step, sha256=sha(folder / 'adapters.safetensors'), auxiliary_sha256=sha(folder / 'auxiliary.safetensors'), training_sha256=sha(folder / 'training.pt')))
    write_json(out / 'latest.json', dict(step=step, checkpoint=str(folder.resolve())))
    return folder

def restore_checkpoint(path, model, optimizers, scalers, data, common=False):
    state = torch.load(path / 'training.pt', map_location='cpu', weights_only=False)
    model.load_adapter_state(state['weights'])
    optimizers[0].load_state_dict(state['optimizer'])
    scalers[0].load_state_dict(state['scaler'])
    if common:
        model.load_sampling_state(state['auxiliary'])
    else:
        model.load_auxiliary_weights(state['auxiliary_weights'])
        optimizers[1].load_state_dict(state['auxiliary_optimizer'])
        scalers[1].load_state_dict(state['auxiliary_scaler'])
        model.load_sampling_state(state['sampling'])
    data.load_state_dict(state['data'])
    restore_rng(state['rng'])
    return int(state['step'])

def status(out, config, step, state, **extra):
    write_json(out / 'status.json', dict(state=state, step=step, target_steps=config['steps'], pid=os.getpid(), time_unix=time.time(), **extra))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_dehazing.json')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-output')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    if args.smoke:
        config['output'] = args.smoke_output or config['output'] + '-smoke'
        config['steps'] = config['resume_step'] + 2
    out = Path(config['output'])
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'STOP').exists():
        raise RuntimeError('STOP marker exists; choose a new run or remove it deliberately')
    if (out / 'latest.json').exists() and (not args.resume):
        raise RuntimeError('Output already contains a checkpoint; use --resume or a new output')
    checkpoint = Path(args.resume or config['resume_checkpoint'])
    if not args.resume and sha(checkpoint / 'training.pt') != config['initial_checkpoint_sha256']:
        raise ValueError('Common checkpoint hash mismatch')
    if args.resume:
        saved = json.loads((checkpoint / 'config.json').read_text(encoding='utf-8'))
        if saved != config:
            raise ValueError('Resume configuration changed')
    assert config['alpha'] == 1 and config['timestep'] == 1 and (config['align_method'] == 'nofix')
    assert config['auxiliary_weighting'] == 'one' and config['auxiliary_updates_per_student'] == 1
    assert config['batch_size'] == 1 and config['precision'] == 'fp16'
    step = config['resume_step']
    writer = None
    try:
        status(out, config, step, 'initializing')
        write_json(out / 'config.json', config)
        print('INITIALIZING', str(out.resolve()), flush=True)
        torch.set_num_threads(4)
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        data = ShuffledPairs(config['sources'], config['resolution'], config['seed'])
        model = AuxiliaryPisa(config)
        groups = [model.adapter_parameters(), model.auxiliary_parameters()]
        optimizers = [torch.optim.AdamW(group, lr=lr, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-08) for (group, lr) in zip(groups, [config['learning_rate'], config['auxiliary_learning_rate']])]
        scalers = [torch.cuda.amp.GradScaler(), torch.cuda.amp.GradScaler(init_scale=1024.0)]
        step = restore_checkpoint(checkpoint, model, optimizers, scalers, data, common=not args.resume)
        if not args.resume and step != config['resume_step']:
            raise ValueError('Common checkpoint step mismatch')
        frozen = {n: p._version for (n, p) in model.named_parameters() if not p.requires_grad}
        writer = SummaryWriter(str(out / 'tensorboard'), purge_step=step + 1, flush_secs=10)
        writer.add_text('protocol/method', 'ASDSR Eq8/9 adapted to dehazing: w(t)=1; gamma=-1; empty unconditional text; single student LoRA, online auxiliary LoRA sharing only frozen teacher base; simultaneous accumulated 1:1 updates from previous psi; MSE+2LPIPS; no CLIP-SCL.', step)
        code = ['dehaze/auxiliary_score.py', 'dehaze/pisa_auxiliary.py', 'scripts/train_dehazing.py', 'dehaze/pisa_single.py', 'dehaze/pisa_data.py']
        protocol = dict(start_checkpoint=str(checkpoint.resolve()), start_sha256=sha(checkpoint / 'training.pt'), method='ASDSR Eq8/9 dehazing adaptation', code_hashes={p: sha(p) for p in code}, student_trainable=sum((p.numel() for p in groups[0])), auxiliary_trainable=sum((p.numel() for p in groups[1])), pairing='shared timestep/noise for student and GT; posterior.mode GT', surrogate='MSE(z, stopgrad(z-g)); gradient 2g/numel', compared_to_csd='online auxiliary; empty rather than negative text; epsilon w(t)=1 rather than PiSA x0 normalization', excluded=['GDCM', 'TSM', 'spatial masks', 'gradient projection', 'CLIP semantic loss'])
        if args.resume:
            previous_protocol = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
            if previous_protocol['code_hashes'] != protocol['code_hashes']:
                raise ValueError('Training code changed since this run; use a new experiment directory')
        else:
            write_json(out / 'protocol.json', protocol)
        if not args.resume:
            save_checkpoint(out, model, optimizers, scalers, data, step, config)
        rejected = 0
        verified = False
        while step < config['steps']:
            if (out / 'STOP').exists():
                save_checkpoint(out, model, optimizers, scalers, data, step, config)
                status(out, config, step, 'stopped_by_user')
                return
            (before_rng, before_data, before_sampling) = (rng_state(), data.state_dict(), model.sampling_state())
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            before = [[p.detach().clone() for p in group] for group in groups] if not verified else None
            model.train()
            model.eval_frozen()
            (totals, diagnostics) = ({}, {})
            samples = hashlib.sha256()
            started = time.perf_counter()
            for micro in range(config['accumulation']):
                item = data.sample()
                samples.update((item['source'] + str(item['name'])).encode('utf-8'))
                samples.update(item['hazy'].contiguous().numpy().tobytes())
                (image, target) = (item['hazy'][None].cuda(), item['gt'][None].cuda())
                (losses, pack, stats, monitor) = model.student_losses(image, target)
                if not all((torch.isfinite(v).all() for v in losses.values())):
                    raise FloatingPointError('Nonfinite student objective')
                scalers[0].scale(losses['total'] / config['accumulation']).backward()
                check_isolation = not verified and micro == 0
                if check_isolation:
                    if any((p.grad is not None for p in groups[1])):
                        raise RuntimeError('Student guidance leaked gradients to auxiliary')
                    student_grad_before_aux = [p.grad.detach().clone() if p.grad is not None else None for p in groups[0]]
                auxiliary = model.auxiliary_loss(pack)
                if not all((torch.isfinite(v).all() for v in auxiliary.values())):
                    raise FloatingPointError('Nonfinite auxiliary objective')
                scalers[1].scale(auxiliary['total'] / config['accumulation']).backward()
                if check_isolation:
                    if not all((p.grad is None and old is None or (p.grad is not None and old is not None and torch.equal(p.grad, old)) for (p, old) in zip(groups[0], student_grad_before_aux))):
                        raise RuntimeError('Auxiliary regression leaked gradients to student')
                    del student_grad_before_aux
                for (prefix, values) in [('student', losses), ('auxiliary', auxiliary)]:
                    for (name, value) in values.items():
                        key = prefix + '/' + name
                        totals[key] = totals.get(key, 0.0) + float(value.detach()) / config['accumulation']
                for (name, value) in stats.items():
                    diagnostics[name] = diagnostics.get(name, 0.0) + float(value) / config['accumulation']
                if micro == config['accumulation'] - 1 and (not verified or (step + 1) % 100 == 0):
                    for (name, value) in monitor.items():
                        writer.add_image('images/' + name, value.cpu().clamp(0, 1), step + 1)
                    writer.add_text('images/sample', item['source'] + '/' + str(item['name']), step + 1)
                del losses, auxiliary, image, target, item, monitor
            (accepted, norms) = paired_update(optimizers, scalers, groups)
            if not accepted:
                rejected += 1
                print('REJECTED_AMP_WINDOW', step, [s.get_scale() for s in scalers], flush=True)
                restore_rng(before_rng)
                data.load_state_dict(before_data)
                model.load_sampling_state(before_sampling)
                if rejected > 12:
                    raise FloatingPointError('Repeated AMP overflow')
                continue
            rejected = 0
            for (name, p) in model.named_parameters():
                if name in frozen and (p._version != frozen[name] or p.grad is not None):
                    raise RuntimeError('Frozen parameter changed or received a gradient: ' + name)
            if not verified:
                with torch.no_grad(), model.frozen_teacher():
                    teacher_again = model.predict_score(pack['noisy_real'], pack['timestep'], pack['positive'])
                drift = float((teacher_again - pack['real_teacher']).abs().max())
                changes = [sum((float((p - old).square().sum()) for (p, old) in zip(group, old_group))) ** 0.5 for (group, old_group) in zip(groups, before)]
                if drift != 0 or min(changes) <= 0:
                    raise RuntimeError(f'Update-isolation check failed: phi drift={drift}, changes={changes}')
                write_json(out / 'update_verification.json', dict(teacher_max_drift=drift, student_delta_norm=changes[0], auxiliary_delta_norm=changes[1], initial_auxiliary_teacher_mse=diagnostics['auxiliary_vs_teacher_mse'], frozen_parameter_versions_unchanged=True, finite_gradient_norms=norms, student_auxiliary_backward_isolated=True))
                verified = True
                del before, teacher_again
            del pack
            step += 1
            seconds = time.perf_counter() - started
            for (name, value) in totals.items():
                writer.add_scalar('loss/' + name, value, step)
            for (name, value) in diagnostics.items():
                writer.add_scalar('guidance/' + name, value, step)
            for (index, name) in enumerate(('student', 'auxiliary')):
                writer.add_scalar('optimization/' + name + '_grad_norm', norms[index], step)
                writer.add_scalar('optimization/' + name + '_amp_scale', scalers[index].get_scale(), step)
                writer.add_scalar('optimization/' + name + '_lr', optimizers[index].param_groups[0]['lr'], step)
            writer.add_scalar('performance/optimizer_step_seconds', seconds, step)
            writer.add_scalar('performance/peak_vram_gb', torch.cuda.max_memory_allocated() / 1024 ** 3, step)
            row = dict(step=step, seconds=seconds, losses=totals, guidance=diagnostics, gradient_norms=norms, sample_sha256=samples.hexdigest())
            with (out / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            status(out, config, step, 'training', last_step_seconds=seconds)
            stop = (out / 'STOP').exists()
            if step % config['save_every'] == 0 or step in (config['resume_step'] + 1, config['steps']) or stop:
                folder = save_checkpoint(out, model, optimizers, scalers, data, step, config)
            if stop:
                status(out, config, step, 'stopped_by_user')
                return
        if args.smoke:
            saved_student = [p.detach().clone() for p in groups[0]]
            saved_aux = [p.detach().clone() for p in groups[1]]
            counters = [[float(s['step']) for s in opt.state.values()] for opt in optimizers]
            saved_scales = [s.state_dict() for s in scalers]
            saved_order = data.state_dict()
            saved_sampling = model.sampling_state()
            with torch.no_grad():
                groups[0][0].add_(1)
                groups[1][0].add_(1)
            loaded_step = restore_checkpoint(folder, model, optimizers, scalers, data)
            assert loaded_step == step
            assert all((torch.equal(p, old) for (p, old) in zip(groups[0], saved_student)))
            assert all((torch.equal(p, old) for (p, old) in zip(groups[1], saved_aux)))
            assert counters == [[float(s['step']) for s in opt.state.values()] for opt in optimizers]
            assert saved_scales == [s.state_dict() for s in scalers]
            assert saved_order['order'] == data.order and saved_order['cursor'] == data.cursor
            assert all((torch.equal(v, model.sampling_state()[k]) for (k, v) in saved_sampling.items()))
            state = rng_state()
            model.eval()
            with torch.no_grad():
                image = data.sample()['hazy'][None].cuda()
                (prediction, _) = model(image)
                assert prediction.shape == image.shape and torch.isfinite(prediction).all()
            restore_rng(state)
            write_json(out / 'smoke_verified.json', dict(student_auxiliary_checkpoint_roundtrip=True, optimizer_counters_restored=True, scaler_sampling_and_data_restored=True, finite_inference=True, two_optimizer_updates=True, step=step))
        status(out, config, step, 'complete')
    except BaseException as error:
        status(out, config, step, 'failed', error=repr(error))
        raise
    finally:
        if writer is not None:
            writer.close()
if __name__ == '__main__':
    main()
