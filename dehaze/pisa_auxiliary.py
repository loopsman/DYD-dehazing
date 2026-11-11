"""Single-LoRA dehazer with a training-only online diffusion LoRA.

The frozen teacher and online model share immutable pretrained weights. Disabling
the distribution adapter evaluates phi; enabling it evaluates psi. No graph may
span a switch of this adapter. All trainable parameters stay FP32 under AMP.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from peft import LoraConfig
from safetensors.torch import save_file
from .pisa_single import PisaSingle, target_modules
from .auxiliary_score import asd_direction, direction_surrogate, auxiliary_objective


class AuxiliaryPisa(PisaSingle):
    def __init__(self, config):
        super().__init__(config, training_resources=True)
        self.teacher.add_adapter(LoraConfig(r=config['auxiliary_rank'], lora_alpha=8,
            init_lora_weights='gaussian', target_modules=target_modules(self.teacher)),
            adapter_name='distribution')
        for name, p in self.teacher.named_parameters():
            p.requires_grad_('lora_' in name)
            if p.requires_grad:
                p.data = p.data.float()
        self.teacher.enable_gradient_checkpointing()
        self.time_rng = torch.Generator(device='cuda').manual_seed(config['seed'] + 101)
        self.noise_rng = torch.Generator(device='cuda').manual_seed(config['seed'] + 202)
        self.teacher.eval()

    def auxiliary_parameters(self):
        return [p for name, p in self.teacher.named_parameters() if 'lora_' in name]

    def auxiliary_weights(self):
        return {n: p.detach().cpu().clone() for n, p in self.teacher.named_parameters() if 'lora_' in n}

    def load_auxiliary_weights(self, weights):
        named = {n: p for n, p in self.teacher.named_parameters() if 'lora_' in n}
        if named.keys() != weights.keys():
            raise ValueError('Auxiliary adapter key mismatch')
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(weights[n])

    def save_auxiliary(self, path):
        save_file({n: p.contiguous() for n, p in self.auxiliary_weights().items()}, str(path))

    def sampling_state(self):
        return dict(time_rng=self.time_rng.get_state(), noise_rng=self.noise_rng.get_state())

    def load_sampling_state(self, state):
        self.time_rng.set_state(state['time_rng'])
        self.noise_rng.set_state(state['noise_rng'])

    @contextmanager
    def frozen_teacher(self):
        self.teacher.eval()
        self.teacher.disable_adapters()
        try:
            yield
        finally:
            self.teacher.enable_adapters()

    def predict_score(self, noisy, timestep, text):
        with torch.autocast('cuda', dtype=torch.float16):
            return self.teacher(noisy.half(), timestep,
                                encoder_hidden_states=text.half()).sample.float()

    def student_losses(self, image, target):
        captions = [f'{caption}, ' for caption in self.captions(target)]
        output, aux = self(image, captions)
        mse = F.mse_loss(output.float(), target.float())
        lpips = self.perceptual(output.float(), target.float()).mean()
        z = aux['latent']
        with torch.no_grad():
            hq = self.vae.encode(target.float()).latent_dist.mode() * self.vae.config.scaling_factor
            tau = torch.randint(self.config['teacher_t_min'], self.config['teacher_t_max'],
                (z.shape[0],), generator=self.time_rng, device=z.device)
            noise = torch.randn(z.shape, device=z.device, dtype=torch.float32, generator=self.noise_rng)
            noisy_fake = self.scheduler.add_noise(z.detach().float(), noise, tau)
            noisy_real = self.scheduler.add_noise(hq.float(), noise, tau)
            empty = self.encode_text([''] * z.shape[0])
            text = torch.cat([empty, aux['positive']])
            paired_time = torch.cat([tau, tau])
            paired_fake = torch.cat([noisy_fake, noisy_fake])
            with self.frozen_teacher():
                tu, tc = self.predict_score(paired_fake, paired_time, text).chunk(2)
                real_teacher = self.predict_score(noisy_real, tau, aux['positive'])
            self.teacher.eval()
            au, ac = self.predict_score(paired_fake, paired_time, text).chunk(2)
            direction = asd_direction(tu, tc, au, ac, self.config['teacher_cfg'])
            td, ad = tc-tu, ac-au
            cosine = F.cosine_similarity(td.flatten(1), ad.flatten(1), dim=1).mean()
            stats = dict(timestep=tau.float().mean(), teacher_contrast_rms=td.square().mean().sqrt(),
                auxiliary_contrast_rms=ad.square().mean().sqrt(), direction_rms=direction.square().mean().sqrt(),
                teacher_auxiliary_cosine=cosine, auxiliary_vs_teacher_mse=F.mse_loss(ac, tc))
            pack = dict(noisy_fake=noisy_fake.detach(), noisy_real=noisy_real.detach(),
                timestep=tau, noise=noise, positive=aux['positive'].detach(), real_teacher=real_teacher)
            monitor = dict(lq=image[0].detach().add(1).div(2), gt=target[0].detach().add(1).div(2),
                student=output[0].detach().add(1).div(2))
        proxy = direction_surrogate(z, direction)
        total = self.config['lambda_mse']*mse + self.config['lambda_lpips']*lpips + self.config['lambda_asd']*proxy
        return dict(total=total, mse=mse, lpips=lpips, asd_proxy=proxy), pack, stats, monitor

    def auxiliary_loss(self, pack):
        # Enable checkpointing only for psi's differentiable forward. Backward
        # must finish before frozen_teacher() can disable this adapter again.
        self.teacher.train()
        pred_fake, pred_real = self.predict_score(
            torch.cat([pack['noisy_fake'], pack['noisy_real']]),
            torch.cat([pack['timestep'], pack['timestep']]),
            torch.cat([pack['positive'], pack['positive']])).chunk(2)
        total, diffusion, separation = auxiliary_objective(pred_fake, pack['noise'], pred_real,
            pack['real_teacher'], self.config['auxiliary_gamma'])
        return dict(total=total, diffusion=diffusion, gt_separation=separation,
                    gt_weighted=self.config['auxiliary_gamma'] * separation)
