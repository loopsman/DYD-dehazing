"""ASDSR Eq. 8/9 primitives, with explicit stop-gradient boundaries."""
import torch
import torch.nn.functional as F


def asd_direction(teacher_uncond, teacher_cond, auxiliary_uncond, auxiliary_cond, cfg):
    """Epsilon coordinates; w(t)=1. The returned direction is not differentiated."""
    return (cfg * (teacher_cond.float() - teacher_uncond.float())
            - (auxiliary_cond.float() - auxiliary_uncond.float())).detach()


def direction_surrogate(latent, direction):
    """Mean MSE injects 2*direction/numel, consistent with the host PiSA proxy."""
    return F.mse_loss(latent.float(), (latent.float() - direction).detach())


def auxiliary_objective(fake_prediction, known_noise, real_prediction, real_teacher, gamma):
    diffusion = F.mse_loss(fake_prediction.float(), known_noise.detach().float())
    gt_separation = F.mse_loss(real_prediction.float(), real_teacher.detach().float())
    return diffusion + gamma * gt_separation, diffusion, gt_separation
