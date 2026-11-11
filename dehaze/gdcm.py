from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorOrFloat = Union[torch.Tensor, float]


@dataclass(frozen=True)
class DetailBypassConfig:
    channels: int = 32
    levels: int = 3
    blocks_per_level: int = 2
    # Legacy checkpoints omitted the final clamp during training.
    clamp_output: bool = False

    def to_dict(self) -> Dict[str, Union[int, bool]]:
        return {
            "channels": self.channels,
            "levels": self.levels,
            "blocks_per_level": self.blocks_per_level,
            "clamp_output": self.clamp_output,
        }

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Union[int, bool]]]) -> "DetailBypassConfig":
        if not values:
            return cls()
        return cls(
            channels=int(values.get("channels", cls.channels)),
            levels=int(values.get("levels", cls.levels)),
            blocks_per_level=int(values.get("blocks_per_level", cls.blocks_per_level)),
            clamp_output=bool(values.get("clamp_output", False)),
        )


class LaplacianPyramid(nn.Module):
    """Fixed Gaussian/Laplacian decomposition with exact shape restoration."""

    def __init__(self, levels: int = 3):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be at least 1")
        kernel_1d = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
        kernel_2d = torch.outer(kernel_1d, kernel_1d)[None, None]
        self.levels = levels
        self.register_buffer("kernel", kernel_2d, persistent=False)

    def blur(self, image: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        kernel = self.kernel.to(device=image.device, dtype=image.dtype).expand(
            channels, 1, -1, -1
        )
        padded = F.pad(image, (2, 2, 2, 2), mode="reflect")
        return F.conv2d(padded, kernel, groups=channels)

    def decompose(self, image: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        bands: List[torch.Tensor] = []
        current = image
        for _ in range(self.levels):
            low = self.blur(current)[:, :, ::2, ::2]
            up = F.interpolate(low, size=current.shape[-2:], mode="bilinear", align_corners=False)
            bands.append(current - up)
            current = low
        return bands, current

    @staticmethod
    def synthesize(
        bands: List[torch.Tensor], low: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not bands:
            raise ValueError("at least one Laplacian band is required")
        if low is None:
            current = torch.zeros_like(bands[-1])
        else:
            current = low
        for band in reversed(bands):
            if current.shape[-2:] != band.shape[-2:]:
                current = F.interpolate(
                    current, size=band.shape[-2:], mode="bilinear", align_corners=False
                )
            current = current + band
        return current


class ResidualDetailBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels * 2
        self.conv_in = nn.Conv2d(channels, hidden, 3, padding=1)
        self.depthwise = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv_out = nn.Conv2d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.conv_in(features)
        residual = F.gelu(self.depthwise(residual))
        residual = self.conv_out(residual)
        return features + self.scale * residual


class ReliabilityGate(nn.Module):
    """Selects hazy-input structures that agree with the coarse output."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 2, 8)
        self.net = nn.Sequential(
            nn.Conv2d(9, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, hazy_band: torch.Tensor, coarse_band: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat(
            [hazy_band.abs(), coarse_band.abs(), hazy_band * coarse_band], dim=1
        )
        return self.net(gate_input)


class DetailBypassRefiner(nn.Module):
    """Restores VAE-suppressed details through a gated pixel-domain bypass."""

    def __init__(self, config: Optional[DetailBypassConfig] = None):
        super().__init__()
        self.config = config or DetailBypassConfig()
        if self.config.channels < 8:
            raise ValueError("channels must be at least 8")

        self.pyramid = LaplacianPyramid(self.config.levels)
        input_channels = 6 + self.config.channels
        self.gates = nn.ModuleList(
            ReliabilityGate(self.config.channels) for _ in range(self.config.levels)
        )
        self.stems = nn.ModuleList(
            nn.Conv2d(input_channels, self.config.channels, 3, padding=1)
            for _ in range(self.config.levels)
        )
        self.blocks = nn.ModuleList(
            nn.Sequential(
                *[
                    ResidualDetailBlock(self.config.channels)
                    for _ in range(self.config.blocks_per_level)
                ]
            )
            for _ in range(self.config.levels)
        )
        self.outputs = nn.ModuleList(
            nn.Conv2d(self.config.channels, 3, 3, padding=1)
            for _ in range(self.config.levels)
        )
        self.reset_output_layers()

    def reset_output_layers(self) -> None:
        for output in self.outputs:
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def predict_correction(
        self, hazy: torch.Tensor, coarse: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]:
        if hazy.shape != coarse.shape:
            raise ValueError(
                "hazy and coarse images must have identical BCHW shapes, got "
                f"{tuple(hazy.shape)} and {tuple(coarse.shape)}"
            )
        if hazy.ndim != 4 or hazy.shape[1] != 3:
            raise ValueError("DetailBypassRefiner expects BCHW RGB tensors")

        hazy_bands, _ = self.pyramid.decompose(hazy)
        coarse_bands, _ = self.pyramid.decompose(coarse)
        predicted_bands: List[Optional[torch.Tensor]] = [None] * self.config.levels
        gates: List[Optional[torch.Tensor]] = [None] * self.config.levels
        previous: Optional[torch.Tensor] = None

        for level in reversed(range(self.config.levels)):
            size = coarse_bands[level].shape[-2:]
            if previous is None:
                previous = coarse.new_zeros(
                    coarse.shape[0], self.config.channels, size[0], size[1]
                )
            else:
                previous = F.interpolate(
                    previous, size=size, mode="bilinear", align_corners=False
                )

            gate = self.gates[level](hazy_bands[level], coarse_bands[level])
            inputs = torch.cat(
                [coarse_bands[level], gate * hazy_bands[level], previous], dim=1
            )
            features = self.blocks[level](self.stems[level](inputs))
            predicted_bands[level] = self.outputs[level](features)
            gates[level] = gate
            previous = features

        typed_bands = [band for band in predicted_bands if band is not None]
        typed_gates = [gate for gate in gates if gate is not None]
        correction = self.pyramid.synthesize(typed_bands)
        return correction, {"bands": typed_bands, "gates": typed_gates}

    @staticmethod
    def _prepare_mask(mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        if mask is None:
            return reference.new_ones(reference.shape[0], 1, *reference.shape[-2:])
        if mask.ndim != 4:
            raise ValueError("mask must be a BCHW tensor")
        if mask.shape[1] != 1:
            mask = mask.mean(dim=1, keepdim=True)
        mask = F.interpolate(
            mask.to(device=reference.device, dtype=reference.dtype),
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _prepare_strength(strength: TensorOrFloat, reference: torch.Tensor) -> torch.Tensor:
        strength_tensor = torch.as_tensor(
            strength, device=reference.device, dtype=reference.dtype
        )
        while strength_tensor.ndim < 4:
            strength_tensor = strength_tensor.unsqueeze(-1)
        return strength_tensor.clamp(min=0.0)

    def forward(
        self,
        hazy: torch.Tensor,
        coarse: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        strength: TensorOrFloat = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        correction, details = self.predict_correction(hazy, coarse)
        pixel_mask = self._prepare_mask(mask, coarse)
        detail_strength = self._prepare_strength(strength, coarse)
        applied_correction = pixel_mask * detail_strength * correction
        refined = coarse + applied_correction
        if self.config.clamp_output:
            refined = refined.clamp(-1.0, 1.0)
        return refined, {
            "correction": correction,
            "applied_correction": applied_correction,
            "bands": details["bands"],
            "gates": details["gates"],
        }


@torch.no_grad()
def deterministic_vae_reconstruction(
    vae: nn.Module, image: torch.Tensor, *, clamp_output: bool = False
) -> torch.Tensor:
    """D(E_mu(J)) in Eq. (12), without clipping the decoder's reconstruction.

    The raw posterior latent goes directly to the same VAE decoder. The SD
    scaling factor is unnecessary here: scaling and unscaling would cancel.
    ``clamp_output`` only reproduces targets from the older experiment.
    """
    latent = vae.encode(image).latent_dist.mode()
    reconstruction = vae.decode(latent).sample
    return reconstruction.clamp(-1.0, 1.0) if clamp_output else reconstruction


@torch.no_grad()
def vae_detail_targets(
    clean: torch.Tensor,
    vae_reconstruction: torch.Tensor,
    pyramid: LaplacianPyramid,
) -> List[torch.Tensor]:
    """Supervise VAE compression loss in clean GT, not the dehazing residual."""
    if clean.shape != vae_reconstruction.shape:
        raise ValueError("clean and VAE reconstruction tensors must have the same shape")
    clean_bands, _ = pyramid.decompose(clean)
    vae_bands, _ = pyramid.decompose(vae_reconstruction)
    return [clean_band - vae_band for clean_band, vae_band in zip(clean_bands, vae_bands)]


def charbonnier_loss(
    prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return charbonnier_loss(pred_x, target_x) + charbonnier_loss(pred_y, target_y)


def fft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_fft = torch.fft.rfft2(prediction.float(), norm="ortho")
    target_fft = torch.fft.rfft2(target.float(), norm="ortho")
    return (pred_fft.abs() - target_fft.abs()).abs().mean()


class DetailBypassLoss(nn.Module):
    def __init__(
        self,
        final_weight: float = 1.0,
        compensation_weight: float = 0.5,
        gradient_weight: float = 0.1,
        fft_weight: float = 0.05,
        correction_weight: float = 0.01,
        eps: float = 1e-3,
    ):
        super().__init__()
        self.final_weight = final_weight
        self.compensation_weight = compensation_weight
        self.gradient_weight = gradient_weight
        self.fft_weight = fft_weight
        self.correction_weight = correction_weight
        self.eps = eps

    def forward(
        self,
        refined: torch.Tensor,
        clean: torch.Tensor,
        predicted_bands: List[torch.Tensor],
        target_bands: List[torch.Tensor],
        correction: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if len(predicted_bands) != len(target_bands):
            raise ValueError("predicted and target band counts must match")
        final = charbonnier_loss(refined, clean, self.eps)
        compensation = torch.stack(
            [
                charbonnier_loss(predicted, target, self.eps)
                for predicted, target in zip(predicted_bands, target_bands)
            ]
        ).mean()
        gradients = gradient_loss(refined, clean)
        frequency = fft_loss(refined, clean)
        correction_size = correction.abs().mean()
        total = (
            self.final_weight * final
            + self.compensation_weight * compensation
            + self.gradient_weight * gradients
            + self.fft_weight * frequency
            + self.correction_weight * correction_size
        )
        return {
            "loss": total,
            "final": final,
            "compensation": compensation,
            "gradient": gradients,
            "fft": frequency,
            "correction": correction_size,
        }


def detail_bypass_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
