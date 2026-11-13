from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class AugmentationSpec:
    rotation_radians: float = 0.025
    translation_fraction: float = 0.04
    zoom_fraction: float = 0.06
    contrast_fraction: float = 0.10

    @property
    def rotation_degrees(self) -> float:
        return math.degrees(self.rotation_radians)

    @property
    def scale_min(self) -> float:
        return 1.0 - self.zoom_fraction

    @property
    def scale_max(self) -> float:
        return 1.0 + self.zoom_fraction

    @property
    def contrast_min(self) -> float:
        return 1.0 - self.contrast_fraction

    @property
    def contrast_max(self) -> float:
        return 1.0 + self.contrast_fraction


DEFAULT_AUGMENTATION = AugmentationSpec()


class ProtocolAugment:
    """Study-locked affine/contrast augmentation with an explicit Torch generator.

    Input is a float tensor in [0,1], shape CxHxW. The same sampled transform is
    applied to all channels. Parameters exactly encode the study protocol:
    rotation +/-0.025 rad, x/y translation +/-4%, isotropic scale 0.94..1.06,
    and contrast 0.90..1.10.
    """

    def __init__(self, spec: AugmentationSpec = DEFAULT_AUGMENTATION, interpolation=InterpolationMode.BILINEAR):
        self.spec = spec
        self.interpolation = interpolation

    def sample_parameters(self, h: int, w: int, generator: torch.Generator) -> Dict[str, float | tuple[int, int]]:
        u = torch.rand(4, generator=generator)
        angle = (float(u[0]) * 2.0 - 1.0) * self.spec.rotation_degrees
        tx = int(round((float(u[1]) * 2.0 - 1.0) * self.spec.translation_fraction * w))
        ty = int(round((float(u[2]) * 2.0 - 1.0) * self.spec.translation_fraction * h))
        scale = self.spec.scale_min + float(u[3]) * (self.spec.scale_max - self.spec.scale_min)
        # Draw contrast independently so changing affine implementation cannot shift contrast RNG state.
        c = torch.rand(1, generator=generator).item()
        contrast = self.spec.contrast_min + c * (self.spec.contrast_max - self.spec.contrast_min)
        return {"angle_degrees": angle, "translate_pixels": (tx, ty), "scale": scale, "contrast": contrast}

    def __call__(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected CxHxW tensor, received {tuple(x.shape)}")
        _, h, w = x.shape
        p = self.sample_parameters(h, w, generator)
        out = TF.affine(
            x,
            angle=float(p["angle_degrees"]),
            translate=list(p["translate_pixels"]),
            scale=float(p["scale"]),
            shear=[0.0, 0.0],
            interpolation=self.interpolation,
            fill=0.0,
        )
        out = TF.adjust_contrast(out, float(p["contrast"]))
        return out.clamp_(0.0, 1.0)


def augmentation_audit_dict(spec: AugmentationSpec = DEFAULT_AUGMENTATION) -> Dict[str, float]:
    return {
        "rotation_radians_min": -spec.rotation_radians,
        "rotation_radians_max": spec.rotation_radians,
        "rotation_degrees_min": -spec.rotation_degrees,
        "rotation_degrees_max": spec.rotation_degrees,
        "translation_fraction_x_min": -spec.translation_fraction,
        "translation_fraction_x_max": spec.translation_fraction,
        "translation_fraction_y_min": -spec.translation_fraction,
        "translation_fraction_y_max": spec.translation_fraction,
        "scale_min": spec.scale_min,
        "scale_max": spec.scale_max,
        "contrast_min": spec.contrast_min,
        "contrast_max": spec.contrast_max,
    }
