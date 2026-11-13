from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

import torch
import torchvision
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    Swin_T_Weights,
    convnext_tiny,
    swin_t,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    framework: str
    builder_identifier: str
    weight_identifier: str
    pretrained_checkpoint_url: str
    pretrained_parameter_count: int
    input_size: int
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    official_inference_resize: int
    official_inference_interpolation: str
    stage1_trainable_prefixes: tuple[str, ...]
    stage2_trainable_prefixes: tuple[str, ...]
    notes: str


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "swin_tiny": ModelSpec(
        key="swin_tiny",
        display_name="Swin Transformer Tiny",
        framework="torchvision",
        builder_identifier="torchvision.models.swin_t",
        weight_identifier="torchvision.models.Swin_T_Weights.IMAGENET1K_V1",
        pretrained_checkpoint_url=Swin_T_Weights.IMAGENET1K_V1.url,
        pretrained_parameter_count=28_288_354,
        input_size=224,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        official_inference_resize=232,
        official_inference_interpolation="bicubic",
        stage1_trainable_prefixes=("head",),
        # TorchVision Swin-T stage 4 is features.7; final LayerNorm and classifier are also unfrozen.
        stage2_trainable_prefixes=("features.7", "norm", "head"),
        notes=(
            "Official TorchVision Swin-T ImageNet-1K weights. The study protocol first resizes MRI slices "
            "directly to 224x224 using area interpolation; therefore the ImageNet normalization is retained "
            "but the official 232->224 center-crop inference recipe is not inserted, because that would alter "
            "the established MRI preprocessing geometry."
        ),
    ),
    "convnext_tiny": ModelSpec(
        key="convnext_tiny",
        display_name="ConvNeXt Tiny",
        framework="torchvision",
        builder_identifier="torchvision.models.convnext_tiny",
        weight_identifier="torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        pretrained_checkpoint_url=ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,
        pretrained_parameter_count=28_589_128,
        input_size=224,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        official_inference_resize=236,
        official_inference_interpolation="bilinear",
        stage1_trainable_prefixes=("classifier",),
        # TorchVision ConvNeXt-T stage 4 is features.7. The classifier includes final normalization.
        stage2_trainable_prefixes=("features.7", "classifier"),
        notes=(
            "Official TorchVision ConvNeXt-T ImageNet-1K weights. The established MRI protocol keeps the "
            "direct area resize to 224x224 and applies ImageNet normalization without adding a new center crop."
        ),
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown architecture {key!r}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def _replace_binary_head(model: nn.Module, key: str) -> nn.Module:
    if key == "swin_tiny":
        if not isinstance(model.head, nn.Linear):
            raise RuntimeError("Unexpected torchvision Swin-T head structure")
        model.head = nn.Linear(model.head.in_features, 1)
    elif key == "convnext_tiny":
        if not isinstance(model.classifier, nn.Sequential) or not isinstance(model.classifier[-1], nn.Linear):
            raise RuntimeError("Unexpected torchvision ConvNeXt-T classifier structure")
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    else:  # pragma: no cover
        raise KeyError(key)
    return model


def build_model(key: str, pretrained: bool = True, progress: bool = True) -> nn.Module:
    """Build a binary single-logit model.

    `pretrained=False` exists for offline unit tests only. Official experiment configs require pretrained=True.
    """
    if key == "swin_tiny":
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights, progress=progress)
    elif key == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights, progress=progress)
    else:
        raise KeyError(f"Unsupported architecture {key}")
    return _replace_binary_head(model, key)


def _set_trainable_by_prefixes(model: nn.Module, prefixes: Iterable[str]) -> None:
    prefixes = tuple(prefixes)
    for name, param in model.named_parameters():
        param.requires_grad = any(name == p or name.startswith(p + ".") for p in prefixes)


def configure_stage(model: nn.Module, key: str, stage: int) -> Dict[str, object]:
    spec = get_model_spec(key)
    if stage == 1:
        prefixes = spec.stage1_trainable_prefixes
    elif stage == 2:
        prefixes = spec.stage2_trainable_prefixes
    else:
        raise ValueError("stage must be 1 or 2")
    _set_trainable_by_prefixes(model, prefixes)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters after configuring {key} stage {stage}")
    return {
        "stage": stage,
        "trainable_prefixes": list(prefixes),
        "trainable_parameter_names": trainable,
        "n_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "n_frozen_parameters": int(sum(p.numel() for p in model.parameters() if not p.requires_grad)),
        "n_trainable_tensors": len(trainable),
        "n_frozen_tensors": len(frozen),
    }


def model_metadata(model: nn.Module, key: str) -> Dict[str, object]:
    spec = get_model_spec(key)
    return {
        **asdict(spec),
        "binary_model_parameter_count": int(sum(p.numel() for p in model.parameters())),
        "binary_model_trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }


def assert_official_pretrained_config(config: dict) -> None:
    if not bool(config.get("pretrained", True)):
        raise RuntimeError("Official runs require ImageNet-pretrained weights; pretrained=False is test-only.")
