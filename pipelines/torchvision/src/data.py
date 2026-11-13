from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import ProtocolAugment
from .models import get_model_spec


class MRISliceDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        data_root: str | Path,
        architecture: str,
        training: bool,
        base_seed: int,
        augment: Optional[ProtocolAugment] = None,
        verify_files: bool = True,
    ):
        required = {"relative_path", "anonymous_subject_id", "class_label"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Dataset frame missing columns: {sorted(missing)}")
        self.frame = frame.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.architecture = architecture
        self.spec = get_model_spec(architecture)
        self.training = bool(training)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.augment = augment or ProtocolAugment()
        self.verify_files = bool(verify_files)
        if self.verify_files:
            missing_paths = [str(self.data_root / p) for p in self.frame.relative_path if not (self.data_root / p).is_file()]
            if missing_paths:
                preview = "\n".join(missing_paths[:10])
                raise FileNotFoundError(f"{len(missing_paths)} manifest images are missing under DATA_ROOT. First examples:\n{preview}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def _generator(self, index: int) -> torch.Generator:
        # Order/worker independent augmentation seed.
        seed = (self.base_seed * 1_000_003 + self.epoch * 100_003 + int(index)) % (2**63 - 1)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        return g

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # 1x1xHxW
        # Exact study geometry: direct area resize to 224x224.
        x = F.interpolate(x, size=(self.spec.input_size, self.spec.input_size), mode="area")
        x = x.squeeze(0).repeat(3, 1, 1)
        return x

    def __getitem__(self, index: int):
        row = self.frame.iloc[int(index)]
        path = self.data_root / str(row.relative_path)
        x = self._load_image(path)
        if self.training:
            x = self.augment(x, self._generator(index))
        mean = torch.tensor(self.spec.normalize_mean, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor(self.spec.normalize_std, dtype=x.dtype).view(3, 1, 1)
        x = (x - mean) / std
        y = torch.tensor(float(row.class_label), dtype=torch.float32)
        meta = {
            "file_identifier": str(row.relative_path),
            "anonymous_subject_id": str(row.anonymous_subject_id),
            "class_label": int(row.class_label),
        }
        return x, y, meta


def collate_with_metadata(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(metas)
