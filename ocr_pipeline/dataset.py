from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class OCRVocab:
    chars: str
    blank_idx: int = 0

    @classmethod
    def from_charset_file(cls, path: Path) -> "OCRVocab":
        return cls(chars=path.read_text(encoding="utf-8"))

    @property
    def size(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        lookup = {char: idx + 1 for idx, char in enumerate(self.chars)}
        return [lookup[ch] for ch in text if ch in lookup]

    def decode(self, tokens: list[int]) -> str:
        chars = []
        for idx in tokens:
            if idx <= 0:
                continue
            if idx - 1 < len(self.chars):
                chars.append(self.chars[idx - 1])
        return "".join(chars)


class OCRCropDataset(Dataset):
    def __init__(self, prepared_dir: Path, split: str, vocab: OCRVocab, image_height: int, max_width: int) -> None:
        self.prepared_dir = prepared_dir
        self.samples = load_manifest(prepared_dir / f"{split}.jsonl")
        self.vocab = vocab
        self.image_height = image_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row = self.samples[index]
        image_path = self.prepared_dir / row["image"]
        image = imread_unicode(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read crop: {image_path}")

        h, w = image.shape[:2]
        scale = self.image_height / max(h, 1)
        resized_w = max(8, min(self.max_width, int(round(w * scale))))
        image = cv2.resize(image, (resized_w, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = image.astype("float32") / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        image = (image - 0.5) / 0.5

        encoded = self.vocab.encode(row["text"])
        if not encoded:
            encoded = [self.vocab.blank_idx]

        return {
            "image": image,
            "label": torch.tensor(encoded, dtype=torch.long),
            "text": row["text"],
            "image_path": row["image"],
        }


def collate_fn(batch: list[dict]) -> dict:
    max_width = max(sample["image"].shape[-1] for sample in batch)
    images = []
    labels = []
    label_lengths = []
    texts = []
    image_paths = []

    for sample in batch:
        image = sample["image"]
        pad_width = max_width - image.shape[-1]
        if pad_width > 0:
            image = torch.nn.functional.pad(image, (0, pad_width, 0, 0), value=-1.0)
        images.append(image)
        labels.append(sample["label"])
        label_lengths.append(sample["label"].numel())
        texts.append(sample["text"])
        image_paths.append(sample["image_path"])

    return {
        "images": torch.stack(images, dim=0),
        "labels": torch.cat(labels, dim=0),
        "label_lengths": torch.tensor(label_lengths, dtype=torch.long),
        "texts": texts,
        "image_paths": image_paths,
    }
