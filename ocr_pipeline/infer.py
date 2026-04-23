from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from ocr_pipeline.dataset import OCRVocab
from ocr_pipeline.model import CRNN
from ocr_pipeline.train import greedy_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCR inference on one prepared crop.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--charset", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def main() -> None:
    args = parse_args()
    vocab = OCRVocab.from_charset_file(args.charset)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = CRNN(num_classes=vocab.size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    image = imread_unicode(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    h, w = image.shape[:2]
    scale = args.image_height / max(h, 1)
    resized_w = max(8, min(args.max_width, int(round(w * scale))))
    image = cv2.resize(image, (resized_w, args.image_height), interpolation=cv2.INTER_CUBIC)
    image = image.astype("float32") / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    image = (image - 0.5) / 0.5
    image = image.to(args.device)

    with torch.no_grad():
        logits = model(image)
        prediction = greedy_decode(logits.log_softmax(dim=-1).cpu(), vocab)[0]
    print(prediction)


if __name__ == "__main__":
    main()
