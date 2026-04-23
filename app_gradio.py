from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch

from ocr_pipeline.dataset import OCRVocab
from ocr_pipeline.model import CRNN
from ocr_pipeline.train import greedy_decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for the OCR recognizer.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/smoke_test/best.pt"))
    parser.add_argument("--charset", type=Path, default=Path("runs/smoke_test/charset.txt"))
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio sharing.")
    return parser.parse_args()


def preprocess(image: np.ndarray, image_height: int, max_width: int) -> torch.Tensor:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    scale = image_height / max(h, 1)
    resized_w = max(8, min(max_width, int(round(w * scale))))
    gray = cv2.resize(gray, (resized_w, image_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(gray.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def extract_editor_image(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return to_rgb(value)
    if isinstance(value, dict):
        for key in ("composite", "background"):
            image = value.get(key)
            if isinstance(image, np.ndarray):
                return to_rgb(image)
    return None


class OCRDemo:
    def __init__(self, checkpoint_path: Path, charset_path: Path, image_height: int, max_width: int, device: str) -> None:
        self.device = torch.device(device)
        self.vocab = OCRVocab.from_charset_file(charset_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = CRNN(num_classes=self.vocab.size)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.image_height = image_height
        self.max_width = max_width
        self.checkpoint_path = checkpoint_path

    def predict_from_editor(self, editor_value: Any) -> tuple[str, str]:
        image = extract_editor_image(editor_value)
        if image is None:
            return "", "Upload ảnh tài liệu, dùng Crop để cắt một dòng/cụm chữ rồi bấm Run OCR."

        tensor = preprocess(image, self.image_height, self.max_width).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prediction = greedy_decode(logits.log_softmax(dim=-1).cpu(), self.vocab)[0]

        note = (
            f"Checkpoint: {self.checkpoint_path.name}\n"
            "Mẹo: crop sát một dòng hoặc một cụm chữ ngang để model đọc ổn định hơn."
        )
        return prediction, note


def sample_examples() -> list[list[str]]:
    sample_dir = Path("demo_samples")
    candidates = [
        sample_dir / "student_card_title.png",
        sample_dir / "student_name.png",
        sample_dir / "student_birth.png",
        sample_dir / "bill_resort.png",
        sample_dir / "bill_code.png",
    ]
    return [[str(path)] for path in candidates if path.exists()]


def build_interface(demo: OCRDemo) -> gr.Blocks:
    description = (
        "Upload ảnh tài liệu, dùng công cụ Crop để cắt vùng chữ cần đọc, rồi bấm Run OCR. "
        "Demo này phù hợp nhất với một dòng hoặc một cụm chữ đã crop gọn."
    )
    with gr.Blocks(title="Vietnamese OCR Demo") as app:
        gr.Markdown("# Vietnamese OCR Demo")
        gr.Markdown(description)
        with gr.Row():
            editor = gr.ImageEditor(
                type="numpy",
                image_mode="RGB",
                label="Upload ảnh tài liệu rồi crop vùng cần đọc",
                transforms=("crop",),
                layers=False,
                eraser=False,
                brush=False,
                sources=("upload", "clipboard"),
                height=520,
            )
            with gr.Column():
                text_output = gr.Textbox(label="Text prediction", lines=3)
                note_output = gr.Textbox(label="Notes", lines=4)
                run_button = gr.Button("Run OCR", variant="primary")

        examples = sample_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[editor],
                label="Sample crops to test quickly",
            )

        run_button.click(
            fn=demo.predict_from_editor,
            inputs=editor,
            outputs=[text_output, note_output],
        )
    return app


def main() -> None:
    args = parse_args()
    demo = OCRDemo(args.checkpoint, args.charset, args.image_height, args.max_width, args.device)
    app = build_interface(demo)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
