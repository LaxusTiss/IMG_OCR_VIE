from __future__ import annotations

import argparse
from pathlib import Path

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
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    scale = image_height / max(h, 1)
    resized_w = max(8, min(max_width, int(round(w * scale))))
    gray = cv2.resize(gray, (resized_w, image_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(gray.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5


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

    def predict(self, image: np.ndarray | None) -> tuple[str, str]:
        if image is None:
            return "", "Hay upload một ảnh crop chỉ chứa một vùng chữ."

        tensor = preprocess(image, self.image_height, self.max_width).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prediction = greedy_decode(logits.log_softmax(dim=-1).cpu(), self.vocab)[0]

        note = (
            f"Checkpoint: {self.checkpoint_path.name}\n"
            "Lưu ý: model này đọc tốt nhất khi ảnh chỉ chứa một dòng/cụm chữ đã crop sẵn."
        )
        return prediction, note


def build_interface(demo: OCRDemo) -> gr.Blocks:
    description = (
        "Demo OCR baseline CRNN + CTC. "
        "Model hiện tại dành cho ảnh crop vùng chữ, không phải toàn bộ trang tài liệu."
    )
    with gr.Blocks(title="Vietnamese OCR Demo") as app:
        gr.Markdown("# Vietnamese OCR Demo")
        gr.Markdown(description)
        with gr.Row():
            image_input = gr.Image(type="numpy", label="Upload crop ảnh chữ")
            with gr.Column():
                text_output = gr.Textbox(label="Text prediction", lines=3)
                note_output = gr.Textbox(label="Notes", lines=4)
        run_button = gr.Button("Run OCR", variant="primary")
        run_button.click(fn=demo.predict, inputs=image_input, outputs=[text_output, note_output])
    return app


def main() -> None:
    args = parse_args()
    demo = OCRDemo(args.checkpoint, args.charset, args.image_height, args.max_width, args.device)
    app = build_interface(demo)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
