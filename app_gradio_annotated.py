from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import torch

from ocr_pipeline.dataset import OCRVocab
from ocr_pipeline.model import CRNN
from ocr_pipeline.prepare_dataset import crop_polygon, imread_unicode, normalize_text, points_to_quad
from ocr_pipeline.train import greedy_decode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio demo for full-document OCR using existing annotations.")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/smoke_test/best.pt"))
    parser.add_argument("--charset", type=Path, default=Path("runs/smoke_test/charset.txt"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=320)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true", help="Enable public Gradio sharing.")
    return parser.parse_args()


def preprocess(image: np.ndarray, image_height: int, max_width: int) -> torch.Tensor:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = image_height / max(h, 1)
    resized_w = max(8, min(max_width, int(round(w * scale))))
    gray = cv2.resize(gray, (resized_w, image_height), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(gray.astype("float32") / 255.0).unsqueeze(0).unsqueeze(0)
    return (tensor - 0.5) / 0.5


def discover_images(root: Path) -> list[str]:
    images: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(part in {"prepared_data", "runs", "demo_samples", ".git"} for part in path.parts):
            continue
        if path.with_suffix(".json").exists():
            images.append(path.relative_to(root).as_posix())
    return images


def sort_boxes(records: list[dict]) -> list[dict]:
    if not records:
        return records
    heights = [record["bottom"] - record["top"] for record in records]
    line_threshold = max(12.0, float(np.median(heights)) * 0.7)
    ordered = sorted(records, key=lambda item: (item["top"], item["left"]))

    lines: list[list[dict]] = []
    current_line: list[dict] = []
    current_y: float | None = None
    for record in ordered:
        center_y = (record["top"] + record["bottom"]) / 2.0
        if current_y is None or abs(center_y - current_y) <= line_threshold:
            current_line.append(record)
            current_y = center_y if current_y is None else (current_y + center_y) / 2.0
        else:
            lines.append(sorted(current_line, key=lambda item: item["left"]))
            current_line = [record]
            current_y = center_y
    if current_line:
        lines.append(sorted(current_line, key=lambda item: item["left"]))

    flattened: list[dict] = []
    for line_idx, line in enumerate(lines):
        for order_idx, record in enumerate(line):
            record["line_idx"] = line_idx
            record["order_idx"] = order_idx
            flattened.append(record)
    return flattened


class FullDocumentDemo:
    def __init__(self, checkpoint_path: Path, charset_path: Path, data_root: Path, image_height: int, max_width: int, device: str) -> None:
        self.device = torch.device(device)
        self.vocab = OCRVocab.from_charset_file(charset_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = CRNN(num_classes=self.vocab.size)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()
        self.data_root = data_root.resolve()
        self.image_height = image_height
        self.max_width = max_width
        self.checkpoint_path = checkpoint_path
        self.choices = discover_images(self.data_root)

    def recognize_crop(self, crop: np.ndarray) -> str:
        tensor = preprocess(crop, self.image_height, self.max_width).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            return greedy_decode(logits.log_softmax(dim=-1).cpu(), self.vocab)[0]

    def run_on_dataset_image(self, relative_image_path: str) -> tuple[np.ndarray | None, str, pd.DataFrame, str]:
        if not relative_image_path:
            return None, "", pd.DataFrame(columns=["idx", "ground_truth", "prediction"]), "Chọn một ảnh trong dataset."

        image_path = self.data_root / relative_image_path
        json_path = image_path.with_suffix(".json")
        image = imread_unicode(image_path)
        if image is None or not json_path.exists():
            return None, "", pd.DataFrame(columns=["idx", "ground_truth", "prediction"]), "Không đọc được ảnh hoặc thiếu file json."

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        shapes = payload.get("shapes", [])
        records: list[dict] = []
        for idx, shape in enumerate(shapes):
            text = normalize_text(shape.get("label", ""))
            points = shape.get("points", [])
            if not text or len(points) < 2:
                continue
            quad = points_to_quad(points)
            crop = crop_polygon(image, points)
            if quad is None or crop is None:
                continue
            prediction = self.recognize_crop(crop)
            quad = quad.astype(np.int32)
            left = float(quad[:, 0].min())
            right = float(quad[:, 0].max())
            top = float(quad[:, 1].min())
            bottom = float(quad[:, 1].max())
            records.append(
                {
                    "idx": idx,
                    "ground_truth": text,
                    "prediction": prediction,
                    "quad": quad,
                    "left": left,
                    "right": right,
                    "top": top,
                    "bottom": bottom,
                }
            )

        records = sort_boxes(records)

        annotated = image.copy()
        for row_idx, record in enumerate(records, start=1):
            cv2.polylines(annotated, [record["quad"]], isClosed=True, color=(0, 255, 0), thickness=2)
            anchor = tuple(record["quad"][0])
            cv2.putText(
                annotated,
                str(row_idx),
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        lines: dict[int, list[str]] = {}
        for record in records:
            lines.setdefault(record["line_idx"], []).append(record["prediction"])
        merged_text = "\n".join(" ".join(part for part in parts if part).strip() for _, parts in sorted(lines.items()))

        table = pd.DataFrame(
            [
                {
                    "idx": idx + 1,
                    "ground_truth": record["ground_truth"],
                    "prediction": record["prediction"],
                }
                for idx, record in enumerate(records)
            ]
        )
        note = (
            f"Checkpoint: {self.checkpoint_path.name}\n"
            f"Ảnh: {relative_image_path}\n"
            f"Số box dùng để OCR: {len(records)}"
        )
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        return annotated, merged_text, table, note


def build_interface(demo: FullDocumentDemo) -> gr.Blocks:
    with gr.Blocks(title="Full Document OCR Demo") as app:
        gr.Markdown("# Full Document OCR Demo")
        gr.Markdown(
            "Chọn một ảnh gốc trong dataset. App sẽ tự đọc file json đi kèm, crop từng vùng chữ, OCR từng box rồi ghép lại theo thứ tự đọc."
        )
        with gr.Row():
            image_selector = gr.Dropdown(choices=demo.choices, label="Dataset image", value=demo.choices[0] if demo.choices else None)
            run_button = gr.Button("Run Full OCR", variant="primary")
        with gr.Row():
            annotated_output = gr.Image(type="numpy", label="Detected text boxes")
            with gr.Column():
                text_output = gr.Textbox(label="Merged OCR text", lines=16)
                note_output = gr.Textbox(label="Notes", lines=4)
        table_output = gr.Dataframe(label="Ground truth vs prediction", interactive=False)

        run_button.click(
            fn=demo.run_on_dataset_image,
            inputs=image_selector,
            outputs=[annotated_output, text_output, table_output, note_output],
        )
    return app


def main() -> None:
    args = parse_args()
    demo = FullDocumentDemo(args.checkpoint, args.charset, args.data_root, args.image_height, args.max_width, args.device)
    app = build_interface(demo)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
