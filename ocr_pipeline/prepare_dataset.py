from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))
    return True


def find_image_for_json(json_path: Path) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def points_to_quad(points: Iterable[Iterable[float]]) -> np.ndarray | None:
    pts = np.array(list(points), dtype=np.float32)
    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        left, right = sorted([x1, x2])
        top, bottom = sorted([y1, y2])
        pts = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )
        return pts
    if len(pts) == 4:
        return order_quad(pts)
    if len(pts) > 2:
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        return order_quad(box)
    return None


def crop_polygon(image: np.ndarray, points: Iterable[Iterable[float]]) -> np.ndarray | None:
    quad = points_to_quad(points)
    if quad is None:
        return None
    width_top = np.linalg.norm(quad[1] - quad[0])
    width_bottom = np.linalg.norm(quad[2] - quad[3])
    height_left = np.linalg.norm(quad[3] - quad[0])
    height_right = np.linalg.norm(quad[2] - quad[1])
    width = max(int(round(max(width_top, width_bottom))), 4)
    height = max(int(round(max(height_left, height_right))), 4)

    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, destination)
    warped = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.size == 0:
        return None
    return warped


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def build_splits(items: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_category[item["category"]].append(item)

    rng = random.Random(seed)
    train_items: list[dict] = []
    val_items: list[dict] = []
    for _, category_items in sorted(by_category.items()):
        rng.shuffle(category_items)
        split_idx = max(1, int(len(category_items) * train_ratio))
        if split_idx >= len(category_items):
            split_idx = len(category_items) - 1
        train_items.extend(category_items[:split_idx])
        val_items.extend(category_items[split_idx:])
    return train_items, val_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OCR crops from LabelMe-style polygons.")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Dataset root containing image/json pairs.")
    parser.add_argument("--output-dir", type=Path, default=Path("prepared_data"), help="Output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio by document.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict] = []
    for json_path in sorted(data_root.rglob("*.json")):
        if output_dir in json_path.parents:
            continue
        image_path = find_image_for_json(json_path)
        if image_path is None:
            continue
        documents.append(
            {
                "json_path": json_path,
                "image_path": image_path,
                "category": json_path.parent.name,
                "stem": json_path.stem,
            }
        )

    train_docs, val_docs = build_splits(documents, args.train_ratio, args.seed)
    split_lookup = {doc["json_path"]: "train" for doc in train_docs}
    split_lookup.update({doc["json_path"]: "val" for doc in val_docs})

    manifests = {"train": [], "val": []}
    charset = Counter()
    stats = Counter()

    for document in documents:
        split = split_lookup[document["json_path"]]
        image = imread_unicode(document["image_path"])
        if image is None:
            stats["missing_images"] += 1
            continue

        payload = json.loads(document["json_path"].read_text(encoding="utf-8"))
        shapes = payload.get("shapes", [])
        for idx, shape in enumerate(shapes):
            text = normalize_text(shape.get("label", ""))
            if not text:
                stats["empty_labels"] += 1
                continue

            points = shape.get("points", [])
            if len(points) < 2:
                stats["bad_polygons"] += 1
                continue

            crop = crop_polygon(image, points)
            if crop is None or min(crop.shape[:2]) < 4:
                stats["bad_crops"] += 1
                continue

            crop_rel = Path("crops") / split / document["category"] / f"{document['stem']}_{idx:03d}.png"
            crop_path = output_dir / crop_rel
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            ok = imwrite_unicode(crop_path, crop)
            if not ok:
                stats["write_failures"] += 1
                continue

            manifests[split].append(
                {
                    "image": crop_rel.as_posix(),
                    "text": text,
                    "category": document["category"],
                    "source_image": document["image_path"].relative_to(data_root).as_posix(),
                }
            )
            charset.update(text)
            stats[f"{split}_samples"] += 1

    for split, rows in manifests.items():
        manifest_path = output_dir / f"{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    charset_path = output_dir / "charset.txt"
    charset_path.write_text("".join(sorted(charset)), encoding="utf-8")

    metadata = {
        "documents": len(documents),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_samples": stats["train_samples"],
        "val_samples": stats["val_samples"],
        "unique_chars": len(charset),
        "stats": dict(stats),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
