# OCR training pipeline

This folder now includes a local OCR training pipeline that does not use any paid API.

## What it does

- Reads the existing `*.json` polygon annotations in this dataset.
- Crops each text region into a word-level image.
- Builds `train/val` manifests and a charset from the labels.
- Trains a CRNN + CTC recognizer in PyTorch.

## Structure

- `ocr_pipeline/prepare_dataset.py`: create crops and manifests
- `ocr_pipeline/train.py`: train the recognizer
- `ocr_pipeline/infer.py`: run inference on one crop
- `requirements-kaggle.txt`: minimal packages for Kaggle

## Run locally or on Kaggle

```bash
python -m pip install -r requirements-kaggle.txt
python -m ocr_pipeline.prepare_dataset --data-root . --output-dir prepared_data
python -m ocr_pipeline.train --prepared-dir prepared_data --output-dir runs/crnn_baseline --epochs 30 --batch-size 64
```

## Kaggle notes

If you clone this repo in Kaggle:

```bash
!git clone <your-repo-url>
%cd <your-repo-folder>/IMG_OCR_VIE_CN
!python -m pip install -r requirements-kaggle.txt
!python -m ocr_pipeline.prepare_dataset --data-root . --output-dir prepared_data
!python -m ocr_pipeline.train --prepared-dir prepared_data --output-dir /kaggle/working/crnn_run --epochs 30 --batch-size 64 --device cuda
```

Recommended Kaggle settings:

- Accelerator: `GPU`
- Internet: optional for package install or git clone
- Batch size: start with `64`, reduce to `32` if GPU memory is tight

## Outputs

- `prepared_data/train.jsonl`, `prepared_data/val.jsonl`
- `prepared_data/charset.txt`
- `runs/.../best.pt`
- `runs/.../last.pt`
- `runs/.../history.json`

## Important limitation

This baseline is a text recognition model trained on cropped word regions. It does not detect text boxes at inference time by itself. Since your dataset already contains polygons, this is the fastest path to a working baseline. If you want, the next step can be adding a detection model or exporting to PaddleOCR/MMOCR format.
