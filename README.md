# Vietnamese OCR Training Starter

This folder contains a local OCR training pipeline for the `IMG_OCR_VIE_CN` dataset. It does not use any paid API and is designed to be pushed to GitHub, then cloned and trained on Kaggle GPU.

## Quick start

Run locally:

```bash
python -m pip install -r requirements-kaggle.txt
python -m ocr_pipeline.prepare_dataset --data-root . --output-dir prepared_data
python -m ocr_pipeline.train --prepared-dir prepared_data --output-dir runs/crnn_baseline --epochs 30 --batch-size 64
```

Continue from the latest checkpoint:

```bash
python -m ocr_pipeline.train --prepared-dir prepared_data --output-dir runs/crnn_baseline --epochs 60 --batch-size 32 --resume runs/crnn_baseline/last.pt
```

Run on Kaggle after cloning the repo:

```bash
!git clone <your-repo-url>
%cd <your-repo-folder>/IMG_OCR_VIE_CN
!python -m pip install -r requirements-kaggle.txt
!python -m ocr_pipeline.prepare_dataset --data-root . --output-dir prepared_data
!python -m ocr_pipeline.train --prepared-dir prepared_data --output-dir /kaggle/working/crnn_run --epochs 30 --batch-size 64 --device cuda
```

Continue training on Kaggle:

```bash
!python -m ocr_pipeline.train --prepared-dir prepared_data --output-dir /kaggle/working/crnn_run --epochs 60 --batch-size 32 --device cuda --resume /kaggle/working/crnn_run/last.pt
```

## Files

- `ocr_pipeline/prepare_dataset.py`: converts LabelMe-style boxes to cropped text images
- `ocr_pipeline/train.py`: trains a CRNN + CTC baseline
- `ocr_pipeline/infer.py`: runs inference on one crop
- `requirements-kaggle.txt`: minimal package list for Kaggle
- `kaggle_train.ipynb`: ready-to-run Kaggle notebook

## Notes

- The current baseline is a text recognition model, not a full text detector.
- `.gitignore` excludes generated folders like `prepared_data/` and `runs/`.
- If your GitHub repo becomes too large, keep only the code in GitHub and upload the raw dataset separately as a Kaggle Dataset.

More detail is in [README_KAGGLE.md](./README_KAGGLE.md).

## Gradio demo

Run a simple demo for cropped text images:

```bash
python app_gradio.py --checkpoint runs/crnn_baseline/best.pt --charset runs/crnn_baseline/charset.txt
```
