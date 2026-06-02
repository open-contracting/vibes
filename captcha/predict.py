# ruff: noqa: INP001
"""
Predict the 6-character text of an Assam tenders captcha.

Library usage:
    from predict import predict, predict_with_confidence

    text = predict("samples/captcha_0001.png")
    text = predict(base64_str)              # base64-encoded PNG, with or without data: prefix
    text = predict(png_bytes)               # raw bytes
    text = predict(pil_image)               # PIL.Image
    text, conf = predict_with_confidence(image)  # conf is list[float], one per character

CLI usage:
    ./predict.py samples/captcha_0001.png
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import sys
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from manage import (
    CROP_H,
    CROP_W,
    IDX_TO_CHAR,
    TinyCNN,
    char_crops,
    resize_to,
)
from PIL import Image

ImageInput = str | bytes | Path | Image.Image | np.ndarray
DEFAULT_MODEL_PATH = Path(__file__).parent / "captcha_cnn.pt"

RGB_NDIM = 3  # height, width, channel axes
RGB_CHANNELS = 3
MAX_PATH_LEN = 260  # paths longer than this can't be filesystem paths, so treat as base64


@cache
def _get_model(model_path: Path = DEFAULT_MODEL_PATH) -> tuple[TinyCNN, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model, device


def _to_rgb(image: ImageInput) -> np.ndarray:
    """Coerce any supported input into an HxWx3 uint8 array composited over white."""
    if isinstance(image, np.ndarray):
        if image.ndim == RGB_NDIM and image.shape[2] == RGB_CHANNELS and image.dtype == np.uint8:
            return image
        pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image
    elif isinstance(image, (str, Path)):
        s = str(image)
        # Heuristic: if it looks like base64 (not a filesystem path), decode it.
        pil = Image.open(s) if isinstance(image, Path) or _looks_like_path(s) else _open_b64(s)
    elif isinstance(image, bytes):
        pil = Image.open(io.BytesIO(image))
    else:
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    pil = pil.convert("RGBA")
    bg = Image.new("RGBA", pil.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(bg, pil).convert("RGB"))


def _looks_like_path(s: str) -> bool:
    # Treat anything short with a known image suffix or an existing file as a path.
    if len(s) < MAX_PATH_LEN and (s.endswith((".png", ".jpg", ".jpeg"))):
        return True
    return Path(s).exists()


def _open_b64(s: str) -> Image.Image:
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    try:
        data = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("input is neither a valid file path nor base64") from e
    return Image.open(io.BytesIO(data))


def _infer(rgb: np.ndarray, model_path: Path) -> tuple[str, list[float]]:
    model, device = _get_model(model_path)
    crops = char_crops(rgb)
    batch = []
    for crop in crops:
        arr = resize_to(crop, CROP_H, CROP_W)
        x = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
        x = (x - 0.5) / 0.5
        batch.append(x)
    xs = torch.stack(batch).to(device)
    with torch.no_grad():
        logits = model(xs)
        probs = F.softmax(logits, dim=1)
        top_p, top_idx = probs.max(dim=1)
    text = "".join(IDX_TO_CHAR[i.item()] for i in top_idx)
    conf = [float(p) for p in top_p]
    return text, conf


def predict(image: ImageInput, model_path: Path = DEFAULT_MODEL_PATH) -> str:
    """Return the 6-character prediction for the given captcha image."""
    return _infer(_to_rgb(image), model_path)[0]


def predict_with_confidence(image: ImageInput, model_path: Path = DEFAULT_MODEL_PATH) -> tuple[str, list[float]]:
    """Return (prediction, per-character confidence) for the captcha image."""
    return _infer(_to_rgb(image), model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a PNG captcha")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--confidence", action="store_true", help="Also print per-char confidence")
    args = parser.parse_args()

    if args.confidence:
        text, conf = predict_with_confidence(args.image, args.model)
        print(text)
        print(" ".join(f"{c}={p:.2f}" for c, p in zip(text, conf, strict=False)))
    else:
        print(predict(args.image, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
