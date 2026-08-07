#!/usr/bin/env python3
"""
Captcha solver management CLI.

All operational commands live here. The library API for callers (`predict()`,
`predict_with_confidence()`) is in predict.py.

    ./manage.py fetch 500 --start 301
    ./manage.py labelsheet --start 801 --end 830 -o sheet.png
    ./manage.py train --epochs 80
    ./manage.py xval
    ./manage.py review
    ./manage.py predict samples/captcha_0001.png
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import re
import shutil
import string
import sys
import time
from pathlib import Path

import click
import numpy as np
import requests
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torch import nn
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase  # 62 classes
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = dict(enumerate(ALPHABET))
NUM_CLASSES = len(ALPHABET)

N_CHARS = 6
CROP_H = 40
CROP_W = 28
PAIR_LEN = 2  # a --pair option names exactly two characters

CHROMA_MAX = 30
INTENSITY_MAX = 140
TEXT_GRAY_MAX = 200  # pixels darker than this count as text

# Data augmentation probabilities.
PEPPER_PROB = 0.5
SALT_PROB = 0.3
NOISE_FRACTION = 0.01
MIN_GLYPH_HEIGHT = 15  # letter bboxes are ~25-35 tall; noise blobs are ~4-8

DEFAULT_LABELS = HERE / "labels.json"
DEFAULT_SAMPLES = HERE / "samples"
DEFAULT_MODEL = HERE / "captcha_cnn.pt"
DEFAULT_SUSPECTS = HERE / "suspects.txt"
DEFAULT_SUSPECTS_SHEET = HERE / "suspects.png"


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def clean(rgb: np.ndarray) -> np.ndarray:
    """
    Return a grayscale image where text pixels stay dark and everything else is white.

    Drops chromatic noise via a color filter, then keeps only connected
    components whose bounding box is tall enough to be a letter. The pepper
    noise in this captcha is small square blobs much shorter than glyphs.
    """
    rgb_i = rgb.astype(np.int16)
    chroma = rgb_i.max(axis=-1) - rgb_i.min(axis=-1)
    intensity = rgb_i.mean(axis=-1)
    is_text = (chroma < CHROMA_MAX) & (intensity < INTENSITY_MAX)
    # Close 1-pixel vertical gaps inside glyphs (some letters render as separated top/bottom strokes).
    bridged = ndimage.binary_closing(is_text, structure=np.ones((3, 1), dtype=bool))
    # 8-connectivity so diagonally adjacent letter strokes stay one component.
    labels, n = ndimage.label(bridged, structure=np.ones((3, 3), dtype=bool))
    if n:
        slices = ndimage.find_objects(labels)
        keep_ids = [i + 1 for i, sl in enumerate(slices) if (sl[0].stop - sl[0].start) >= MIN_GLYPH_HEIGHT]
        bridged = np.isin(labels, keep_ids)
    out = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    out[bridged & is_text] = intensity[bridged & is_text].astype(np.uint8)
    out[bridged & ~is_text] = 0
    return out


def text_bbox(gray: np.ndarray) -> tuple[int, int]:
    """Return the (left, right) columns spanning all text pixels."""
    cols = (gray < TEXT_GRAY_MAX).any(axis=0)
    if not cols.any():
        return 0, gray.shape[1]
    xs = np.flatnonzero(cols)
    return int(xs[0]), int(xs[-1]) + 1


def component_bboxes(rgb: np.ndarray, max_count: int | None = None) -> list[tuple[int, int]]:
    """
    Horizontal (x0, x1) ranges of each kept text component, sorted left to right.

    Returns components surviving the chroma + intensity filter, vertical 3x1
    bridging, and MIN_GLYPH_HEIGHT filter. If `max_count` is set and more than
    that many components pass the height filter, keeps only the top-N by *pixel
    area* (not bbox area — thin tall edge artifacts have large bbox area but
    few actual pixels). Result is sorted left-to-right.
    """
    rgb_i = rgb.astype(np.int16)
    chroma = rgb_i.max(axis=-1) - rgb_i.min(axis=-1)
    intensity = rgb_i.mean(axis=-1)
    is_text = (chroma < CHROMA_MAX) & (intensity < INTENSITY_MAX)
    bridged = ndimage.binary_closing(is_text, structure=np.ones((3, 1), dtype=bool))
    labels, n = ndimage.label(bridged, structure=np.ones((3, 3), dtype=bool))
    if not n:
        return []
    slices = ndimage.find_objects(labels)
    pixel_counts = np.bincount(labels.ravel())
    candidates: list[tuple[int, int, int]] = []
    for i, sl in enumerate(slices):
        if (sl[0].stop - sl[0].start) >= MIN_GLYPH_HEIGHT:
            candidates.append((sl[1].start, sl[1].stop, int(pixel_counts[i + 1])))
    if max_count is not None and len(candidates) > max_count:
        candidates.sort(key=lambda c: -c[2])
        candidates = candidates[:max_count]
    candidates.sort(key=lambda c: c[0])
    return [(x0, x1) for x0, x1, _ in candidates]


# ---------------------------------------------------------------------------
# Library helpers (also imported by predict.py)
# ---------------------------------------------------------------------------


def load_original(path: Path) -> np.ndarray:
    """Composite a captcha PNG over white and return an HxWx3 uint8 array."""
    src = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
    return np.array(Image.alpha_composite(bg, src).convert("RGB"))


def char_x_ranges(rgb: np.ndarray) -> list[tuple[int, int]]:
    """
    Per-character (x0, x1) column ranges for slicing the original image.

    When cleaning produces exactly N_CHARS components, slot widths stay uniform
    (matching the training distribution) but each slot is centered on its
    component — so wide letters like B or W aren't chopped by the slot boundary.
    Falls back to uniform N-way division of the text bbox when CC count != N_CHARS
    (cleaning lost or merged a glyph).
    """
    w = rgb.shape[1]
    cleaned = clean(rgb)
    left, right = text_bbox(cleaned)
    bboxes = component_bboxes(rgb, max_count=N_CHARS)
    if len(bboxes) == N_CHARS:
        slot_w = (right - left) / N_CHARS
        half = slot_w / 2
        ranges: list[tuple[int, int]] = []
        for x0, x1 in bboxes:
            c = (x0 + x1) / 2
            ranges.append((max(0, round(c - half)), min(w, round(c + half))))
        return ranges
    step = (right - left) / N_CHARS
    return [(int(left + i * step), int(left + (i + 1) * step)) for i in range(N_CHARS)]


def char_crops(rgb: np.ndarray) -> list[np.ndarray]:
    """Slice the *original* image into N_CHARS grayscale crops aligned to detected letters."""
    gray = rgb.mean(axis=-1).astype(np.uint8)
    return [gray[:, x0:x1] for x0, x1 in char_x_ranges(rgb)]


def resize_to(crop: np.ndarray, h: int = CROP_H, w: int = CROP_W) -> np.ndarray:
    return np.array(Image.fromarray(crop).resize((w, h), Image.BILINEAR))


def build_items(labels: dict[str, str], samples_dir: Path) -> list[tuple[Path, int, str]]:
    """Flatten labels.json into (path, char_index, char) triples for training."""
    items: list[tuple[Path, int, str]] = []
    for fname, label in labels.items():
        if len(label) != N_CHARS:
            click.echo(f"skip {fname}: label '{label}' is not {N_CHARS} chars", err=True)
            continue
        path = samples_dir / fname
        if not path.exists():
            click.echo(f"skip {fname}: file missing", err=True)
            continue
        for i, c in enumerate(label):
            if c in CHAR_TO_IDX:
                items.append((path, i, c))
            else:
                click.echo(f"skip {fname}[{i}]: char '{c}' not in alphabet", err=True)
    return items


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CharDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int, str]], *, augment: bool = False, seed: int | None = None) -> None:
        self.items = items
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, char_idx, label = self.items[idx]
        rgb = load_original(path)
        crops = char_crops(rgb)
        crop = resize_to(crops[char_idx])
        if self.augment:
            crop = self._augment(crop)
        x = torch.from_numpy(crop).float().unsqueeze(0) / 255.0
        x = (x - 0.5) / 0.5
        return x, CHAR_TO_IDX[label]

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        img = Image.fromarray(crop)
        angle = self.rng.uniform(-8, 8)
        img = img.rotate(angle, fillcolor=255, resample=Image.BILINEAR)
        arr = np.array(img)
        if self.rng.random() < PEPPER_PROB:
            arr[self.rng.random(arr.shape) < NOISE_FRACTION] = 0
        if self.rng.random() < SALT_PROB:
            arr[self.rng.random(arr.shape) < NOISE_FRACTION] = 255
        return arr


class TinyCNN(nn.Module):
    """
    Small CNN with a 2-row adaptive pool to preserve vertical position.

    The (2, 1) pool keeps a top-half vs bottom-half distinction in the feature
    vector — useful for distinguishing characters that differ mainly in where a
    stroke extends (e.g. p has a descender below the baseline while b has an
    ascender above the x-height; n vs h is the same pattern).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 1)),
        )
        self.classifier = nn.Linear(128 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def fit_one(
    train_items: list[tuple[Path, int, str]],
    val_items: list[tuple[Path, int, str]] | None,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
    save_best_to: Path | None,
    log_prefix: str = "",
    seed: int | None = None,
) -> tuple[TinyCNN, float, int]:
    """
    Train one model.

    If save_best_to is set and val_items is non-empty, save the best-by-val-acc
    checkpoint. Returns (model, best_val_acc, best_epoch).
    """
    train_dl = DataLoader(
        CharDataset(train_items, augment=True, seed=seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_dl = (
        DataLoader(
            CharDataset(val_items, augment=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        if val_items
        else None
    )
    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = 0.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = train_total = train_loss = 0
        for x, y in train_dl:
            xb, yb = x.to(device), y.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()
            train_total += xb.size(0)
        sched.step()
        val_acc = None
        marker = ""
        if val_dl is not None:
            model.eval()
            v_correct = v_total = 0
            with torch.no_grad():
                for x, y in val_dl:
                    xb, yb = x.to(device), y.to(device)
                    v_correct += (model(xb).argmax(1) == yb).sum().item()
                    v_total += xb.size(0)
            val_acc = v_correct / v_total
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                if save_best_to is not None:
                    torch.save(model.state_dict(), save_best_to)
                    marker = " *"
        line = (
            f"{log_prefix}epoch {epoch:02d}/{epochs} "
            f"loss={train_loss / train_total:.4f} acc={train_correct / train_total:.3f}"
        )
        if val_acc is not None:
            line += f" val={val_acc:.3f}{marker}"
        click.echo(line)
    return model, best_val, best_epoch


# ---------------------------------------------------------------------------
# Font + terminal protocol helpers
# ---------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


ITERM_TEMPLATE = "\033]1337;File=inline=1;preserveAspectRatio=1;height={h}:{data}\007"
KITTY_CHUNK_SIZE = 4096


def detect_terminal_protocol() -> str:
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if "kitty" in term or "ghostty" in term:
        return "kitty"
    if term_program in {"ghostty", "kitty"}:
        return "kitty"
    if "KITTY_WINDOW_ID" in os.environ or "GHOSTTY_RESOURCES_DIR" in os.environ:
        return "kitty"
    return "iterm"


def print_inline_iterm(img: Image.Image, term_rows: int) -> None:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    sys.stdout.write(ITERM_TEMPLATE.format(h=term_rows, data=encoded))
    sys.stdout.write("\n")
    sys.stdout.flush()


def print_inline_kitty(img: Image.Image, term_rows: int) -> None:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode()
    base_params = f"a=T,f=100,r={term_rows}"
    if len(data) <= KITTY_CHUNK_SIZE:
        sys.stdout.write(f"\033_G{base_params};{data}\033\\")
    else:
        first = True
        for i in range(0, len(data), KITTY_CHUNK_SIZE):
            chunk = data[i : i + KITTY_CHUNK_SIZE]
            more = i + KITTY_CHUNK_SIZE < len(data)
            params = f"{base_params},m={1 if more else 0}" if first else f"m={1 if more else 0}"
            first = False
            sys.stdout.write(f"\033_G{params};{chunk}\033\\")
    sys.stdout.write("\n")
    sys.stdout.flush()


def print_inline(img: Image.Image, term_rows: int, protocol: str) -> None:
    if protocol == "kitty":
        print_inline_kitty(img, term_rows)
    else:
        print_inline_iterm(img, term_rows)


# ---------------------------------------------------------------------------
# Suspect parsing (used by review subcommand)
# ---------------------------------------------------------------------------


def parse_suspects(path: Path) -> list[tuple[str, int, str, str, float]]:
    rgx = re.compile(r"(\S+)\s+pos=(\d+)\s+label='(.)'\s+pred='(.)'\s+conf=([\d.]+)")
    out: list[tuple[str, int, str, str, float]] = []
    for line in path.read_text().splitlines():
        m = rgx.match(line.strip())
        if m:
            out.append((m.group(1), int(m.group(2)), m.group(3), m.group(4), float(m.group(5))))
    return out


def annotated_suspect_image(samples_dir: Path, fname: str, pos: int, scale: int = 4) -> Image.Image:
    """
    Captcha scaled up with a red box around the suspect column.

    Uses the same per-character slicing as `char_crops`, so the box reflects
    exactly what the CNN sees during training.
    """
    rgb = load_original(samples_dir / fname)
    x0, x1 = char_x_ranges(rgb)[pos]
    h = rgb.shape[0]
    out = rgb.copy()
    out[:2, x0:x1] = [255, 0, 0]
    out[-2:, x0:x1] = [255, 0, 0]
    out[:, x0 : x0 + 2] = [255, 0, 0]
    out[:, max(x1 - 2, x0) : x1] = [255, 0, 0]
    return Image.fromarray(out).resize((out.shape[1] * scale, h * scale), Image.NEAREST)


# ---------------------------------------------------------------------------
# Click app
# ---------------------------------------------------------------------------


@click.group(help=__doc__)
def cli() -> None:
    pass


# ----- fetch -----

BASE_URL = "https://assamtenders.gov.in/nicgep/app"
PAGE_PARAMS = {"page": "WebTenderStatusLists", "service": "page"}
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
CAPTCHA_RE = re.compile(rb'id="captchaImage"[^>]*src="data:image/png;base64,([^"]+)"')


def _fetch_form_data(html: bytes):
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", id="frmSearchFilter")
    if form is None:
        raise RuntimeError("frmSearchFilter form not found on page")
    return {i["name"]: i.get("value", "") for i in form.find_all("input", {"type": "hidden"}) if i.get("name")}


def _fetch_b64(html: bytes) -> str:
    match = CAPTCHA_RE.search(html)
    if not match:
        raise RuntimeError("captchaImage not found in response")
    return match.group(1).decode("ascii")


@cli.command()
@click.argument("n", type=int)
@click.option("--start", default=1, show_default=True, type=int, help="First file index")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SAMPLES,
    show_default=True,
    help="Directory to write captcha PNGs into.",
)
@click.option(
    "--delay",
    default=0.5,
    show_default=True,
    type=float,
    help="Seconds to wait between fetches (politeness delay against the portal).",
)
def fetch(n: int, start: int, output_dir: Path, delay: float) -> None:
    """Download N captchas from the Assam tenders portal."""
    if n < 1 or start < 1:
        raise click.BadParameter("n and --start must be >= 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    r = session.get(BASE_URL, params=PAGE_PARAMS, timeout=30)
    r.raise_for_status()
    html = r.content
    form_data = _fetch_form_data(html)
    b64 = _fetch_b64(html)

    last_idx = start + n - 1
    width = max(4, len(str(last_idx)))
    seen: set[str] = set()
    for i in range(start, last_idx + 1):
        path = output_dir / f"captcha_{i:0{width}d}.png"
        path.write_bytes(base64.b64decode(b64))
        marker = "new" if b64 not in seen else "DUPLICATE"
        seen.add(b64)
        click.echo(f"[{i}/{last_idx}] {path} ({marker})", err=True)
        if i < last_idx:
            if delay:
                time.sleep(delay)
            payload = dict(form_data)
            payload.update(
                {
                    "beventname": "onclick",
                    "bcomponentidpath": "WebTenderStatusLists/captcha",
                    "bcomponentid": "captcha",
                    "beventtarget.id": "captcha",
                    "captchaText": "",
                }
            )
            r = session.post(BASE_URL, data=payload, timeout=30)
            r.raise_for_status()
            html = r.content
            form_data = _fetch_form_data(html)
            b64 = _fetch_b64(html)


# ----- labelsheet -----


@cli.command()
@click.option("--start", required=True, type=int, help="First captcha index (inclusive)")
@click.option("--end", required=True, type=int, help="Last captcha index (inclusive)")
@click.option(
    "--samples",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SAMPLES,
    show_default=True,
    help="Directory containing the captcha PNGs.",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output PNG path for the tiled transcription sheet.",
)
def labelsheet(start: int, end: int, samples: Path, output: Path) -> None:
    """Tile cleaned captchas [start..end] into a transcription PNG."""
    font = _font(22)
    rows = []
    for i in range(start, end + 1):
        path = samples / f"captcha_{i:04d}.png"
        if not path.exists():
            continue
        rgb = load_original(path)
        cleaned = clean(rgb)
        cleaned_rgb = np.stack([cleaned] * 3, axis=-1)
        h, w = cleaned_rgb.shape[:2]
        label_w = 90
        row_h = max(h, 50)
        row = np.full((row_h, label_w + w + 4, 3), 255, dtype=np.uint8)
        row[:, label_w : label_w + 4] = (255, 0, 0)
        y0 = (row_h - h) // 2
        row[y0 : y0 + h, label_w + 4 :] = cleaned_rgb
        pil = Image.fromarray(row)
        ImageDraw.Draw(pil).text((8, (row_h - 24) // 2), f"{i:04d}", fill=(0, 0, 0), font=font)
        rows.append(np.array(pil))
    if not rows:
        click.echo("no captchas found in range", err=True)
        return
    max_w = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    sheet = np.vstack(padded)
    Image.fromarray(sheet).save(output)
    click.echo(f"Wrote {output} ({len(rows)} rows, {sheet.shape[1]}x{sheet.shape[0]} px)")


# ----- train -----


@cli.command()
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_LABELS,
    show_default=True,
    help="labels.json mapping {filename: '6chars'}.",
)
@click.option(
    "--samples",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SAMPLES,
    show_default=True,
    help="Directory containing the captcha PNGs.",
)
@click.option(
    "--epochs",
    default=60,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--batch-size",
    default=64,
    show_default=True,
    help="Mini-batch size for both training and validation.",
)
@click.option("--lr", default=1e-3, show_default=True, help="AdamW learning rate.")
@click.option(
    "--weight-decay",
    default=1e-3,
    show_default=True,
    help="AdamW weight decay (L2 regularisation).",
)
@click.option(
    "--val-frac",
    default=0.2,
    show_default=True,
    help="Fraction of character samples held out for validation.",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Random seed for the train/val shuffle and model init.",
)
@click.option(
    "-o",
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Where to save the best-by-val-accuracy checkpoint.",
)
def train(
    labels_path: Path,
    samples: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_frac: float,
    seed: int,
    out: Path,
) -> None:
    """Train the CNN. Saves the best-by-val checkpoint to --out."""
    random.seed(seed)
    torch.manual_seed(seed)
    labels = json.loads(labels_path.read_text())
    items = build_items(labels, samples)
    random.shuffle(items)
    n_val = int(len(items) * val_frac)
    val_items, train_items = items[:n_val], items[n_val:]
    click.echo(f"train chars: {len(train_items)}  val chars: {len(val_items)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, best_val, best_epoch = fit_one(
        train_items,
        val_items,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        save_best_to=out,
        seed=seed,
    )
    click.echo(f"best val acc {best_val:.3f} at epoch {best_epoch}; saved to {out}")


# ----- xval -----


@cli.command()
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_LABELS,
    show_default=True,
    help="labels.json mapping {filename: '6chars'}.",
)
@click.option(
    "--samples",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SAMPLES,
    show_default=True,
    help="Directory containing the captcha PNGs.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SUSPECTS,
    show_default=True,
    help="Output path for the suspect list (label/pred disagreements).",
)
@click.option(
    "--sheet",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SUSPECTS_SHEET,
    show_default=True,
    help="Output PNG path for the suspect contact sheet.",
)
@click.option(
    "--sheet-limit",
    default=100,
    show_default=True,
    help="Maximum number of suspects rendered in the contact sheet.",
)
@click.option(
    "--folds",
    default=5,
    show_default=True,
    help="Number of cross-validation folds.",
)
@click.option(
    "--epochs",
    default=40,
    show_default=True,
    help="Training epochs per fold.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for fold assignment and model init.",
)
@click.option(
    "--report-only",
    is_flag=True,
    help="Skip cross-validation training; just regenerate the summary from existing suspects.txt.",
)
def xval(
    labels_path: Path,
    samples: Path,
    output: Path,
    sheet: Path,
    sheet_limit: int,
    folds: int,
    epochs: int,
    seed: int,
    *,
    report_only: bool,
) -> None:
    """K-fold cross-validation; write suspects.txt + suspects.png."""
    labels = json.loads(labels_path.read_text())
    total_chars = sum(len(v) for v in labels.values() if len(v) == N_CHARS)

    if report_only:
        if not output.exists():
            raise click.ClickException(f"{output} not found; run xval without --report-only first.")
        suspects = parse_suspects(output)
        _pair_report(suspects, total_chars)
        return

    started = time.monotonic()
    random.seed(seed)
    torch.manual_seed(seed)
    fnames = sorted(labels)
    random.shuffle(fnames)
    fold_size = len(fnames) // folds
    fold_groups = [fnames[i * fold_size : (i + 1) * fold_size] for i in range(folds - 1)]
    fold_groups.append(fnames[(folds - 1) * fold_size :])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions: dict[tuple[str, int], tuple[str, float]] = {}
    for fi, val_fnames in enumerate(fold_groups, 1):
        val_set = set(val_fnames)
        train_labels = {f: lbl for f, lbl in labels.items() if f not in val_set}
        train_items = build_items(train_labels, samples)
        click.echo(f"Fold {fi}/{folds}: training on {len(train_items)} chars, val {len(val_fnames)} captchas")
        model, _, _ = fit_one(
            train_items,
            None,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-3,
            batch_size=64,
            device=device,
            save_best_to=None,
            log_prefix=f"  fold {fi}/{folds} ",
            seed=seed + fi,
        )
        model.eval()
        with torch.no_grad():
            for fname in val_fnames:
                rgb = load_original(samples / fname)
                crops = char_crops(rgb)
                batch = torch.stack(
                    [
                        ((torch.from_numpy(resize_to(c, CROP_H, CROP_W)).float() / 255.0 - 0.5) / 0.5).unsqueeze(0)
                        for c in crops
                    ]
                ).to(device)
                probs = F.softmax(model(batch), dim=1)
                top_p, top_idx = probs.max(dim=1)
                for pos in range(N_CHARS):
                    predictions[(fname, pos)] = (
                        IDX_TO_CHAR[top_idx[pos].item()],
                        float(top_p[pos]),
                    )

    suspects: list[tuple[str, int, str, str, float]] = []
    correct = total = 0
    for (fname, pos), (pred, conf) in predictions.items():
        label = labels[fname]
        if pos >= len(label):
            continue
        expected = label[pos]
        total += 1
        if pred == expected:
            correct += 1
        else:
            suspects.append((fname, pos, expected, pred, conf))
    suspects.sort(key=lambda s: -s[4])

    output.write_text(
        "\n".join(f"{f}  pos={p}  label={e!r}  pred={pr!r}  conf={c:.3f}" for f, p, e, pr, c in suspects) + "\n"
    )
    click.echo(f"Wrote {output}")

    _render_suspect_sheet(suspects, labels, samples, sheet, sheet_limit)

    click.echo()
    _pair_report(suspects, total_chars)

    elapsed = time.monotonic() - started
    mins, secs = divmod(elapsed, 60)
    click.echo(f"xval finished in {int(mins)}m {secs:.1f}s")


def _pair_report(suspects: list[tuple[str, int, str, str, float]], total_chars: int) -> None:
    """Print per-char accuracy + a breakdown of the top confusion pairs."""
    n_dis = len(suspects)
    correct = total_chars - n_dis
    click.echo(f"Cross-validated per-char accuracy: {correct}/{total_chars} = {correct / total_chars:.3%}")
    click.echo(f"Total disagreements: {n_dis}")
    if not n_dis:
        return
    directed: dict[tuple[str, str], int] = {}
    for _, _, lab, pred, _ in suspects:
        directed[(lab, pred)] = directed.get((lab, pred), 0) + 1
    groups: dict[frozenset[str], dict[tuple[str, str], int]] = {}
    for pair, n in directed.items():
        groups.setdefault(frozenset(pair), {})[pair] = n
    rows = [(sum(d.values()), key, d) for key, d in groups.items()]
    rows.sort(key=lambda r: -r[0])
    click.echo()
    click.echo("Top confusion pairs:")
    for total, key, dirs in rows:
        chars = sorted(key)
        if len(chars) == 1:
            click.echo(f"  {chars[0]} : {total}")
            continue
        a, b = chars
        ab = dirs.get((a, b), 0)
        ba = dirs.get((b, a), 0)
        if ab and ba:
            click.echo(f"  {a} ↔ {b} : {total} ({ab} {a}→{b}, {ba} {b}→{a})")
        elif ab:
            click.echo(f"  {a} → {b} : {total}")
        else:
            click.echo(f"  {b} → {a} : {total}")


def _render_suspect_sheet(
    suspects: list[tuple[str, int, str, str, float]],
    labels: dict[str, str],
    samples_dir: Path,
    output: Path,
    limit: int,
) -> None:
    font = _font(18)
    rows = []
    seen: set[str] = set()
    for fname, pos, expected, predicted, conf in suspects[:limit]:
        if fname in seen:
            continue
        seen.add(fname)
        rgb = load_original(samples_dir / fname)
        cleaned = clean(rgb)
        cleaned_rgb = np.stack([cleaned] * 3, axis=-1)
        h, w = cleaned_rgb.shape[:2]
        label_w = 250
        row_h = max(h, 60)
        row = np.full((row_h, label_w + w + 12, 3), 240, dtype=np.uint8)
        y0 = (row_h - h) // 2
        row[y0 : y0 + h, label_w + 8 : label_w + 8 + w] = cleaned_rgb
        pil = Image.fromarray(row)
        draw = ImageDraw.Draw(pil)
        draw.text((6, 4), f"{fname}  conf={conf:.2f}", fill=(40, 40, 40), font=font)
        label_str = labels[fname]
        marked_label = label_str[:pos] + f"[{expected}]" + label_str[pos + 1 :]
        marked_pred = label_str[:pos] + f"[{predicted}]" + label_str[pos + 1 :]
        draw.text((6, 24), f"label: {marked_label}", fill=(20, 20, 20), font=font)
        draw.text((6, 42), f"pred:  {marked_pred}", fill=(160, 0, 0), font=font)
        rows.append(np.array(pil))
    if not rows:
        return
    max_w = max(r.shape[1] for r in rows)
    padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)), constant_values=240) for r in rows]
    Image.fromarray(np.vstack(padded)).save(output)
    click.echo(f"Wrote {output} ({len(rows)} rows)")


# ----- review -----


@cli.command()
@click.option(
    "--suspects",
    "suspects_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_SUSPECTS,
    show_default=True,
    help="Suspect list produced by `xval` (sorted by confidence).",
)
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_LABELS,
    show_default=True,
    help="labels.json to update if you accept any corrections.",
)
@click.option(
    "--samples",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SAMPLES,
    show_default=True,
    help="Directory containing the captcha PNGs.",
)
@click.option(
    "--min-conf",
    default=0.0,
    show_default=True,
    type=float,
    help="Minimum suspect confidence to include.",
)
@click.option(
    "--max-conf",
    default=1.0,
    show_default=True,
    type=float,
    help="Maximum suspect confidence to include.",
)
@click.option("--start", default=0, show_default=True, type=int, help="Skip the first N suspects.")
@click.option(
    "--no-image",
    is_flag=True,
    help="Print the file path instead of an inline image (fallback for terminals without graphics).",
)
@click.option(
    "--scale",
    default=4,
    show_default=True,
    type=int,
    help="Upscale factor applied to the captcha before sending it to the terminal.",
)
@click.option(
    "--term-rows",
    default=8,
    show_default=True,
    type=int,
    help="Image height in terminal rows.",
)
@click.option(
    "--protocol",
    type=click.Choice(["auto", "iterm", "kitty"]),
    default="auto",
    show_default=True,
    help="Inline image protocol; auto detects from $TERM / $TERM_PROGRAM.",
)
@click.option(
    "--pair",
    default=None,
    help="Only review confusions between two characters, e.g. --pair n,h "
    "(matches both label=n pred=h and label=h pred=n).",
)
def review(
    suspects_path: Path,
    labels_path: Path,
    samples: Path,
    min_conf: float,
    max_conf: float,
    start: int,
    *,
    no_image: bool,
    scale: int,
    term_rows: int,
    protocol: str,
    pair: str | None,
) -> None:
    """Interactive triage of xval suspects: keep / use prediction / enter / skip."""
    proto = detect_terminal_protocol() if protocol == "auto" else protocol
    pair_chars: set[frozenset[str]] | None = None
    if pair:
        parts = [p.strip() for p in pair.split(",")]
        if len(parts) != PAIR_LEN or any(len(p) != 1 for p in parts):
            raise click.BadParameter(f"--pair must be two characters separated by a comma, e.g. n,h (got {pair!r})")
        pair_chars = {frozenset(parts)}
    suspects = parse_suspects(suspects_path)
    suspects = [s for s in suspects if min_conf <= s[4] <= max_conf]
    if pair_chars is not None:
        suspects = [s for s in suspects if frozenset((s[2], s[3])) in pair_chars]
    suspects = suspects[start:]
    if not suspects:
        click.echo("No suspects in range.")
        return
    labels: dict[str, str] = json.loads(labels_path.read_text())
    click.echo(
        f"Reviewing {len(suspects)} suspects (conf {min_conf}..{max_conf}); "
        f"image protocol: {proto if not no_image else 'none'}"
    )
    click.echo("Keys: enter/1=keep label, 2=use prediction, e=enter char, q=save & quit, ?=help")
    click.echo()

    decisions = {"keep": 0, "fix": 0}
    try:
        for i, (fname, pos, expected, predicted, conf) in enumerate(suspects, 1):
            current = labels.get(fname, "")
            if pos >= len(current) or current[pos] != expected:
                continue
            if no_image:
                click.echo(f"  -> {samples / fname}")
            else:
                img = annotated_suspect_image(samples, fname, pos, scale)
                print_inline(img, term_rows, proto)
            click.echo(f"[{i}/{len(suspects)}] {fname}  pos={pos + 1}  conf={conf:.2f}")
            click.echo(f"  1) {expected}  [enter]")
            click.echo(f"  2) {predicted}")
            click.echo("  e) enter a different character")
            click.echo("  q) save and quit")
            while True:
                choice = input("? ").strip().lower()
                if choice == "":
                    choice = "1"
                if choice in ("1", "2", "e", "q", "?"):
                    break
                click.echo("  unknown key; press enter, 2, e, q, or ?")
            if choice == "q":
                break
            if choice == "?":
                click.echo("  enter/1=keep label, 2=use prediction, e=enter custom char, q=save & quit")
                continue
            if choice == "1":
                decisions["keep"] += 1
            elif choice == "2":
                labels[fname] = current[:pos] + predicted + current[pos + 1 :]
                decisions["fix"] += 1
            elif choice == "e":
                while True:
                    custom = input("  character: ").strip()
                    if len(custom) == 1 and custom in ALPHABET:
                        break
                    click.echo(f"  must be a single character from the alphabet (0-9, a-z, A-Z); got {custom!r}")
                labels[fname] = current[:pos] + custom + current[pos + 1 :]
                decisions["fix"] += 1
            click.echo()
    except KeyboardInterrupt, EOFError:
        click.echo("\nInterrupted.")

    click.echo()
    click.echo(f"Decisions: kept={decisions['keep']}  fixed={decisions['fix']}")
    if decisions["fix"]:
        backup = labels_path.with_suffix(labels_path.suffix + ".cli.bak")
        shutil.copy(labels_path, backup)
        labels_path.write_text(json.dumps(labels, indent=2) + "\n")
        click.echo(f"Wrote {decisions['fix']} corrections to {labels_path}")
        click.echo(f"Backup at {backup}")
    else:
        click.echo("No corrections applied; labels.json unchanged.")


# ----- verify -----


@cli.command()
@click.argument("chars", type=str)
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_LABELS,
    show_default=True,
    help="labels.json to read and (if any corrections accepted) write back to.",
)
@click.option(
    "--samples",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_SAMPLES,
    show_default=True,
    help="Directory containing the captcha PNGs.",
)
@click.option(
    "--start",
    default=0,
    show_default=True,
    type=int,
    help="Skip the first N positions (useful when resuming).",
)
@click.option(
    "--no-image",
    is_flag=True,
    help="Print the file path instead of an inline image (fallback for terminals without graphics).",
)
@click.option(
    "--scale",
    default=4,
    show_default=True,
    type=int,
    help="Upscale factor applied to the captcha before sending it to the terminal.",
)
@click.option(
    "--term-rows",
    default=8,
    show_default=True,
    type=int,
    help="Image height in terminal rows.",
)
@click.option(
    "--protocol",
    type=click.Choice(["auto", "iterm", "kitty"]),
    default="auto",
    show_default=True,
    help="Inline image protocol; auto detects from $TERM / $TERM_PROGRAM.",
)
def verify(
    chars: str,
    labels_path: Path,
    samples: Path,
    start: int,
    *,
    no_image: bool,
    scale: int,
    term_rows: int,
    protocol: str,
) -> None:
    """
    Re-verify every position whose label is one of the given characters.

    Unlike review, this iterates over labels.json directly — it doesn't depend
    on suspects.txt — so it can catch systematic labelling biases that the
    model has memorised and so won't flag as disagreements.

        ./manage.py verify nh             # all n and h positions
        ./manage.py verify Kk             # all K and k positions
        ./manage.py verify n,h            # commas optional
    """
    target_chars = set(chars.replace(",", ""))
    invalid = sorted(c for c in target_chars if c not in ALPHABET)
    if invalid:
        raise click.BadParameter(f"invalid characters (must be a-z, A-Z, 0-9): {invalid}")

    proto = detect_terminal_protocol() if protocol == "auto" else protocol
    labels: dict[str, str] = json.loads(labels_path.read_text())

    # Group by target character so the user stays in "one letter mode" rather
    # than jumping between letters within a captcha.
    todo: list[tuple[str, int, str]] = []
    for c in sorted(target_chars):
        for fname in sorted(labels):
            label = labels[fname]
            if len(label) != N_CHARS:
                continue
            for pos, ch in enumerate(label):
                if ch == c:
                    todo.append((fname, pos, c))
    todo = todo[start:]

    if not todo:
        click.echo(f"No labels containing {sorted(target_chars)}.")
        return

    captchas_touched = len({t[0] for t in todo})
    char_counts = {c: sum(1 for t in todo if t[2] == c) for c in sorted(target_chars)}
    click.echo(
        f"Verifying {len(todo)} positions across {captchas_touched} captchas "
        f"(chars: {sorted(target_chars)}); image protocol: "
        f"{proto if not no_image else 'none'}"
    )
    click.echo("Keys: Enter=keep, <char>=replace with that char, ?=help, Ctrl-C=save & quit")
    click.echo()

    decisions = {"keep": 0, "fix": 0}
    prev_char: str | None = None
    try:
        for i, (fname, pos, current_char) in enumerate(todo, 1):
            current = labels.get(fname, "")
            if pos >= len(current) or current[pos] != current_char:
                continue
            if current_char != prev_char:
                if prev_char is not None:
                    click.echo()
                click.echo("=" * 60)
                click.echo(f"  Now reviewing: '{current_char}'  ({char_counts[current_char]} positions)")
                click.echo("=" * 60)
                input("Press Enter to start (Ctrl-C to quit) ")
                click.echo()
                prev_char = current_char
            if no_image:
                click.echo(f"  -> {samples / fname}")
            else:
                img = annotated_suspect_image(samples, fname, pos, scale)
                print_inline(img, term_rows, proto)
            marked = current[:pos] + f"[{current_char}]" + current[pos + 1 :]
            click.echo(f"[{i}/{len(todo)}] {fname}  pos={pos + 1}  label: {marked}")
            while True:
                choice = input("? ").strip()
                if choice == "":
                    decisions["keep"] += 1
                    break
                if choice == "?":
                    click.echo(
                        f"  Enter=keep '{current_char}', type any single character to "
                        "replace (case-sensitive), Ctrl-C=save & quit"
                    )
                    continue
                if len(choice) == 1 and choice in ALPHABET:
                    labels[fname] = current[:pos] + choice + current[pos + 1 :]
                    decisions["fix"] += 1
                    break
                click.echo(f"  must be Enter, a single character from 0-9/a-z/A-Z, or ?; got {choice!r}")
            click.echo()
    except KeyboardInterrupt, EOFError:
        click.echo("\nInterrupted.")

    click.echo()
    click.echo(f"Decisions: kept={decisions['keep']}  fixed={decisions['fix']}")
    if decisions["fix"]:
        backup = labels_path.with_suffix(labels_path.suffix + ".verify.bak")
        shutil.copy(labels_path, backup)
        labels_path.write_text(json.dumps(labels, indent=2) + "\n")
        click.echo(f"Wrote {decisions['fix']} corrections to {labels_path}")
        click.echo(f"Backup at {backup}")
    else:
        click.echo("No corrections applied; labels.json unchanged.")


# ----- predict -----


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--model",
    type=click.Path(dir_okay=False, path_type=Path),
    default=DEFAULT_MODEL,
    show_default=True,
    help="Trained TinyCNN checkpoint to load.",
)
@click.option(
    "--confidence",
    is_flag=True,
    help="Also print the model's per-character softmax confidence.",
)
def predict_cmd(image_path: Path, model: Path, *, confidence: bool) -> None:
    """Predict a captcha's text. (Library API: import from predict.py.)."""
    from predict import predict, predict_with_confidence  # noqa: PLC0415  # local import to avoid cycle

    if confidence:
        text, conf = predict_with_confidence(image_path, model)
        click.echo(text)
        click.echo(" ".join(f"{c}={p:.2f}" for c, p in zip(text, conf, strict=False)))
    else:
        click.echo(predict(image_path, model))


# Click 8 doesn't support dashes in command names by default; rename via `name=`.
predict_cmd.name = "predict"


# ----- debug -----


@cli.command()
@click.argument("image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=HERE / "debug",
    show_default=True,
    help="Parent dir; outputs go into <out-dir>/<captcha_stem>/",
)
def debug(image_path: Path, out_dir: Path) -> None:
    """
    Write every intermediate preprocessing artefact for one captcha to disk.

    Useful for understanding why a particular suspect is what it is, or for
    spotting segmentation drift in the slice boundaries.
    """
    target = out_dir / image_path.stem
    target.mkdir(parents=True, exist_ok=True)

    rgb = load_original(image_path)
    Image.fromarray(rgb).save(target / "01_original.png")

    rgb_i = rgb.astype(np.int16)
    chroma = (rgb_i.max(axis=-1) - rgb_i.min(axis=-1)).astype(np.uint8)
    intensity = rgb_i.mean(axis=-1).astype(np.uint8)
    Image.fromarray(intensity).save(target / "02_intensity.png")
    Image.fromarray(chroma).save(target / "03_chroma.png")

    is_text = (chroma < CHROMA_MAX) & (intensity < INTENSITY_MAX)
    Image.fromarray((~is_text * 255).astype(np.uint8)).save(target / "04_is_text_mask.png")

    bridged = ndimage.binary_closing(is_text, structure=np.ones((3, 1), dtype=bool))
    Image.fromarray((~bridged * 255).astype(np.uint8)).save(target / "05_bridged_mask.png")

    labels, n = ndimage.label(bridged, structure=np.ones((3, 3), dtype=bool))
    rng = np.random.default_rng(0)
    palette = np.zeros((n + 1, 3), dtype=np.uint8)
    palette[0] = [255, 255, 255]
    for i in range(1, n + 1):
        palette[i] = rng.integers(40, 220, size=3)
    Image.fromarray(palette[labels]).save(target / "06_components.png")

    slices = ndimage.find_objects(labels)
    keep_ids = [i + 1 for i, sl in enumerate(slices) if (sl[0].stop - sl[0].start) >= MIN_GLYPH_HEIGHT]
    kept = np.isin(labels, keep_ids)
    Image.fromarray((~kept * 255).astype(np.uint8)).save(target / "07_kept_components.png")

    click.echo(f"{n} connected components in 05_bridged_mask:")
    for i, sl in enumerate(slices, 1):
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        keep = "KEEP" if (y1 - y0) >= MIN_GLYPH_HEIGHT else "drop"
        click.echo(
            f"  id={i:2d} bbox=(y={y0:>3d}..{y1:<3d} h={y1 - y0:2d}, x={x0:>3d}..{x1:<3d} w={x1 - x0:2d})  {keep}"
        )

    cleaned = clean(rgb)
    Image.fromarray(cleaned).save(target / "08_cleaned.png")

    left, right = text_bbox(cleaned)
    bbox_overlay = np.stack([cleaned] * 3, axis=-1)
    bbox_overlay[:, left] = [255, 0, 0]
    bbox_overlay[:, min(right - 1, cleaned.shape[1] - 1)] = [255, 0, 0]
    Image.fromarray(bbox_overlay).save(target / "09_bbox_on_cleaned.png")
    click.echo(f"\ntext_bbox: left={left}, right={right}, width={right - left}")

    # Steps 10/11/12/13 use char_x_ranges — the same slicing the CNN sees.
    ranges = char_x_ranges(rgb)
    slices_overlay = np.stack([cleaned] * 3, axis=-1)
    for x0, _ in ranges:
        slices_overlay[:, min(x0, cleaned.shape[1] - 1)] = [255, 0, 0]
    last_x = ranges[-1][1]
    slices_overlay[:, min(last_x, cleaned.shape[1] - 1) - 1] = [255, 0, 0]
    Image.fromarray(slices_overlay).save(target / "10_slices_on_cleaned.png")

    orig_overlay = rgb.copy()
    for x0, _ in ranges:
        orig_overlay[:, min(x0, rgb.shape[1] - 1)] = [255, 0, 0]
    orig_overlay[:, min(last_x, rgb.shape[1] - 1) - 1] = [255, 0, 0]
    Image.fromarray(orig_overlay).save(target / "11_slices_on_original.png")

    click.echo("\nper-slot horizontal slice (the CNN's input columns):")
    for i, (x0, x1) in enumerate(ranges):
        click.echo(f"  slot {i}: x={x0}..{x1} (width={x1 - x0})")

    crops_dir = target / "12_crops"
    crops_dir.mkdir(exist_ok=True)
    gray = rgb.mean(axis=-1).astype(np.uint8)
    for i, (x0, x1) in enumerate(ranges):
        Image.fromarray(gray[:, x0:x1]).save(crops_dir / f"crop_{i}.png")

    resized_dir = target / "13_crops_resized"
    resized_dir.mkdir(exist_ok=True)
    for i, (x0, x1) in enumerate(ranges):
        crop = gray[:, x0:x1]
        Image.fromarray(resize_to(crop)).save(resized_dir / f"crop_{i}.png")

    click.echo(f"\nWrote {len(list(target.rglob('*.png')))} files to {target}/")


if __name__ == "__main__":
    cli()
