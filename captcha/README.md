# Assam Tenders Captcha Solver

A small PyTorch CNN that reads 6-character captchas from the Assam tenders portal. Trained on 800 labeled samples, currently scores **99.6% per-character** on held-out validation.

## Setup

Local virtualenv via `uv`:

```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

After that, every script in this directory just uses `#!/usr/bin/env python` and picks up the activated venv. All ops go through `./manage.py`; the library API is `predict.py`.

## Quickstart

```python
from predict import predict, predict_with_confidence

text = predict("samples/captcha_0001.png")          # "15u2A3"
text = predict(png_bytes)                            # raw PNG bytes
text = predict(base64_str)                           # base64 string, with or without "data:image/png;base64," prefix
text = predict(pil_image)                            # PIL.Image

text, conf = predict_with_confidence(image)
# text  -> "15u2A3"
# conf  -> [0.98, 0.75, 1.00, 0.98, 1.00, 0.99]   (one float per char)
```

CLI (same prediction, no library import):

```sh
./manage.py predict samples/captcha_0001.png
./manage.py predict samples/captcha_0001.png --confidence
```

A retry-aware caller can refresh the captcha when `min(conf) < 0.5` rather than blind-retrying on rejection.

## How it works

1. **Preprocess** (`manage.py`): composite the PNG over white, drop chromatic noise (colored lines/dots) via a chroma + intensity filter, drop short connected components (pepper noise), bridge 1-pixel vertical gaps inside glyphs, find the connected components of the surviving text.
2. **Segment**: if exactly 6 components remain, each crop is a uniform-width slot centered on its component (so wide letters like `B` aren't chopped by the slot boundary). When cleaning loses or merges a glyph, fall back to dividing the full text bbox into 6 equal parts.
3. **Classify** each crop with `TinyCNN` (defined in `manage.py`): three conv blocks with channel widths **32 → 64 → 128** (each `Conv2d(3×3) + BatchNorm + ReLU`, max-pool after the first two), then `AdaptiveAvgPool2d((2, 1))` → 256-dim feature vector (two vertical bins × 128 channels) → `Linear(256, 62)`. Roughly 100k parameters total. The (2, 1) pool preserves a top-half/bottom-half signal, which is what distinguishes pairs like p/b (descender vs ascender) and n/h (no ascender vs ascender). Concatenate the 6 predictions into the final string.

The captcha is case-sensitive, so the alphabet is 62 classes (`0-9 a-z A-Z`). See [A note on model size](#a-note-on-model-size) for why this architecture isn't a tuning lever.

## CLI overview

```sh
./manage.py --help                       # list commands
./manage.py fetch 500 --start 301        # download captchas 0301..0800
./manage.py labelsheet --start 801 --end 830 -o sheet.png
./manage.py train --epochs 60            # train, save best-by-val to captcha_cnn.pt
./manage.py xval                         # 5-fold cross-validation, find mislabels
./manage.py review                       # interactive triage of suspects.txt
./manage.py verify nh                    # re-verify every label position whose char is n or h
./manage.py predict samples/captcha_0001.png
```

## Refining the model

The training data has some labeling noise (mostly subtle case calls — K/k, S/s — and i/1/I/J confusions). The model accuracy is currently labeling-noise-bound. To push higher, either add more samples or clean more labels.

### Add more samples

```sh
# fetch 200 more samples, numbered captcha_0801.png .. captcha_1000.png
./manage.py fetch 200 --start 801
```

Then label them, then retrain.

### Label new samples

`./manage.py labelsheet` tiles N captchas into a single PNG for transcription:

```sh
./manage.py labelsheet --start 801 --end 830 -o sheet.png
# open sheet.png, write down the 6 characters for each row,
# then add the entries to labels.json:
#   "captcha_0801.png": "Xy3aBc",
#   "captcha_0802.png": "...",
```

`labels.json` is just a dict of `{filename: "6chars"}`. Hand-edit it freely. Backups from prior cleanup rounds are kept as `labels.json.bak`, `labels.json.iter1.bak`, etc.

### Find likely mislabels (cross-validation)

`./manage.py xval` trains 5 models on 5-fold splits and reports characters where the held-out model confidently disagrees with the current label. These are likely mislabels — `xval` does NOT need a pre-trained model, it trains fresh ones internally.

```sh
./manage.py xval    # ~5 min on CPU
# writes suspects.txt (sorted by model confidence)
# writes suspects.png (contact sheet for visual review)
```

Look at the top entries in `suspects.txt`:

```
captcha_0113.png  pos=4  label='G'  pred='6'  conf=0.999
captcha_0679.png  pos=2  label='B'  pred='3'  conf=0.996
...
```

`pos` is the 0-indexed character position. At high confidence (`conf >= 0.85`), the model is right ~85-90% of the time in our experience. Lower confidence is more mixed.

Tuning knobs: `--folds 5 --epochs 40` (defaults). Training within each fold typically plateaus by epoch ~30 — reducing `--epochs 30` saves ~25% of runtime with negligible accuracy loss.

Already have a `suspects.txt` from a previous run and just want to re-print the summary (top confusion pairs etc.) without retraining?

```sh
./manage.py xval --report-only
```

### Find systematic mislabels (verify)

xval has a blind spot: if you mislabeled the same character the same way in many captchas, the model learns the wrong mapping. On the held-out fold it predicts what you taught it — agreeing with the wrong label — so the mistake never shows up as a suspect. This is most likely on visually-ambiguous pairs like `n/h`, `p/b`, `K/k`, `S/s`, `V/v`, `J/j`, `1/I/L`.

`./manage.py verify <chars>` iterates over every position in `labels.json` whose label is one of the given characters, regardless of what the model thinks, and lets you confirm or correct each:

```sh
./manage.py verify nh        # all n and h positions
./manage.py verify Kk        # all K and k positions
./manage.py verify pbnh      # multiple pairs in one pass
```

The signal to act on this comes from the asymmetric pairs in `xval`'s pair report. If you see `n → h : 6 (one-directional)` with no balancing `h → n` cases, that asymmetry hints at a systematic labeling bias — and `verify nh` will surface the captchas you're labeling inconsistently with the model's learned norm.

Like `review`, corrections are written to `labels.json` with a `.verify.bak` backup.

### Apply xval corrections

Three ways (`verify` above is a fourth, complementary approach):

**Interactive CLI** (`./manage.py review`) — recommended for careful review. Shows each suspect captcha inline in the terminal with a red box around the suspect character. Press `Enter` (or `1`) to keep your label, `2` to accept the prediction, `e` to enter a different character, `q` to save and quit. Corrections are written to `labels.json` at the end with a `.cli.bak` backup of the previous file.

Supports two image protocols, auto-detected from your terminal:
- **iTerm2 protocol** — iTerm2, WezTerm
- **Kitty protocol** — Kitty, Ghostty, WezTerm

```sh
./manage.py review                       # all suspects, sorted by confidence
./manage.py review --min-conf 0.85       # focus on high-confidence ones only
./manage.py review --start 30            # resume after a previous session
./manage.py review --protocol kitty      # force a specific protocol
./manage.py review --no-image            # fallback: print file paths instead
```

**Manual** — open `labels.json`, change the relevant character. Best for the borderline cases the model gets wrong.

**Bulk auto-apply by confidence** — for high-confidence suspects, you can accept the model's prediction wholesale:

```python
import json, re, shutil
labels = json.load(open("labels.json"))
suspects = []
with open("suspects.txt") as f:
    for line in f:
        m = re.match(r"(\S+)\s+pos=(\d+)\s+label='(.)'\s+pred='(.)'\s+conf=([\d.]+)", line.strip())
        if m:
            suspects.append((m.group(1), int(m.group(2)), m.group(3), m.group(4), float(m.group(5))))

suspects.sort(key=lambda s: -s[4])
TOP_N = 30                                              # tune this
for fname, pos, expected, predicted, _ in suspects[:TOP_N]:
    cur = labels[fname]
    if cur[pos] == expected:
        labels[fname] = cur[:pos] + predicted + cur[pos+1:]

shutil.copy("labels.json", "labels.json.bak")
json.dump(labels, open("labels.json", "w"), indent=2)
```

Start with a small `TOP_N` (~30) and retrain; expand if accuracy keeps climbing.

### Retrain

```sh
./manage.py train                              # default: 60 epochs, AdamW + cosine LR
./manage.py train --epochs 100 --lr 5e-4       # tune hyperparams
```

Outputs:
- Console log with per-epoch `train loss / train acc / val acc`. The `*` marker indicates a new best val acc.
- `captcha_cnn.pt` — best model by val accuracy

Per-char val accuracy is the headline number. Per-captcha accuracy is roughly `val_acc^6`.

### Verify the full pipeline

After retraining, sanity-check a few samples:

```sh
./manage.py predict samples/captcha_0001.png --confidence
```

And evaluate against your labels:

```python
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from predict import predict

labels = json.load(open("labels.json"))
correct_char = correct_cap = total = 0
for fname, label in labels.items():
    pred = predict(Path("samples") / fname)
    correct_char += sum(p == l for p, l in zip(pred, label))
    correct_cap += int(pred == label)
    total += 1

print(f"per-char: {correct_char}/{total*6} = {correct_char/(total*6):.1%}")
print(f"per-captcha: {correct_cap}/{total} = {correct_cap/total:.1%}")
```

(These numbers are contaminated by training-set memorization. The honest accuracy comes from `./manage.py train`'s val output.)

## Files

In roughly the order you'd touch them in the data → train → refine workflow:

| File | Purpose |
|---|---|
| `requirements.txt` | Python deps. Install with `uv pip install -r requirements.txt` into the local `.venv`. |
| `manage.py` | The CLI plus all model + image-processing code. Subcommands: `fetch`, `labelsheet`, `train`, `xval`, `review`, `predict`, `debug`. |
| `predict.py` | Public Python API for callers: `predict(image)`, `predict_with_confidence(image)`. |

Data files:

| File | Purpose |
|---|---|
| `samples/` | Captcha PNGs (`captcha_NNNN.png`). |
| `labels.json` | `{filename: "6chars"}` mapping — hand-edited or updated by the cleanup tools. |
| `captcha_cnn.pt` | Trained model weights (output of `manage.py train`, input to `predict.py`). |
| `suspects.txt`, `suspects.png` | xval outputs — list of likely mislabels and a contact-sheet preview. |
| `.venv/` | Local Python virtualenv (created by `uv venv`). Not committed. |

## Tuning ideas

If you want to push beyond the current per-char accuracy:

1. **More cleanup iterations of `manage.py xval`** — diminishing returns past iteration 2 in our experience.
2. **More data** — pushes past the labeling-noise ceiling. Aim for 1500-2000 labeled.
3. **Inspect failure modes** — look at the cross-val disagreements that *aren't* obviously mislabels. Those are systematic model weaknesses that need a code change, not a label fix. The per-component slot centering in `char_x_ranges` came from exactly this: a "suspect" turned out to be a correctly-labeled captcha whose crop was misaligned by uniform slicing, chopping a wide letter so it visually read as a different character. Look for: characters that the model gets wrong consistently across many captchas, ones where `min(conf)` in `predict_with_confidence` is low on otherwise-clear images, and ones where `./manage.py debug <captcha>`'s `10_slices_on_cleaned.png` shows a slice boundary cutting through a glyph.

### A note on model size

The conventional advice when accuracy plateaus is "bigger CNN" — more channel width, more layers, more parameters. **For this captcha that won't help.** TinyCNN's train/val gap is already small (~1.5 pp), which means the model isn't underfitting — adding capacity won't reach features it's currently missing. The remaining error is at the labels (case calls like K/k that even a perfect model can't beat without ground-truth height information) or at upstream segmentation (the slot-centering fix). Going *smaller* doesn't help either — you'd match current accuracy with a slightly faster, smaller checkpoint, but nothing else changes. The architecture is the right size for this task.

### A note on augmentation

The standard CNN-tuning playbook recommends piling on geometric augmentations — random affine, scale, shear, cutout — when accuracy plateaus. **For this captcha that won't help.** The generator doesn't rotate, translate, scale, or shear its letters, so those transforms train the model on a distribution it'll never see at inference. The noise that does appear in real captchas (per-pixel pepper, occasional thin white streaks from cleaned chroma lines) is already covered by the salt + pepper in `CharDataset._augment`. If you want to push augmentation further, the on-distribution moves are slightly higher pepper density or simulated thin white streaks — not the geometric transforms. In our experience the bigger lever at this stage is cleaning more labels, not augmenting harder.
