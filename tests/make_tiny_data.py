"""Tiny synthetic training sets for the training tests (no dataset download needed).

    python tests/make_tiny_data.py --out runs/tests/data [--jit_from checkpoints/budgetpix_jit_b16_256.pth]

Writes, under --out:
  imagenet/train-00000.parquet   ImageNet-1k parquet format (image.bytes JPEG + label), for the JiT and pMF trainers
  minit2i/chunk_000000.pt        MiniT2I chunk format (uint8 pixels + flan-t5 token ids), for the MiniT2I trainer
  pixeldit_images/               images with .txt captions, the input of pixeldit/scripts/precompute_data.py
  jit_released.pth               with --jit_from: a JiT checkpoint in the released format (`model_ema1`), made from the
                                 backbone of a BudgetPix JiT checkpoint, for the JiT trainer (the released JiT weights
                                 are distributed outside the Hugging Face Hub)
The images are random shapes on gradients with noisy patches, so the entropy layouts vary across the image.
"""
import argparse
import importlib.util
import io
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTIONS = ["a red circle on a blue gradient", "two yellow squares on a dark background",
            "a green triangle next to a noisy patch", "colourful shapes on a smooth sky"]


def synthetic(size, k, rng):
    y, x = np.mgrid[0:size, 0:size] / size
    a = rng.uniform(0, 1, 3)
    img = np.stack([255 * (a[c] * x + (1 - a[c]) * y) for c in range(3)], -1).astype(np.uint8)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    for _ in range(4):
        x0, y0 = rng.integers(0, size * 3 // 4, 2)
        w = int(rng.integers(size // 8, size // 3))
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([int(x0), int(y0), int(x0) + w, int(y0) + w], fill=col)
    arr = np.array(pil)
    py, px = rng.integers(0, size // 2, 2)
    arr[py:py + size // 4, px:px + size // 4] = rng.integers(0, 256, (size // 4, size // 4, 3))
    return Image.fromarray(arr)


def jpeg(pil):
    b = io.BytesIO()
    pil.save(b, format="JPEG", quality=92)
    return b.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32, help="images per set")
    ap.add_argument("--jit_from", default="", help="a BudgetPix JiT checkpoint, to write jit_released.pth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)

    # ImageNet parquet
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = os.path.join(a.out, "imagenet")
    os.makedirs(d, exist_ok=True)
    rows = [{"bytes": jpeg(synthetic(320, i, rng)), "path": f"{i:05d}.jpg"} for i in range(a.n)]
    table = pa.table({"image": rows, "label": [int(v) for v in rng.integers(0, 1000, a.n)]})
    pq.write_table(table, os.path.join(d, "train-00000.parquet"), row_group_size=16)
    print("[data]", os.path.join(d, "train-00000.parquet"), flush=True)

    # MiniT2I chunk
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("google/flan-t5-large")
    caps = [CAPTIONS[i % len(CAPTIONS)] for i in range(a.n)]
    enc = tok(caps, padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    pix = torch.stack([torch.from_numpy(np.array(synthetic(512, i, rng))).permute(2, 0, 1) for i in range(a.n)])
    d = os.path.join(a.out, "minit2i")
    os.makedirs(d, exist_ok=True)
    torch.save({"pixel_values": pix, "input_ids": enc["input_ids"].long(), "attention_mask": enc["attention_mask"].long(),
                "caption": caps, "key": [f"tiny/{i}" for i in range(a.n)]}, os.path.join(d, "chunk_000000.pt"))
    print("[data]", os.path.join(d, "chunk_000000.pt"), flush=True)

    # PixelDiT image folder (scripts/precompute_data.py --images encodes it)
    d = os.path.join(a.out, "pixeldit_images")
    os.makedirs(d, exist_ok=True)
    for i in range(min(a.n, 8)):
        synthetic(1024, i, rng).save(os.path.join(d, f"{i:03d}.jpg"), quality=92)
        open(os.path.join(d, f"{i:03d}.txt"), "w").write(CAPTIONS[i % len(CAPTIONS)])
    print("[data]", d, flush=True)

    if a.jit_from:
        spec = importlib.util.spec_from_file_location("jit_model", os.path.join(ROOT, "jit", "model.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = m
        sys.path.insert(0, os.path.join(ROOT, "jit"))
        spec.loader.exec_module(m)
        ck = torch.load(a.jit_from, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        base = {k: v for k, v in sd.items() if not k.startswith(m.NEW_MODULES)}
        torch.save({"model_ema1": base}, os.path.join(a.out, "jit_released.pth"))
        print(f"[data] {os.path.join(a.out, 'jit_released.pth')} ({len(base)} of {len(sd)} tensors)", flush=True)


if __name__ == "__main__":
    main()
