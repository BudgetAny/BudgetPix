"""Compare two folders of sample images pixel for pixel (same file names).

    python tests/compare_images.py <new> <reference> [--tol 0]

Prints, per image, the largest absolute difference (0-255) and the share of differing pixels; exits non-zero when an
image is missing or differs by more than --tol. Use it to check that a code change leaves the samples unchanged.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new"); ap.add_argument("ref")
    ap.add_argument("--tol", type=int, default=0, help="largest allowed per-pixel difference")
    a = ap.parse_args()
    names = sorted(f for f in os.listdir(a.ref) if f.endswith(".png"))
    if not names:
        sys.exit(f"no .png in {a.ref}")
    bad = 0
    for f in names:
        p = os.path.join(a.new, f)
        if not os.path.exists(p):
            print(f"[compare] {f}: MISSING"); bad += 1; continue
        x = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
        y = np.asarray(Image.open(os.path.join(a.ref, f)).convert("RGB"), dtype=np.int16)
        if x.shape != y.shape:
            print(f"[compare] {f}: shape {x.shape} != {y.shape}"); bad += 1; continue
        d = np.abs(x - y)
        ok = int(d.max()) <= a.tol
        bad += not ok
        print(f"[compare] {f}: max diff {int(d.max())}, {100 * float((d > 0).any(-1).mean()):.3f} % of pixels differ"
              f"{'' if ok else '  <-- FAIL'}")
    print(f"[compare] {len(names) - bad}/{len(names)} images match (tol {a.tol})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
