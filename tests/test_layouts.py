"""CPU tests of every family's token layouts (no weights, no GPU, a few seconds).

    python tests/test_layouts.py

For each family and geometry: the dense layout and every budget layout tile the image exactly once with aligned tokens
of the allowed sizes; a budget is never exceeded, the dense budget gives the dense grid, smaller budgets never give
more tokens, a flat image merges completely; pack() keeps every token.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(family):
    spec = importlib.util.spec_from_file_location(f"{family}_layout", os.path.join(ROOT, family, "layout.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m            # dataclasses look their module up
    spec.loader.exec_module(m)
    return m


def images(geo, n=3, seed=0):
    """Half noise, half flat, plus a smooth gradient: content that merges unevenly."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (n, 3, geo.img, geo.img), generator=g, dtype=torch.uint8)
    x[:, :, :, : geo.img // 2] = 128
    ramp = torch.linspace(0, 255, geo.img).round().to(torch.uint8)
    x[-1] = ramp[None, :, None].expand(3, geo.img, geo.img)
    return x


def check_tiling(toks, geo, what):
    cover = np.zeros((geo.img, geo.img), dtype=np.int32)
    for s, y, x in toks:
        assert s in geo.scales, f"{what}: token size {s} not in {geo.scales}"
        assert y % s == 0 and x % s == 0, f"{what}: token ({s}, {y}, {x}) is not aligned"
        cover[y:y + s, x:x + s] += 1
    assert (cover == 1).all(), f"{what}: tokens do not tile the image exactly once"


def check_family(family, geos):
    L = load(family)
    n = 0
    for kw in geos:
        geo = L.Geometry(**kw)
        dense = L.all_base_tokens(geo)
        assert len(dense) == geo.n_dense
        check_tiling(dense, geo, f"{family} dense")
        imgs = images(geo)
        pyr = L.detail_pyramid(imgs, geo)
        budgets = [geo.n_dense, int(0.75 * geo.n_dense), geo.n_dense // 2, geo.n_dense // 4, geo.n_min]
        prev = None
        for b in budgets:
            lays = L.layouts_for_budget(pyr, geo, b)
            counts = [len(t) for t in lays]
            for i, t in enumerate(lays):
                check_tiling(t, geo, f"{family} img {geo.img} budget {b} image {i}")
            assert max(counts) <= b, f"{family}: budget {b} exceeded ({counts})"
            if b == geo.n_dense:
                assert counts == [geo.n_dense] * len(lays), f"{family}: dense budget gave {counts}"
            if prev is not None:
                assert all(c <= p for c, p in zip(counts, prev)), f"{family}: budget {b} gave more tokens than a larger one"
            prev = counts
            n += 1
        flat = torch.full((1, 3, geo.img, geo.img), 77, dtype=torch.uint8)
        lay = L.layouts_for_budget(L.detail_pyramid(flat, geo), geo, geo.n_min)[0]
        assert len(lay) == geo.n_min and all(s == geo.scales[-1] for s, _, _ in lay), f"{family}: flat image did not merge"
        packed = L.pack(lays, geo, "cpu") if family != "jit" else L.pack(lays, geo)
        assert [int(v) for v in packed["n_per"]] == [len(t) for t in lays], f"{family}: pack lost tokens"
        n += 2
    return n


FAMILIES = {
    "jit": [dict(img=256, base_patch=16, scales=(16, 32, 64)), dict(img=512, base_patch=32, scales=(32, 64, 128))],
    "minit2i": [dict(img=512, base_patch=16, scales=(16, 32, 64), thresholds=(6.0, 4.0))],
    "pixeldit": [dict(img=1024, base_patch=16, scales=(16, 32, 64), thresholds=(6.0, 4.0))],
    "pmf": [dict(img=256, base_patch=16, scales=(16, 32, 64), thresholds=(6.0, 4.0))],
}

if __name__ == "__main__":
    total = 0
    for fam, geos in FAMILIES.items():
        k = check_family(fam, geos)
        total += k
        print(f"[layouts] {fam}: {k} checks passed", flush=True)
    print(f"[layouts] all {total} checks passed")
