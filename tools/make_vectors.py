"""Generate the XeFP4 v1.0 conformance vectors in spec/vectors/ with the reference encoder.

usage: python tools/make_vectors.py [--out spec/vectors]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), "."]   # find xefp4_ref and --encoder modules
import xefp4_ref as R  # noqa: E402

F = np.float32


def cases():
    rng = np.random.default_rng(20261117)
    yield "gauss", "random Gaussian rows", (rng.standard_normal((8, 256)) * 0.02).astype(F)
    yield "heavy_tail", "Student-t (3) rows with outliers", (rng.standard_t(3, (8, 256)) * 0.02).astype(F)

    # exact factor tie: block 1 is all 6·rs, exact with m=8 (g=6) and with m=4 (g=12) -> smaller m (4) wins
    w = np.zeros((2, 64), F)
    w[:, :32] = 12.0
    w[0, 32:] = 6.0
    w[1, 32:] = -3.0                       # exact with m=2 (g=12), m=4 (g=6), m=8 (g=3) -> m=2
    yield "factor_tie", "blocks where several factors are exact; the smallest m must win", w

    # every rounding boundary at m = 8 (the block also holds 12 so that m = 8 is a candidate)
    b = np.array([0.5, 1.5, 2.5, 3.5, 5, 7, 10], F)
    row = np.concatenate([[12.0], b, -b, b + 1e-3, b - 1e-3, [-12.0, 0.25, 0]]).astype(F)
    assert row.size == 32, row.size
    w = np.tile(row, (2, 2))
    w[1] *= 0.37
    yield "boundaries", "values on and next to every magnitude boundary", w

    w = (rng.standard_normal((2, 64)) * 0.5).astype(F)
    w[:, ::3] = -0.0
    yield "negative_zero", "-0.0 inputs must be stored as code 0", w

    w = (rng.standard_normal((4, 64)) * 0.1).astype(F)
    w[0] = 0.0                                                      # all-zero row: rs rounds to 0
    w[1] = (rng.standard_normal(64) * 1e-8).astype(F)               # max|W| < 7e-7: rs rounds to 0
    w[2] = (rng.standard_normal(64) * 2e-5).astype(F)               # FP16 subnormal row scale
    w[3, :32] = 0.0                                                  # all-zero block in a normal row
    yield "zero_and_tiny_rows", "row scale 0, subnormal row scale, all-zero block", w

    w = (rng.standard_normal((2, 64)) * 1e4).astype(F)
    w[:, 5] = 65504.0 * 12                                          # largest encodable row: rs = 65504
    yield "largest_row", "row scale at the FP16 maximum", w

    yield "k64", "K = 64: 6 factor bits, last word padded", (rng.standard_normal((3, 64)) * 0.02).astype(F)
    yield "k96", "K = 96: 9 factor bits", (rng.standard_normal((3, 96)) * 0.02).astype(F)
    yield "k1408", "K = 1408 (Gemma expert width): 132 factor bits, padded to 5 words", \
        (rng.standard_normal((2, 1408)) * 0.02).astype(F)

    # all of the above at K = 1024 (factor bits fill whole words), so converters limited to such K can be checked
    w = (rng.standard_normal((8, 1024)) * 0.02).astype(F)
    w[1] = (rng.standard_t(3, 1024) * 0.02).astype(F)
    w[2, :32], w[2, 32:64], w[2, 64:96] = 12.0, 6.0, -3.0                 # factor ties
    w[3] = np.tile(row, 32) * 0.37                                          # boundaries
    w[4, ::3] = -0.0
    w[5] = 0.0
    w[6] = (rng.standard_normal(1024) * 2e-5).astype(F)
    w[7, 7] = 65504.0 * 12
    yield "mixed_k1024", "the edge cases above at K = 1024", w


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "spec" / "vectors"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    index = {"xefp4_version": "1.0", "fmt": "xe8", "G": R.G, "cases": {}}
    for name, desc, W in cases():
        codes, fac, rs = R.encode(W)
        W_hat = R.decode(codes, fac, rs, W.shape[1])
        arrs = {"W": W, "codes": codes, "fac": fac, "rs": rs, "W_hat": W_hat}
        np.savez(out / f"{name}.npz", **arrs)
        index["cases"][name] = {"description": desc, "shape": list(W.shape),
                                "sha256": {k: sha(v) for k, v in arrs.items()}}
        print(f"{name:20s} {str(W.shape):12s} rs = {rs.astype(np.float32).min():.3g}..{rs.astype(np.float32).max():.3g}")
    (out / "vectors.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
