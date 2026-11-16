"""Check an XeFP4 encoder against the conformance vectors in spec/vectors/.

usage: python tools/check_vectors.py [--vectors spec/vectors] [--encoder module:function]

The encoder takes W (float32 [N, K]) and returns (codes uint8 [N, K/2], fac uint32 [N, FW], rs float16 [N]).
The default is the reference encoder, xefp4_ref:encode. The reference decoder is always checked against W_hat.
Exit code 0 means every case passed.
"""
import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

import xefp4_ref as R


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", default=str(Path(__file__).resolve().parent.parent / "spec" / "vectors"))
    ap.add_argument("--encoder", default="xefp4_ref:encode")
    a = ap.parse_args()
    mod, fn = a.encoder.split(":")
    encode = getattr(importlib.import_module(mod), fn)
    vdir = Path(a.vectors)
    index = json.loads((vdir / "vectors.json").read_text(encoding="utf-8"))
    bad = skipped = 0
    for name, meta in index["cases"].items():
        v = dict(np.load(vdir / f"{name}.npz"))
        errs = [k for k, h in meta["sha256"].items() if sha(v[k]) != h]
        if errs:
            print(f"FAIL {name}: vector file damaged ({', '.join(errs)})")
            bad += 1
            continue
        try:
            codes, fac, rs = encode(v["W"])
        except NotImplementedError as e:
            print(f"skip {name:20s} {meta['description']}  ({e})")
            skipped += 1
            continue
        got = {"codes": np.asarray(codes, np.uint8), "fac": np.asarray(fac).astype("<u4"),
               "rs": np.asarray(rs, np.float16)}
        live = v["rs"].astype(np.float32) != 0            # rows with rs == 0: codes and factors carry no meaning
        diff = []
        for k in got:
            if got[k].shape != v[k].shape:
                diff.append(k)
            elif k == "rs":
                if got[k].tobytes() != v[k].tobytes():
                    diff.append(k)
            elif got[k][live].tobytes() != v[k][live].tobytes():
                diff.append(k)
        dec = R.decode(v["codes"], v["fac"], v["rs"], v["W"].shape[1])
        if dec.tobytes() != v["W_hat"].tobytes():
            diff.append("decode")
        print(f"{'ok  ' if not diff else 'FAIL'} {name:20s} {meta['description']}" + (f"  -> {diff}" if diff else ""))
        bad += bool(diff)
    print(f"{len(index['cases']) - bad - skipped}/{len(index['cases'])} cases passed, {skipped} skipped, {bad} failed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
