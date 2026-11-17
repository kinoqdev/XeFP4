"""Self-test: fixed chat prompts, 64 greedy tokens each, compared with the reference token hashes in reference.json
(recorded on Arc A770; Arc 140T gives the same tokens). Qwen3.5 also checks that MTP decoding returns the same tokens
as plain decoding. Ends with a decode-speed figure (at least 3 seconds of decoding)."""
from __future__ import annotations

import hashlib
import json
import os

PROMPTS = {
    "en": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
    "code": [{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively."}],
    "ko": [{"role": "user", "content": "대한민국의 수도와 그 도시의 특징을 세 문장으로 설명해 줘."}],
}
N = 64
REF = os.path.join(os.path.dirname(__file__), "reference.json")


def thash(toks):
    return hashlib.sha256(json.dumps([int(t) for t in toks]).encode()).hexdigest()[:16]


def run(model, ctx=None):
    from .engines import open_model
    eng, chat = open_model(model, ctx=ctx or 4096)
    key = os.path.basename(eng.files["main"])
    ref = json.load(open(REF, encoding="utf-8")).get(key, {}) if os.path.exists(REF) else {}
    res, ok = {}, True
    for name, msgs in PROMPTS.items():
        ids, _ = chat.encode(msgs)
        out, _, _ = eng.generate(ids, N, reuse=False)
        h = thash(out); res[name] = h
        line = f"{name:5s} {len(out):3d} tokens  {h}"
        if getattr(eng, "sp", None) is not None:                       # same tokens without MTP
            sp, eng.sp = eng.sp, None
            out2, _, _ = eng.generate(ids, N, reuse=False); eng.sp = sp
            same = out2 == out; ok &= same; line += "  MTP = plain" if same else "  MTP != plain (FAIL)"
        if name in ref:
            good = ref[name] == h; ok &= good; line += "  ref OK" if good else f"  ref {ref[name]} (FAIL)"
        else:
            line += "  (no reference)"
        print(line, flush=True)
    ids, _ = chat.encode(PROMPTS["en"]); t, n = 0.0, 0
    while t < 3.0:
        out, _, (tp, tg, k, _) = eng.generate(ids, 256, reuse=False); t += tg; n += max(k - 1, 0)
    print(f"decode {n / t:.1f} tok/s ({'MTP' if getattr(eng, 'sp', None) else 'plain'}, {n} tokens in {t:.1f}s)", flush=True)
    print("PASS" if ok and ref else ("no reference for " + key if not ref else "FAIL"), flush=True)
    if os.environ.get("XEQ_SELFTEST_RECORD"):
        allr = json.load(open(REF, encoding="utf-8")) if os.path.exists(REF) else {}
        allr[key] = res; json.dump(allr, open(REF, "w", encoding="utf-8"), indent=1); print("recorded", key)
    return ok
