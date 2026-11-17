"""The published benchmark: decode speed by context position, prefill speed and time to first token, VRAM.
Every timed measurement runs for at least 3 seconds (short ones are repeated and summed) and is read at the end.
Speed does not depend on the token values except for MTP (Qwen3.5), whose acceptance depends on the text:
pass --text FILE to measure MTP on your own text; without it a fixed English prompt is used for the short case only."""
from __future__ import annotations

import json
import os
import platform
import statistics
import time

import numpy as np
import pyopencl as cl

MIN_T = 3.0


def rep_until(fn):
    n, t = 0, 0.0
    while t < MIN_T:
        t0 = time.perf_counter(); fn(); t += time.perf_counter() - t0; n += 1
    return n, t


def env():
    from .device import device
    d = device()
    return {"time": time.strftime("%Y-%m-%d %H:%M"), "os": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
            "pyopencl": cl.VERSION_TEXT, "cpu": platform.processor(),
            "gpu": {"name": d.name, "driver": d.driver_version, "opencl": d.version, "compute_units": d.max_compute_units,
                    "max_clock_mhz": d.max_clock_frequency, "global_mem_gib": round(d.global_mem_size / 2**30, 2),
                    "split_dpas": "cl_intel_subgroup_split_matrix_multiply_accumulate" in d.extensions},
            "conditions": {"min_measure_s": MIN_T, "batch": 1, "decoding": "greedy"}}


def synth(n, vocab):
    ids = (np.arange(n + 8) * 7919 % (vocab - 2000) + 1000).astype(np.int32); return ids


def gemma(eng):
    from .device import vram
    m, pf = eng.m, eng.pf
    r = {"tmax": m.TM, "vram_alloc_gib": round(vram() / 2**30, 2), "decode_at_pos": {}, "prefill": {}}
    for P in (0, 1024, 4096, 10240, 16384, 32000):
        if P + 400 > m.TM:
            continue
        m.reset(); m.load_ids(np.full(m.TM + 1, 818, np.int32))
        for i in range(8):
            m.step(P + i, feed=0)
        m.q.finish(); n = 320; t0 = time.perf_counter()
        for i in range(n):
            m.step(P + 8 + i, out_step=i)
        m.q.finish(); dt = time.perf_counter() - t0
        r["decode_at_pos"][str(P)] = {"tok_s": round(n / dt, 2), "ms_tok": round(dt * 1e3 / n, 3), "measure_s": round(dt, 2)}
        print("decode at", P, r["decode_at_pos"][str(P)], flush=True)
    for N in (512, 4096, 16384, 32000):
        if N + 300 > m.TM:
            continue
        ids = synth(N, m.V); ids[0] = 2

        def pre():
            m.reset(); m.load_ids(ids); pf.prompt(N); m.q.finish()
        pre(); n, t = rep_until(pre)
        r["prefill"][str(N)] = {"tok_s": round(N * n / t, 1), "ttft_s": round(t / n, 3), "reps": n}
        print("prefill", N, r["prefill"][str(N)], flush=True)
    return r


def qwen(eng, chat, text=None):
    from .device import vram
    m, pf, sp = eng.m, eng.pf, eng.sp
    r = {"tmax": m.TM, "vram_alloc_gib": round(vram() / 2**30, 2), "mtp": sp is not None}
    p = chat.tk.encode("The capital of France is", add_special_tokens=False).ids
    m.generate(p, 8); off = []
    for _ in range(3):
        _, _, tg = m.generate(p, 320); off.append(319 / tg)
    r["decode_short"] = {"prompt": "The capital of France is", "mtp_off_tok_s": round(statistics.median(off), 2), "reps": 3, "stat": "median"}
    if sp:
        sp.prompt(p); sp.iterate(4); sp.read(); on, acc = [], []
        for _ in range(3):
            sp.prompt(p); t0 = time.perf_counter(); sp.iterate(250); o, a = sp.read(); on.append(len(o) / (time.perf_counter() - t0)); acc.append(a / max(1, sp.it))
        r["decode_short"].update(mtp_on_tok_s=round(statistics.median(on), 2), mtp_accept=round(statistics.median(acc), 3))
    print("short", r["decode_short"], flush=True)
    src = None
    if text:
        src = np.array(chat.tk.encode(open(text, encoding="utf-8").read(), add_special_tokens=False).ids, np.int32)
        r["text"] = {"file": os.path.basename(text), "tokens": int(len(src))}
    r["long"] = {}
    for L in (2048, 8192, 16384, 32768):
        if L + 400 > m.TM:
            continue
        ids = (np.resize(src, L) if src is not None and len(src) else synth(L, m.V)[:L]).tolist(); e = {"L": L}

        def pre():
            m.reset(); m.load_ids(list(ids) + [0]); pf.run_long(L); pf.first_token(0); m.q.finish()
        n, t = rep_until(pre); e["prefill_tok_s"] = round(L * n / t, 1); e["prefill_s"] = round(t / n, 3); e["prefill_reps"] = n
        ng = 320; t0 = time.perf_counter()
        for j in range(1, ng):
            m.step(L + j - 1, out_step=j)
        offt = np.zeros(ng, np.int32); cl.enqueue_copy(m.q, offt, m.outs); m.q.finish(); e["mtp_off_tok_s"] = round((ng - 1) / (time.perf_counter() - t0), 2)
        if sp and src is not None:
            sp.prompt_fast(ids, pf); it = 170; t0 = time.perf_counter(); sp.iterate(it); o, a = sp.read()
            e["mtp_on_tok_s"] = round(len(o) / (time.perf_counter() - t0), 2); e["mtp_accept"] = round(a / it, 3)
            g = [int(offt[0])] + list(o); k = min(len(g), ng); e["mtp_on_equals_off"] = g[:k] == offt[:k].tolist()
        r["long"][str(L)] = e; print("long", e, flush=True)
    return r


def gemma_mtp(model, out, ctx=None):
    """Gemma 4 MTP: fixed chat prompts, 400 greedy tokens, MTP off and on (auto) with the same output check."""
    from .engines import open_model
    eng, chat = open_model(model, ctx=ctx or 4096)
    if eng.sp is None:
        raise SystemExit("no assistant/ folder next to the model: MTP not available")
    prompts = {"en": "Explain in a few paragraphs how a refrigerator works.",
               "code": "Write a Python function that reads a CSV file and returns the average of every numeric column. Include comments.",
               "think": "How many prime numbers are there between 1 and 60? Think it through."}
    r = {"env": env(), "model": os.path.basename(eng.files["main"]), "ctx": eng.ctx, "tokens": 400, "result": {}}
    for name, text in prompts.items():
        ids, _ = chat.encode([{"role": "user", "content": text}], thinking=(name == "think"))
        e = {}
        for lab, on in (("off", False), ("on", True)):
            sp = eng.sp; eng.sp = sp if on else None
            eng.generate(ids, 32, reuse=False)
            t, n, k = 0.0, 0, 0
            while t < MIN_T:
                o, _, (tp, tg, m_, _) = eng.generate(ids, 400, reuse=False); t += tg; n += max(m_ - 1, 0); k += 1
            e[lab + "_tok_s"] = round(n / t, 2); e[lab + "_tokens"] = o; eng.sp = sp
        e["same"] = e.pop("off_tokens") == e.pop("on_tokens")
        r["result"][name] = e; print("mtp", name, e, flush=True)
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out, flush=True)


def run(model, out, ctx=None, text=None):
    from .engines import find, open_model
    if not ctx:                                     # published settings: 33280, Gemma Quality 17408 (fits 16 GB)
        f = find(model); ctx = 17408 if json.load(open(f["main"][:-4] + ".json", encoding="utf-8")).get("nonexp8") else 33280
    eng, chat = open_model(model, ctx=ctx, mtp=False if find(model)["family"] == "gemma4" else True)   # Gemma: plain figures; MTP: bench --mtp
    r = {"env": env(), "model": os.path.basename(eng.files["main"]), "family": eng.family}
    r["result"] = gemma(eng) if eng.family == "gemma4" else qwen(eng, chat, text)
    json.dump(r, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", out, flush=True)
