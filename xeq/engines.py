"""Model folder discovery and greedy generation for both families.
generate(ids, max_new, on_tokens) -> (tokens without the stop token, "stop" | "length", (prefill s, decode s, tokens, recomputed prompt tokens))."""
from __future__ import annotations

import glob
import json
import os
import threading
import time

import numpy as np
import pyopencl as cl

from .device import knob as knob_g4

SHORT = 32          # prompts shorter than this are fed token by token


def find(path):
    """A model folder (or a .bin inside it) -> {"family", "folder", "main", "mtp", "vision"}.
    The main model is the .bin whose header has "config"; the Qwen MTP head is the one with "mtp.*" tensors only;
    the Gemma vision encoder has "fmt": "g4v"."""
    folder = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    r = {"folder": folder, "main": None, "mtp": None, "vision": None, "assistant": None}
    cands = sorted(glob.glob(os.path.join(folder, "*.bin")))
    for v in (os.path.join(folder, "vision"), os.path.join(folder, "..", "vision")):   # optional vision encoder folder
        cands += sorted(glob.glob(os.path.join(v, "*.bin")))
    for b in cands:
        j = b[:-4] + ".json"
        if not os.path.exists(j):
            continue
        h = json.load(open(j, encoding="utf-8"))
        if h.get("fmt") == "g4v":
            if r["vision"] is None or h.get("enc") == "i8":           # INT8 encoder by default (fits 16 GB with both configs)
                r["vision"] = b
        elif "config" in h and os.path.dirname(os.path.abspath(b)) == os.path.abspath(folder):
            if r["main"] is None or os.path.abspath(b) == os.path.abspath(path):
                r["main"], cfg = b, h["config"]
        elif any(k.startswith("mtp.") for k in h.get("tensors", {})):
            r["mtp"] = b
    for a in (os.path.join(folder, "assistant"), os.path.join(folder, "..", "assistant")):   # Gemma 4 MTP assistant (original files)
        if os.path.exists(os.path.join(a, "model.safetensors")) and os.path.exists(os.path.join(a, "config.json")):
            r["assistant"] = os.path.abspath(a); break
    if r["main"] is None:
        raise SystemExit(f"no XeFP4 model (.bin + .json with a config) in {folder}")
    mt = str(cfg.get("model_type", ""))
    r["family"] = "gemma4" if "gemma" in mt else "qwen35" if "qwen" in mt else None
    if r["family"] is None:
        raise SystemExit(f"unknown model_type {mt!r}")
    return r


def open_model(path, ctx=None, mtp=True, vision=False, kv=None):
    """-> (engine, Chat). kv: "fp16" (default) or "fp32" KV cache (Gemma 4; fixed when the kernels are first built)."""
    import sys
    from .chat import Chat
    f = find(path)
    if kv not in (None, "fp32", "fp16"):
        raise SystemExit(f"--kv must be fp32 or fp16, not {kv!r}")
    if kv == "fp16" and f["family"] != "gemma4":
        print("[xeq] note: --kv fp16 applies to Gemma 4 only; Qwen3.5 keeps an FP32 KV cache", file=sys.stderr, flush=True)
    if f["family"] == "gemma4" and kv:
        want = "1" if kv == "fp16" else "0"
        mod = sys.modules.get(__package__ + ".gemma4.model")
        if mod is not None and int(mod.KV16) != int(want):
            raise SystemExit("the KV cache type is fixed once a Gemma model has been loaded in this process")
        os.environ["XEQ_G4_KV16"] = want
    chat = Chat(f["folder"], f["family"])
    if f["family"] == "gemma4":
        eng = Gemma(f["main"], chat.stop, ctx, f["vision"] if vision else None, f["assistant"] if mtp else None)
    else:
        eng = Qwen(f["main"], chat.stop, ctx, f["mtp"] if mtp else None)
    eng.files = f
    _check_vram(eng)
    return eng, chat


def _check_vram(eng):
    """Warns when the allocated buffers come close to the GPU memory (the driver then spills to system memory: slow)."""
    from .device import vram
    total = eng.m.ctx.devices[0].global_mem_size; used = vram()
    if used > 0.99 * total:
        import sys
        print(f"[xeq] warning: {used / 2**30:.2f} GiB allocated of {total / 2**30:.2f} GiB GPU memory; "
              "lower --ctx or use --no-mtp if decoding is slow", file=sys.stderr, flush=True)


def _rd(m, buf, n):
    v = np.zeros(n, np.int32); cl.enqueue_copy(m.q, v, buf); m.q.finish(); return v.tolist()


def _cut(got, stop):
    for i, t in enumerate(got):
        if t in stop:
            return got[:i], True
    return got, False


def _decode(m, N, got, mx, stop, on_tokens, step):
    """Plain greedy decoding after the first token: enqueues `step` tokens at a time and keeps up to two reads in flight,
    so the GPU never waits for the host."""
    j = 1; done = got[0] in stop
    if on_tokens and not done:
        on_tokens(list(got))
    pend = []
    while not done and (j < mx or pend):
        if j < mx and len(pend) < 2:
            n = min(step, mx - j); arr = np.zeros(n, np.int32)
            for k in range(n):
                m.step(N + j + k - 1, out_step=j + k)
            ev = cl.enqueue_copy(m.q, arr, m.outs, src_offset=j * 4, is_blocking=False); m.q.flush()
            pend.append((arr, ev)); j += n
            if len(pend) < 2 and j < mx:
                continue
        arr, ev = pend.pop(0); ev.wait(); got += arr.tolist()
        done = any(t in stop for t in got)
        if on_tokens:
            on_tokens(_cut(got, stop)[0])
    for arr, ev in pend:
        ev.wait(); got += arr.tolist()
    m.q.finish()
    return got, j


class _Sampled:
    """generate() = greedy unless `sampling` options (see sampling.options) are given."""
    sampler = None; _sopt = None

    def generate(self, ids, mx, on_tokens=None, sampling=None, **kw):
        self._sopt = sampling; self._splans = {}
        if not sampling:
            return self._generate(ids, mx, on_tokens, **kw)
        from .sampling import Sampler
        with self.lock:
            if self.sampler is None:
                self.sampler = Sampler(self.m)
            self.sampler.begin(sampling, ids, spec=self.sp if self.family == "qwen35" else None)
            try:
                return self._generate(ids, mx, on_tokens, **kw)
            finally:
                self.sampler.end()


class Gemma(_Sampled):
    family = "gemma4"

    def __init__(self, path, stop, ctx=None, vision=None, assistant=None):
        from .gemma4 import model as GR
        from .gemma4.prefill import Prefill
        self.m = GR.Model(path, tmax=ctx); self.pf = Prefill(self.m); self.lock = threading.RLock(); self.stop = stop
        self.cache = []
        self.CAPS = min(L["CAP"] for L in self.m.lcfg if not L["kv"]); self.WIN = self.m.WIN
        self.vis = None
        if vision:
            from .gemma4.vision import Vision
            self.vis = Vision(vision, self.m.ctx, self.m.q); self.m.q.finish()
        self.mtp = None; self.sp = None
        if assistant:
            from .gemma4 import mtp as MT
            from .device import knob
            self.MT = MT; self.mp = MT.Multi(self.m); self.dr = MT.Draft(self.m, assistant, self.mp); self.m.q.finish()
            self.mtp = assistant; self.k_fixed = int(knob("G4_K", "0"))
            for T in sorted({(self.k_fixed or 4) + 1, 4, 5, 6}):              # build the verification plans now, not mid-generation
                self.mp.plan(T)
            self.m.q.finish()
            self._calibrate()
            self.sp = self                                                   # MTP on (set to None to decode without it)

    @property
    def ctx(self):
        return self.m.TM

    def encode_images(self, images):
        """PIL images -> (projected rows buffer, [token count per image])."""
        if self.vis is None:
            raise ValueError("no vision encoder loaded (needs the g4v .bin in the model folder and --vision)")
        H = self.m.H; ns, bufs = [], []
        for im in images:
            soft, n = self.vis.encode(im=im); a = np.zeros((n, H), np.float32); cl.enqueue_copy(self.m.q, a, soft); self.m.q.finish()
            bufs.append(a); ns.append(n)
        return (np.ascontiguousarray(np.concatenate(bufs)) if bufs else np.zeros((0, H), np.float32)), ns

    def _generate(self, ids, mx, on_tokens=None, step=4, reuse=True, spans=None, soft=None):
        """Greedy. KV of the previous request is reused for the common prefix (text-only requests); a sliding-layer ring
        that was overwritten past the prefix is recomputed from prefix - ring size."""
        m = self.m; N = len(ids); use_mtp = self.mtp is not None and self.sp is not None
        mx = min(mx, m.TM - N - 8 - (self.MT.TB if use_mtp else 0))
        if mx < 1:
            raise ValueError(f"prompt too long ({N} tokens, limit {m.TM - 9})")
        with self.lock:
            c = self.cache if (reuse and not spans) else []; P = 0
            while P < min(len(c), N) and c[P] == ids[P]:
                P += 1
            P = min(P, N - 1); E = len(c) + (8 if c else 0)                  # MTP may have written a few positions past the cache
            if P < 32:
                s = 0
            elif E - P <= self.CAPS - self.WIN:
                s = P
            else:
                s = max(0, P - self.CAPS)
            self.cache = []
            if s == 0:
                m.reset()
            if spans:
                self.pf.set_images(spans, soft)
            m.load_ids(np.array(list(ids) + [0] * 8, np.int32))
            t0 = time.perf_counter(); self.pf.prompt(N, start=s)
            got = _rd(m, m.outs, 1); t1 = time.perf_counter()
            if use_mtp:
                got = self._mtp_decode(N, got, mx, on_tokens); j = len(got)
            else:
                got, j = _decode(m, N, got, mx, self.stop, on_tokens, step)
            t2 = time.perf_counter()
            if spans:
                self.pf.set_images([])
            else:
                self.cache = list(ids) + got[:j - 1]
        out, hit = _cut(got, self.stop)
        return out, "stop" if hit else "length", (t1 - t0, t2 - t1, len(out), N - s)


    def _plan(self, T):
        """Verification plan for T tokens: greedy argmax rows, or (sampling) the sampler kernels in their place."""
        P = self.mp.plan(T)
        if not self._sopt:
            return P, None
        if T not in self._splans:
            k1, k2 = self.sampler.multi(self.mp.logit, T, self.dr.douts, T - 1, self.mp.outs)
            Ps = self.MT.Plan(self.m.q); Ps.ks = P.ks[:-2 * T] + [list(k1) + [None], list(k2) + [None]]; Ps.dyn = P.dyn
            self._splans[T] = (Ps, (k1, k2))
        return self._splans[T]

    def _prime(self, p, y):
        """Starts drafting at position p with token y (main hidden state of position p-1 in m.t)."""
        m, mp, dr = self.m, self.mp, self.dr
        cl.enqueue_copy(m.q, dr.Zh, m.t)
        cl.enqueue_copy(m.q, dr.tok, np.array([y], np.int32)); dr.embed()
        cl.enqueue_copy(m.q, mp.toks, np.array([y], np.int32))

    def _mtp_seg(self, out, p, k, mx, cycles, on_tokens):
        """Up to `cycles` MTP cycles (drafting state primed). Returns (p, k, (tokens, seconds, cycles) after the first cycle, stopped)."""
        m, mp, dr, TB = self.m, self.mp, self.dr, self.MT.TB; H = m.H; stop = self.stop; samp = bool(self._sopt)
        accb = np.zeros(TB, np.int32); n0 = len(out); t0 = time.perf_counter(); c = 0
        while c < cycles and len(out) < mx:
            k = max(1, min(k, TB - 1, m.TM - p - 2))
            for i in range(k):
                dr.step(p + i, p, i)
            cl.enqueue_copy(m.q, mp.toks, dr.douts, dst_offset=4, byte_count=4 * k)
            P, sk = self._plan(k + 1)
            if sk:
                self.sampler.set_pos(sk[0], sk[1], p)
            P.run(pos0=p)
            o = np.zeros(k + 1, np.int32); d = np.zeros(k, np.int32)
            cl.enqueue_copy(m.q, o, mp.outs); cl.enqueue_copy(m.q, d, dr.douts)
            if samp:
                cl.enqueue_copy(m.q, accb, self.sampler.acc)
            m.q.finish()
            a = 0
            if samp:
                while a < k and accb[a]:
                    a += 1
            else:
                while a < k and d[a] == o[a]:
                    a += 1
            new = d[:a].tolist() + [int(o[a])]; out += new; self._hist.append(a); p += a + 1; c += 1
            if c == 1:                                                     # rate excludes the first cycle (priming)
                n0, t0 = len(out), time.perf_counter()
            if samp:
                self.sampler.mark(new)
            cl.enqueue_copy(m.q, dr.Zh, mp.t, src_offset=a * H * 4, byte_count=H * 4)
            cl.enqueue_copy(m.q, dr.tok, mp.outs, src_offset=a * 4, byte_count=4)
            cl.enqueue_copy(m.q, mp.toks, mp.outs, src_offset=a * 4, byte_count=4)
            cl.enqueue_copy(m.q, m.tok, mp.outs, src_offset=a * 4, byte_count=4)       # in case plain decoding takes over
            dr.embed(); m.q.flush()
            if any(t in stop for t in new):
                return p, k, (len(out) - n0, time.perf_counter() - t0, c - 1), True
            if on_tokens:
                on_tokens(_cut(out, stop)[0][:mx])
            h = self._hist
            if not self.k_fixed and len(h) >= 4:
                r = sum(h[-4:]) / 4
                if r >= k - 1 and k < 5:
                    k += 1
                elif r < k - 2.5 and k > 3:
                    k -= 1
        return p, k, (len(out) - n0, time.perf_counter() - t0, c - 1), False

    def _plain_seg(self, out, p, n, on_tokens, mx=1 << 30):
        """n plain decoding steps from position p (token to feed in m.tok, hidden of p-1 in m.t). Returns (p, tokens per second, stopped)."""
        m, stop = self.m, self.stop; t0 = time.perf_counter(); got, pend, i = [], [], 0; done = False
        while (i < n or pend) and not done:
            if i < n and len(pend) < 2:
                c = min(8, n - i); arr = np.zeros(c, np.int32)
                for q in range(c):
                    m.step(p + i + q, out_step=i + q)
                ev = cl.enqueue_copy(m.q, arr, m.outs, src_offset=i * 4, is_blocking=False); m.q.flush()
                pend.append((arr, ev)); i += c
                if len(pend) < 2 and i < n:
                    continue
            arr, ev = pend.pop(0); ev.wait(); got += arr.tolist(); done = any(t in stop for t in got)
            if on_tokens:
                on_tokens(_cut(out + got, stop)[0][:mx])
        for arr, ev in pend:
            ev.wait(); got += arr.tolist()
        m.q.finish(); out += got
        return p + i, i / max(time.perf_counter() - t0, 1e-9), done or any(t in stop for t in got)

    def _calibrate(self):
        """Plain decoding speed (tokens/s) at a short context, measured once at load; refreshed by plain segments."""
        m = self.m; m.reset(); m.load_ids(np.full(64, 2, np.int32))
        for i in range(8):
            m.step(i, feed=0)
        m.q.finish(); t0 = time.perf_counter()
        for i in range(8, 40):
            m.step(i, out_step=i)
        m.q.finish(); self.r_plain = 32 / (time.perf_counter() - t0); m.reset()

    def _mtp_decode(self, N, got, mx, on_tokens):
        """MTP decoding that falls back to plain decoding on text where drafts are rarely accepted. The MTP speed is
        accumulated over the request (first cycle excluded) and compared with plain decoding once at least 12 MTP cycles
        have run; if MTP is more than 3% slower, plain decoding runs for 64 tokens (128, 256 after further losses) and MTP
        is tried again with fresh statistics. A request that follows one where MTP lost starts with 64 plain tokens.
        Both are exact: greedy tokens and the sampling distribution do not change.
        The assistant drafts k tokens (k adapts between 3 and 5), the main model verifies k+1."""
        stop = self.stop; out = [got[0]]; p = N; self._hist = []
        if out[0] in stop:
            return out
        if on_tokens:
            on_tokens(list(out))
        auto = int(knob_g4("G4_MTPAUTO", "1")); dbg = knob_g4("G4_MTPDBG"); k = self.k_fixed or 4
        primed = False; back = 64; acc_n = acc_t = 0.0; acc_c = 0; verdict = None
        if auto and getattr(self, "_mtp_lost", False):                   # the previous request lost with MTP: start plain
            p, r, stopped = self._plain_seg(out, p, min(64, mx - len(out)), on_tokens, mx)
            if stopped:
                self.last_cycles = self.last_accept = 0
                return out[:mx]
        while len(out) < mx:
            if not primed:
                self._prime(p, out[-1]); primed = True
            p, k, (n, dt, c), stopped = self._mtp_seg(out, p, k, mx, 6 if auto else 1 << 30, on_tokens)
            acc_n += n; acc_t += dt; acc_c += c
            if stopped or len(out) >= mx:
                break
            r_mtp = acc_n / max(acc_t, 1e-9)
            if dbg:
                import sys
                print(f"[mtp] p {p} k {k} out {len(out)} cycles {acc_c} r_mtp {r_mtp:.1f} r_plain {self.r_plain:.1f} hist {self._hist[-6:]}", file=sys.stderr, flush=True)
            if acc_c >= 12:
                verdict = r_mtp < 0.97 * self.r_plain
            if auto and acc_c >= 12 and r_mtp < 0.97 * self.r_plain:
                n = min(back, mx - len(out))
                p, r, stopped = self._plain_seg(out, p, n, on_tokens, mx); primed = False
                if n >= 32:
                    self.r_plain = 0.5 * self.r_plain + 0.5 * r
                back = min(back * 2, 256); acc_n = acc_t = 0.0; acc_c = 0
                if stopped:
                    break
            elif acc_c >= 12 and r_mtp >= 0.97 * self.r_plain:
                back = 64
        self.m.q.finish()
        if acc_c >= 12:
            verdict = acc_n / max(acc_t, 1e-9) < 0.97 * self.r_plain
        if verdict is not None:
            self._mtp_lost = verdict                                       # remembered for the next request
        self.last_cycles, self.last_accept = len(self._hist), sum(self._hist)
        return out[:mx]

class Qwen(_Sampled):
    family = "qwen35"

    def __init__(self, path, stop, ctx=None, mtp=None):
        from .qwen35.model import Model
        from .qwen35.prefill import Prefill
        self.m = Model(path, tmax=ctx); self.pf = Prefill(self.m); self.lock = threading.RLock(); self.stop = stop
        self.sp = None
        if mtp:
            from .qwen35.spec import Spec
            self.sp = Spec(self.m, mtp)

    @property
    def ctx(self):
        return self.m.TM

    def _generate(self, ids, mx, on_tokens=None, step=16, reuse=True, spans=None, soft=None):
        """Greedy, with MTP speculative decoding when the MTP head is loaded (same tokens as without it).
        The linear-attention state is not kept between requests, so every prompt is processed in full."""
        if spans:
            raise ValueError("this model has no vision encoder")
        m = self.m; N = len(ids)
        mx = min(mx, m.TM - N - 24)
        if mx < 1:
            raise ValueError(f"prompt too long ({N} tokens, limit {m.TM - 25})")
        with self.lock:
            t0 = time.perf_counter()
            if self.sp is not None:
                y0 = self.sp.prompt(ids) if N < SHORT else self.sp.prompt_fast(ids, self.pf)
                got = [y0]; t1 = time.perf_counter(); done = y0 in self.stop
                if on_tokens and not done:
                    on_tokens(got)
                while not done and len(got) < mx:
                    self.sp.iterate(step); out, _ = self.sp.read(); got = [y0] + out
                    done = any(t in self.stop for t in got)
                    if on_tokens:
                        on_tokens(_cut(got, self.stop)[0][:mx])
                got = got[:mx]
            else:
                m.reset()
                if N < SHORT:
                    m.load_ids(ids)
                    for i in range(N):
                        m.step(i, out_step=0 if i == N - 1 else None, feed=i)
                else:
                    m.load_ids(list(ids) + [0]); self.pf.run_long(N); self.pf.first_token(0)
                got = _rd(m, m.outs, 1); t1 = time.perf_counter()
                got, _ = _decode(m, N, got, mx, self.stop, on_tokens, 4)
            m.q.finish(); t2 = time.perf_counter()
        out, hit = _cut(got, self.stop)
        return out, "stop" if hit else "length", (t1 - t0, t2 - t1, len(out), N)
