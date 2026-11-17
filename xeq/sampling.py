"""Sampling on the GPU (no host round trip per token).

Order, as in common practice: repetition penalty (prompt + output tokens) -> presence penalty (output tokens)
-> temperature -> top-k -> top-p -> min-p. temperature 0 (or no sampling options) keeps greedy decoding untouched.
- Without top-k / top-p / min-p, a token is drawn from the full vocabulary with the Gumbel-max trick.
- With them, the 64 most likely tokens are the candidates (top_k is capped at 64; top_p probabilities are taken
  over the full vocabulary, and the nucleus is cut at 64 tokens if it would be larger).
- Random numbers come from a counter-based hash of (seed, position, purpose), so a seed reproduces a run.

MTP (Qwen3.5): the draft d is greedy, i.e. a one-hot proposal. Speculative sampling then reduces to: accept d with
probability p(d); on rejection draw from p with d removed. The output distribution equals plain sampling."""
from __future__ import annotations

import os

import numpy as np
import pyopencl as cl

CAP = 64
SALT_PLAIN, SALT_SPEC = 1, 2

SSRC = r"""
#define CAP 64
#define NS 4096
inline ulong mix64(ulong z) { z += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
inline float uni(ulong seed, ulong ctr, ulong salt, ulong i) {
    ulong h = mix64(seed ^ mix64((ctr << 3) ^ salt) ^ mix64(i * 0xD1B54A32D192ED03UL + 1UL));
    return ((float)(h >> 40) + 0.5f) * (1.0f / 16777216.0f);
}
inline float proc(float x, int s, float invT, float pres, float rep) {
    if (rep != 1.0f && s) x = x > 0.0f ? x / rep : x * rep;
    if (s & 2) x -= pres;
    return x * invT;
}
inline int sx(int i, int s, __global const int* xt, int m) {          // extra "generated" tokens: drafts before this row
    for (int j = 0; j < m; j++) if (xt[j] == i) s |= 2;
    return s;
}
inline void bsort(__local float* sk, __local int* si, int l) {          // descending, NS elements, 256 work-items
    for (int k = 2; k <= NS; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int t = l; t < NS; t += 256) {
                int p = t ^ j;
                if (p > t) {
                    float a = sk[t], b = sk[p]; int up = (t & k) == 0;
                    if (up ? (a < b) : (a > b)) { sk[t] = b; sk[p] = a; int c = si[t]; si[t] = si[p]; si[p] = c; }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
}
// stage 1: work-group (g, row t) over logits [g*4096, (g+1)*4096): candidates (mode 1), Gumbel argmax (mode 0), log-sum-exp parts
__kernel void sp1(__global const float* v, int N, __global const int* seen, __global const int* xt, int nd,
                  float invT, float pres, float rep, int mode, uint slo, uint shi, __global const int* ctrb, int ctroff, int salt,
                  __global float* cv, __global int* ci, __global float* gm, __global int* gi, __global float* lm, __global float* ls) {
    __local float sk[NS]; __local int si[NS]; __local float rm[256], rs[256]; __local int ri[256];
    int l = get_local_id(0), g = get_group_id(0), t = get_group_id(1), P = get_num_groups(0);
    ulong seed = ((ulong)shi << 32) | slo, ctr = (ulong)(ctrb[0] + ctroff);
    int ex = t < nd ? xt[t] : -1, nx = min(t, nd);
    __global const float* vt = v + (size_t)t * N;
    float mx = -INFINITY, sm = 0.0f, gb = -INFINITY; int gbi = 0;
    for (int k = l; k < NS; k += 256) {
        int i = g * NS + k; float x = -INFINITY;
        if (i < N) x = proc(vt[i], sx(i, seen[i], xt, nx), invT, pres, rep);
        sk[k] = x; si[k] = i;
        if (x > -INFINITY) {
            if (x > mx) { sm = sm * exp(mx - x) + 1.0f; mx = x; } else sm += exp(x - mx);
            if (mode == 0 && i != ex) { float y = x - log(-log(uni(seed, ctr, (ulong)(salt * 16 + t), (ulong)i))); if (y > gb) { gb = y; gbi = i; } }
        }
    }
    rm[l] = mx; rs[l] = sm; barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) {
        float M = -INFINITY, S = 0.0f;
        for (int k = 0; k < 256; k++) M = fmax(M, rm[k]);
        if (M > -INFINITY) for (int k = 0; k < 256; k++) if (rm[k] > -INFINITY) S += rs[k] * exp(rm[k] - M);
        lm[t * P + g] = M; ls[t * P + g] = S;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (mode == 0) {
        rm[l] = gb; ri[l] = gbi; barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 128; s > 0; s >>= 1) { if (l < s && rm[l + s] > rm[l]) { rm[l] = rm[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
        if (l == 0) { gm[t * P + g] = rm[0]; gi[t * P + g] = ri[0]; }
        return;
    }
    bsort(sk, si, l);
    if (l < CAP) { cv[((size_t)t * P + g) * CAP + l] = sk[l]; ci[((size_t)t * P + g) * CAP + l] = si[l]; }
}
// stage 2: one work-group per row. Arguments 3..5 match argmax2 (tok, outs, step) so it can stand in for it.
__kernel void sp2(__global const float* cv, __global const int* ci, int P, __global int* tok, __global int* outs, int step,
                  __global const float* gm, __global const int* gi, __global const float* lm, __global const float* ls,
                  __global const float* v, int N, __global int* seen, __global const int* xt, int nd,
                  float invT, float pres, float rep, int mode, int topk, float topp, float minp,
                  uint slo, uint shi, __global int* ctrb, int ctroff, int salt, int toO, __global int* O, __global int* acc, int setseen, int inc) {
    __local float sk[NS]; __local int si[NS]; __local float rm[256]; __local int ri[256];
    int l = get_local_id(0), t = get_group_id(1);
    ulong seed = ((ulong)shi << 32) | slo, ctr = (ulong)(ctrb[0] + ctroff);
    int ex = t < nd ? xt[t] : -1, nx = min(t, nd);
    if (l == 0) {                                                      // log-sum-exp over the full vocabulary
        float M = -INFINITY, S = 0.0f;
        for (int g = 0; g < P; g++) M = fmax(M, lm[t * P + g]);
        for (int g = 0; g < P; g++) if (lm[t * P + g] > -INFINITY) S += ls[t * P + g] * exp(lm[t * P + g] - M);
        rm[0] = M + log(S);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float LSE = rm[0]; int pick = 0; float pd = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (mode == 0) {
        rm[l] = l < P ? gm[t * P + l] : -INFINITY; ri[l] = l < P ? gi[t * P + l] : 0; barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = 128; s > 0; s >>= 1) { if (l < s && rm[l + s] > rm[l]) { rm[l] = rm[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
        pick = ri[0];
        if (ex >= 0) pd = exp(proc(v[(size_t)t * N + ex], sx(ex, seen[ex], xt, nx), invT, pres, rep) - LSE);
    } else {
        for (int k = l; k < NS; k += 256) {
            int g = k / CAP;
            sk[k] = g < P ? cv[((size_t)t * P) * CAP + k] : -INFINITY; si[k] = g < P ? ci[((size_t)t * P) * CAP + k] : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        bsort(sk, si, l);
        if (l == 0) {
            int K = topk > 0 ? min(topk, CAP) : CAP; float base;
            if (topk > 0) { float S = 0.0f; for (int k = 0; k < K; k++) if (sk[k] > -INFINITY) S += exp(sk[k] - sk[0]); base = sk[0] + log(S); }
            else base = LSE;
            float p[CAP], c = 0.0f; int n = 0;
            for (int k = 0; k < K; k++) {
                if (sk[k] == -INFINITY) break;
                p[k] = exp(sk[k] - base);
                if (n > 0 && (c >= topp || p[k] < minp * p[0])) break;
                c += p[k]; n++;
            }
            float tot = 0.0f; int dk = -1;
            for (int k = 0; k < n; k++) { tot += p[k]; if (si[k] == ex) dk = k; }
            if (dk >= 0) pd = p[dk] / tot;
            float rest = tot - (dk >= 0 ? p[dk] : 0.0f);
            if (rest <= 0.0f) pick = si[0];
            else {
                float u = uni(seed, ctr, (ulong)(salt * 16 + t), 0UL) * rest, a = 0.0f; pick = -1;
                for (int k = 0; k < n; k++) { if (k == dk) continue; a += p[k]; if (u < a) { pick = si[k]; break; } }
                if (pick < 0) for (int k = n - 1; k >= 0; k--) if (k != dk) { pick = si[k]; break; }
            }
            ri[0] = pick; rm[1] = pd;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        pick = ri[0]; pd = rm[1];
    }
    if (l == 0) {
        if (toO) O[t] = pick; else { tok[0] = pick; outs[step] = pick; }
        if (setseen) seen[pick] |= 2;
        if (ex >= 0) acc[t] = uni(seed, ctr, (ulong)(1024 + salt * 16 + t), 1UL) < pd;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (inc && l == 0 && t == 0) ctrb[0] += 1;                         // plain decoding: counter for the next token
}
// MTP judge with sampling: accept d when acc[0] (drawn with probability p(d)); otherwise emit O[0], drawn without d.
__kernel void judgeS(__global int* ta, __global const int* td, __global const int* O, __global int* flag, __global int* pos, __global int* mpa, __global int* mpb,
                     __global int* oa, __global int* ob, __global int* outs, __global int* cnt, __global int* nacc, __global const int* acc, __global int* seen) {
    int d = td[0], o0 = O[0], o1 = O[1], c = cnt[0], p = pos[0];
    mpa[0] = p + 1; mpb[0] = p + 2; ob[0] = o1;
    if (acc[0]) { oa[0] = d; outs[c] = d; outs[c + 1] = o1; cnt[0] = c + 2; flag[0] = 1; ta[0] = o1; pos[0] = p + 2; nacc[0] += 1; seen[d] |= 2; seen[o1] |= 2; }
    else { oa[0] = o0; outs[c] = o0; cnt[0] = c + 1; flag[0] = 0; ta[0] = o0; pos[0] = p + 1; seen[o0] |= 2; }
}
"""

KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "seed")


def options(src):
    """dict (CLI args or request body) -> sampling options, or None for greedy."""
    o = {k: src.get(k) for k in KEYS if src.get(k) is not None}
    if float(o.get("temperature") or 0) <= 0:
        return None
    return o


class Sampler:
    """Stands in for the model's argmax pair (m.am1, m.am2) while a request samples; for Qwen3.5 MTP it also replaces
    the verification argmax and judge kernels."""

    def __init__(self, m):
        self.m = m; ctx = m.ctx; mf = cl.mem_flags
        self.prg = cl.Program(ctx, SSRC).build()
        self.V = m.V; self.P = -(-self.V // 4096); P = self.P
        B = lambda n: cl.Buffer(ctx, mf.READ_WRITE, max(4, int(n) * 4))
        self.seen = B(self.V); self.zero = B(1); cl.enqueue_fill_buffer(m.q, self.zero, np.int32(0), 0, 4)
        R = 8                                                            # rows: up to k+1 verified tokens
        self.cv, self.ci, self.gm, self.gi, self.lm, self.ls = B(R * P * CAP), B(R * P * CAP), B(R * P), B(R * P), B(R * P), B(R * P)
        self.acc = B(R); self.dummy = B(2); self.z0 = B(1); cl.enqueue_fill_buffer(m.q, self.z0, np.int32(0), 0, 4)
        self.saved = None; self.spec = None

    def _args(self, o, logits, rows, xt, nd, ctrb, salt, toO, O, setseen, inc):
        T = float(o["temperature"]); tk = int(o.get("top_k") or 0); tp = float(o.get("top_p") if o.get("top_p") is not None else 1.0)
        mp = float(o.get("min_p") or 0.0); pres = float(o.get("presence_penalty") or 0.0); rep = float(o.get("repetition_penalty") or 1.0)
        mode = 0 if (tk <= 0 and tp >= 1.0 and mp <= 0.0) else 1
        s = o["_seed"]; slo, shi = np.uint32(s & 0xFFFFFFFF), np.uint32(s >> 32)
        I, f = np.int32, np.float32; m = self.m
        k1 = cl.Kernel(self.prg, "sp1"); k2 = cl.Kernel(self.prg, "sp2")
        k1.set_args(logits, I(m.V), self.seen, xt, I(nd), f(1.0 / T), f(pres), f(rep), I(mode), slo, shi, ctrb, I(0), I(salt),
                    self.cv, self.ci, self.gm, self.gi, self.lm, self.ls)
        k2.set_args(self.cv, self.ci, I(self.P), m.tok, m.outs, I(0), self.gm, self.gi, self.lm, self.ls, logits, I(m.V), self.seen, xt, I(nd),
                    f(1.0 / T), f(pres), f(rep), I(mode), I(tk), f(tp), f(mp), slo, shi, ctrb, I(0), I(salt), I(toO), O, self.acc, I(setseen), I(inc))
        return [k1, (self.P * 256, rows), (256, 1)], [k2, (256, rows), (256, 1)]

    def begin(self, o, prompt_ids, spec=None):
        """Installs the sampling kernels for one request."""
        m = self.m; o = dict(o); self.o = o
        seed = o.get("seed")
        o["_seed"] = (int(seed) & 0xFFFFFFFFFFFFFFFF) if seed is not None else int.from_bytes(os.urandom(8), "little")
        bits = np.zeros(self.V, np.int32); ids = np.asarray(prompt_ids, np.int64); ids = ids[(ids >= 0) & (ids < self.V)]; bits[ids] = 1
        cl.enqueue_copy(m.q, self.seen, bits)
        neg = self.dummy; cl.enqueue_fill_buffer(m.q, neg, np.int32(-1), 0, 8)
        self.saved = (m.am1, m.am2)
        cl.enqueue_fill_buffer(m.q, self.zero, np.int32(0), 0, 4)       # per-request token counter (plain decoding)
        m.am1, m.am2 = self._args(o, m.logit, 1, neg, 0, self.zero, SALT_PLAIN, 0, self.dummy, 1, 1)
        if spec is not None:
            k1, k2 = self._args(o, spec.lg2, 2, spec.TD, 1, spec.POS, SALT_SPEC, 1, spec.O, 0, 0)
            kj = cl.Kernel(self.prg, "judgeS")
            kj.set_args(spec.TA, spec.TD, spec.O, spec.FLAG, spec.POS, spec.MPA, spec.MPB, spec.OA, spec.OB, spec.OUTS, spec.CNT, spec.NACC, self.acc, self.seen)
            names = [e[0].function_name for e in spec.main]
            i1, i2, ij = names.index("argmax1b"), names.index("argmax2b"), names.index("judge")
            new = list(spec.main); new[i1] = k1; new[i2] = k2; new[ij] = [kj, (1,), (1,)]
            self.spec = (spec, spec.main); spec.main = new
        m.q.finish()

    def multi(self, logits, rows, drafts, nd, O):
        """Kernels for a multi-token verification (Gemma 4 MTP): row t < nd accepts drafts[t] with probability p_t(drafts[t])
        (acc[t]) and otherwise draws O[t] without it; row nd draws O[nd]. Set the position with set_pos() before each run."""
        k1, k2 = self._args(self.o, logits, rows, drafts, nd, self.z0, SALT_SPEC, 1, O, 0, 0)
        return k1, k2

    @staticmethod
    def set_pos(k1, k2, pos):
        k1[0].set_arg(12, np.int32(pos)); k2[0].set_arg(25, np.int32(pos))

    def mark(self, toks):
        """Marks generated tokens as seen (presence / repetition penalties)."""
        for t in toks:
            cl.enqueue_copy(self.m.q, self.seen, np.array([3], np.int32), dst_offset=4 * int(t), is_blocking=False)

    def end(self):
        m = self.m
        if self.saved:
            m.am1, m.am2 = self.saved; self.saved = None
        if self.spec:
            sp, main = self.spec; sp.main = main; self.spec = None
