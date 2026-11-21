"""Qwen3.5 decoding with multi-token MTP drafts (greedy). One iteration runs on the GPU without host round trips:
1) The main model processes T = ND + 1 tokens [accepted token a, drafts d0..d(ND-1)] in one pass (weights read once).
2) Judge: c = 1 + number of leading drafts equal to the main model's own prediction; the accepted tokens are O[0..c-1],
   so the output is exactly the greedy sequence. Linear-attention updates of tokens 1..T-1 are deferred and applied at
   the start of the next pass for the accepted ones; the causal-conv window is rebuilt from the accepted inputs.
3) MTP writes its KV for the c accepted positions, drafts d0 from the last one, then is fed its own draft and output
   ND-1 more times (chained drafts). For ND = 1 this is the same scheme as spec.Spec.
Written for XeQ; not used with sampling."""
from __future__ import annotations

import numpy as np
import pyopencl as cl

from .model import Model, SRC
from .spec import BSRC
from ..device import knob

NSRC = r"""
#ifndef TT
#define TT 3
#endif
__kernel void embedN(__global const uchar* codes, __global const uint* fac, __global const half* rs, __global const ushort* E, int xe,
                     __global const int* ta, __global const int* td, __global float* X, int H) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    size_t n = t == 0 ? ta[0] : td[t - 1];
    float v;
    if (xe) {
        int FW = (H / G) * 3 / 32, bo = (i / G) * 3;
        __global const uint* fr = fac + n * FW;
        ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        uint by = codes[n * (H / 2) + i / 2], c = (i & 1) ? (by >> 4) : (by & 15u);
        v = edq(c) * vload_half(n, rs) * 0.125f * (float)(((fw >> (bo & 31)) & 7) + 1);
    } else v = as_float(((uint)E[n * H + i]) << 16);
    X[(size_t)t * H + i] = v;
}
// gemvB4 for TT input vectors (same per-vector accumulation order as gemvB4)
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemvBT(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, int add) {
    int r0 = get_global_id(1) * RB; if (r0 >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32, K8 = K / 8;
    float a[TT][RB];
    UNR for (int t = 0; t < TT; t++) UNR for (int r = 0; r < RB; r++) a[t][r] = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        float mm[RB], s[TT][RB];
        UNR for (int r = 0; r < RB; r++) {
            __global const uint* fr = fac + (size_t)(r0 + r) * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            mm[r] = (float)(((fw >> sh) & 7) + 1);
            UNR for (int t = 0; t < TT; t++) s[t][r] = 0.0f;
        }
        UNR for (int j = 0; j < G / 8; j++) {
            int c8 = b * (G / 8) + j;
            float8 v[TT];
            UNR for (int t = 0; t < TT; t++) v[t] = vload8(c8 + t * K8, x);
            UNR for (int r = 0; r < RB; r++) {
                uint u = codes[(size_t)(r0 + r) * K8 + c8];
                float4 lo = convert_float4(as_char4(dlo(u))), hi = convert_float4(as_char4(dhi(u)));
                UNR for (int t = 0; t < TT; t++) s[t][r] += dotp(lo, hi, v[t]);
            }
        }
        UNR for (int r = 0; r < RB; r++) UNR for (int t = 0; t < TT; t++) a[t][r] += s[t][r] * mm[r];
    }
    UNR for (int r = 0; r < RB; r++) UNR for (int t = 0; t < TT; t++) {
        float v = sub_group_reduce_add(a[t][r]);
        if (l == 0) { int row = r0 + r; float sc = vload_half(row, rs) * 0.125f; v *= sc;
            size_t o = (size_t)t * N + row; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lmheadT(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K) {
    int r0 = get_global_id(1) * 2; if (r0 >= N) return;
    int l = get_local_id(0);
    float acc[2][TT];
    UNR for (int r = 0; r < 2; r++) UNR for (int t = 0; t < TT; t++) acc[r][t] = 0.0f;
    for (int k = l * 16; k < K; k += 256) {
        float16 v[TT];
        UNR for (int t = 0; t < TT; t++) v[t] = vload16(0, x + (size_t)t * K + k);
        UNR for (int r = 0; r < 2; r++) {
            float16 a = convert_float16(vload16(0, q + (size_t)(r0 + r) * K + k));
            UNR for (int t = 0; t < TT; t++)
                acc[r][t] += dot(a.s0123, v[t].s0123) + dot(a.s4567, v[t].s4567) + dot(a.s89ab, v[t].s89ab) + dot(a.scdef, v[t].scdef);
        }
    }
    UNR for (int r = 0; r < 2; r++) UNR for (int t = 0; t < TT; t++) {
        float v = sub_group_reduce_add(acc[r][t]); if (l == 0) y[(size_t)t * N + r0 + r] = v * sc[r0 + r];
    }
}
// causal conv (kernel 4) for TT tokens. cw: window (3 inputs) before the previous pass, cx: the previous pass's TT inputs,
// acc: tokens accepted from the previous pass (0 right after the prompt: cw is the window).
__kernel void convN(__global const float* P, int ldp, __global float* cw, __global float* cx, __global const float* w, __global float* Cx, int C, __global const int* acc) {
    int c = get_global_id(0); if (c >= C) return;
    int n = acc[0];
    float seq[3 + TT];
    seq[0] = cw[c]; seq[1] = cw[C + c]; seq[2] = cw[2 * C + c];
    for (int t = 0; t < TT; t++) seq[3 + t] = cx[(size_t)t * C + c];
    float s0 = seq[n], s1 = seq[n + 1], s2 = seq[n + 2];                 // last three of window ++ accepted inputs
    cw[c] = s0; cw[C + c] = s1; cw[2 * C + c] = s2;
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    UNR for (int t = 0; t < TT; t++) {
        float x = P[(size_t)t * ldp + c];
        float v = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        cx[(size_t)t * C + c] = x; Cx[(size_t)t * C + c] = siluB(v);
        s0 = s1; s1 = s2; s2 = x;
    }
}
// gated delta rule for TT tokens with deferred updates (generalizes dnetB: same formulas for TT = 2)
__kernel void dnetN(__global const float* Cx, int C, int KD, __global const float* P, int ldp, int ZO, int BO, int AO,
                    __global const float* A_log, __global const float* dtb, __global float* S, __global float* PK, __global float* PD, __global float* PE,
                    __global const float* gw, __global float* out, int VH, float eps, int gplus1, __global const int* acc) {
    __local float sk[TT][128], sq[TT][128], spk[TT][128], r[256], pa[2 * TT][2][128];
    int h = get_group_id(0), lid = get_local_id(0), d = lid & 127, part = lid >> 7, kh = h / (VH / (KD / 128));
    float z0 = part == 0 ? 1.0f : 0.0f;
    float q[TT], k[TT], v[TT];
    UNR for (int t = 0; t < TT; t++) {
        q[t] = Cx[(size_t)t * C + kh * 128 + d]; k[t] = Cx[(size_t)t * C + KD + kh * 128 + d]; v[t] = Cx[(size_t)t * C + 2 * KD + h * 128 + d];
        float nq = wsumB(z0 * q[t] * q[t], r), nk = wsumB(z0 * k[t] * k[t], r);
        q[t] *= rsqrt(nq + 1e-6f) * 0.08838834764831845f; k[t] *= rsqrt(nk + 1e-6f);
    }
    int na = acc[0];                                                    // deferred updates 1..na-1 of the previous pass are accepted
    if (part == 0) {
        UNR for (int t = 0; t < TT; t++) { sk[t][d] = k[t]; sq[t][d] = q[t]; }
        UNR for (int t = 0; t < TT - 1; t++) spk[t][d] = PK[((size_t)t * VH + h) * 128 + d];
    }
    float pdl[TT], pe[TT];
    UNR for (int t = 0; t < TT - 1; t++) { pdl[t] = PD[((size_t)t * VH + h) * 128 + d]; pe[t] = PE[t * VH + h]; }
    float kk[TT][TT], kq[TT][TT];
    UNR for (int a_ = 0; a_ < TT; a_++) UNR for (int b_ = 0; b_ < TT; b_++) {
        kq[a_][b_] = b_ >= a_ ? wsumB(z0 * k[a_] * q[b_], r) : 0.0f;
        kk[a_][b_] = b_ > a_ ? wsumB(z0 * k[a_] * k[b_], r) : 0.0f;
    }
    __global float* Sh = S + (size_t)h * 128 * 128;
    int i0 = part * 64;
    float sv[64], bk[TT], bq[TT];
    UNR for (int t = 0; t < TT; t++) { bk[t] = 0.0f; bq[t] = 0.0f; }
    UNR for (int j = 0; j < 64; j++) sv[j] = Sh[(i0 + j) * 128 + d];
    for (int t = 0; t < TT - 1; t++) {
        if (t + 1 < na) { UNR for (int j = 0; j < 64; j++) sv[j] = sv[j] * pe[t] + spk[t][i0 + j] * pdl[t]; }
    }
    UNR for (int j = 0; j < 64; j++) {
        float s = sv[j];
        UNR for (int t = 0; t < TT; t++) { bk[t] += s * sk[t][i0 + j]; bq[t] += s * sq[t][i0 + j]; }
    }
    UNR for (int t = 0; t < TT; t++) { pa[2 * t][part][d] = bk[t]; pa[2 * t + 1][part][d] = bq[t]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    UNR for (int t = 0; t < TT; t++) { bk[t] = pa[2 * t][0][d] + pa[2 * t][1][d]; bq[t] = pa[2 * t + 1][0][d] + pa[2 * t + 1][1][d]; }
    float al = exp(A_log[h]), o[TT], e[TT], dl[TT];
    UNR for (int t = 0; t < TT; t++) {
        __global const float* pt = P + (size_t)t * ldp;
        float bt = 1.0f / (1.0f + exp(-pt[BO + h]));
        float av = pt[AO + h] + dtb[h];
        e[t] = exp(-al * (av > 20.0f ? av : log1p(exp(av))));
        dl[t] = (v[t] - e[t] * bk[t]) * bt;
        o[t] = e[t] * bq[t] + dl[t] * kq[t][t];
        UNR for (int j = t + 1; j < TT; j++) { bk[j] = e[t] * bk[j] + dl[t] * kk[t][j]; bq[j] = e[t] * bq[j] + dl[t] * kq[t][j]; }
    }
    UNR for (int j = 0; j < 64; j++) Sh[(i0 + j) * 128 + d] = sv[j] * e[0] + sk[0][i0 + j] * dl[0];
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (part == 0) { UNR for (int t = 1; t < TT; t++) { PK[((size_t)(t - 1) * VH + h) * 128 + d] = k[t]; PD[((size_t)(t - 1) * VH + h) * 128 + d] = dl[t]; } }
    if (lid == 0) { UNR for (int t = 1; t < TT; t++) PE[(t - 1) * VH + h] = e[t]; }
    float g = gplus1 ? 1.0f + gw[d] : gw[d];
    UNR for (int t = 0; t < TT; t++) {
        float ms = wsumB(z0 * o[t] * o[t], r) / 128.0f;
        __global const float* pt = P + (size_t)t * ldp;
        if (part == 0) out[(size_t)t * VH * 128 + h * 128 + d] = o[t] * rsqrt(ms + eps) * g * siluB(pt[ZO + h * 128 + d]);
    }
}
// judge for TT tokens: O[j] = main prediction after token j (token 0 = a, token j = draft j-1)
__kernel void judgeN(__global int* ta, __global const int* td, __global const int* O, __global int* acc, __global int* pos,
                     __global int* ge, __global int* eq, __global int* mp, __global int* mtok, __global int* outs, __global int* cnt, __global int* nacc) {
    int p = pos[0], c = 1, n = cnt[0];
    while (c < TT && O[c - 1] == td[c - 1]) c++;
    for (int j = 0; j < c; j++) outs[n + j] = O[j];
    cnt[0] = n + c; nacc[0] += c - 1; acc[0] = c; ta[0] = O[c - 1]; pos[0] = p + c;
    for (int j = 1; j <= TT; j++) { ge[j - 1] = c >= j; eq[j - 1] = c == j; mp[j - 1] = p + j; mtok[j - 1] = O[j - 1]; }
    for (int k = 1; k < TT - 1 + 1; k++) mp[TT + k - 1] = p + c + k;             // chained draft positions
}
"""


class SpecN:
    def __init__(self, m: Model, mtp_path, nd=2):
        self.m = m; self.ND = nd; T = self.T = nd + 1
        m.load_mtp(mtp_path, "eh", "post")
        self._pfx = "N:"
        c, H, F, I, eps = m.cfg, m.H, m.F, np.int32, np.float32(m.eps)
        G = m.meta["G"]; xe8 = int(m.meta["fmt"] == "xe8")
        kvo = ["-DKV16"] if m.kv16 else []
        opt = [f"-DG={G}", f"-DXE8={xe8}", "-DR=1", "-DDB=64", f"-DNP={m.np_}", f"-DQPG={m.GQ}", f"-DTT={T}"] + kvo
        rb = dict(zip(("proj", "out", "gu", "down"), (int(x) for x in knob("Q35_SRB", "4,2,4,2").split(","))))
        self.rb = rb
        self.bpr = {r: cl.Program(m.ctx, SRC + BSRC + NSRC).build(options=opt + [f"-DRB={r}"]) for r in sorted(set(rb.values()) | {1})}
        self.bp = self.bpr[1]
        self.cp = cp = {r: cl.Program(m.ctx, SRC).build(options=[f"-DG={G}", f"-DXE8={xe8}", f"-DR={r}", "-DDB=64", f"-DNP={m.np_}", "-DCOND", "-DPOSB", f"-DQPG={m.GQ}"] + kvo) for r in (1, 2)}
        mf = cl.mem_flags
        f32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        i32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        NPm = max(m.tm[f"layers.{i}.proj.codes"]["shape"][0] for i in range(c["num_hidden_layers"]))
        CC = 2 * m.KD + m.VH * 128
        self.hT, self.tT, self.aT = f32(T * H), f32(T * H), f32(T * max(H, m.VH * 128, m.NH * 256))
        self.PT, self.CT, self.GUT, self.MT = f32(T * NPm), f32(T * CC), f32(T * 2 * F), f32(T * F)
        self.QT, self.lgT = f32(T * m.NH * 256), f32(T * m.V)
        self.PMT, self.PLT, self.POT = f32(T * m.NH * m.SM), f32(T * m.NH * m.SM), f32(T * m.NH * m.SM * 256)
        self.pvT, self.piT = f32(T * m.npart), i32(T * m.npart)
        self.TA, self.ACC, self.POS, self.CNT, self.NACC, self.ALWAYS = (i32(1) for _ in range(6))
        self.TDA = i32(nd)                                    # drafts, contiguous (main pass input)
        self.TD = [i32(1) for _ in range(nd)]                 # draft k written by the k-th MTP pass
        self.O = i32(T)
        self.GE = [i32(1) for _ in range(T)]; self.EQ = [i32(1) for _ in range(T)]
        self.MP = [i32(1) for _ in range(2 * T)]; self.MTOK = [i32(1) for _ in range(T)]
        self.GEa, self.EQa, self.MPa, self.MTOKa = i32(T), i32(T), i32(2 * T), i32(T)
        self.OUTS = i32(m.TM)
        self.lin = {}
        for i, lt in enumerate(c["layer_types"]):
            if lt == "linear_attention":
                self.lin[i] = (f32(3 * CC), f32(T * CC), f32((T - 1) * m.VH * 128), f32((T - 1) * m.VH * 128), f32((T - 1) * m.VH))
        self.keep = [self.tT.get_sub_region(j * H * 4, H * 4) for j in range(T)]
        bk = lambda name, g, l, *a: self._k(self.bp, name, g, l, *a)
        xe = "embed.ecodes" in m.tm; dummy = m.buf["norm"]
        ks = [bk("embedN", (-(-H // 256) * 256, T), (256, 1), m.buf["embed.ecodes"] if xe else dummy, m.buf["embed.efac"] if xe else dummy,
                 m.buf["embed.ers"] if xe else dummy, dummy if xe else m.buf["embed"], I(int(xe)), self.TA, self.TDA, self.hT, I(H))]
        nb = lambda x, w, y, p1: bk("rmsnorm4b", (1024, T), (1024, 1), x, w, y, I(H), eps, I(p1))

        def gv(name, x, y, add):
            t = m.tm[name + ".codes"]; N, K = t["shape"][0], t["shape"][1] * 2
            kind = name.split(".")[-1]; r = self.rb.get(kind, 1)
            assert N % r == 0
            return self._k(self.bpr[r], "gemvBT", (16, -(-(N // r) // 8) * 8), (16, 8), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, y, I(N), I(K), I(add),
                           label="N:gemv." + kind)
        for i, lt in enumerate(c["layer_types"]):
            p = f"layers.{i}."; NPl = m.tm[p + "proj.codes"]["shape"][0]
            ks.append(nb(self.hT, m.buf[p + "in_norm"], self.tT, m.p1))
            ks.append(gv(p + "proj", self.tT, self.PT, 0))
            if lt == "linear_attention":
                S = m.state[i][1]; cw, cx, PK, PD, PE = self.lin[i]; ZO = CC; BO = ZO + m.VH * 128; AO = BO + m.VH
                ks.append(bk("convN", (-(-CC // 256) * 256,), (256,), self.PT, I(NPl), cw, cx, m.buf[p + "conv"], self.CT, I(CC), self.ACC))
                ks.append(bk("dnetN", (m.VH * 256,), (256,), self.CT, I(CC), I(m.KD), self.PT, I(NPl), I(ZO), I(BO), I(AO), m.buf[p + "A_log"], m.buf[p + "dt_bias"],
                             S, PK, PD, PE, m.buf[p + "gnorm"], self.aT, I(m.VH), eps, I(m.gp1), self.ACC))
            else:
                kc, vc = m.state[i][:2]
                ks.append(bk("attnprep2", ((m.NH + m.KH) * 256, T), (256, 1), self.PT, I(NPl), m.buf[p + "q_norm"], m.buf[p + "k_norm"], self.QT, kc, vc,
                             self.POS, np.float32(m.theta), I(m.NH), I(m.KH), I(m.RD), eps, I(m.qp1)))
                ks.append(list(self._k(cp[1], "fattn", (m.KH * 128, m.SM, T), (128, 1, 1), self.QT, kc, vc, self.PMT, self.PLT, self.POT, I(0), I(m.KH), I(m.SM), I(m.CH),
                                       self.POS, self.ALWAYS, I(-1), label="N:attn")))
                ks.append(self._k(cp[1], "fcomb", (m.NH * 256, T), (256, 1), self.PMT, self.PLT, self.POT, self.PT, I(NPl), self.aT, I(0), I(m.KH), I(m.SM), I(m.CH),
                                  self.POS, self.ALWAYS, I(-1), label="N:attn"))
            ks.append(gv(p + "out", self.aT, self.hT, 1))
            ks.append(nb(self.hT, m.buf[p + "post_norm"], self.tT, m.p1))
            ks.append(gv(p + "gu", self.tT, self.GUT, 0))
            ks.append(bk("swiglu2", (-(-F // 256) * 256, T), (256, 1), self.GUT, self.MT, I(F)))
            ks.append(gv(p + "down", self.MT, self.hT, 1))
        ks.append(nb(self.hT, m.buf["norm"], self.tT, m.p1))
        ks.append(bk("lmheadT", (16, -(-(m.V // 2) // 8) * 8), (16, 8), m.buf["lm_head.q"], m.buf["lm_head.s"], self.tT, self.lgT, I(m.V), I(H)))
        ks.append(bk("argmax1b", (m.npart * 256, T), (256, 1), self.lgT, I(m.V), self.pvT, self.piT))
        ks.append(bk("argmax2b", (T * 256,), (256,), self.pvT, self.piT, I(m.npart), self.O))
        ks.append(bk("judgeN", (1,), (1,), self.TA, self.TDA, self.O, self.ACC, self.POS, self.GEa, self.EQa, self.MPa, self.MTOKa, self.OUTS, self.CNT, self.NACC))
        self.main = ks
        self._vc = {}; self.vk = 0
        import os
        vp = knob("Q35_MTPVOCAB", os.path.join(os.path.dirname(m.path), "vocab_order.npy"))
        vk = int(knob("Q35_MTPK", "65536"))
        if vk and os.path.exists(vp):
            self._load_vocab(np.load(vp), vk)
        self._build_mtp()

    def _load_vocab(self, order, K):
        m = self.m; idx = np.sort(order[:K]).astype(np.int32)
        t, ts = m.tm["lm_head.q"], m.tm["lm_head.s"]
        raw = np.memmap(m.path, dtype=np.uint8, mode="r")
        q = np.asarray(raw[t["off"]:t["off"] + t["nbytes"]]).reshape(m.V, m.H)[idx]
        sc = np.asarray(raw[ts["off"]:ts["off"] + ts["nbytes"]]).view(np.float32)[idx]
        mf = cl.mem_flags; mk = lambda a: cl.Buffer(m.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))
        self._vc[K] = (mk(q), mk(sc), mk(idx), idx); self.vk = K

    def _k(self, prg, name, g, l, *a, label=None):
        k = cl.Kernel(prg, name); k.set_args(*a)
        self.m.lab[id(k)] = label or (self._pfx + name)
        return (k, g, l)

    def _mtp(self, tok, hid, mp, cf_first, w_first, cf_rest, w_rest, out_td):
        """One MTP pass: embedding of `tok` + hidden `hid` at position mp. The part up to the KV write runs when
        cf_first == w_first, the rest (attention output .. draft into out_td) when cf_rest == w_rest (w = -1: always)."""
        m, H, F, I, eps, cp = self.m, self.m.H, self.m.F, np.int32, np.float32(self.m.eps), self.cp
        K = lambda prg_r, name, g, l, cf, w, *a: self._k(cp[prg_r], name, g, l, *a, cf, I(w))
        xe = "embed.ecodes" in m.tm; e1, e2 = m._keep; out = []
        if xe:
            out.append(K(1, "embed_xe", (-(-H // 256) * 256,), (256,), cf_first, w_first, m.buf["embed.ecodes"], m.buf["embed.efac"], m.buf["embed.ers"], tok, m.mE, I(H)))
        else:
            out.append(K(1, "embed", (-(-H // 256) * 256,), (256,), cf_first, w_first, m.buf["embed"], tok, m.mE, I(H)))
        nrm = lambda cf, w, x, wt, y: K(1, "rmsnorm4", (1024,), (1024,), cf, w, x, wt, y, I(H), eps, I(1))

        def gv(cf, w, name, x, y, add):
            t = m.tm[name + ".codes"]; N, Kk = t["shape"][0], t["shape"][1] * 2
            R = 2 if name.endswith(".gu") else 1
            return K(R, "gemv2", (16, -(-(N // R) // 8) * 8), (16, 8), cf, w, m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, y, I(N), I(Kk), I(add))
        A, B = (cf_first, w_first), (cf_rest, w_rest)
        out += [nrm(*A, m.mE, m.buf["mtp.pe"], e1), nrm(*A, hid, m.buf["mtp.ph"], e2),
                gv(*A, "mtp.fcA", e1, m.mh, 0), gv(*A, "mtp.fcB", e2, m.mh, 1),
                nrm(*A, m.mh, m.buf["mtp.in_norm"], m.t), gv(*A, "mtp.proj", m.t, m.P, 0)]
        out.append(self._k(cp[1], "attn_prep", ((m.NH + m.KH) * 256,), (256,), m.P, m.buf["mtp.q_norm"], m.buf["mtp.k_norm"], m.Q, m.mkc, m.mvc,
                           I(0), np.float32(m.theta), I(m.NH), I(m.KH), I(m.RD), eps, I(1), mp, cf_first, I(w_first)))
        out.append(list(self._k(cp[1], m.fa1, (m.KH * m._fls(), m.SM, 1), (m._fls(), 1, 1), m.Q, m.mkc, m.mvc, m.PM, m.PL, m.PO, I(0), I(m.KH), I(m.SM), I(m.CH), mp, cf_rest, I(w_rest))))
        out.append(self._k(cp[1], "fcomb", (m.NH * 256, 1), (256, 1), m.PM, m.PL, m.PO, m.P, I(0), m.a, I(0), I(m.KH), I(m.SM), I(m.CH), mp, cf_rest, I(w_rest)))
        out += [gv(*B, "mtp.out", m.a, m.mh, 1), nrm(*B, m.mh, m.buf["mtp.post_norm"], m.t), gv(*B, "mtp.gu", m.t, m.GU, 0),
                K(1, "swiglu", (-(-F // 256) * 256,), (256,), *B, m.GU, m.M, I(F)), gv(*B, "mtp.down", m.M, m.mh, 1),
                nrm(*B, m.mh, m.buf["mtp.norm"], m.t)]
        if self.vk:
            qb, sb, ib, idx = self._vc[self.vk]; Nv = len(idx); npv = (Nv + 4095) // 4096
            out += [K(2, "gemv_i8r", (16, -(-(Nv // 2) // 8) * 8), (16, 8), *B, qb, sb, m.t, m.logit, I(Nv), I(H)),
                    K(1, "argmax1", (npv * 256,), (256,), *B, m.logit, I(Nv), m.pv, m.pi),
                    K(1, "argmax2", (256,), (256,), *B, m.pv, m.pi, I(npv), out_td, m.mouts, I(0)),
                    K(1, "mapid", (1,), (1,), *B, out_td, ib)]
        else:
            out += [K(2, "gemv_i8r", (16, -(-(m.V // 2) // 8) * 8), (16, 8), *B, m.buf["lm_head.q"], m.buf["lm_head.s"], m.t, m.logit, I(m.V), I(H)),
                    K(1, "argmax1", (m.npart * 256,), (256,), *B, m.logit, I(m.V), m.pv, m.pi),
                    K(1, "argmax2", (256,), (256,), *B, m.pv, m.pi, I(m.npart), out_td, m.mouts, I(0))]
        return out

    def _build_mtp(self):
        """Per accepted count c: passes j = 1..T write MTP KV at p+j (token O[j-1], hidden row j-1) when c >= j; the pass with
        j == c also drafts d0. Then ND-1 chained passes (token = previous draft, hidden = MTP output) at p+c+1.."""
        self._pfx = "M:"
        lst = []
        for j in range(1, self.T + 1):
            lst += self._mtp(self.MTOK[j - 1], self.keep[j - 1], self.MP[j - 1], self.GE[j - 1], 1, self.EQ[j - 1], 1, self.TD[0])
        for k in range(1, self.ND):
            lst += self._mtp(self.TD[k - 1], self.m.t, self.MP[self.T + k - 1], self.ALWAYS, -1, self.ALWAYS, -1, self.TD[k])
        self.mtp = lst

    def _scatter(self):
        """judgeN writes arrays; the MTP passes read single-int buffers: device-side copies (no host wait)."""
        q = self.m.q
        for j in range(self.T):
            cl.enqueue_copy(q, self.GE[j], self.GEa, src_offset=4 * j, byte_count=4)
            cl.enqueue_copy(q, self.EQ[j], self.EQa, src_offset=4 * j, byte_count=4)
            cl.enqueue_copy(q, self.MTOK[j], self.MTOKa, src_offset=4 * j, byte_count=4)
        for j in range(2 * self.T - 1):
            cl.enqueue_copy(q, self.MP[j], self.MPa, src_offset=4 * j, byte_count=4)

    def _gather(self):
        q = self.m.q
        for k in range(self.ND):
            cl.enqueue_copy(q, self.TDA, self.TD[k], dst_offset=4 * k, byte_count=4)

    def _start(self, T):
        """After the prompt (main state, KV and MTP KV up to T): first token y0, drafts by chained MTP from position T."""
        m, I = self.m, np.int32
        m.mtp_step(T, 0)
        for k in range(1, self.ND):
            cl.enqueue_copy(m.q, m.tok, m.mtok); m.mtp_step(T + k, k)
        y0, d = np.zeros(1, np.int32), np.zeros(self.ND, np.int32)
        cl.enqueue_copy(m.q, y0, m.outs); cl.enqueue_copy(m.q, d, m.mouts); m.q.finish()
        for b, v in ((self.TA, y0[0]), (self.ACC, 0), (self.POS, T), (self.CNT, 0), (self.NACC, 0), (self.ALWAYS, 1)):
            cl.enqueue_copy(m.q, b, np.array([v], np.int32))
        cl.enqueue_copy(m.q, self.TDA, d)
        for i, (cw, cx, PK, PD, PE) in self.lin.items():
            st, S, PK0, PD0, PE0 = m.state[i]
            cl.enqueue_copy(m.q, cw, st)                                    # prefill/decode conv state = last three inputs
            # fold any deferred update of the plain decoder into S (prefill leaves PD = 0, PE = 1, so this is exact)
        m.q.finish()
        self.base, self.it = T, 0
        return int(y0[0])

    def prompt(self, ids):
        m = self.m; m.reset(); m.load_ids(ids)
        for kq in (m.mkc, m.mvc):
            cl.enqueue_fill_buffer(m.q, kq, np.float32(0), 0, kq.size)
        T = len(ids)
        for i in range(T):
            m.step(i, out_step=0 if i == T - 1 else None, feed=i)
            if i < T - 1:
                m.st[0].set_arg(1, np.int32(i + 1)); m.run([m.st]); m.mtp_step(i + 1, 0)
        self._fold_plain()
        return self._start(T)

    def _fold_plain(self):
        """The plain decoder keeps one deferred delta-rule update (S*PE + PK*PD); fold it into S before multi-token passes."""
        m = self.m; c = m.cfg
        if not hasattr(self, "_fk"):
            src = r"""__kernel void fold(__global float* S, __global const float* PK, __global float* PD, __global float* PE) {
                int h = get_group_id(0), d = get_local_id(0); __global float* Sh = S + (size_t)h * 16384;
                float pe = PE[h], pd = PD[h * 128 + d];
                for (int i = 0; i < 128; i++) Sh[i * 128 + d] = Sh[i * 128 + d] * pe + PK[h * 128 + i] * pd;
                barrier(CLK_GLOBAL_MEM_FENCE); PD[h * 128 + d] = 0.0f; if (d == 0) PE[h] = 1.0f; }"""
            self._fk = cl.Kernel(cl.Program(m.ctx, src).build(), "fold")
        for i, lt in enumerate(c["layer_types"]):
            if lt == "linear_attention":
                st, S, PK, PD, PE = m.state[i]
                self._fk.set_args(S, PK, PD, PE); cl.enqueue_nd_range_kernel(m.q, self._fk, (m.VH * 128,), (128,))

    def prompt_fast(self, ids, pf, start=0, cuts=(), on_cut=None):
        m = self.m; T = len(ids)
        if start == 0:
            m.reset()
        m.load_ids(list(ids) + [0])
        if start == 0:
            for kq in (m.mkc, m.mvc):
                cl.enqueue_fill_buffer(m.q, kq, np.float32(0), 0, kq.size)
        pf.run_long(T, mtp=True, start=start, cuts=cuts, on_cut=on_cut)
        pf.first_token(0)
        self._fold_plain()
        return self._start(T)

    def iterate(self, n):
        m = self.m
        ents = [e for e in self.main + self.mtp if isinstance(e, list) and e[0].function_name in ("fattn", "fattn1", "fattn2")]
        for _ in range(n):
            S = min(m.SM, -(-(self.base + self.T * self.it + 2 * self.T) // m.CH))
            for e in ents:
                e[1] = (e[1][0], S, e[1][2])
            m.run(self.main); self._scatter(); m.run(self.mtp); self._gather(); m.q.flush()
            self.it += 1

    def read(self):
        c, a = np.zeros(1, np.int32), np.zeros(1, np.int32)
        cl.enqueue_copy(self.m.q, c, self.CNT); cl.enqueue_copy(self.m.q, a, self.NACC); self.m.q.finish()
        o = np.zeros(max(1, int(c[0])), np.int32); cl.enqueue_copy(self.m.q, o, self.OUTS); self.m.q.finish()
        return o[:int(c[0])].tolist(), int(a[0])
