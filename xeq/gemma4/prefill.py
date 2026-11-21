"""Gemma 4 26B-A4B XeFP4 prompt processing (prefill), chunks of up to 512 tokens. Per layer:
dense GEMM (W4A8 XMX) -> per-head norm, rotary, KV cache -> attention (two BF16 XMX GEMMs past 512 positions)
-> MoE: router -> top 8 -> per-expert token lists and tiles -> expert gate/up and down GEMMs -> weighted sum.
The sliding-layer KV ring holds window + chunk slots, so writing one chunk never overwrites keys still needed.
"""
from __future__ import annotations


import numpy as np
import pyopencl as cl

from . import model as GR
from ..device import knob

PSRC = r"""
inline float dotE(uint w, float8 v) { return dot(convert_float4(as_char4(dlo(w))), v.even) + dot(convert_float4(as_char4(dhi(w))), v.odd); }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemmT(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, int ldx,
                     __global float* y, int ldy, int NR, int K, int add, int mode, int Tn,
                     __global const int* tiles, __global const int* xidx, __global const int* yidx) {
    int r = get_global_id(1), l = get_local_id(0), tl = get_group_id(2);
    if (r >= NR) return;
    int e, st, cnt;
    if (mode == 0) { e = 0; st = tl * TB; cnt = min(TB, Tn - st); } else { e = tiles[tl * 4]; st = tiles[tl * 4 + 1]; cnt = tiles[tl * 4 + 2]; }
    if (cnt <= 0) return;
    size_t row = (size_t)e * NR + r;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    __global const uint* cr = codes + row * (K / 8);
    __global const uint* fr = fac + row * FW;
    int xr[TB]; float acc[TB];
    UNR for (int k = 0; k < TB; k++) { int kk = min(k, cnt - 1); xr[k] = mode ? xidx[st + kk] : st + kk; acc[k] = 0.0f; }
    for (int b = l; b < nb; b += 16) {
        uint4 u = vload4(b, cr);
        int bo = b * 3; ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        float mm = (float)(((fw >> (bo & 31)) & 7) + 1);
        UNR for (int k = 0; k < TB; k++) {
            if (k < cnt) {
                __global const float* xp = x + (size_t)xr[k] * ldx + b * 32;
                float s = dotE(u.s0, vload8(0, xp)) + dotE(u.s1, vload8(1, xp)) + dotE(u.s2, vload8(2, xp)) + dotE(u.s3, vload8(3, xp));
                acc[k] += s * mm;
            }
        }
    }
    float sc = vload_half(row, rs) * 0.125f;
    UNR for (int k = 0; k < TB; k++) {
        float v = sub_group_reduce_add(acc[k]) * sc;
        if (l == 0 && k < cnt) { int yr = mode ? yidx[st + k] : st + k; size_t o = (size_t)yr * ldy + r; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void igemmT(__global const char* q, __global const float* sc, __global const float* x, int ldx, __global float* y, int ldy, int N, int K, int add, int Tn) {
    int r = get_global_id(1), l = get_local_id(0), st = get_group_id(2) * TB; if (r >= N) return;
    int cnt = min(TB, Tn - st); if (cnt <= 0) return;
    float acc[TB];
    UNR for (int k = 0; k < TB; k++) acc[k] = 0.0f;
    for (int k0 = l * 16; k0 < K; k0 += 256) {
        float16 w = convert_float16(vload16(0, q + (size_t)r * K + k0));
        UNR for (int k = 0; k < TB; k++) if (k < cnt) acc[k] += d16(w, vload16(0, x + (size_t)(st + k) * ldx + k0));
    }
    UNR for (int k = 0; k < TB; k++) {
        float v = sub_group_reduce_add(acc[k]) * sc[r];
        if (l == 0 && k < cnt) { size_t o = (size_t)(st + k) * ldy + r; y[o] = add ? y[o] + v : v; }
    }
}
__kernel void embedT(__global const char* q, __global const float* sc, __global const int* ids, int i0, __global float* x, int H, float nrm,
                     __global const int* smap, __global const float* soft) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    int si = smap[i0 + t];
    if (si >= 0) { x[(size_t)t * H + i] = soft[(size_t)si * H + i]; return; }
    size_t n = ids[i0 + t];
    x[(size_t)t * H + i] = bfr(bfr((float)q[n * H + i] * sc[n]) * nrm);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gnormT(__global const float* x, __global const float* w, __global float* y, int H, float eps) {
    __local float r[64]; __local float tot;
    int l = get_local_id(0); size_t ro = (size_t)get_group_id(1) * H; x += ro; y += ro;
    float s = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = x[i]; s += v * v; }
    float k = rsqrt(bsum(s, r, &tot) / H + eps);
    for (int i = l; i < H; i += 1024) y[i] = x[i] * k * w[i];
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void postattnT(__global const float* o, __global const float* wpa, __global float* h, __global const float* w1, __global const float* w3,
                        __global float* t1, __global float* t3, __global float* t2, int H, float eps) {
    __local float r[64]; __local float tot; __local float hh[4096];
    int l = get_local_id(0); size_t ro = (size_t)get_group_id(1) * H; o += ro; h += ro; t1 += ro; t3 += ro; t2 += ro;
    float s = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = o[i]; s += v * v; }
    float k = rsqrt(bsum(s, r, &tot) / H + eps), s2 = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = h[i] + o[i] * k * wpa[i]; hh[i] = v; h[i] = v; s2 += v * v; }
    float k2 = rsqrt(bsum(s2, r, &tot) / H + eps);
    for (int i = l; i < H; i += 1024) { float v = hh[i] * k2; t1[i] = v * w1[i]; t3[i] = v * w3[i]; t2[i] = v; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ffend2T(__global const float* d1, __global const float* e, __global const float* w1, __global const float* w2, __global const float* w,
                      __global const float* ls, __global float* h, __global const float* wn, __global float* tn, int H, float eps) {
    __local float r[64]; __local float tot; __local float t[4096];
    int l = get_local_id(0); size_t ro = (size_t)get_group_id(1) * H; d1 += ro; e += ro; h += ro; tn += ro;
    float s1 = 0.0f, s2 = 0.0f;
    for (int i = l; i < H; i += 1024) { float a = d1[i], b = e[i]; s1 += a * a; s2 += b * b; }
    float k1 = rsqrt(bsum(s1, r, &tot) / H + eps), k2 = rsqrt(bsum(s2, r, &tot) / H + eps);
    float s3 = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = d1[i] * k1 * w1[i] + e[i] * k2 * w2[i]; t[i] = v; s3 += v * v; }
    float k3 = rsqrt(bsum(s3, r, &tot) / H + eps), sc = ls[0], s4 = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = (h[i] + t[i] * k3 * w[i]) * sc; h[i] = v; t[i] = v; s4 += v * v; }
    float k4 = rsqrt(bsum(s4, r, &tot) / H + eps);
    for (int i = l; i < H; i += 1024) tn[i] = t[i] * k4 * wn[i];
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gprepT(__global const float* p, int ldp, __global const float* qn, __global const float* kn, __global float* Q, __global KT* kc, __global KT* vc,
                     int pos0, float theta, int NH, int KH, int HD, int NR, int VOFF, int CAP, float eps) {
    __local float r[64]; __local float tot; __local float sh[512];
    int g = get_group_id(0), t = get_group_id(1), l = get_local_id(0), NC = HD / 256, half_ = HD / 2, pos = pos0 + t;
    p += (size_t)t * ldp; Q += (size_t)t * NH * HD;
    __global const float* src = g < NH ? p + g * HD : p + NH * HD + (g - NH) * HD;
    __global const float* w = g < NH ? qn : kn;
    float xv[2], s = 0.0f;
    for (int c = 0; c < NC; c++) { xv[c] = src[l + c * 256]; s += xv[c] * xv[c]; }
    float k = rsqrt(bsum(s, r, &tot) / HD + eps);
    for (int c = 0; c < NC; c++) { int d = l + c * 256; sh[d] = xv[c] * k * w[d]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    size_t slot = (size_t)(pos % CAP);
    for (int c = 0; c < NC; c++) {
        int d = l + c * 256, i = d < half_ ? d : d - half_;
        float y = sh[d];
        if (i < NR) {
            float f = (float)pos * pow(theta, -2.0f * (float)i / (float)HD), cs = cos(f), sn = sin(f);
            y = d < half_ ? sh[d] * cs - sh[d + half_] * sn : sh[d] * cs + sh[d - half_] * sn;
        }
        if (g < NH) Q[g * HD + d] = y; else KST(y, (slot * KH + (g - NH)) * HD + d, kc);
    }
    if (g >= NH) {
        __global const float* vs = p + VOFF + (g - NH) * HD;
        float s2 = 0.0f, vv[2];
        for (int c = 0; c < NC; c++) { vv[c] = vs[l + c * 256]; s2 += vv[c] * vv[c]; }
        float k2 = rsqrt(bsum(s2, r, &tot) / HD + eps);
        for (int c = 0; c < NC; c++) KST(vv[c] * k2, (slot * KH + (g - NH)) * HD + l + c * 256, vc);
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gattnT(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* A,
                     int pos0, int Tn, int CAP, int KH, int GQ, int HD, int WIN, __global const int* ub, int bid) {
    int h = get_group_id(0), NH = get_num_groups(0), sg = get_sub_group_id(), l = get_sub_group_local_id(), t = get_group_id(1) * 8 + sg;
    if (t >= Tn) return;
    int L = pos0 + t + 1, lo = max(0, L - WIN), kh = h / GQ, NC = HD / 256;
    if (bid) L = max(L, ub[pos0 + t]);
    __global const float* qp = Q + ((size_t)t * NH + h) * HD;
    float16 q0 = vload16(0, qp + l * 16), q1 = NC > 1 ? vload16(0, qp + 256 + l * 16) : (float16)(0.0f);
    float m = -INFINITY, su = 0.0f; float16 o0 = (float16)(0.0f), o1 = (float16)(0.0f);
    for (int j = lo; j < L; j++) {
        size_t sl = (size_t)(j % CAP);
        __global const KT* kr = kc + (sl * KH + kh) * HD; __global const KT* vr = vc + (sl * KH + kh) * HD;
        float c = d16(q0, KLD16(0, kr + l * 16));
        if (NC > 1) c += d16(q1, KLD16(0, kr + 256 + l * 16));
        c = sub_group_reduce_add(c);
        float mn = fmax(m, c), a = exp(m - mn), b = exp(c - mn);
        su = su * a + b; o0 = o0 * a + KLD16(0, vr + l * 16) * b;
        if (NC > 1) o1 = o1 * a + KLD16(0, vr + 256 + l * 16) * b;
        m = mn;
    }
    __global float* ap = A + ((size_t)t * NH + h) * HD;
    vstore16(o0 / su, 0, ap + l * 16); if (NC > 1) vstore16(o1 / su, 0, ap + 256 + l * 16);
}
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((intel_reqd_sub_group_size(16)))      // r[8] assumes 8 sub-groups of 16 in a 128-wide work-group
__kernel void routerT(__global const float* w, __global const float* x, __global float* lg, int E, int K) {
    __local float r[8];
    int e = get_group_id(0), t = get_group_id(1), l = get_local_id(0);
    float acc = 0.0f;
    for (int k = l * 4; k < K; k += 512) acc += dot(vload4(0, w + (size_t)e * K + k), vload4(0, x + (size_t)t * K + k));
    acc = sub_group_reduce_add(acc);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) { float s = 0.0f; for (int i = 0; i < 8; i++) s += r[i]; lg[(size_t)t * E + e] = s; }
}
#define REB 16
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void routerT2(__global const float* w, __global const float* x, __global float* lg, int E, int K) {
    // routerT for REB experts per work-group: x is read once, and every (expert, token) sum keeps routerT's order
    // (thread l: k = 4l, 4l+512, ...; sub-group sums; then sub-groups 0..7 in order), so the logits are the same bits.
    __local float r[REB][8];
    int e0 = get_group_id(0) * REB, t = get_group_id(1), l = get_local_id(0), sg = get_sub_group_id();
    float4 xv[6];
    UNR for (int i = 0; i < 6; i++) { int k = l * 4 + i * 512; xv[i] = k < K ? vload4(0, x + (size_t)t * K + k) : (float4)(0.0f); }
    for (int j = 0; j < REB; j++) {
        __global const float* we = w + (size_t)(e0 + j) * K; float acc = 0.0f;
        UNR for (int i = 0; i < 6; i++) { int k = l * 4 + i * 512; if (k < K) acc += dot(vload4(0, we + k), xv[i]); }
        acc = sub_group_reduce_add(acc);
        if (get_sub_group_local_id() == 0) r[j][sg] = acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < REB) { float s = 0.0f; for (int i = 0; i < 8; i++) s += r[l][i]; lg[(size_t)t * E + e0 + l] = s; }
}
__kernel void topkT(__global const float* lg, __global const float* pes, __global int* idx, __global float* wt, int E, int TK) {
    __local float v[128]; __local int sel[16]; __local float pe[16];
    int l = get_local_id(0), t = get_group_id(1);
    float xv = l < E ? lg[(size_t)t * E + l] : -INFINITY; v[l] = xv; barrier(CLK_LOCAL_MEM_FENCE);
    int rk = 0;
    for (int i = 0; i < 128; i++) { float y = v[i]; rk += (y > xv) || (y == xv && i < l); }
    if (rk < TK) sel[rk] = l;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < TK) pe[l] = exp(v[sel[l]] - v[sel[0]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < TK) { float ps = 0.0f; for (int k = 0; k < TK; k++) ps += pe[k]; idx[t * TK + l] = sel[l]; wt[t * TK + l] = pe[l] / ps * pes[sel[l]]; }
}
__kernel void buildl(__global const int* idx, int A, int E, int TK, __global int* ltok, __global int* lasg, __global int* tiles, int maxt) {
    __local int cnt[128], off[128], cur[128], toff[128];
    int l = get_local_id(0);
    if (l < E) { cnt[l] = 0; cur[l] = 0; }
    for (int i = l; i < maxt * 4; i += 1024) tiles[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int a = l; a < A; a += 1024) atomic_inc(&cnt[idx[a]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) { int s = 0, ts = 0; for (int e = 0; e < E; e++) { off[e] = s; s += cnt[e]; toff[e] = ts; ts += (cnt[e] + TB - 1) / TB; } }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int a = l; a < A; a += 1024) { int e = idx[a], p = atomic_inc(&cur[e]); ltok[off[e] + p] = a / TK; lasg[off[e] + p] = a; }
    if (l < E) for (int k = 0; k * TB < cnt[l]; k++) { int ti = toff[l] + k; tiles[ti * 4] = l; tiles[ti * 4 + 1] = off[l] + k * TB; tiles[ti * 4 + 2] = min(TB, cnt[l] - k * TB); }
}
__kernel void ecomb(__global const float* eo, __global const float* wt, __global float* e, int H, int TK) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    float s = 0.0f;
    for (int j = 0; j < TK; j++) s += wt[t * TK + j] * eo[((size_t)t * TK + j) * H + i];
    e[(size_t)t * H + i] = s;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void nllT(__global const float* lg, int V, float cap, __global const int* ids, int tgt0, __global float* out, int out0) {
    __local float r[64]; __local float tot; __local float rv[1024]; __local int ri[1024];
    int l = get_local_id(0), row = get_group_id(1); lg += (size_t)row * V;
    float mx = -INFINITY; int mi = 0;
    for (int i = l; i < V; i += 1024) { float v = cap * tanh(lg[i] / cap); if (v > mx) { mx = v; mi = i; } }
    rv[l] = mx; ri[l] = mi; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 512; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    float M = rv[0]; int am = ri[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.0f; for (int i = l; i < V; i += 1024) s += exp(cap * tanh(lg[i] / cap) - M);
    s = bsum(s, r, &tot);
    if (l == 0) { int tt = ids[tgt0 + row]; float lt = cap * tanh(lg[tt] / cap); out[2 * (out0 + row)] = M + log(s) - lt; out[2 * (out0 + row) + 1] = am == tt ? 1.0f : 0.0f; }
}
"""


AMSRC = r"""
inline uint bfh(float x) { uint u = as_uint(x); return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
__kernel void packQg(__global const float* Q, __global uint* Aq, int NH, int kh, int GQ, int HD, int rb0, int R) {
    int c2 = get_global_id(0), r = get_global_id(1), rg = rb0 + r, KB = HD / 16; uint v = 0;
    if (rg < R) { int t = rg / GQ, g = rg % GQ; __global const float* q = Q + ((size_t)t * NH + kh * GQ + g) * HD; v = bfh(q[2 * c2]) | (bfh(q[2 * c2 + 1]) << 16); }
    Aq[((size_t)(r / 8) * KB + c2 / 8) * 64 + (r % 8) * 8 + c2 % 8] = v;
}
__kernel void packKg(__global const KT* kc, __global uint* Bk, int KH, int kh, int HD, int jlo, int jhi, int CAP) {
    int n = get_global_id(0), c2 = get_global_id(1), KB = HD / 16, p = jlo + n; uint v = 0;
    if (p < jhi) { __global const KT* k = kc + ((size_t)(p % CAP) * KH + kh) * HD; v = bfh(KLD1(2 * c2, k)) | (bfh(KLD1(2 * c2 + 1, k)) << 16); }
    Bk[((size_t)(n / 8) * KB + c2 / 8) * 64 + (c2 % 8) * 8 + n % 8] = v;
}
__kernel void packVg(__global const KT* vc, __global uint* Bv, int KH, int kh, int HD, int jlo, int jhi, int CAP, int KBp) {
    int d = get_global_id(0), p2 = get_global_id(1), p = jlo + 2 * p2; float a = 0.0f, b = 0.0f;
    if (p < jhi) a = KLD1(((size_t)(p % CAP) * KH + kh) * HD + d, vc);
    if (p + 1 < jhi) b = KLD1(((size_t)((p + 1) % CAP) * KH + kh) * HD + d, vc);
    Bv[((size_t)(d / 8) * KBp + p2 / 8) * 64 + (p2 % 8) * 8 + d % 8] = bfh(a) | (bfh(b) << 16);
}
__kernel void softPg(__global const float* S, int Lp, __global uint* Ap, int pos0, int rb0, int R, int GQ, int jlo, int WIN, __global const int* ub, int bid) {
    __local float red[256];
    int r = get_group_id(0), l = get_local_id(0), KB = Lp / 16, rg = rb0 + r, a0 = 0, a1 = 0;
    if (rg < R) { int L = pos0 + rg / GQ + 1, lo = max(0, L - WIN); if (bid) L = max(L, ub[pos0 + rg / GQ]); a0 = lo - jlo; a1 = L - jlo; }
    __global const float* s = S + (size_t)r * Lp;
    float mx = -INFINITY;
    for (int j = a0 + l; j < a1; j += 256) mx = fmax(mx, s[j]);
    red[l] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) red[l] = fmax(red[l], red[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sm = 0.0f;
    for (int j = a0 + l; j < a1; j += 256) sm += exp(s[j] - mx);
    red[l] = sm; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) red[l] += red[l + q]; barrier(CLK_LOCAL_MEM_FENCE); }
    float inv = a1 > a0 ? 1.0f / red[0] : 0.0f;
    for (int c2 = l; c2 < Lp / 2; c2 += 256) {
        int j = 2 * c2;
        float a = (j >= a0 && j < a1) ? exp(s[j] - mx) * inv : 0.0f, b = (j + 1 >= a0 && j + 1 < a1) ? exp(s[j + 1] - mx) * inv : 0.0f;
        Ap[((size_t)(r / 8) * KB + c2 / 8) * 64 + (r % 8) * 8 + c2 % 8] = bfh(a) | (bfh(b) << 16);
    }
}
__kernel void writeOg(__global const float* O, __global float* A, int NH, int kh, int GQ, int HD, int rb0, int R) {
    int d = get_global_id(0), r = get_global_id(1), rg = rb0 + r; if (rg >= R) return;
    int t = rg / GQ, g = rg % GQ; A[((size_t)t * NH + kh * GQ + g) * HD + d] = O[(size_t)r * HD + d];
}
"""

# Skinny expert GEMM for few rows per expert (prefill chunk <= 1024): no SLM, no barriers; each sub-group decodes its
# own columns straight into the DPAS B operand and reads A from global. Same per-element accumulation order as pgemm8e.
SKSRC = r"""#ifndef SR
#define SR 4
#endif
#ifndef SNT
#define SNT 4
#endif
#ifndef SWN
#define SWN 4
#endif
inline int8 decB(uint4 w, char mq) {
    uint l0 = as_uint(as_char4(dlo(w.s0)) * mq), h0 = as_uint(as_char4(dhi(w.s0)) * mq);
    uint l1 = as_uint(as_char4(dlo(w.s1)) * mq), h1 = as_uint(as_char4(dhi(w.s1)) * mq);
    uint l2 = as_uint(as_char4(dlo(w.s2)) * mq), h2 = as_uint(as_char4(dhi(w.s2)) * mq);
    uint l3 = as_uint(as_char4(dlo(w.s3)) * mq), h3 = as_uint(as_char4(dhi(w.s3)) * mq);
    return as_int8((uint8)(il0(l0, h0), il1(l0, h0), il0(l1, h1), il1(l1, h1), il0(l2, h2), il1(l2, h2), il0(l3, h3), il1(l3, h3)));
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void pgemm8s(__global const uint* A, __global const float* SAt, int Mp, __global const uint* codes, __global const uint* fac, __global const half* rs,
                      __global float* Y, int M, int N, int K, int ldy, int add, __global const int* tileE, int useE) {
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int KB = K / 32, FW = (((K / G) * 3 + 31) / 32), KW = K / 8;
    int rt0 = get_group_id(1) * SR;
    int eid = useE ? tileE[get_group_id(1)] : 0;
    int valid = eid >= 0; eid = max(eid, 0);
    codes += (size_t)eid * N * KW; fac += (size_t)eid * N * FW; rs += (size_t)eid * N;
    int ct0 = (get_group_id(0) * SWN + sg) * SNT;
    __global const uint* cp[SNT]; __global const uint* fp[SNT]; int ok[SNT];
    UN for (int c = 0; c < SNT; c++) { int n = (ct0 + c) * 8 + l; ok[c] = n < N; n = min(n, N - 1); cp[c] = codes + (size_t)n * KW; fp[c] = fac + (size_t)n * FW; }
    float8 acc[SR][SNT];
    UN for (int r = 0; r < SR; r++) UN for (int c = 0; c < SNT; c++) acc[r][c] = (float8)(0.f);
    uint4 wn[SNT];
    UN for (int c = 0; c < SNT; c++) wn[c] = vload4(0, cp[c]);
    for (int kq = 0; kq < KB; kq++) {
        uint4 w[SNT];
        UN for (int c = 0; c < SNT; c++) w[c] = wn[c];
        if (kq + 1 < KB) { UN for (int c = 0; c < SNT; c++) wn[c] = vload4(0, cp[c] + (kq + 1) * 4); }
        int bo = ((kq * 32) / G) * 3;
        int8 b[SNT];
        UN for (int c = 0; c < SNT; c++) {
            ulong fw = (ulong)fp[c][bo >> 5] | ((ulong)fp[c][(bo >> 5) + 1] << 32);
            b[c] = ok[c] ? decB(w[c], (char)(((fw >> (bo & 31)) & 7) + 1)) : (int8)(0);
        }
        ATILE a[SR]; float8 sa[SR];
        UN for (int r = 0; r < SR; r++) {
            a[r] = as_int8(intel_sub_group_block_read8(A + ((size_t)(rt0 + r) * KB + kq) * 64));
            sa[r] = vload8(0, SAt + (size_t)kq * Mp + (rt0 + r) * 8);
        }
        UN for (int r = 0; r < SR; r++) UN for (int c = 0; c < SNT; c++)
            acc[r][c] += convert_float8(MADI8(a[r], b[c], (int8)(0))) * sa[r];
    }
    UN for (int r = 0; r < SR; r++) UN for (int c = 0; c < SNT; c++) UN for (int m = 0; m < 8; m++) {
        int row = (rt0 + r) * 8 + m, col = (ct0 + c) * 8 + l;
        if (valid && row < M && col < N) {
            float v = acc[r][c][m] * vload_half(col, rs) * 0.125f;
            size_t o = (size_t)row * ldy + col; Y[o] = add ? Y[o] + v : v;
        }
    }
}
"""


def _x8src():
    """W4A8 XMX: takes pgemm8/pgemm8t from the Qwen prefill, rounds the factor-row length up (K = 704, 2112, 2816)
    and adds pgemm8e, which reads the expert id per row tile."""
    from ..qwen35 import prefill as QP
    s = QP.GSRC.replace("(K / G) * 3 / 32", "(((K / G) * 3 + 31) / 32)")
    a = s.index("__attribute__((intel_reqd_sub_group_size(8)))\n__kernel void pgemm8(")
    b = s.index("// TILE-SCRATCH")
    i8 = s[a:b].replace("__kernel void pgemm8(", "__kernel void pgemm8i(", 1)
    d0 = i8.index("uint4 w = vload4(0, codes + (size_t)n * (K / 8) + kq * 4);")
    d1 = i8.index("o = (uint8)(il0(l0, h0)"); d1 = i8.index("\\", d1)
    i8 = i8[:d0] + "o = vload8(0, codes + ((size_t)n * K + kq * 32) / 4);   /* INT8 row bytes = DPAS B operand */ " + i8[d1:]
    # pgemm8i: INT8 weights read row-major straight into the B operand (no tileI8 pass); codes = q, rs = s*8 (FP16)
    e = s[a:b].replace("__kernel void pgemm8(", "__kernel void pgemm8e(", 1)
    e = e.replace("__global float* Y, int M, int N, int K, int ldy, int add) {", "__global float* Y, int M, int N, int K, int ldy, int add, __global const int* tileE) {", 1)
    anchor = "int KB = K / 32, FW = (((K / G) * 3 + 31) / 32);"
    assert anchor in e
    e = e.replace(anchor, anchor + "\n    int eid = tileE[get_group_id(1)]; int valid = eid >= 0; eid = max(eid, 0);"
                                    "\n    codes += (size_t)eid * N * (K / 8); fac += (size_t)eid * N * FW; rs += (size_t)eid * N;", 1)
    assert e.count("if (row < M && col < N) {") == 1
    e = e.replace("if (row < M && col < N) {", "if (valid && row < M && col < N) {")
    extra = r"""
#define UNR __attribute__((opencl_unroll_hint))
__kernel void quantA8m(__global const float* X, int ldx, __global const int* map, int usemap, int Mreal, __global uint* A8, __global float* SAt, int K, int Mp) {
    int b = get_global_id(0), t = get_global_id(1), KB = K / 32;
    int src = usemap ? map[t] : t;
    __global uint* o = A8 + ((size_t)(t / 8) * KB + b) * 64 + (t % 8) * 8;
    if (src < 0 || src >= Mreal) { UNR for (int i = 0; i < 8; i++) o[i] = 0u; SAt[(size_t)b * Mp + t] = 0.0f; return; }
    __global const float* x = X + (size_t)src * ldx + b * 32;
    float mx = 0.0f;
    UNR for (int i = 0; i < 8; i++) { float4 v = fabs(vload4(i, x)); mx = fmax(mx, fmax(fmax(v.x, v.y), fmax(v.z, v.w))); }
    float s = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / s;
    UNR for (int i = 0; i < 8; i++) o[i] = as_uint(convert_char4_sat_rte(vload4(i, x) * inv));
    SAt[(size_t)b * Mp + t] = s;
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void tileI8(__global const char* q, __global uint* Bt, int N, int K) {
    int l = get_sub_group_local_id(), kq = get_global_id(0) / 8, nt = get_global_id(1), n = nt * 8 + l, KB = K / 32;
    if (kq >= KB) return;
    uint8 o = (uint8)(0);
    if (n < N) o = vload8(0, (__global const uint*)(q + (size_t)n * K + kq * 32));
    intel_sub_group_block_write8(Bt + ((size_t)nt * KB + kq) * 64, o);
}
__kernel void buildl2(__global const int* idx, int A, int E, int TK, int TRE, __global int* prow, __global int* rmap, __global int* tileE, int maxrows) {
    __local int cnt[128], off[128], cur[128];
    int l = get_local_id(0);
    if (l < E) { cnt[l] = 0; cur[l] = 0; }
    for (int i = l; i < maxrows; i += 1024) rmap[i] = -1;
    for (int i = l; i < maxrows / TRE; i += 1024) tileE[i] = -1;
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    for (int a = l; a < A; a += 1024) atomic_inc(&cnt[idx[a]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) { int s = 0; for (int e = 0; e < E; e++) { off[e] = s; s += (cnt[e] + TRE - 1) / TRE * TRE; } }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int a = l; a < A; a += 1024) { int e = idx[a], p = atomic_inc(&cur[e]), r = off[e] + p; prow[a] = r; rmap[r] = a / TK; }
    if (l < E) for (int k = 0; k * TRE < cnt[l]; k++) tileE[off[l] / TRE + k] = l;
}
__kernel void gq8(__global const float* GU, int F, int Mreal, __global uint* A8, __global float* SAt, int Mp) {
    // geglu + quantA8m (no row map) in one pass: same values as gelu(g) * u written out and re-read.
    int b = get_global_id(0), t = get_global_id(1), KB = F / 32;
    __global uint* o = A8 + ((size_t)(t / 8) * KB + b) * 64 + (t % 8) * 8;
    if (t >= Mreal) { UNR for (int i = 0; i < 8; i++) o[i] = 0u; SAt[(size_t)b * Mp + t] = 0.0f; return; }
    __global const float* g = GU + (size_t)t * 2 * F + b * 32;
    float v[32]; float mx = 0.0f;
    UNR for (int i = 0; i < 32; i++) { v[i] = gelu(g[i]) * g[F + i]; mx = fmax(mx, fabs(v[i])); }
    float s = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / s;
    UNR for (int i = 0; i < 8; i++) o[i] = as_uint(convert_char4_sat_rte((float4)(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]) * inv));
    SAt[(size_t)b * Mp + t] = s;
}
__kernel void ecomb2(__global const float* eo, __global const int* prow, __global const float* wt, __global float* e, int H, int TK) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    float s = 0.0f;
    for (int j = 0; j < TK; j++) s += wt[t * TK + j] * eo[(size_t)prow[t * TK + j] * H + i];
    e[(size_t)t * H + i] = s;
}
"""
    return s + e + i8, extra


class Prefill:
    def __init__(self, m: GR.Model):
        self.m = m; self.T = m.PCH; self.TB = int(knob("G4_TB", "8"))
        self.prg = cl.Program(m.ctx, GR.GSRC + PSRC).build(options=[f"-DG={m.meta['G']}", "-DR=1", f"-DTB={self.TB}"])
        self.a8 = int(knob("G4_A8", "1"))
        self.a8d, self.a8e = int(knob("G4_A8D", "1")), int(knob("G4_A8E", "1"))
        ns_ = "1" if GR.K2.nosplit(m.q.device) else "0"                 # skinny expert kernel: GPUs without split DPAS only
        self.fuse = int(knob("G4_FUSE", "1"))
        self.i8direct = int(knob("G4_I8DIRECT", "1"))
        if self.a8:
            from ..qwen35.prefill import GCFG
            X8, XAUX = _x8src(); ctx = m.ctx; base = [f"-DG={m.meta['G']}", "-DXE8=1", "-DNODEC=0", "-DIB=1", "-cl-intel-256-GRF-per-thread"] + GR.K2.nosplit(m.q.device)
            from ..qwen35.prefill import _slmbw
            base += _slmbw()
            def cfg(env, dflt):
                c = dict(GCFG); c.update(zip(("RM", "NT", "WM", "WN", "UK"), (int(x) for x in knob(env, dflt).split(",")))); return c
            self.cd, self.ce = cfg("G4_P8D", "4,4,8,4,2"), cfg("G4_P8E", "2,4,4,4,2")
            self.pd = cl.Program(ctx, X8).build(options=[f"-D{k}={v}" for k, v in self.cd.items()] + base)
            ex = []
            self.skinny = int(knob("G4_SKINNY", "1" if self.T <= 1024 and ns_ == "1" else "0"))
            self.SR, self.SNT, self.SWN = (int(x) for x in knob("G4_SK", "4,2,8").split(","))
            sko = [f"-DSR={self.SR}", f"-DSNT={self.SNT}", f"-DSWN={self.SWN}"]
            self.pe = cl.Program(ctx, X8 + (SKSRC if self.skinny else "")).build(options=[f"-D{k}={v}" for k, v in self.ce.items()] + base + ex + sko)
            self.pa = cl.Program(ctx, GR.GSRC + XAUX).build(options=[f"-DG={m.meta['G']}", "-DR=1"])
            for nm, pr, kn in (("dense", self.pd, "pgemm8"), ("dense", self.pd, "pgemm8t"), ("expert", self.pe, "pgemm8e")):
                zb = bytes(pr.get_info(cl.program_info.BINARIES)[0]); i = zb.find(("name:            " + kn + chr(10)).encode())
                j = zb.find(b"- name:", i + 10); seg = zb[i:j if j > 0 else len(zb)]
                if b"require_disable_eufusion: true" in seg:
                    print(f"warning: {nm} {kn} was built with EU fusion disabled; split DPAS results may be wrong", flush=True)
            self.TRD = 8 * self.cd["RM"] * self.cd["WM"]; self.TCD = 8 * self.cd["NT"] * self.cd["WN"]
            self.TRE = 8 * self.ce["RM"] * self.ce["WM"]; self.TCE = 8 * self.ce["NT"] * self.ce["WN"]
            if self.skinny:
                self.TRE, self.TCE = 8 * self.SR, 8 * self.SNT * self.SWN
            self.Tp = -(-self.T // self.TRD) * self.TRD
            self.MPE = -(-(self.T * m.TK + m.NE * (self.TRE - 1)) // self.TRE) * self.TRE
            mf = cl.mem_flags; i32 = lambda n: cl.Buffer(ctx, mf.READ_WRITE, max(4, int(n) * 4))
            Kmax = max(max(t["shape"][1] * 2 for k, t in m.tm.items() if k.endswith(".codes")), max([t["shape"][1] for k, t in m.tm.items() if k.endswith(".q")] or [0]))
            self.A8 = cl.Buffer(ctx, mf.READ_WRITE, max(self.Tp, self.MPE) * Kmax)
            self.SA = i32(max(self.Tp, self.MPE) * Kmax // 32)
            self.PROW, self.RMAP, self.TLE = i32(self.T * m.TK), i32(self.MPE), i32(self.MPE // self.TRE)
            self.EGUp, self.EMp, self.EOp = i32(self.MPE * 2 * m.FE), i32(1 if self.fuse else self.MPE * m.FE), i32(self.MPE * m.H)   # EMp only on the unfused path
            i8 = [k[:-2] for k in m.tm if k.endswith(".q") and k != "lm_head.q"]
            if i8:
                self.BT = cl.Buffer(ctx, mf.READ_WRITE, max(-(-m.tm[k + ".q"]["shape"][0] // 8) * 8 * m.tm[k + ".q"]["shape"][1] for k in i8))
                with open(m.path, "rb") as fh:
                    for k in i8:
                        t = m.tm[k + ".s"]; fh.seek(t["off"]); s = np.frombuffer(fh.read(t["nbytes"]), np.float32)
                        m.buf[k + ".rs8"] = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=(s * 8).astype(np.float16))
        mf = cl.mem_flags; T, H, NH, F, TK, FE = self.T, m.H, m.NH, m.F, m.TK, m.FE
        f32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        self.X, self.TN, self.T2, self.T3, self.O, self.D1, self.E = (f32(T * H) for _ in range(7))
        self.ldP = max(NH * L["HD"] + (1 if L["kv"] else 2) * L["KH"] * L["HD"] for L in m.lcfg)
        self.P = f32(T * self.ldP); self.Qb, self.Ab = f32(T * NH * 512), f32(T * NH * 512)
        self.GU, self.M = f32(T * 2 * F), f32(T * F)
        self.A = T * TK; self.maxt = self.A // self.TB + m.NE
        self.RL, self.IDX, self.WT = f32(T * m.NE), f32(self.A), f32(self.A)
        self.LT, self.LA, self.TL = f32(self.A), f32(self.A), f32(self.maxt * 4)
        fb = not (self.a8 and self.a8e)                                    # float expert path buffers only when it is used
        self.EGU, self.EM, self.EO = (f32(self.A * 2 * FE), f32(self.A * FE), f32(self.A * H)) if fb else (f32(1), f32(1), f32(1))
        self.LR = 64; self.LOG = f32(self.LR * m.V)
        self.amm = int(knob("G4_AMM", "1"))
        self.amm_min = int(knob("G4_AMM_MIN", "512"))
        self.aband = int(knob("G4_ABAND", "1"))                 # band-limited key range per row block on sliding layers
        if self.amm:
            from ..qwen35 import prefill as QP
            gc = dict(QP.GCFG); gc.update(WM=8, UK=2)
            # 2: one pass (row offset, unnormalized BF16 P, exact fallback on overflow); 1: exact two passes; 0: score matrix + softPg
            self.att2 = int(knob("G4_ATT2", "0"))   # off by default until checked against BF16 on translation / writing text
            self.att2_min = int(knob("G4_ATT2_MIN", "2048"))   # key range from which the new path is used (smaller S stays in cache)
            from ..qwen35.attn2_src import ATT2_FAST, ATT2_EXACT, ATT2_GSRC
            gopts = [f"-D{k}={v}" for k, v in gc.items()] + [f"-DG={m.meta['G']}", "-DXE8=1", "-DNODEC=0", "-DIB=1", "-cl-intel-256-GRF-per-thread"] + QP._nosplit(m.q.device) + QP._slmbw()
            self.gpm = cl.Program(m.ctx, QP.GSRC + (ATT2_FAST if self.att2 == 2 else "")).build(options=gopts)
            self.gpx = cl.Program(m.ctx, QP.GSRC + ATT2_EXACT).build(options=gopts) if self.att2 else None
            zb = bytes(self.gpm.get_info(cl.program_info.BINARIES)[0]); i = zb.find(("name:            pgemm" + chr(10)).encode()); j = zb.find(b"- name:", i + 10)
            self.amm_nofuse = b"require_disable_eufusion: true" in zb[i:j if j > 0 else len(zb)]
            if self.amm_nofuse:
                print("warning: attention pgemm was built with EU fusion disabled", flush=True)
            self.pam = cl.Program(m.ctx, GR.KVM + AMSRC + (ATT2_GSRC if self.att2 else "")).build(
                options=[f"-DOFFADJ={knob('G4_OFFADJ')}.0f"] if knob("G4_OFFADJ") else [])
            self.ATR, self.ATC, self.ans = 8 * gc["RM"] * gc["WM"], 8 * gc["NT"] * gc["WN"], gc["WM"] * gc["WN"]
            self.RB = 1024; self.LPM = -(-(m.TM + self.T) // 128) * 128
            bb = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, int(n))
            self.aq, self.bk, self.bv = bb(self.RB * 512 * 2), bb(self.LPM * 512 * 2), bb(self.LPM * 512 * 2)
            self.chk = knob("G4_CHK") == "1"
            self.Sm, self.ap, self.ot = bb(max(4, self.RB * (min(self.LPM, self.att2_min) if self.att2 and not self.chk else self.LPM) * 4)), bb(self.RB * self.LPM * 2), bb(self.RB * 512 * 4)
            if self.att2:
                self.aMX, self.aINV, self.aGR, self.aOFF = bb(self.RB * 4), bb(self.RB * 4), bb(self.RB * 4), bb(self.RB * 4)
                self.aKN, self.aPS, self.aFLAG, self.aONE = bb(self.LPM * 4), bb(self.RB * self.LPM // 32 * 4), bb(4), bb(4)
                cl.enqueue_fill_buffer(m.q, self.aONE, np.int32(1), 0, 4)
        self.NP = m.TM + self.T
        self.SMAP = cl.Buffer(m.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.full(self.NP, -1, np.int32))
        self.UB = cl.Buffer(m.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(self.NP, np.int32))
        self.SOFT = f32(1); self.spans = []
        self.cache = {}

    def set_images(self, spans, soft=None):
        """Image token spans [(b0, b1)] and projected rows [sum(b1-b0), H] in span order. Positions inside a span take the projected row
        instead of the embedding and attend bidirectionally within the span on sliding layers. spans=[] restores text-only."""
        m = self.m; smap = np.full(self.NP, -1, np.int32); ub = np.zeros(self.NP, np.int32); r = 0
        for b0, b1 in spans:
            smap[b0:b1] = np.arange(r, r + b1 - b0); ub[b0:b1] = b1; r += b1 - b0
        cl.enqueue_copy(m.q, self.SMAP, smap); cl.enqueue_copy(m.q, self.UB, ub)
        if soft is not None:
            assert isinstance(soft, cl.Buffer) or soft.shape == (r, m.H), (soft.shape, r)
            self.SOFT = soft if isinstance(soft, cl.Buffer) else cl.Buffer(m.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(soft, np.float32))
        self.spans = list(spans); m.q.finish()

    def K(self, name, g, l, *args):
        k = cl.Kernel(self.prg, name); k.set_args(*args); return (k, g, l)

    def gemm8(self, name, x, ldx, y, ldy, Tn, add=0):
        """W4A8 dense: INT8 activations -> pgemm8 (XeFP4) or tileI8 + pgemm8t (INT8 weights). Returns a kernel list."""
        m, I = self.m, np.int32; Tp = -(-Tn // self.TRD) * self.TRD; ns = self.cd["WM"] * self.cd["WN"]
        i8 = name + ".q" in m.tm
        N, Kd = (m.tm[name + ".q"]["shape"] if i8 else (m.tm[name + ".codes"]["shape"][0], m.tm[name + ".codes"]["shape"][1] * 2))
        qa = (cl.Kernel(self.pa, "quantA8m"), (Kd // 32, Tp), None)
        qa[0].set_args(x, I(ldx), self.PROW, I(0), I(Tn), self.A8, self.SA, I(Kd), I(Tp))
        g = (-(-N // self.TCD) * ns * 8, Tp // self.TRD); lsz = (ns * 8, 1)
        if i8:
            if self.i8direct:
                k = cl.Kernel(self.pd, "pgemm8i"); k.set_args(self.A8, self.SA, I(Tp), m.buf[name + ".q"], m.buf[name + ".q"], m.buf[name + ".rs8"], y, I(Tn), I(N), I(Kd), I(ldy), I(add))
                return [qa, (k, g, lsz)]
            tb = (cl.Kernel(self.pa, "tileI8"), (-(-(Kd // 32 * 8) // 64) * 64, -(-N // 8)), (64, 1)); tb[0].set_args(m.buf[name + ".q"], self.BT, I(N), I(Kd))
            k = cl.Kernel(self.pd, "pgemm8t"); k.set_args(self.A8, self.SA, I(Tp), self.BT, m.buf[name + ".rs8"], y, I(Tn), I(N), I(Kd), I(ldy), I(add))
            return [qa, tb, (k, g, lsz)]
        k = cl.Kernel(self.pd, "pgemm8"); k.set_args(self.A8, self.SA, I(Tp), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], y, I(Tn), I(N), I(Kd), I(ldy), I(add))
        return [qa, (k, g, lsz)]

    def experts8(self, p, Tn):
        """W4A8 experts: gather -> gate/up (pgemm8e) -> gelu product -> down (pgemm8e) -> weighted scatter back to tokens."""
        m, I = self.m, np.int32; H, FE, TK, A = m.H, m.FE, m.TK, Tn * m.TK; ns = self.ce["WM"] * self.ce["WN"]; b = m.buf
        out = [self.K2(self.pa, "buildl2", (1024,), (1024,), self.IDX, I(A), I(m.NE), I(TK), I(self.TRE), self.PROW, self.RMAP, self.TLE, I(self.MPE)),
               self.K2(self.pa, "quantA8m", (H // 32, self.MPE), None, self.T3, I(H), self.RMAP, I(1), I(Tn), self.A8, self.SA, I(H), I(self.MPE))]
        def ek(w, Y, N, Kd):
            if self.skinny:
                k = cl.Kernel(self.pe, "pgemm8s"); k.set_args(self.A8, self.SA, I(self.MPE), b[p + w + ".codes"], b[p + w + ".fac"], b[p + w + ".rs"], Y,
                                                              I(self.MPE), I(N), I(Kd), I(N), I(0), self.TLE, I(1))
                return (k, (-(-N // self.TCE) * self.SWN * 8, self.MPE // self.TRE), (self.SWN * 8, 1))
            k = cl.Kernel(self.pe, "pgemm8e"); k.set_args(self.A8, self.SA, I(self.MPE), b[p + w + ".codes"], b[p + w + ".fac"], b[p + w + ".rs"], Y,
                                                          I(self.MPE), I(N), I(Kd), I(N), I(0), self.TLE)
            return (k, (-(-N // self.TCE) * ns * 8, self.MPE // self.TRE), (ns * 8, 1))
        out.append(ek("egu", self.EGUp, 2 * FE, H))
        if self.fuse:
            out += [self.K2(self.pa, "gq8", (FE // 32, self.MPE), None, self.EGUp, I(FE), I(self.MPE), self.A8, self.SA, I(self.MPE))]
        else:
            out += [self.K("geglu", (-(-FE * self.MPE // 256) * 256,), (256,), self.EGUp, self.EMp, I(FE), I(self.MPE)),
                    self.K2(self.pa, "quantA8m", (FE // 32, self.MPE), None, self.EMp, I(FE), self.RMAP, I(0), I(self.MPE), self.A8, self.SA, I(FE), I(self.MPE))]
        out.append(ek("edown", self.EOp, H, FE))
        out.append(self.K2(self.pa, "ecomb2", (-(-H // 256) * 256, Tn), (256, 1), self.EOp, self.PROW, self.WT, self.E, I(H), I(TK)))
        return out

    def K2(self, prg, name, g, l, *args):
        k = cl.Kernel(prg, name); k.set_args(*args); return (k, g, l)

    def gemm(self, name, x, ldx, y, ldy, Tn, add=0):
        m, I = self.m, np.int32
        if name + ".q" in m.tm:
            N, Kd = m.tm[name + ".q"]["shape"]
            return self.K("igemmT", (16, -(-N // 8) * 8, -(-Tn // self.TB)), (16, 8, 1), m.buf[name + ".q"], m.buf[name + ".s"], x, I(ldx), y, I(ldy), I(N), I(Kd), I(add), I(Tn))
        t = m.tm[name + ".codes"]; N, Kd = t["shape"][0], t["shape"][1] * 2
        return self.K("xgemmT", (16, -(-N // 8) * 8, -(-Tn // self.TB)), (16, 8, 1), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, I(ldx), y, I(ldy),
                      I(N), I(Kd), I(add), I(0), I(Tn), self.TL, self.LT, self.LA)

    def egemm(self, nm, x, ldx, y, ldy, NR, Kd, xidx):
        m, I = self.m, np.int32
        return self.K("xgemmT", (16, -(-NR // 8) * 8, self.maxt), (16, 8, 1), m.buf[nm + ".codes"], m.buf[nm + ".fac"], m.buf[nm + ".rs"], x, I(ldx), y, I(ldy),
                      I(NR), I(Kd), I(0), I(1), I(0), self.TL, xidx, self.LA)

    def kernels(self, Tn):
        """Kernel list for chunk length Tn (pos0 is set at run time). Returns (list, kernels and argument indices taking pos0)."""
        if Tn in self.cache:
            return self.cache[Tn]
        m, I, H, NH, F, TK, FE = self.m, np.int32, self.m.H, self.m.NH, self.m.F, self.m.TK, self.m.FE
        eps = np.float32(m.eps); A = Tn * TK; b = m.buf; ks, posk = [], []
        u = int(np.array([np.sqrt(H)], np.float32).view(np.uint32)[0]); nrm = float(np.array([(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000], np.uint32).view(np.float32)[0])
        ke = self.K("embedT", (-(-H // 256) * 256, Tn), (256, 1), b["lm_head.q"], b["lm_head.s"], m.ids, I(0), self.X, I(H), np.float32(nrm), self.SMAP, self.SOFT)
        ks += [ke, self.K("gnormT", (1024, Tn), (1024, 1), self.X, b["layers.0.in_norm"], self.TN, I(H), eps)]
        RW = lambda n: (-(-n // 256) * 256, Tn)
        for i, L in enumerate(m.lcfg):
            p = f"layers.{i}."; kc, vc = m.kv[i]; KH, HD = L["KH"], L["HD"]
            voff = NH * HD if L["kv"] else NH * HD + KH * HD
            kp = self.K("gprepT", ((NH + KH) * 256, Tn), (256, 1), self.P, I(self.ldP), b[p + "q_norm"], b[p + "k_norm"], self.Qb, kc, vc,
                        I(0), np.float32(L["theta"]), I(NH), I(KH), I(HD), I(L["NR"]), I(voff), I(L["CAP"]), eps)
            ka = self.K("gattnT", (NH * 128, -(-Tn // 8)), (128, 1), self.Qb, kc, vc, self.Ab, I(0), I(Tn), I(L["CAP"]), I(KH), I(NH // KH), I(HD), I(L["WIN"]), self.UB, I(0 if L["kv"] else 1))
            if self.amm:
                ka = ("ATTN", i, ka); posk += [(kp, 7)]
            else:
                posk += [(kp, 7), (ka, 4)]
            nxt = b[f"layers.{i + 1}.in_norm"] if i + 1 < len(m.lcfg) else b["norm"]
            GM = (lambda *a: self.gemm8(*a)) if (self.a8 and self.a8d) else (lambda *a: [self.gemm(*a)])
            ks += GM(p + "proj", self.TN, H, self.P, self.ldP, Tn) + [kp, ka] + GM(p + "out", self.Ab, NH * HD, self.O, H, Tn)
            ks += [self.K("postattnT", (1024, Tn), (1024, 1), self.O, b[p + "post_attn"], self.X, b[p + "pre_ff"], b[p + "pre_ff2"], self.TN, self.T3, self.T2, I(H), eps)]
            if self.a8 and self.a8d and self.fuse:                                   # gemm8 list = [quantA8m, (tileI8,) gemm]
                Tpd = -(-Tn // self.TRD) * self.TRD
                ks += GM(p + "gu", self.TN, H, self.GU, 2 * F, Tn) + [self.K2(self.pa, "gq8", (F // 32, Tpd), None, self.GU, I(F), I(Tn), self.A8, self.SA, I(Tpd))]
                ks += GM(p + "down", self.M, F, self.D1, H, Tn)[1:]          # activations already quantized by gq8
            else:
                ks += GM(p + "gu", self.TN, H, self.GU, 2 * F, Tn) + [self.K("geglu", (-(-F * Tn // 256) * 256,), (256,), self.GU, self.M, I(F), I(Tn))]
                ks += GM(p + "down", self.M, F, self.D1, H, Tn)
            r2 = knob("G4_ROUTER2", "1") == "1" and H <= 3072 and m.NE % 16 == 0
            ks += [self.K("routerT2", (m.NE // 16 * 128, Tn), (128, 1), b[p + "r_projF"], self.T2, self.RL, I(m.NE), I(H)) if r2 else
                   self.K("routerT", (m.NE * 128, Tn), (128, 1), b[p + "r_projF"], self.T2, self.RL, I(m.NE), I(H)),
                   self.K("topkT", (128, Tn), (128, 1), self.RL, b[p + "r_pes"], self.IDX, self.WT, I(m.NE), I(TK))]
            if self.a8 and self.a8e:
                ks += self.experts8(p, Tn)
            else:
                ks += [self.K("buildl", (1024,), (1024,), self.IDX, I(A), I(m.NE), I(TK), self.LT, self.LA, self.TL, I(self.maxt)),
                       self.egemm(p + "egu", self.T3, H, self.EGU, 2 * FE, 2 * FE, H, self.LT),
                       self.K("geglu", (-(-FE * A // 256) * 256,), (256,), self.EGU, self.EM, I(FE), I(A)),
                       self.egemm(p + "edown", self.EM, FE, self.EO, H, H, FE, self.LA),
                       self.K("ecomb", RW(H), (256, 1), self.EO, self.WT, self.E, I(H), I(TK))]
            ks += [self.K("ffend2T", (1024, Tn), (1024, 1), self.D1, self.E, b[p + "post_ff1"], b[p + "post_ff2"], b[p + "post_ff"], b[p + "ls"], self.X, nxt, self.TN, I(H), eps)]
        self.cache[Tn] = (ks, posk, ke)
        return self.cache[Tn]

    def attn_mm(self, i, Tn, pos0, q=None, evs=None):
        """Layer i attention as two matrix products per KV head. Key range [jlo, jhi) covers every position this chunk attends to."""
        m, I = self.m, np.int32; L = m.lcfg[i]; kc, vc = m.kv[i]; KH, HD, NH = L["KH"], L["HD"], m.NH; GQ = NH // KH
        q = q or m.q; jhi = pos0 + Tn; jlo = max(0, pos0 + 1 - L["WIN"]); Lp = -(-(jhi - jlo) // 128) * 128; R = Tn * GQ
        RB = min(self.RB, -(-R // self.ATR) * self.ATR); ns = self.ans
        if not hasattr(self, "akc"):
            names = ("packKg", "packVg", "packQg", "softPg", "writeOg") + (("knormg", "rowoffg", "rowsumg", "writeOg2") if self.att2 else ())
            self.akc = {n: cl.Kernel(self.pam, n) for n in names}; self.akg = cl.Kernel(self.gpm, "pgemm")
            if self.att2:
                self.akc.update({n: cl.Kernel(self.gpx, n) for n in ("qkstat", "qkp")})
                if self.att2 == 2:
                    self.akc["qke"] = cl.Kernel(self.gpm, "qke")
        kg = self.akg

        def go(name, g, l, *args):
            k = self.akc[name]; k.set_args(*args); e = cl.enqueue_nd_range_kernel(q, k, g, l)
            if evs is not None:
                evs.append(("attn." + name, e))
        # Band: on sliding layers a row block only needs keys [first position - WIN + 1, last position], so K/V are packed
        # and multiplied per row block over that range instead of the whole chunk range (same keys per row, less work).
        # Not with image spans (bidirectional rows can reach past the band).
        band = self.aband and L["WIN"] < jhi and not self.spans
        for kh in range(KH):
            if not band:
                go("packKg", (Lp, HD // 2), None, kc, self.bk, I(KH), I(kh), I(HD), I(jlo), I(jhi), I(L["CAP"]))
                go("packVg", (HD, Lp // 2), None, vc, self.bv, I(KH), I(kh), I(HD), I(jlo), I(jhi), I(L["CAP"]), I(Lp // 16))
            for rb0 in range(0, R, RB):
                jl, Lq, jh = jlo, Lp, jhi
                if band:
                    t_first, t_last = pos0 + rb0 // GQ, pos0 + (min(R, rb0 + RB) - 1) // GQ
                    jl = max(jlo, t_first + 1 - L["WIN"]); jh = min(jhi, t_last + 1)
                    jl -= jl % 128                                   # absolute 128 grid: the same key blocks whatever the chunking (masked keys add 0)
                    Lq = -(-(jh - jl) // 128) * 128
                    go("packKg", (Lq, HD // 2), None, kc, self.bk, I(KH), I(kh), I(HD), I(jl), I(jh), I(L["CAP"]))
                    go("packVg", (HD, Lq // 2), None, vc, self.bv, I(KH), I(kh), I(HD), I(jl), I(jh), I(L["CAP"]), I(Lq // 16))
                go("packQg", (HD // 2, RB), None, self.Qb, self.aq, I(NH), I(kh), I(GQ), I(HD), I(rb0), I(R))
                bid = I(0 if L["kv"] else 1)
                new = self.att2 and Lq >= self.att2_min
                if new:
                    ra = (I(R), I(pos0), I(rb0), I(GQ), I(jl), I(L["WIN"]), self.UB, bid if self.spans else I(0), np.float32(1.0))   # UB is all zero without images
                    gate = self.aONE
                    if self.att2 == 2:
                        go("knormg", (-(-Lq // 256) * 256,), (256,), kc, I(KH), I(kh), I(HD), I(jl), I(jh), I(L["CAP"]), I(Lq), self.aKN)
                        go("rowoffg", (RB,), (256,), self.Qb, kc, self.aKN, I(NH), I(KH), I(kh), I(GQ), I(HD), I(L["CAP"]), I(Lq), I(pos0), I(rb0), I(R), self.aOFF)
                        go("qke", (Lq // self.ATC * ns * 8, RB // self.ATR), (ns * 8, 1), self.aq, self.bk, I(Lq), I(HD), *ra, self.aOFF, self.ap, self.aPS)
                        cl.enqueue_fill_buffer(q, self.aFLAG, np.int32(0), 0, 4)
                        go("rowsumg", (RB,), (256,), self.aPS, I(Lq // 32), I(rb0), I(R), self.aGR, self.aFLAG)
                        gate = self.aFLAG
                    go("qkstat", (ns * 8, RB // self.ATR), (ns * 8, 1), self.aq, self.bk, I(Lq), I(HD), *ra, self.aMX, self.aINV, gate, self.aGR)
                    go("qkp", (Lq // self.ATC * ns * 8, RB // self.ATR), (ns * 8, 1), self.aq, self.bk, I(Lq), I(HD), *ra, self.aMX, self.aINV, self.ap, gate)
                    if self.chk:
                        self._chk(q, kg, i, kh, rb0, RB, Lq, HD, pos0, R, GQ, jl, L, bid, ns)
                else:
                    kg.set_args(self.aq, self.bk, self.Sm, I(RB), I(Lq), I(HD), I(Lq), I(0))
                    e = cl.enqueue_nd_range_kernel(q, kg, (Lq // self.ATC * ns * 8, RB // self.ATR), (ns * 8, 1))
                    if evs is not None:
                        evs.append(("attn.QK", e))
                    go("softPg", (RB * 256,), (256,), self.Sm, I(Lq), self.ap, I(pos0), I(rb0), I(R), I(GQ), I(jl), I(L["WIN"]), self.UB, bid)
                kg.set_args(self.ap, self.bv, self.ot, I(RB), I(HD), I(Lq), I(HD), I(0))
                e = cl.enqueue_nd_range_kernel(q, kg, (HD // self.ATC * ns * 8, RB // self.ATR), (ns * 8, 1))
                if evs is not None:
                    evs.append(("attn.PV", e))
                if new:
                    go("writeOg2", (HD, RB), None, self.ot, self.aGR, self.Ab, I(NH), I(kh), I(GQ), I(HD), I(rb0), I(R))
                else:
                    go("writeOg", (HD, RB), None, self.ot, self.Ab, I(NH), I(kh), I(GQ), I(HD), I(rb0), I(R))

    def _chk(self, q, kg, i, kh, rb0, RB, Lq, HD, pos0, R, GQ, jl, L, bid, ns):
        I = np.int32; n = RB * Lq
        pn = np.empty(n, np.uint16); cl.enqueue_copy(q, pn, self.ap); gr = np.empty(RB, np.float32); cl.enqueue_copy(q, gr, self.aGR)
        kg.set_args(self.aq, self.bk, self.Sm, I(RB), I(Lq), I(HD), I(Lq), I(0))
        cl.enqueue_nd_range_kernel(q, kg, (Lq // self.ATC * ns * 8, RB // self.ATR), (ns * 8, 1))
        k = self.akc["softPg"]; k.set_args(self.Sm, I(Lq), self.ap, I(pos0), I(rb0), I(R), I(GQ), I(jl), I(L["WIN"]), self.UB, bid)
        cl.enqueue_nd_range_kernel(q, k, (RB * 256,), (256,))
        po = np.empty(n, np.uint16); cl.enqueue_copy(q, po, self.ap); q.finish()
        f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
        # unpack: block (r/8, c/16) of 64 uints = 128 halves; uint index (r%8)*8 + c2%8, c2 = c/2
        def unpack(u):
            x = f(u).reshape(RB // 8, Lq // 16, 8, 8, 2)                     # [rb][kb][r%8][c2%8][pair]
            return x.transpose(0, 2, 1, 3, 4).reshape(RB, Lq)
        a, b = unpack(pn), unpack(po)
        if self.att2 == 2:
            a = a * gr[:, None]
        d = np.abs(a - b); rows = min(RB, R - rb0)
        nzo, nzn = (b[:rows] != 0).sum(), (a[:rows] != 0).sum()
        print(f"chk layer {i} kh {kh} rb0 {rb0} Lq {Lq} jl {jl} WIN {L['WIN']}: max|d| {d[:rows].max():.3e} rel {(d[:rows].max() / max(b[:rows].max(), 1e-30)):.3e} "
              f"nonzero old {nzo} new {nzn} rowsum old {b[:rows].sum(1).min():.4f}..{b[:rows].sum(1).max():.4f} new {a[:rows].sum(1).min():.4f}..{a[:rows].sum(1).max():.4f}", flush=True)

    def chunk(self, i0, Tn, pos0, q=None, evs=None):
        """Feeds ids[i0 : i0+Tn] at positions pos0.. . Afterwards TN holds the final normed hidden states [Tn, H]."""
        ks, posk, ke = self.kernels(Tn); q = q or self.m.q
        ke[0].set_arg(3, np.int32(i0)); ke[0].set_arg(8, self.SOFT)
        for k, a in posk:
            k[0].set_arg(a, np.int32(pos0))
        for k, g, l in ks:
            if isinstance(k, str):
                if pos0 + Tn > self.amm_min:
                    self.attn_mm(g, Tn, pos0, q, evs)
                    continue
                k, g, l = l; k.set_arg(4, np.int32(pos0))
            e = cl.enqueue_nd_range_kernel(q, k, g, l)
            if evs is not None:
                evs.append((k.function_name, e))
        q.flush()

    def prompt(self, N, start=0):
        """Feeds m.ids[start:N] in chunks (KV before start is assumed present), then the last logits and greedy next token (outs[0], tok). Next position = N."""
        m = self.m; c0 = start
        while c0 < N:
            c1 = min(c0 + self.T, N)
            for b0, b1 in self.spans:
                if b0 < c1 < b1 and b0 > c0:
                    c1 = b0
            self.chunk(c0, c1 - c0, c0); Tn = c1 - c0; c0 = c1
        cl.enqueue_copy(m.q, m.t, self.TN, src_offset=(Tn - 1) * m.H * 4, byte_count=m.H * 4)
        m.run(m.tail); m.am2[0].set_arg(5, np.int32(0)); m.run([m.am1, m.am2])

    def nll_rows(self, Tn, tgt0, out0):
        """All Tn rows of TN through lm_head -> NLL and top-1 against ids[tgt0 + row] -> m.NLL[out0 + row]."""
        m, I = self.m, np.int32; V, H = m.V, m.H
        for r0 in range(0, Tn, self.LR):
            n = min(self.LR, Tn - r0)
            sub = self.TN.get_sub_region(r0 * H * 4, n * H * 4)
            k = self.K("igemmT", (16, V, -(-n // self.TB)), (16, 8, 1), m.buf["lm_head.q"], m.buf["lm_head.s"], sub, I(H), self.LOG, I(V), I(V), I(H), I(0), I(n))
            k2 = self.K("nllT", (1024, n), (1024, 1), self.LOG, I(V), np.float32(m.cap), m.ids, I(tgt0 + r0), m.NLL, I(out0 + r0))
            for kk, g, l in (k, k2):
                cl.enqueue_nd_range_kernel(m.q, kk, g, l)
            m.q.finish()
