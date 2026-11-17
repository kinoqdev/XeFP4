"""Qwen3.5-9B XeFP4 decoder with MTP speculative decoding. One iteration runs entirely on the GPU:
1) The main model processes [accepted token a, draft d] in one pass (weights are read once): o0 follows a, o1 follows d.
2) Judge: if o0 == d, accept d and o1 (2 tokens); otherwise accept o0 only and roll the linear-attention state
   back to the first token (the delta-rule update of the second token is deferred, the conv window is restored).
3) MTP drafts the next token from [norm(embed(next token)), normed main hidden state].
"""
from __future__ import annotations


import numpy as np
import pyopencl as cl

from .model import Model, SRC
from ..device import knob

BSRC = r"""
#define UNR __attribute__((opencl_unroll_hint))
inline float wsumB(float v, __local float* r) {
    int l = get_local_id(0), n = get_local_size(0); r[l] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) { if (l < s) r[l] += r[l + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float o = r[0]; barrier(CLK_LOCAL_MEM_FENCE); return o;
}
inline float siluB(float v) { return v / (1.0f + exp(-v)); }
__kernel void embed2(__global const uchar* codes, __global const uint* fac, __global const half* rs, __global const ushort* E, int xe,
                     __global const int* ta, __global const int* td, __global float* X, int H) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    size_t n = t == 0 ? ta[0] : td[0];
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
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm4b(__global const float* X, __global const float* w, __global float* Y, int H, float eps, int plus1) {
    __local float r[64]; __local float tot;
    int l = get_local_id(0), t = get_group_id(1), n4 = H / 4;
    __global const float* x = X + (size_t)t * H; __global float* y = Y + (size_t)t * H;
    float s = 0.0f;
    for (int i = l; i < n4; i += 1024) { float4 v = vload4(i, x); s += dot(v, v); }
    s = sub_group_reduce_add(s);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_id() == 0) { int j = get_sub_group_local_id(); float q = r[j] + r[j + 16] + r[j + 32] + r[j + 48]; q = sub_group_reduce_add(q); if (j == 0) tot = q; }
    barrier(CLK_LOCAL_MEM_FENCE);
    float k = rsqrt(tot / H + eps);
    for (int i = l; i < n4; i += 1024) { float4 v = vload4(i, x), g = vload4(i, w); vstore4(v * k * (plus1 ? 1.0f + g : g), i, y); }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemvB(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                    int N, int K, int add) {
    int row = get_global_id(1); if (row >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32;
    __global const uint* cr = codes + (size_t)row * (K / 8);
    __global const uint* fr = fac + (size_t)row * FW;
    float a0 = 0.0f, a1 = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3; ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        float m = (float)(((fw >> (bo & 31)) & 7) + 1);
        float s0 = 0.0f, s1 = 0.0f;
        UNR for (int j = 0; j < G / 32; j++) {
            uint4 u = vload4(0, cr + b * (G / 8) + j * 4);
            __global const float* x0 = x + b * G + j * 32; __global const float* x1 = x0 + K;
            s0 += dot8(u.s0, vload8(0, x0)) + dot8(u.s1, vload8(1, x0)) + dot8(u.s2, vload8(2, x0)) + dot8(u.s3, vload8(3, x0));
            s1 += dot8(u.s0, vload8(0, x1)) + dot8(u.s1, vload8(1, x1)) + dot8(u.s2, vload8(2, x1)) + dot8(u.s3, vload8(3, x1));
        }
        a0 += s0 * m; a1 += s1 * m;
    }
    a0 = sub_group_reduce_add(a0); a1 = sub_group_reduce_add(a1);
    if (l == 0) {
        float sc = vload_half(row, rs) * 0.125f;
        y[row] = add ? y[row] + a0 * sc : a0 * sc; y[N + row] = add ? y[N + row] + a1 * sc : a1 * sc;
    }
}
inline void dec8(uint u, float4* lo, float4* hi) { *lo = convert_float4(as_char4(dlo(u))); *hi = convert_float4(as_char4(dhi(u))); }
inline float dotp(float4 lo, float4 hi, float8 v) { return dot(lo, (float4)(v.s0, v.s2, v.s4, v.s6)) + dot(hi, (float4)(v.s1, v.s3, v.s5, v.s7)); }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemvB1(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, int add) {
    int row = get_global_id(1); if (row >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32;
    __global const uint* cr = codes + (size_t)row * (K / 8);
    __global const uint* fr = fac + (size_t)row * FW;
    float a0 = 0.0f, a1 = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3; ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        float m = (float)(((fw >> (bo & 31)) & 7) + 1);
        float s0 = 0.0f, s1 = 0.0f;
        UNR for (int j = 0; j < G / 32; j++) {
            uint4 u = vload4(0, cr + b * (G / 8) + j * 4);
            __global const float* x0 = x + b * G + j * 32; __global const float* x1 = x0 + K;
            float4 lo, hi;
            dec8(u.s0, &lo, &hi); s0 += dotp(lo, hi, vload8(0, x0)); s1 += dotp(lo, hi, vload8(0, x1));
            dec8(u.s1, &lo, &hi); s0 += dotp(lo, hi, vload8(1, x0)); s1 += dotp(lo, hi, vload8(1, x1));
            dec8(u.s2, &lo, &hi); s0 += dotp(lo, hi, vload8(2, x0)); s1 += dotp(lo, hi, vload8(2, x1));
            dec8(u.s3, &lo, &hi); s0 += dotp(lo, hi, vload8(3, x0)); s1 += dotp(lo, hi, vload8(3, x1));
        }
        a0 += s0 * m; a1 += s1 * m;
    }
    a0 = sub_group_reduce_add(a0); a1 = sub_group_reduce_add(a1);
    if (l == 0) {
        float sc = vload_half(row, rs) * 0.125f;
        y[row] = add ? y[row] + a0 * sc : a0 * sc; y[N + row] = add ? y[N + row] + a1 * sc : a1 * sc;
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemvB4(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, int add) {
    int r0 = get_global_id(1) * RB; if (r0 >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32, K8 = K / 8;
    float a0[RB], a1[RB];
    UNR for (int r = 0; r < RB; r++) { a0[r] = 0.0f; a1[r] = 0.0f; }
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        float mm[RB], s0[RB], s1[RB];
        UNR for (int r = 0; r < RB; r++) {
            __global const uint* fr = fac + (size_t)(r0 + r) * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            mm[r] = (float)(((fw >> sh) & 7) + 1); s0[r] = 0.0f; s1[r] = 0.0f;
        }
        UNR for (int j = 0; j < G / 8; j++) {
            int c8 = b * (G / 8) + j;
            float8 v0 = vload8(c8, x), v1 = vload8(c8 + K8, x);
            UNR for (int r = 0; r < RB; r++) {
                uint u = codes[(size_t)(r0 + r) * K8 + c8];
                float4 lo = convert_float4(as_char4(dlo(u))), hi = convert_float4(as_char4(dhi(u)));
                s0[r] += dotp(lo, hi, v0); s1[r] += dotp(lo, hi, v1);
            }
        }
        UNR for (int r = 0; r < RB; r++) { a0[r] += s0[r] * mm[r]; a1[r] += s1[r] * mm[r]; }
    }
    UNR for (int r = 0; r < RB; r++) {
        float v0 = sub_group_reduce_add(a0[r]), v1 = sub_group_reduce_add(a1[r]);
        if (l == 0) { int row = r0 + r; float sc = vload_half(row, rs) * 0.125f; v0 *= sc; v1 *= sc;
            y[row] = add ? y[row] + v0 : v0; y[N + row] = add ? y[N + row] + v1 : v1; }
    }
}
inline float dot8h(uint u, __global const half* p, int o) {
    float8 v = vload_half8(o, p); return dot8(u, v);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemvBh(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const half* xh, __global float* y,
                     int N, int K, int add) {
    int row = get_global_id(1); if (row >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32;
    __global const uint* cr = codes + (size_t)row * (K / 8);
    __global const uint* fr = fac + (size_t)row * FW;
    float a0 = 0.0f, a1 = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3; ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        float m = (float)(((fw >> (bo & 31)) & 7) + 1);
        float s0 = 0.0f, s1 = 0.0f;
        UNR for (int j = 0; j < G / 32; j++) {
            uint4 u = vload4(0, cr + b * (G / 8) + j * 4);
            int o = (b * G + j * 32) / 8;
            s0 += dot8h(u.s0, xh, o) + dot8h(u.s1, xh, o + 1) + dot8h(u.s2, xh, o + 2) + dot8h(u.s3, xh, o + 3);
            int o1 = o + K / 8;
            s1 += dot8h(u.s0, xh, o1) + dot8h(u.s1, xh, o1 + 1) + dot8h(u.s2, xh, o1 + 2) + dot8h(u.s3, xh, o1 + 3);
        }
        a0 += s0 * m; a1 += s1 * m;
    }
    a0 = sub_group_reduce_add(a0); a1 = sub_group_reduce_add(a1);
    if (l == 0) {
        float sc = vload_half(row, rs) * 0.125f;
        y[row] = add ? y[row] + a0 * sc : a0 * sc; y[N + row] = add ? y[N + row] + a1 * sc : a1 * sc;
    }
}
__kernel void f2h(__global const float* x, __global half* y, int n) { int i = get_global_id(0); if (i < n) vstore_half(x[i], i, y); }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemvBR(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, int add) {
    int r0 = get_global_id(1) * RB; if (r0 >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32;
    float a0[RB], a1[RB];
    UNR for (int r = 0; r < RB; r++) { a0[r] = 0.0f; a1[r] = 0.0f; }
    for (int b = l; b < nb; b += 16) {
        float8 x0[G / 8], x1[G / 8];
        UNR for (int j = 0; j < G / 8; j++) { x0[j] = vload8(j, x + b * G); x1[j] = vload8(j, x + K + b * G); }
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        UNR for (int r = 0; r < RB; r++) {
            size_t row = r0 + r;
            __global const uint* cr = codes + row * (K / 8) + b * (G / 8);
            __global const uint* fr = fac + row * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            float m = (float)(((fw >> sh) & 7) + 1), s0 = 0.0f, s1 = 0.0f;
            UNR for (int j = 0; j < G / 32; j++) {
                uint4 u = vload4(j, cr);
                s0 += dot8(u.s0, x0[4 * j]) + dot8(u.s1, x0[4 * j + 1]) + dot8(u.s2, x0[4 * j + 2]) + dot8(u.s3, x0[4 * j + 3]);
                s1 += dot8(u.s0, x1[4 * j]) + dot8(u.s1, x1[4 * j + 1]) + dot8(u.s2, x1[4 * j + 2]) + dot8(u.s3, x1[4 * j + 3]);
            }
            a0[r] += s0 * m; a1[r] += s1 * m;
        }
    }
    UNR for (int r = 0; r < RB; r++) {
        float v0 = sub_group_reduce_add(a0[r]), v1 = sub_group_reduce_add(a1[r]);
        if (l == 0) { int row = r0 + r; float sc = vload_half(row, rs) * 0.125f; v0 *= sc; v1 *= sc;
            y[row] = add ? y[row] + v0 : v0; y[N + row] = add ? y[N + row] + v1 : v1; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void lmheadB(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K) {
    int r0 = get_global_id(1) * 2; if (r0 >= N) return;
    int l = get_local_id(0);
    float acc[2][2] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    for (int k = l * 16; k < K; k += 256) {
        float16 v0 = vload16(0, x + k), v1 = vload16(0, x + K + k);
        UNR for (int r = 0; r < 2; r++) {
            float16 a = convert_float16(vload16(0, q + (size_t)(r0 + r) * K + k));
            acc[r][0] += dot(a.s0123, v0.s0123) + dot(a.s4567, v0.s4567) + dot(a.s89ab, v0.s89ab) + dot(a.scdef, v0.scdef);
            acc[r][1] += dot(a.s0123, v1.s0123) + dot(a.s4567, v1.s4567) + dot(a.s89ab, v1.s89ab) + dot(a.scdef, v1.scdef);
        }
    }
    UNR for (int r = 0; r < 2; r++) UNR for (int t = 0; t < 2; t++) {
        float v = sub_group_reduce_add(acc[r][t]); if (l == 0) y[(size_t)t * N + r0 + r] = v * sc[r0 + r];
    }
}
__kernel void conv2(__global const float* P, int ldp, __global float* st, __global float* bak, __global const float* w, __global float* Cx, int C, __global const int* flag) {
    int c = get_global_id(0); if (c >= C) return;
    float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    if (flag[0] == 0) { s2 = s1; s1 = s0; s0 = bak[c]; }
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float x0 = P[c], x1 = P[ldp + c];
    float v0 = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x0;
    float v1 = w0 * s1 + w1 * s2 + w2 * x0 + w3 * x1;
    bak[c] = s1; st[c] = s2; st[C + c] = x0; st[2 * C + c] = x1;
    Cx[c] = siluB(v0); Cx[C + c] = siluB(v1);
}
__kernel void dnetB(__global const float* Cx, int C, int KD, __global const float* P, int ldp, int ZO, int BO, int AO,
                    __global const float* A_log, __global const float* dtb, __global float* S, __global float* PK, __global float* PD, __global float* PE,
                    __global const float* gw, __global float* out, int VH, float eps, int gplus1, __global const int* flag) {
    __local float sk0[128], sq0[128], sk1[128], sq1[128], spk[128], r[256], pa[4][2][128];
    int h = get_group_id(0), lid = get_local_id(0), d = lid & 127, part = lid >> 7, kh = h / (VH / (KD / 128));
    float z0 = part == 0 ? 1.0f : 0.0f;
    float q0 = Cx[kh * 128 + d], k0 = Cx[KD + kh * 128 + d], v0 = Cx[2 * KD + h * 128 + d];
    float q1 = Cx[C + kh * 128 + d], k1 = Cx[C + KD + kh * 128 + d], v1 = Cx[C + 2 * KD + h * 128 + d];
    float nq0 = wsumB(z0 * q0 * q0, r), nk0 = wsumB(z0 * k0 * k0, r), nq1 = wsumB(z0 * q1 * q1, r), nk1 = wsumB(z0 * k1 * k1, r);
    q0 *= rsqrt(nq0 + 1e-6f) * 0.08838834764831845f; k0 *= rsqrt(nk0 + 1e-6f);
    q1 *= rsqrt(nq1 + 1e-6f) * 0.08838834764831845f; k1 *= rsqrt(nk1 + 1e-6f);
    int acc = flag[0];
    if (part == 0) { sk0[d] = k0; sq0[d] = q0; sk1[d] = k1; sq1[d] = q1; spk[d] = acc ? PK[h * 128 + d] : 0.0f; }
    float pdl = acc ? PD[h * 128 + d] : 0.0f, pe = acc ? PE[h] : 1.0f;
    float k0q0 = wsumB(z0 * k0 * q0, r), k0k1 = wsumB(z0 * k0 * k1, r), k0q1 = wsumB(z0 * k0 * q1, r), k1q1 = wsumB(z0 * k1 * q1, r);
    __global float* Sh = S + (size_t)h * 128 * 128;
    int i0 = part * 64;
    float sv[64], b0 = 0.0f, c0 = 0.0f, b1 = 0.0f, c1 = 0.0f;
    UNR for (int j = 0; j < 64; j++) sv[j] = Sh[(i0 + j) * 128 + d];
    UNR for (int j = 0; j < 64; j++) {
        float s = sv[j] * pe + spk[i0 + j] * pdl; sv[j] = s;
        b0 += s * sk0[i0 + j]; c0 += s * sq0[i0 + j]; b1 += s * sk1[i0 + j]; c1 += s * sq1[i0 + j];
    }
    pa[0][part][d] = b0; pa[1][part][d] = c0; pa[2][part][d] = b1; pa[3][part][d] = c1;
    barrier(CLK_LOCAL_MEM_FENCE);
    b0 = pa[0][0][d] + pa[0][1][d]; c0 = pa[1][0][d] + pa[1][1][d]; b1 = pa[2][0][d] + pa[2][1][d]; c1 = pa[3][0][d] + pa[3][1][d];
    __global const float* p0 = P; __global const float* p1 = P + ldp;
    float bt0 = 1.0f / (1.0f + exp(-p0[BO + h])), bt1 = 1.0f / (1.0f + exp(-p1[BO + h]));
    float av0 = p0[AO + h] + dtb[h], av1 = p1[AO + h] + dtb[h], al = exp(A_log[h]);
    float e0 = exp(-al * (av0 > 20.0f ? av0 : log1p(exp(av0)))), e1 = exp(-al * (av1 > 20.0f ? av1 : log1p(exp(av1))));
    float dl0 = (v0 - e0 * b0) * bt0;
    float o0 = e0 * c0 + dl0 * k0q0;
    float sk1v = e0 * b1 + dl0 * k0k1, sq1v = e0 * c1 + dl0 * k0q1;     // S0ᵀk1, S0ᵀq1
    float dl1 = (v1 - e1 * sk1v) * bt1;
    float o1 = e1 * sq1v + dl1 * k1q1;
    UNR for (int j = 0; j < 64; j++) Sh[(i0 + j) * 128 + d] = sv[j] * e0 + sk0[i0 + j] * dl0;
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (part == 0) { PK[h * 128 + d] = k1; PD[h * 128 + d] = dl1; }
    if (lid == 0) PE[h] = e1;
    float ms0 = wsumB(z0 * o0 * o0, r) / 128.0f, ms1 = wsumB(z0 * o1 * o1, r) / 128.0f;
    if (part == 0) {
        float g = gplus1 ? 1.0f + gw[d] : gw[d];
        out[h * 128 + d] = o0 * rsqrt(ms0 + eps) * g * siluB(p0[ZO + h * 128 + d]);
        out[VH * 128 + h * 128 + d] = o1 * rsqrt(ms1 + eps) * g * siluB(p1[ZO + h * 128 + d]);
    }
}
__kernel void attnprep2(__global const float* P, int ldp, __global const float* qn, __global const float* kn, __global float* Q, __global float* kc, __global float* vc,
                        __global const int* pb, float theta, int NH, int KH, int RD, float eps, int plus1) {
    __local float r[256], sh[256];
    int g = get_group_id(0), t = get_group_id(1), d = get_local_id(0), pos = pb[0] + t;
    __global const float* p = P + (size_t)t * ldp;
    float x = g < NH ? p[g * 512 + d] : p[NH * 512 + (g - NH) * 256 + d];
    float k = rsqrt(wsumB(x * x, r) / 256.0f + eps);
    __global const float* w = g < NH ? qn : kn;
    float y = x * k * (plus1 ? 1.0f + w[d] : w[d]);
    sh[d] = y; barrier(CLK_LOCAL_MEM_FENCE);
    int hr = RD / 2;
    if (d < RD) {
        int i = d < hr ? d : d - hr;
        float f = (float)pos * pow(theta, -2.0f * (float)i / (float)RD);
        float c = cos(f), s = sin(f);
        y = d < hr ? sh[d] * c - sh[d + hr] * s : sh[d] * c + sh[d - hr] * s;
    }
    if (g < NH) Q[((size_t)t * NH + g) * 256 + d] = y;
    else { size_t o = ((size_t)pos * KH + (g - NH)) * 256 + d; kc[o] = y; vc[o] = p[NH * 512 + KH * 256 + (g - NH) * 256 + d]; }
}
__kernel void attn2(__global const float* Q, __global const float* kc, __global const float* vc, __global const float* P, int ldp, __global float* A,
                    __global float* sc, __global const int* pb, int TM, int KH, int GQ) {
    __local float sq[256], r[256];
    int h = get_group_id(0), t = get_group_id(1), l = get_local_id(0), NH = get_num_groups(0), kh = h / GQ;
    int L = pb[0] + t + 1;
    __global float* s_ = sc + ((size_t)t * NH + h) * TM;
    sq[l] = Q[((size_t)t * NH + h) * 256 + l]; barrier(CLK_LOCAL_MEM_FENCE);
    float mx = -INFINITY;
    for (int j = l; j < L; j += 256) {
        __global const float* kr = kc + ((size_t)j * KH + kh) * 256; float s = 0.0f;
        for (int i = 0; i < 256; i += 4) s += dot(vload4(0, sq + i), vload4(0, kr + i));
        s *= 0.0625f; s_[j] = s; mx = fmax(mx, s);
    }
    r[l] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) r[l] = fmax(r[l], r[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    mx = r[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int j = l; j < L; j += 256) { float e = exp(s_[j] - mx); s_[j] = e; sum += e; }
    sum = wsumB(sum, r);
    barrier(CLK_GLOBAL_MEM_FENCE);
    float o = 0.0f;
    for (int j = 0; j < L; j++) o += s_[j] * vc[((size_t)j * KH + kh) * 256 + l];
    float gt = P[(size_t)t * ldp + h * 512 + 256 + l];
    A[((size_t)t * NH + h) * 256 + l] = o / sum / (1.0f + exp(-gt));
}
__kernel void swiglu2(__global const float* GU, __global float* M, int F) {
    int i = get_global_id(0), t = get_global_id(1); if (i < F) M[(size_t)t * F + i] = siluB(GU[(size_t)t * 2 * F + i]) * GU[(size_t)t * 2 * F + F + i];
}
__kernel void argmax1b(__global const float* v, int N, __global float* pv, __global int* pi) {
    __local float rv[256]; __local int ri[256];
    int l = get_local_id(0), g = get_group_id(0), t = get_group_id(1), P = get_num_groups(0); float bv = -INFINITY; int bi = 0;
    __global const float* vt = v + (size_t)t * N;
    for (int i = g * 4096 + l; i < min(N, (g + 1) * 4096); i += 256) { float x = vt[i]; if (x > bv) { bv = x; bi = i; } }
    rv[l] = bv; ri[l] = bi; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) { pv[t * P + g] = rv[0]; pi[t * P + g] = ri[0]; }
}
__kernel void argmax2b(__global const float* pv, __global const int* pi, int P, __global int* O) {
    __local float rv[256]; __local int ri[256]; int l = get_local_id(0), t = get_group_id(0);
    rv[l] = l < P ? pv[t * P + l] : -INFINITY; ri[l] = l < P ? pi[t * P + l] : 0; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) O[t] = ri[0];
}
__kernel void judge(__global int* ta, __global const int* td, __global const int* O, __global int* flag, __global int* pos, __global int* mpa, __global int* mpb,
                    __global int* oa, __global int* ob, __global int* outs, __global int* cnt, __global int* nacc) {
    int d = td[0], o0 = O[0], o1 = O[1], c = cnt[0], p = pos[0];
    mpa[0] = p + 1; mpb[0] = p + 2; oa[0] = o0; ob[0] = o1;
    if (o0 == d) { outs[c] = d; outs[c + 1] = o1; cnt[0] = c + 2; flag[0] = 1; ta[0] = o1; pos[0] = p + 2; nacc[0] += 1; }
    else { outs[c] = o0; cnt[0] = c + 1; flag[0] = 0; ta[0] = o0; pos[0] = p + 1; }
}
"""


class Spec:
    def __init__(self, m: Model, mtp_path):
        self.m = m
        m.load_mtp(mtp_path, "eh", "post")
        self._pfx = "B:" 
        c, H, F, I, eps = m.cfg, m.H, m.F, np.int32, np.float32(m.eps)
        G = m.meta["G"]; xe8 = int(m.meta["fmt"] == "xe8")
        opt = [f"-DG={G}", f"-DXE8={xe8}", "-DR=1", "-DDB=64", f"-DNP={m.np_}"]
        import os
        self.bp = cl.Program(m.ctx, SRC + BSRC).build(options=opt + ["-DRB=1"])
        rb = dict(zip(("proj", "out", "gu", "down"), (int(x) for x in knob("Q35_SRB", "4,2,4,2").split(","))))
        self.rb = rb
        self.bpr = {r: (self.bp if r == 1 else cl.Program(m.ctx, SRC + BSRC).build(options=opt + [f"-DRB={r}"])) for r in sorted(set(rb.values()))}
        self.gvk = knob("Q35_SGV", "gemvB4")
        self.cp = cp = {r: cl.Program(m.ctx, SRC).build(options=[f"-DG={G}", f"-DXE8={xe8}", f"-DR={r}", "-DDB=64", f"-DNP={m.np_}", "-DCOND", "-DPOSB"]) for r in (1, 2)}
        mf = cl.mem_flags
        f32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        self.i32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        NPm = max(m.tm[f"layers.{i}.proj.codes"]["shape"][0] for i in range(c["num_hidden_layers"]))
        CC = 2 * m.KD + m.VH * 128
        self.h2, self.t2, self.a2 = f32(2 * H), f32(2 * H), f32(2 * max(H, m.VH * 128, m.NH * 256))
        self.P2, self.C2, self.GU2, self.M2 = f32(2 * NPm), f32(2 * CC), f32(4 * F), f32(2 * F)
        self.Q2, self.SC2, self.lg2 = f32(2 * m.NH * 256), f32(2 * m.NH * m.TM), f32(2 * m.V)
        self.PM2, self.PL2, self.PO2 = f32(2 * m.NH * m.SM), f32(2 * m.NH * m.SM), f32(2 * m.NH * m.SM * 256)
        self.pv2, self.pi2 = f32(2 * m.npart), self.i32(2 * m.npart)
        self.TA, self.TD, self.FLAG, self.POS, self.MPA, self.MPB, self.OA, self.OB = (self.i32(1) for _ in range(8))
        self.O = self.i32(2)
        self.OUTS, self.CNT, self.NACC = self.i32(m.TM), self.i32(1), self.i32(1)
        self.bak = {i: f32(CC) for i, lt in enumerate(c["layer_types"]) if lt == "linear_attention"}
        self.keep = [self.t2.get_sub_region(0, H * 4), self.t2.get_sub_region(H * 4, H * 4)]
        bk = lambda name, g, l, *a: self._k(self.bp, name, g, l, *a)
        xe = "embed.ecodes" in m.tm
        dummy = m.buf["norm"]
        ks = [bk("embed2", (-(-H // 256) * 256, 2), (256, 1), m.buf["embed.ecodes"] if xe else dummy, m.buf["embed.efac"] if xe else dummy,
                 m.buf["embed.ers"] if xe else dummy, dummy if xe else m.buf["embed"], I(int(xe)), self.TA, self.TD, self.h2, I(H))]
        nb = lambda x, w, y, p1: bk("rmsnorm4b", (1024, 2), (1024, 1), x, w, y, I(H), eps, I(p1))

        def gv(name, x, y, add):
            t = m.tm[name + ".codes"]; N, K = t["shape"][0], t["shape"][1] * 2
            kind = name.split(".")[-1]; r = self.rb.get(kind, 1)
            assert N % r == 0
            return self._k(self.bpr[r], self.gvk, (16, -(-(N // r) // 8) * 8), (16, 8), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, y, I(N), I(K), I(add),
                           label="B:gemv." + kind)
        for i, lt in enumerate(c["layer_types"]):
            p = f"layers.{i}."; NPl = m.tm[p + "proj.codes"]["shape"][0]
            ks.append(nb(self.h2, m.buf[p + "in_norm"], self.t2, m.p1))
            ks.append(gv(p + "proj", self.t2, self.P2, 0))
            if lt == "linear_attention":
                st, S, PK, PD, PE = m.state[i]; ZO = CC; BO = ZO + m.VH * 128; AO = BO + m.VH
                ks.append(bk("conv2", (-(-CC // 256) * 256,), (256,), self.P2, I(NPl), st, self.bak[i], m.buf[p + "conv"], self.C2, I(CC), self.FLAG))
                ks.append(bk("dnetB", (m.VH * 256,), (256,), self.C2, I(CC), I(m.KD), self.P2, I(NPl), I(ZO), I(BO), I(AO), m.buf[p + "A_log"], m.buf[p + "dt_bias"],
                             S, PK, PD, PE, m.buf[p + "gnorm"], self.a2, I(m.VH), eps, I(m.gp1), self.FLAG))
                ain = m.VH * 128
            else:
                kc, vc = m.state[i][:2]
                ks.append(bk("attnprep2", ((m.NH + m.KH) * 256, 2), (256, 1), self.P2, I(NPl), m.buf[p + "q_norm"], m.buf[p + "k_norm"], self.Q2, kc, vc,
                             self.POS, np.float32(m.theta), I(m.NH), I(m.KH), I(m.RD), eps, I(m.qp1)))
                if m.fa:
                    if knob("Q35_FATTN2", "1") == "1":
                        ks.append(list(self._k(cp[1], "fattn2", (m.KH * 128, m.SM, 1), (128, 1, 1), self.Q2, kc, vc, self.PM2, self.PL2, self.PO2, I(0), I(m.KH), I(m.SM), I(m.CH),
                                               self.POS, self.FLAG, I(-1), label="B:attn2")))
                    else:
                        ks.append(list(self._k(cp[1], "fattn", (m.KH * 128, m.SM, 2), (128, 1, 1), self.Q2, kc, vc, self.PM2, self.PL2, self.PO2, I(0), I(m.KH), I(m.SM), I(m.CH),
                                               self.POS, self.FLAG, I(-1), label="B:attn2")))
                    ks.append(self._k(cp[1], "fcomb", (m.NH * 256, 2), (256, 1), self.PM2, self.PL2, self.PO2, self.P2, I(NPl), self.a2, I(0), I(m.KH), I(m.SM), I(m.CH),
                                      self.POS, self.FLAG, I(-1), label="B:attn2"))
                else:
                    ks.append(bk("attn2", (m.NH * 256, 2), (256, 1), self.Q2, kc, vc, self.P2, I(NPl), self.a2, self.SC2, self.POS, I(m.TM), I(m.KH), I(m.NH // m.KH)))
                ain = m.NH * 256
            ks.append(gv(p + "out", self.a2, self.h2, 1))
            ks.append(nb(self.h2, m.buf[p + "post_norm"], self.t2, m.p1))
            ks.append(gv(p + "gu", self.t2, self.GU2, 0))
            ks.append(bk("swiglu2", (-(-F // 256) * 256, 2), (256, 1), self.GU2, self.M2, I(F)))
            ks.append(gv(p + "down", self.M2, self.h2, 1))
        ks.append(nb(self.h2, m.buf["norm"], self.t2, m.p1))
        ks.append(bk("lmheadB", (16, -(-(m.V // 2) // 8) * 8), (16, 8), m.buf["lm_head.q"], m.buf["lm_head.s"], self.t2, self.lg2, I(m.V), I(H)))
        ks.append(bk("argmax1b", (m.npart * 256, 2), (256, 1), self.lg2, I(m.V), self.pv2, self.pi2))
        ks.append(bk("argmax2b", (2 * 256,), (256,), self.pv2, self.pi2, I(m.npart), self.O))
        ks.append(bk("judge", (1,), (1,), self.TA, self.TD, self.O, self.FLAG, self.POS, self.MPA, self.MPB, self.OA, self.OB, self.OUTS, self.CNT, self.NACC))
        self.main = ks
        self._pfx = "MA:"; ma = None
        self._vc = {}; self.vk = 0
        self.mtpA = self._mtp(cp, self.OA, self.keep[0], self.MPA, -1, 0)
        self._pfx = "MB:"
        self.mtpB = self._mtp(cp, self.OB, self.keep[1], self.MPB, 1, 1)
        vp = knob("Q35_MTPVOCAB", os.path.join(os.path.dirname(m.path), "vocab_order.npy"))
        vk = int(knob("Q35_MTPK", "65536"))
        if vk and os.path.exists(vp):
            self.set_vocab(np.load(vp), vk)

    def set_vocab(self, order, K):
        """Restricts the MTP draft lm_head to the top-K vocabulary rows (K = 0: full vocabulary). Verification always uses the full vocabulary."""
        m = self.m
        if K:
            if K not in self._vc:
                idx = np.sort(order[:K]).astype(np.int32)
                t, ts = m.tm["lm_head.q"], m.tm["lm_head.s"]
                raw = np.memmap(m.path, dtype=np.uint8, mode="r")
                q = np.asarray(raw[t["off"]:t["off"] + t["nbytes"]]).reshape(m.V, m.H)[idx]
                sc = np.asarray(raw[ts["off"]:ts["off"] + ts["nbytes"]]).view(np.float32)[idx]
                mf = cl.mem_flags
                mk = lambda a: cl.Buffer(m.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))
                self._vc[K] = (mk(q), mk(sc), mk(idx), idx)
        self.vk = K
        self._pfx = "MA:"; self.mtpA = self._mtp(self.cp, self.OA, self.keep[0], self.MPA, -1, 0)
        self._pfx = "MB:"; self.mtpB = self._mtp(self.cp, self.OB, self.keep[1], self.MPB, 1, 1)

    def _k(self, prg, name, g, l, *a, label=None):
        k = cl.Kernel(prg, name); k.set_args(*a)
        self.m.lab[id(k)] = label or (self._pfx + name)
        return (k, g, l)

    def _mtp(self, cp, tok, hid, mp, w_first, w_rest):
        """One conditional MTP step. w_first: condition for the part up to the KV write, w_rest: condition for the rest (-1 = always)."""
        m, H, F, I, eps = self.m, self.m.H, self.m.F, np.int32, np.float32(self.m.eps)
        cf = self.FLAG

        def K(prg_r, name, g, l, w, *a):
            return self._k(cp[prg_r], name, g, l, *a, cf, I(w))
        xe = "embed.ecodes" in m.tm
        e1, e2 = m._keep
        out = []
        if xe:
            out.append(K(1, "embed_xe", (-(-H // 256) * 256,), (256,), w_first, m.buf["embed.ecodes"], m.buf["embed.efac"], m.buf["embed.ers"], tok, m.mE, I(H)))
        else:
            out.append(K(1, "embed", (-(-H // 256) * 256,), (256,), w_first, m.buf["embed"], tok, m.mE, I(H)))
        nrm = lambda w, x, wt, y: K(1, "rmsnorm4", (1024,), (1024,), w, x, wt, y, I(H), eps, I(1))

        def gv(w, name, x, y, add):
            t = m.tm[name + ".codes"]; N, Kk = t["shape"][0], t["shape"][1] * 2
            R = 2 if name.endswith(".gu") else 1
            return K(R, "gemv2", (16, -(-(N // R) // 8) * 8), (16, 8), w, m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, y, I(N), I(Kk), I(add))
        out += [nrm(w_first, m.mE, m.buf["mtp.pe"], e1), nrm(w_first, hid, m.buf["mtp.ph"], e2),
                gv(w_first, "mtp.fcA", e1, m.mh, 0), gv(w_first, "mtp.fcB", e2, m.mh, 1),
                nrm(w_first, m.mh, m.buf["mtp.in_norm"], m.t), gv(w_first, "mtp.proj", m.t, m.P, 0)]
        out.append(self._k(cp[1], "attn_prep", ((m.NH + m.KH) * 256,), (256,), m.P, m.buf["mtp.q_norm"], m.buf["mtp.k_norm"], m.Q, m.mkc, m.mvc,
                           I(0), np.float32(m.theta), I(m.NH), I(m.KH), I(m.RD), eps, I(1), mp, cf, I(w_first)))
        if m.fa:
            out.append(list(self._k(cp[1], m.fa1, (m.KH * 128, m.SM, 1), (128, 1, 1), m.Q, m.mkc, m.mvc, m.PM, m.PL, m.PO, I(0), I(m.KH), I(m.SM), I(m.CH), mp, cf, I(w_rest))))
            out.append(self._k(cp[1], "fcomb", (m.NH * 256, 1), (256, 1), m.PM, m.PL, m.PO, m.P, I(0), m.a, I(0), I(m.KH), I(m.SM), I(m.CH), mp, cf, I(w_rest)))
        else:
            out.append(self._k(cp[1], "attn", (m.NH * 256,), (256,), m.Q, m.mkc, m.mvc, m.P, m.a, m.SC, I(1), I(m.TM), I(m.KH), I(m.NH // m.KH), mp, cf, I(w_rest)))
        out += [gv(w_rest, "mtp.out", m.a, m.mh, 1), nrm(w_rest, m.mh, m.buf["mtp.post_norm"], m.t), gv(w_rest, "mtp.gu", m.t, m.GU, 0),
                K(1, "swiglu", (-(-F // 256) * 256,), (256,), w_rest, m.GU, m.M, I(F)), gv(w_rest, "mtp.down", m.M, m.mh, 1),
                nrm(w_rest, m.mh, m.buf["mtp.norm"], m.t),
                ]
        if getattr(self, "vk", 0):
            qb, sb, ib, idx = self._vc[self.vk]; Nv = len(idx); npv = (Nv + 4095) // 4096
            out += [K(2, "gemv_i8r", (16, -(-(Nv // 2) // 8) * 8), (16, 8), w_rest, qb, sb, m.t, m.logit, I(Nv), I(H)),
                    K(1, "argmax1", (npv * 256,), (256,), w_rest, m.logit, I(Nv), m.pv, m.pi),
                    K(1, "argmax2", (256,), (256,), w_rest, m.pv, m.pi, I(npv), self.TD, m.mouts, I(0)),
                    K(1, "mapid", (1,), (1,), w_rest, self.TD, ib)]
        else:
            out += [K(2, "gemv_i8r", (16, -(-(m.V // 2) // 8) * 8), (16, 8), w_rest, m.buf["lm_head.q"], m.buf["lm_head.s"], m.t, m.logit, I(m.V), I(H)),
                    K(1, "argmax1", (m.npart * 256,), (256,), w_rest, m.logit, I(m.V), m.pv, m.pi),
                    K(1, "argmax2", (256,), (256,), w_rest, m.pv, m.pi, I(m.npart), self.TD, m.mouts, I(0))]
        return out

    def prompt(self, ids):
        """Prompt token by token (main model + MTP KV) -> first token y0 and first draft d0."""
        m = self.m
        m.reset(); m.load_ids(ids)
        for kq in (m.mkc, m.mvc):
            cl.enqueue_fill_buffer(m.q, kq, np.float32(0), 0, kq.size)
        T = len(ids)
        for i in range(T):
            m.step(i, out_step=0 if i == T - 1 else None, feed=i)
            if i < T - 1:
                m.st[0].set_arg(1, np.int32(i + 1)); m.run([m.st]); m.mtp_step(i + 1, 0)
        m.mtp_step(T, 0)
        y0, d0 = np.zeros(1, np.int32), np.zeros(1, np.int32)
        cl.enqueue_copy(m.q, y0, m.outs); cl.enqueue_copy(m.q, d0, m.mouts); m.q.finish()
        for b, v in ((self.TA, y0[0]), (self.TD, d0[0]), (self.FLAG, 1), (self.POS, T), (self.CNT, 0), (self.NACC, 0)):
            cl.enqueue_copy(m.q, b, np.array([v], np.int32))
        m.q.finish()
        self.base, self.it = T, 0
        return int(y0[0])

    def prompt_fast(self, ids, pf):
        """Long prompt: chunked prefill with the MTP KV filled at once -> first token y0, first draft d0."""
        m = self.m; T = len(ids)
        m.reset(); m.load_ids(list(ids) + [0])
        for kq in (m.mkc, m.mvc):
            cl.enqueue_fill_buffer(m.q, kq, np.float32(0), 0, kq.size)
        pf.run_long(T, mtp=True)
        pf.first_token(0)
        m.mtp_step(T, 0)
        y0, d0 = np.zeros(1, np.int32), np.zeros(1, np.int32)
        cl.enqueue_copy(m.q, y0, m.outs); cl.enqueue_copy(m.q, d0, m.mouts); m.q.finish()
        for b, v in ((self.TA, y0[0]), (self.TD, d0[0]), (self.FLAG, 1), (self.POS, T), (self.CNT, 0), (self.NACC, 0)):
            cl.enqueue_copy(m.q, b, np.array([v], np.int32))
        m.q.finish()
        self.base, self.it = T, 0
        return int(y0[0])

    def iterate(self, n):
        """Runs n iterations. Positions live on the GPU, but after i iterations position <= start + 2i, so attention launches only that many chunks."""
        m = self.m
        dyn = knob("Q35_SDYN", "1") == "1"
        ents = [e for lst in (self.main, self.mtpA, self.mtpB) for e in lst if isinstance(e, list) and e[0].function_name in ("fattn", "fattn1", "fattn2")]
        for _ in range(n):
            if dyn:
                S = min(m.SM, -(-(self.base + 2 * self.it + 4) // m.CH))
                for e in ents:
                    e[1] = (e[1][0], S, e[1][2])
            self.m.run(self.main); self.m.run(self.mtpA); self.m.run(self.mtpB); self.m.q.flush()
            self.it += 1

    def read(self):
        c, a = np.zeros(1, np.int32), np.zeros(1, np.int32)
        cl.enqueue_copy(self.m.q, c, self.CNT); cl.enqueue_copy(self.m.q, a, self.NACC); self.m.q.finish()
        o = np.zeros(max(1, int(c[0])), np.int32); cl.enqueue_copy(self.m.q, o, self.OUTS); self.m.q.finish()
        return o[:int(c[0])].tolist(), int(a[0])
