"""Gemma 4 26B-A4B XeFP4 decoder (OpenCL, batch 1).
One layer for one token: norm -> qkv GEMV -> per-head norm, rotary, KV cache -> attention (split + combine) -> o GEMV
-> dense MLP (gelu gate * up -> down) in parallel with MoE (router -> top 8 -> expert gate/up -> down, weighted sum)
-> normed sum of both branches -> residual * layer_scalar.
Sliding-window layers keep a ring of window + prefill-chunk KV slots; global layers keep TMAX slots.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyopencl as cl

from .. import device as K2
from ..device import knob

KV16 = knob("G4_KV16", "1") == "1"                     # public default: FP16 KV cache
KVB = 2 if KV16 else 4
KVM = ("#define KV16 1" + chr(10) if KV16 else "") + r"""
#ifdef KV16
#define KT half
#define KLD1(o, p) vload_half(o, p)
#define KLD2(o, p) vload_half2(o, p)
#define KLD4(o, p) vload_half4(o, p)
#define KLD8(o, p) vload_half8(o, p)
#define KLD16(o, p) vload_half16(o, p)
#define KST(v, o, p) vstore_half(v, o, p)
#else
#define KT float
#define KLD1(o, p) ((p)[o])
#define KLD2(o, p) vload2(o, p)
#define KLD4(o, p) vload4(o, p)
#define KLD8(o, p) vload8(o, p)
#define KLD16(o, p) vload16(o, p)
#define KST(v, o, p) ((p)[o] = (v))
#endif
"""

GSRC = KVM + r"""
#define UNR __attribute__((opencl_unroll_hint))
__constant float ELUT[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
inline uint e2(uint x) {
    uint m = x & 0x07070707u, lo = m & 0x03030303u;
    uint t = 0x04040404u | ((lo & 0x01010101u) << 1);
    uint t2 = t + (t & (((lo >> 1) & 0x01010101u) * 0xFFu));
    uint hm = ((m >> 2) & 0x01010101u) * 0xFFu;
    uint mag = (t2 & hm) | (lo & ~hm);
    uint s = ((x >> 3) & 0x01010101u) & (((mag + 0x7F7F7F7Fu) >> 7) & 0x01010101u);
    return (mag ^ (s * 0xFFu)) + s;
}
inline uint dlo(uint u) { return e2(u & 0x0F0F0F0Fu); }
inline uint dhi(uint u) { return e2((u >> 4) & 0x0F0F0F0Fu); }
inline float dot8(uint u, float8 v) {
    return dot(convert_float4(as_char4(dlo(u))), (float4)(v.s0, v.s2, v.s4, v.s6)) + dot(convert_float4(as_char4(dhi(u))), (float4)(v.s1, v.s3, v.s5, v.s7));
}
inline float gelu(float x) { return 0.5f * x * (1.0f + tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); }
inline float bfr(float x) { uint u = as_uint(x); u += 0x7FFFu + ((u >> 16) & 1u); return as_float(u & 0xFFFF0000u); }

// XeFP4-8 lin8 GEMV: R rows per sub-group (expert mode: rows of expert idx[j])
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemv(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                    int N, int K, int add, __global const int* idx, int NE) {
    int gid = get_global_id(1), l = get_local_id(0), nb = K / G, FW = (nb * 3 + 31) >> 5;
    size_t rbase; int ybase, r0;
    if (NE > 0) { int per = NE / R, j = gid / per; r0 = (gid % per) * R; if (j >= N) return; rbase = (size_t)idx[j] * NE; ybase = j * NE; }
    else { r0 = gid * R; if (r0 >= N) return; rbase = 0; ybase = 0; }
    float acc[R];
    UNR for (int r = 0; r < R; r++) acc[r] = 0.0f;
    for (int b = l; b < nb; b += 16) {
        __global const float* xp = x + b * G;
        float8 xv[G / 8];
        UNR for (int j = 0; j < G / 8; j++) xv[j] = vload8(j, xp);
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        UNR for (int r = 0; r < R; r++) {
            size_t row = rbase + r0 + r;
            __global const uint* cr = codes + row * (K / 8) + b * (G / 8);
            __global const uint* fr = fac + row * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            float s = 0.0f;
            UNR for (int j = 0; j < G / 32; j++) {
                uint4 u = vload4(j, cr);
                s += dot8(u.s0, xv[4 * j]) + dot8(u.s1, xv[4 * j + 1]) + dot8(u.s2, xv[4 * j + 2]) + dot8(u.s3, xv[4 * j + 3]);
            }
            acc[r] += s * (float)(((fw >> sh) & 7) + 1);
        }
    }
    UNR for (int r = 0; r < R; r++) {
        float v = sub_group_reduce_add(acc[r]);
        if (l == 0) { size_t row = rbase + r0 + r; v *= vload_half(row, rs) * 0.125f; int o = ybase + r0 + r; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xdown(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                    int N, int K, __global const int* idx, __global const float* wt, int TK) {
    int row = get_global_id(1), l = get_local_id(0); if (row >= N) return;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    float acc = 0.0f;
    for (int t = l; t < TK * nb; t += 16) {
        int j = t / nb, b = t - j * nb;
        size_t rr = (size_t)idx[j] * N + row;
        __global const float* xp = x + j * K + b * G;
        __global const uint* cr = codes + rr * (K / 8) + b * (G / 8);
        __global const uint* fr = fac + rr * FW;
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
        float s = 0.0f;
        UNR for (int q = 0; q < G / 32; q++) {
            uint4 u = vload4(q, cr);
            s += dot8(u.s0, vload8(4 * q, xp)) + dot8(u.s1, vload8(4 * q + 1, xp)) + dot8(u.s2, vload8(4 * q + 2, xp)) + dot8(u.s3, vload8(4 * q + 3, xp));
        }
        acc += s * (float)(((fw >> sh) & 7) + 1) * vload_half(rr, rs) * 0.125f * wt[j];
    }
    acc = sub_group_reduce_add(acc);
    if (l == 0) y[row] = acc;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemvS(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, int add, __global const int* idx, int NE, int SPL) {
    __local float red[8][R];
    int gid = get_global_id(1), sgl = get_local_id(1), l = get_local_id(0), part = gid % SPL, rg = gid / SPL;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    size_t rbase = 0; int ybase = 0, r0; bool ok;
    if (NE > 0) { int per = NE / R, j = rg / per; r0 = (rg % per) * R; ok = j < N; if (ok) { rbase = (size_t)idx[j] * NE; ybase = j * NE; } }
    else { r0 = rg * R; ok = r0 < N; }
    float acc[R];
    UNR for (int r = 0; r < R; r++) acc[r] = 0.0f;
    if (ok) {
        for (int b = l + 16 * part; b < nb; b += 16 * SPL) {
            __global const float* xp = x + b * G;
            float8 xv[G / 8];
            UNR for (int j = 0; j < G / 8; j++) xv[j] = vload8(j, xp);
            int bo = b * 3, wo = bo >> 5, sh = bo & 31;
            UNR for (int r = 0; r < R; r++) {
                size_t row = rbase + r0 + r;
                __global const uint* cr = codes + row * (K / 8) + b * (G / 8);
                __global const uint* fr = fac + row * FW;
                ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
                float s = 0.0f;
                UNR for (int j = 0; j < G / 32; j++) {
                    uint4 u = vload4(j, cr);
                    s += dot8(u.s0, xv[4 * j]) + dot8(u.s1, xv[4 * j + 1]) + dot8(u.s2, xv[4 * j + 2]) + dot8(u.s3, xv[4 * j + 3]);
                }
                acc[r] += s * (float)(((fw >> sh) & 7) + 1);
            }
        }
    }
    UNR for (int r = 0; r < R; r++) { float v = sub_group_reduce_add(acc[r]); if (l == 0) red[sgl][r] = v; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (ok && part == 0 && l < R) {
        float v = 0.0f; for (int q = 0; q < SPL; q++) v += red[sgl + q][l];
        size_t row = rbase + r0 + l; v *= vload_half(row, rs) * 0.125f; int o = ybase + r0 + l; y[o] = add ? y[o] + v : v;
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xdownS(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, __global const int* idx, __global const float* wt, int TK, int SPL) {
    __local float red[8];
    int gid = get_global_id(1), sgl = get_local_id(1), l = get_local_id(0), part = gid % SPL, row = gid / SPL;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    float acc = 0.0f;
    if (row < N) {
        for (int t = l + 16 * part; t < TK * nb; t += 16 * SPL) {
            int j = t / nb, b = t - j * nb;
            size_t rr = (size_t)idx[j] * N + row;
            __global const float* xp = x + j * K + b * G;
            __global const uint* cr = codes + rr * (K / 8) + b * (G / 8);
            __global const uint* fr = fac + rr * FW;
            int bo = b * 3, wo = bo >> 5, sh = bo & 31;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            float s = 0.0f;
            UNR for (int q = 0; q < G / 32; q++) {
                uint4 u = vload4(q, cr);
                s += dot8(u.s0, vload8(4 * q, xp)) + dot8(u.s1, vload8(4 * q + 1, xp)) + dot8(u.s2, vload8(4 * q + 2, xp)) + dot8(u.s3, vload8(4 * q + 3, xp));
            }
            acc += s * (float)(((fw >> sh) & 7) + 1) * vload_half(rr, rs) * 0.125f * wt[j];
        }
    }
    acc = sub_group_reduce_add(acc);
    if (l == 0) red[sgl] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (row < N && part == 0 && l == 0) { float v = 0.0f; for (int q = 0; q < SPL; q++) v += red[sgl + q]; y[row] = v; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void igemv(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K, int add) {
    int r0 = get_global_id(1) * R; if (r0 >= N) return;
    int l = get_local_id(0);
    float acc[R];
    UNR for (int r = 0; r < R; r++) acc[r] = 0.0f;
    for (int k = l * 16; k < K; k += 256) {
        float16 v = vload16(0, x + k);
        UNR for (int r = 0; r < R; r++) {
            char16 a = vload16(0, q + (size_t)(r0 + r) * K + k);
            acc[r] += dot(convert_float4(a.s0123), v.s0123) + dot(convert_float4(a.s4567), v.s4567) + dot(convert_float4(a.s89ab), v.s89ab) + dot(convert_float4(a.scdef), v.scdef);
        }
    }
    UNR for (int r = 0; r < R; r++) { float t = sub_group_reduce_add(acc[r]); if (l == 0) { float v = t * sc[r0 + r]; y[r0 + r] = add ? y[r0 + r] + v : v; } }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void fgemv(__global const float* w, __global const float* x, __global float* y, int N, int K) {
    int row = get_global_id(1), l = get_local_id(0); if (row >= N) return;
    float acc = 0.0f;
    for (int k = l * 4; k < K; k += 64) acc += dot(vload4(0, w + (size_t)row * K + k), vload4(0, x + k));
    acc = sub_group_reduce_add(acc);
    if (l == 0) y[row] = acc;
}
__kernel void embed8(__global const char* q, __global const float* sc, __global const int* tok, __global float* x, int H, float nrm) {
    int i = get_global_id(0); if (i >= H) return;
    size_t n = tok[0];
    x[i] = bfr(bfr((float)q[n * H + i] * sc[n]) * nrm);
}
inline float bsum(float v, __local float* r, __local float* tot) {
    v = sub_group_reduce_add(v);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_id() == 0) { int j = get_sub_group_local_id(); float t = 0.0f; for (int k = j; k < (int)get_num_sub_groups(); k += 16) t += r[k]; t = sub_group_reduce_add(t); if (j == 0) *tot = t; }
    barrier(CLK_LOCAL_MEM_FENCE);
    float o = *tot; barrier(CLK_LOCAL_MEM_FENCE); return o;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gnorm(__global const float* x, __global const float* w, __global const float* mul, __global float* y, int H, float eps, float c, int hasw, int hasm) {
    __local float r[64]; __local float tot;
    int l = get_local_id(0); float s = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = x[i]; s += v * v; }
    float k = rsqrt(bsum(s, r, &tot) / H + eps) * c;
    for (int i = l; i < H; i += 1024) { float v = x[i] * k; if (hasw) v *= w[i]; if (hasm) v *= mul[i]; y[i] = v; }
}
// h += rms(x)·w
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void normadd(__global const float* x, __global const float* w, __global float* h, int H, float eps) {
    __local float r[64]; __local float tot;
    int l = get_local_id(0); float s = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = x[i]; s += v * v; }
    float k = rsqrt(bsum(s, r, &tot) / H + eps);
    for (int i = l; i < H; i += 1024) h[i] += x[i] * k * w[i];
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ffend(__global const float* d1, __global const float* e, __global const float* w1, __global const float* w2, __global const float* w,
                    __global const float* ls, __global float* h, int H, float eps) {
    __local float r[64]; __local float tot; __local float t[4096];
    int l = get_local_id(0); float s1 = 0.0f, s2 = 0.0f;
    for (int i = l; i < H; i += 1024) { float a = d1[i], b = e[i]; s1 += a * a; s2 += b * b; }
    float k1 = rsqrt(bsum(s1, r, &tot) / H + eps), k2 = rsqrt(bsum(s2, r, &tot) / H + eps);
    float s3 = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = d1[i] * k1 * w1[i] + e[i] * k2 * w2[i]; t[i] = v; s3 += v * v; }
    float k3 = rsqrt(bsum(s3, r, &tot) / H + eps), sc = ls[0];
    for (int i = l; i < H; i += 1024) h[i] = (h[i] + t[i] * k3 * w[i]) * sc;
}
__kernel void geglu(__global const float* gu, __global float* o, int F, int TK) {
    int i = get_global_id(0); if (i >= F * TK) return;
    int j = i / F, c = i - j * F;
    o[i] = gelu(gu[j * 2 * F + c]) * gu[j * 2 * F + F + c];
}
__kernel void route(__global const float* lg, __global const float* pes, __global int* idx, __global float* wt, int E, int TK) {
    __local float v[128]; __local int sel[16]; __local float pe[16];
    int l = get_local_id(0);
    float x = l < E ? lg[l] : -INFINITY; v[l] = x; barrier(CLK_LOCAL_MEM_FENCE);
    int rk = 0;
    for (int i = 0; i < 128; i++) { float y = v[i]; rk += (y > x) || (y == x && i < l); }
    if (rk < TK) sel[rk] = l;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < TK) pe[l] = exp(v[sel[l]] - v[sel[0]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < TK) { float ps = 0.0f; for (int k = 0; k < TK; k++) ps += pe[k]; idx[l] = sel[l]; wt[l] = pe[l] / ps * pes[sel[l]]; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gprep(__global const float* p, __global const float* qw, __global const float* kw, __global float* Q, __global KT* kc, __global KT* vc,
                    int pos, float theta, int NH, int KH, int HD, int NR, int VOFF, int CAP, float eps) {
    __local float r[64]; __local float tot; __local float sh[512];
    int g = get_group_id(0), l = get_local_id(0), NC = HD / 256, half_ = HD / 2;
    __global const float* src = g < NH ? p + g * HD : p + NH * HD + (g - NH) * HD;
    __global const float* w = g < NH ? qw : kw;
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
inline float d16(float16 a, float16 b) { return dot(a.lo.lo, b.lo.lo) + dot(a.lo.hi, b.lo.hi) + dot(a.hi.lo, b.hi.lo) + dot(a.hi.hi, b.hi.hi); }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gattn(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                    int L, int CAP, int KH, int GQ, int HD, int S, int CH, int WIN) {
    __local float sm[8], sl[8], so[8 * 512];
    int h = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0);
    int lo = max(0, L - WIN), n = L - lo, j0 = s * CH; if (j0 >= n) return;
    int j1 = min(n, j0 + CH), kh = h / GQ, NC = HD / 256;
    float16 q0 = vload16(0, Q + h * HD + l * 16), q1 = NC > 1 ? vload16(0, Q + h * HD + 256 + l * 16) : (float16)(0.0f);
    float m = -INFINITY, su = 0.0f; float16 o0 = (float16)(0.0f), o1 = (float16)(0.0f);
    for (int jj = j0 + sg; jj < j1; jj += 8) {
        size_t j = (size_t)((lo + jj) % CAP);
        __global const KT* kr = kc + (j * KH + kh) * HD; __global const KT* vr = vc + (j * KH + kh) * HD;
        float c = d16(q0, KLD16(0, kr + l * 16));
        if (NC > 1) c += d16(q1, KLD16(0, kr + 256 + l * 16));
        c = sub_group_reduce_add(c);
        float mn = fmax(m, c), a = exp(m - mn), b = exp(c - mn);
        su = su * a + b; o0 = o0 * a + KLD16(0, vr + l * 16) * b;
        if (NC > 1) o1 = o1 * a + KLD16(0, vr + 256 + l * 16) * b;
        m = mn;
    }
    if (l == 0) { sm[sg] = m; sl[sg] = su; }
    vstore16(o0, 0, so + sg * 512 + l * 16); if (NC > 1) vstore16(o1, 0, so + sg * 512 + 256 + l * 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    float M = -INFINITY; for (int i = 0; i < 8; i++) M = fmax(M, sm[i]);
    size_t pb = (size_t)h * S + s;
    float tl = 0.0f; for (int i = 0; i < 8; i++) tl += sm[i] == -INFINITY ? 0.0f : sl[i] * exp(sm[i] - M);
    for (int d = lid; d < HD; d += 128) {
        float ov = 0.0f; for (int i = 0; i < 8; i++) ov += sm[i] == -INFINITY ? 0.0f : so[i * 512 + d] * exp(sm[i] - M);
        PO[pb * 512 + d] = ov;
    }
    if (lid == 0) { PM[pb] = M; PL[pb] = tl; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gattn2(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                     int L, int CAP, int KH, int GQ, int HD, int S, int CH, int WIN) {
    __local float sm[8], sl[8], so[8 * 512];
    int kh = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0);
    int lo = max(0, L - WIN), n = L - lo, j0 = s * CH; if (j0 >= n) return;
    int j1 = min(n, j0 + CH), NC = HD / 256, NS = 8 / GQ, g = sg % GQ, ps = sg / GQ, h = kh * GQ + g;
    float16 q0 = vload16(0, Q + h * HD + l * 16), q1 = NC > 1 ? vload16(0, Q + h * HD + 256 + l * 16) : (float16)(0.0f);
    float m = -INFINITY, su = 0.0f; float16 o0 = (float16)(0.0f), o1 = (float16)(0.0f);
    for (int jj = j0 + ps; jj < j1; jj += NS) {
        size_t j = (size_t)((lo + jj) % CAP);
        __global const KT* kr = kc + (j * KH + kh) * HD; __global const KT* vr = vc + (j * KH + kh) * HD;
        float c = d16(q0, KLD16(0, kr + l * 16));
        if (NC > 1) c += d16(q1, KLD16(0, kr + 256 + l * 16));
        c = sub_group_reduce_add(c);
        float mn = fmax(m, c), a = exp(m - mn), b = exp(c - mn);
        su = su * a + b; o0 = o0 * a + KLD16(0, vr + l * 16) * b;
        if (NC > 1) o1 = o1 * a + KLD16(0, vr + 256 + l * 16) * b;
        m = mn;
    }
    if (l == 0) { sm[sg] = m; sl[sg] = su; }
    vstore16(o0, 0, so + sg * 512 + l * 16); if (NC > 1) vstore16(o1, 0, so + sg * 512 + 256 + l * 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int idx = lid; idx < GQ * HD; idx += 128) {
        int gg = idx / HD, d = idx - gg * HD;
        float M = -INFINITY; for (int t = 0; t < NS; t++) M = fmax(M, sm[gg + GQ * t]);
        float ov = 0.0f, tl = 0.0f;
        for (int t = 0; t < NS; t++) { int i = gg + GQ * t; if (sm[i] != -INFINITY) { float w = exp(sm[i] - M); ov += so[i * 512 + d] * w; tl += sl[i] * w; } }
        size_t pb = (size_t)(kh * GQ + gg) * S + s;
        PO[pb * 512 + d] = ov;
        if (d == 0) { PM[pb] = M; PL[pb] = tl; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gattn3g(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                      int L, int CAP, int KH, int GQ_, int HD_, int S, int CH, int WIN) {
    __local float qs[4096], ps[8 * 128], st[24];
    int kh = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0);
    int lo = max(0, L - WIN), n = L - lo, j0 = s * CH; if (j0 >= n) return;
    int j1 = min(n, j0 + CH);
    for (int i = lid; i < 8 * 512; i += 128) qs[i] = Q[kh * 8 * 512 + i];
    if (lid < 8) { st[lid] = -INFINITY; st[8 + lid] = 0.0f; }
    float o[8][4];
    for (int h = 0; h < 8; h++) for (int e = 0; e < 4; e++) o[h][e] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t0 = j0; t0 < j1; t0 += 128) {
        int jj = t0 + sg * 16 + l; float acc[8];
        for (int h = 0; h < 8; h++) acc[h] = 0.0f;
        if (jj < j1) {
            __global const KT* kr = kc + ((size_t)((lo + jj) % CAP) * KH + kh) * 512;
            for (int c = 0; c < 512; c += 16) {
                float16 k = KLD16(0, kr + c);
                for (int h = 0; h < 8; h++) acc[h] += d16(vload16(0, qs + h * 512 + c), k);
            }
        }
        for (int h = 0; h < 8; h++) ps[h * 128 + sg * 16 + l] = jj < j1 ? acc[h] : -INFINITY;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg < 8) {
            float v[8], mx = -INFINITY;
            for (int i = 0; i < 8; i++) { v[i] = ps[sg * 128 + i * 16 + l]; mx = fmax(mx, v[i]); }
            float Mo = st[sg], Mn = fmax(Mo, sub_group_reduce_max(mx)), al = Mo == -INFINITY ? 0.0f : exp(Mo - Mn), su = 0.0f;
            for (int i = 0; i < 8; i++) { float pp = v[i] == -INFINITY ? 0.0f : exp(v[i] - Mn); ps[sg * 128 + i * 16 + l] = pp; su += pp; }
            su = sub_group_reduce_add(su);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            if (l == 0) { st[sg] = Mn; st[8 + sg] = st[8 + sg] * al + su; st[16 + sg] = al; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int h = 0; h < 8; h++) { float al = st[16 + h]; for (int e = 0; e < 4; e++) o[h][e] *= al; }
        int nt = min(128, j1 - t0), i = 0;
        for (; i + 4 <= nt; i += 4) {
            float4 v0 = KLD4(lid, vc + ((size_t)((lo + t0 + i) % CAP) * KH + kh) * 512);
            float4 v1 = KLD4(lid, vc + ((size_t)((lo + t0 + i + 1) % CAP) * KH + kh) * 512);
            float4 v2 = KLD4(lid, vc + ((size_t)((lo + t0 + i + 2) % CAP) * KH + kh) * 512);
            float4 v3 = KLD4(lid, vc + ((size_t)((lo + t0 + i + 3) % CAP) * KH + kh) * 512);
            for (int h = 0; h < 8; h++) {
                float4 a = vload4(0, o[h]); __local const float* pp = ps + h * 128 + i;
                a += pp[0] * v0 + pp[1] * v1 + pp[2] * v2 + pp[3] * v3; vstore4(a, 0, o[h]);
            }
        }
        for (; i < nt; i++) {
            float4 v0 = KLD4(lid, vc + ((size_t)((lo + t0 + i) % CAP) * KH + kh) * 512);
            for (int h = 0; h < 8; h++) { float4 a = vload4(0, o[h]); a += ps[h * 128 + i] * v0; vstore4(a, 0, o[h]); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int h = 0; h < 8; h++) {
        size_t pb = (size_t)(kh * 8 + h) * S + s;
        for (int e = 0; e < 4; e++) PO[pb * 512 + lid * (4) + e] = o[h][e];
        if (lid == 0) { PM[pb] = st[h]; PL[pb] = st[8 + h]; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gattn3s(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                      int L, int CAP, int KH, int GQ_, int HD_, int S, int CH, int WIN) {
    __local float qs[512], ps[2 * 128], st[24];
    int kh = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0);
    int lo = max(0, L - WIN), n = L - lo, j0 = s * CH; if (j0 >= n) return;
    int j1 = min(n, j0 + CH);
    for (int i = lid; i < 2 * 256; i += 128) qs[i] = Q[kh * 2 * 256 + i];
    if (lid < 2) { st[lid] = -INFINITY; st[8 + lid] = 0.0f; }
    float o[2][2];
    for (int h = 0; h < 2; h++) for (int e = 0; e < 2; e++) o[h][e] = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t0 = j0; t0 < j1; t0 += 128) {
        int jj = t0 + sg * 16 + l; float acc[2];
        for (int h = 0; h < 2; h++) acc[h] = 0.0f;
        if (jj < j1) {
            __global const KT* kr = kc + ((size_t)((lo + jj) % CAP) * KH + kh) * 256;
            for (int c = 0; c < 256; c += 16) {
                float16 k = KLD16(0, kr + c);
                for (int h = 0; h < 2; h++) acc[h] += d16(vload16(0, qs + h * 256 + c), k);
            }
        }
        for (int h = 0; h < 2; h++) ps[h * 128 + sg * 16 + l] = jj < j1 ? acc[h] : -INFINITY;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sg < 2) {
            float v[8], mx = -INFINITY;
            for (int i = 0; i < 8; i++) { v[i] = ps[sg * 128 + i * 16 + l]; mx = fmax(mx, v[i]); }
            float Mo = st[sg], Mn = fmax(Mo, sub_group_reduce_max(mx)), al = Mo == -INFINITY ? 0.0f : exp(Mo - Mn), su = 0.0f;
            for (int i = 0; i < 8; i++) { float pp = v[i] == -INFINITY ? 0.0f : exp(v[i] - Mn); ps[sg * 128 + i * 16 + l] = pp; su += pp; }
            su = sub_group_reduce_add(su);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            if (l == 0) { st[sg] = Mn; st[8 + sg] = st[8 + sg] * al + su; st[16 + sg] = al; }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int h = 0; h < 2; h++) { float al = st[16 + h]; for (int e = 0; e < 2; e++) o[h][e] *= al; }
        int nt = min(128, j1 - t0), i = 0;
        for (; i + 4 <= nt; i += 4) {
            float2 v0 = KLD2(lid, vc + ((size_t)((lo + t0 + i) % CAP) * KH + kh) * 256);
            float2 v1 = KLD2(lid, vc + ((size_t)((lo + t0 + i + 1) % CAP) * KH + kh) * 256);
            float2 v2 = KLD2(lid, vc + ((size_t)((lo + t0 + i + 2) % CAP) * KH + kh) * 256);
            float2 v3 = KLD2(lid, vc + ((size_t)((lo + t0 + i + 3) % CAP) * KH + kh) * 256);
            for (int h = 0; h < 2; h++) {
                float2 a = vload2(0, o[h]); __local const float* pp = ps + h * 128 + i;
                a += pp[0] * v0 + pp[1] * v1 + pp[2] * v2 + pp[3] * v3; vstore2(a, 0, o[h]);
            }
        }
        for (; i < nt; i++) {
            float2 v0 = KLD2(lid, vc + ((size_t)((lo + t0 + i) % CAP) * KH + kh) * 256);
            for (int h = 0; h < 2; h++) { float2 a = vload2(0, o[h]); a += ps[h * 128 + i] * v0; vstore2(a, 0, o[h]); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int h = 0; h < 2; h++) {
        size_t pb = (size_t)(kh * 2 + h) * S + s;
        for (int e = 0; e < 2; e++) PO[pb * 512 + lid * (2) + e] = o[h][e];
        if (lid == 0) { PM[pb] = st[h]; PL[pb] = st[8 + h]; }
    }
}
__kernel void gcomb(__global const float* PM, __global const float* PL, __global const float* PO, __global float* A, int L, int CAP, int HD, int S, int CH, int WIN) {
    int h = get_group_id(0), l = get_local_id(0), n = L - max(0, L - WIN), ns = (n + CH - 1) / CH;
    size_t base = (size_t)h * S;
    float M = -INFINITY; for (int s = 0; s < ns; s++) M = fmax(M, PM[base + s]);
    float tl = 0.0f; for (int s = 0; s < ns; s++) tl += PL[base + s] * exp(PM[base + s] - M);
    for (int d = l; d < HD; d += 256) {
        float o = 0.0f; for (int s = 0; s < ns; s++) o += PO[(base + s) * 512 + d] * exp(PM[base + s] - M);
        A[h * HD + d] = o / tl;
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void postattn(__global const float* o, __global const float* wpa, __global float* h, __global const float* w1, __global const float* w3,
                       __global float* t1, __global float* t3, __global float* t2, int H, float eps) {
    __local float r[64]; __local float tot; __local float hh[4096];
    int l = get_local_id(0); float s = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = o[i]; s += v * v; }
    float k = rsqrt(bsum(s, r, &tot) / H + eps), s2 = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = h[i] + o[i] * k * wpa[i]; hh[i] = v; h[i] = v; s2 += v * v; }
    float k2 = rsqrt(bsum(s2, r, &tot) / H + eps);
    for (int i = l; i < H; i += 1024) { float v = hh[i] * k2; t1[i] = v * w1[i]; t3[i] = v * w3[i]; t2[i] = v; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ffend2(__global const float* d1, __global const float* e, __global const float* w1, __global const float* w2, __global const float* w,
                     __global const float* ls, __global float* h, __global const float* wn, __global float* tn, int H, float eps) {
    __local float r[64]; __local float tot; __local float t[4096];
    int l = get_local_id(0); float s1 = 0.0f, s2 = 0.0f;
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
__kernel void rroute(__global const float* w, __global const float* x, __global volatile float* lg, __global const float* pes, __global int* idx, __global float* wt,
                     __global volatile int* cnt, int E, int K, int TK) {
    __local float r[8]; __local float v[128]; __local int last; __local int sel[16]; __local float pe[16];
    int e = get_group_id(0), l = get_local_id(0);
    float acc = 0.0f;
    for (int k = l * 4; k < K; k += 512) acc += dot(vload4(0, w + (size_t)e * K + k), vload4(0, x + k));
    acc = sub_group_reduce_add(acc);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) {
        float t = 0.0f; for (int i = 0; i < 8; i++) t += r[i];
        lg[e] = t; mem_fence(CLK_GLOBAL_MEM_FENCE);
        last = atomic_inc(cnt) == E - 1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (!last) return;
    mem_fence(CLK_GLOBAL_MEM_FENCE);
    float xv = l < E ? lg[l] : -INFINITY; v[l] = xv; barrier(CLK_LOCAL_MEM_FENCE);
    int rk = 0;
    for (int i = 0; i < 128; i += 4) { float4 y = vload4(0, v + i);
        rk += (y.x > xv) || (y.x == xv && i < l); rk += (y.y > xv) || (y.y == xv && i + 1 < l); rk += (y.z > xv) || (y.z == xv && i + 2 < l); rk += (y.w > xv) || (y.w == xv && i + 3 < l); }
    if (rk < TK) sel[rk] = l;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < TK) pe[l] = exp(v[sel[l]] - v[sel[0]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l < TK) { float ps = 0.0f; for (int k = 0; k < TK; k++) ps += pe[k]; idx[l] = sel[l]; wt[l] = pe[l] / ps * pes[sel[l]]; }
    if (l == 0) cnt[0] = 0;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rlogit(__global const float* w, __global const float* x, __global float* lg, int K) {
    __local float r[8];
    int e = get_group_id(0), l = get_local_id(0);
    float acc = 0.0f;
    for (int k = l * 4; k < K; k += 512) acc += dot(vload4(0, w + (size_t)e * K + k), vload4(0, x + k));
    acc = sub_group_reduce_add(acc);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) { float t = 0.0f; for (int i = 0; i < 8; i++) t += r[i]; lg[e] = t; }
}
__kernel void rtop(__global const float* lg, __global const float* pes, __global int* idx, __global float* wt, int TK) {
    __local float v[128]; __local int rk[128]; __local int sel[16]; __local float pe[16];
    int L = get_local_id(0), e = L >> 3, j = L & 7;
    if (L < 128) { v[L] = lg[L]; rk[L] = 0; }
    barrier(CLK_LOCAL_MEM_FENCE);
    float x = v[e]; int c = 0;
    for (int i = j * 16; i < j * 16 + 16; i += 4) {
        float4 y = vload4(0, v + i);
        c += (y.x > x) || (y.x == x && i < e); c += (y.y > x) || (y.y == x && i + 1 < e);
        c += (y.z > x) || (y.z == x && i + 2 < e); c += (y.w > x) || (y.w == x && i + 3 < e);
    }
    atomic_add(&rk[e], c);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (L < 128 && rk[L] < TK) sel[rk[L]] = L;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (L < TK) pe[L] = exp(v[sel[L]] - v[sel[0]]);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (L < TK) { float ps = 0.0f; for (int k = 0; k < TK; k++) ps += pe[k]; idx[L] = sel[L]; wt[L] = pe[L] / ps * pes[sel[L]]; }
}
__kernel void settok(__global const int* ids, int i, __global int* tok) { tok[0] = ids[i]; }
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void nll(__global const float* lg, int V, float cap, __global const int* ids, int tgt_i, __global float* out, int step) {
    __local float r[64]; __local float tot; __local float rv[1024]; __local int ri[1024];
    int l = get_local_id(0); float mx = -INFINITY; int mi = 0;
    for (int i = l; i < V; i += 1024) { float v = cap * tanh(lg[i] / cap); if (v > mx) { mx = v; mi = i; } }
    rv[l] = mx; ri[l] = mi; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 512; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    float M = rv[0]; int am = ri[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.0f; for (int i = l; i < V; i += 1024) s += exp(cap * tanh(lg[i] / cap) - M);
    s = bsum(s, r, &tot);
    if (l == 0) { int t = ids[tgt_i]; float lt = cap * tanh(lg[t] / cap); out[2 * step] = M + log(s) - lt; out[2 * step + 1] = am == t ? 1.0f : 0.0f; }
}
__kernel void argmax1(__global const float* v, int N, __global float* pv, __global int* pi) {
    __local float rv[256]; __local int ri[256];
    int l = get_local_id(0), g = get_group_id(0); float bv = -INFINITY; int bi = 0;
    for (int i = g * 4096 + l; i < min(N, (g + 1) * 4096); i += 256) { float t = v[i]; if (t > bv) { bv = t; bi = i; } }
    rv[l] = bv; ri[l] = bi; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) { pv[g] = rv[0]; pi[g] = ri[0]; }
}
__kernel void argmax2(__global const float* pv, __global const int* pi, int P, __global int* tok, __global int* outs, int step) {
    __local float rv[256]; __local int ri[256]; int l = get_local_id(0);
    rv[l] = l < P ? pv[l] : -INFINITY; ri[l] = l < P ? pi[l] : 0; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) { tok[0] = ri[0]; outs[step] = ri[0]; }
}
"""


class Model:
    def __init__(self, path, tmax=None, prof=False):
        self.path = path
        meta = json.load(open(path[:-4] + ".json"))
        self.meta, self.cfg, self.tm = meta, meta["config"], meta["tensors"]
        nf = meta.get("norm_form")
        if nf is not None and any(v != "w" for v in nf.values()):
            raise NotImplementedError(f"the Gemma 4 engine supports norm_form 'w' only: {nf}")
        c = self.cfg
        self.H, self.F, self.V = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
        self.NE, self.FE, self.TK = c["num_experts"], c["moe_intermediate_size"], c["top_k_experts"]
        self.NH = c["num_attention_heads"]; self.eps = float(c["rms_norm_eps"]); self.cap = float(c["final_logit_softcapping"])
        self.TM = tmax or int(knob("G4_TMAX", "4096")); self.WIN = c["sliding_window"]
        self.CH = int(knob("G4_CH", "256")); self.gqa = int(knob("G4_GQA", "2"))
        dch = 128 if self.gqa == 2 else 32; self.tile_s = int(knob("G4_TILES", "0"))
        self.CHG, self.CHS = int(knob("G4_CHG", dch)), int(knob("G4_CHS", 128 if self.gqa == 2 else 32))
        self.SM = -(-self.TM // (min(self.CHG, self.CHS) if self.gqa else self.CH))
        from ..device import device as _dev
        # Prefill chunk: 2048 tokens gives the experts about 4x more rows per weight pass (Arc 140T prompt processing
        # +40%) but needs about 1.2 GiB more (larger sliding-window ring and work buffers); only with a 20 GiB+ device.
        self.PCH = int(knob("G4_PCH", "2048" if _dev().global_mem_size >= 20 * 2**30 else "512"))
        rp = c["rope_parameters"]
        self.lcfg = []
        for lt in c["layer_types"]:
            if lt == "sliding_attention":
                self.lcfg.append(dict(KH=c["num_key_value_heads"], HD=c["head_dim"], NR=c["head_dim"] // 2, theta=float(rp[lt]["rope_theta"]), CAP=self.WIN + self.PCH, WIN=self.WIN, kv=False))
            else:
                hd = c["global_head_dim"]; nr = int(hd * rp[lt]["partial_rotary_factor"]) // 2
                self.lcfg.append(dict(KH=c["num_global_key_value_heads"], HD=hd, NR=nr, theta=float(rp[lt]["rope_theta"]), CAP=self.TM, WIN=self.TM, kv=True))
        dev = K2.device(); self.ctx = cl.Context([dev])
        self.prof = prof; self.ev = []; self.lab = {}; self.kbuf = {}; self.evk = []
        self.q = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) if prof else cl.CommandQueue(self.ctx)
        self.R = {k: int(v) for k, v in (x.split("=") for x in knob("G4_R", "proj=2,out=1,gu=2,down=1,egu=4,lm_head=2").split(","))}
        self.S = {k: int(v) for k, v in (x.split("=") for x in knob("G4_S", "proj=1,out=1,gu=1,down=1,egu=1,edown=1").split(","))}
        opt = [f"-DG={meta['G']}"]
        self.prgs = {r: cl.Program(self.ctx, GSRC).build(options=opt + [f"-DR={r}"]) for r in sorted(set(self.R.values()) | {1})}
        self.prg = self.prgs[1]
        try:
            self.prg3 = cl.Program(self.ctx, GSRC).build(options=opt + ["-DR=1", "-cl-intel-256-GRF-per-thread"])
        except cl.Error as e:
            print(f"[xeq] 256-GRF build failed, using the default build for tiled attention: {str(e).splitlines()[0][:120]}", flush=True)
            self.prg3 = self.prgs[1]
        mf = cl.mem_flags; self.buf = {}
        with open(path, "rb") as fh:
            for k, t in self.tm.items():
                fh.seek(t["off"]); a = np.frombuffer(fh.read(t["nbytes"]), np.uint8)
                if t["kind"] == "fac":
                    a = np.concatenate([a, np.zeros(16, np.uint8)])
                self.buf[k] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        f32 = lambda n: cl.Buffer(self.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        H, F, NH = self.H, self.F, self.NH
        self.h, self.t, self.t2, self.o, self.d1, self.e = f32(H), f32(H), f32(H), f32(H), f32(H), f32(H)
        self.P = f32(max(NH * L["HD"] + 2 * L["KH"] * L["HD"] for L in self.lcfg))
        self.Q, self.A = f32(NH * 512), f32(NH * 512)
        self.GU, self.M = f32(2 * F), f32(F)
        self.rl, self.idx, self.wt = f32(self.NE), f32(self.TK), f32(self.TK)
        self.EGU, self.EM = f32(self.TK * 2 * self.FE), f32(self.TK * self.FE)
        self.PM, self.PL, self.PO = f32(NH * self.SM), f32(NH * self.SM), f32(NH * self.SM * 512)
        self.logit = f32(self.V); self.npart = (self.V + 4095) // 4096
        self.pv, self.pi = f32(self.npart), f32(self.npart)
        self.tok = f32(1); self.outs = f32(self.TM); self.ids = f32(self.TM + 1); self.NLL = f32(2 * self.TM)
        kvb = lambda n: cl.Buffer(self.ctx, mf.READ_WRITE, int(n) * KVB)
        self.kv = [(kvb(L["CAP"] * L["KH"] * L["HD"]), kvb(L["CAP"] * L["KH"] * L["HD"])) for L in self.lcfg]
        self.t3 = f32(H); self.cnt = f32(1); cl.enqueue_fill_buffer(self.q, self.cnt, np.int32(0), 0, 4)
        self.fuse = int(knob("G4_FUSE", "1"))
        self.route2 = int(knob("G4_ROUTE", "2")) == 2
        with open(path, "rb") as fh:
            for i in range(len(self.lcfg)):
                p = f"layers.{i}."
                rd = lambda k: (fh.seek(self.tm[k]["off"]), np.frombuffer(fh.read(self.tm[k]["nbytes"]), np.float32).reshape(self.tm[k]["shape"]))[1]
                wf = (rd(p + "r_proj") * (rd(p + "r_scale") * np.float32(H ** -0.5))[None, :]).astype(np.float32)
                self.buf[p + "r_projF"] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(wf))
        self.build()

    def reset(self):
        for a, b in self.kv:
            for x in (a, b):
                cl.enqueue_fill_buffer(self.q, x, np.uint32(0), 0, x.size)
        self.q.finish()

    def K(self, name, g, l, *args, label=None, prg=None):
        k = cl.Kernel(prg or self.prg, name); k.set_args(*args); self.lab[id(k)] = label or name
        self.kbuf[id(k)] = [a for a in args if isinstance(a, cl.Buffer)]
        return [k, g, l]

    def lin(self, name, x, y, add, kind):
        """XeFP4 or INT8 (nonexp8) linear layer."""
        I = np.int32
        if name + ".q" in self.tm:
            N, Kd = self.tm[name + ".q"]["shape"]; R = self.R.get("lm_head" if kind == "lm_head" else "i8", 1)
            return self.K("igemv", (16, -(-(N // R) // 8) * 8), (16, 8), self.buf[name + ".q"], self.buf[name + ".s"], x, y, I(N), I(Kd), I(add), label="i8." + kind, prg=self.prgs[R])
        t = self.tm[name + ".codes"]; N, Kd = t["shape"][0], t["shape"][1] * 2; R = self.R.get(kind, 1); S = self.S.get(kind, 1)
        assert N % R == 0, (name, N, R)
        if S > 1:
            return self.K("xgemvS", (16, -(-(N // R * S) // 8) * 8), (16, 8), self.buf[name + ".codes"], self.buf[name + ".fac"], self.buf[name + ".rs"], x, y,
                          I(N), I(Kd), I(add), self.idx, I(0), I(S), label="xe." + kind, prg=self.prgs[R])
        return self.K("xgemv", (16, -(-(N // R) // 8) * 8), (16, 8), self.buf[name + ".codes"], self.buf[name + ".fac"], self.buf[name + ".rs"], x, y,
                      I(N), I(Kd), I(add), self.idx, I(0), label="xe." + kind, prg=self.prgs[R])

    def egu(self, p, x):
        I = np.int32; R = self.R["egu"]; S = self.S.get("egu", 1); NEr = 2 * self.FE; b = self.buf
        if S > 1:
            return self.K("xgemvS", (16, -(-(self.TK * NEr // R * S) // 8) * 8), (16, 8), b[p + "egu.codes"], b[p + "egu.fac"], b[p + "egu.rs"], x, self.EGU,
                          I(self.TK), I(self.H), I(0), self.idx, I(NEr), I(S), label="xe.egu", prg=self.prgs[R])
        return self.K("xgemv", (16, -(-(self.TK * NEr // R) // 8) * 8), (16, 8), b[p + "egu.codes"], b[p + "egu.fac"], b[p + "egu.rs"], x, self.EGU,
                      I(self.TK), I(self.H), I(0), self.idx, I(NEr), label="xe.egu", prg=self.prgs[R])

    def edown(self, p):
        I = np.int32; S = self.S.get("edown", 1); b = self.buf; H = self.H
        if S > 1:
            return self.K("xdownS", (16, -(-H * S // 8) * 8), (16, 8), b[p + "edown.codes"], b[p + "edown.fac"], b[p + "edown.rs"], self.EM, self.e,
                          I(H), I(self.FE), self.idx, self.wt, I(self.TK), I(S), label="xe.edown")
        return self.K("xdown", (16, -(-H // 8) * 8), (16, 8), b[p + "edown.codes"], b[p + "edown.fac"], b[p + "edown.rs"], self.EM, self.e,
                      I(H), I(self.FE), self.idx, self.wt, I(self.TK), label="xe.edown")

    def build(self):
        H, F, I, eps, NH = self.H, self.F, np.int32, np.float32(self.eps), self.NH
        u = int(np.array([np.sqrt(H)], np.float32).view(np.uint32)[0])
        nrm = float(np.array([(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000], np.uint32).view(np.float32)[0])   # BF16(√H) = 53.0
        self.head = [self.K("embed8", (-(-H // 256) * 256,), (256,), self.buf["lm_head.q"], self.buf["lm_head.s"], self.tok, self.h, I(H), np.float32(nrm))]
        G1 = lambda x, w, y: self.K("gnorm", (1024,), (1024,), x, w, w, y, I(H), eps, np.float32(1.0), I(1), I(0))
        self.layers = []
        for i, L in enumerate(self.lcfg):
            p = f"layers.{i}."; kc, vc = self.kv[i]; KH, HD = L["KH"], L["HD"]
            ks = [G1(self.h, self.buf[p + "in_norm"], self.t), self.lin(p + "proj", self.t, self.P, 0, "proj")]
            voff = NH * HD if L["kv"] else NH * HD + KH * HD
            kp = self.K("gprep", ((NH + KH) * 256,), (256,), self.P, self.buf[p + "q_norm"], self.buf[p + "k_norm"], self.Q, kc, vc,
                        I(0), np.float32(L["theta"]), I(NH), I(KH), I(HD), I(L["NR"]), I(voff), I(L["CAP"]), eps)
            if self.gqa == 2 and (L["kv"] or self.tile_s):
                assert (NH // KH, HD) in ((8, 512), (2, 256))
                L["CHD"], L["NG"] = (self.CHG if L["kv"] else self.CHS), KH
                ka = self.K("gattn3g" if HD == 512 else "gattn3s", (KH * 128, 1), (128, 1), self.Q, kc, vc, self.PM, self.PL, self.PO, I(1), I(L["CAP"]), I(KH), I(NH // KH), I(HD), I(self.SM), I(L["CHD"]), I(L["WIN"]), label="gattn.global" if L["kv"] else "gattn.sliding", prg=self.prg3)
            elif self.gqa == 1:
                assert 8 % (NH // KH) == 0
                L["CHD"], L["NG"] = (self.CHG if L["kv"] else self.CHS), KH
                ka = self.K("gattn2", (KH * 128, 1), (128, 1), self.Q, kc, vc, self.PM, self.PL, self.PO, I(1), I(L["CAP"]), I(KH), I(NH // KH), I(HD), I(self.SM), I(L["CHD"]), I(L["WIN"]), label="gattn.global" if L["kv"] else "gattn.sliding")
            else:
                L["CHD"], L["NG"] = (self.CHS if self.gqa == 2 else self.CH), NH
                ka = self.K("gattn", (NH * 128, 1), (128, 1), self.Q, kc, vc, self.PM, self.PL, self.PO, I(1), I(L["CAP"]), I(KH), I(NH // KH), I(HD), I(self.SM), I(L["CHD"]), I(L["WIN"]), label="gattn.global" if L["kv"] else "gattn.sliding")
            kb = self.K("gcomb", (NH * 256,), (256,), self.PM, self.PL, self.PO, self.A, I(1), I(L["CAP"]), I(HD), I(self.SM), I(L["CHD"]), I(L["WIN"]), label="gcomb.global" if L["kv"] else "gcomb.sliding")
            ks += [kp, ka, kb, self.lin(p + "out", self.A, self.o, 0, "out"),
                   self.K("normadd", (1024,), (1024,), self.o, self.buf[p + "post_attn"], self.h, I(H), eps)]
            ks += [G1(self.h, self.buf[p + "pre_ff"], self.t), self.lin(p + "gu", self.t, self.GU, 0, "gu"),
                   self.K("geglu", (-(-F // 256) * 256,), (256,), self.GU, self.M, I(F), I(1)), self.lin(p + "down", self.M, self.d1, 0, "down")]
            # MoE
            ks += [self.K("gnorm", (1024,), (1024,), self.h, self.buf[p + "r_scale"], self.buf[p + "r_scale"], self.t2, I(H), eps, np.float32(H ** -0.5), I(0), I(1)),
                   self.K("fgemv", (16, -(-self.NE // 8) * 8), (16, 8), self.buf[p + "r_proj"], self.t2, self.rl, I(self.NE), I(H), label="router"),
                   self.K("route", (128,), (128,), self.rl, self.buf[p + "r_pes"], self.idx, self.wt, I(self.NE), I(self.TK)),
                   G1(self.h, self.buf[p + "pre_ff2"], self.t)]
            R = self.R["egu"]; NEr = 2 * self.FE
            ks += [self.K("xgemv", (16, -(-(self.TK * NEr // R) // 8) * 8), (16, 8), self.buf[p + "egu.codes"], self.buf[p + "egu.fac"], self.buf[p + "egu.rs"], self.t, self.EGU,
                          I(self.TK), I(H), I(0), self.idx, I(NEr), label="xe.egu", prg=self.prgs[R]),
                   self.K("geglu", (-(-self.TK * self.FE // 256) * 256,), (256,), self.EGU, self.EM, I(self.FE), I(self.TK), label="geglu.e"),
                   self.K("xdown", (16, -(-H // 8) * 8), (16, 8), self.buf[p + "edown.codes"], self.buf[p + "edown.fac"], self.buf[p + "edown.rs"], self.EM, self.e,
                          I(H), I(self.FE), self.idx, self.wt, I(self.TK), label="xe.edown"),
                   self.K("ffend", (1024,), (1024,), self.d1, self.e, self.buf[p + "post_ff1"], self.buf[p + "post_ff2"], self.buf[p + "post_ff"], self.buf[p + "ls"], self.h, I(H), eps)]
            if self.fuse:
                nxt = self.buf[f"layers.{i + 1}.in_norm"] if i + 1 < len(self.lcfg) else self.buf["norm"]
                ks = [self.lin(p + "proj", self.t, self.P, 0, "proj"), kp, ka, kb, self.lin(p + "out", self.A, self.o, 0, "out"),
                      self.K("postattn", (1024,), (1024,), self.o, self.buf[p + "post_attn"], self.h, self.buf[p + "pre_ff"], self.buf[p + "pre_ff2"], self.t, self.t3, self.t2, I(H), eps),
                      self.lin(p + "gu", self.t, self.GU, 0, "gu"), self.K("geglu", (-(-F // 256) * 256,), (256,), self.GU, self.M, I(F), I(1)),
                      self.lin(p + "down", self.M, self.d1, 0, "down"),
                      *([self.K("rlogit", (self.NE * 128,), (128,), self.buf[p + "r_projF"], self.t2, self.rl, I(H), label="rlogit"),
                         self.K("rtop", (1024,), (1024,), self.rl, self.buf[p + "r_pes"], self.idx, self.wt, I(self.TK), label="rtop")] if self.route2 else
                        [self.K("rroute", (self.NE * 128,), (128,), self.buf[p + "r_projF"], self.t2, self.rl, self.buf[p + "r_pes"], self.idx, self.wt, self.cnt, I(self.NE), I(H), I(self.TK))]),
                      self.egu(p, self.t3),
                      self.K("geglu", (-(-self.TK * self.FE // 256) * 256,), (256,), self.EGU, self.EM, I(self.FE), I(self.TK), label="geglu.e"),
                      self.edown(p),
                      self.K("ffend2", (1024,), (1024,), self.d1, self.e, self.buf[p + "post_ff1"], self.buf[p + "post_ff2"], self.buf[p + "post_ff"], self.buf[p + "ls"], self.h,
                             nxt, self.t, I(H), eps)]
            self.layers.append((ks, kp, ka, kb, L))
        self.innorm = [G1(self.h, self.buf[f"layers.{i}.in_norm"], self.t) for i in range(len(self.lcfg))]
        if self.fuse:
            self.head.append(self.innorm[0]); self.tail = [self.lin("lm_head", self.t, self.logit, 0, "lm_head")]
        else:
            self.tail = [G1(self.h, self.buf["norm"], self.t), self.lin("lm_head", self.t, self.logit, 0, "lm_head")]
        self.am1 = self.K("argmax1", (self.npart * 256,), (256,), self.logit, I(self.V), self.pv, self.pi)
        self.am2 = self.K("argmax2", (256,), (256,), self.pv, self.pi, I(self.npart), self.tok, self.outs, I(0))
        self.st = self.K("settok", (1,), (1,), self.ids, I(0), self.tok)
        self.kw = self.K("nll", (1024,), (1024,), self.logit, I(self.V), np.float32(self.cap), self.ids, I(0), self.NLL, I(0))

    def run(self, kl):
        for k, g, l in kl:
            e = cl.enqueue_nd_range_kernel(self.q, k, g, l)
            if self.prof:
                self.ev.append((self.lab[id(k)], e)); self.evk.append(id(k))

    def step(self, pos, out_step=None, feed=None, nll_step=None):
        if feed is not None:
            self.st[0].set_arg(1, np.int32(feed)); self.run([self.st])
        self.run(self.head)
        for ks, kp, ka, kb, L in self.layers:
            n = min(pos + 1, L["WIN"])
            kp[0].set_arg(6, np.int32(pos)); ka[0].set_arg(6, np.int32(pos + 1)); ka[1] = (L["NG"] * 128, -(-n // L["CHD"])); kb[0].set_arg(4, np.int32(pos + 1))
            self.run(ks)
        self.q.flush()
        self.run(self.tail)
        if nll_step is not None:
            self.kw[0].set_arg(4, np.int32(feed + 1)); self.kw[0].set_arg(6, np.int32(nll_step)); self.run([self.kw])
        if out_step is not None:
            self.am2[0].set_arg(5, np.int32(out_step)); self.run([self.am1, self.am2])
        if pos % 64 == 63:
            self.q.finish()

    def load_ids(self, ids):
        a = np.asarray(ids, np.int32); cl.enqueue_copy(self.q, self.ids, a); self.q.finish()

    def generate(self, ids, n):
        self.reset(); self.load_ids(ids)
        t0 = time.perf_counter()
        for i in range(len(ids)):
            self.step(i, out_step=0 if i == len(ids) - 1 else None, feed=i)
        o = np.zeros(1, np.int32); cl.enqueue_copy(self.q, o, self.outs); self.q.finish()
        t1 = time.perf_counter()
        for j in range(1, n):
            self.step(len(ids) + j - 1, out_step=j)
        out = np.zeros(n, np.int32); cl.enqueue_copy(self.q, out, self.outs); self.q.finish()
        return out.tolist(), t1 - t0, time.perf_counter() - t1

    def rd(self, b, n, dt=np.float32):
        v = np.zeros(n, dt); cl.enqueue_copy(self.q, v, b); self.q.finish(); return v

    def rdkv(self, b, n):
        """Reads a KV cache buffer as float32 (fp16 or fp32 storage)."""
        return self.rd(b, n, np.float16 if KV16 else np.float32).astype(np.float32)
