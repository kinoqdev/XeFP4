"""Qwen3.5-9B XeFP4 decoder (OpenCL, batch 1).
32 layers: 24 linear-attention layers (gated delta rule) and 8 full-attention layers (every 4th layer).
One token is about 290 kernel launches. The next token id is passed on the GPU, so decoding needs no host
synchronisation: the host enqueues several tokens ahead.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pyopencl as cl

from .. import device as K2
from ..device import knob

SRC = r"""
#ifdef KV16
#define KT half
#define KLD16(o, p) vload_half16(o, p)
#define KLD4(o, p) vload_half4(o, p)
#define KLD1(o, p) vload_half(o, p)
#define KST(v, o, p) vstore_half(v, o, p)
#else
#define KT float
#define KLD16(o, p) vload16(o, p)
#define KLD4(o, p) vload4(o, p)
#define KLD1(o, p) ((p)[o])
#define KST(v, o, p) ((p)[o] = (v))
#endif

#ifdef COND
#define CARGS , __global const int* cf, int cw
#define CCHECK if (cw >= 0 && cf[0] != cw) return;
#else
#define CARGS
#define CCHECK
#endif
#ifdef POSB
#define PARGS , __global const int* pb
#define PADJ(x) (pb[0] + (x))
#else
#define PARGS
#define PADJ(x) (x)
#endif
#define UNR __attribute__((opencl_unroll_hint))
inline float bf(ushort u) { return as_float(((uint)u) << 16); }
inline float wsum(float v, __local float* r) {
    int l = get_local_id(0), n = get_local_size(0); r[l] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) { if (l < s) r[l] += r[l + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float o = r[0]; barrier(CLK_LOCAL_MEM_FENCE); return o;
}
inline float wmax(float v, __local float* r) {
    int l = get_local_id(0), n = get_local_size(0); r[l] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) { if (l < s) r[l] = fmax(r[l], r[l + s]); barrier(CLK_LOCAL_MEM_FENCE); }
    float o = r[0]; barrier(CLK_LOCAL_MEM_FENCE); return o;
}
inline float wsum16(float v, __local float* r) {
    int ns = get_local_size(0) / 16; float s = sub_group_reduce_add(v);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    float t = 0.0f; for (int i = 0; i < ns; i++) t += r[i];
    barrier(CLK_LOCAL_MEM_FENCE); return t;
}
inline float silu(float v) { return v / (1.0f + exp(-v)); }

__kernel void embed(__global const ushort* E, __global const int* tok, __global float* x, int H CARGS) {
    CCHECK
    int i = get_global_id(0); if (i < H) x[i] = bf(E[(size_t)tok[0] * H + i]);
}
#if XE8
__constant float ELUT[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
inline float edq(uint c) { float v = ELUT[c & 7]; return (c & 8) ? -v : v; }
#else
inline float edq(uint c) { return (float)((int)c - 8); }
#endif
__kernel void embed_xe(__global const uchar* codes, __global const uint* fac, __global const half* rs, __global const int* tok, __global float* x, int H CARGS) {
    CCHECK
    int i = get_global_id(0); if (i >= H) return;
    size_t n = tok[0]; int FW = (H / G) * 3 / 32, bo = (i / G) * 3;
    __global const uint* fr = fac + n * FW;
    ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
    uint by = codes[n * (H / 2) + i / 2], c = (i & 1) ? (by >> 4) : (by & 15u);
    x[i] = edq(c) * vload_half(n, rs) * 0.125f * (float)(((fw >> (bo & 31)) & 7) + 1);
}
__kernel void rmsnorm(__global const float* x, __global const float* w, __global float* y, int H, float eps, int plus1) {
    __local float r[256]; int l = get_local_id(0); float s = 0.0f;
    for (int i = l; i < H; i += 256) { float v = x[i]; s += v * v; }
    float k = rsqrt(wsum(s, r) / H + eps);
    for (int i = l; i < H; i += 256) y[i] = x[i] * k * (plus1 ? 1.0f + w[i] : w[i]);
}

#if XE8
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
#else
inline uint dlo(uint u) { return ((u & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u; }
inline uint dhi(uint u) { return (((u >> 4) & 0x0F0F0F0Fu) + 0x78787878u) ^ 0x80808080u; }
#endif
inline float dot8(uint u, float8 v) {
    return dot(convert_float4(as_char4(dlo(u))), (float4)(v.s0, v.s2, v.s4, v.s6)) + dot(convert_float4(as_char4(dhi(u))), (float4)(v.s1, v.s3, v.s5, v.s7));
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                   int N, int K, int add) {
    int row = get_global_id(1); if (row >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32;
    __global const uint* cr = codes + (size_t)row * (K / 8);
    __global const uint* fr = fac + (size_t)row * FW;
    float acc = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3; ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        float m = (float)(((fw >> (bo & 31)) & 7) + 1);
        float s = 0.0f;
        UNR for (int j = 0; j < G / 32; j++) {
            uint4 u = vload4(0, cr + b * (G / 8) + j * 4);
            __global const float* xp = x + b * G + j * 32;
            s += dot8(u.s0, vload8(0, xp)) + dot8(u.s1, vload8(1, xp)) + dot8(u.s2, vload8(2, xp)) + dot8(u.s3, vload8(3, xp));
        }
        acc += s * m;
    }
    acc = sub_group_reduce_add(acc);
    if (l == 0) { float v = acc * vload_half(row, rs) * 0.125f; y[row] = add ? y[row] + v : v; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv2(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                    int N, int K, int add CARGS) {
    CCHECK
    int r0 = get_global_id(1) * R; if (r0 >= N) return;
    int l = get_local_id(0), nb = K / G, FW = nb * 3 / 32;
    float acc[R];
    UNR for (int r = 0; r < R; r++) acc[r] = 0.0f;
    for (int b = l; b < nb; b += 16) {
        __global const float* xp = x + b * G;
        float8 xv[G / 8];
        UNR for (int j = 0; j < G / 8; j++) xv[j] = vload8(j, xp);
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        UNR for (int r = 0; r < R; r++) {
            size_t row = r0 + r;
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
        if (l == 0) { int row = r0 + r; v *= vload_half(row, rs) * 0.125f; y[row] = add ? y[row] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_i8r(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K CARGS) {
    CCHECK
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
    UNR for (int r = 0; r < R; r++) { float t = sub_group_reduce_add(acc[r]); if (l == 0) y[r0 + r] = t * sc[r0 + r]; }
}
__kernel void rdpeak(__global const uint4* a, __global uint* o, int n) {
    uint4 s = (uint4)(0); for (int i = get_global_id(0); i < n; i += get_global_size(0)) s ^= a[i];
    if ((s.x ^ s.y ^ s.z ^ s.w) == 0x12345678u) o[0] = 1;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gemv_i8(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K) {
    int row = get_global_id(1); if (row >= N) return;
    int l = get_local_id(0); __global const char* qr = q + (size_t)row * K; float acc = 0.0f;
    for (int k = l * 16; k < K; k += 256) {
        char16 a = vload16(0, qr + k); float16 v = vload16(0, x + k);
        acc += dot(convert_float4(a.s0123), v.s0123) + dot(convert_float4(a.s4567), v.s4567) + dot(convert_float4(a.s89ab), v.s89ab) + dot(convert_float4(a.scdef), v.scdef);
    }
    acc = sub_group_reduce_add(acc);
    if (l == 0) y[row] = acc * sc[row];
}
__kernel void argmax1(__global const float* v, int N, __global float* pv, __global int* pi CARGS) {
    CCHECK
    __local float rv[256]; __local int ri[256];
    int l = get_local_id(0), g = get_group_id(0); float bv = -INFINITY; int bi = 0;
    for (int i = g * 4096 + l; i < min(N, (g + 1) * 4096); i += 256) { float t = v[i]; if (t > bv) { bv = t; bi = i; } }
    rv[l] = bv; ri[l] = bi; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) { pv[g] = rv[0]; pi[g] = ri[0]; }
}
__kernel void argmax2(__global const float* pv, __global const int* pi, int P, __global int* tok, __global int* outs, int step CARGS) {
    CCHECK
    __local float rv[256]; __local int ri[256]; int l = get_local_id(0);
    rv[l] = l < P ? pv[l] : -INFINITY; ri[l] = l < P ? pi[l] : 0; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) { tok[0] = ri[0]; outs[step] = ri[0]; }
}
inline float dot16(float16 a, float16 b) { return dot(a.lo.lo, b.lo.lo) + dot(a.lo.hi, b.lo.hi) + dot(a.hi.lo, b.hi.lo) + dot(a.hi.hi, b.hi.hi); }
#ifndef QPG
#define QPG 4                // query heads per KV head (Qwen3.5-9B: 4, Qwen3.8-27B: 6); fattn1/fattn2 run 32*QPG threads
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void fattn(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                    int pos0, int KH, int S, int CH PARGS CARGS) {
    CCHECK
    __local float sm[8][QPG], sl[8][QPG], so[8][QPG][256];
    int kh = get_group_id(0), s = get_group_id(1), t = get_group_id(2);
    int sg = get_sub_group_id(), l = get_sub_group_local_id(), NSG = get_num_sub_groups();
    int L = PADJ(pos0) + t + 1, j0 = s * CH;
    if (j0 >= L) return;
    int j1 = min(L, j0 + CH), NH = KH * QPG;
    float16 q[QPG], o[QPG]; float mx[QPG], su[QPG];
    UNR for (int g = 0; g < QPG; g++) { q[g] = vload16(0, Q + ((size_t)t * NH + kh * QPG + g) * 256 + l * 16); mx[g] = -INFINITY; su[g] = 0.0f; o[g] = (float16)(0.0f); }
    for (int j = j0 + sg; j < j1; j += NSG) {
        size_t r = ((size_t)j * KH + kh) * 256 + l * 16;
        float16 kv = KLD16(0, kc + r), vv = KLD16(0, vc + r);
#define ONL(m_, s_, o_, c_) { float mn = fmax(m_, c_), a = exp(m_ - mn), b = exp(c_ - mn); s_ = s_ * a + b; o_ = o_ * a + vv * b; m_ = mn; }
        UNR for (int g = 0; g < QPG; g++) { float c = sub_group_reduce_add(dot16(q[g], kv)) * 0.0625f; ONL(mx[g], su[g], o[g], c) }
    }
    if (l == 0) { UNR for (int g = 0; g < QPG; g++) { sm[sg][g] = mx[g]; sl[sg][g] = su[g]; } }
    UNR for (int g = 0; g < QPG; g++) vstore16(o[g], 0, &so[sg][g][l * 16]);
    barrier(CLK_LOCAL_MEM_FENCE);
    int lid = get_local_id(0), d0 = (lid & 31) * 8;
    for (int g = lid >> 5; g < QPG; g += (int)(get_local_size(0) >> 5)) {
        float M = -INFINITY;
        for (int i = 0; i < NSG; i++) M = fmax(M, sm[i][g]);
        float tl = 0.0f; float8 ov = (float8)(0.0f);
        for (int i = 0; i < NSG; i++) { float w = sm[i][g] == -INFINITY ? 0.0f : exp(sm[i][g] - M); tl += sl[i][g] * w; ov += vload8(0, &so[i][g][d0]) * w; }
        size_t pb_ = (((size_t)t * KH + kh) * QPG + g) * S + s;
        if ((lid & 31) == 0) { PM[pb_] = M; PL[pb_] = tl; }
        vstore8(ov, 0, PO + pb_ * 256 + d0);
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void fattn2(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                     int pos0, int KH, int S, int CH PARGS CARGS) {
    // 2 tokens x QPG heads, one sub-group each: 32*QPG threads
    CCHECK
    __local float sk[16 * 256], sv[16 * 256];
    int kh = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0), LS = get_local_size(0);
    int t = sg / QPG, g = sg % QPG, NH = KH * QPG;
    int L0 = PADJ(pos0) + 1, Lt = L0 + t, j0 = s * CH;
    if (j0 >= L0 + 1) return;
    int jend = min(L0 + 1, j0 + CH);
    float16 q = vload16(0, Q + ((size_t)t * NH + kh * QPG + g) * 256 + l * 16);
    float m = -INFINITY, sm = 0.0f; float16 o = (float16)(0.0f);
    for (int jt = j0; jt < jend; jt += 16) {
        for (int e = lid; e < 1024; e += LS) {
            int pj = e >> 6, c = e & 63, j = jt + pj;
            float4 kk = (float4)(0.0f), vv = (float4)(0.0f);
            if (j < jend) { size_t r = ((size_t)j * KH + kh) * 256; kk = KLD4(c, kc + r); vv = KLD4(c, vc + r); }
            vstore4(kk, e, sk); vstore4(vv, e, sv);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int np = min(16, jend - jt);
        for (int p = 0; p < np; p++) {
            if (jt + p < Lt) {
                float c = sub_group_reduce_add(dot16(q, vload16(0, sk + p * 256 + l * 16))) * 0.0625f;
                float mn = fmax(m, c), a = exp(m - mn), b = exp(c - mn);
                sm = sm * a + b; o = o * a + vload16(0, sv + p * 256 + l * 16) * b; m = mn;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    size_t pb_ = (((size_t)t * KH + kh) * QPG + g) * S + s;
    if (l == 0) { PM[pb_] = m; PL[pb_] = sm; }
    vstore16(o, 0, PO + pb_ * 256 + l * 16);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void fattn1(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                     int pos0, int KH, int S, int CH PARGS CARGS) {
    // 1 token: QPG heads x 2 key halves, one sub-group each: 32*QPG threads
    CCHECK
    __local float sk[16 * 256], sv[16 * 256], cm[2 * QPG], cs[2 * QPG], co[2 * QPG * 256];
    int kh = get_group_id(0), s = get_group_id(1), sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0), LS = get_local_size(0);
    int g = sg % QPG, hf = sg / QPG;
    int L = PADJ(pos0) + 1, j0 = s * CH;
    if (j0 >= L) return;
    int jend = min(L, j0 + CH);
    float16 q = vload16(0, Q + ((size_t)kh * QPG + g) * 256 + l * 16);
    float m = -INFINITY, sm = 0.0f; float16 o = (float16)(0.0f);
    for (int jt = j0; jt < jend; jt += 16) {
        for (int e = lid; e < 1024; e += LS) {
            int pj = e >> 6, c = e & 63, j = jt + pj;
            float4 kk = (float4)(0.0f), vv = (float4)(0.0f);
            if (j < jend) { size_t r = ((size_t)j * KH + kh) * 256; kk = KLD4(c, kc + r); vv = KLD4(c, vc + r); }
            vstore4(kk, e, sk); vstore4(vv, e, sv);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        int np = min(16, jend - jt);
        for (int p = hf; p < np; p += 2) {
            float c = sub_group_reduce_add(dot16(q, vload16(0, sk + p * 256 + l * 16))) * 0.0625f;
            float mn = fmax(m, c), a = exp(m - mn), b = exp(c - mn);
            sm = sm * a + b; o = o * a + vload16(0, sv + p * 256 + l * 16) * b; m = mn;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0) { cm[sg] = m; cs[sg] = sm; }
    vstore16(o, 0, co + sg * 256 + l * 16);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (hf == 0) {
        float m2 = cm[sg + QPG], s2 = cs[sg + QPG]; float16 o2 = vload16(0, co + (sg + QPG) * 256 + l * 16);
        float M = fmax(m, m2), wa = m == -INFINITY ? 0.0f : exp(m - M), wb = m2 == -INFINITY ? 0.0f : exp(m2 - M);
        sm = sm * wa + s2 * wb; o = o * wa + o2 * wb;
        size_t pb_ = ((size_t)kh * QPG + g) * S + s;
        if (l == 0) { PM[pb_] = M; PL[pb_] = sm; }
        vstore16(o, 0, PO + pb_ * 256 + l * 16);
    }
}
__kernel void fcomb(__global const float* PM, __global const float* PL, __global const float* PO, __global const float* P, int ldp, __global float* A,
                    int pos0, int KH, int S, int CH PARGS CARGS) {
    CCHECK
    int h = get_group_id(0), t = get_group_id(1), d = get_local_id(0), NH = KH * QPG, kh = h / QPG, g = h % QPG;
    int L = PADJ(pos0) + t + 1, ns = (L + CH - 1) / CH;
    size_t base = (((size_t)t * KH + kh) * QPG + g) * S;
    float M = -INFINITY;
    for (int s = 0; s < ns; s++) M = fmax(M, PM[base + s]);
    float tl = 0.0f, o = 0.0f;
    for (int s = 0; s < ns; s++) { float w = exp(PM[base + s] - M); tl += PL[base + s] * w; o += PO[(base + s) * 256 + d] * w; }
    float gt = P[(size_t)t * ldp + h * 512 + 256 + d];
    A[((size_t)t * NH + h) * 256 + d] = o / tl / (1.0f + exp(-gt));
}
__kernel void mapid(__global int* t, __global const int* idx CARGS) {
    CCHECK
    t[0] = idx[t[0]];
}
__kernel void settok(__global const int* ids, int i, __global int* tok) { tok[0] = ids[i]; }

__kernel void conv_silu(__global const float* p, __global float* st, __global const float* w, __global float* o, int C) {
    int c = get_global_id(0); if (c >= C) return;
    float x = p[c], s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    float v = w[c * 4] * s0 + w[c * 4 + 1] * s1 + w[c * 4 + 2] * s2 + w[c * 4 + 3] * x;
    st[c] = s1; st[C + c] = s2; st[2 * C + c] = x;
    o[c] = silu(v);
}
__kernel void dnet(__global const float* qkv, __global const float* p, __global const float* A_log, __global const float* dtb,
                   __global float* S, __global const float* gw, __global float* out, int KD, int VH, int ZO, int BO, int AO, float eps, int gplus1) {
    __local float sq[128], sk[128], r[128];
    int h = get_group_id(0), d = get_local_id(0), kh = h / (VH / 16);
    float qv = qkv[kh * 128 + d], kv = qkv[KD + kh * 128 + d], vv = qkv[2 * KD + h * 128 + d];
    float nq = wsum(qv * qv, r), nk = wsum(kv * kv, r);
    sq[d] = qv * rsqrt(nq + 1e-6f) * 0.08838834764831845f; sk[d] = kv * rsqrt(nk + 1e-6f);
    barrier(CLK_LOCAL_MEM_FENCE);
    float bt = 1.0f / (1.0f + exp(-p[BO + h]));
    float av = p[AO + h] + dtb[h]; float sp = av > 20.0f ? av : log1p(exp(av));
    float eg = exp(-exp(A_log[h]) * sp);
    __global float* Sh = S + (size_t)h * 128 * 128;
    float km = 0.0f;
    for (int i = 0; i < 128; i++) km += Sh[i * 128 + d] * sk[i];
    km *= eg;
    float dl = (vv - km) * bt, o = 0.0f;
    for (int i = 0; i < 128; i++) { float sv = Sh[i * 128 + d] * eg + sk[i] * dl; Sh[i * 128 + d] = sv; o += sv * sq[i]; }
    float ms = wsum(o * o, r) / 128.0f; float z = p[ZO + h * 128 + d];
    out[h * 128 + d] = o * rsqrt(ms + eps) * (gplus1 ? 1.0f + gw[d] : gw[d]) * silu(z);
}
__kernel void attn_prep(__global const float* p, __global const float* qn, __global const float* kn, __global float* q, __global KT* kc, __global KT* vc,
                        int pos, float theta, int NH, int KH, int RD, float eps, int plus1 PARGS CARGS) {
    CCHECK
    __local float r[256], sh[256];
    int g = get_group_id(0), d = get_local_id(0);
    float x = g < NH ? p[g * 512 + d] : p[NH * 512 + (g - NH) * 256 + d];
    float k = rsqrt(wsum(x * x, r) / 256.0f + eps);
    __global const float* w = g < NH ? qn : kn;
    float y = x * k * (plus1 ? 1.0f + w[d] : w[d]);
    sh[d] = y; barrier(CLK_LOCAL_MEM_FENCE);
    int hr = RD / 2;
    if (d < RD) {
        int i = d < hr ? d : d - hr;
        float f = (float)PADJ(pos) * pow(theta, -2.0f * (float)i / (float)RD);
        float c = cos(f), s = sin(f);
        y = d < hr ? sh[d] * c - sh[d + hr] * s : sh[d] * c + sh[d - hr] * s;
    }
    if (g < NH) q[g * 256 + d] = y;
    else { size_t o = ((size_t)PADJ(pos) * KH + (g - NH)) * 256 + d; KST(y, o, kc); KST(p[NH * 512 + KH * 256 + (g - NH) * 256 + d], o, vc); }
}
__kernel void attn(__global const float* q, __global const KT* kc, __global const KT* vc, __global const float* p, __global float* out,
                   __global float* sc, int T, int TM, int KH, int GQ PARGS CARGS) {
    CCHECK
    __local float sq[256], r[256];
    int h = get_group_id(0), l = get_local_id(0), kh = h / GQ;
    sq[l] = q[h * 256 + l]; barrier(CLK_LOCAL_MEM_FENCE);
    float mx = -INFINITY;
    for (int t = l; t < PADJ(T); t += 256) {
        __global const KT* kr = kc + ((size_t)t * KH + kh) * 256; float s = 0.0f;
        for (int i = 0; i < 256; i += 4) s += dot(vload4(0, sq + i), KLD4(0, kr + i));
        s *= 0.0625f; sc[h * TM + t] = s; mx = fmax(mx, s);
    }
    mx = wmax(mx, r); float sum = 0.0f;
    for (int t = l; t < PADJ(T); t += 256) { float e = exp(sc[h * TM + t] - mx); sc[h * TM + t] = e; sum += e; }
    sum = wsum(sum, r);
    barrier(CLK_GLOBAL_MEM_FENCE);
    float o = 0.0f;
    for (int t = 0; t < PADJ(T); t++) o += sc[h * TM + t] * KLD1(((size_t)t * KH + kh) * 256 + l, vc);
    float gt = p[h * 512 + 256 + l];
    out[h * 256 + l] = o / sum / (1.0f + exp(-gt));
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnorm4(__global const float* x, __global const float* w, __global float* y, int H, float eps, int plus1 CARGS) {
    CCHECK
    __local float r[64]; __local float tot;
    int l = get_local_id(0), n4 = H / 4;
    float s = 0.0f;
    for (int i = l; i < n4; i += 1024) { float4 v = vload4(i, x); s += dot(v, v); }
    s = sub_group_reduce_add(s);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_id() == 0) {
        int j = get_sub_group_local_id(); float t = r[j] + r[j + 16] + r[j + 32] + r[j + 48];
        t = sub_group_reduce_add(t); if (j == 0) tot = t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float k = rsqrt(tot / H + eps);
    for (int i = l; i < n4; i += 1024) {
        float4 v = vload4(i, x), g = vload4(i, w);
        vstore4(v * k * (plus1 ? 1.0f + g : g), i, y);
    }
}
__kernel void dnet2(__global const float* qkv, __global const float* p, __global const float* A_log, __global const float* dtb,
                    __global float* S, __global float* PK, __global float* PD, __global float* PE, __global const float* gw, __global float* out,
                    int KD, int VH, int ZO, int BO, int AO, float eps, int gplus1) {
    __local float sq[128], sk[128], spk[128], r[128];
    int h = get_group_id(0), d = get_local_id(0), kh = h / (VH / (KD / 128));
    float qv = qkv[kh * 128 + d], kv = qkv[KD + kh * 128 + d], vv = qkv[2 * KD + h * 128 + d];
    float nq = wsum(qv * qv, r), nk = wsum(kv * kv, r);
    float qn = qv * rsqrt(nq + 1e-6f) * 0.08838834764831845f, kn = kv * rsqrt(nk + 1e-6f);
    sq[d] = qn; sk[d] = kn; spk[d] = PK[h * 128 + d];
    float pdl = PD[h * 128 + d], pe = PE[h];
    float kq = wsum(qn * kn, r);
    float bt = 1.0f / (1.0f + exp(-p[BO + h]));
    float av = p[AO + h] + dtb[h]; float sp = av > 20.0f ? av : log1p(exp(av));
    float eg = exp(-exp(A_log[h]) * sp);
    __global float* Sh = S + (size_t)h * 128 * 128;
    float a1 = 0.0f, a2 = 0.0f;
    for (int i0 = 0; i0 < 128; i0 += DB) {
        float sv[DB];
        UNR for (int j = 0; j < DB; j++) sv[j] = Sh[(i0 + j) * 128 + d];
        UNR for (int j = 0; j < DB; j++) {
            float s = sv[j] * pe + spk[i0 + j] * pdl;
            Sh[(i0 + j) * 128 + d] = s; a1 += s * sk[i0 + j]; a2 += s * sq[i0 + j];
        }
    }
    float dl = (vv - eg * a1) * bt;
    float o = eg * a2 + dl * kq;
    PK[h * 128 + d] = kn; PD[h * 128 + d] = dl;
    if (d == 0) PE[h] = eg;
    float ms = wsum(o * o, r) / 128.0f; float z = p[ZO + h * 128 + d];
    out[h * 128 + d] = o * rsqrt(ms + eps) * (gplus1 ? 1.0f + gw[d] : gw[d]) * silu(z);
}
__kernel void dnet3(__global const float* qkv, __global const float* p, __global const float* A_log, __global const float* dtb,
                    __global float* S, __global float* PK, __global float* PD, __global float* PE, __global const float* gw, __global float* out,
                    int KD, int VH, int ZO, int BO, int AO, float eps, int gplus1) {
    __local float sq[128], sk[128], spk[128], r[128 * NP], pa1[NP][128], pa2[NP][128];
    int h = get_group_id(0), lid = get_local_id(0), d = lid & 127, part = lid >> 7, kh = h / (VH / (KD / 128));
    float qv = qkv[kh * 128 + d], kv = qkv[KD + kh * 128 + d], vv = qkv[2 * KD + h * 128 + d];
    float z0 = part == 0 ? 1.0f : 0.0f;
    float nq = wsum(z0 * qv * qv, r), nk = wsum(z0 * kv * kv, r);
    float qn = qv * rsqrt(nq + 1e-6f) * 0.08838834764831845f, kn = kv * rsqrt(nk + 1e-6f);
    if (part == 0) { sq[d] = qn; sk[d] = kn; spk[d] = PK[h * 128 + d]; }
    float pdl = PD[h * 128 + d], pe = PE[h];
    float kq = wsum(z0 * qn * kn, r);
    __global float* Sh = S + (size_t)h * 128 * 128;
    float a1 = 0.0f, a2 = 0.0f;
    int i0 = part * (128 / NP);
    float sv[128 / NP];
    UNR for (int j = 0; j < 128 / NP; j++) sv[j] = Sh[(i0 + j) * 128 + d];
    UNR for (int j = 0; j < 128 / NP; j++) {
        float s = sv[j] * pe + spk[i0 + j] * pdl;
        Sh[(i0 + j) * 128 + d] = s; a1 += s * sk[i0 + j]; a2 += s * sq[i0 + j];
    }
    pa1[part][d] = a1; pa2[part][d] = a2;
    barrier(CLK_LOCAL_MEM_FENCE);
    float o = 0.0f;
    if (part == 0) {
        UNR for (int q = 1; q < NP; q++) { a1 += pa1[q][d]; a2 += pa2[q][d]; }
        float bt = 1.0f / (1.0f + exp(-p[BO + h]));
        float av = p[AO + h] + dtb[h]; float sp = av > 20.0f ? av : log1p(exp(av));
        float eg = exp(-exp(A_log[h]) * sp);
        float dl = (vv - eg * a1) * bt;
        o = eg * a2 + dl * kq;
        PK[h * 128 + d] = kn; PD[h * 128 + d] = dl;
        if (d == 0) PE[h] = eg;
    }
    float ms = wsum(o * o, r) / 128.0f;
    if (part == 0) { float z = p[ZO + h * 128 + d]; out[h * 128 + d] = o * rsqrt(ms + eps) * (gplus1 ? 1.0f + gw[d] : gw[d]) * silu(z); }
}
__kernel void swiglu(__global const float* gu, __global float* o, int F CARGS) {
    CCHECK
    int i = get_global_id(0); if (i < F) o[i] = silu(gu[i]) * gu[F + i];
}
"""



def norm_form(meta, groups):
    """The container json "norm_form" key -> {group: 1 for (1+w), 0 for w}. None when the key is absent."""
    nf = meta.get("norm_form")
    if nf is None:
        return None
    out = {}
    for g in groups:
        v = nf.get(g)
        if v not in ("1+w", "w"):
            raise ValueError(f"norm_form[{g!r}] = {v!r}: must be '1+w' or 'w'")
        out[g] = int(v == "1+w")
    return out

class Model:
    def __init__(self, path, tmax=None, gv=None, prof=False):
        if gv is None:
            gv = tuple(int(x) for x in knob("Q35_GV", "2,0,8").split(","))
        tmax = tmax or int(knob("Q35_TMAX", "4096"))
        self.gv = gv; self.path = path
        self.flush = int(knob("Q35_FLUSH", "32"))
        self.k2 = int(knob("Q35_K2", "3"))
        self.np_ = int(knob("Q35_NP", "2"))
        meta = json.load(open(path[:-4] + ".json"))
        self.meta, self.cfg, self.tm = meta, meta["config"], meta["tensors"]
        c = self.cfg
        self.H, self.F, self.V = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
        self.KD = c["linear_num_key_heads"] * c["linear_key_head_dim"]; self.VH = c["linear_num_value_heads"]
        self.NH, self.KH = c["num_attention_heads"], c["num_key_value_heads"]
        self.RD = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"]); self.theta = float(c["rope_parameters"]["rope_theta"])
        self.eps = float(c["rms_norm_eps"]); self.TM = tmax
        nf = norm_form(meta, ("rms", "gated", "qk", "mtp"))
        if nf:
            self.p1, self.gp1, self.qp1 = nf["rms"], nf["gated"], nf["qk"]
            if nf["mtp"] != 1:
                raise NotImplementedError("only the 1+w norm form is supported for MTP")
        else:
            ns = meta["norm_stats"]
            self.p1 = int(abs(ns["layers.0.in_norm"][0]) < 0.5)
            self.gp1 = int(abs(ns["layers.0.gnorm"][0]) < 0.5)
            self.qp1 = int(abs(ns["layers.3.q_norm"][0]) < 0.5)
        dev = K2.device()
        self.ctx = cl.Context([dev])
        self.prof = prof; self.ev = []; self.lab = {}
        self.q = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE) if prof else cl.CommandQueue(self.ctx)
        opt = [f"-DG={meta['G']}", f"-DXE8={int(meta['fmt'] == 'xe8')}", f"-DDB={int(knob('Q35_DB', '64'))}", f"-DNP={self.np_}"]
        self.GQ = self.NH // self.KH; self.ALS = 32 * self.GQ               # query heads per KV head; fattn1/fattn2 work-group size
        opt += [f"-DQPG={self.GQ}"]
        self.kv16 = int(knob("Q35_KV16", "1")); self.KVB = 2 if self.kv16 else 4      # KV cache element: FP16 (default) or FP32
        opt += ["-DKV16"] if self.kv16 else []
        self.rmap = {"proj": 1, "out": 1, "gu": 2 if meta["G"] == 32 else 1, "down": 1, "lm_head": 2} if gv[1] == 0 else {k: gv[1] for k in ("proj", "out", "gu", "down", "lm_head")}
        self.prgs = {r: cl.Program(self.ctx, SRC).build(options=opt + [f"-DR={r}"]) for r in sorted(set(self.rmap.values()))}
        self.prg = self.prgs[min(self.prgs)]
        raw = np.memmap(path, dtype=np.uint8, mode="r")
        mf = cl.mem_flags
        self.buf = {}
        for k, t in self.tm.items():
            a = np.asarray(raw[t["off"]:t["off"] + t["nbytes"]])
            if t["kind"] in ("fac", "efac"):
                a = np.concatenate([a, np.zeros(16, np.uint8)])
            self.buf[k] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        f32 = lambda n: cl.Buffer(self.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        H, F = self.H, self.F
        self.h, self.t, self.a = f32(H), f32(H), f32(max(H, self.VH * 128))
        self.P = f32(max(self.tm[f"layers.{i}.proj.codes"]["shape"][0] for i in range(c["num_hidden_layers"])))
        self.C = f32(2 * self.KD + self.VH * 128)
        self.GU, self.M = f32(2 * F), f32(F)
        self.Q = f32(self.NH * 256); self.SC = f32(self.NH * tmax)
        self.fa = int(knob("Q35_FATTN", "1")); self.CH = 256; self.SM = -(-tmax // self.CH)
        self.fa1 = "fattn1" if knob("Q35_FATTN1", "1") == "1" else "fattn"
        self.PM, self.PL, self.PO = f32(self.NH * self.SM), f32(self.NH * self.SM), f32(self.NH * self.SM * 256)
        self.logit = f32(self.V); self.npart = (self.V + 4095) // 4096
        self.pv, self.pi = f32(self.npart), f32(self.npart)
        self.tok = f32(1); self.outs = f32(tmax); self.ids = f32(tmax)
        self.state = {}
        for i, lt in enumerate(c["layer_types"]):
            if lt == "linear_attention":
                self.state[i] = (f32(3 * (2 * self.KD + self.VH * 128)), f32(self.VH * 128 * 128), f32(self.VH * 128), f32(self.VH * 128), f32(self.VH))
            else:
                self.state[i] = tuple(cl.Buffer(self.ctx, mf.READ_WRITE, tmax * self.KH * 256 * self.KVB) for _ in range(2))
        self.build()

    def reset(self):
        for i, bufs in self.state.items():
            for x in bufs:
                cl.enqueue_fill_buffer(self.q, x, np.float32(0), 0, x.size)
        self.q.finish()

    def K(self, name, g, l, *args, label=None, prg=None):
        k = cl.Kernel(prg or self.prg, name); k.set_args(*args)
        self.lab[id(k)] = label or name
        return (k, g, l)

    def gemv(self, name, x, y, add):
        t = self.tm[name + ".codes"]; N, K = t["shape"][0], t["shape"][1] * 2
        v, _, W = self.gv
        kind = name.split(".")[-1]; lab = "gemv." + kind
        if v == 1:
            return self.K("gemv", (16, -(-N // 8) * 8), (16, 8), self.buf[name + ".codes"], self.buf[name + ".fac"], self.buf[name + ".rs"], x, y, np.int32(N), np.int32(K), np.int32(add), label=lab)
        R = self.rmap.get(kind, 1)
        assert N % R == 0, (name, N, R)
        return self.K("gemv2", (16, -(-(N // R) // W) * W), (16, W), self.buf[name + ".codes"], self.buf[name + ".fac"], self.buf[name + ".rs"], x, y, np.int32(N), np.int32(K), np.int32(add), label=lab, prg=self.prgs[R])

    def norm(self, x, w, y, plus1):
        if self.k2:
            return self.K("rmsnorm4", (1024,), (1024,), x, w, y, np.int32(self.H), np.float32(self.eps), np.int32(plus1), label="rmsnorm")
        return self.K("rmsnorm", (256,), (256,), x, w, y, np.int32(self.H), np.float32(self.eps), np.int32(plus1))

    def lmhead(self):
        v, _, W = self.gv
        if v == 1:
            return self.K("gemv_i8", (16, -(-self.V // 8) * 8), (16, 8), self.buf["lm_head.q"], self.buf["lm_head.s"], self.t, self.logit, np.int32(self.V), np.int32(self.H), label="lm_head")
        R = self.rmap["lm_head"]
        return self.K("gemv_i8r", (16, -(-(self.V // R) // W) * W), (16, W), self.buf["lm_head.q"], self.buf["lm_head.s"], self.t, self.logit, np.int32(self.V), np.int32(self.H), label="lm_head", prg=self.prgs[R])

    def _fls(self):
        """Work-group size of the single-token attention kernel (fattn1: 32 * QPG, fattn: 128)."""
        return self.ALS if self.fa1 == "fattn1" else 128

    def build(self):
        c, H, F, I = self.cfg, self.H, self.F, np.int32
        eps = np.float32(self.eps)
        if "embed.ecodes" in self.tm:
            self.head = [self.K("embed_xe", (-(-H // 256) * 256,), (256,), self.buf["embed.ecodes"], self.buf["embed.efac"], self.buf["embed.ers"], self.tok, self.h, I(H))]
        else:
            self.head = [self.K("embed", (-(-H // 256) * 256,), (256,), self.buf["embed"], self.tok, self.h, I(H))]
        self.layers = []
        for i, lt in enumerate(c["layer_types"]):
            p = f"layers.{i}."
            ks, posk = [], []
            ks.append(self.norm(self.h, self.buf[p + "in_norm"], self.t, self.p1))
            ks.append(self.gemv(p + "proj", self.t, self.P, 0))
            if lt == "linear_attention":
                st, S, PK, PD, PE = self.state[i]; CC = 2 * self.KD + self.VH * 128
                ks.append(self.K("conv_silu", (-(-CC // 256) * 256,), (256,), self.P, st, self.buf[p + "conv"], self.C, I(CC)))
                ZO = CC; BO = ZO + self.VH * 128; AO = BO + self.VH
                if self.k2 == 3:
                    ks.append(self.K("dnet3", (self.VH * 128 * self.np_,), (128 * self.np_,), self.C, self.P, self.buf[p + "A_log"], self.buf[p + "dt_bias"], S, PK, PD, PE, self.buf[p + "gnorm"], self.a,
                                     I(self.KD), I(self.VH), I(ZO), I(BO), I(AO), eps, I(self.gp1), label="dnet"))
                elif self.k2:
                    ks.append(self.K("dnet2", (self.VH * 128,), (128,), self.C, self.P, self.buf[p + "A_log"], self.buf[p + "dt_bias"], S, PK, PD, PE, self.buf[p + "gnorm"], self.a,
                                     I(self.KD), I(self.VH), I(ZO), I(BO), I(AO), eps, I(self.gp1)))
                else:
                    ks.append(self.K("dnet", (self.VH * 128,), (128,), self.C, self.P, self.buf[p + "A_log"], self.buf[p + "dt_bias"], S, self.buf[p + "gnorm"], self.a,
                                     I(self.KD), I(self.VH), I(ZO), I(BO), I(AO), eps, I(self.gp1)))
            else:
                kc, vc = self.state[i][:2]
                kp = self.K("attn_prep", ((self.NH + self.KH) * 256,), (256,), self.P, self.buf[p + "q_norm"], self.buf[p + "k_norm"], self.Q, kc, vc,
                            I(0), np.float32(self.theta), I(self.NH), I(self.KH), I(self.RD), eps, I(self.qp1))
                if self.fa:
                    ka1 = list(self.K(self.fa1, (self.KH * self._fls(), 1, 1), (self._fls(), 1, 1), self.Q, kc, vc, self.PM, self.PL, self.PO, I(0), I(self.KH), I(self.SM), I(self.CH)))
                    ka2 = self.K("fcomb", (self.NH * 256, 1), (256, 1), self.PM, self.PL, self.PO, self.P, I(0), self.a, I(0), I(self.KH), I(self.SM), I(self.CH))
                    ks += [kp, ka1, ka2]; posk = [kp, ka1, ka2]
                else:
                    ka = self.K("attn", (self.NH * 256,), (256,), self.Q, kc, vc, self.P, self.a, self.SC, I(1), I(self.TM), I(self.KH), I(self.NH // self.KH))
                    ks += [kp, ka]; posk = [kp, ka]
            ks.append(self.gemv(p + "out", self.a, self.h, 1))
            ks.append(self.norm(self.h, self.buf[p + "post_norm"], self.t, self.p1))
            ks.append(self.gemv(p + "gu", self.t, self.GU, 0))
            ks.append(self.K("swiglu", (-(-F // 256) * 256,), (256,), self.GU, self.M, I(F)))
            ks.append(self.gemv(p + "down", self.M, self.h, 1))
            self.layers.append((ks, posk))
        self.tail = [self.norm(self.h, self.buf["norm"], self.t, self.p1), self.lmhead()]
        self.am1 = self.K("argmax1", (self.npart * 256,), (256,), self.logit, I(self.V), self.pv, self.pi)
        self.am2 = self.K("argmax2", (256,), (256,), self.pv, self.pi, I(self.npart), self.tok, self.outs, I(0))
        self.st = self.K("settok", (1,), (1,), self.ids, I(0), self.tok)

    def load_mtp(self, path, order="eh", hid="pre"):
        """MTP layer: input = [norm(embed(next token)), norm(main-model hidden state)] -> fc -> one attention layer -> norm -> shared lm_head.
        order: "eh" = embedding first, "he" = hidden state first. hid: "pre" = last layer output, "post" = after the final norm."""
        meta = json.load(open(path[:-4] + ".json")); raw = np.memmap(path, dtype=np.uint8, mode="r"); mf = cl.mem_flags
        for k, t in meta["tensors"].items():
            a = np.asarray(raw[t["off"]:t["off"] + t["nbytes"]])
            if t["kind"] == "fac":
                a = np.concatenate([a, np.zeros(16, np.uint8)])
            self.buf[k] = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
            self.tm[k] = t
        ns = meta["norm_stats"]; pl = lambda k: 1
        H, F, I, eps = self.H, self.F, np.int32, np.float32(self.eps)
        f32 = lambda n: cl.Buffer(self.ctx, mf.READ_WRITE, int(n) * 4)
        self.mE, self.mcat, self.mh = f32(H), f32(2 * H), f32(H)
        self.mkc, self.mvc = (cl.Buffer(self.ctx, mf.READ_WRITE, self.TM * self.KH * 256 * self.KVB) for _ in range(2))
        self.mtok, self.mouts = f32(1), f32(self.TM)
        first, second = (self.mcat.get_sub_region(0, H * 4), self.mcat.get_sub_region(H * 4, H * 4))
        self._keep = (first, second)
        eo, ho = (first, second) if order == "eh" else (second, first)
        hsrc = self.h if hid == "pre" else self.t
        if "embed.ecodes" in self.tm:
            ke = self.K("embed_xe", (-(-H // 256) * 256,), (256,), self.buf["embed.ecodes"], self.buf["embed.efac"], self.buf["embed.ers"], self.tok, self.mE, I(H))
        else:
            ke = self.K("embed", (-(-H // 256) * 256,), (256,), self.buf["embed"], self.tok, self.mE, I(H))
        self.mkp = self.K("attn_prep", ((self.NH + self.KH) * 256,), (256,), self.P, self.buf["mtp.q_norm"], self.buf["mtp.k_norm"], self.Q, self.mkc, self.mvc,
                          I(0), np.float32(self.theta), I(self.NH), I(self.KH), I(self.RD), eps, I(pl("mtp.q_norm")))
        self.mka = self.K("attn", (self.NH * 256,), (256,), self.Q, self.mkc, self.mvc, self.P, self.a, self.SC, I(1), I(self.TM), I(self.KH), I(self.NH // self.KH))
        self.mk = [ke, self.norm(self.mE, self.buf["mtp.pe"], eo, pl("mtp.pe")), self.norm(hsrc, self.buf["mtp.ph"], ho, pl("mtp.ph")),
                   *([self.gemv("mtp.fc", self.mcat, self.mh, 0)] if "mtp.fc.codes" in self.tm else
                     [self.gemv("mtp.fcA", first, self.mh, 0), self.gemv("mtp.fcB", second, self.mh, 1)]),
                   self.norm(self.mh, self.buf["mtp.in_norm"], self.t, pl("mtp.in_norm")), self.gemv("mtp.proj", self.t, self.P, 0),
                   self.mkp, self.mka, self.gemv("mtp.out", self.a, self.mh, 1),
                   self.norm(self.mh, self.buf["mtp.post_norm"], self.t, pl("mtp.post_norm")), self.gemv("mtp.gu", self.t, self.GU, 0),
                   self.K("swiglu", (-(-F // 256) * 256,), (256,), self.GU, self.M, I(F)), self.gemv("mtp.down", self.M, self.mh, 1),
                   self.norm(self.mh, self.buf["mtp.norm"], self.t, pl("mtp.norm")), self.lmhead(),
                   self.am1, self.K("argmax2", (256,), (256,), self.pv, self.pi, I(self.npart), self.mtok, self.mouts, I(0))]
        self.mtp_ns = {k: v for k, v in ns.items()}

    def mtp_step(self, pos, out_step):
        """Call right after a main-model step: embed(tok) + hidden state -> MTP at position pos -> mouts[out_step]."""
        self.mkp[0].set_arg(6, np.int32(pos)); self.mka[0].set_arg(6, np.int32(pos + 1))
        self.mk[-1][0].set_arg(5, np.int32(out_step))
        self.run(self.mk)

    def run(self, kl):
        for k, g, l in kl:
            e = cl.enqueue_nd_range_kernel(self.q, k, g, l)
            if self.prof:
                self.ev.append((self.lab[id(k)], e))

    def step(self, pos, out_step=None, feed=None):
        """One token: feeds ids[feed] if given, else the previous greedy result. Writes the result to outs[out_step] if given."""
        if feed is not None:
            self.st[0].set_arg(1, np.int32(feed)); self.run([self.st])
        self.run(self.head)
        for li, (ks, posk) in enumerate(self.layers):
            if posk:
                posk[0][0].set_arg(6, np.int32(pos))
                if len(posk) == 3:
                    posk[1][0].set_arg(6, np.int32(pos)); posk[1][1] = (self.KH * self._fls(), -(-(pos + 1) // self.CH), 1); posk[2][0].set_arg(6, np.int32(pos))
                else:
                    posk[1][0].set_arg(6, np.int32(pos + 1))
            self.run(ks)
            if self.flush and li % self.flush == self.flush - 1:
                self.q.flush()
        self.run(self.tail)
        if out_step is not None:
            self.am2[0].set_arg(5, np.int32(out_step)); self.run([self.am1, self.am2])

    def load_ids(self, ids):
        a = np.asarray(ids, np.int32); cl.enqueue_copy(self.q, self.ids, a); self.q.finish()

    def generate(self, ids, n):
        """Feeds the prompt token by token and generates n tokens greedily. Returns (tokens, prompt time, generation time)."""
        self.reset(); self.load_ids(ids)
        t0 = time.perf_counter()
        for i in range(len(ids)):
            self.step(i, out_step=0 if i == len(ids) - 1 else None, feed=i)
        o = np.zeros(1, np.int32); cl.enqueue_copy(self.q, o, self.outs); self.q.finish()
        t1 = time.perf_counter()
        for j in range(1, n):
            self.step(len(ids) + j - 1, out_step=j)
        out = np.zeros(n, np.int32); cl.enqueue_copy(self.q, out, self.outs); self.q.finish()
        t2 = time.perf_counter()
        return out.tolist(), t1 - t0, t2 - t1

    def logits(self):
        v = np.zeros(self.V, np.float32); cl.enqueue_copy(self.q, v, self.logit); self.q.finish(); return v
