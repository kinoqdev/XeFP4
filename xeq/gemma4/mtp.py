"""Gemma 4 26B-A4B MTP speculative decoding: the assistant model drafts k tokens, the main model verifies k+1 tokens in one pass.
Assistant input = pre_projection([main embedding (x53) ; main model's last normed hidden state]); 4 layers (3 sliding + 1 global)
with q_proj only, attending to the KV of the main model's last sliding and last global layer; output = its own tied embed_tokens
on norm(x); next hidden state = post_projection(norm(x)).
Multi-token step of the main model (T <= 8): weights are read once for T tokens; experts run once each over the union chosen by
the T tokens; attention kernels are T-versions of the one-token kernels.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pyopencl as cl

from . import model as GR
from ..device import knob

TB = 8
DBG = knob("G4M_DBG") == "1"
FL = int(knob("G4M_FL", "48")); FL0 = int(knob("G4M_FL0", "7")); FLE = knob("G4M_FLE", "0") == "1"
PROF = None
MSRC = r"""
#ifndef TB
#define TB 8
#endif
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemvT(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, int T, int add) {
    int r0 = get_global_id(1) * R, l = get_local_id(0); if (r0 >= N) return;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    float acc[TB][R];
    UNR for (int t = 0; t < TB; t++) UNR for (int r = 0; r < R; r++) acc[t][r] = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        float4 lo[R][4], hi[R][4]; float m[R];
        UNR for (int r = 0; r < R; r++) {
            size_t row = r0 + r;
            __global const uint* fr = fac + row * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            m[r] = (float)(((fw >> sh) & 7) + 1);
            uint4 u = vload4(0, codes + row * (K / 8) + b * (G / 8));
            lo[r][0] = convert_float4(as_char4(dlo(u.s0))); hi[r][0] = convert_float4(as_char4(dhi(u.s0)));
            lo[r][1] = convert_float4(as_char4(dlo(u.s1))); hi[r][1] = convert_float4(as_char4(dhi(u.s1)));
            lo[r][2] = convert_float4(as_char4(dlo(u.s2))); hi[r][2] = convert_float4(as_char4(dhi(u.s2)));
            lo[r][3] = convert_float4(as_char4(dlo(u.s3))); hi[r][3] = convert_float4(as_char4(dhi(u.s3)));
        }
        UNR for (int t = 0; t < TB; t++) if (t < T) {
            __global const float* xp = x + (size_t)t * K + b * G;
            float8 v0 = vload8(0, xp), v1 = vload8(1, xp), v2 = vload8(2, xp), v3 = vload8(3, xp);
            float4 e0 = (float4)(v0.s0, v0.s2, v0.s4, v0.s6), o0 = (float4)(v0.s1, v0.s3, v0.s5, v0.s7), e1 = (float4)(v1.s0, v1.s2, v1.s4, v1.s6), o1 = (float4)(v1.s1, v1.s3, v1.s5, v1.s7);
            float4 e2v = (float4)(v2.s0, v2.s2, v2.s4, v2.s6), o2 = (float4)(v2.s1, v2.s3, v2.s5, v2.s7), e3 = (float4)(v3.s0, v3.s2, v3.s4, v3.s6), o3 = (float4)(v3.s1, v3.s3, v3.s5, v3.s7);
            UNR for (int r = 0; r < R; r++) {
                float sm = (dot(lo[r][0], e0) + dot(hi[r][0], o0)) + (dot(lo[r][1], e1) + dot(hi[r][1], o1)) + (dot(lo[r][2], e2v) + dot(hi[r][2], o2)) + (dot(lo[r][3], e3) + dot(hi[r][3], o3));
                acc[t][r] += sm * m[r];
            }
        }
    }
    UNR for (int t = 0; t < TB; t++) if (t < T) UNR for (int r = 0; r < R; r++) {
        float v = sub_group_reduce_add(acc[t][r]);
        if (l == 0) { size_t row = r0 + r; v *= vload_half(row, rs) * 0.125f; size_t o = (size_t)t * N + row; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void igemvT(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K, int T, int add) {
    int r0 = get_global_id(1) * R; if (r0 >= N) return;
    int l = get_local_id(0);
    float acc[TB][R];
    UNR for (int t = 0; t < TB; t++) UNR for (int r = 0; r < R; r++) acc[t][r] = 0.0f;
    for (int k = l * 16; k < K; k += 256) {
        float4 w[R][4];
        UNR for (int r = 0; r < R; r++) {
            char16 a = vload16(0, q + (size_t)(r0 + r) * K + k);
            w[r][0] = convert_float4(a.s0123); w[r][1] = convert_float4(a.s4567); w[r][2] = convert_float4(a.s89ab); w[r][3] = convert_float4(a.scdef);
        }
        UNR for (int t = 0; t < TB; t++) if (t < T) {
            float16 v = vload16(0, x + (size_t)t * K + k);
            UNR for (int r = 0; r < R; r++) acc[t][r] += dot(w[r][0], v.s0123) + dot(w[r][1], v.s4567) + dot(w[r][2], v.s89ab) + dot(w[r][3], v.scdef);
        }
    }
    UNR for (int t = 0; t < TB; t++) if (t < T) UNR for (int r = 0; r < R; r++) {
        float s = sub_group_reduce_add(acc[t][r]);
        if (l == 0) { size_t o = (size_t)t * N + r0 + r; float v = s * sc[r0 + r]; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void eunion(__global const int* idx, int n, __global int* uni, __global uint* mlo, __global uint* mhi, __global int* cnt) {
    __local int v[64]; __local int fs[64]; __local uint lo[64]; __local uint hi[64];
    int i = get_local_id(0);
    int e = i < n ? idx[i] : -1 - i; v[i] = e; lo[i] = 0u; hi[i] = 0u;
    barrier(CLK_LOCAL_MEM_FENCE);
    int fi = i; for (int k = 0; k < i; k++) if (v[k] == e) { fi = k; break; }
    fs[i] = (i < n && fi == i) ? 1 : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    int s = 0; for (int k = 0; k < fi; k++) s += fs[k];
    if (i < n) { if (i < 32) atomic_or(&lo[s], 1u << i); else atomic_or(&hi[s], 1u << (i - 32)); }
    if (i < n && fi == i) uni[s] = e;
    barrier(CLK_LOCAL_MEM_FENCE);
    mlo[i] = lo[i]; mhi[i] = hi[i];
    if (i == 0) { int c = 0; for (int k = 0; k < n; k++) c += fs[k]; cnt[0] = c; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemvE(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int NE, int K, __global const int* uni, __global const uint* mlo, __global const uint* mhi, __global const int* cnt, int TK) {
    int gid = get_global_id(1), l = get_local_id(0), per = NE / R, u = gid / per, r0 = (gid % per) * R;
    if (u >= cnt[0]) return;
    int sl[TB], ns = 0; uint a = mlo[u], c = mhi[u];
    UNR for (int q = 0; q < TB; q++) sl[q] = 0;
    while (a) { int i = ctz(a); a &= a - 1u; UNR for (int q = 0; q < TB; q++) if (q == ns) sl[q] = i; ns++; }
    while (c) { int i = 32 + ctz(c); c &= c - 1u; UNR for (int q = 0; q < TB; q++) if (q == ns) sl[q] = i; ns++; }
    size_t rbase = (size_t)uni[u] * NE;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    float acc[TB][R];
    UNR for (int t = 0; t < TB; t++) UNR for (int r = 0; r < R; r++) acc[t][r] = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        float4 lo[R][4], hi[R][4]; float m[R];
        UNR for (int r = 0; r < R; r++) {
            size_t row = rbase + r0 + r;
            __global const uint* fr = fac + row * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            m[r] = (float)(((fw >> sh) & 7) + 1);
            uint4 u = vload4(0, codes + row * (K / 8) + b * (G / 8));
            lo[r][0] = convert_float4(as_char4(dlo(u.s0))); hi[r][0] = convert_float4(as_char4(dhi(u.s0)));
            lo[r][1] = convert_float4(as_char4(dlo(u.s1))); hi[r][1] = convert_float4(as_char4(dhi(u.s1)));
            lo[r][2] = convert_float4(as_char4(dlo(u.s2))); hi[r][2] = convert_float4(as_char4(dhi(u.s2)));
            lo[r][3] = convert_float4(as_char4(dlo(u.s3))); hi[r][3] = convert_float4(as_char4(dhi(u.s3)));
        }
        UNR for (int q = 0; q < TB; q++) if (q < ns) {
            __global const float* xp = x + (size_t)(sl[q] / TK) * K + b * G;
            float8 v0 = vload8(0, xp), v1 = vload8(1, xp), v2 = vload8(2, xp), v3 = vload8(3, xp);
            float4 e0 = (float4)(v0.s0, v0.s2, v0.s4, v0.s6), o0 = (float4)(v0.s1, v0.s3, v0.s5, v0.s7), e1 = (float4)(v1.s0, v1.s2, v1.s4, v1.s6), o1 = (float4)(v1.s1, v1.s3, v1.s5, v1.s7);
            float4 e2v = (float4)(v2.s0, v2.s2, v2.s4, v2.s6), o2 = (float4)(v2.s1, v2.s3, v2.s5, v2.s7), e3 = (float4)(v3.s0, v3.s2, v3.s4, v3.s6), o3 = (float4)(v3.s1, v3.s3, v3.s5, v3.s7);
            UNR for (int r = 0; r < R; r++) {
                float sm = (dot(lo[r][0], e0) + dot(hi[r][0], o0)) + (dot(lo[r][1], e1) + dot(hi[r][1], o1)) + (dot(lo[r][2], e2v) + dot(hi[r][2], o2)) + (dot(lo[r][3], e3) + dot(hi[r][3], o3));
                acc[q][r] += sm * m[r];
            }
        }
    }
    UNR for (int q = 0; q < TB; q++) if (q < ns) UNR for (int r = 0; r < R; r++) {
        float v = sub_group_reduce_add(acc[q][r]);
        if (l == 0) { size_t row = rbase + r0 + r; v *= vload_half(row, rs) * 0.125f; y[(size_t)sl[q] * NE + r0 + r] = v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemvE2(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                      int NE, int K, __global const int* uni, __global const uint* mlo, __global const uint* mhi, __global const int* cnt, int TK) {
    int gid = get_global_id(1), l = get_local_id(0), per = NE / R2E, u = gid / per, r0 = (gid % per) * R2E;
    if (u >= cnt[0]) return;
    int sl[TB], ns = 0; uint a = mlo[u], c = mhi[u];
    UNR for (int q = 0; q < TB; q++) sl[q] = 0;
    while (a) { int i = ctz(a); a &= a - 1u; UNR for (int q = 0; q < TB; q++) if (q == ns) sl[q] = i; ns++; }
    while (c) { int i = 32 + ctz(c); c &= c - 1u; UNR for (int q = 0; q < TB; q++) if (q == ns) sl[q] = i; ns++; }
    size_t rbase = (size_t)uni[u] * NE;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    float acc[TB][R2E];
    UNR for (int t = 0; t < TB; t++) UNR for (int r = 0; r < R2E; r++) acc[t][r] = 0.0f;
    for (int b = l; b < nb; b += 16) {
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        uint wl[R2E][4], wh[R2E][4]; float m[R2E];
        UNR for (int r = 0; r < R2E; r++) {
            size_t row = rbase + r0 + r;
            __global const uint* fr = fac + row * FW;
            ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
            m[r] = (float)(((fw >> sh) & 7) + 1);
            uint4 uu = vload4(0, codes + row * (K / 8) + b * (G / 8));
            wl[r][0] = dlo(uu.s0); wh[r][0] = dhi(uu.s0); wl[r][1] = dlo(uu.s1); wh[r][1] = dhi(uu.s1);
            wl[r][2] = dlo(uu.s2); wh[r][2] = dhi(uu.s2); wl[r][3] = dlo(uu.s3); wh[r][3] = dhi(uu.s3);
        }
        UNR for (int q = 0; q < TB; q++) if (q < ns) {
            __global const float* xp = x + (size_t)(sl[q] / TK) * K + b * G;
            float8 v0 = vload8(0, xp), v1 = vload8(1, xp), v2 = vload8(2, xp), v3 = vload8(3, xp);
            UNR for (int r = 0; r < R2E; r++) {
                float sm = (dot(convert_float4(as_char4(wl[r][0])), v0.even) + dot(convert_float4(as_char4(wh[r][0])), v0.odd))
                         + (dot(convert_float4(as_char4(wl[r][1])), v1.even) + dot(convert_float4(as_char4(wh[r][1])), v1.odd))
                         + (dot(convert_float4(as_char4(wl[r][2])), v2.even) + dot(convert_float4(as_char4(wh[r][2])), v2.odd))
                         + (dot(convert_float4(as_char4(wl[r][3])), v3.even) + dot(convert_float4(as_char4(wh[r][3])), v3.odd));
                acc[q][r] += sm * m[r];
            }
        }
    }
    UNR for (int q = 0; q < TB; q++) if (q < ns) UNR for (int r = 0; r < R2E; r++) {
        float v = sub_group_reduce_add(acc[q][r]);
        if (l == 0) { size_t row = rbase + r0 + r; v *= vload_half(row, rs) * 0.125f; y[(size_t)sl[q] * NE + r0 + r] = v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xdownE(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                     int N, int K, __global const int* uni, __global const uint* mlo, __global const uint* mhi, __global const int* cnt,
                     __global const float* wt, int TK, int T) {
    __local int lsl[64 * TB]; __local int lns[64];
    int lid = get_local_id(0) + get_local_id(1) * 16, nu = cnt[0];
    if (lid < 64) {
        int ns = 0;
        if (lid < nu) { uint a = mlo[lid], c = mhi[lid]; for (int i = 0; i < 64 && ns < TB; i++) if (((i < 32 ? a >> i : c >> (i - 32)) & 1u)) lsl[lid * TB + ns++] = i; }
        lns[lid] = ns;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int row = get_global_id(1), l = get_local_id(0); if (row >= N) return;
    int nb = K / G, FW = (nb * 3 + 31) >> 5;
    float acc[TB];
    UNR for (int t = 0; t < TB; t++) acc[t] = 0.0f;
    for (int p = l; p < nu * nb; p += 16) {
        int u = p / nb, b = p - u * nb;
        size_t rr = (size_t)uni[u] * N + row;
        __global const uint* cr = codes + rr * (K / 8) + b * (G / 8);
        __global const uint* fr = fac + rr * FW;
        int bo = b * 3, wo = bo >> 5, sh = bo & 31;
        ulong fw = (ulong)fr[wo] | ((ulong)fr[wo + 1] << 32);
        float m = (float)(((fw >> sh) & 7) + 1) * vload_half(rr, rs) * 0.125f;
        uint4 u4 = vload4(0, cr);
        float4 lo0 = convert_float4(as_char4(dlo(u4.s0))), hi0 = convert_float4(as_char4(dhi(u4.s0)));
        float4 lo1 = convert_float4(as_char4(dlo(u4.s1))), hi1 = convert_float4(as_char4(dhi(u4.s1)));
        float4 lo2 = convert_float4(as_char4(dlo(u4.s2))), hi2 = convert_float4(as_char4(dhi(u4.s2)));
        float4 lo3 = convert_float4(as_char4(dlo(u4.s3))), hi3 = convert_float4(as_char4(dhi(u4.s3)));
        int ns = lns[u];
        for (int q = 0; q < ns; q++) {
            int i = lsl[u * TB + q];
            __global const float* xp = x + (size_t)i * K + b * G;
            float8 v0 = vload8(0, xp), v1 = vload8(1, xp), v2 = vload8(2, xp), v3 = vload8(3, xp);
            float s = (dot(lo0, (float4)(v0.s0, v0.s2, v0.s4, v0.s6)) + dot(hi0, (float4)(v0.s1, v0.s3, v0.s5, v0.s7)))
                    + (dot(lo1, (float4)(v1.s0, v1.s2, v1.s4, v1.s6)) + dot(hi1, (float4)(v1.s1, v1.s3, v1.s5, v1.s7)))
                    + (dot(lo2, (float4)(v2.s0, v2.s2, v2.s4, v2.s6)) + dot(hi2, (float4)(v2.s1, v2.s3, v2.s5, v2.s7)))
                    + (dot(lo3, (float4)(v3.s0, v3.s2, v3.s4, v3.s6)) + dot(hi3, (float4)(v3.s1, v3.s3, v3.s5, v3.s7)));
            float val = s * m * wt[i]; int tt = i / TK;
            UNR for (int z = 0; z < TB; z++) if (z == tt) acc[z] += val;
        }
    }
    UNR for (int t = 0; t < TB; t++) if (t < T) { float v = sub_group_reduce_add(acc[t]); if (l == 0) y[(size_t)t * N + row] = v; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void embed8T(__global const char* q, __global const float* sc, __global const int* tok, __global float* x, int H, float nrm) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    size_t n = tok[t];
    x[(size_t)t * H + i] = bfr(bfr((float)q[n * H + i] * sc[n]) * nrm);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gnormT(__global const float* x, __global const float* w, __global float* y, int H, float eps) {
    __local float r[64]; __local float tot;
    int l = get_local_id(0); size_t o = (size_t)get_group_id(0) * H; x += o; y += o; float s = 0.0f;
    for (int i = l; i < H; i += 1024) { float v = x[i]; s += v * v; }
    float k = rsqrt(bsum(s, r, &tot) / H + eps) * 1.0f;
    for (int i = l; i < H; i += 1024) { float v = x[i] * k; v *= w[i]; y[i] = v; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void postattnT(__global const float* o, __global const float* wpa, __global float* h, __global const float* w1, __global const float* w3,
                        __global float* t1, __global float* t3, __global float* t2, int H, float eps) {
    __local float r[64]; __local float tot; __local float hh[4096];
    size_t off = (size_t)get_group_id(0) * H; o += off; h += off; t1 += off; t3 += off; t2 += off;
    int l = get_local_id(0); float s = 0.0f;
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
    size_t off = (size_t)get_group_id(0) * H; d1 += off; e += off; h += off; tn += off;
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
__kernel void rlogitT(__global const float* w, __global const float* x, __global float* lg, int K, int E) {
    __local float r[8];
    int e = get_group_id(0), t = get_group_id(1), l = get_local_id(0); x += (size_t)t * K;
    float acc = 0.0f;
    for (int k = l * 4; k < K; k += 512) acc += dot(vload4(0, w + (size_t)e * K + k), vload4(0, x + k));
    acc = sub_group_reduce_add(acc);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (l == 0) { float s = 0.0f; for (int i = 0; i < 8; i++) s += r[i]; lg[t * E + e] = s; }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rtopT(__global const float* lg, __global const float* pes, __global int* idx, __global float* wt, int TK) {
    __local float v[128]; __local int rk[128]; __local int sel[16]; __local float pe[16];
    int L = get_local_id(0), e = L >> 3, j = L & 7, t = get_group_id(0); lg += t * 128; idx += t * TK; wt += t * TK;
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
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void gprepT(__global const float* p, __global const float* qw, __global const float* kw, __global float* Q, __global KT* kc, __global KT* vc,
                     int pos0, float theta, int NH, int KH, int HD, int NR, int VOFF, int CAP, float eps) {
    __local float r[64]; __local float tot; __local float sh[512];
    int g = get_group_id(0), t = get_group_id(1), l = get_local_id(0), NC = HD / 256, half_ = HD / 2, pos = pos0 + t;
    p += (size_t)t * (VOFF + KH * HD); Q += (size_t)t * NH * HD;
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
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void argmax1T(__global const float* v, int N, __global float* pv, __global int* pi, int t) {
    __local float rv[256]; __local int ri[256];
    int l = get_local_id(0), g = get_group_id(0); float bv = -INFINITY; int bi = 0; v += (size_t)t * N;
    for (int i = g * 4096 + l; i < min(N, (g + 1) * 4096); i += 256) { float x = v[i]; if (x > bv) { bv = x; bi = i; } }
    rv[l] = bv; ri[l] = bi; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = 128; s > 0; s >>= 1) { if (l < s && (rv[l + s] > rv[l] || (rv[l + s] == rv[l] && ri[l + s] < ri[l]))) { rv[l] = rv[l + s]; ri[l] = ri[l + s]; } barrier(CLK_LOCAL_MEM_FENCE); }
    if (l == 0) { pv[g] = rv[0]; pi[g] = ri[0]; }
}
__kernel void vscale(__global float* x, __global const float* s, int n) { int i = get_global_id(0); if (i < n) x[i] *= s[0]; }
"""



def _tsrc():
    """T-versions of the attention kernels (gattn, gattn3g, gattn3s, gcomb): same bodies, dimension 0 = (head, token tq), tokens fastest.
    Neighbouring groups read the same KV chunk, shared in L2. L = length of the first token (pos0+1), token tq has L + tq."""
    src, out = GR.GSRC, []
    for name, hv, GQ, HD in (("gattn", "h", "GQ", "HD"), ("gattn3g", "kh", "8", "512"), ("gattn3s", "kh", "2", "256"), ("gcomb", None, None, None)):
        a = src.index(f"__kernel void {name}("); b = src.index("\n}\n", a) + 3
        if name != "gcomb":
            a = src.rindex("__attribute__", 0, a)
        k = src[a:b].replace(f"__kernel void {name}(", f"__kernel void {name}T(", 1).replace("int WIN) {", "int WIN, int T) {", 1)
        if name == "gcomb":
            old = "int h = get_group_id(0), l = get_local_id(0), n = L - max(0, L - WIN), ns = (n + CH - 1) / CH;"
            new = ("int h = get_group_id(0) / T, tq = get_group_id(0) % T, l = get_local_id(0), NHh = get_num_groups(0) / T;\n"
                   "    L += tq; A += (size_t)tq * NHh * HD; { size_t ob = (size_t)tq * NHh * S; PM += ob; PL += ob; PO += ob * 512; }\n"
                   "    int n = L - max(0, L - WIN), ns = (n + CH - 1) / CH;")
            assert k.count(old) == 1; k = k.replace(old, new)
        else:
            old = f"int {hv} = get_group_id(0), s = get_group_id(1),"
            assert k.count(old) == 1, name
            k = k.replace(old, f"int {hv} = get_group_id(0) / T, tq = get_group_id(0) % T, s = get_group_id(1),")
            e = k.index("\n", k.index(f"int {hv} = get_group_id(0) / T")) + 1
            k = k[:e] + f"    L += tq; Q += (size_t)tq * KH * {GQ} * {HD}; {{ size_t ob = (size_t)tq * KH * {GQ} * S; PM += ob; PL += ob; PO += ob * 512; }}\n" + k[e:]
        out.append(k)
    return "\n".join(out) + "\n"



def _qsrc(name, HD, GQ, TG):
    """Grouped attention: work-group = (KV head kh, token group tg, chunk s). NQR = TG*GQ query rows
    stay in SLM; each key/value row is read once for all rows. Position range = union over the group, causal and window masks per row.
    Chunk output pb = (token*NH + head)*S + s, combined by gcombQ (fully masked chunks have M = -inf, L = 0)."""
    NQR, DV = GQ * TG, HD // 128
    F = f"float{DV}" if DV > 1 else "float"
    return f"""
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void {name}(__global const float* Q, __global const KT* kc, __global const KT* vc, __global float* PM, __global float* PL, __global float* PO,
                     int L0, int CAP, int KH, int T, int S, int CH, int WIN) {{
    __local float qs[{NQR * HD}], ps[{NQR} * 128], st[{3 * NQR}];
    int ntg = (T + {TG - 1}) / {TG}, kh = get_group_id(0) / ntg, tg = get_group_id(0) % ntg, s = get_group_id(1);
    int sg = get_sub_group_id(), l = get_sub_group_local_id(), lid = get_local_id(0), NH = KH * {GQ};
    int lo = max(0, L0 - WIN), n = L0 + T - 1 - lo, j0 = s * CH; if (j0 >= n) return;
    int j1 = min(n, j0 + CH), nt = min({TG}, T - tg * {TG});
    for (int i = lid; i < {NQR * HD}; i += 128) {{
        int r = i / {HD}, d = i - r * {HD}, tl = r / {GQ}, g = r - tl * {GQ};
        qs[i] = tl < nt ? Q[((size_t)(tg * {TG} + tl) * NH + kh * {GQ} + g) * {HD} + d] : 0.0f;
    }}
    if (lid < {NQR}) {{ st[lid] = -INFINITY; st[{NQR} + lid] = 0.0f; }}
    {F} o[{NQR}];
    UNR for (int r = 0; r < {NQR}; r++) o[r] = ({F})(0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int t0 = j0; t0 < j1; t0 += 128) {{
        int jj = t0 + sg * 16 + l, ja = lo + jj; float acc[{NQR}];
        UNR for (int r = 0; r < {NQR}; r++) acc[r] = 0.0f;
        if (jj < j1) {{
            __global const KT* kr = kc + ((size_t)(ja % CAP) * KH + kh) * {HD};
            for (int c = 0; c < {HD}; c += 8) {{
                float8 k = KLD8(0, kr + c);
                UNR for (int r = 0; r < {NQR}; r++) {{ float8 q = vload8(0, qs + r * {HD} + c); acc[r] += dot(q.lo, k.lo) + dot(q.hi, k.hi); }}
            }}
        }}
        UNR for (int r = 0; r < {NQR}; r++) {{
            int tl = r / {GQ}, Lt = L0 + tg * {TG} + tl;
            ps[r * 128 + sg * 16 + l] = (jj < j1 && tl < nt && ja < Lt && ja >= Lt - WIN) ? acc[r] : -INFINITY;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int r = sg; r < {NQR}; r += 8) {{
            float v[8], mx = -INFINITY;
            for (int i = 0; i < 8; i++) {{ v[i] = ps[r * 128 + i * 16 + l]; mx = fmax(mx, v[i]); }}
            float Mo = st[r], Mn = fmax(Mo, sub_group_reduce_max(mx)), al = Mo == -INFINITY ? 0.0f : exp(Mo - Mn), su = 0.0f;
            for (int i = 0; i < 8; i++) {{ float pp = v[i] == -INFINITY ? 0.0f : exp(v[i] - Mn); ps[r * 128 + i * 16 + l] = pp; su += pp; }}
            su = sub_group_reduce_add(su);
            sub_group_barrier(CLK_LOCAL_MEM_FENCE);
            if (l == 0) {{ st[r] = Mn; st[{NQR} + r] = st[{NQR} + r] * al + su; st[{2 * NQR} + r] = al; }}
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
        UNR for (int r = 0; r < {NQR}; r++) o[r] *= st[{2 * NQR} + r];
        int ntp = min(128, j1 - t0), i = 0;
        for (; i + 4 <= ntp; i += 4) {{
            {F} v0 = KLD{DV}(lid, vc + ((size_t)((lo + t0 + i) % CAP) * KH + kh) * {HD});
            {F} v1 = KLD{DV}(lid, vc + ((size_t)((lo + t0 + i + 1) % CAP) * KH + kh) * {HD});
            {F} v2 = KLD{DV}(lid, vc + ((size_t)((lo + t0 + i + 2) % CAP) * KH + kh) * {HD});
            {F} v3 = KLD{DV}(lid, vc + ((size_t)((lo + t0 + i + 3) % CAP) * KH + kh) * {HD});
            UNR for (int r = 0; r < {NQR}; r++) {{ __local const float* pp = ps + r * 128 + i; o[r] += pp[0] * v0 + pp[1] * v1 + pp[2] * v2 + pp[3] * v3; }}
        }}
        for (; i < ntp; i++) {{
            {F} v0 = KLD{DV}(lid, vc + ((size_t)((lo + t0 + i) % CAP) * KH + kh) * {HD});
            UNR for (int r = 0; r < {NQR}; r++) o[r] += ps[r * 128 + i] * v0;
        }}
        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    UNR for (int r = 0; r < {NQR}; r++) {{
        int tl = r / {GQ}, g = r - tl * {GQ};
        if (tl < nt) {{
            size_t pb = ((size_t)(tg * {TG} + tl) * NH + kh * {GQ} + g) * S + s;
            vstore{DV}(o[r], lid, PO + pb * 512);
            if (lid == 0) {{ PM[pb] = st[r]; PL[pb] = st[{NQR} + r]; }}
        }}
    }}
}}
"""


QCOMB = r"""
__kernel void gcombQ(__global const float* PM, __global const float* PL, __global const float* PO, __global float* A, int L0, int T, int HD, int S, int CH, int WIN) {
    int h = get_group_id(0), t = get_group_id(1), l = get_local_id(0), NH = get_num_groups(0);
    int lo = max(0, L0 - WIN), n = L0 + T - 1 - lo, ns = (n + CH - 1) / CH;
    size_t base = ((size_t)t * NH + h) * S;
    float M = -INFINITY; for (int s = 0; s < ns; s++) M = fmax(M, PM[base + s]);
    float tl = 0.0f; for (int s = 0; s < ns; s++) tl += PM[base + s] == -INFINITY ? 0.0f : PL[base + s] * exp(PM[base + s] - M);
    for (int d = l; d < HD; d += 256) {
        float o = 0.0f; for (int s = 0; s < ns; s++) o += PM[base + s] == -INFINITY ? 0.0f : PO[(base + s) * 512 + d] * exp(PM[base + s] - M);
        A[((size_t)t * NH + h) * HD + d] = o / tl;
    }
}
"""



LSRC = r"""
#define NL (16 * KS)
#define REDUCE_STORE(RLv) \
    UNR for (int t = 0; t < TT; t++) UNR for (int r = 0; r < RLv; r++) { float v_ = sub_group_reduce_add(acc[t][r]); if (l == 0) red[(sg * TT + t) * RLv + r] = v_; } \
    barrier(CLK_LOCAL_MEM_FENCE);
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemvL(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y, int N, int K, int add) {
    __local float red[@KD@ * TT * @RD@];
    int li = get_local_id(0), sg = li / 16, l = li % 16, r0 = get_group_id(1) * @RD@, KU = K / 8, FW = ((K / G) * 3 + 31) >> 5;
    float acc[TT][@RD@];
    UNR for (int t = 0; t < TT; t++) UNR for (int r = 0; r < @RD@; r++) acc[t][r] = 0.0f;
    for (int j = li; j < KU; j += 16 * @KD@) {
        float4 e[TT], o[TT];
        UNR for (int t = 0; t < TT; t++) { float8 v = vload8(j, x + (size_t)t * K); e[t] = v.even; o[t] = v.odd; }
        int bo = (j / (G / 8)) * 3, wo = bo >> 5, sh = bo & 31;
        UNR for (int r = 0; r < @RD@; r++) {
            size_t row = min(r0 + r, N - 1);
            __global const uint* fr = fac + row * FW;
            float m = (float)(((((ulong)fr[wo] | ((ulong)fr[wo + 1] << 32)) >> sh) & 7) + 1);
            uint w = codes[row * KU + j];
            float4 lo = convert_float4(as_char4(dlo(w))), hi = convert_float4(as_char4(dhi(w)));
            UNR for (int t = 0; t < TT; t++) acc[t][r] += (dot(lo, e[t]) + dot(hi, o[t])) * m;
        }
    }
    REDUCE_STORE(@RD@)
    for (int ix = li; ix < TT * @RD@; ix += 16 * @KD@) {
        int t = ix / @RD@, r = ix % @RD@, row = r0 + r;
        if (row < N) { float v = 0.0f; for (int q = 0; q < @KD@; q++) v += red[(q * TT + t) * @RD@ + r]; v *= vload_half(row, rs) * 0.125f; size_t o = (size_t)t * N + row; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void igemvL(__global const char* q, __global const float* sc, __global const float* x, __global float* y, int N, int K, int add) {
    __local float red[@KI@ * TT * @RI@];
    int li = get_local_id(0), sg = li / 16, l = li % 16, r0 = get_group_id(1) * @RI@, KU = K / 8;
    float acc[TT][@RI@];
    UNR for (int t = 0; t < TT; t++) UNR for (int r = 0; r < @RI@; r++) acc[t][r] = 0.0f;
    for (int j = li; j < KU; j += 16 * @KI@) {
        float8 xv[TT];
        UNR for (int t = 0; t < TT; t++) xv[t] = vload8(j, x + (size_t)t * K);
        UNR for (int r = 0; r < @RI@; r++) {
            float8 w = convert_float8(vload8(j, q + (size_t)min(r0 + r, N - 1) * K));
            UNR for (int t = 0; t < TT; t++) acc[t][r] += dot(w.lo, xv[t].lo) + dot(w.hi, xv[t].hi);
        }
    }
    REDUCE_STORE(@RI@)
    for (int ix = li; ix < TT * @RI@; ix += 16 * @KI@) {
        int t = ix / @RI@, r = ix % @RI@, row = r0 + r;
        if (row < N) { float v = 0.0f; for (int qq = 0; qq < @KI@; qq++) v += red[(qq * TT + t) * @RI@ + r]; v *= sc[row]; size_t o = (size_t)t * N + row; y[o] = add ? y[o] + v : v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xgemvEL(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                      int NE, int K, __global const int* uni, __global const uint* mlo, __global const uint* mhi, __global const int* cnt, int TK) {
    __local float red[@KE@ * TT * @RE@];
    int per = NE / @RE@, gi = get_group_id(1), u = gi / per, r0 = (gi % per) * @RE@;
    if (u >= cnt[0]) return;
    int li = get_local_id(0), sg = li / 16, l = li % 16, KU = K / 8, FW = ((K / G) * 3 + 31) >> 5;
    uint ma = mlo[u], mc = mhi[u]; int sl[TT];
    UNR for (int t = 0; t < TT; t++) { sl[t] = -1; for (int k = 0; k < TK; k++) { int i = t * TK + k; if (((i < 32 ? ma >> i : mc >> (i - 32)) & 1u)) sl[t] = i; } }
    size_t rbase = (size_t)uni[u] * NE;
    float acc[TT][@RE@];
    UNR for (int t = 0; t < TT; t++) UNR for (int r = 0; r < @RE@; r++) acc[t][r] = 0.0f;
    for (int j = li; j < KU; j += 16 * @KE@) {
        float4 e[TT], o[TT];
        UNR for (int t = 0; t < TT; t++) { float8 v = sl[t] >= 0 ? vload8(j, x + (size_t)t * K) : (float8)(0.0f); e[t] = v.even; o[t] = v.odd; }
        int bo = (j / (G / 8)) * 3, wo = bo >> 5, sh = bo & 31;
        UNR for (int r = 0; r < @RE@; r++) {
            size_t row = rbase + r0 + r;
            __global const uint* fr = fac + row * FW;
            float m = (float)(((((ulong)fr[wo] | ((ulong)fr[wo + 1] << 32)) >> sh) & 7) + 1);
            uint w = codes[row * KU + j];
            float4 lo = convert_float4(as_char4(dlo(w))), hi = convert_float4(as_char4(dhi(w)));
            UNR for (int t = 0; t < TT; t++) if (sl[t] >= 0) acc[t][r] += (dot(lo, e[t]) + dot(hi, o[t])) * m;
        }
    }
    REDUCE_STORE(@RE@)
    for (int ix = li; ix < TT * @RE@; ix += 16 * @KE@) {
        int t = ix / @RE@, r = ix % @RE@;
        if (sl[t] >= 0) { float v = 0.0f; for (int q = 0; q < @KE@; q++) v += red[(q * TT + t) * @RE@ + r]; v *= vload_half(rbase + r0 + r, rs) * 0.125f; y[(size_t)sl[t] * NE + r0 + r] = v; }
    }
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xdownEL(__global const uint* codes, __global const uint* fac, __global const half* rs, __global const float* x, __global float* y,
                      int N, int K, __global const int* uni, __global const uint* mlo, __global const uint* mhi, __global const int* cnt, __global const float* wt, int TK) {
    __local int lsl[64 * TT]; __local float red[@KO@ * TT * @RO@];
    int li = get_local_id(0), sg = li / 16, l = li % 16, nu = cnt[0], r0 = get_group_id(1) * @RO@, KU = K / 8, FW = ((K / G) * 3 + 31) >> 5;
    for (int uu = li; uu < nu; uu += 16 * @KO@) {
        uint ma = mlo[uu], mc = mhi[uu];
        for (int t = 0; t < TT; t++) { int s_ = -1; for (int k = 0; k < TK; k++) { int i = t * TK + k; if (((i < 32 ? ma >> i : mc >> (i - 32)) & 1u)) s_ = i; } lsl[uu * TT + t] = s_; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float acc[TT][@RO@];
    UNR for (int t = 0; t < TT; t++) UNR for (int r = 0; r < @RO@; r++) acc[t][r] = 0.0f;
    for (int p = li; p < nu * KU; p += 16 * @KO@) {
        int u = p / KU, j = p - u * KU;
        float4 e[TT], o[TT]; float wq[TT];
        UNR for (int t = 0; t < TT; t++) {
            int i = lsl[u * TT + t]; wq[t] = i >= 0 ? wt[i] : 0.0f;
            float8 v = vload8(j, x + (size_t)max(i, 0) * K); if (i < 0) v = (float8)(0.0f); e[t] = v.even; o[t] = v.odd;
        }
        int bo = (j / (G / 8)) * 3, wo = bo >> 5, sh = bo & 31;
        size_t eb = (size_t)uni[u] * N;
        UNR for (int r = 0; r < @RO@; r++) {
            size_t rr = eb + min(r0 + r, N - 1);
            __global const uint* fr = fac + rr * FW;
            float m = (float)(((((ulong)fr[wo] | ((ulong)fr[wo + 1] << 32)) >> sh) & 7) + 1) * vload_half(rr, rs) * 0.125f;
            uint w = codes[rr * KU + j];
            float4 lo = convert_float4(as_char4(dlo(w))), hi = convert_float4(as_char4(dhi(w)));
            UNR for (int t = 0; t < TT; t++) acc[t][r] += (dot(lo, e[t]) + dot(hi, o[t])) * (m * wq[t]);
        }
    }
    REDUCE_STORE(@RO@)
    for (int ix = li; ix < TT * @RO@; ix += 16 * @KO@) {
        int t = ix / @RO@, r = ix % @RO@, row = r0 + r;
        if (row < N) { float v = 0.0f; for (int q = 0; q < @KO@; q++) v += red[(q * TT + t) * @RO@ + r]; y[(size_t)t * N + row] = v; }
    }
}
"""


def st_load(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n; out = {}
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]; f.seek(base + a); raw = np.frombuffer(f.read(b - a), np.uint16)
        out[k] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(v["shape"])
    return out


def q8(w):
    s = np.abs(w).max(1) / 127.0; s[s == 0] = 1.0
    return np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)


class Plan:
    """A fixed kernel list; only position-dependent arguments and grids are updated before each run."""

    def __init__(self, q):
        self.q, self.ks, self.dyn = q, [], []

    def add(self, prg, name, g, l, *args, dyn=None):
        k = cl.Kernel(prg, name); k.set_args(*args); e = [k, g, l, args]; self.ks.append(e)
        if dyn:
            self.dyn.append((e, dyn))
        return e

    def run(self, **v):
        for e, f in self.dyn:
            f(e, v)
        for j, (k, g, l, _) in enumerate(self.ks):
            ev = cl.enqueue_nd_range_kernel(self.q, k, g, l)
            if PROF is not None:
                PROF.append((k.function_name, ev))
            if j % FL == FL - 1 or j == FL0:
                self.q.flush()
            if DBG:
                self.q.finish(); print("  ok", k.function_name, g, l, flush=True)
        if FLE:
            self.q.flush()


def _attn(m, prg3, L, Q, kc, vc, A, P: Plan, Lfun):
    """One attention (chunks + combine). Lfun(v) = current length."""
    I = np.int32; KH, HD, NH = L["KH"], L["HD"], m.NH
    if m.gqa == 2 and (L["kv"] or m.tile_s):
        prg, name, G = prg3, ("gattn3g" if HD == 512 else "gattn3s"), KH * 128
    else:
        prg, name, G = m.prg, "gattn", NH * 128

    def da(e, v):
        Lt = Lfun(v); n = min(Lt, L["WIN"]); e[0].set_arg(6, I(Lt)); e[1] = (G, -(-n // L["CHD"]))

    P.add(prg, name, (G, 1), (128, 1), Q, kc, vc, m.PM, m.PL, m.PO, I(1), I(L["CAP"]), I(KH), I(NH // KH), I(HD), I(m.SM), I(L["CHD"]), I(L["WIN"]), dyn=da)
    P.add(m.prg, "gcomb", (NH * 256,), (256,), m.PM, m.PL, m.PO, A, I(1), I(L["CAP"]), I(HD), I(m.SM), I(L["CHD"]), I(L["WIN"]), dyn=lambda e, v: e[0].set_arg(4, I(Lfun(v))))


class Multi:
    """Multi-token step (T <= TB) for the main model m."""

    def __init__(self, m: GR.Model):
        self.m = m; mf = cl.mem_flags; H, NH = m.H, m.NH
        opt = [f"-DG={m.meta['G']}", f"-DTB={TB}", f"-DR2E={knob('G4M_R2E', '2')}"]
        self.RT = int(knob("G4M_R", "2"))
        gw = next(L for L in m.lcfg if L["kv"]); sw = next(L for L in m.lcfg if not L["kv"])
        self.TGG = int(knob("G4M_TGG", "1")); self.TGS = int(knob("G4M_TGS", "2")); self.QG = knob("G4M_QG", "0") == "1"
        SRC = GR.GSRC + MSRC + _tsrc() + _qsrc("gattnQg", gw["HD"], NH // gw["KH"], self.TGG) + _qsrc("gattnQs", sw["HD"], NH // sw["KH"], self.TGS) + QCOMB
        self.prg = cl.Program(m.ctx, SRC).build(options=opt + [f"-DR={self.RT}"])
        self.prgG = cl.Program(m.ctx, SRC).build(options=opt + [f"-DR={self.RT}"] + (["-cl-intel-256-GRF-per-thread"] if knob("G4M_G256", "1") == "1" else []))
        f32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        PM = max(L["KH"] * L["HD"] * 2 + NH * L["HD"] for L in m.lcfg)
        self.toks, self.h, self.t, self.t2, self.t3, self.o, self.d1, self.e = f32(TB), f32(TB * H), f32(TB * H), f32(TB * H), f32(TB * H), f32(TB * H), f32(TB * H), f32(TB * H)
        self.P, self.Q, self.A = f32(TB * PM), f32(TB * NH * 512), f32(TB * NH * 512)
        self.GU, self.M = f32(TB * 2 * m.F), f32(TB * m.F)
        self.rl, self.idx, self.wt = f32(TB * m.NE), f32(TB * m.TK), f32(TB * m.TK)
        self.uni, self.mlo, self.mhi, self.cnt = f32(64), f32(64), f32(64), f32(1)
        self.EGU, self.EM = f32(TB * m.TK * 2 * m.FE), f32(TB * m.TK * m.FE)
        self.logit, self.outs = f32(TB * m.V), f32(TB)
        self.PM, self.PL, self.PO = f32(TB * NH * m.SM), f32(TB * NH * m.SM), f32(TB * NH * m.SM * 512)
        self.AT = int(knob("G4M_AT", "2"))
        self.plans = {}
        self.LG = knob("G4M_L", "1") == "1"
        self.KS = int(knob("G4M_KS", "8")); self.KSs = {k: int(knob("G4M_" + k, self.KS)) for k in ("KD", "KI", "KE", "KO")}; self.LE = knob("G4M_LE", "0") == "1"; self.E2 = knob("G4M_E2", "1") == "1"
        self.RLs = {k: int(knob("G4M_" + k, v)) for k, v in (("RD", "4"), ("RI", "4"), ("RE", "4"), ("RO", "4"))}
        self.prgLs = {}

    def prgL(self, T):
        if T not in self.prgLs:
            hdr = GR.GSRC[:GR.GSRC.index("// XeFP4-8 lin8 GEMV")]
            src = LSRC
            for k, v in list(self.RLs.items()) + list(self.KSs.items()):
                src = src.replace(f"@{k}@", str(v))
            opt = [f"-DG={self.m.meta['G']}", f"-DTT={T}", f"-DKS={self.KS}"] + (["-cl-intel-256-GRF-per-thread"] if knob("G4M_G256", "1") == "1" else [])
            self.prgLs[T] = cl.Program(self.m.ctx, hdr + src).build(options=opt)
        return self.prgLs[T]

    def lin(self, P, name, x, y, T):
        m, I = self.m, np.int32
        if self.LG:
            if name + ".q" in m.tm:
                N, Kd = m.tm[name + ".q"]["shape"]
                P.add(self.prgL(T), "igemvL", (16 * self.KSs["KI"], -(-N // self.RLs["RI"])), (16 * self.KSs["KI"], 1), m.buf[name + ".q"], m.buf[name + ".s"], x, y, I(N), I(Kd), I(0))
            else:
                t = m.tm[name + ".codes"]; N, Kd = t["shape"][0], t["shape"][1] * 2
                P.add(self.prgL(T), "xgemvL", (16 * self.KSs["KD"], -(-N // self.RLs["RD"])), (16 * self.KSs["KD"], 1), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, y, I(N), I(Kd), I(0))
            return
        if name + ".q" in m.tm:
            N, Kd = m.tm[name + ".q"]["shape"]
            P.add(self.prgG, "igemvT", (16, -(-(N // self.RT) // 8) * 8), (16, 8), m.buf[name + ".q"], m.buf[name + ".s"], x, y, I(N), I(Kd), I(T), I(0))
        else:
            t = m.tm[name + ".codes"]; N, Kd = t["shape"][0], t["shape"][1] * 2
            P.add(self.prgG, "xgemvT", (16, -(-(N // self.RT) // 8) * 8), (16, 8), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], x, y, I(N), I(Kd), I(T), I(0))

    def attnQ(self, P, L, kc, vc, T):
        m, I = self.m, np.int32; KH, HD, NH = L["KH"], L["HD"], m.NH
        name, TG = ("gattnQg", self.TGG) if L["kv"] else ("gattnQs", self.TGS)
        G = KH * (-(-T // TG)) * 128; CH = int(knob("G4M_QCH", "128"))

        def da(e, v):
            L0 = v["pos0"] + 1; n = L0 + T - 1 - max(0, L0 - L["WIN"]); e[0].set_arg(6, I(L0)); e[1] = (G, -(-n // CH))

        P.add(self.prgG, name, (G, 1), (128, 1), self.Q, kc, vc, self.PM, self.PL, self.PO, I(1), I(L["CAP"]), I(KH), I(T), I(m.SM), I(CH), I(L["WIN"]), dyn=da)
        P.add(self.prg, "gcombQ", (NH * 256, T), (256, 1), self.PM, self.PL, self.PO, self.A, I(1), I(T), I(HD), I(m.SM), I(CH), I(L["WIN"]),
              dyn=lambda e, v: e[0].set_arg(4, I(v["pos0"] + 1)))

    def attnT(self, P, L, kc, vc, T):
        m, I = self.m, np.int32; KH, HD, NH = L["KH"], L["HD"], m.NH
        if m.gqa == 2 and (L["kv"] or m.tile_s):
            prg, name, G = self.prgG, ("gattn3gT" if HD == 512 else "gattn3sT"), KH * 128 * T
        else:
            prg, name, G = self.prg, "gattnT", NH * 128 * T

        def da(e, v):
            n = min(v["pos0"] + T, L["WIN"]); e[0].set_arg(6, I(v["pos0"] + 1)); e[1] = (G, -(-n // L["CHD"]))

        P.add(prg, name, (G, 1), (128, 1), self.Q, kc, vc, self.PM, self.PL, self.PO, I(1), I(L["CAP"]), I(KH), I(NH // KH), I(HD), I(m.SM), I(L["CHD"]), I(L["WIN"]), I(T), dyn=da)
        P.add(self.prg, "gcombT", (NH * 256 * T,), (256,), self.PM, self.PL, self.PO, self.A, I(1), I(L["CAP"]), I(HD), I(m.SM), I(L["CHD"]), I(L["WIN"]), I(T),
              dyn=lambda e, v: e[0].set_arg(4, I(v["pos0"] + 1)))

    def plan(self, T):
        if T in self.plans:
            return self.plans[T]
        m, I, H, NH, eps = self.m, np.int32, self.m.H, self.m.NH, np.float32(self.m.eps)
        P = Plan(m.q); pr = self.prg
        P.add(pr, "embed8T", (-(-H // 256) * 256, T), (256, 1), m.buf["lm_head.q"], m.buf["lm_head.s"], self.toks, self.h, I(H), np.float32(53.0))
        P.add(pr, "gnormT", (1024 * T,), (1024,), self.h, m.buf["layers.0.in_norm"], self.t, I(H), eps)
        for i, L in enumerate(m.lcfg[:int(knob("G4M_NL", "99"))]):
            p = f"layers.{i}."; kc, vc = m.kv[i]; KH, HD = L["KH"], L["HD"]
            voff = NH * HD if L["kv"] else NH * HD + KH * HD
            self.lin(P, p + "proj", self.t, self.P, T)
            P.add(pr, "gprepT", ((NH + KH) * 256, T), (256, 1), self.P, m.buf[p + "q_norm"], m.buf[p + "k_norm"], self.Q, kc, vc, I(0), np.float32(L["theta"]), I(NH), I(KH), I(HD),
                  I(L["NR"]), I(voff), I(L["CAP"]), eps, dyn=lambda e, v: e[0].set_arg(6, I(v["pos0"])))
            if self.AT == 2 and (not L["kv"] or self.QG):
                self.attnQ(P, L, kc, vc, T)
            elif self.AT >= 1:
                self.attnT(P, L, kc, vc, T)
            else:
              for t in range(T):
                Qt = self.Q.get_sub_region(t * NH * HD * 4, NH * HD * 4); At = self.A.get_sub_region(t * NH * HD * 4, NH * HD * 4)
                _attn(m, m.prg3, L, Qt, kc, vc, At, P, (lambda tt: (lambda v: v["pos0"] + tt + 1))(t))
            self.lin(P, p + "out", self.A, self.o, T)
            P.add(pr, "postattnT", (1024 * T,), (1024,), self.o, m.buf[p + "post_attn"], self.h, m.buf[p + "pre_ff"], m.buf[p + "pre_ff2"], self.t, self.t3, self.t2, I(H), eps)
            self.lin(P, p + "gu", self.t, self.GU, T)
            P.add(m.prg, "geglu", (-(-m.F * T // 256) * 256,), (256,), self.GU, self.M, I(m.F), I(T))
            self.lin(P, p + "down", self.M, self.d1, T)
            P.add(pr, "rlogitT", (m.NE * 128, T), (128, 1), m.buf[p + "r_projF"], self.t2, self.rl, I(H), I(m.NE))
            P.add(pr, "rtopT", (1024 * T,), (1024,), self.rl, m.buf[p + "r_pes"], self.idx, self.wt, I(m.TK))
            n = T * m.TK; NEr = 2 * m.FE
            P.add(pr, "eunion", (64,), (64,), self.idx, I(n), self.uni, self.mlo, self.mhi, self.cnt)
            if self.LG and self.LE:
                P.add(self.prgL(T), "xgemvEL", (16 * self.KSs["KE"], min(64, n) * (NEr // self.RLs["RE"])), (16 * self.KSs["KE"], 1), m.buf[p + "egu.codes"], m.buf[p + "egu.fac"], m.buf[p + "egu.rs"], self.t3, self.EGU,
                      I(NEr), I(H), self.uni, self.mlo, self.mhi, self.cnt, I(m.TK))
            elif self.E2:
                R2E = int(knob("G4M_R2E", "2"))
                P.add(self.prgG, "xgemvE2", (16, -(-(min(64, n) * (NEr // R2E)) // 8) * 8), (16, 8), m.buf[p + "egu.codes"], m.buf[p + "egu.fac"], m.buf[p + "egu.rs"], self.t3, self.EGU,
                      I(NEr), I(H), self.uni, self.mlo, self.mhi, self.cnt, I(m.TK))
            else:
                P.add(self.prgG, "xgemvE", (16, -(-(min(64, n) * (NEr // self.RT)) // 8) * 8), (16, 8), m.buf[p + "egu.codes"], m.buf[p + "egu.fac"], m.buf[p + "egu.rs"], self.t3, self.EGU,
                      I(NEr), I(H), self.uni, self.mlo, self.mhi, self.cnt, I(m.TK))
            P.add(m.prg, "geglu", (-(-m.FE * n // 256) * 256,), (256,), self.EGU, self.EM, I(m.FE), I(n))
            if self.LG:
                P.add(self.prgL(T), "xdownEL", (16 * self.KSs["KO"], -(-H // self.RLs["RO"])), (16 * self.KSs["KO"], 1), m.buf[p + "edown.codes"], m.buf[p + "edown.fac"], m.buf[p + "edown.rs"], self.EM, self.e, I(H), I(m.FE),
                      self.uni, self.mlo, self.mhi, self.cnt, self.wt, I(m.TK))
            else:
                P.add(pr, "xdownE", (16, -(-H // 8) * 8), (16, 8), m.buf[p + "edown.codes"], m.buf[p + "edown.fac"], m.buf[p + "edown.rs"], self.EM, self.e, I(H), I(m.FE),
                      self.uni, self.mlo, self.mhi, self.cnt, self.wt, I(m.TK), I(T))
            nxt = m.buf[f"layers.{i + 1}.in_norm"] if i + 1 < len(m.lcfg) else m.buf["norm"]
            P.add(pr, "ffend2T", (1024 * T,), (1024,), self.d1, self.e, m.buf[p + "post_ff1"], m.buf[p + "post_ff2"], m.buf[p + "post_ff"], m.buf[p + "ls"], self.h, nxt, self.t, I(H), eps)
        self.lin(P, "lm_head", self.t, self.logit, T)
        for t in range(T):
            P.add(pr, "argmax1T", (m.npart * 256,), (256,), self.logit, I(m.V), m.pv, m.pi, I(t))
            P.add(m.prg, "argmax2", (256,), (256,), m.pv, m.pi, I(m.npart), m.tok, self.outs, I(t))
        self.plans[T] = P; return P

    def step(self, pos0, T):
        """toks[0:T] at positions pos0.. in one pass. outs[0:T] = greedy next tokens, t = final normed hidden states."""
        self.plan(T).run(pos0=pos0)


class Draft:
    """The assistant model (loaded as row-scaled INT8). Reads the KV of the last sliding and last global layer of the main model."""

    def __init__(self, m: GR.Model, folder, mp: Multi):
        self.m = m; mf = cl.mem_flags; c = json.load(open(folder + "/config.json"))["text_config"]
        self.Hd, self.F, self.NHd = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"]
        self.lt = c["layer_types"]; self.eps = np.float32(c["rms_norm_eps"])
        W = st_load(folder + "/model.safetensors"); self.buf = {}; self.shape = {}

        def up(name, a):
            self.buf[name] = cl.Buffer(m.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))

        def up8(name, w):
            q, s = q8(w); up(name + ".q", q); up(name + ".s", s); self.shape[name] = w.shape
        up8("pre", W["pre_projection.weight"]); up8("post", W["post_projection.weight"]); up8("emb", W["model.embed_tokens.weight"]); up("norm", W["model.norm.weight"])
        for i in range(len(self.lt)):
            p = f"model.layers.{i}."
            up8(f"{i}.q", W[p + "self_attn.q_proj.weight"]); up8(f"{i}.o", W[p + "self_attn.o_proj.weight"])
            up8(f"{i}.gu", np.concatenate([W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"]], 0)); up8(f"{i}.down", W[p + "mlp.down_proj.weight"])
            for a, b in (("in", "input_layernorm"), ("pa", "post_attention_layernorm"), ("pf", "pre_feedforward_layernorm"), ("pff", "post_feedforward_layernorm"), ("qw", "self_attn.q_norm")):
                up(f"{i}.{a}", W[p + b + ".weight"])
            up(f"{i}.ls", W[p + "layer_scalar"].astype(np.float32))
        del W
        f32 = lambda n: cl.Buffer(m.ctx, mf.READ_WRITE, max(4, int(n) * 4))
        self.Z = f32(2 * m.H); self.Ze, self.Zh = self.Z.get_sub_region(0, m.H * 4), self.Z.get_sub_region(m.H * 4, m.H * 4)
        self.x, self.a, self.q, self.Q, self.A, self.ob = f32(self.Hd), f32(self.Hd), f32(self.NHd * 512), f32(self.NHd * 512), f32(self.NHd * 512), f32(self.Hd)
        self.GU, self.Mb, self.xn, self.logit = f32(2 * self.F), f32(self.F), f32(self.Hd), f32(m.V)
        self.tok, self.douts = f32(1), f32(TB)
        sw = [i for i, L in enumerate(m.lcfg) if not L["kv"]]; gw = [i for i, L in enumerate(m.lcfg) if L["kv"]]
        self.src = {"sliding_attention": sw[-1], "full_attention": gw[-1]}
        I, H = np.int32, m.H; R = 2 if 2 in m.prgs else 1; pi = m.prgs[R]
        self.pe = Plan(m.q); self.pe.add(m.prg, "embed8", (-(-H // 256) * 256,), (256,), m.buf["lm_head.q"], m.buf["lm_head.s"], self.tok, self.Ze, I(H), np.float32(53.0))
        P = Plan(m.q); Hd = self.Hd

        def gemv(name, x, y):
            N, Kd = self.shape[name]
            P.add(pi, "igemv", (16, -(-(N // R) // 8) * 8), (16, 8), self.buf[name + ".q"], self.buf[name + ".s"], x, y, I(N), I(Kd), I(0))

        def gnorm(x, w, y):
            P.add(m.prg, "gnorm", (1024,), (1024,), x, w, w, y, I(Hd), self.eps, np.float32(1.0), I(1), I(0))
        gemv("pre", self.Z, self.x)
        for li, lt in enumerate(self.lt):
            L = m.lcfg[self.src[lt]]; kc, vc = m.kv[self.src[lt]]; KH, HD = L["KH"], L["HD"]
            gnorm(self.x, self.buf[f"{li}.in"], self.a); gemv(f"{li}.q", self.a, self.q)
            P.add(m.prg, "gprep", (self.NHd * 256,), (256,), self.q, self.buf[f"{li}.qw"], self.buf[f"{li}.qw"], self.Q, kc, vc, I(0), np.float32(L["theta"]), I(self.NHd), I(KH), I(HD),
                  I(L["NR"]), I(0), I(L["CAP"]), self.eps, dyn=lambda e, v: e[0].set_arg(6, I(v["pos"])))
            _attn(m, m.prg3, L, self.Q, kc, vc, self.A, P, lambda v: v["kvlen"])
            gemv(f"{li}.o", self.A, self.ob)
            P.add(m.prg, "normadd", (1024,), (1024,), self.ob, self.buf[f"{li}.pa"], self.x, I(Hd), self.eps)
            gnorm(self.x, self.buf[f"{li}.pf"], self.a); gemv(f"{li}.gu", self.a, self.GU)
            P.add(m.prg, "geglu", (-(-self.F // 256) * 256,), (256,), self.GU, self.Mb, I(self.F), I(1))
            gemv(f"{li}.down", self.Mb, self.ob)
            P.add(m.prg, "normadd", (1024,), (1024,), self.ob, self.buf[f"{li}.pff"], self.x, I(Hd), self.eps)
            P.add(mp.prg, "vscale", (-(-Hd // 256) * 256,), (256,), self.x, self.buf[f"{li}.ls"], I(Hd))
        gnorm(self.x, self.buf["norm"], self.xn)
        gemv("emb", self.xn, self.logit)
        P.add(m.prg, "argmax1", (m.npart * 256,), (256,), self.logit, I(m.V), m.pv, m.pi)
        P.add(m.prg, "argmax2", (256,), (256,), m.pv, m.pi, I(m.npart), self.tok, self.douts, I(0), dyn=lambda e, v: e[0].set_arg(5, I(v["i"])))
        gemv("post", self.xn, self.Zh)
        P.ks += self.pe.ks
        self.ps = P

    def embed(self):
        self.pe.run()

    def step(self, pos, kvlen, i):
        """One draft step: douts[i] and tok = drafted token, Zh = next hidden state, Ze = next embedding."""
        self.ps.run(pos=pos, kvlen=kvlen, i=i)
