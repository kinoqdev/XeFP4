"""Qwen3.5-9B XeFP4 prompt processing (prefill): T prompt tokens at once.
- Linear layers: XeFP4 weights are decoded once per layer into the BF16 XMX B layout and multiplied with
  BF16 (or INT8, W4A8) activations by DPAS; output FP32 plus residual.
- Linear attention: causal convolution per channel, then the gated delta rule with the state kept in registers.
- Full attention: per-head norm, rotary, KV cache write, causal attention as two XMX matrix products.
- Only the last token goes through lm_head; decoding continues in qwen35.model.
"""
from __future__ import annotations


import numpy as np
import pyopencl as cl

from .model import Model
from ..device import knob


def _nosplit(dev):
    return [] if "cl_intel_subgroup_split_matrix_multiply_accumulate" in dev.extensions else ["-DNOSPLIT"]


def _slmbw():
    """SLM stores as sub-group block writes (default; checked on the Arc A770 and 140T). XEQ_SLMBW=0 restores scalar stores."""
    return ["-DSLMBW"] if knob("SLMBW", "1") == "1" else []

GSRC = r"""
#define UN __attribute__((opencl_unroll_hint))
#ifdef NOSPLIT
#define ATILE int8
#define AREAD(p, h) as_int8(intel_sub_group_block_read8(p))
#define MADBF(a, b, c) intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, c)
#define MADI8(a, b, c) intel_sub_group_i8_i8_matrix_mad_k32(a, b, c)
#else
#define ATILE int4
#define AREAD(p, h) as_int4(intel_sub_group_block_read4((p) + (h) * 32))
#define MADBF(a, b, c) intel_sub_group_bf16_bf16_split_matrix_mad_k16(a, b, c)
#define MADI8(a, b, c) intel_sub_group_i8_i8_split_matrix_mad_k32(a, b, c)
#endif
#define NSG (WM * WN)
#define TR (WM * RM)
#define TC (WN * NT)
#define NA (TR * UK)
#define TOT ((TR + TC) * UK)
#define PER ((TOT + NSG - 1) / NSG)
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void pgemm(__global const uint* A, __global const uint* B, __global float* Y, int M, int N, int K, int ldy, int add) {
    __local uint sA[2][NA * 64], sB[2][TC * UK * 64];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 16;
    int r0 = get_group_id(1) * TR, c0 = get_group_id(0) * TC;
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    __global const uint* gb = B + (size_t)c0 * KB * 64;
    float8 acc[RM][NT];
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = (float8)(0.f);
    uint8 st[PER];
#define GSRC(j, kb) ((j) < NA ? ga + ((size_t)((j) / UK) * KB + (kb) + (j) % UK) * 64 \
                              : gb + ((size_t)(((j) - NA) / UK) * KB + (kb) + ((j) - NA) % UK) * 64)
#define LDST(buf, j) ((j) < NA ? &sA[buf][(j) * 64] : &sB[buf][((j) - NA) * 64])
#define LOADG(kb) UN for (int q = 0; q < PER; q++) { int j = sg + q * NSG; if (j < TOT) st[q] = intel_sub_group_block_read8(GSRC(j, kb)); }
#ifdef SLMBW   /* SLM stores as sub-group block writes (same layout) */
#define STORE(buf) UN for (int q = 0; q < PER; q++) { int j = sg + q * NSG; if (j < TOT) intel_sub_group_block_write8(LDST(buf, j), st[q]); }
#else
#define STORE(buf) UN for (int q = 0; q < PER; q++) { int j = sg + q * NSG; if (j < TOT) { __local uint* d = LDST(buf, j); UN for (int i = 0; i < 8; i++) d[i * 8 + l] = st[q][i]; } }
#endif
    LOADG(0);
    STORE(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kb = 0; kb < KB; kb += UK) {
        int cur = (kb / UK) & 1;
        int more = kb + UK < KB;
        if (more) { LOADG(kb + UK); }
        UN for (int u = 0; u < UK; u++) {
            int8 b[NT];
            UN for (int c = 0; c < NT; c++) b[c] = as_int8(intel_sub_group_block_read8(&sB[cur][((sc * NT + c) * UK + u) * 64]));
            ATILE a[RM];
            UN for (int r = 0; r < RM; r++) a[r] = AREAD(&sA[cur][((sr * RM + r) * UK + u) * 64], h);
            UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = MADBF(a[r], b[c], acc[r][c]);
        }
        if (more) { STORE(cur ^ 1); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int rt = r0 + sr * RM, ct = c0 + sc * NT;
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) UN for (int m = 0; m < 8; m++) {
        int row = (rt + r) * 8 + m, col = (ct + c) * 8 + l;
        if (row < M && col < N) { size_t o = (size_t)row * ldy + col; Y[o] = add ? Y[o] + acc[r][c][m] : acc[r][c][m]; }
    }
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
inline uint bfp(float a, float b) {
    uint x = as_uint(a), y = as_uint(b);
    return ((x + 0x7FFFu + ((x >> 16) & 1u)) >> 16) | (((y + 0x7FFFu + ((y >> 16) & 1u)) >> 16) << 16);
}
#define NB (TC * UK)
#define PA ((NA + NSG - 1) / NSG)
#define PB ((NB + NSG - 1) / NSG)
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void pgemm4(__global const uint* A, __global const uint* codes, __global const uint* fac, __global const half* rs,
                     __global float* Y, int M, int N, int K, int ldy, int add) {
    __local uint sA[2][NA * 64], sB[2][NB * 64];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 16, nbk = K / G, FW = nbk * 3 / 32;
    int r0 = get_group_id(1) * TR, c0 = get_group_id(0) * TC;
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    float8 acc[RM][NT];
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = (float8)(0.f);
    uint8 stA[PA], stB[PB];
#define LOADG4(kb) \
    UN for (int q = 0; q < PA; q++) { int j = sg + q * NSG; if (j < NA) stA[q] = intel_sub_group_block_read8(ga + ((size_t)(j / UK) * KB + (kb) + j % UK) * 64); } \
    UN for (int q = 0; q < PB; q++) { int j = sg + q * NSG; if (j < NB) { \
        int n = (c0 + j / UK) * 8 + l, kq = (kb) + j % UK; uint8 o = (uint8)(0); \
        if (n < N) { \
            uint2 w = vload2(0, codes + (size_t)n * (K / 8) + kq * 2); \
            if (NODEC) { o = (uint8)(w.x, w.y, w.x ^ 1u, w.y ^ 1u, w.x ^ 2u, w.y ^ 2u, w.x ^ 3u, w.y ^ 3u); } else if (IB) { \
            int bo = ((kq * 16) / G) * 3; __global const uint* fr = fac + (size_t)n * FW; \
            ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32); \
            char mq = (char)(((fw >> (bo & 31)) & 7) + 1); \
            char4 a0 = as_char4(dlo(w.x)) * mq, b0 = as_char4(dhi(w.x)) * mq, a1 = as_char4(dlo(w.y)) * mq, b1 = as_char4(dhi(w.y)) * mq; \
            uint4 ua0 = as_uint4(convert_float4(a0)) >> 16, ub0 = as_uint4(convert_float4(b0)) >> 16; \
            uint4 ua1 = as_uint4(convert_float4(a1)) >> 16, ub1 = as_uint4(convert_float4(b1)) >> 16; \
            uint4 p0 = ua0 | (ub0 << 16), p1 = ua1 | (ub1 << 16); \
            o = (uint8)(p0.s0, p0.s1, p0.s2, p0.s3, p1.s0, p1.s1, p1.s2, p1.s3); } else { \
            int bo =((kq * 16) / G) * 3; __global const uint* fr = fac + (size_t)n * FW; \
            ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32); \
            float s = vload_half(n, rs) * 0.125f * (float)(((fw >> (bo & 31)) & 7) + 1); \
            float4 a0 = convert_float4(as_char4(dlo(w.x))) * s, b0 = convert_float4(as_char4(dhi(w.x))) * s; \
            float4 a1 = convert_float4(as_char4(dlo(w.y))) * s, b1 = convert_float4(as_char4(dhi(w.y))) * s; \
            o = (uint8)(bfp(a0.s0, b0.s0), bfp(a0.s1, b0.s1), bfp(a0.s2, b0.s2), bfp(a0.s3, b0.s3), \
                        bfp(a1.s0, b1.s0), bfp(a1.s1, b1.s1), bfp(a1.s2, b1.s2), bfp(a1.s3, b1.s3)); } \
        } \
        stB[q] = o; } }
#ifdef SLMBW
#define STORE4(buf) \
    UN for (int q = 0; q < PA; q++) { int j = sg + q * NSG; if (j < NA) intel_sub_group_block_write8(&sA[buf][j * 64], stA[q]); } \
    UN for (int q = 0; q < PB; q++) { int j = sg + q * NSG; if (j < NB) intel_sub_group_block_write8(&sB[buf][j * 64], stB[q]); }
#else
#define STORE4(buf) \
    UN for (int q = 0; q < PA; q++) { int j = sg + q * NSG; if (j < NA) { UN for (int i = 0; i < 8; i++) sA[buf][j * 64 + i * 8 + l] = stA[q][i]; } } \
    UN for (int q = 0; q < PB; q++) { int j = sg + q * NSG; if (j < NB) { UN for (int i = 0; i < 8; i++) sB[buf][j * 64 + i * 8 + l] = stB[q][i]; } }
#endif
    LOADG4(0);
    STORE4(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kb = 0; kb < KB; kb += UK) {
        int cur = (kb / UK) & 1;
        int more = kb + UK < KB;
        if (more) { LOADG4(kb + UK); }
        UN for (int u = 0; u < UK; u++) {
            int8 b[NT];
            UN for (int c = 0; c < NT; c++) b[c] = as_int8(intel_sub_group_block_read8(&sB[cur][((sc * NT + c) * UK + u) * 64]));
            ATILE a[RM];
            UN for (int r = 0; r < RM; r++) a[r] = AREAD(&sA[cur][((sr * RM + r) * UK + u) * 64], h);
            UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = MADBF(a[r], b[c], acc[r][c]);
        }
        if (more) { STORE4(cur ^ 1); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int rt = r0 + sr * RM, ct = c0 + sc * NT;
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) UN for (int m = 0; m < 8; m++) {
        int row = (rt + r) * 8 + m, col = (ct + c) * 8 + l;
        if (row < M && col < N) {
            float v = acc[r][c][m];
            if (IB) v *= vload_half(col, rs) * 0.125f;
            size_t o = (size_t)row * ldy + col; Y[o] = add ? Y[o] + v : v;
        }
    }
}

inline uint il0(uint lo, uint hi) { return (lo & 0xFFu) | ((hi & 0xFFu) << 8) | ((lo & 0xFF00u) << 8) | ((hi & 0xFF00u) << 16); }
inline uint il1(uint lo, uint hi) { return ((lo >> 16) & 0xFFu) | (((hi >> 16) & 0xFFu) << 8) | ((lo >> 8) & 0xFF0000u) | (hi & 0xFF000000u); }
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void pgemm8(__global const uint* A, __global const float* SAt, int Mp, __global const uint* codes, __global const uint* fac, __global const half* rs,
                     __global float* Y, int M, int N, int K, int ldy, int add) {
    __local uint sA[2][NA * 64], sB[2][NB * 64];
    __local float sSA[2][TR * 8 * UK];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 32, FW = (K / G) * 3 / 32;
    int r0 = get_group_id(1) * TR, c0 = get_group_id(0) * TC;
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    float8 acc[RM][NT];
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = (float8)(0.f);
    uint8 stA[PA], stB[PB];
#define NSA (TR * 8 * UK)
#define PSA ((NSA + NSG * 8 - 1) / (NSG * 8))
    float stS[PSA];
    int tid = sg * 8 + l;
#define LOADG8(kb)     UN for (int q = 0; q < PSA; q++) { int x = tid + q * NSG * 8; if (x < NSA) stS[q] = SAt[(size_t)((kb) + x / (TR * 8)) * Mp + r0 * 8 + x % (TR * 8)]; } \
    UN for (int q = 0; q < PA; q++) { int j = sg + q * NSG; if (j < NA) stA[q] = intel_sub_group_block_read8(ga + ((size_t)(j / UK) * KB + (kb) + j % UK) * 64); } \
    UN for (int q = 0; q < PB; q++) { int j = sg + q * NSG; if (j < NB) { \
        int n = (c0 + j / UK) * 8 + l, kq = (kb) + j % UK; uint8 o = (uint8)(0); \
        if (n < N) { \
            uint4 w = vload4(0, codes + (size_t)n * (K / 8) + kq * 4); \
            int bo = ((kq * 32) / G) * 3; __global const uint* fr = fac + (size_t)n * FW; \
            ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32); \
            char mq = (char)(((fw >> (bo & 31)) & 7) + 1); \
            uint l0 = as_uint(as_char4(dlo(w.s0)) * mq), h0 = as_uint(as_char4(dhi(w.s0)) * mq); \
            uint l1 = as_uint(as_char4(dlo(w.s1)) * mq), h1 = as_uint(as_char4(dhi(w.s1)) * mq); \
            uint l2 = as_uint(as_char4(dlo(w.s2)) * mq), h2 = as_uint(as_char4(dhi(w.s2)) * mq); \
            uint l3 = as_uint(as_char4(dlo(w.s3)) * mq), h3 = as_uint(as_char4(dhi(w.s3)) * mq); \
            o = (uint8)(il0(l0, h0), il1(l0, h0), il0(l1, h1), il1(l1, h1), il0(l2, h2), il1(l2, h2), il0(l3, h3), il1(l3, h3)); \
        } \
        stB[q] = o; } }
#define STORE8(buf) STORE4(buf); UN for (int q = 0; q < PSA; q++) { int x = tid + q * NSG * 8; if (x < NSA) sSA[buf][x] = stS[q]; }
    LOADG8(0);
    STORE8(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kb = 0; kb < KB; kb += UK) {
        int cur = (kb / UK) & 1;
        int more = kb + UK < KB;
        if (more) { LOADG8(kb + UK); }
        UN for (int u = 0; u < UK; u++) {
            int8 b[NT];
            UN for (int c = 0; c < NT; c++) b[c] = as_int8(intel_sub_group_block_read8(&sB[cur][((sc * NT + c) * UK + u) * 64]));
            ATILE a[RM]; float8 sa[RM];
            UN for (int r = 0; r < RM; r++) {
                a[r] = AREAD(&sA[cur][((sr * RM + r) * UK + u) * 64], h);
                sa[r] = vload8(0, &sSA[cur][u * TR * 8 + (sr * RM + r) * 8]);
            }
            UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++)
                acc[r][c] += convert_float8(MADI8(a[r], b[c], (int8)(0))) * sa[r];
        }
        if (more) { STORE8(cur ^ 1); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int rt = r0 + sr * RM, ct = c0 + sc * NT;
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) UN for (int m = 0; m < 8; m++) {
        int row = (rt + r) * 8 + m, col = (ct + c) * 8 + l;
        if (row < M && col < N) {
            float v = acc[r][c][m] * vload_half(col, rs) * 0.125f;
            size_t o = (size_t)row * ldy + col; Y[o] = add ? Y[o] + v : v;
        }
    }
}

// TILE-SCRATCH: XeFP4 row-major -> INT8 q*m in the DPAS B block layout
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void tileB8(__global const uint* codes, __global const uint* fac, __global uint* Bt, int N, int K) {
    int l = get_sub_group_local_id(), kq = get_global_id(0) / 8, nt = get_global_id(1), n = nt * 8 + l;
    int KB = K / 32, FW = (K / G) * 3 / 32, bo = ((kq * 32) / G) * 3;
    uint8 o = (uint8)(0);
    if (n < N) {
        __global const uint* fr = fac + (size_t)n * FW;
        ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
        char mq = (char)(((fw >> (bo & 31)) & 7) + 1);
        uint4 w = vload4(0, codes + (size_t)n * (K / 8) + kq * 4);
        uint l0 = as_uint(as_char4(dlo(w.s0)) * mq), h0 = as_uint(as_char4(dhi(w.s0)) * mq);
        uint l1 = as_uint(as_char4(dlo(w.s1)) * mq), h1 = as_uint(as_char4(dhi(w.s1)) * mq);
        uint l2 = as_uint(as_char4(dlo(w.s2)) * mq), h2 = as_uint(as_char4(dhi(w.s2)) * mq);
        uint l3 = as_uint(as_char4(dlo(w.s3)) * mq), h3 = as_uint(as_char4(dhi(w.s3)) * mq);
        o = (uint8)(il0(l0, h0), il1(l0, h0), il0(l1, h1), il1(l1, h1), il0(l2, h2), il1(l2, h2), il0(l3, h3), il1(l3, h3));
    }
    intel_sub_group_block_write8(Bt + ((size_t)nt * KB + kq) * 64, o);
}
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void pgemm8t(__global const uint* A, __global const float* SAt, int Mp, __global const uint* Bt, __global const half* rs,
                      __global float* Y, int M, int N, int K, int ldy, int add) {
    __local uint sA[2][NA * 64], sB[2][NB * 64];
    __local float sSA[2][TR * 8 * UK];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 32;
    int r0 = get_group_id(1) * TR, c0 = get_group_id(0) * TC;
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    __global const uint* gb = Bt + (size_t)c0 * KB * 64;
    float8 acc[RM][NT];
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = (float8)(0.f);
    uint8 stA[PA], stB[PB];
    float stS[PSA];
    int tid = sg * 8 + l;
#define LOADGT(kb) \
    UN for (int q = 0; q < PSA; q++) { int x = tid + q * NSG * 8; if (x < NSA) stS[q] = SAt[(size_t)((kb) + x / (TR * 8)) * Mp + r0 * 8 + x % (TR * 8)]; } \
    UN for (int q = 0; q < PA; q++) { int j = sg + q * NSG; if (j < NA) stA[q] = intel_sub_group_block_read8(ga + ((size_t)(j / UK) * KB + (kb) + j % UK) * 64); } \
    UN for (int q = 0; q < PB; q++) { int j = sg + q * NSG; if (j < NB) stB[q] = intel_sub_group_block_read8(gb + ((size_t)(j / UK) * KB + (kb) + j % UK) * 64); }
    LOADGT(0);
    STORE8(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kb = 0; kb < KB; kb += UK) {
        int cur = (kb / UK) & 1;
        int more = kb + UK < KB;
        if (more) { LOADGT(kb + UK); }
        UN for (int u = 0; u < UK; u++) {
            int8 b[NT];
            UN for (int c = 0; c < NT; c++) b[c] = as_int8(intel_sub_group_block_read8(&sB[cur][((sc * NT + c) * UK + u) * 64]));
            ATILE a[RM]; float8 sa[RM];
            UN for (int r = 0; r < RM; r++) {
                a[r] = AREAD(&sA[cur][((sr * RM + r) * UK + u) * 64], h);
                sa[r] = vload8(0, &sSA[cur][u * TR * 8 + (sr * RM + r) * 8]);
            }
            UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++)
                acc[r][c] += convert_float8(MADI8(a[r], b[c], (int8)(0))) * sa[r];
        }
        if (more) { STORE8(cur ^ 1); }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int rt = r0 + sr * RM, ct = c0 + sc * NT;
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) UN for (int m = 0; m < 8; m++) {
        int row = (rt + r) * 8 + m, col = (ct + c) * 8 + l;
        if (row < M && col < N) {
            float v = acc[r][c][m] * vload_half(col, rs) * 0.125f;
            size_t o = (size_t)row * ldy + col; Y[o] = add ? Y[o] + v : v;
        }
    }
}
"""

PSRC = r"""
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

#ifndef QPG
#define QPG 4
#endif
#define UNR __attribute__((opencl_unroll_hint))
inline uint bf16(float f) { uint u = as_uint(f); return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
inline float wsum(float v, __local float* r) {
    int l = get_local_id(0), n = get_local_size(0); r[l] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) { if (l < s) r[l] += r[l + s]; barrier(CLK_LOCAL_MEM_FENCE); }
    float o = r[0]; barrier(CLK_LOCAL_MEM_FENCE); return o;
}
inline float silu(float v) { return v / (1.0f + exp(-v)); }
#if XE8
__constant float LUT[8] = {0.f, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 12.f};
inline float dq(uint c) { float v = LUT[c & 7]; return (c & 8) ? -v : v; }
#else
inline float dq(uint c) { return (float)((int)c - 8); }
#endif
__kernel void decB(__global const uint* codes, __global const uint* fac, __global const half* rs, __global uint* Bp, int N, int K) {
    int n = get_global_id(0), c = get_global_id(1); if (n >= N) return;
    int nb = K / G, FW = nb * 3 / 32, blk = (c * 32) / G, bo = blk * 3;
    __global const uint* fr = fac + (size_t)n * FW;
    ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
    float sc = vload_half(n, rs) * 0.125f * (float)(((fw >> (bo & 31)) & 7) + 1);
    uint4 u = vload4(c, codes + (size_t)n * (K / 8));
    int KB = K / 16, nt = n / 8, nn = n % 8;
    __global uint* bp = Bp + (size_t)nt * KB * 64 + nn;
    UNR for (int q = 0; q < 16; q++) {
        uint w = q < 4 ? u.s0 : q < 8 ? u.s1 : q < 12 ? u.s2 : u.s3;
        uint by = (w >> (8 * (q & 3))) & 0xFFu;
        uint pr = bf16(dq(by & 15u) * sc) | (bf16(dq(by >> 4) * sc) << 16);
        int kb = 2 * c + q / 8, kk = q % 8;
        bp[(size_t)kb * 64 + kk * 8] = pr;
    }
}
__kernel void packA(__global const float* X, __global uint* Ap, int K, int ldx) {
    int c2 = get_global_id(0), t = get_global_id(1);
    int k = 2 * c2, KB = K / 16;
    float a = X[(size_t)t * ldx + k], b = X[(size_t)t * ldx + k + 1];
    Ap[((size_t)(t / 8) * KB + k / 16) * 64 + (t % 8) * 8 + c2 % 8] = bf16(a) | (bf16(b) << 16);
}
__kernel void embedT_xe(__global const uchar* codes, __global const uint* fac, __global const half* rs, __global const int* ids, int i0, __global float* X, int H) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    size_t n = ids[i0 + t]; int FW = (H / G) * 3 / 32, bo = (i / G) * 3;
    __global const uint* fr = fac + n * FW;
    ulong fw = (ulong)fr[bo >> 5] | ((ulong)fr[(bo >> 5) + 1] << 32);
    uint by = codes[n * (H / 2) + i / 2], c = (i & 1) ? (by >> 4) : (by & 15u);
    X[(size_t)t * H + i] = dq(c) * vload_half(n, rs) * 0.125f * (float)(((fw >> (bo & 31)) & 7) + 1);
}
__kernel void embedT(__global const ushort* E, __global const int* ids, int i0, __global float* X, int H) {
    int i = get_global_id(0), t = get_global_id(1); if (i < H) X[(size_t)t * H + i] = as_float(((uint)E[(size_t)ids[i0 + t] * H + i]) << 16);
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsnormT(__global const float* X, __global const float* w, __global float* Y, int H, float eps, int plus1) {
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
__kernel void convT(__global const float* P, int ldp, __global float* st, __global const float* w, __global float* Cx, int C, int T) {
    int c = get_global_id(0); if (c >= C) return;
    float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; t++) {
        float x = P[(size_t)t * ldp + c];
        float v = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
        s0 = s1; s1 = s2; s2 = x;
        Cx[(size_t)t * C + c] = silu(v);
    }
    st[c] = s0; st[C + c] = s1; st[2 * C + c] = s2;
}
__kernel void qkprep(__global const float* Cx, int C, int KD, __global float* Qn, __global float* Kn, __global float* KQ) {
    __local float r[128];
    int kh = get_group_id(0), t = get_group_id(1), d = get_local_id(0), NKH = get_num_groups(0);
    float q = Cx[(size_t)t * C + kh * 128 + d], k = Cx[(size_t)t * C + KD + kh * 128 + d];
    float nq = wsum(q * q, r), nk = wsum(k * k, r);
    float qn = q * rsqrt(nq + 1e-6f) * 0.08838834764831845f, kn = k * rsqrt(nk + 1e-6f);
    float kq = wsum(qn * kn, r);
    size_t o = ((size_t)t * NKH + kh) * 128 + d;
    Qn[o] = qn; Kn[o] = kn; if (d == 0) KQ[(size_t)t * NKH + kh] = kq;
}
__kernel void dnetT(__global const float* Cx, int C, int KD, __global const float* P, int ldp, int BO, int AO,
                    __global const float* Qn, __global const float* Kn, __global const float* KQ,
                    __global const float* A_log, __global const float* dtb,
                    __global float* S, __global float* PK, __global float* PD, __global float* PE, __global float* O, int VH, int T) {
    __local float pa1[2][4][32], pa2[2][4][32];
    int h = get_group_id(0), dc = get_group_id(1), lid = get_local_id(0), dd = lid & 31, part = lid >> 5;
    int d = dc * 32 + dd, NKH = KD / 128, kh = h / (VH / NKH), i0 = part * 32;
    __global float* Sh = S + (size_t)h * 128 * 128;
    float pe = PE[h], pdl = PD[h * 128 + d];
    float s[32];
    UNR for (int j = 0; j < 32; j++) s[j] = Sh[(i0 + j) * 128 + d] * pe + PK[h * 128 + i0 + j] * pdl;
    float alh = exp(A_log[h]), db = dtb[h];
    int buf = 0;
    for (int t = 0; t < T; t++) {
        __global const float* kp = Kn + ((size_t)t * NKH + kh) * 128 + i0;
        __global const float* qp = Qn + ((size_t)t * NKH + kh) * 128 + i0;
        float a1 = 0.0f, a2 = 0.0f;
        UNR for (int j = 0; j < 32; j++) { a1 += s[j] * kp[j]; a2 += s[j] * qp[j]; }
        pa1[buf][part][dd] = a1; pa2[buf][part][dd] = a2;
        barrier(CLK_LOCAL_MEM_FENCE);
        a1 = pa1[buf][0][dd] + pa1[buf][1][dd] + pa1[buf][2][dd] + pa1[buf][3][dd];
        a2 = pa2[buf][0][dd] + pa2[buf][1][dd] + pa2[buf][2][dd] + pa2[buf][3][dd];
        __global const float* pr = P + (size_t)t * ldp;
        float bt = 1.0f / (1.0f + exp(-pr[BO + h]));
        float av = pr[AO + h] + db; float sp = av > 20.0f ? av : log1p(exp(av));
        float eg = exp(-alh * sp);
        float v = Cx[(size_t)t * C + 2 * KD + h * 128 + d];
        float dl = (v - eg * a1) * bt;
        float o = eg * a2 + dl * KQ[(size_t)t * NKH + kh];
        UNR for (int j = 0; j < 32; j++) s[j] = s[j] * eg + kp[j] * dl;
        if (part == 0) O[(size_t)t * VH * 128 + h * 128 + d] = o;
        buf ^= 1;
    }
    UNR for (int j = 0; j < 32; j++) Sh[(i0 + j) * 128 + d] = s[j];
    if (part == 0) PD[h * 128 + d] = 0.0f;
}
__kernel void setone(__global float* x, int n) { int i = get_global_id(0); if (i < n) x[i] = 1.0f; }
__kernel void gnormT(__global const float* O, __global const float* P, int ldp, int ZO, __global const float* gw, __global float* A, float eps, int gplus1) {
    __local float r[128];
    int h = get_group_id(0), t = get_group_id(1), d = get_local_id(0), VH = get_num_groups(0);
    float o = O[(size_t)t * VH * 128 + h * 128 + d];
    float ms = wsum(o * o, r) / 128.0f; float z = P[(size_t)t * ldp + ZO + h * 128 + d];
    A[(size_t)t * VH * 128 + h * 128 + d] = o * rsqrt(ms + eps) * (gplus1 ? 1.0f + gw[d] : gw[d]) * silu(z);
}
__kernel void attnprepT(__global const float* P, int ldp, __global const float* qn, __global const float* kn, __global float* Q, __global KT* kc, __global KT* vc,
                        int pos0, float theta, int NH, int KH, int RD, float eps, int plus1) {
    __local float r[256], sh[256];
    int g = get_group_id(0), t = get_group_id(1), d = get_local_id(0), pos = pos0 + t;
    __global const float* p = P + (size_t)t * ldp;
    float x = g < NH ? p[g * 512 + d] : p[NH * 512 + (g - NH) * 256 + d];
    float k = rsqrt(wsum(x * x, r) / 256.0f + eps);
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
    else { size_t o = ((size_t)pos * KH + (g - NH)) * 256 + d; KST(y, o, kc); KST(p[NH * 512 + KH * 256 + (g - NH) * 256 + d], o, vc); }
}
__kernel void attnT(__global const float* Q, __global const KT* kc, __global const KT* vc, __global const float* P, int ldp, __global float* A,
                    __global float* sc, int pos0, int TT, int KH, int GQ) {
    __local float sq[256], r[256];
    int h = get_group_id(0), t = get_group_id(1), l = get_local_id(0), NH = get_num_groups(0), kh = h / GQ;
    int L = pos0 + t + 1;
    __global float* s_ = sc + ((size_t)t * NH + h) * TT;
    sq[l] = Q[((size_t)t * NH + h) * 256 + l]; barrier(CLK_LOCAL_MEM_FENCE);
    float mx = -INFINITY;
    for (int j = l; j < L; j += 256) {
        __global const KT* kr = kc + ((size_t)j * KH + kh) * 256; float s = 0.0f;
        for (int i = 0; i < 256; i += 4) s += dot(vload4(0, sq + i), KLD4(0, kr + i));
        s *= 0.0625f; s_[j] = s; mx = fmax(mx, s);
    }
    r[l] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) r[l] = fmax(r[l], r[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    mx = r[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sum = 0.0f;
    for (int j = l; j < L; j += 256) { float e = exp(s_[j] - mx); s_[j] = e; sum += e; }
    sum = wsum(sum, r);
    barrier(CLK_GLOBAL_MEM_FENCE);
    float o = 0.0f;
    for (int j = 0; j < L; j++) o += s_[j] * KLD1(((size_t)j * KH + kh) * 256 + l, vc);
    float gt = P[(size_t)t * ldp + h * 512 + 256 + l];
    A[((size_t)t * NH + h) * 256 + l] = o / sum / (1.0f + exp(-gt));
}
__kernel void quantA8(__global const float* X, int ldx, __global uint* A8, __global float* SAt, int K, int Mp) {
    int b = get_global_id(0), t = get_global_id(1), KB = K / 32;
    __global const float* x = X + (size_t)t * ldx + b * 32;
    float mx = 0.0f;
    UNR for (int i = 0; i < 8; i++) { float4 v = fabs(vload4(i, x)); mx = fmax(mx, fmax(fmax(v.x, v.y), fmax(v.z, v.w))); }
    float s = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / s;
    __global uint* o = A8 + ((size_t)(t / 8) * KB + b) * 64 + (t % 8) * 8;
    UNR for (int i = 0; i < 8; i++) o[i] = as_uint(convert_char4_sat_rte(vload4(i, x) * inv));
    SAt[(size_t)b * Mp + t] = s;
}
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void rmsq8(__global const float* X, __global const float* w, int H, float eps, int plus1, __global uint* A8, __global float* SAt, int Mp, int T) {
    // rmsnormT + quantA8 in one pass (4096 <= H <= 8192: thread l handles float4 l and, if present, l + 1024). The sum
    // uses rmsnormT's order, so the normalized values and therefore the INT8 codes are the same bits as the two-kernel path.
    __local float r[64]; __local float tot;
    int l = get_local_id(0), t = get_group_id(1), KB = H / 32, n4 = H / 4, two = l + 1024 < n4;
    __global const float* x = X + (size_t)t * H;
    float4 v0 = t < T ? vload4(l, x) : (float4)(0.0f), v1 = (two && t < T) ? vload4(l + 1024, x) : (float4)(0.0f);
    float s = dot(v0, v0); if (two) s += dot(v1, v1);
    s = sub_group_reduce_add(s);
    if (get_sub_group_local_id() == 0) r[get_sub_group_id()] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_sub_group_id() == 0) { int j = get_sub_group_local_id(); float q = r[j] + r[j + 16] + r[j + 32] + r[j + 48]; q = sub_group_reduce_add(q); if (j == 0) tot = q; }
    barrier(CLK_LOCAL_MEM_FENCE);
    float k = rsqrt(tot / H + eps);
    for (int part = 0; part < 2; part++) {
        if (part == 1 && !two) break;                                  // uniform per sub-group (n4 - 1024 is a multiple of 16)
        int i = l + part * 1024, b = i >> 3;
        float4 g = vload4(i, w), v = part ? v1 : v0;
        float4 y = v * k * (plus1 ? 1.0f + g : g);
        float4 a = fabs(y); float mx = fmax(fmax(a.x, a.y), fmax(a.z, a.w));
        mx = fmax(mx, intel_sub_group_shuffle_xor(mx, 1u)); mx = fmax(mx, intel_sub_group_shuffle_xor(mx, 2u)); mx = fmax(mx, intel_sub_group_shuffle_xor(mx, 4u));
        float sc = mx > 0.0f ? mx / 127.0f : 1.0f;
        A8[((size_t)(t / 8) * KB + b) * 64 + (t % 8) * 8 + (i & 7)] = as_uint(convert_char4_sat_rte(y * (1.0f / sc)));
        if ((i & 7) == 0) SAt[(size_t)b * Mp + t] = sc;
    }
}
__kernel void swq8(__global const float* GU, int F, __global uint* A8, __global float* SAt, int Mp, int T) {
    // swigluT + quantA8 for the down projection in one pass: same float values as silu(g) * u written to M and re-read.
    int b = get_global_id(0), t = get_global_id(1), KB = F / 32;
    __global uint* o = A8 + ((size_t)(t / 8) * KB + b) * 64 + (t % 8) * 8;
    if (t >= T) { UNR for (int i = 0; i < 8; i++) o[i] = 0u; SAt[(size_t)b * Mp + t] = 1.0f; return; }
    __global const float* g = GU + (size_t)t * 2 * F + b * 32; __global const float* u = g + F;
    float4 v[8]; float mx = 0.0f;
    UNR for (int i = 0; i < 8; i++) {
        float4 gg = vload4(i, g), uu = vload4(i, u);
        v[i] = (float4)(silu(gg.x) * uu.x, silu(gg.y) * uu.y, silu(gg.z) * uu.z, silu(gg.w) * uu.w);
        float4 a = fabs(v[i]); mx = fmax(mx, fmax(fmax(a.x, a.y), fmax(a.z, a.w)));
    }
    float s = mx > 0.0f ? mx / 127.0f : 1.0f, inv = 1.0f / s;
    UNR for (int i = 0; i < 8; i++) o[i] = as_uint(convert_char4_sat_rte(v[i] * inv));
    SAt[(size_t)b * Mp + t] = s;
}
__kernel void packQ(__global const float* Q, __global uint* Aq, int NH, int kh) {
    int c2 = get_global_id(0), r = get_global_id(1);
    int t = r / QPG, g = r % QPG; __global const float* q = Q + ((size_t)t * NH + kh * QPG + g) * 256;
    Aq[((size_t)(r / 8) * 16 + c2 / 8) * 64 + (r % 8) * 8 + c2 % 8] = bf16(q[2 * c2]) | (bf16(q[2 * c2 + 1]) << 16);
}
__kernel void packK(__global const KT* kc, __global uint* Bk, int KH, int kh, int L) {
    int n = get_global_id(0), c2 = get_global_id(1);
    uint pr = 0;
    if (n < L) { __global const KT* k = kc + ((size_t)n * KH + kh) * 256; pr = bf16(KLD1(2 * c2, k)) | (bf16(KLD1(2 * c2 + 1, k)) << 16); }
    Bk[((size_t)(n / 8) * 16 + c2 / 8) * 64 + (c2 % 8) * 8 + n % 8] = pr;
}
__kernel void packVT(__global const KT* vc, __global uint* Bv, int KH, int kh, int L, int KB) {
    int n = get_global_id(0), p2 = get_global_id(1);
    int p = 2 * p2; float a = 0.0f, b = 0.0f;
    if (p < L) a = KLD1(((size_t)p * KH + kh) * 256 + n, vc);
    if (p + 1 < L) b = KLD1(((size_t)(p + 1) * KH + kh) * 256 + n, vc);
    Bv[((size_t)(n / 8) * KB + p2 / 8) * 64 + (p2 % 8) * 8 + n % 8] = bf16(a) | (bf16(b) << 16);
}
__kernel void softP(__global const float* S, int Lp, __global uint* Ap, int pos0, int R) {
    __local float red[256];
    int r = get_group_id(0), l = get_local_id(0), KB = Lp / 16;
    int Lr = r < R ? pos0 + r / QPG + 1 : 0;
    __global const float* s = S + (size_t)r * Lp;
    float mx = -INFINITY;
    for (int j = l; j < Lr; j += 256) mx = fmax(mx, s[j] * 0.0625f);
    red[l] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) red[l] = fmax(red[l], red[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sm = 0.0f;
    for (int j = l; j < Lr; j += 256) sm += exp(s[j] * 0.0625f - mx);
    sm = wsum(sm, red);
    float inv = Lr > 0 ? 1.0f / sm : 0.0f;
    for (int c2 = l; c2 < Lp / 2; c2 += 256) {
        int j = 2 * c2;
        float a = j < Lr ? exp(s[j] * 0.0625f - mx) * inv : 0.0f, b = j + 1 < Lr ? exp(s[j + 1] * 0.0625f - mx) * inv : 0.0f;
        Ap[((size_t)(r / 8) * KB + c2 / 8) * 64 + (r % 8) * 8 + c2 % 8] = bf16(a) | (bf16(b) << 16);
    }
}
__kernel void gateO(__global const float* O, __global const float* P, int ldp, __global float* A, int NH, int kh) {
    int d = get_global_id(0), r = get_global_id(1), t = r / QPG, g = r % QPG, h = kh * QPG + g;
    float gt = P[(size_t)t * ldp + h * 512 + 256 + d];
    A[((size_t)t * NH + h) * 256 + d] = O[(size_t)r * 256 + d] / (1.0f + exp(-gt));
}
__kernel void swigluT(__global const float* GU, __global float* M, int F) {
    int i = get_global_id(0), t = get_global_id(1);
    if (i < F) M[(size_t)t * F + i] = silu(GU[(size_t)t * 2 * F + i]) * GU[(size_t)t * 2 * F + F + i];
}
"""

GCFG = dict(RM=4, NT=4, WM=4, WN=4, UK=1, HSW=0)


class Prefill:
    def __init__(self, m: Model, tmax=None):
        import os
        # 2048-token chunks (prefill +7-13% on the A770 and 140T; about +1.1 GiB at a 32K context).
        # A large model on a smaller GPU (27B on a 16 GB card) uses 1024: fits the Arc A770 at an 8K context with MTP
        # (2048 does not: 15.58 of 15.56 GiB) and is +24% over 512 there.
        wbytes = sum(t["nbytes"] for t in m.tm.values())
        auto = "2048" if (m.q.device.global_mem_size >= 20 * 2**30 or wbytes < 8 * 2**30) else "1024"
        tmax = tmax or int(knob("Q35_PCHUNK", auto))
        self.m = m
        ctx = m.ctx
        import os
        self.ib = int(knob("Q35_IB", "1"))
        self.g4 = int(knob("Q35_PG4", "1"))
        gc = dict(GCFG); gc.update(WM=8, UK=2)
        if knob("Q35_PCFG"):
            gc.update(zip(("RM", "NT", "WM", "WN", "UK"), (int(x) for x in os.environ["Q35_PCFG"].split(","))))
        self.gcfg = gc
        # attention without the FP32 score matrix: 2 = one pass (row offset, unnormalized BF16 P, exact two-pass fallback on overflow),
        # 1 = exact two passes (row max / sum, then P), 0 = score matrix + softP
        self.att2 = int(knob("Q35_ATT2", "2"))
        from .attn2_src import ATT2_FAST, ATT2_EXACT, ATT2_PSRC
        gopts = [f"-D{k}={v}" for k, v in gc.items()] + [f"-DG={m.meta['G']}", f"-DXE8={int(m.meta['fmt'] == 'xe8')}", "-DNODEC=0", f"-DIB={self.ib}", f"-DQPG={m.GQ}", "-cl-intel-256-GRF-per-thread"] + _nosplit(m.q.device) + _slmbw()
        self.gp = cl.Program(ctx, GSRC + (ATT2_FAST if self.att2 == 2 else "")).build(options=gopts)
        self.gpx = cl.Program(ctx, GSRC + ATT2_EXACT).build(options=gopts) if self.att2 else None
        self.kg4 = cl.Kernel(self.gp, "pgemm4")
        self.a8 = int(knob("Q35_A8", "1" if m.meta["fmt"] == "xe8" else "0"))
        self.fuse = int(knob("Q35_FUSE", "1"))   # fused norm/SwiGLU + INT8 quantization
        self.fq = self.a8 and self.fuse and 4096 <= m.H <= 8192 and (m.H // 4 - 1024) % 16 == 0   # rmsq8: one or two float4 per thread
        c8 = dict(GCFG); c8.update(zip(("RM", "NT", "WM", "WN", "UK"), (int(x) for x in knob("Q35_P8CFG", "4,4,8,4,1").split(","))))   # K step 32 (A770 +3-4%, 140T +5%)
        self.c8 = c8; self.TR8 = 8 * c8["RM"] * c8["WM"]; self.TC8 = 8 * c8["NT"] * c8["WN"]
        self.gp8 = cl.Program(ctx, GSRC).build(options=[f"-D{k}={v}" for k, v in c8.items()] + [f"-DG={m.meta['G']}", f"-DXE8={int(m.meta['fmt'] == 'xe8')}", "-DNODEC=0", "-DIB=1", "-cl-intel-256-GRF-per-thread"] + _nosplit(m.q.device) + _slmbw())
        self.kg8 = cl.Kernel(self.gp8, "pgemm8")
        self.bt = int(knob("Q35_BT", "-1"))                    # -1: tile B once per GEMM when T >= bt_min (all devices)
        self.bt_min = int(knob("Q35_BT_MIN", str(min(768, tmax)))); self.tkeep = {}   # never above the chunk size
        self.kg8t = cl.Kernel(self.gp8, "pgemm8t"); self.ktb = cl.Kernel(self.gp8, "tileB8")
        self.gopt = [f"-D{k}={v}" for k, v in gc.items()] + [f"-DG={m.meta['G']}", f"-DXE8={int(m.meta['fmt'] == 'xe8')}", "-cl-intel-256-GRF-per-thread"] + _nosplit(m.q.device)
        zb = bytes(self.gp.get_info(cl.program_info.BINARIES)[0])
        self.nofuse = {}
        zb8 = bytes(self.gp8.get_info(cl.program_info.BINARIES)[0])
        for kn, zb in (("pgemm4", zb), ("pgemm", zb), ("pgemm8", zb8), ("pgemm8t", zb8)):
            i = zb.find(("name:            " + kn + chr(10)).encode()); i = i if i >= 0 else zb.find(("name:            " + kn).encode())
            j = zb.find(b"- name:", i + 10); seg = zb[i:j if j > 0 else len(zb)]
            self.nofuse[kn] = b"require_disable_eufusion: true" in seg
        self.pp = cl.Program(ctx, PSRC + (ATT2_PSRC if self.att2 else "")).build(options=([f"-DOFFADJ={knob('Q35_OFFADJ')}.0f"] if knob("Q35_OFFADJ") else []) + [f"-DG={m.meta['G']}", f"-DXE8={int(m.meta['fmt'] == 'xe8')}", f"-DQPG={m.GQ}"] + (["-DKV16"] if m.kv16 else []))
        self.kg = cl.Kernel(self.gp, "pgemm")
        self.TR = 8 * gc["RM"] * gc["WM"]; self.TC = 8 * gc["NT"] * gc["WN"]
        self.Tmax = -(-tmax // self.TR) * self.TR
        mf = cl.mem_flags; f32 = lambda n: cl.Buffer(ctx, mf.READ_WRITE, max(4, int(n) * 4))
        c, Tm = m.cfg, self.Tmax
        H, F = m.H, m.F
        self.NP = max(m.tm[f"layers.{i}.proj.codes"]["shape"][0] for i in range(c["num_hidden_layers"]))
        kmax = max(t["shape"][1] * 2 for k, t in m.tm.items() if t["kind"] == "codes")
        nmax = max(-(-t["shape"][0] // self.TC) * self.TC for k, t in m.tm.items() if t["kind"] == "codes")
        # weight tile scratch: the largest single tensor (INT8 tiles on the W4A8 path, BF16 otherwise), not max N x max K
        tcm = max(self.TC, 8 * c8["NT"] * c8["WN"])
        bpe = 1 if self.a8 else 2
        self.Bp = cl.Buffer(ctx, mf.READ_WRITE, max(-(-t["shape"][0] // tcm) * tcm * t["shape"][1] * 2 * bpe for k, t in m.tm.items() if t["kind"] == "codes"))
        self.Ap = cl.Buffer(ctx, mf.READ_WRITE, Tm * kmax * 2)
        self.SAt = f32(Tm * kmax // 32)
        self.X, self.Xn, self.A = f32(Tm * H), f32(Tm * H), f32(Tm * max(H, m.VH * 128, m.NH * 256))
        self.P, self.GU, self.M = f32(Tm * self.NP), f32(Tm * 2 * F), f32(Tm * F)
        CC = 2 * m.KD + m.VH * 128; self.CC = CC
        self.Cx = f32(Tm * CC); NKH = m.KD // 128
        self.Qn, self.Kn, self.KQ, self.O = f32(Tm * NKH * 128), f32(Tm * NKH * 128), f32(Tm * NKH), f32(Tm * m.VH * 128)
        self.Q = f32(Tm * m.NH * 256)
        import os
        self.pfa = int(knob("Q35_PFA", "1"))
        if self.pfa:
            Mp = -(-m.GQ * Tm // self.TR) * self.TR; Lp = -(-m.TM // 128) * 128
            self.SC = f32(1); self.aq = cl.Buffer(ctx, mf.READ_WRITE, Mp * 256 * 2); self.bk = cl.Buffer(ctx, mf.READ_WRITE, Lp * 256 * 2)
            self.bv = cl.Buffer(ctx, mf.READ_WRITE, Lp * 256 * 2); self.Sm = f32(1 if self.att2 else Mp * Lp); self.MX, self.INV, self.G, self.OFF = f32(Mp), f32(Mp), f32(Mp), f32(Mp)
            self.KN, self.PS = f32(m.TM), f32(Mp * Lp // 32); self.FLAG = f32(1); self.ONE = f32(1)
            cl.enqueue_fill_buffer(m.q, self.ONE, np.int32(1), 0, 4); self.ap = cl.Buffer(ctx, mf.READ_WRITE, Mp * Lp * 2); self.ot = f32(Mp * 256)
        elif m.fa:
            self.SC = f32(1); self.PM, self.PL, self.PO = f32(Tm * m.NH * m.SM), f32(Tm * m.NH * m.SM), f32(Tm * m.NH * m.SM * 256)
        else:
            self.SC = f32(Tm * m.NH * m.TM)
        self.ev = []

    def k(self, name, g, l, *a, label=None):
        kk = cl.Kernel(self.pp, name); kk.set_args(*a)
        e = cl.enqueue_nd_range_kernel(self.m.q, kk, g, l)
        if self.m.prof:
            self.ev.append((label or name, e))
        return e

    def gemm(self, name, Xb, ldx, Y, ldy, add, T, quantized=False):
        """quantized=True: the INT8 activations are already in Ap/SAt (fused producer kernel)."""
        m = self.m; t = m.tm[name + ".codes"]; N, K = t["shape"][0], t["shape"][1] * 2
        Tp = -(-T // self.TR) * self.TR
        I = np.int32
        if self.a8:
            if not quantized:
                self.k("quantA8", (K // 32, Tp), None, Xb, I(ldx), self.Ap, self.SAt, I(K), I(Tp))
            Np8 = -(-N // self.TC8) * self.TC8; ns8 = self.c8["WM"] * self.c8["WN"]
            bt = self.bt if self.bt >= 0 else int(T >= self.bt_min)
            if bt:
                if bt == 2 and name in self.tkeep:
                    tb = self.tkeep[name]
                else:
                    tb = self.Bp
                    if bt == 2:
                        tb = cl.Buffer(m.ctx, cl.mem_flags.READ_WRITE, -(-N // 8) * 8 * K); self.tkeep[name] = tb
                    self.ktb.set_args(m.buf[name + ".codes"], m.buf[name + ".fac"], tb, I(N), I(K))
                    e0 = cl.enqueue_nd_range_kernel(m.q, self.ktb, (K // 32 * 8, -(-N // 8)), (64, 1))
                    if m.prof:
                        self.ev.append(("tileB8", e0))
                kg = self.kg8t; kg.set_args(self.Ap, self.SAt, I(Tp), tb, m.buf[name + ".rs"], Y, I(T), I(N), I(K), I(ldy), I(add))
            else:
                kg = self.kg8; kg.set_args(self.Ap, self.SAt, I(Tp), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], Y, I(T), I(N), I(K), I(ldy), I(add))
            e = cl.enqueue_nd_range_kernel(m.q, kg, (Np8 // self.TC8 * ns8 * 8, Tp // self.TR8), (ns8 * 8, 1))
            if m.prof:
                self.ev.append(("gemm." + name.split(".")[-1], e))
            return
        self.k("packA", (K // 2, Tp), None, Xb, self.Ap, I(K), I(ldx))
        Np = -(-N // self.TC) * self.TC
        if self.g4:
            kg = self.kg4; kg.set_args(self.Ap, m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], Y, I(T), I(N), I(K), I(ldy), I(add))
        else:
            self.k("decB", (-(-N // 64) * 64, K // 32), (64, 1), m.buf[name + ".codes"], m.buf[name + ".fac"], m.buf[name + ".rs"], self.Bp, I(N), I(K))
            kg = self.kg; kg.set_args(self.Ap, self.Bp, Y, I(T), I(N), I(K), I(ldy), I(add))
        nsg = self.gcfg["WM"] * self.gcfg["WN"]
        e = cl.enqueue_nd_range_kernel(m.q, kg, (Np // self.TC * nsg * 8, Tp // self.TR), (nsg * 8, 1))
        if m.prof:
            self.ev.append(("gemm." + name.split(".")[-1], e))

    def run(self, i0, T, pos0=0):
        """Feeds ids[i0 : i0+T] at positions pos0.. . Afterwards m.h holds the last hidden state; KV cache and linear-attention state are filled."""
        m, I, f = self.m, np.int32, np.float32
        assert T <= self.Tmax and pos0 + T <= m.TM
        H, F, eps = m.H, m.F, f(m.eps)
        Tp = -(-T // self.TR) * self.TR
        cl.enqueue_fill_buffer(m.q, self.X, np.float32(0), 0, Tp * H * 4)
        if "embed.ecodes" in m.tm:
            self.k("embedT_xe", (-(-H // 256) * 256, T), (256, 1), m.buf["embed.ecodes"], m.buf["embed.efac"], m.buf["embed.ers"], m.ids, I(i0), self.X, I(H))
        else:
            self.k("embedT", (-(-H // 256) * 256, T), (256, 1), m.buf["embed"], m.ids, I(i0), self.X, I(H))
        for li, lt in enumerate(m.cfg["layer_types"]):
            p = f"layers.{li}."
            NPl = m.tm[p + "proj.codes"]["shape"][0]
            if self.fq:
                self.k("rmsq8", (1024, Tp), (1024, 1), self.X, m.buf[p + "in_norm"], I(H), eps, I(m.p1), self.Ap, self.SAt, I(Tp), I(T))
                self.gemm(p + "proj", None, H, self.P, NPl, 0, T, quantized=True)
            else:
                self.k("rmsnormT", (1024, Tp), (1024, 1), self.X, m.buf[p + "in_norm"], self.Xn, I(H), eps, I(m.p1))
                self.gemm(p + "proj", self.Xn, H, self.P, NPl, 0, T)
            if lt == "linear_attention":
                st, S, PK, PD, PE = m.state[li]; CC = self.CC
                ZO = CC; BO = ZO + m.VH * 128; AO = BO + m.VH
                self.k("convT", (-(-CC // 256) * 256,), (256,), self.P, I(NPl), st, m.buf[p + "conv"], self.Cx, I(CC), I(T))
                self.k("qkprep", (m.KD // 128 * 128, T), (128, 1), self.Cx, I(CC), I(m.KD), self.Qn, self.Kn, self.KQ)
                self.k("dnetT", (m.VH * 128, 4), (128, 1), self.Cx, I(CC), I(m.KD), self.P, I(NPl), I(BO), I(AO), self.Qn, self.Kn, self.KQ,
                       m.buf[p + "A_log"], m.buf[p + "dt_bias"], S, PK, PD, PE, self.O, I(m.VH), I(T))
                self.k("setone", (m.VH,), None, PE, I(m.VH))
                self.k("gnormT", (m.VH * 128, T), (128, 1), self.O, self.P, I(NPl), I(ZO), m.buf[p + "gnorm"], self.A, eps, I(m.gp1))
                ain = m.VH * 128
            else:
                kc, vc = m.state[li][:2]
                self.k("attnprepT", ((m.NH + m.KH) * 256, T), (256, 1), self.P, I(NPl), m.buf[p + "q_norm"], m.buf[p + "k_norm"], self.Q, kc, vc,
                       I(pos0), f(m.theta), I(m.NH), I(m.KH), I(m.RD), eps, I(m.qp1))
                if self.pfa:
                    self.attn_mm(T, pos0, kc, vc, NPl)
                elif m.fa:
                    for kn, g, l, a in (("fattn", (m.KH * 128, -(-(pos0 + T) // m.CH), T), (128, 1, 1), (self.Q, kc, vc, self.PM, self.PL, self.PO, I(pos0), I(m.KH), I(m.SM), I(m.CH))),
                                        ("fcomb", (m.NH * 256, T), (256, 1), (self.PM, self.PL, self.PO, self.P, I(NPl), self.A, I(pos0), I(m.KH), I(m.SM), I(m.CH)))):
                        kk = cl.Kernel(m.prg, kn); kk.set_args(*a); e = cl.enqueue_nd_range_kernel(m.q, kk, g, l)
                        if m.prof:
                            self.ev.append(("attnT", e))
                else:
                    self.k("attnT", (m.NH * 256, T), (256, 1), self.Q, kc, vc, self.P, I(NPl), self.A, self.SC, I(pos0), I(pos0 + T), I(m.KH), I(m.NH // m.KH))
                ain = m.NH * 256
            self.gemm(p + "out", self.A, ain, self.X, H, 1, T)
            if self.fq:
                self.k("rmsq8", (1024, Tp), (1024, 1), self.X, m.buf[p + "post_norm"], I(H), eps, I(m.p1), self.Ap, self.SAt, I(Tp), I(T))
                self.gemm(p + "gu", None, H, self.GU, 2 * F, 0, T, quantized=True)
            else:
                self.k("rmsnormT", (1024, Tp), (1024, 1), self.X, m.buf[p + "post_norm"], self.Xn, I(H), eps, I(m.p1))
                self.gemm(p + "gu", self.Xn, H, self.GU, 2 * F, 0, T)
            if self.a8 and self.fuse:
                self.k("swq8", (F // 32, Tp), None, self.GU, I(F), self.Ap, self.SAt, I(Tp), I(T))
                self.gemm(p + "down", None, F, self.X, H, 1, T, quantized=True)
            else:
                self.k("swigluT", (-(-F // 256) * 256, T), (256, 1), self.GU, self.M, I(F))
                self.gemm(p + "down", self.M, F, self.X, H, 1, T)
            m.q.flush()
        cl.enqueue_copy(m.q, m.h, self.X, src_offset=(T - 1) * H * 4, byte_count=H * 4)

    def attn_mm(self, T, pos0, kc, vc, NPl):
        """Per KV head: S = Q K^T -> causal softmax -> O = P V (two BF16 XMX GEMMs) -> output gate."""
        m, I = self.m, np.int32
        L = pos0 + T; Lp = -(-L // 128) * 128; R = m.GQ * T; Mp = -(-R // self.TR) * self.TR
        nsg = self.gcfg["WM"] * self.gcfg["WN"]
        ra = (I(R), I(pos0), I(0), I(m.GQ), I(0), I(1 << 30), self.ONE, I(0), np.float32(0.0625))   # row masks: causal, no window
        for kh in range(m.KH):
            self.k("packQ", (128, R), None, self.Q, self.aq, I(m.NH), I(kh), label="attnT.pack")
            self.k("packK", (Lp, 128), None, kc, self.bk, I(m.KH), I(kh), I(L), label="attnT.pack")
            self.k("packVT", (256, Lp // 2), None, vc, self.bv, I(m.KH), I(kh), I(L), I(Lp // 16), label="attnT.pack")
            if self.att2 == 2:
                self.k("knorm", (-(-L // 256) * 256,), (256,), kc, I(m.KH), I(kh), I(L), self.KN, label="attnT.pack")
                self.k("rowoff", (Mp,), (256,), self.Q, kc, self.KN, I(m.NH), I(m.KH), I(kh), I(L), I(pos0), I(R), self.OFF, label="attnT.pack")
                ke = cl.Kernel(self.gp, "qke"); ke.set_args(self.aq, self.bk, I(Lp), I(256), *ra, self.OFF, self.ap, self.PS)
                e = cl.enqueue_nd_range_kernel(m.q, ke, (Lp // self.TC * nsg * 8, Mp // self.TR), (nsg * 8, 1))
                if m.prof:
                    self.ev.append(("attnT.qke", e))
                cl.enqueue_fill_buffer(m.q, self.FLAG, np.int32(0), 0, 4)
                self.k("rowsum", (Mp,), (256,), self.PS, I(Lp // 32), I(R), self.G, self.FLAG, label="attnT.pack")
                gate = self.FLAG
            else:
                gate = self.ONE
            if self.att2:
                ks = cl.Kernel(self.gpx, "qkstat"); ks.set_args(self.aq, self.bk, I(Lp), I(256), *ra, self.MX, self.INV, gate, self.G)
                e = cl.enqueue_nd_range_kernel(m.q, ks, (nsg * 8, Mp // self.TR), (nsg * 8, 1))
                if m.prof:
                    self.ev.append(("attnT.stat", e))
                kp = cl.Kernel(self.gpx, "qkp"); kp.set_args(self.aq, self.bk, I(Lp), I(256), *ra, self.MX, self.INV, self.ap, gate)
                e = cl.enqueue_nd_range_kernel(m.q, kp, (Lp // self.TC * nsg * 8, Mp // self.TR), (nsg * 8, 1))
                if m.prof:
                    self.ev.append(("attnT.qkp", e))
            else:
                self.kg.set_args(self.aq, self.bk, self.Sm, I(R), I(Lp), I(256), I(Lp), I(0))
                e = cl.enqueue_nd_range_kernel(m.q, self.kg, (Lp // self.TC * nsg * 8, Mp // self.TR), (nsg * 8, 1))
                if m.prof:
                    self.ev.append(("attnT.qk", e))
                self.k("softP", (Mp * 256,), (256,), self.Sm, I(Lp), self.ap, I(pos0), I(R), label="attnT.soft")
            self.kg.set_args(self.ap, self.bv, self.ot, I(R), I(256), I(Lp), I(256), I(0))
            e = cl.enqueue_nd_range_kernel(m.q, self.kg, (256 // self.TC * nsg * 8, Mp // self.TR), (nsg * 8, 1))
            if m.prof:
                self.ev.append(("attnT.pv", e))
            if self.att2:
                self.k("gateO2", (256, R), None, self.ot, self.G, self.P, I(NPl), self.A, I(m.NH), I(kh), label="attnT.gate")
            else:
                self.k("gateO", (256, R), None, self.ot, self.P, I(NPl), self.A, I(m.NH), I(kh), label="attnT.gate")

    def run_long(self, N, mtp=False, start=0, cuts=(), on_cut=None):
        """Feeds ids[start:N] in chunks (state and KV before `start` must already be in place). With mtp=True the MTP KV
        cache (positions start+1..N) is filled per chunk as well. A chunk ends at every position in `cuts`, and
        on_cut(p) is called right after it (used to snapshot the linear-attention state for the chat cache).
        Each token's result does not depend on where chunks end, so the outputs are the same bits as without cuts."""
        cuts = sorted(p for p in set(cuts) if start < p <= N)
        c0 = start
        while c0 < N:
            c1 = min(c0 + self.Tmax, N)
            for p in cuts:
                if c0 < p < c1:
                    c1 = p; break
            self.run(c0, c1 - c0, c0)
            if mtp:
                self.mtp_fill(c0, c1 - c0)
            if on_cut is not None and c1 in cuts:
                on_cut(c1)
            self.m.q.flush(); c0 = c1

    def mtp_fill(self, c0, T):
        """Row t: (embed(ids[c0+t+1]), normed hidden t) -> MTP KV at position c0+t+1 (the last row is overwritten by the next MTP step)."""
        m, I, f = self.m, np.int32, np.float32
        H, eps = m.H, f(m.eps); Tp = -(-T // self.TR) * self.TR
        if "embed.ecodes" in m.tm:
            self.k("embedT_xe", (-(-H // 256) * 256, T), (256, 1), m.buf["embed.ecodes"], m.buf["embed.efac"], m.buf["embed.ers"], m.ids, I(c0 + 1), self.A, I(H))
        else:
            self.k("embedT", (-(-H // 256) * 256, T), (256, 1), m.buf["embed"], m.ids, I(c0 + 1), self.A, I(H))
        self.k("rmsnormT", (1024, Tp), (1024, 1), self.A, m.buf["mtp.pe"], self.M, I(H), eps, I(1))
        self.k("rmsnormT", (1024, Tp), (1024, 1), self.X, m.buf["norm"], self.Xn, I(H), eps, I(m.p1))
        self.k("rmsnormT", (1024, Tp), (1024, 1), self.Xn, m.buf["mtp.ph"], self.Q, I(H), eps, I(1))
        self.gemm("mtp.fcA", self.M, H, self.GU, H, 0, T)
        self.gemm("mtp.fcB", self.Q, H, self.GU, H, 1, T)
        self.k("rmsnormT", (1024, Tp), (1024, 1), self.GU, m.buf["mtp.in_norm"], self.Xn, I(H), eps, I(1))
        NPm = m.tm["mtp.proj.codes"]["shape"][0]
        self.gemm("mtp.proj", self.Xn, H, self.P, NPm, 0, T)
        self.k("attnprepT", ((m.NH + m.KH) * 256, T), (256, 1), self.P, I(NPm), m.buf["mtp.q_norm"], m.buf["mtp.k_norm"], self.Q, m.mkc, m.mvc,
               I(c0 + 1), f(m.theta), I(m.NH), I(m.KH), I(m.RD), eps, I(1))

    def first_token(self, out_step=0):
        m = self.m
        m.run(m.tail)
        m.am2[0].set_arg(5, np.int32(out_step)); m.run([m.am1, m.am2])
