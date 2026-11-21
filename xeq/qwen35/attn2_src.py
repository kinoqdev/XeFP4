"""Prefill attention without an FP32 score matrix. ATT2_FAST (qke) is appended to GSRC (no early returns, so EU fusion
stays on for pgemm); ATT2_EXACT (gated two-pass fallback) is built as its own program with GSRC. Row masks are arguments (RARGS): Qwen and Gemma.
qkstat: one work-group per 8*TR rows of Q (packed as in packQ), loops over K's column tiles inside the causal range,
        S = Q K^T in registers, online row max / sum of exp(S/16 - max). Writes MX[row] and INV[row] = 1/sum.
qkp:    S tile again (same products, same bits), P = exp(S/16 - MX) * INV for causal columns, else 0, stored as BF16 pairs
        in the PV GEMM's A layout (as softP did)."""

ATT2_HEAD = r"""
inline uint bf16g(float f) { uint u = as_uint(f); return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
#define QKTILE(c0) { \
    __global const uint* gb = B + (size_t)(c0) * KB * 64; \
    UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = (float8)(0.f); \
    LOADG(0); STORE(0); barrier(CLK_LOCAL_MEM_FENCE); \
    for (int kb = 0; kb < KB; kb += UK) { \
        int cur = (kb / UK) & 1, more = kb + UK < KB; \
        if (more) { LOADG(kb + UK); } \
        UN for (int u = 0; u < UK; u++) { \
            int8 b[NT]; \
            UN for (int c = 0; c < NT; c++) b[c] = as_int8(intel_sub_group_block_read8(&sB[cur][((sc * NT + c) * UK + u) * 64])); \
            ATILE a[RM]; \
            UN for (int r = 0; r < RM; r++) a[r] = AREAD(&sA[cur][((sr * RM + r) * UK + u) * 64], h); \
            UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c++) acc[r][c] = MADBF(a[r], b[c], acc[r][c]); \
        } \
        if (more) { STORE(cur ^ 1); } \
        barrier(CLK_LOCAL_MEM_FENCE); \
    } }
/* columns [a0, a1) of block row `row` (key position jl + column): causal, sliding window WIN, bidirectional image span (ub) */
#define RR(row, a0, a1) { int rg_ = rb0 + (row); a0 = 0; a1 = 0; \
    if (rg_ < R) { int t_ = rg_ / GQ, L_ = pos0 + t_ + 1; a0 = max(0, L_ - WIN) - jl; if (bid) L_ = max(L_, ub[pos0 + t_]); a1 = L_ - jl; } }
/* column range of this work-group's rows */
#define WGRANGE(cmin, cmax) { int fr_ = r0 * 8, lr_ = min(r0 * 8 + TR * 8, R - rb0) - 1, x_; cmin = 0; cmax = 0; \
    if (lr_ >= fr_) { if (bid) cmax = N; else { RR(fr_, cmin, x_); RR(lr_, x_, cmax); } } cmax = min(cmax, N); }
#define RARGS int R, int pos0, int rb0, int GQ, int jl, int WIN, __global const int* ub, int bid, float scl
#define GSRC2(j, kb) ((j) < NA ? ga + ((size_t)((j) / UK) * KB + (kb) + (j) % UK) * 64 \
                               : gb + ((size_t)(((j) - NA) / UK) * KB + (kb) + ((j) - NA) % UK) * 64)

"""

ATT2_EXACT = ATT2_HEAD + r"""__attribute__((intel_reqd_sub_group_size(8)))
__kernel void qkstat(__global const uint* A, __global const uint* B, int N, int K, RARGS, __global float* MX, __global float* INV,
                     __global const int* gate, __global float* GR) {
    if (gate[0] == 0) return;
    __local uint sA[2][NA * 64], sB[2][TC * UK * 64];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 16;
    int r0 = get_group_id(1) * TR;
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    float8 acc[RM][NT];
    uint8 st[PER];
#undef GSRC
#define GSRC GSRC2
    int cmin, cmax; WGRANGE(cmin, cmax)
    int t0 = cmin / (TC * 8), ntile = cmax > cmin ? (cmax + TC * 8 - 1) / (TC * 8) : 0;
    int rt = r0 + sr * RM;
    float rm[RM][8], rsum[RM][8];
    UN for (int r = 0; r < RM; r++) UN for (int m = 0; m < 8; m++) { rm[r][m] = -INFINITY; rsum[r][m] = 0.0f; }
    for (int tj = t0; tj < ntile; tj++) {
        int c0 = tj * TC;
        QKTILE(c0)
        int ct = c0 + sc * NT;
        UN for (int r = 0; r < RM; r++) UN for (int m = 0; m < 8; m++) {
            int row = (rt + r) * 8 + m, a0, a1; RR(row, a0, a1)
            float v[NT]; float tm = -INFINITY;
            UN for (int c = 0; c < NT; c++) { int col = (ct + c) * 8 + l; v[c] = col >= a0 && col < a1 ? acc[r][c][m] * scl : -INFINITY; tm = fmax(tm, v[c]); }
            if (tm > -INFINITY) {
                float nm = fmax(rm[r][m], tm), s = rsum[r][m] * exp(rm[r][m] - nm);
                UN for (int c = 0; c < NT; c++) s += exp(v[c] - nm);
                rsum[r][m] = s; rm[r][m] = nm;
            }
        }
    }
    /* combine: 8 lanes, then the WN sub-groups of a row (through SLM) */
    __local float* xs = (__local float*)&sA[0][0];
    UN for (int r = 0; r < RM; r++) UN for (int m = 0; m < 8; m++) {
        float mm = sub_group_reduce_max(rm[r][m]);
        float ss = sub_group_reduce_add(mm > -INFINITY && rm[r][m] > -INFINITY ? rsum[r][m] * exp(rm[r][m] - mm) : 0.0f);
        int rl = ((sr * RM + r) * 8 + m);
        if (l == 0) { xs[(rl * WN + sc) * 2] = mm; xs[(rl * WN + sc) * 2 + 1] = ss; }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int rl = get_local_id(0); rl < TR * 8; rl += NSG * 8) {
        int row = r0 * 8 + rl;
        float mm = -INFINITY;
        for (int q = 0; q < WN; q++) mm = fmax(mm, xs[(rl * WN + q) * 2]);
        float ss = 0.0f;
        for (int q = 0; q < WN; q++) { float a = xs[(rl * WN + q) * 2]; if (a > -INFINITY) ss += xs[(rl * WN + q) * 2 + 1] * exp(a - mm); }
        MX[row] = mm > -INFINITY ? mm : 0.0f; INV[row] = ss > 0.0f ? 1.0f / ss : 0.0f; GR[row] = 1.0f;
    }
}

__attribute__((intel_reqd_sub_group_size(8)))
__kernel void qkp(__global const uint* A, __global const uint* B, int N, int K, RARGS, __global const float* MX, __global const float* INV, __global uint* Ap,
                  __global const int* gate) {
    if (gate[0] == 0) return;
    __local uint sA[2][NA * 64], sB[2][TC * UK * 64];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 16, KBp = N / 16;
    int r0 = get_group_id(1) * TR, c0 = get_group_id(0) * TC;
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    float8 acc[RM][NT];
    uint8 st[PER];
    int cmin, cmax; WGRANGE(cmin, cmax)
    int rt = r0 + sr * RM, ct = c0 + sc * NT;
    /* output: the 8 rows x 16 columns of acc blocks (c, c+1) are one 64-uint block of the PV A layout; lane j writes column pair j */
    int src0 = (2 * l) & 7, src1 = (2 * l + 1) & 7, hi = l >= 4;
    if (c0 * 8 >= cmax || c0 * 8 + TC * 8 <= cmin) {                       /* whole tile outside the attended range */
        UN for (int r = 0; r < RM; r++) UN for (int c = 0; c < NT; c += 2) {
            int rb = rt + r, cb2 = (ct + c) / 2;
            intel_sub_group_block_write8(Ap + ((size_t)rb * KBp + cb2) * 64, (uint8)(0u));
        }
        return;
    }
    QKTILE(c0)
    UN for (int r = 0; r < RM; r++) {
        int rb = rt + r;
        float mx[8], inv[8]; int A0[8], A1[8];
        UN for (int m = 0; m < 8; m++) { int row = rb * 8 + m, ok = rb0 + row < R; RR(row, A0[m], A1[m]) mx[m] = ok ? MX[row] : 0.0f; inv[m] = ok ? INV[row] : 0.0f; }
        UN for (int c = 0; c < NT; c += 2) {
            uint8 o;
            UN for (int m = 0; m < 8; m++) {
                int col = (ct + c) * 8 + l;
                float v0 = col >= A0[m] && col < A1[m] ? exp(acc[r][c][m] * scl - mx[m]) * inv[m] : 0.0f;
                float v1 = col + 8 >= A0[m] && col + 8 < A1[m] ? exp(acc[r][c + 1][m] * scl - mx[m]) * inv[m] : 0.0f;
                float a0 = intel_sub_group_shuffle(v0, src0), a1 = intel_sub_group_shuffle(v0, src1);
                float b0 = intel_sub_group_shuffle(v1, src0), b1 = intel_sub_group_shuffle(v1, src1);
                o[m] = bf16g(hi ? b0 : a0) | (bf16g(hi ? b1 : a1) << 16);
            }
            intel_sub_group_block_write8(Ap + ((size_t)rb * KBp + (ct + c) / 2) * 64, o);
        }
    }
}

#undef GSRC
"""

ATT2_FAST = ATT2_HEAD + r"""/* One pass: P~ = exp(S/16 - OFF[row]) (unnormalized, BF16) and per-(row, 32-column) partial sums PS (row sum = sum of PS row). */
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void qke(__global const uint* A, __global const uint* B, int N, int K, RARGS, __global const float* OFF, __global uint* Ap, __global float* PS) {
    __local uint sA[2][NA * 64], sB[2][TC * UK * 64];
    int sg = get_sub_group_id(), l = get_sub_group_local_id();
    int sr = sg / WN, sc = sg % WN;
    int h = (sg & 1) ^ HSW;
    int KB = K / 16, KBp = N / 16, NPS = N / (8 * TC) * WN;
    int r0 = get_group_id(1) * TR, c0 = get_group_id(0) * TC, tj = get_group_id(0);
    __global const uint* ga = A + (size_t)r0 * KB * 64;
    float8 acc[RM][NT];
    uint8 st[PER];
    int rt = r0 + sr * RM, ct = c0 + sc * NT;
    int src0 = (2 * l) & 7, src1 = (2 * l + 1) & 7, hi = l >= 4;
    /* every tile is computed: masked columns give 0 (no early return, see the module note) */
    QKTILE(c0)
    UN for (int r = 0; r < RM; r++) {
        int rb = rt + r;
        float of[8], pl[8]; int A0[8], A1[8]; float ps = 0.0f;
        UN for (int m = 0; m < 8; m++) { int row = rb * 8 + m; RR(row, A0[m], A1[m]) of[m] = rb0 + row < R ? OFF[row] : 0.0f; pl[m] = 0.0f; }
        UN for (int c = 0; c < NT; c += 2) {
            uint8 o;
            UN for (int m = 0; m < 8; m++) {
                int col = (ct + c) * 8 + l;
                float v0 = col >= A0[m] && col < A1[m] ? exp(acc[r][c][m] * scl - of[m]) : 0.0f;
                float v1 = col + 8 >= A0[m] && col + 8 < A1[m] ? exp(acc[r][c + 1][m] * scl - of[m]) : 0.0f;
                pl[m] += v0 + v1;
                float a0 = intel_sub_group_shuffle(v0, src0), a1 = intel_sub_group_shuffle(v0, src1);
                float b0 = intel_sub_group_shuffle(v1, src0), b1 = intel_sub_group_shuffle(v1, src1);
                o[m] = bf16g(hi ? b0 : a0) | (bf16g(hi ? b1 : a1) << 16);
            }
            intel_sub_group_block_write8(Ap + ((size_t)rb * KBp + (ct + c) / 2) * 64, o);
        }
        UN for (int m = 0; m < 8; m++) { float sm = sub_group_reduce_add(pl[m]); if (l == m) ps = sm; }
        PS[(size_t)(rb * 8 + l) * NPS + tj * WN + sc] = ps;                 /* lane l: row rb*8+l */
    }
}
#undef GSRC
"""



ATT2_PSRC = r"""
#ifndef OFFADJ
#define OFFADJ 0.0f
#endif
__kernel void knorm(__global const KT* kc, int KH, int kh, int L, __global float* KN) {
    int n = get_global_id(0); if (n >= L) return;
    __global const KT* k = kc + ((size_t)n * KH + kh) * 256; float s = 0.0f;
    for (int d = 0; d < 256; d++) { float v = KLD1(d, k); s += v * v; }
    KN[n] = sqrt(s);
}
/* row offset: min(|q| max|k| / 16, q.k_self / 16 + 20): no exp overflow when the bound is used, the own key keeps the sum > 0 */
__kernel void rowoff(__global const float* Q, __global const KT* kc, __global const float* KN, int NH, int KH, int kh, int L, int pos0, int R, __global float* OFF) {
    __local float red[256];
    int l = get_local_id(0), r = get_global_id(0);
    float km = 0.0f;
    for (int j = l; j < L; j += 256) km = fmax(km, KN[j]);
    red[l] = km; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) red[l] = fmax(red[l], red[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    km = red[0];
    if (r >= R) { OFF[r] = 0.0f; return; }
    int t = r / QPG, g = r % QPG;
    __global const float* q = Q + ((size_t)t * NH + kh * QPG + g) * 256;
    __global const KT* k = kc + ((size_t)(pos0 + t) * KH + kh) * 256;
    float qq = 0.0f, qk = 0.0f;
    for (int d = 0; d < 256; d++) { float a = q[d]; qq += a * a; qk += a * KLD1(d, k); }
    OFF[r] = fmin(sqrt(qq) * km * 0.0625f, qk * 0.0625f + 20.0f) - OFFADJ;   /* OFFADJ: test knob (forces the fallback) */
}
__kernel void rowsum(__global const float* PS, int NPS, int R, __global float* GR, __global int* FLAG) {
    int r = get_global_id(0);
    if (r >= R) { GR[r] = 0.0f; return; }
    __global const float* p = PS + (size_t)r * NPS; float s = 0.0f;
    for (int i = 0; i < NPS; i++) s += p[i];
    GR[r] = s > 0.0f ? 1.0f / s : 0.0f;
    if (!(s > 0.0f && s < 1e30f)) atomic_max(FLAG, 1);
}
__kernel void gateO2(__global const float* O, __global const float* GR, __global const float* P, int ldp, __global float* A, int NH, int kh) {
    int d = get_global_id(0), r = get_global_id(1), t = r / QPG, g = r % QPG, h = kh * QPG + g;
    float gt = P[(size_t)t * ldp + h * 512 + 256 + d];
    A[((size_t)t * NH + h) * 256 + d] = O[(size_t)r * 256 + d] * GR[r] / (1.0f + exp(-gt));
}
"""


ATT2_GSRC = r"""
#ifndef OFFADJ
#define OFFADJ 0.0f
#endif
__kernel void knormg(__global const KT* kc, int KH, int kh, int HD, int jl, int jh, int CAP, int Lq, __global float* KN) {
    int n = get_global_id(0); if (n >= Lq) return;
    int p = jl + n; float s = 0.0f;
    if (p < jh) { __global const KT* k = kc + ((size_t)(p % CAP) * KH + kh) * HD; for (int d = 0; d < HD; d++) { float v = KLD1(d, k); s += v * v; } }
    KN[n] = sqrt(s);
}
/* row offset (scale 1): min(|q| max|k|, q.k_self + 20) */
__kernel void rowoffg(__global const float* Q, __global const KT* kc, __global const float* KN, int NH, int KH, int kh, int GQ, int HD, int CAP, int Lq,
                      int pos0, int rb0, int R, __global float* OFF) {
    __local float red[256];
    int l = get_local_id(0), r = get_global_id(0), rg = rb0 + r;
    float km = 0.0f;
    for (int j = l; j < Lq; j += 256) km = fmax(km, KN[j]);
    red[l] = km; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) red[l] = fmax(red[l], red[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    km = red[0];
    if (rg >= R) { OFF[r] = 0.0f; return; }
    int t = rg / GQ, g = rg % GQ;
    __global const float* q = Q + ((size_t)t * NH + kh * GQ + g) * HD;
    __global const KT* k = kc + ((size_t)((pos0 + t) % CAP) * KH + kh) * HD;
    float qq = 0.0f, qk = 0.0f;
    for (int d = 0; d < HD; d++) { float a = q[d]; qq += a * a; qk += a * KLD1(d, k); }
    OFF[r] = fmin(sqrt(qq) * km, qk + 20.0f) - OFFADJ;
}
__kernel void rowsumg(__global const float* PS, int NPS, int rb0, int R, __global float* GR, __global int* FLAG) {
    int r = get_global_id(0);
    if (rb0 + r >= R) { GR[r] = 0.0f; return; }
    __global const float* p = PS + (size_t)r * NPS; float s = 0.0f;
    for (int i = 0; i < NPS; i++) s += p[i];
    GR[r] = s > 0.0f ? 1.0f / s : 0.0f;
    if (!(s > 0.0f && s < 1e30f)) atomic_max(FLAG, 1);
}
__kernel void writeOg2(__global const float* O, __global const float* GR, __global float* A, int NH, int kh, int GQ, int HD, int rb0, int R) {
    int d = get_global_id(0), r = get_global_id(1), rg = rb0 + r; if (rg >= R) return;
    int t = rg / GQ, g = rg % GQ; A[((size_t)t * NH + kh * GQ + g) * HD + d] = O[(size_t)r * HD + d] * GR[r];
}
"""
