"""Gemma 4 vision encoder: preprocessing and OpenCL forward pass.
Preprocess: scale to about 2520 x 256 pixels, round width and height down to multiples of 48 (patch 16 x pooling 3),
bicubic resize, /255, 16x16 patches in row-major order, padded to 2520 patches (position -1).
Patch embedding: Linear(2x - 1) + pos_table[0][x] + pos_table[1][y].
27 layers with sandwich norms; q/k per-head RMSNorm, unweighted v norm, 2D rotary (theta 100), bidirectional over real patches.
Pooling: 3x3 mean * sqrt(1152) -> (x - std_bias) * std_scale -> RMSNorm -> Linear 1152 -> 2816.
"""
import json

import numpy as np

MAXP, PS, POOL = 2520, 16, 3


def preprocess(im):
    """PIL image -> (patches [MAXP, 768] fp32 in 0..1, positions [MAXP, 2] int32 (-1 = padding), grid (gx, gy), soft token count)."""
    from PIL import Image
    im = im.convert("RGB"); W0, H0 = im.size
    s = (MAXP * PS * PS / (W0 * H0)) ** 0.5; U = PS * POOL
    Wn, Hn = max(U, int(W0 * s) // U * U), max(U, int(H0 * s) // U * U)
    while (Wn // PS) * (Hn // PS) > MAXP:
        if Wn >= Hn: Wn -= U
        else: Hn -= U
    a = np.asarray(im.resize((Wn, Hn), Image.BICUBIC), np.float32) / 255.0
    gx, gy = Wn // PS, Hn // PS
    P = a.reshape(gy, PS, gx, PS, 3).transpose(0, 2, 1, 3, 4).reshape(gy * gx, PS * PS * 3)
    pix = np.zeros((MAXP, PS * PS * 3), np.float32); pix[:gy * gx] = P
    pos = np.full((MAXP, 2), -1, np.int32)
    pos[:gy * gx] = np.stack(np.meshgrid(np.arange(gx), np.arange(gy)), -1).reshape(-1, 2)
    return pix, pos, (gx, gy), (gx // POOL) * (gy // POOL)


# ---------------------------------------------------------------- GPU
VSRC = r"""
inline uint bfh(float x) { uint u = as_uint(x); return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16; }
__kernel void vpackA(__global const float* X, int ldx, int M, int Kr, __global uint* A, int KB) {
    int c2 = get_global_id(0), r = get_global_id(1), k = 2 * c2; float a = 0.0f, b = 0.0f;
    if (r < M) { if (k < Kr) a = X[(size_t)r * ldx + k]; if (k + 1 < Kr) b = X[(size_t)r * ldx + k + 1]; }
    A[((size_t)(r / 8) * KB + c2 / 8) * 64 + (r % 8) * 8 + c2 % 8] = bfh(a) | (bfh(b) << 16);
}
__kernel void vpackBr(__global const float* X, int ldx, int N, int Kr, __global uint* B, int KB) {
    int n = get_global_id(0), c2 = get_global_id(1), k = 2 * c2; float a = 0.0f, b = 0.0f;
    if (n < N) { if (k < Kr) a = X[(size_t)n * ldx + k]; if (k + 1 < Kr) b = X[(size_t)n * ldx + k + 1]; }
    B[((size_t)(n / 8) * KB + c2 / 8) * 64 + (c2 % 8) * 8 + n % 8] = bfh(a) | (bfh(b) << 16);
}
__kernel void vpackBT(__global const float* X, int ldx, int N, int Kr, __global uint* B, int KB) {
    int n = get_global_id(0), c2 = get_global_id(1), k = 2 * c2; float a = 0.0f, b = 0.0f;
    if (n < N) { if (k < Kr) a = X[(size_t)k * ldx + n]; if (k + 1 < Kr) b = X[(size_t)(k + 1) * ldx + n]; }
    B[((size_t)(n / 8) * KB + c2 / 8) * 64 + (c2 % 8) * 8 + n % 8] = bfh(a) | (bfh(b) << 16);
}
__kernel void vdeqB(__global const char* q, __global const float* s, int N, int Kr, __global uint* B, int KB) {
    int n = get_global_id(0), c2 = get_global_id(1), k = 2 * c2; float a = 0.0f, b = 0.0f;
    if (n < N) { float sc = s[n]; if (k < Kr) a = q[(size_t)n * Kr + k] * sc; if (k + 1 < Kr) b = q[(size_t)n * Kr + k + 1] * sc; }
    B[((size_t)(n / 8) * KB + c2 / 8) * 64 + (c2 % 8) * 8 + n % 8] = bfh(a) | (bfh(b) << 16);
}
inline float bsum(float v, __local float* r) {
    int l = get_local_id(0); r[l] = v; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = get_local_size(0) / 2; q > 0; q >>= 1) { if (l < q) r[l] += r[l + q]; barrier(CLK_LOCAL_MEM_FENCE); }
    float o = r[0]; barrier(CLK_LOCAL_MEM_FENCE); return o;
}
__kernel void vrms(__global const float* x, __global const float* w, __global float* y, int H, int usew, float eps) {
    __local float r[256]; int t = get_group_id(0), l = get_local_id(0); x += (size_t)t * H; y += (size_t)t * H;
    float s = 0.0f; for (int i = l; i < H; i += 256) s += x[i] * x[i];
    float inv = rsqrt(bsum(s, r) / H + eps);
    for (int i = l; i < H; i += 256) y[i] = x[i] * inv * (usew ? w[i] : 1.0f);
}
// h += rms(y)·w
__kernel void vnadd(__global const float* yv, __global const float* w, __global float* h, int H, float eps) {
    __local float r[256]; int t = get_group_id(0), l = get_local_id(0); yv += (size_t)t * H; h += (size_t)t * H;
    float s = 0.0f; for (int i = l; i < H; i += 256) s += yv[i] * yv[i];
    float inv = rsqrt(bsum(s, r) / H + eps);
    for (int i = l; i < H; i += 256) h[i] += yv[i] * inv * w[i];
}
__kernel void vpos(__global float* x, __global const ushort* tab, __global const int* pos, int H, int NPOS) {
    int i = get_global_id(0), t = get_global_id(1); if (i >= H) return;
    int px = pos[2 * t], py = pos[2 * t + 1]; if (px < 0) return;
    float a = as_float((uint)tab[(size_t)px * H + i] << 16), b = as_float((uint)tab[((size_t)NPOS + py) * H + i] << 16);
    x[(size_t)t * H + i] += a + b;
}
__kernel void vprep(__global const float* qkv, __global const float* qw, __global const float* kw, __global const int* pos,
                    __global float* Qf, __global float* Kf, __global float* Vf, int NH, int HD, int Mp, float theta, float eps) {
    __local float xq[128], xk[128], r[128];
    int h = get_group_id(0), t = get_group_id(1), d = get_local_id(0), W = NH * HD;
    __global const float* row = qkv + (size_t)t * 3 * W + h * HD;
    float q = d < HD ? row[d] : 0.0f, k = d < HD ? row[W + d] : 0.0f, v = d < HD ? row[2 * W + d] : 0.0f;
    float iq = rsqrt(bsum(q * q, r) / HD + eps), ik = rsqrt(bsum(k * k, r) / HD + eps), iv = rsqrt(bsum(v * v, r) / HD + eps);
    if (d < HD) { xq[d] = q * iq * qw[d]; xk[d] = k * ik * kw[d]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (d >= HD) return;
    int hf = HD / 2, qd = HD / 4, seg = d / hf, i = d % hf, j = i % qd;
    int p = max(pos[2 * t + seg], 0);
    float ang = (float)p * pow(theta, -(float)(2 * j) / (float)hf), c = cos(ang), s = sin(ang);
    int o = seg * hf + (i < qd ? i + qd : i - qd); float sg = i < qd ? -1.0f : 1.0f;
    size_t oi = ((size_t)h * Mp + t) * HD + d;
    Qf[oi] = xq[d] * c + sg * xq[o] * s; Kf[oi] = xk[d] * c + sg * xk[o] * s; Vf[oi] = v * iv;
}
__kernel void vsoft(__global const float* S, int Lp, int nk, __global uint* Ap) {
    __local float red[256];
    int r = get_group_id(0), l = get_local_id(0), KB = Lp / 16; __global const float* s = S + (size_t)r * Lp;
    float mx = -INFINITY; for (int j = l; j < nk; j += 256) mx = fmax(mx, s[j]);
    red[l] = mx; barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = 128; q > 0; q >>= 1) { if (l < q) red[l] = fmax(red[l], red[l + q]); barrier(CLK_LOCAL_MEM_FENCE); }
    mx = red[0]; barrier(CLK_LOCAL_MEM_FENCE);
    float sm = 0.0f; for (int j = l; j < nk; j += 256) sm += exp(s[j] - mx);
    float inv = 1.0f / bsum(sm, red);
    for (int c2 = l; c2 < Lp / 2; c2 += 256) {
        int j = 2 * c2; float a = j < nk ? exp(s[j] - mx) * inv : 0.0f, b = j + 1 < nk ? exp(s[j + 1] - mx) * inv : 0.0f;
        Ap[((size_t)(r / 8) * KB + c2 / 8) * 64 + (r % 8) * 8 + c2 % 8] = bfh(a) | (bfh(b) << 16);
    }
}
__kernel void vgelu(__global const float* gu, __global float* o, int F, int Fp) {
    int j = get_global_id(0), t = get_global_id(1); if (j >= Fp) return;
    float g = gu[(size_t)t * 2 * Fp + j], u = gu[(size_t)t * 2 * Fp + Fp + j];
    float a = 0.5f * g * (1.0f + tanh(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    o[(size_t)t * Fp + j] = j < F ? a * u : 0.0f;
}
__kernel void vpool(__global const float* x, int gx, int ncx, __global const float* sb, __global const float* ss, __global float* y, int H, float eps) {
    __local float r[256]; int c = get_group_id(0), l = get_local_id(0), cy = c / ncx, cx = c % ncx;
    float v[8]; float s = 0.0f; int n = 0;
    for (int i = l; i < H; i += 256, n++) {
        float a = 0.0f;
        for (int dy = 0; dy < 3; dy++) for (int dx = 0; dx < 3; dx++) a += x[((size_t)(3 * cy + dy) * gx + 3 * cx + dx) * H + i];
        a = (a / 9.0f * sqrt((float)H) - sb[i]) * ss[i]; v[n] = a; s += a * a;
    }
    float inv = rsqrt(bsum(s, r) / H + eps); n = 0;
    for (int i = l; i < H; i += 256, n++) y[(size_t)c * H + i] = v[n] * inv;
}
"""


def _ceil(a, b):
    return -(-a // b) * b


def pack_b_host(u16, Np, Kp):
    """bf16 bits [N, K] -> pgemm B layout as uint32 (zero padded)."""
    N, K = u16.shape; w = np.zeros((Np, Kp), np.uint16); w[:N, :K] = u16
    pr = w[:, 0::2].astype(np.uint32) | (w[:, 1::2].astype(np.uint32) << 16)        # [Np, Kp/2]
    KB = Kp // 16
    return np.ascontiguousarray(pr.reshape(Np // 8, 8, KB, 8).transpose(0, 2, 3, 1)).reshape(-1)


class Vision:
    def __init__(self, path, ctx, q):
        import pyopencl as cl
        from ..qwen35 import prefill as QP
        self.cl, self.ctx, self.q = cl, ctx, q
        meta = json.load(open(path[:-4] + ".json")); self.meta, tm = meta, meta["tensors"]
        vc = meta["vision_config"]; self.H, self.NH, self.HD, self.F = vc["hidden_size"], vc["num_attention_heads"], vc["head_dim"], vc["intermediate_size"]
        self.NL, self.eps, self.theta = vc["num_hidden_layers"], float(vc["rms_norm_eps"]), float(vc["rope_parameters"]["rope_theta"])
        self.NPOS = vc["position_embedding_size"]; self.TH = meta["text_hidden"]
        gc = dict(QP.GCFG); gc.update(UK=1)
        self.TR, self.TC, self.ns = 8 * gc["RM"] * gc["WM"], 8 * gc["NT"] * gc["WN"], gc["WM"] * gc["WN"]
        self.pg = cl.Program(ctx, QP.GSRC).build(options=[f"-D{k}={v}" for k, v in gc.items()] + ["-DG=32", "-DXE8=1", "-DNODEC=0", "-DIB=1", "-cl-intel-256-GRF-per-thread"] + QP._nosplit(q.device))
        zb = bytes(self.pg.get_info(cl.program_info.BINARIES)[0]); i = zb.find(("name:            pgemm" + chr(10)).encode()); j = zb.find(b"- name:", i + 10)
        if b"require_disable_eufusion: true" in zb[i:j if j > 0 else len(zb)]:
            print("warning: vision pgemm was built with EU fusion disabled", flush=True)
        self.pv = cl.Program(ctx, VSRC).build()
        self.kg = cl.Kernel(self.pg, "pgemm"); self.kk = {}
        mf = cl.mem_flags; self.Fp = _ceil(self.F, self.TC)
        fh = open(path, "rb")

        def raw(name):
            t = tm[name]; fh.seek(t["off"]); b = fh.read(t["nbytes"])
            return np.frombuffer(b, {"bf16": np.uint16, "f32": np.float32, "i8": np.int8}[t["kind"]]).reshape(t["shape"])
        ro = lambda a: cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(a))
        enc = meta["enc"]; self.w = {}; self.nbytes = 0

        def lin(key, parts, Np, Kp):
            """Concatenates weights [n_i, K] by rows into the B layout; INT8 weights are kept with row scales and expanded to bf16 on use."""
            if enc == "bf16":
                rows = []
                for nm, padn in parts:
                    u = raw(nm); z = np.zeros((padn, u.shape[1]), np.uint16); z[:u.shape[0]] = u; rows.append(z)
                b = pack_b_host(np.concatenate(rows), Np, Kp); self.w[key] = ("bf16", ro(b), Np, Kp); self.nbytes += b.nbytes
            else:
                qs, ss = [], []
                for nm, padn in parts:
                    qq, s = raw(nm + ".q"), raw(nm + ".s"); z = np.zeros((padn, qq.shape[1]), np.int8); z[:qq.shape[0]] = qq
                    zs = np.zeros(padn, np.float32); zs[:qq.shape[0]] = s; qs.append(z); ss.append(zs)
                qq, s = np.concatenate(qs), np.concatenate(ss); kr = qq.shape[1]
                self.w[key] = ("i8", ro(qq), ro(s), qq.shape[0], kr, Np, Kp); self.nbytes += qq.nbytes + s.nbytes
        H, F, Fp = self.H, self.F, self.Fp
        lin("patch", [("v.patch_embedder.input_proj.weight", H)], H, _ceil(768, 16))
        self.tab = ro(raw("v.patch_embedder.position_embedding_table")); self.nbytes += 2 * self.NPOS * H * 2
        self.sb, self.ss = ro(raw("v.std_bias").astype(np.float32)), ro(raw("v.std_scale").astype(np.float32))
        for i in range(self.NL):
            p = f"v.encoder.layers.{i}."
            lin(p + "qkv", [(p + "self_attn.q_proj.weight", H), (p + "self_attn.k_proj.weight", H), (p + "self_attn.v_proj.weight", H)], 3 * H, H)
            lin(p + "o", [(p + "self_attn.o_proj.weight", H)], H, H)
            lin(p + "gu", [(p + "mlp.gate_proj.weight", Fp), (p + "mlp.up_proj.weight", Fp)], 2 * Fp, H)
            if enc == "bf16":
                u = raw(p + "mlp.down_proj.weight"); b = pack_b_host(u, H, Fp); self.w[p + "down"] = ("bf16", ro(b), H, Fp); self.nbytes += b.nbytes
            else:
                qq, s = raw(p + "mlp.down_proj.weight.q"), raw(p + "mlp.down_proj.weight.s")
                self.w[p + "down"] = ("i8", ro(qq), ro(s), H, F, H, Fp); self.nbytes += qq.nbytes + s.nbytes
            for k in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "self_attn.q_norm", "self_attn.k_norm"):
                self.w[p + k] = ro(raw(p + k + ".weight").astype(np.float32))
        lin("proj", [("e.embedding_projection.weight", self.TH)], self.TH, H)
        fh.close()
        self.WB = None
        if enc == "i8":
            self.WB = cl.Buffer(ctx, mf.READ_WRITE, max(v[5] * v[6] // 2 for v in self.w.values() if isinstance(v, tuple) and v[0] == "i8") * 4)

    def K(self, name, g, l, *args):
        if name not in self.kk:
            self.kk[name] = self.cl.Kernel(self.pv, name)
        k = self.kk[name]; k.set_args(*args); self.cl.enqueue_nd_range_kernel(self.q, k, g, l)

    def gemm(self, A, key, Y, M, ldy, add=0, B=None, N=None, Kp=None):
        I = np.int32
        if B is None:
            w = self.w[key]
            if w[0] == "bf16":
                B, Np, Kp = w[1], w[2], w[3]; N = N or Np
            else:
                _, qq, s, n, kr, Np, Kp = w
                self.K("vdeqB", (Np, Kp // 2), None, qq, s, I(n), I(kr), self.WB, I(Kp // 16)); B = self.WB; N = N or Np
        Np = _ceil(N, self.TC)
        self.kg.set_args(A, B, Y, I(M), I(N), I(Kp), I(ldy), I(add))
        self.cl.enqueue_nd_range_kernel(self.q, self.kg, (Np // self.TC * self.ns * 8, _ceil(M, self.TR) // self.TR), (self.ns * 8, 1))

    def encode(self, im=None, pre=None, dump=None):
        """Image -> projected rows (language-model hidden size) as cl.Buffer [n, TH], and n."""
        cl, I, f = self.cl, np.int32, np.float32
        pix, pos, (gx, gy), ns = pre if pre is not None else preprocess(im)
        T = gx * gy; Mp = _ceil(MAXP, self.TR); H, NH, HD, Fp = self.H, self.NH, self.HD, self.Fp; HDp = _ceil(HD, 16)
        mf = cl.mem_flags; B = lambda n: cl.Buffer(self.ctx, mf.READ_WRITE, int(n) * 4)
        X, XN, AT = B(Mp * H), B(Mp * H), B(Mp * H)
        QKV, GU, MM = B(Mp * 3 * H), B(Mp * 2 * Fp), B(Mp * Fp)
        Qf, Kf, Vf = B(NH * Mp * HD), B(NH * Mp * HD), B(NH * Mp * HD)
        AP = B(Mp * max(Fp, Mp) // 2); S = B(Mp * Mp); PB = B(Mp * Mp // 2); BK = B(Mp * HDp // 2); BV = B(_ceil(HD, self.TC) * Mp // 2)
        X2 = 2 * pix[:T] - 1
        xin = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(X2, f))
        posb = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.ascontiguousarray(pos[:Mp] if Mp <= MAXP else np.concatenate([pos, np.full((Mp - MAXP, 2), -1, np.int32)]), np.int32))
        Kp = self.w["patch"][3] if self.w["patch"][0] == "bf16" else self.w["patch"][6]
        self.K("vpackA", (Kp // 2, Mp), None, xin, I(768), I(T), I(768), AP, I(Kp // 16))
        self.gemm(AP, "patch", X, T, H)
        self.K("vpos", (_ceil(H, 64), T), (64, 1), X, self.tab, posb, I(H), I(self.NPOS))
        eps = f(self.eps)
        for i in range(self.NL):
            p = f"v.encoder.layers.{i}."; w = self.w
            self.K("vrms", (T * 256,), (256,), X, w[p + "input_layernorm"], XN, I(H), I(1), eps)
            self.K("vpackA", (H // 2, Mp), None, XN, I(H), I(T), I(H), AP, I(H // 16))
            self.gemm(AP, p + "qkv", QKV, T, 3 * H)
            self.K("vprep", (NH * 128, T), (128, 1), QKV, w[p + "self_attn.q_norm"], w[p + "self_attn.k_norm"], posb, Qf, Kf, Vf, I(NH), I(HD), I(Mp), f(self.theta), eps)
            for h in range(NH):
                qh = Qf.get_sub_region(h * Mp * HD * 4, Mp * HD * 4); kh = Kf.get_sub_region(h * Mp * HD * 4, Mp * HD * 4); vh = Vf.get_sub_region(h * Mp * HD * 4, Mp * HD * 4)
                self.K("vpackA", (HDp // 2, Mp), None, qh, I(HD), I(T), I(HD), AP, I(HDp // 16))
                self.K("vpackBr", (Mp, HDp // 2), None, kh, I(HD), I(T), I(HD), BK, I(HDp // 16))
                self.gemm(AP, None, S, T, Mp, B=BK, N=Mp, Kp=HDp)
                self.K("vsoft", (T * 256,), (256,), S, I(Mp), I(T), PB)
                self.K("vpackBT", (_ceil(HD, self.TC), Mp // 2), None, vh, I(HD), I(HD), I(T), BV, I(Mp // 16))
                self.gemm(PB, None, XN, T, HD, B=BV, N=HD, Kp=Mp)
                self.K("vcopycols", (HD, T), None, XN, I(HD), AT, I(H), I(h * HD))
            self.K("vpackA", (H // 2, Mp), None, AT, I(H), I(T), I(H), AP, I(H // 16))
            self.gemm(AP, p + "o", XN, T, H)
            self.K("vnadd", (T * 256,), (256,), XN, w[p + "post_attention_layernorm"], X, I(H), eps)
            self.K("vrms", (T * 256,), (256,), X, w[p + "pre_feedforward_layernorm"], XN, I(H), I(1), eps)
            self.K("vpackA", (H // 2, Mp), None, XN, I(H), I(T), I(H), AP, I(H // 16))
            self.gemm(AP, p + "gu", GU, T, 2 * Fp)
            self.K("vgelu", (Fp, T), None, GU, MM, I(self.F), I(Fp))
            self.K("vpackA", (Fp // 2, Mp), None, MM, I(Fp), I(T), I(Fp), AP, I(Fp // 16))
            self.gemm(AP, p + "down", XN, T, H, N=H)
            self.K("vnadd", (T * 256,), (256,), XN, w[p + "post_feedforward_layernorm"], X, I(H), eps)
            if dump is not None and (i in (0, 1) or i == self.NL - 1):
                a = np.empty((T, H), f); cl.enqueue_copy(self.q, a, X); dump[f"layer{i}"] = a
        ncx = gx // POOL
        PL = B(ns * H); SOFT = cl.Buffer(self.ctx, mf.READ_WRITE, ns * self.TH * 4)
        self.K("vpool", (ns * 256,), (256,), X, I(gx), I(ncx), self.sb, self.ss, PL, I(H), eps)
        self.K("vpackA", (H // 2, _ceil(ns, self.TR)), None, PL, I(H), I(ns), I(H), AP, I(H // 16))
        self.gemm(AP, "proj", SOFT, ns, self.TH)
        if dump is not None:
            a = np.empty((ns, H), f); cl.enqueue_copy(self.q, a, PL); dump["pooled_normed"] = a
        return SOFT, ns


VSRC += r"""
__kernel void vcopycols(__global const float* src, int lds, __global float* dst, int ldd, int c0) {
    int d = get_global_id(0), t = get_global_id(1); dst[(size_t)t * ldd + c0 + d] = src[(size_t)t * lds + d];
}
"""
