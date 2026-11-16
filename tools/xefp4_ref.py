"""XeFP4 v1.0 reference encoder and decoder (NumPy only).

This is the executable form of spec/XEFP4_SPEC.md, sections 2-3. It is written for clarity, not speed.

    codes, fac, rs = encode(W)          # W: float32 [N, K], K % 32 == 0
    W_hat = decode(codes, fac, rs, K)   # float32 [N, K]
"""
from __future__ import annotations

import numpy as np

G = 32
L = 12.0                                                  # largest grid magnitude
GRID = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.float32)   # magnitude index -> value (E2M1 x 2)
BOUNDS = np.array([0.5, 1.5, 2.5, 3.5, 5, 7, 10], np.float32)
RS_MIN = 1e-12                                            # applied before FP16 rounding (rounds to 0 in FP16)


def row_scale(W: np.ndarray) -> np.ndarray:
    """FP16 row scale rs = fp16(max(max|W| / 12, 1e-12)), returned as float16 [N]."""
    amax = np.abs(W).max(axis=1).astype(np.float32)
    return np.maximum(amax / np.float32(L), np.float32(RS_MIN)).astype(np.float16)


def magnitude_index(a: np.ndarray) -> np.ndarray:
    """Number of boundaries strictly smaller than |z| (a value on a boundary goes to the smaller magnitude)."""
    return np.searchsorted(BOUNDS, a, side="left").astype(np.int64)


def encode(W: np.ndarray):
    """W float32 [N, K] -> (codes uint8 [N, K/2], fac uint32 [N, FW], rs float16 [N])."""
    W = np.ascontiguousarray(W, np.float32)
    N, K = W.shape
    assert K % G == 0, "K must be a multiple of 32"
    nb = K // G
    rs16 = row_scale(W)
    rs = rs16.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        Wb = (W / rs[:, None]).reshape(N, nb, G)           # float32; inf/nan only where rs == 0
    zero_row = rs == 0
    Wb[zero_row] = 0

    best_e = np.full((N, nb), np.inf, np.float64)
    best_m = np.ones((N, nb), np.int64)
    for m in range(1, 9):                                  # m = 1..8 in order, strict "<": ties keep the smaller m
        f = np.float32(m / 8)
        z = Wb / f
        q = GRID[magnitude_index(np.abs(z))] * np.sign(z)
        d = q * f - Wb                                     # float32 element terms
        e = (d.astype(np.float64) ** 2).sum(-1)            # summed in float64 (see spec 2.3 note)
        better = e < best_e
        best_e[better] = e[better]
        best_m[better] = m

    z = Wb / (best_m.astype(np.float32) / 8)[..., None]
    idx = magnitude_index(np.abs(z))
    neg = (z < 0) & (idx > 0)                               # zero is always stored with sign 0
    code = (idx | (neg.astype(np.int64) << 3)).reshape(N, K)
    best_m[zero_row] = 1                                    # rs == 0: codes and factors carry no meaning
    code[zero_row] = 0
    codes = (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)
    return codes, pack_factors(best_m - 1), rs16


def pack_factors(j: np.ndarray) -> np.ndarray:
    """j = m - 1, int [N, nb] (0..7) -> uint32 [N, ceil(3·nb/32)], LSB first per row."""
    N, nb = j.shape
    bits = ((j[:, :, None] >> np.arange(3)) & 1).astype(np.uint8).reshape(N, nb * 3)
    FW = -(-nb * 3 // 32)
    pad = np.zeros((N, FW * 32 - nb * 3), np.uint8)
    by = np.packbits(np.concatenate([bits, pad], 1), axis=1, bitorder="little")
    return np.ascontiguousarray(by).view("<u4")


def unpack_factors(fac: np.ndarray, nb: int) -> np.ndarray:
    """uint32 [N, FW] -> m = 1..8, int [N, nb]."""
    by = np.ascontiguousarray(fac.astype("<u4")).view(np.uint8)
    bits = np.unpackbits(by, axis=1, bitorder="little")[:, :nb * 3].reshape(-1, nb, 3).astype(np.int64)
    return (bits * (1 << np.arange(3))).sum(-1) + 1


def decode(codes: np.ndarray, fac: np.ndarray, rs: np.ndarray, K: int) -> np.ndarray:
    """(codes, fac, rs) -> float32 [N, K]. W = rs · (m/8) · g, computed in float32."""
    N = codes.shape[0]
    nb = K // G
    code = np.empty((N, K), np.int64)
    code[:, 0::2] = codes & 15
    code[:, 1::2] = codes >> 4
    g = GRID[code & 7] * np.where(code & 8, np.float32(-1), np.float32(1))
    m = unpack_factors(fac, nb).astype(np.float32)
    sf = rs.astype(np.float32)[:, None] * (m / 8)           # [N, nb]
    return (g.reshape(N, nb, G) * sf[..., None]).reshape(N, K).astype(np.float32)
