"""OpenCL device selection and small shared helpers."""
from __future__ import annotations

import os

import pyopencl as cl


def knob(name, default=None):
    """Internal tuning knob, read from the environment variable XEQ_<name>.
    These are development switches, not a supported interface; the defaults reproduce the published benchmarks."""
    return os.environ.get("XEQ_" + name, default)


def device():
    """The OpenCL GPU to use. XEQ_DEVICE selects one by index (in the order of `python -m xeq devices`)
    or by a substring of its name; the default is the first Intel Arc GPU."""
    gpus = [d for p in cl.get_platforms() for d in p.get_devices() if d.type & cl.device_type.GPU]
    want = os.environ.get("XEQ_DEVICE")
    if want:
        if want.isdigit() and int(want) < len(gpus):
            return gpus[int(want)]
        for d in gpus:
            if want.lower() in d.name.lower():
                return d
        raise SystemExit(f"XEQ_DEVICE={want!r} matches no GPU: {[d.name for d in gpus]}")
    for d in gpus:
        if "Arc" in d.name:
            return d
    raise SystemExit("no Intel Arc GPU found (OpenCL). Found: " + (", ".join(d.name for d in gpus) or "none"))


def nosplit(dev):
    """Build option for GPUs without split DPAS (Xe-LPG+, e.g. Arc 140T): one sub-group reads the whole A tile."""
    return [] if "cl_intel_subgroup_split_matrix_multiply_accumulate" in dev.extensions else ["-DNOSPLIT"]


def vram():
    """Total size in bytes of the OpenCL buffers alive in this process."""
    import gc
    seen, tot = set(), 0
    for o in gc.get_objects():
        if isinstance(o, cl.Buffer):
            try:
                if o.int_ptr not in seen:
                    seen.add(o.int_ptr); tot += o.size
            except Exception:
                pass
    return tot
