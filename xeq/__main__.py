"""XeQ Runtime

python -m xeq <command> ...

  devices                                   list OpenCL GPUs (XEQ_DEVICE selects one by index or name)
  run      <model> -p TEXT [options]        one prompt, streamed to stdout
  chat     <model> [options]                interactive multi-turn chat (empty line or /exit to quit, /reset to clear)
  serve    <model> [--host H] [--port P]    OpenAI-compatible server
  selftest <model>                          greedy outputs against the reference token hashes, plus decode speed
  bench    <model> [--out FILE.json] [--text FILE]   the published benchmark (speed by context length, Gemma without MTP)
  bench    <gemma model> --mtp [--out FILE.json]        Gemma 4 MTP off/on on three chat prompts (needs assistant/)

<model> is a model folder (or a .bin in it) holding the XeFP4 .bin/.json pair, tokenizer.json and chat_template.jinja.
Options: --ctx N (KV cache length), -n N (max new tokens), --think (reasoning on), --system TEXT,
         --image FILE (Gemma 4, repeatable; loads the vision encoder), --no-mtp (plain decoding),
         --kv fp16|fp32 (Gemma 4 KV cache, default fp16; fp32 doubles the KV memory),
         --prefill-chunk auto|small|N (prompt tokens per prefill pass; auto = 2048 where it helps and fits:
         Qwen3.5 always, Gemma 4 on GPUs with 20 GiB or more; small = 512 / 1024, less memory).
Sampling (default greedy): --temperature T --top-p P --top-k K --min-p P --presence-penalty X --repetition-penalty X --seed N"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _args(argv):
    ap = argparse.ArgumentParser(prog="python -m xeq", add_help=False)
    ap.add_argument("cmd"); ap.add_argument("model", nargs="?")
    ap.add_argument("-p", "--prompt"); ap.add_argument("-n", "--max-tokens", type=int, default=1024)
    ap.add_argument("--ctx", type=int); ap.add_argument("--think", action="store_true"); ap.add_argument("--system")
    ap.add_argument("--image", action="append", default=[]); ap.add_argument("--vision", action="store_true")
    ap.add_argument("--no-mtp", action="store_true"); ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--kv", choices=("fp32", "fp16")); ap.add_argument("--prefill-chunk", default="auto")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8080); ap.add_argument("--name")
    ap.add_argument("--out"); ap.add_argument("--text"); ap.add_argument("-h", "--help", action="store_true")
    for k, t in (("temperature", float), ("top-p", float), ("top-k", int), ("min-p", float), ("presence-penalty", float), ("repetition-penalty", float), ("seed", int)):
        ap.add_argument("--" + k, type=t)
    return ap.parse_args(argv)


def _stream(chat, think):
    """on_tokens callback that prints new text as it arrives."""
    st = {"n": 0}

    def f(toks):
        txt = chat.decode(toks)
        if txt.endswith("�"):
            return
        sys.stdout.write(txt[st["n"]:]); sys.stdout.flush(); st["n"] = len(txt)
    return f


def _turn(eng, chat, msgs, a):
    from .serve import prepare
    ids, kw = prepare(eng, chat, msgs, a.think)
    from .sampling import options
    out, fin, (tp, tg, n, re_n) = eng.generate(ids, a.max_tokens, on_tokens=_stream(chat, a.think), sampling=options(vars(a)), **kw)
    r, c, _ = chat.split(chat.decode(out), a.think)
    print(f"\n[{len(ids)} prompt tokens ({re_n} computed) {tp:.2f}s, {n} tokens {max(n - 1, 1) / max(tg, 1e-9):.1f} tok/s, {fin}]", file=sys.stderr, flush=True)
    return c, r


def _chunk(v):
    """--prefill-chunk -> the engines' chunk settings (read when a model is loaded)."""
    if v == "auto":
        return
    if v == "small":
        g, q = "512", "1024"
    elif v.isdigit() and int(v) >= 64:
        g = q = str(int(v))
    else:
        raise SystemExit("--prefill-chunk must be auto, small or a token count >= 64")
    os.environ["XEQ_G4_PCH"], os.environ["XEQ_Q35_PCHUNK"] = g, q


def _user(text, images):
    if not images:
        return {"role": "user", "content": text}
    import base64
    parts = []
    for f in images:
        parts.append({"type": "image_url", "image_url": {"url": "data:image;base64," + base64.b64encode(open(f, "rb").read()).decode()}})
    return {"role": "user", "content": parts + [{"type": "text", "text": text}]}


def main(argv=None):
    a = _args(sys.argv[1:] if argv is None else argv)
    if a.help or a.cmd in ("help", "-h"):
        print(__doc__); return
    if a.cmd == "devices":
        import pyopencl as cl
        gpus = [d for p in cl.get_platforms() for d in p.get_devices() if d.type & cl.device_type.GPU]
        for i, d in enumerate(gpus):
            split = "cl_intel_subgroup_split_matrix_multiply_accumulate" in d.extensions
            print(f"{i}: {d.name} | driver {d.driver_version} | {d.global_mem_size / 2**30:.1f} GiB | split DPAS {'yes' if split else 'no'}")
        return
    if not a.model:
        print(__doc__); sys.exit(2)
    from .engines import open_model
    _chunk(a.prefill_chunk)
    if a.kv:
        import os as _os
        _os.environ["XEQ_G4_KV16"] = "1" if a.kv == "fp16" else "0"
    if a.cmd == "bench":
        if a.mtp:
            from .bench import gemma_mtp
            return gemma_mtp(a.model, a.out or "xeq_bench_mtp.json", a.ctx)
        from .bench import run
        return run(a.model, a.out or "xeq_bench.json", a.ctx, a.text)
    if a.cmd == "selftest":
        from .selftest import run
        return run(a.model, a.ctx)
    t0 = time.perf_counter()
    eng, chat = open_model(a.model, ctx=a.ctx, mtp=not a.no_mtp, vision=bool(a.image) or a.vision, kv=a.kv)
    print(f"[loaded {os.path.basename(eng.files['main'])}"
          + (" + MTP" if getattr(eng, "sp", None) else "") + (" + vision" if getattr(eng, "vis", None) else "")
          + f", context {eng.ctx}, {time.perf_counter() - t0:.1f}s]", file=sys.stderr, flush=True)
    sysm = [{"role": "system", "content": a.system}] if a.system else []
    if a.cmd == "run":
        if a.prompt is None:
            a.prompt = sys.stdin.read()
        _turn(eng, chat, sysm + [_user(a.prompt, a.image)], a)
    elif a.cmd == "chat":
        msgs = list(sysm)
        while True:
            try:
                line = input("\n> ")
            except EOFError:
                break
            if line.strip() in ("", "/exit", "/quit"):
                break
            if line.strip() == "/reset":
                msgs = list(sysm); continue
            msgs.append(_user(line, a.image)); a.image = []
            c, r = _turn(eng, chat, msgs, a)
            msgs.append({"role": "assistant", "content": c})
    elif a.cmd == "serve":
        from .serve import serve
        serve(eng, chat, a.host, a.port, a.name or os.path.basename(os.path.normpath(eng.files["folder"])))
    else:
        print(__doc__); sys.exit(2)


if __name__ == "__main__":
    main()
