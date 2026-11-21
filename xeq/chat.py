"""Tokenizer, chat template and output parsing (reasoning blocks and tool calls) for both model families.
The chat template is the original model's `chat_template.jinja`, rendered as is."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime


class Chat:
    """tokenizer.json + chat_template.jinja from the model folder."""

    def __init__(self, folder, family):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer
        for d in (folder, os.path.dirname(os.path.abspath(folder))):      # model folder, then its parent (repo root)
            if all(os.path.exists(os.path.join(d, f)) for f in ("tokenizer.json", "chat_template.jinja")):
                folder = d; break
        else:
            raise SystemExit(f"tokenizer.json and chat_template.jinja not found in {folder} or its parent folder")
        self.family = family
        self.tk = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        class Generation(jinja2.ext.Extension):
            """{% generation %} ... {% endgeneration %} (marks the assistant text for training masks): rendered as its content."""
            tags = {"generation"}

            def parse(self, parser):
                lineno = next(parser.stream).lineno
                body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
                return jinja2.nodes.Scope(body, lineno=lineno)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols, Generation])

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)
        env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False, **k: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda f: datetime.now().strftime(f)
        self.tmpl = env.from_string(open(os.path.join(folder, "chat_template.jinja"), encoding="utf-8").read())
        tid = self.tk.token_to_id
        if family == "gemma4":
            self.bos = "<bos>"
            self.stop = {1, 106, 50} | {tid("<|tool_response>")}          # <eos>, <turn|>, <|tool_response>
            self.image_tok, self.boi, self.eoi = tid("<|image|>"), tid("<|image>"), tid("<image|>")
        else:
            self.bos = ""
            self.stop = {t for t in (tid("<|im_end|>"), tid("<|endoftext|>")) if t is not None}
            self.image_tok = None
        self.stop.discard(None)

    def encode(self, messages, thinking=False, tools=None):
        txt = self.tmpl.render(messages=messages, add_generation_prompt=True, bos_token=self.bos, enable_thinking=thinking, tools=tools or None)
        return self.tk.encode(txt, add_special_tokens=False).ids, txt

    def decode(self, ids):
        return self.tk.decode(ids, skip_special_tokens=False)

    def expand_images(self, ids, counts):
        """Gemma: each <|image|> from the template becomes <|image> + <|image|> * n + <image|>. Returns (ids, spans)."""
        out, spans, k = [], [], 0
        for t in ids:
            if t == self.image_tok:
                out.append(self.boi); a = len(out); out += [self.image_tok] * counts[k]; spans.append((a, len(out))); out.append(self.eoi); k += 1
            else:
                out.append(t)
        if k != len(counts):
            raise ValueError(f"{len(counts)} images given, the prompt has {k} image slots")
        return out, spans

    def split(self, text, thinking=False):
        """Model output -> (reasoning, answer, [(tool name, arguments dict)])."""
        if self.family == "gemma4":
            r, c = _g4_thought(text)
            c, calls = _g4_calls(c)
        else:
            r, c = _q35_thought(text, thinking)
            c, calls = _q35_calls(c)
        return r, c, calls

    def visible(self, text, thinking=False):
        """Partial output while streaming -> (reasoning, answer without a started tool call)."""
        if self.family == "gemma4":
            r, c = _g4_thought(text); k = c.find("<|tool_call>")
        else:
            r, c = _q35_thought(text, thinking); k = c.find("<tool_call>")
        return r, (c[:k] if k >= 0 else c)


# ------------------------------------------------------------------ Gemma 4
Q = '<|"|>'


def _g4_thought(text):
    """Removes every '<|channel>thought ... <channel|>' block wherever it appears (after a tool response the template does not
    open one, so the model may open it itself). An unclosed block runs to the end. A tool call inside reasoning moves to the answer."""
    r, c, i = [], [], 0
    while True:
        a = text.find("<|channel>", i)
        if a < 0:
            c.append(text[i:]); break
        c.append(text[i:a]); b = text.find("<channel|>", a)
        body = text[a + len("<|channel>"):b if b >= 0 else len(text)]
        if body.startswith("thought"):
            body = body[len("thought"):]
        elif b < 0 and not "thought".startswith(body):
            c.append(body); break
        r.append(body.lstrip("\n"))
        if b < 0:
            break
        i = b + len("<channel|>")
    r, c = "\n".join(x.rstrip("\n") if k < len(r) - 1 else x for k, x in enumerate(r) if x), "".join(c).replace("<channel|>", "")
    k = r.find("<|tool_call>")
    if k >= 0 and "<|tool_call>" not in c:
        r, c = r[:k].rstrip("\n"), c + r[k:]
    return r, c


def _g4_val(s, i):
    """One Gemma argument value -> (python value, next index). Strings are <|"|>...<|"|>; keys are bare or quoted."""
    while s[i] in " \n\t":
        i += 1
    if s.startswith(Q, i):
        j = s.index(Q, i + len(Q)); return s[i + len(Q):j], j + len(Q)
    c = s[i]
    if c == "{":
        o = {}; i += 1
        while True:
            while s[i] in " \n\t,":
                i += 1
            if s[i] == "}":
                return o, i + 1
            if s.startswith(Q, i):
                j = s.index(Q, i + len(Q)); k = s[i + len(Q):j]; i = j + len(Q)
            else:
                j = s.index(":", i); k = s[i:j].strip(); i = j
            assert s[i] == ":", s[i:i + 20]
            v, i = _g4_val(s, i + 1); o[k] = v
    if c == "[":
        a = []; i += 1
        while True:
            while s[i] in " \n\t,":
                i += 1
            if s[i] == "]":
                return a, i + 1
            v, i = _g4_val(s, i); a.append(v)
    m = re.compile(r"[^,}\]]*").match(s, i); tok = m.group(0).strip(); i = m.end()
    if tok in ("true", "false"):
        return tok == "true", i
    if tok == "null":
        return None, i
    try:
        return (int(tok) if re.fullmatch(r"-?\d+", tok) else float(tok)), i
    except ValueError:
        return tok, i


def _g4_calls(text):
    """'<|tool_call>call:name{args}<tool_call|>' -> (text before the calls, [(name, args)]). On a parse error the text is kept."""
    k = text.find("<|tool_call>")
    if k < 0:
        return text, []
    head, calls, i = text[:k], [], k
    while True:
        i = text.find("<|tool_call>call:", i)
        if i < 0:
            break
        j = text.index("{", i); name = text[i + len("<|tool_call>call:"):j].strip()
        try:
            args, e = _g4_val(text, j)
        except Exception:
            return text, []
        calls.append((name, args)); i = e
    return head.rstrip() if calls else head, calls


# ------------------------------------------------------------------ Qwen3.5
def _q35_thought(text, thinking):
    """With thinking on, the template ends the prompt with '<think>\n', so the output starts inside the block."""
    if thinking and "<think>" not in text[:16]:
        text = "<think>" + text
    a = text.find("<think>")
    if a < 0:
        return "", text
    b = text.find("</think>", a)
    if b < 0:
        return text[a + len("<think>"):].strip("\n"), text[:a]
    return text[a + len("<think>"):b].strip("\n"), text[:a] + text[b + len("</think>"):].lstrip("\n")


def _q35_arg(v):
    v = v.strip("\n")
    try:
        return json.loads(v)
    except (json.JSONDecodeError, ValueError):
        return v


def _q35_calls(text):
    """'<tool_call>\\n<function=name>\\n<parameter=p>\\nvalue\\n</parameter>...</function>\\n</tool_call>' -> (text, [(name, args)])."""
    k = text.find("<tool_call>")
    if k < 0:
        return text, []
    calls = []
    for blk in re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text[k:], re.S):
        m = re.search(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", blk, re.S)
        if not m:
            return text, []
        args = {p: _q35_arg(v) for p, v in re.findall(r"<parameter=([^>\n]+)>(.*?)</parameter>", m.group(2), re.S)}
        calls.append((m.group(1).strip(), args))
    return text[:k].rstrip(), calls
