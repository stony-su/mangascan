"""Japanese -> English translation of the text blocks of a page.

Three engines, same interface:

``claude-code``  Claude, through the Claude Code command line that is already
                 installed and logged in on this computer.  No API key.
``claude``       Claude, through the Anthropic API with an API key.
``offline``      FuguMT, a compact translation model that runs locally.

The two Claude engines *localise*: the whole page is translated in one go,
with the page image, the story so far and the names already settled, by a
model that is asked for the line an English-language writer would have given
that character in that moment, not for the Japanese sentence in English
words.  They also read lettering the local OCR could not.  Both send the page
to Anthropic and use the user's account.

The offline engine translates each block on its own and literally.  It is a
rough draft and cannot be made to localise; it exists so the tool works with
no account and no network.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .ocr import has_japanese

MIN_CONFIDENCE = 0.70            # below this the OCR result is treated as unread
CLAUDE_MODEL = "claude-opus-5-5"         # API engine
CLAUDE_CODE_MODEL = "opus"               # alias understood by the Claude Code CLI
CLAUDE_EFFORT = "medium"
ENGINES = ("claude-code", "claude", "offline")

_PUNCT_ONLY = re.compile(r"^[\s…・.．。｡!！?？~〜ー―—\-、,「」『』()（）]+$")
_PUNCT_MAP = str.maketrans({"！": "!", "？": "?", "〜": "~", "．": ".", "。": ".", "｡": ".", "、": ",",
                            "ー": "-", "―": "-", "—": "-", "（": "(", "）": ")", "「": "", "」": "",
                            "『": "", "』": "", "・": "."})


def punctuation_only(text: str) -> Optional[str]:
    """Blocks such as "…………" or "！？" need no translator, just ASCII lettering."""
    if not text or not _PUNCT_ONLY.match(text):
        return None
    out = text.translate(_PUNCT_MAP).replace("…", "...")
    out = re.sub(r"\.{7,}", "......", re.sub(r"\s+", "", out))
    return out


def polish(text: str) -> str:
    """Tidy a translation for lettering."""
    text = text.replace("…", "...").replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"').replace("　", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    text = re.sub(r"\.{4,}", "...", text)
    return text


def readable(item: dict) -> bool:
    return item.get("confidence", 1.0) >= MIN_CONFIDENCE and has_japanese(item.get("text", ""))


class TranslationUnavailable(RuntimeError):
    """The engine could not be used; the message says why in plain words."""


# ------------------------------------------------------------------ context
@dataclass
class PageContext:
    """What a localiser knows before reading a page.

    A line cannot be put into natural English without knowing who says it,
    to whom, and what has just happened; names have to be spelled the same
    on every page.  This carries that from page to page.
    """

    notes: str = ""                                          # the user's notes on the series
    glossary: Dict[str, str] = field(default_factory=dict)   # Japanese -> English, already settled
    previous: List[dict] = field(default_factory=list)       # earlier pages: name, scene, lines

    def add_names(self, names: List[dict]) -> None:
        for n in names or []:
            ja, en = str(n.get("japanese", "")).strip(), str(n.get("english", "")).strip()
            if ja and en and ja not in self.glossary:
                self.glossary[ja] = en

    def remember(self, name: str, scene: str, lines: List[dict], names: List[dict]) -> None:
        self.add_names(names)
        self.previous.append({"name": name, "scene": scene, "lines": lines})

    def brief(self, pages: int = 1) -> str:
        """The context as it is put in front of the translator."""
        parts = []
        if self.notes.strip():
            parts.append("Notes from the editor about this series:\n" + self.notes.strip())
        if self.glossary:
            parts.append("Names and terms already settled (keep these spellings):\n"
                         + json.dumps(self.glossary, ensure_ascii=False))
        story = [p["scene"] for p in self.previous if p.get("scene")]
        if story:
            parts.append("The story so far, page by page:\n" + "\n".join(f"- {s}" for s in story[-12:]))
        script = [{"page": p["name"], "lines": p["lines"]} for p in self.previous[-pages:] if p.get("lines")]
        if script:
            parts.append("The script of the page just before this one, for continuity of voice:\n"
                         + json.dumps(script, ensure_ascii=False))
        return "\n\n".join(parts)


# ------------------------------------------------------------------ offline
class OfflineTranslator:
    """FuguMT (Marian) on ONNX Runtime, with a small beam search."""

    name = "offline"

    def __init__(self, files: Dict[str, Path], threads: int = 0, beams: int = 4):
        import onnxruntime as ort
        import sentencepiece as spm

        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        prov = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(files["encoder_model.onnx"]), sess_options=opts, providers=prov)
        self.decoder = ort.InferenceSession(str(files["decoder_model.onnx"]), sess_options=opts, providers=prov)
        self.sp = spm.SentencePieceProcessor(model_file=str(files["source.spm"]))
        vocab = json.loads(Path(files["vocab.json"]).read_text(encoding="utf-8"))
        self.vocab = vocab
        self.pieces = {i: p for p, i in vocab.items()}
        self.eos = vocab["</s>"]
        self.unk = vocab["<unk>"]
        self.pad = vocab["<pad>"]
        self.beams = beams
        self.last_page: Optional[dict] = None

    def translate(self, text: str) -> str:
        ids = [self.vocab.get(p, self.unk) for p in self.sp.encode(text, out_type=str)][:400] + [self.eos]
        src = np.array([ids], np.int64)
        att = np.ones_like(src)
        memory = self.encoder.run(None, {"input_ids": src, "attention_mask": att})[0]
        best = self._beam_search(memory, att, max_len=int(len(ids) * 2.5) + 12)
        pieces = [self.pieces.get(i, "") for i in best if i not in (self.eos, self.pad)]
        return polish("".join(pieces).replace("▁", " "))

    def _beam_search(self, memory: np.ndarray, att: np.ndarray, max_len: int) -> List[int]:
        seqs = [[self.pad]]                      # Marian starts decoding from <pad>
        scores = np.zeros(1)
        done: List[tuple] = []
        for step in range(max_len):
            n = len(seqs)
            logits = self.decoder.run(["logits"], {
                "input_ids": np.array(seqs, np.int64),
                "encoder_hidden_states": np.repeat(memory, n, axis=0),
                "encoder_attention_mask": np.repeat(att, n, axis=0)})[0][:, -1].astype(np.float64)
            logits[:, self.pad] = -1e9
            logits -= logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            total = scores[:, None] + logp
            flat = total.ravel()
            top = np.argpartition(-flat, min(2 * self.beams, flat.size - 1))[:2 * self.beams]
            top = top[np.argsort(-flat[top])]
            new_seqs, new_scores = [], []
            for idx in top:
                b, tok = divmod(int(idx), logp.shape[1])
                seq = seqs[b] + [tok]
                if tok == self.eos:
                    done.append((flat[idx] / len(seq), seq))     # length-normalised
                elif len(new_seqs) < self.beams:
                    new_seqs.append(seq)
                    new_scores.append(flat[idx])
            if not new_seqs:
                break
            seqs, scores = new_seqs, np.array(new_scores)
            if len(done) >= self.beams and max(d[0] for d in done) >= scores.max() / (step + 2):
                break
        if not done:
            done = [(scores[i] / len(seqs[i]), seqs[i]) for i in range(len(seqs))]
        return max(done, key=lambda d: d[0])[1][1:]

    def translate_page(self, items: List[dict], page_rgb: Optional[np.ndarray] = None,
                       progress: Optional[Callable[[int, int], None]] = None,
                       context: Optional[PageContext] = None) -> List[dict]:
        out = []
        for i, it in enumerate(items):
            text = it.get("text", "")
            simple = punctuation_only(text)
            if it.get("keep"):                    # hand-edited: nothing to do
                english = it.get("english", "")
            elif simple is not None:
                english = simple
            elif readable(it):
                english = self.translate(text)
            else:
                english = ""                      # unread: left blank for a human
            out.append({"id": it["id"], "text": text, "translation": english})
            if progress:
                progress(i + 1, len(items))
        self.last_page = {"scene": "", "names": []}
        return out


# ------------------------------------------------------------------- claude
SYSTEM_PROMPT = """You are localising a page of Japanese manga for an English-language edition read by a \
Western audience. The reader should feel the page was written in English.

What you are given: the page image; a list of its text blocks in reading order (top to bottom, right to \
left) with each block's position as percentages of the page and what a local OCR model read there; close-up \
crops of blocks the OCR could not read; and, when there is any, notes from the editor, names already \
settled, the story so far and the script of the previous page.

Work in this order.

1. Understand the scene before writing a word. From the art and all the lines together, work out who is \
speaking in each balloon, to whom, what their relationship is, what has just happened and what each \
speaker wants. Put that in `scene` in one or two sentences; the next page's translator will read it.

2. Then write each line as the line an English-language writer would have given that character at that \
moment. This is localisation, not translation word by word. Japanese leaves out subjects and objects, \
stacks clauses in front of the verb and leans on set phrases; English that mirrors those habits is stiff \
and robotic, and that is the failure to avoid.
- Recast freely. Supply who and what from context, reorder clauses, turn noun-heavy phrasing into verbs, \
and break a long thought where an English speaker would pause.
- A set phrase or idiom becomes the English expression that does the same job in that situation, never \
a gloss of its words.
- Give every speaker a voice that fits who they are (rank, age, temper, how they feel right now) and \
the world of the story. A war epic set centuries ago should not sound like modern office talk or slang; \
a school comedy should not sound like a legal notice. Speech uses contractions and natural rhythm.
- When one sentence runs across several balloons, make the pieces read as one English sentence broken \
at natural points, even if that means moving words from one balloon to the next.
- Stay faithful to what is meant. Do not add information or jokes, do not drop meaning, and keep the \
emotion at the strength it has in the original.

3. Keep it short. Each line is lettered into a balloon drawn for a few vertical columns of Japanese, so \
of two good phrasings take the tighter one, and never pad.

Names and culture: keep people and place names, romanised without macrons and spelled exactly as in \
the settled list; small ruby text beside a name on the page gives its reading. Do not add honorifics \
the original lacks, and drop ones the relationship does not depend on. Keep what the story's setting \
depends on; where an incidental reference would simply be lost on a Western reader, say it the way an \
English speaker would express the same idea.

The OCR is usually right for ordinary balloons and often wrong for small, stylised or outlined \
lettering and for rare kanji in names. Check every reading against the image, correct it, and read the \
blocks it could not. Blocks that are not story text (chapter title, credits, publisher notices) are \
translated plainly. If a block is not text at all, or truly cannot be read, give an empty `english` \
for it rather than guess. A block that is only an ellipsis or punctuation stays that way. A block \
marked `fixed` was already lettered by the editor: return its English unchanged and let it inform the \
lines around it.

Write in ordinary sentence case with English punctuation; the letterer applies the comic capitals. \
Return every id exactly once, and in `names` list every person, place and special term that appears on \
this page with the English spelling you used."""

SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "japanese": {"type": "string"},
                    "english": {"type": "string"},
                },
                "required": ["id", "japanese", "english"],
                "additionalProperties": False,
            },
        },
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"japanese": {"type": "string"}, "english": {"type": "string"}},
                "required": ["japanese", "english"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scene", "blocks", "names"],
    "additionalProperties": False,
}

MAX_CROPS = 6


def image_block(rgb: np.ndarray, max_side: int = 1568) -> dict:
    """An RGB array as a base64 JPEG image content block."""
    from PIL import Image

    im = Image.fromarray(rgb)
    if max(im.size) > max_side:              # larger adds cost, not legibility
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=88)
    return {"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg",
        "data": base64.standard_b64encode(buf.getvalue()).decode("ascii")}}


def _crop(rgb: np.ndarray, box, pad: int = 12) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    crop = rgb[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    short = min(crop.shape[:2])
    if 0 < short < 120:                          # small lettering: enlarge so it can be read
        from PIL import Image
        f = min(4.0, 120.0 / short)
        crop = np.asarray(Image.fromarray(crop).resize(
            (int(crop.shape[1] * f), int(crop.shape[0] * f)), Image.LANCZOS))
    return crop


def page_content(items: List[dict], page_rgb: Optional[np.ndarray], context: Optional[PageContext]) -> List[dict]:
    """Everything the translator gets for one page: image, crops, context, blocks."""
    content: List[dict] = []
    shape = page_rgb.shape if page_rgb is not None else (1, 1)
    h, w = shape[:2]
    if page_rgb is not None:
        content.append(image_block(page_rgb))
        unread = [it for it in items if not it.get("keep") and not readable(it)
                  and punctuation_only(it.get("text", "")) is None][:MAX_CROPS]
        for it in unread:
            content.append({"type": "text", "text": f"Close-up of block {it['id']}:"})
            content.append(image_block(_crop(page_rgb, it["box"]), max_side=1000))
    listing = []
    for it in items:
        x1, y1, x2, y2 = it.get("box", (0, 0, 0, 0))
        entry = {
            "id": it["id"],
            "ocr": it.get("text", ""),
            "ocr_confidence": round(float(it.get("confidence", 0.0)), 2),
            "left_pct": round(100 * x1 / w), "top_pct": round(100 * y1 / h),
            "width_pct": round(100 * (x2 - x1) / w), "height_pct": round(100 * (y2 - y1) / h),
            "direction": "vertical" if it.get("vertical", True) else "horizontal",
        }
        if it.get("keep"):
            entry["fixed"] = it.get("english", "")
        listing.append(entry)
    parts = []
    brief = context.brief() if context is not None else ""
    if brief:
        parts.append(brief)
    parts.append("Text blocks on this page, in reading order:\n" + json.dumps(listing, ensure_ascii=False, indent=1))
    content.append({"type": "text", "text": "\n\n".join(parts)})
    return content


def read_answer(data: dict, items: List[dict]) -> tuple:
    """(results, scene, names) from the model's JSON, tolerant of gaps."""
    got = {}
    for b in data.get("blocks", []) if isinstance(data, dict) else []:
        try:
            got[int(b["id"])] = b
        except (KeyError, TypeError, ValueError):
            continue
    out = []
    for it in items:
        b = got.get(it["id"])
        if it.get("keep"):
            out.append({"id": it["id"], "text": it.get("text", ""), "translation": it.get("english", "")})
            continue
        if b is None:                         # never trust a model to return every id
            simple = punctuation_only(it.get("text", ""))
            out.append({"id": it["id"], "text": it.get("text", ""), "translation": simple or ""})
            continue
        japanese = str(b.get("japanese") or it.get("text", ""))
        simple = punctuation_only(japanese)
        english = simple if simple is not None else polish(str(b.get("english", "")))
        out.append({"id": it["id"], "text": japanese, "translation": english})
    scene = str(data.get("scene", "")).strip() if isinstance(data, dict) else ""
    names = [n for n in (data.get("names", []) if isinstance(data, dict) else []) if isinstance(n, dict)]
    return out, scene, names


class _PageLocaliser:
    """Shared by the two Claude engines: one request per page."""

    name = ""

    def __init__(self):
        self.last_page: Optional[dict] = None     # scene + names of the page just translated
        self.last_usage: Optional[dict] = None

    def _ask(self, system: str, content: List[dict]) -> dict:
        raise NotImplementedError

    def translate_page(self, items: List[dict], page_rgb: Optional[np.ndarray] = None,
                       progress: Optional[Callable[[int, int], None]] = None,
                       context: Optional[PageContext] = None) -> List[dict]:
        self.last_page = None
        if not items:
            return []
        data = self._ask(SYSTEM_PROMPT, page_content(items, page_rgb, context))
        out, scene, names = read_answer(data, items)
        self.last_page = {"scene": scene, "names": names}
        if progress:
            progress(len(items), len(items))
        return out


class ClaudeTranslator(_PageLocaliser):
    """Claude through the Anthropic API (official ``anthropic`` SDK, API key)."""

    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL, effort: str = CLAUDE_EFFORT, client=None):
        super().__init__()
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise TranslationUnavailable(
                    "The Claude API engine needs the 'anthropic' package (pip install anthropic).") from exc
            client = anthropic.Anthropic()      # credentials come from the environment
        self.client = client
        self.model = model
        self.effort = effort

    def build_request(self, system: str, content: List[dict]) -> dict:
        return {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            # if a safety classifier declines the request, the API re-runs it on
            # Anthropic's recommended fallback model instead of returning nothing
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }

    def _ask(self, system: str, content: List[dict]) -> dict:
        import anthropic

        try:
            response = self.client.beta.messages.create(**self.build_request(system, content))
        except TypeError as exc:
            # the SDK raises this before sending anything when it finds no credentials
            if "authentication" not in str(exc).lower():
                raise
            raise TranslationUnavailable(
                "No Anthropic credentials were found. Set ANTHROPIC_API_KEY (or run 'ant auth login') "
                "and start the app again.") from exc
        except anthropic.AuthenticationError as exc:
            raise TranslationUnavailable(
                "Anthropic rejected the credentials. Set ANTHROPIC_API_KEY (or run 'ant auth login') "
                "and try again.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise TranslationUnavailable(f"This account may not use {self.model}: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise TranslationUnavailable(f"The model {self.model} is not available to this account.") from exc
        except anthropic.RateLimitError as exc:
            raise TranslationUnavailable("Anthropic is rate-limiting this account; wait a minute and retry.") from exc
        except anthropic.BadRequestError as exc:
            raise TranslationUnavailable(f"Anthropic did not accept the request: {exc.message}") from exc
        except anthropic.APIStatusError as exc:
            raise TranslationUnavailable(f"The Anthropic API returned an error ({exc.status_code}).") from exc
        except anthropic.APIConnectionError as exc:
            raise TranslationUnavailable("Could not reach the Anthropic API. Check the internet connection.") from exc

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.last_usage = {"input_tokens": getattr(usage, "input_tokens", 0),
                               "output_tokens": getattr(usage, "output_tokens", 0),
                               "model": getattr(response, "model", self.model)}
        if response.stop_reason == "refusal":
            raise TranslationUnavailable("Claude declined to translate this page.")
        if response.stop_reason == "max_tokens":
            raise TranslationUnavailable("The translation was cut off (page has too much text for one request).")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TranslationUnavailable("Claude's answer could not be read as JSON.") from exc


def find_claude_code() -> Optional[str]:
    """Path of the Claude Code command line, if it is installed."""
    exe = shutil.which("claude")
    if exe:
        return exe
    for cand in (Path.home() / ".local" / "bin" / "claude.exe", Path.home() / ".local" / "bin" / "claude"):
        if cand.exists():
            return str(cand)
    return None


class ClaudeCodeTranslator(_PageLocaliser):
    """Claude through the Claude Code command line, using its existing login.

    Runs ``claude -p`` once per page with every tool switched off and the
    page handed over on standard input.  Nothing is written to the Claude
    Code session history.
    """

    name = "claude-code"

    def __init__(self, model: str = "", effort: str = "", exe: Optional[str] = None,
                 timeout: float = 600.0, runner=None):
        super().__init__()
        self.exe = exe or find_claude_code()
        if self.exe is None and runner is None:
            raise TranslationUnavailable(
                "Claude Code is not installed on this computer (the 'claude' command was not found).")
        self.model = model or os.environ.get("MANGACLEAN_CLAUDE_MODEL", CLAUDE_CODE_MODEL)
        self.effort = effort or os.environ.get("MANGACLEAN_CLAUDE_EFFORT", CLAUDE_EFFORT)
        self.timeout = timeout
        self._run = runner or self._subprocess

    def command(self, system: str) -> List[str]:
        return [self.exe or "claude", "-p",
                "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                "--tools", "",                       # a translator needs no tools
                "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config",
                "--model", self.model, "--effort", self.effort,
                "--system-prompt", system,
                "--json-schema", json.dumps(SCHEMA, separators=(",", ":"))]

    def _subprocess(self, args: List[str], stdin: bytes) -> tuple:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # a neutral folder, so no project instructions or settings are picked up
        proc = subprocess.run(args, input=stdin, capture_output=True, timeout=self.timeout,
                              cwd=tempfile.gettempdir(), creationflags=flags)
        return proc.returncode, proc.stdout, proc.stderr

    def _ask(self, system: str, content: List[dict]) -> dict:
        message = {"type": "user", "message": {"role": "user", "content": content}}
        try:
            code, stdout, stderr = self._run(self.command(system), (json.dumps(message) + "\n").encode("utf-8"))
        except FileNotFoundError as exc:
            raise TranslationUnavailable("The 'claude' command could not be started.") from exc
        except subprocess.TimeoutExpired as exc:
            raise TranslationUnavailable("Claude Code took too long to answer; try the page again.") from exc

        result = None
        for line in stdout.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "result":
                result = event
        if result is None:
            lines = [ln for ln in stderr.decode("utf-8", "replace").strip().splitlines() if ln.strip()]
            detail = lines[-1] if lines else "no answer"
            hint = " Open a terminal, run 'claude' once and log in." if "log" in detail.lower() else ""
            raise TranslationUnavailable(f"Claude Code did not answer ({detail[:160]}).{hint}")
        usage = result.get("usage") or {}
        self.last_usage = {"input_tokens": usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)
                           + usage.get("cache_read_input_tokens", 0),
                           "output_tokens": usage.get("output_tokens", 0),
                           "cost_usd": result.get("total_cost_usd"),
                           "seconds": (result.get("duration_ms") or 0) / 1000.0,
                           "model": ", ".join((result.get("modelUsage") or {}).keys()) or self.model}
        if result.get("is_error") or result.get("subtype") != "success":
            text = str(result.get("result") or result.get("subtype") or "error")
            low = text.lower()
            if "login" in low or "api key" in low or "authenticat" in low:
                raise TranslationUnavailable(
                    "Claude Code is not logged in. Open a terminal, run 'claude' once and log in.")
            if "limit" in low:
                when = ""
                m = re.search(r"\|(\d{9,11})", text)
                if m:
                    import datetime
                    when = " It resets at " + datetime.datetime.fromtimestamp(int(m.group(1))).strftime(
                        "%a %d %b, %H:%M") + "."
                raise TranslationUnavailable("Your Claude usage limit has been reached." + when
                                             + " The offline translator still works meanwhile.")
            raise TranslationUnavailable(f"Claude Code reported a problem: {text[:200]}")
        data = result.get("structured_output")
        if not isinstance(data, dict):
            try:
                data = json.loads(result.get("result") or "")
            except (TypeError, ValueError) as exc:
                raise TranslationUnavailable("Claude Code's answer could not be read.") from exc
        return data


# -------------------------------------------------------------- availability
def claude_available() -> Optional[str]:
    """None if the Claude API engine can be used, else a one-line reason."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "the 'anthropic' package is not installed"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
            or os.environ.get("ANTHROPIC_PROFILE"):
        return None
    if (Path.home() / ".config" / "anthropic").exists():
        return None
    return "no Anthropic API key found (set ANTHROPIC_API_KEY or run 'ant auth login')"


def claude_code_available() -> Optional[str]:
    """None if the Claude Code engine can be used, else a one-line reason."""
    return None if find_claude_code() else "Claude Code is not installed (the 'claude' command was not found)"


def engine_problems() -> Dict[str, Optional[str]]:
    """For every engine: None if usable, else why not."""
    return {"claude-code": claude_code_available(), "claude": claude_available(), "offline": None}


def best_engine() -> str:
    """The best engine that can run here: Claude when possible, else offline."""
    problems = engine_problems()
    for engine in ENGINES:
        if problems[engine] is None:
            return engine
    return "offline"
