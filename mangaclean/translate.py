"""Japanese -> English translation of the text blocks of a page.

Two engines, same interface:

``offline``  FuguMT, a compact translation model that runs on this computer.
             No account, no network, nothing leaves the machine.  It
             translates each block on its own, so the result reads like
             machine translation: fine for a first draft.
``claude``   The Claude API.  The whole page is translated in one request,
             with the page image for context, so names, tone and who is
             speaking carry across balloons, and lettering the local OCR
             could not read is read from the image instead.  Needs Anthropic
             credentials, costs money per page, and sends the page to
             Anthropic - it is therefore never used unless asked for.
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .ocr import has_japanese

MIN_CONFIDENCE = 0.70            # below this the OCR result is treated as unread
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"

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
                       progress: Optional[Callable[[int, int], None]] = None) -> List[dict]:
        out = []
        for i, it in enumerate(items):
            text = it.get("text", "")
            simple = punctuation_only(text)
            if simple is not None:
                english = simple
            elif readable(it):
                english = self.translate(text)
            else:
                english = ""                      # unread: left blank for a human
            out.append({"id": it["id"], "text": text, "translation": english})
            if progress:
                progress(i + 1, len(items))
        return out


# ------------------------------------------------------------------- claude
SYSTEM_PROMPT = """You translate Japanese manga into English for lettering.

You get one page image and a list of its text blocks in reading order (top to \
bottom, right to left). Each block has an id, its position on the page as \
percentages, and what a local OCR model read there together with that model's \
confidence. The OCR is usually right for ordinary speech balloons and often \
wrong for small, stylised or outlined lettering and for rare name kanji, so \
check every reading against the image and correct it.

For every block return the Japanese as it actually appears and an English \
translation that will be lettered into the same balloon:

- Translate the meaning naturally, the way an official English edition would \
read, not word for word. Use the whole page for context: who is speaking, to \
whom, and in what mood.
- Keep it about as short as the Japanese allows. The text has to fit a balloon \
drawn for a few vertical columns, so prefer the tighter of two good phrasings.
- Keep names in their usual romanised form and keep them consistent across \
blocks. Do not add honorifics the original does not have.
- Carry over trailing ellipses, dashes, exclamation and question marks.
- A block that is only an ellipsis or punctuation becomes the same in ASCII.
- Use ordinary sentence case; the letterer applies the comic capitals.
- If a block is not text at all, or truly cannot be read, return an empty \
english string for it rather than guessing.

Return every id exactly once."""

SCHEMA = {
    "type": "object",
    "properties": {
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
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


class TranslationUnavailable(RuntimeError):
    """The engine could not be used; the message says why in plain words."""


class ClaudeTranslator:
    """Whole-page translation with the Claude API (official ``anthropic`` SDK)."""

    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL, effort: str = CLAUDE_EFFORT,
                 send_image: bool = True, client=None):
        if client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise TranslationUnavailable(
                    "The Claude engine needs the 'anthropic' package (pip install anthropic).") from exc
            client = anthropic.Anthropic()      # credentials come from the environment
        self.client = client
        self.model = model
        self.effort = effort
        self.send_image = send_image
        self.last_usage: Optional[dict] = None

    @staticmethod
    def _image_block(rgb: np.ndarray) -> dict:
        from PIL import Image

        im = Image.fromarray(rgb)
        if max(im.size) > 1568:                  # larger adds cost, not legibility
            im.thumbnail((1568, 1568), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        return {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("ascii")}}

    def build_request(self, items: List[dict], page_rgb: Optional[np.ndarray]) -> dict:
        h, w = page_rgb.shape[:2] if page_rgb is not None else (1, 1)
        listing = []
        for it in items:
            x1, y1, x2, y2 = it.get("box", (0, 0, 0, 0))
            listing.append({
                "id": it["id"],
                "ocr": it.get("text", ""),
                "ocr_confidence": round(float(it.get("confidence", 0.0)), 2),
                "left_pct": round(100 * x1 / w), "top_pct": round(100 * y1 / h),
                "width_pct": round(100 * (x2 - x1) / w), "height_pct": round(100 * (y2 - y1) / h),
                "direction": "vertical" if it.get("vertical", True) else "horizontal",
            })
        content = []
        if self.send_image and page_rgb is not None:
            content.append(self._image_block(page_rgb))
        content.append({"type": "text", "text":
                        "Text blocks on this page, in reading order:\n"
                        + json.dumps(listing, ensure_ascii=False, indent=1)})
        return {
            "model": self.model,
            "max_tokens": 16000,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": SCHEMA}},
            # if a safety classifier declines the request, the API re-runs it on
            # Anthropic's recommended fallback model instead of returning nothing
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }

    def translate_page(self, items: List[dict], page_rgb: Optional[np.ndarray] = None,
                       progress: Optional[Callable[[int, int], None]] = None) -> List[dict]:
        if not items:
            return []
        import anthropic

        try:
            response = self.client.beta.messages.create(**self.build_request(items, page_rgb))
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

        return self.parse_response(response, items)

    def parse_response(self, response, items: List[dict]) -> List[dict]:
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
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TranslationUnavailable("Claude's answer could not be read as JSON.") from exc
        got = {int(b["id"]): b for b in data.get("blocks", []) if isinstance(b, dict) and "id" in b}
        out = []
        for it in items:
            b = got.get(it["id"])
            if b is None:                         # never trust a model to return every id
                out.append({"id": it["id"], "text": it.get("text", ""), "translation": ""})
                continue
            simple = punctuation_only(b.get("japanese") or it.get("text", ""))
            english = simple if simple is not None else polish(str(b.get("english", "")))
            out.append({"id": it["id"], "text": str(b.get("japanese") or it.get("text", "")),
                        "translation": english})
        if progress:
            progress(len(items), len(items))
        return out


def claude_available() -> Optional[str]:
    """None if the Claude engine can be used, else a one-line reason."""
    import os

    try:
        import anthropic  # noqa: F401
    except ImportError:
        return "the 'anthropic' package is not installed"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") \
            or os.environ.get("ANTHROPIC_PROFILE"):
        return None
    if (Path.home() / ".config" / "anthropic").exists():
        return None
    return "no Anthropic credentials found (set ANTHROPIC_API_KEY or run 'ant auth login')"
