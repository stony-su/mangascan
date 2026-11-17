"""Reading Japanese lettering: manga-ocr on ONNX Runtime.

manga-ocr is an image-to-text transformer trained on manga: it copes with
vertical and horizontal text, several lines at once, ruby text (which it
skips) and lettering on top of artwork.  It expects one text block per image.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

SIZE = 224
START, END = 2, 3                 # [CLS], [SEP]
_KANA_KANJI = re.compile(r"[぀-ヿ㐀-鿿]")


def has_japanese(text: str) -> bool:
    return bool(_KANA_KANJI.search(text))


def tidy(text: str) -> str:
    """Normalise what the recogniser returns."""
    text = "".join(text.split())
    return re.sub(r"[・.．｡。]{2,}|…+",
                  lambda m: "…" * max(1, round(len(m.group(0)) / (1 if "…" in m.group(0) else 3))), text)


class MangaOcr:
    def __init__(self, files: Dict[str, Path], threads: int = 0):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        prov = ["CPUExecutionProvider"]
        self.encoder = ort.InferenceSession(str(files["encoder_model.onnx"]), sess_options=opts, providers=prov)
        self.decoder = ort.InferenceSession(str(files["decoder_model.onnx"]), sess_options=opts, providers=prov)
        self.vocab: List[str] = Path(files["vocab.txt"]).read_text(encoding="utf-8").split("\n")

    # ------------------------------------------------------------ network
    def read_image(self, gray: np.ndarray, max_tokens: int = 160) -> Tuple[str, float, int]:
        """Read one image showing a single block of text.

        Returns ``(text, confidence, tokens)``; confidence is the geometric
        mean of the probabilities of the characters that were chosen.
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(gray, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
        memory = self.encoder.run(None, {"pixel_values": np.repeat(x[None, None], 3, axis=1)})[0]

        ids, logp = [START], []
        for _ in range(max_tokens):
            logits = self.decoder.run(["logits"], {
                "input_ids": np.array([ids], np.int64), "encoder_hidden_states": memory})[0][0, -1]
            logits = logits - logits.max()
            lp = logits - np.log(np.exp(logits).sum())
            nxt = int(lp.argmax())
            if nxt == END:
                break
            ids.append(nxt)
            logp.append(float(lp[nxt]))
            # a transformer that has lost the plot repeats itself; stop it
            n = len(ids)
            looped = any(n > 3 * k + 1 and ids[-k:] == ids[-2 * k:-k] == ids[-3 * k:-2 * k]
                         and len(set(ids[-k:])) > 1 for k in (2, 3, 4, 5, 6, 8))
            if looped:
                return "", 0.0, n
        chars = []
        for i in ids[1:]:
            tok = self.vocab[i] if 0 <= i < len(self.vocab) else ""
            if (tok.startswith("[") and tok.endswith("]")) or tok.startswith("<unused"):
                continue
            chars.append(tok[2:] if tok.startswith("##") else tok)
        conf = float(np.exp(np.mean(logp))) if logp else 0.0
        return "".join(chars), conf, len(logp)

    # -------------------------------------------------------------- blocks
    @staticmethod
    def _view(rgb: np.ndarray, block, margin: int):
        """Grey crop of a block (with margin), plus the offset of the block in it.

        When the lettering sits on artwork, everything that is not lettering
        is painted over with the tone right next to the strokes, so the
        recogniser sees clean text instead of text tangled in hatching.
        """
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = block.box
        vx1, vy1, vx2, vy2 = max(0, x1 - margin), max(0, y1 - margin), min(W, x2 + margin), min(H, y2 + margin)
        crop = rgb[vy1:vy2, vx1:vx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.copy()
        mask = np.zeros(gray.shape, np.uint8)
        mask[y1 - vy1:y2 - vy1, x1 - vx1:x2 - vx1] = block.mask
        k = max(2, int(round(0.2 * block.font)))
        near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
        hug = near & (cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0)
        if hug.sum() >= 8 and (~near).sum() >= 8:
            tone = float(np.median(gray[hug]))
            busy = float((np.abs(gray[~near].astype(np.float32) - tone) > 40).mean())
            if busy > 0.15:                       # artwork around the lettering
                gray = np.where(near, gray, np.uint8(round(tone))).astype(np.uint8)
        return gray, (x1 - vx1, y1 - vy1)

    def read_block(self, rgb: np.ndarray, block) -> Tuple[str, float]:
        """Read a text block from the original page. Returns ``(text, confidence)``.

        An ordinary balloon (a few lines of up to a dozen glyphs) is read in
        one go, which is what the network was trained on.  The network
        squeezes its input into a square, so anything long and thin is read
        line by line, and long lines piece by piece.
        """
        font = max(1.0, block.font)
        m = int(round(max(2.0, 0.18 * font)))
        view, (ox, oy) = self._view(rgb, block, m)
        vh, vw = view.shape
        bw, bh = block.width, block.height
        along, across = (bh, bw) if block.vertical else (bw, bh)

        def budget(length: float, lines: int = 1) -> int:
            return int(1.6 * lines * length / font) + 10   # a little more than can physically fit

        if along / font <= 12.5 and across <= 4.0 * along:
            text, conf, _ = self.read_image(view, budget(along, block.n_lines))
            return tidy(text), conf

        texts, logsum, count = [], 0.0, 0
        for a, b in block.spans:
            if block.vertical:
                profile = block.mask[:, a:b].any(axis=1)
            else:
                profile = block.mask[a:b, :].any(axis=0)
            limit = 9.0 * font                          # about nine glyphs per piece
            pieces, start = [], 0
            while along - start > 1.35 * limit:
                lo, hi = int(start + 0.7 * limit), int(min(along - 1, start + 1.2 * limit))
                gaps = [p for p in range(lo, hi) if not profile[p]]
                target = start + limit
                cut = min(gaps, key=lambda p: abs(p - target)) if gaps else int(target)
                pieces.append((start, cut))
                start = cut
            pieces.append((start, along))
            for s, e in pieces:
                if block.vertical:
                    crop = view[max(0, oy + s - m):min(vh, oy + e + m), max(0, ox + a - m):min(vw, ox + b + m)]
                else:
                    crop = view[max(0, oy + a - m):min(vh, oy + b + m), max(0, ox + s - m):min(vw, ox + e + m)]
                if crop.size == 0:
                    continue
                text, conf, n = self.read_image(crop, budget(e - s))
                texts.append(text)
                logsum += n * np.log(max(conf, 1e-6))
                count += n
        conf = float(np.exp(logsum / count)) if count else 0.0
        return tidy("".join(texts)), conf
