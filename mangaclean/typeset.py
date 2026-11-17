"""Lettering: draw the translation back into the page.

For every text block this module

1. works out *where* the English may go - the inside of the balloon the
   Japanese sat in (found on the cleaned page), or a box around the original
   lettering when there is no balloon;
2. picks a lettering style that carries over the feel of the original:
   balloon dialogue in a comic hand, oversized balloon text in a shout face,
   big captions in a heavy display face, serif text in a serif face; the same
   ink colour, the same outline if the original had one, and a size that
   keeps the original's proportions between loud and quiet text;
3. fits the words to the *shape* of that area (lines are longer where a
   balloon is wider), choosing the largest size that fits, and draws them.

The fonts are ordinary files in the ``fonts`` folder.  To letter with your
own, drop them there and name them in ``fonts/fonts.json``, for example
``{"dialogue": "MyComicFont.ttf", "shout": "MyLoudFont.otf"}``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .blocks import TextBlock, erased_fraction
from .models import default_fonts_dir, ensure_font
from .textmask import _cc, _cdist, to_gray

SS = 3                                   # text is drawn at 3x and scaled down
FLAT_TOL = 28

ROLES = ("dialogue", "shout", "title", "narration", "handwritten")
ROLE_FONT = {
    "dialogue": "ComicNeue-Bold.ttf",
    "shout": "Bangers-Regular.ttf",
    "title": "Anton-Regular.ttf",
    "narration": "CrimsonText-Bold.ttf",
    "handwritten": "PatrickHand-Regular.ttf",
}
ROLE_CAPS = {"dialogue": True, "shout": True, "title": True, "narration": False, "handwritten": True}
# fonts that ship with Windows, used only if a default cannot be downloaded
SYSTEM_FALLBACK = {
    "dialogue": ("comicbd.ttf", "arialbd.ttf"), "shout": ("impact.ttf", "ariblk.ttf"),
    "title": ("impact.ttf", "ariblk.ttf"), "narration": ("georgiab.ttf", "timesbd.ttf"),
    "handwritten": ("comic.ttf", "arial.ttf"),
}


class FontBook:
    """Resolves lettering roles to font files and caches the loaded faces."""

    def __init__(self, fonts_dir: Optional[Path] = None, log: Optional[Callable[[str], None]] = None):
        self.dir = Path(fonts_dir) if fonts_dir else default_fonts_dir()
        self.log = log or (lambda s: None)
        self.custom: Dict[str, str] = {}
        cfg = self.dir / "fonts.json"
        if cfg.exists():
            try:
                self.custom = {k: str(v) for k, v in json.loads(cfg.read_text(encoding="utf-8")).items()
                               if k in ROLES}
            except (ValueError, OSError):
                self.log("fonts.json could not be read; using the default fonts")
        self._paths: Dict[str, str] = {}
        self._faces: Dict[Tuple[str, float], ImageFont.FreeTypeFont] = {}
        self._caps: Dict[str, float] = {}

    def path(self, role: str) -> str:
        if role in self._paths:
            return self._paths[role]
        found = None
        name = self.custom.get(role)
        if name:
            p = Path(name) if Path(name).is_absolute() else self.dir / name
            if p.exists():
                found = str(p)
            else:
                self.log(f"fonts.json names {name} for '{role}', but that file is missing")
        if found is None:
            p = ensure_font(ROLE_FONT[role], self.dir, self.log)
            found = str(p) if p else None
        if found is None:
            windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
            for cand in SYSTEM_FALLBACK[role]:
                if (windir / cand).exists():
                    found = str(windir / cand)
                    break
        if found is None:
            raise RuntimeError("No usable lettering font was found. Put a .ttf file in the fonts folder.")
        self._paths[role] = found
        return found

    def face(self, role: str, size: float) -> ImageFont.FreeTypeFont:
        key = (role, round(float(size), 2))
        f = self._faces.get(key)
        if f is None:
            f = ImageFont.truetype(self.path(role), key[1])
            if len(self._faces) > 400:
                self._faces.clear()
            self._faces[key] = f
        return f

    def cap(self, role: str) -> float:
        """Height of a capital letter as a share of the font size."""
        if role not in self._caps:
            box = self.face(role, 200).getbbox("H")
            self._caps[role] = max(0.4, (box[3] - box[1]) / 200.0)
        return self._caps[role]


# ------------------------------------------------------------------- words
_hyphenator = None


def _syllable_cuts(word: str) -> List[int]:
    global _hyphenator
    if _hyphenator is None:
        try:
            import pyphen
            _hyphenator = pyphen.Pyphen(lang="en_US")
        except Exception:                                   # noqa: BLE001
            _hyphenator = False
    core = word.strip(".,!?\"'()-:;")
    if not _hyphenator or len(core) < 6 or not core.isalpha():
        return []
    start = word.find(core)
    return [start + c for c in _hyphenator.positions(core.lower()) if 2 <= c <= len(core) - 3]


def split_words(text: str, caps: bool) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if caps:
        text = text.upper()
    text = re.sub(r"\s+(\.\.\.|[!?,.])", r"\1", text)       # never start a line with punctuation
    return [w for w in text.split(" ") if w]


def _wrap(words: List[str], widths: List[float], measure) -> Optional[List[str]]:
    """Greedy wrap into lines of the given widths; None if it does not fit."""
    words = list(words)
    lines: List[str] = []
    i = 0
    for width in widths:
        if i >= len(words):
            break
        line: List[str] = []
        while i < len(words):
            trial = " ".join(line + [words[i]])
            if measure(trial) <= width:
                line.append(words[i])
                i += 1
                continue
            # the word does not fit; break it at a syllable if that helps
            cuts = _syllable_cuts(words[i])
            done = False
            if cuts and (not line or len(words[i]) >= 9):
                for c in reversed(cuts):
                    head = words[i][:c] + "-"
                    if measure(" ".join(line + [head])) <= width:
                        line.append(head)
                        words[i] = words[i][c:]
                        done = True
                        break
            if not done and not line:
                return None                                 # a single word wider than the line
            break
        lines.append(" ".join(line))
    return lines if i >= len(words) else None


# -------------------------------------------------------------------- area
def _row_extents(area: np.ndarray, ax: int):
    """For every row: the run of free pixels that contains column ``ax``."""
    h, w = area.shape
    cols = np.arange(w)
    blocked = ~area
    last = np.maximum.accumulate(np.where(blocked, cols, -1), axis=1)[:, ax]
    first = np.minimum.accumulate(np.where(blocked, cols, w)[:, ::-1], axis=1)[:, ::-1][:, ax]
    ok = area[:, ax]
    left = np.where(ok, last + 1, ax)
    right = np.where(ok, first, ax)
    return left, right, ok


def find_area(clean: np.ndarray, block: TextBlock, cell: Optional[np.ndarray], font_px: float) -> dict:
    """Where may the translation of ``block`` be drawn?

    ``clean`` is the cleaned page (grey or RGB).  Returns the area as a mask
    in a window around the block, and whether it is a balloon.
    """
    H, W = clean.shape[:2]
    x1, y1, x2, y2 = block.box
    bw, bh = x2 - x1, y2 - y1
    f = block.font
    ex = int(0.75 * max(bw, bh) + 2 * f)
    wx1, wy1, wx2, wy2 = max(0, x1 - ex), max(0, y1 - ex), min(W, x2 + ex), min(H, y2 + ex)
    win = clean[wy1:wy2, wx1:wx2]
    bx1, by1, bx2, by2 = x1 - wx1, y1 - wy1, x2 - wx1, y2 - wy1
    inner = win[by1:by2, bx1:bx2]
    px = inner.reshape(-1, inner.shape[2]) if inner.ndim == 3 else inner.reshape(-1, 1)
    bg = np.median(px, axis=0).astype(np.int16)
    same = _cdist(win, bg) <= FLAT_TOL
    flat = float(same[by1:by2, bx1:bx2].mean()) >= 0.9

    # room the translation may take beyond the original lettering when there
    # is no balloon to hold it
    dx = max(0.3 * bw, 1.0 * f) + (0.3 * bh if block.vertical and block.n_lines > 1 else 0.0)
    dy = 0.4 * f
    loose = np.zeros(same.shape, bool)
    loose[max(0, int(by1 - dy)):int(by2 + dy) + 1, max(0, int(bx1 - dx)):int(bx2 + dx) + 1] = True

    balloon = False
    if flat:
        n, lab, st = _cc(same, 4)
        counts = np.bincount(lab[by1:by2, bx1:bx2].ravel(), minlength=n)
        counts[0] = 0
        free = (counts >= 0.2 * counts.sum())[lab]
        # specks left inside the balloon must not carve the area up
        holes = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((3, 3), np.uint8)).astype(bool)
        free = holes
        edge = np.concatenate([free[0], free[-1], free[:, 0], free[:, -1]])
        balloon = float(edge.mean()) < 0.03
        area = free if balloon else (free & loose)
    else:
        area = loose
    if cell is not None:
        area = area & cell[wy1:wy2, wx1:wx2]
    if balloon:                       # keep the lettering off the outline
        k = max(2, int(round(0.22 * font_px)))
        area = cv2.erode(area.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(bool)
    return {"mask": area, "origin": (wx1, wy1), "flat": flat, "balloon": balloon,
            "bg": tuple(int(v) for v in bg)}


def _anchor(area: np.ndarray, origin, block: TextBlock) -> Tuple[int, int]:
    """Point the lettering is centred on: the original text's centre, moved to
    the middle of the balloon at that height."""
    h, w = area.shape
    ox, oy = origin
    x1, y1, x2, y2 = block.box
    ax = int(np.clip((x1 + x2) // 2 - ox, 0, w - 1))
    ay = int(np.clip((y1 + y2) // 2 - oy, 0, h - 1))
    if not area.any():
        return ax, ay
    if not area[ay, ax]:                          # e.g. the centre fell on a leftover speck
        dist = cv2.distanceTransform(area.astype(np.uint8), cv2.DIST_L2, 3)
        ay, ax = np.unravel_index(int(dist.argmax()), dist.shape)
        return int(ax), int(ay)
    rows = slice(max(0, y1 - oy), min(h, y2 - oy))
    left, right, ok = _row_extents(area[rows], ax)
    if ok.any():
        ax = int(np.clip(round(float(np.median((left[ok] + right[ok]) / 2.0))), 0, w - 1))
        if not area[ay, ax]:
            ax = int(np.clip((x1 + x2) // 2 - ox, 0, w - 1))
    return ax, ay


# ------------------------------------------------------------------ layout
def fit_text(words: List[str], book: FontBook, role: str, area: np.ndarray,
             anchor: Tuple[int, int], size_max: float, size_min: float) -> Optional[dict]:
    """Largest size at which ``words`` fit the area, with the line breaks."""
    if not words or not area.any():
        return None
    ax, ay = anchor
    left, right, ok = _row_extents(area, ax)
    if not ok[ay]:
        return None
    top = ay
    while top > 0 and ok[top - 1]:
        top -= 1
    bot = ay + 1
    while bot < len(ok) and ok[bot]:
        bot += 1
    cap_ratio = book.cap(role)
    lead = 1.34 if ROLE_CAPS[role] else 1.62          # line pitch, in cap heights

    size = float(size_max)
    while size >= size_min - 1e-6:
        face = book.face(role, size)
        cap = cap_ratio * size
        pitch = lead * cap
        max_lines = int((bot - top - cap) // pitch) + 1 if bot - top >= cap else 0
        measure = face.getlength
        for n in range(1, max_lines + 1):
            height = (n - 1) * pitch + cap
            y0 = float(np.clip(ay - height / 2.0, top, bot - height))
            widths = []
            for i in range(n):
                a = int(np.floor(y0 + i * pitch))
                b = int(np.ceil(y0 + i * pitch + cap)) + 1
                seg_l, seg_r = left[a:b], right[a:b]
                if not ok[a:b].all():
                    widths.append(0.0)
                else:                                   # lines share one centre axis
                    widths.append(2.0 * float(min(ax - seg_l.max(), seg_r.min() - ax)))
            if min(widths) <= 0:
                continue
            lines = _wrap(words, widths, measure)
            if lines is None or len(lines) > n:
                continue
            # even the lines out: the tightest widths that still fit in n lines
            lo, hi = 0.5, 1.0
            best = lines
            for _ in range(6):
                mid = (lo + hi) / 2
                trial = _wrap(words, [w * mid for w in widths], measure)
                if trial is not None and len(trial) <= len(lines):
                    best, hi = trial, mid
                else:
                    lo = mid
            height = (len(best) - 1) * pitch + cap
            y0 = float(np.clip(ay - height / 2.0, top, bot - height))
            return {"role": role, "size": size, "lines": best, "x": float(ax),
                    "baselines": [y0 + cap + i * pitch for i in range(len(best))],
                    "cap": cap, "pitch": pitch}
        size -= max(0.5, round(size * 0.04 * 2) / 2)
    return None


def _force_fit(words: List[str], book: FontBook, role: str, shape, anchor, size: float) -> dict:
    """Last resort when nothing fits: wrap to the area's width and overflow."""
    h, w = shape
    face = book.face(role, size)
    cap = book.cap(role) * size
    pitch = (1.34 if ROLE_CAPS[role] else 1.62) * cap
    width = max(3.0 * size, 0.8 * w)
    lines = None
    while lines is None:
        lines = _wrap(words, [width] * 200, face.getlength)
        width *= 1.25
    height = (len(lines) - 1) * pitch + cap
    y0 = anchor[1] - height / 2.0
    return {"role": role, "size": size, "lines": lines, "x": float(anchor[0]),
            "baselines": [y0 + cap + i * pitch for i in range(len(lines))], "cap": cap, "pitch": pitch,
            "overflow": True}


# ------------------------------------------------------------------- style
def _snap(colour: np.ndarray) -> Tuple[int, ...]:
    """Lettering is printed in solid ink: pull near-black/white to the extreme."""
    lum = float(np.mean(colour))
    if lum < 80:
        return tuple(0 for _ in colour)
    if lum > 185 and float(np.ptp(colour)) < 30:
        return tuple(255 for _ in colour)
    return tuple(int(v) for v in colour)


def read_style(orig: np.ndarray, block: TextBlock, bg) -> dict:
    """Ink colour and outline of the original lettering of ``block``."""
    x1, y1, x2, y2 = block.box
    crop = orig[y1:y2, x1:x2]
    gray = to_gray(crop).astype(np.int16)
    m8 = block.mask.astype(np.uint8)
    hug = (cv2.dilate(m8, np.ones((7, 7), np.uint8)) > 0)
    vals = gray[hug]
    t, _ = cv2.threshold(vals.reshape(-1, 1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark, light = block.mask & (gray <= t), block.mask & (gray > t)
    # the lettering is the side that differs from what surrounds the block
    bg_l = float(np.mean(bg))
    ink = dark if abs(float(gray[dark].mean()) - bg_l if dark.any() else 0) >= \
        abs(float(gray[light].mean()) - bg_l if light.any() else 0) else light
    if ink.sum() < 6:
        ink = block.mask
    px = crop[ink]
    ink_col = np.median(px.reshape(len(px), -1), axis=0)
    out = {"ink": _snap(ink_col), "outline": None}

    # an outline: a band of one contrasting colour hugging the strokes
    i8 = ink.astype(np.uint8)
    ring = (cv2.dilate(i8, np.ones((5, 5), np.uint8)) > 0) & ~(cv2.dilate(i8, np.ones((3, 3), np.uint8)) > 0)
    if ring.sum() >= 20:
        rp = crop[ring].reshape(int(ring.sum()), -1).astype(np.int16)
        med = np.median(rp, axis=0)
        uniform = float((np.abs(rp - med).max(axis=1) <= FLAT_TOL).mean())
        if uniform >= 0.6 and float(np.abs(med - ink_col).max()) >= 70:
            out["halo"] = _snap(med)
    return out


def choose_role(block: TextBlock, balloon: bool, dominant: float) -> str:
    q = block.font / max(1.0, dominant)
    if balloon:
        return "shout" if q >= 1.5 else "dialogue"
    return "title" if q >= 1.8 else "dialogue"


# --------------------------------------------------------------- rendering
def _draw(shape, layout: dict, book: FontBook, stroke: float = 0.0) -> np.ndarray:
    """Alpha mask (float 0-1, window sized) of the laid-out text."""
    h, w = shape
    im = Image.new("L", (w * SS, h * SS), 0)
    d = ImageDraw.Draw(im)
    face = book.face(layout["role"], layout["size"] * SS)
    sw = int(round(stroke * SS))
    for line, base in zip(layout["lines"], layout["baselines"]):
        d.text((layout["x"] * SS, base * SS), line, font=face, fill=255, anchor="ms",
               stroke_width=sw, stroke_fill=255)
    return np.asarray(im.reduce(SS), np.float32) / 255.0


class Typesetter:
    def __init__(self, fonts_dir: Optional[Path] = None, log: Optional[Callable[[str], None]] = None):
        self.book = FontBook(fonts_dir, log)

    def typeset(self, page, clean_rgb: np.ndarray) -> np.ndarray:
        """Draw every translated block of ``page`` onto the cleaned page."""
        out = clean_rgb.copy()
        H, W = out.shape[:2]
        flat_m, art_m = page.masks()
        todo = [b for b in page.blocks
                if b.enabled and b.translation.strip() and erased_fraction(b, flat_m, art_m) >= 0.6]
        for b in page.blocks:
            if b not in todo:
                b.layout = None
        if not todo:
            return out
        clean = to_gray(out) if page.gray else out

        # every pixel belongs to the nearest block, so lettering never collides
        seeds = np.full((H, W), 255, np.uint8)
        for b in todo:
            seeds[b.box[1]:b.box[3], b.box[0]:b.box[2]] = 0
        _, labels = cv2.distanceTransformWithLabels(seeds, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)

        typical = [b.font for b in todo if b.n_lines >= 2] or [b.font for b in todo]
        dominant = float(np.median(typical))

        plans = []
        for b in todo:
            cy, cx = (b.box[1] + b.box[3]) // 2, (b.box[0] + b.box[2]) // 2
            cell = labels == labels[min(H - 1, cy), min(W - 1, cx)]
            override = b.style.get("role_override")
            guess = 0.9 * b.font
            area = find_area(clean, b, cell, guess)
            role = override if override in ROLES else choose_role(b, area["balloon"], dominant)
            look = read_style(page.work, b, area["bg"])
            caps = ROLE_CAPS[role]
            words = split_words(b.translation, caps)
            # capitals about two thirds as tall as the original glyphs
            size_max = 0.66 * b.font / self.book.cap(role) * float(b.style.get("scale", 1.0))
            if not caps:
                size_max *= 1.25
            size_max = float(np.clip(size_max, 6.0, 0.2 * max(H, W)))
            plans.append({"block": b, "area": area, "role": role, "look": look, "words": words,
                          "size_max": size_max, "size_min": max(5.0, 0.45 * size_max)})

        for p in plans:
            p["fit"] = self._fit(p, p["size_max"])
        # ordinary dialogue is lettered at one size across the page
        peers = [p for p in plans if p["role"] == "dialogue" and p["fit"] and not p["fit"].get("overflow")
                 and 0.8 <= p["block"].font / dominant <= 1.25 and "scale" not in p["block"].style]
        if len(peers) >= 2:
            common = float(np.median([p["fit"]["size"] for p in peers]))
            for p in peers:
                if p["fit"]["size"] > common + 0.25:
                    p["fit"] = self._fit(p, common)

        for p in plans:
            self._paint(out, p, page.gray)
        return out

    # -- one block
    def _rotated(self, p: dict) -> bool:
        """A single tall column outside a balloon is lettered sideways,
        reading downwards, like the spine of a book."""
        b, area = p["block"], p["area"]
        return (b.vertical and b.n_lines == 1 and not area["balloon"]
                and b.height >= 4.5 * b.width)

    def _fit(self, p: dict, size_max: float) -> dict:
        b, area = p["block"], p["area"]
        mask = area["mask"]
        ax, ay = _anchor(mask, area["origin"], b)
        p["turn"] = self._rotated(p)
        if p["turn"]:
            h, w = mask.shape
            mask = np.rot90(mask, 1)                     # lay the column on its side
            ax, ay = ay, w - 1 - ax
        p["shape"] = mask.shape
        fit = fit_text(p["words"], self.book, p["role"], mask, (ax, ay), size_max, p["size_min"])
        if fit is None:
            fit = _force_fit(p["words"], self.book, p["role"], mask.shape, (ax, ay), p["size_min"])
        return fit

    def _paint(self, out: np.ndarray, p: dict, gray_page: bool) -> None:
        b, area, fit, look = p["block"], p["area"], p["fit"], p["look"]
        ox, oy = area["origin"]
        ink = np.array(look["ink"], np.float32)
        halo = look.get("halo")
        stroke = 0.0
        if halo is not None and not area["flat"]:
            stroke = max(1.2, 0.13 * fit["size"])        # the original was outlined: keep that
        alpha = _draw(p["shape"], fit, self.book)
        edge = _draw(p["shape"], fit, self.book, stroke) if stroke else None
        if p["turn"]:
            alpha = np.rot90(alpha, -1)
            edge = np.rot90(edge, -1) if edge is not None else None
        h, w = alpha.shape
        region = out[oy:oy + h, ox:ox + w].astype(np.float32)

        def tone(c):
            c = np.array(c, np.float32)
            if c.size == 1:
                c = np.repeat(c, 3)
            return np.full(3, c.mean(), np.float32) if gray_page else c

        if edge is not None:
            region = region * (1 - edge[..., None]) + tone(halo) * edge[..., None]
        region = region * (1 - alpha[..., None]) + tone(ink) * alpha[..., None]
        out[oy:oy + h, ox:ox + w] = np.clip(region + 0.5, 0, 255).astype(np.uint8)

        ys, xs = np.nonzero(alpha > 0.02)
        box = [ox + int(xs.min()), oy + int(ys.min()), ox + int(xs.max()) + 1, oy + int(ys.max()) + 1] \
            if xs.size else list(b.box)
        b.layout = {"role": p["role"], "size": round(float(fit["size"]), 1), "lines": fit["lines"],
                    "box": box, "balloon": bool(area["balloon"]), "overflow": bool(fit.get("overflow")),
                    "rotated": bool(p["turn"])}
