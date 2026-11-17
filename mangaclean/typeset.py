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
CAP_SHARE = 0.74                         # capital height as a share of the original glyph size
EMPHASIS_GROWTH = 1.7                    # how far loud lettering may grow to fill its space
SQUEEZE = 0.88                           # letterers condense text to ~85-93% before shrinking it

ROLES = ("dialogue", "shout", "title", "narration", "handwritten")
# Open-licensed defaults, chosen by lettering real pages with candidates side
# by side: an upright, heavy, compact comic face for dialogue and its italic
# for captions (the usual English-manga pairing).  In fonts.json,
# "File.ttf@650" picks the weight of a variable font.
ROLE_FONT = {
    "dialogue": "BalsamiqSans-Bold.ttf",
    "shout": "Bangers-Regular.ttf",
    "title": "Anton-Regular.ttf",
    "narration": "BalsamiqSans-BoldItalic.ttf",
    "handwritten": "PatrickHand-Regular.ttf",
}
# Professional comic-lettering families.  They cannot be shipped or fetched
# (paid licences), but if one is installed on this computer, or its file is
# dropped into the fonts folder, it is used for dialogue automatically.
# Ordered by how common they are in licensed English manga.
PRO_DIALOGUE = ("wild words", "wildwords", "anime ace", "comicrazy", "meanwhile", "samaritan",
                "blambot classic", "back issues", "yada yada", "joe kubert", "monologous", "face front",
                "hush hush", "imaginary friend", "jack armstrong", "out of line", "tight spot",
                "duty calls", "cloud splitter", "sequentialist", "whizbang", "mighty zeo", "digital strip")
ROLE_CAPS = {"dialogue": True, "shout": True, "title": True, "narration": True, "handwritten": True}
# fonts that ship with Windows, used only if a default cannot be downloaded
SYSTEM_FALLBACK = {
    "dialogue": ("comicbd.ttf", "arialbd.ttf"), "shout": ("impact.ttf", "ariblk.ttf"),
    "title": ("impact.ttf", "ariblk.ttf"), "narration": ("comicz.ttf", "arialbi.ttf"),
    "handwritten": ("comic.ttf", "arial.ttf"),
}


def _family(path: Path) -> str:
    try:
        name = ImageFont.truetype(str(path), 20).getname()
        return f"{name[0]} {name[1]}".lower()
    except Exception:                                       # noqa: BLE001
        return ""


def find_pro_font(fonts_dir: Path, italic: bool = False) -> Optional[Path]:
    """A professional dialogue font the user owns: in the fonts folder or installed.

    With ``italic`` the italic cut of the best family is preferred (captions).
    """
    folders = [fonts_dir, Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        folders.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    found = {}
    for folder in folders:
        if not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.suffix.lower() not in (".ttf", ".otf", ".ttc"):
                continue
            stem = f.stem.lower().replace("-", " ").replace("_", " ")
            # cheap filter on the file name first; opening hundreds of fonts is slow
            if not any(k.replace(" ", "") in stem.replace(" ", "") for k in PRO_DIALOGUE):
                continue
            fam = _family(f) or stem
            for rank, key in enumerate(PRO_DIALOGUE):
                if key in fam or key.replace(" ", "") in fam.replace(" ", ""):
                    # prefer the upright regular cut of the best-ranked family
                    slant = any(w in fam for w in ("italic", "oblique"))
                    heavy = any(w in fam for w in ("bold", "heavy", "black"))
                    found.setdefault((rank, slant != italic, heavy, str(f).lower()), f)
                    break
    return found[min(found)] if found else None


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
        self._specs: Dict[str, Tuple[str, Optional[float]]] = {}
        self._faces: Dict[Tuple[str, float], ImageFont.FreeTypeFont] = {}
        self._caps: Dict[str, float] = {}
        self._unicase: Dict[str, bool] = {}

    @staticmethod
    def _split(spec: str) -> Tuple[str, Optional[float]]:
        """"File.ttf@650" -> ("File.ttf", 650.0)."""
        if "@" in spec:
            name, weight = spec.rsplit("@", 1)
            try:
                return name, float(weight)
            except ValueError:
                return spec, None
        return spec, None

    def spec(self, role: str) -> Tuple[str, Optional[float]]:
        """(font file, weight for a variable font or None) for a role."""
        if role in self._specs:
            return self._specs[role]
        found, weight = None, None
        name = self.custom.get(role)
        if name:                                            # 1. what fonts.json says
            fname, weight = self._split(name)
            p = Path(fname) if Path(fname).is_absolute() else self.dir / fname
            if p.exists():
                found = str(p)
            else:
                weight = None
                self.log(f"fonts.json names {fname} for '{role}', but that file is missing")
        if found is None and role in ("dialogue", "narration"):   # 2. a professional font the user owns
            pro = find_pro_font(self.dir, italic=role == "narration")
            if pro is not None:
                found = str(pro)
                self.log(f"Lettering {role} with {pro.name}")
        if found is None:                                   # 3. the open-licensed default
            fname, weight = self._split(ROLE_FONT[role])
            p = ensure_font(fname, self.dir, self.log)
            found = str(p) if p else None
        if found is None:                                   # 4. something that ships with Windows
            weight = None
            windir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
            for cand in SYSTEM_FALLBACK[role]:
                if (windir / cand).exists():
                    found = str(windir / cand)
                    break
        if found is None:
            raise RuntimeError("No usable lettering font was found. Put a .ttf file in the fonts folder.")
        self._specs[role] = (found, weight)
        return self._specs[role]

    def path(self, role: str) -> str:
        return self.spec(role)[0]

    def face(self, role: str, size: float) -> ImageFont.FreeTypeFont:
        key = (role, round(float(size), 2))
        f = self._faces.get(key)
        if f is None:
            path, weight = self.spec(role)
            f = ImageFont.truetype(path, key[1])
            if weight is not None:
                try:                                        # variable font: set the weight axis
                    axes = f.get_variation_axes()
                    values = []
                    for axis in axes:
                        label = axis.get("name", "")
                        label = label.decode("ascii", "ignore") if isinstance(label, bytes) else str(label)
                        if "eight" in label:
                            values.append(float(np.clip(weight, axis["minimum"], axis["maximum"])))
                        else:
                            values.append(axis["default"])
                    f.set_variation_by_axes(values)
                except (OSError, AttributeError):
                    pass                                    # not a variable font after all
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

    def unicase(self, role: str) -> bool:
        """True for comic fonts whose lower case is a second set of capitals.

        In those fonts the upper-case I carries the crossbars and is meant
        only for the pronoun; every other i is typed in lower case.
        """
        if role not in self._unicase:
            face = self.face(role, 200)
            lo, up = face.getbbox("x"), face.getbbox("X")
            self._unicase[role] = (lo[3] - lo[1]) >= 0.92 * (up[3] - up[1])
        return self._unicase[role]


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


def _name_cuts(word: str) -> List[int]:
    """Break points of a romanised Japanese name: between syllables (Ka-ryo-ten),
    never inside one the way English hyphenation rules would (Kary-oten)."""
    core = word.strip(".,!?\"'()-:;~")
    start = word.find(core)
    low = core.lower()
    vowels = "aeiou"
    cuts = []
    for i in range(2, len(low) - 1):
        prev, nxt = low[i - 1], low[i]
        if prev in vowels and nxt not in vowels and not (nxt == "n" and (i + 1 >= len(low) or low[i + 1] not in vowels + "y")):
            cuts.append(i)
        elif prev == "n" and low[i - 2] in vowels and nxt not in vowels + "y":
            cuts.append(i)                          # after a syllable-final n: Ken-ji
    return [start + c for c in cuts if 2 <= c <= len(core) - 2]


def split_words(text: str, caps: bool, unicase: bool = False) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if caps and unicase:
        # comic convention: crossbar I for the pronoun only
        text = re.sub(r"\bi(?=\b|')", "I", text.lower())
    elif caps:
        text = text.upper()
    text = re.sub(r"\s+(\.\.\.|[!?,.])", r"\1", text)       # never start a line with punctuation
    return [w for w in text.split(" ") if w]


def _wrap(words: List[str], widths: List[float], measure, hyphenate=True,
          nobreak: frozenset = frozenset()) -> Optional[List[str]]:
    """Greedy wrap into lines of the given widths; None if it does not fit.

    Letterers avoid breaking words; ``hyphenate`` is therefore only switched
    on once a size without broken words has been ruled out.
    """
    words = list(words)
    lines: List[str] = []
    broken = set()                                  # a word is broken at most once
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
            core = words[i].strip(".,!?\"'()-:;~").lower()
            if i in broken:
                cuts = []
            elif core in nobreak:                   # a name: only as the very last resort
                cuts = _name_cuts(words[i]) if hyphenate == "all" else []
            else:
                cuts = _syllable_cuts(words[i]) if hyphenate else []
            done = False
            if cuts and not line:
                for c in reversed(cuts):
                    head = words[i][:c] + "-"
                    if measure(" ".join(line + [head])) <= width:
                        line.append(head)
                        words[i] = words[i][c:]
                        broken.add(i)
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


def find_area(clean: np.ndarray, block: TextBlock, cell: Optional[np.ndarray], font_px: float,
              margin: float = 0.22) -> dict:
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
    lx1, ly1 = max(0, int(bx1 - dx)), max(0, int(by1 - dy))
    lx2, ly2 = min(same.shape[1], int(bx2 + dx) + 1), min(same.shape[0], int(by2 + dy) + 1)
    # ... but never across a panel border: a ruled line running the whole
    # height (or width) of the lettering stops the area on that side
    # (a border contrasts with the background: dark rules on a light page,
    # light gutters on a dark one - a black panel itself is not a wall)
    gw = to_gray(win)
    ink = (gw < 90) if float(np.mean(bg)) >= 128 else (gw > 165)
    rows, cols = slice(max(0, by1), by2), slice(max(0, bx1), bx2)
    if by2 > by1 and bx2 > bx1:
        wall_x = ink[rows].mean(axis=0) >= 0.85
        wall_y = ink[:, cols].mean(axis=1) >= 0.85
        gap = max(2, int(round(0.15 * f)))          # and keeps a little air off it
        right = np.flatnonzero(wall_x[bx2:lx2])
        if right.size:
            lx2 = max(bx2, bx2 + int(right[0]) - gap)
        left_w = np.flatnonzero(wall_x[lx1:max(lx1, bx1)])
        if left_w.size:
            lx1 = min(bx1, lx1 + int(left_w[-1]) + 1 + gap)
        below = np.flatnonzero(wall_y[by2:ly2])
        if below.size:
            ly2 = max(by2, by2 + int(below[0]) - gap)
        above = np.flatnonzero(wall_y[ly1:max(ly1, by1)])
        if above.size:
            ly1 = min(by1, ly1 + int(above[-1]) + 1 + gap)
    loose = np.zeros(same.shape, bool)
    loose[ly1:ly2, lx1:lx2] = True

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
    if not balloon:
        # Text set on the page or the art may cross thin lines and hatching,
        # but not a solid area of contrasting colour (a black label, a dark
        # panel): lettering there would be unreadable.
        far = np.abs(to_gray(win).astype(np.int16) - int(round(float(np.mean(bg))))) > 100
        k = max(3, int(round(0.5 * f)))
        solid = cv2.morphologyEx(far.astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        area = area & ~solid
    if cell is not None:
        area = area & cell[wy1:wy2, wx1:wx2]
    if balloon:                       # keep the lettering off the outline
        k = max(2, int(round(margin * font_px)))
        area = cv2.erode(area.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))).astype(bool)
    boxy = False
    if balloon and area.any():
        ys, xs = np.nonzero(area)
        boxy = area.sum() >= 0.93 * (np.ptp(xs) + 1) * (np.ptp(ys) + 1)     # a rectangle fills its bounding box
    return {"mask": area, "origin": (wx1, wy1), "flat": flat, "balloon": balloon, "boxy": bool(boxy),
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
             anchor: Tuple[int, int], size_max: float, size_min: float,
             left_x: Optional[float] = None, max_lines_allowed: int = 999,
             hyphenate: bool = True, squeeze: float = 1.0,
             nobreak: frozenset = frozenset()) -> Optional[dict]:
    """Largest size at which ``words`` fit the area, with the line breaks.

    Lines are centred on the anchor; with ``left_x`` they all start at that x
    instead (captions keep the left edge the original had).
    """
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
        face = book.face(role, size * SS)        # the face the text is drawn with
        cap = cap_ratio * size
        pitch = lead * cap
        max_lines = int((bot - top - cap) // pitch) + 1 if bot - top >= cap else 0
        max_lines = min(max_lines, max_lines_allowed)

        def measure(text, face=face):
            return face.getlength(text) / SS * squeeze
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
                elif left_x is not None:
                    widths.append(float(seg_r.min() - max(left_x, float(seg_l.max()))))
                else:                                   # lines share one centre axis
                    widths.append(2.0 * float(min(ax - seg_l.max(), seg_r.min() - ax)))
            if min(widths) <= 0:
                continue
            lines = _wrap(words, widths, measure, hyphenate, nobreak)
            if lines is None or len(lines) > n:
                continue
            best = lines
            if left_x is None:
                # even the lines out: the tightest widths that still fit in n lines
                lo, hi = 0.5, 1.0
                for _ in range(6):
                    mid = (lo + hi) / 2
                    trial = _wrap(words, [w * mid for w in widths], measure, hyphenate, nobreak)
                    if trial is not None and len(trial) <= len(lines):
                        best, hi = trial, mid
                    else:
                        lo = mid
            height = (len(best) - 1) * pitch + cap
            y0 = float(np.clip(ay - height / 2.0, top, bot - height))
            return {"role": role, "size": size, "lines": best,
                    "x": float(ax if left_x is None else left_x),
                    "align": "center" if left_x is None else "left",
                    "baselines": [y0 + cap + i * pitch for i in range(len(best))],
                    "cap": cap, "pitch": pitch, "width": max(measure(ln) for ln in best),
                    "squeeze": squeeze}
        size -= max(0.5, round(size * 0.04 * 2) / 2)
    return None


def _force_fit(words: List[str], book: FontBook, role: str, shape, anchor, size: float) -> dict:
    """Last resort when nothing fits: wrap to the area's width and overflow."""
    h, w = shape
    face = book.face(role, size * SS)

    def measure(text):
        return face.getlength(text) / SS

    cap = book.cap(role) * size
    pitch = (1.34 if ROLE_CAPS[role] else 1.62) * cap
    width = max(3.0 * size, 0.8 * w)
    lines = None
    while lines is None:
        lines = _wrap(words, [width] * 200, measure)
        width *= 1.25
    height = (len(lines) - 1) * pitch + cap
    y0 = anchor[1] - height / 2.0
    return {"role": role, "size": size, "lines": lines, "x": float(anchor[0]), "align": "center",
            "baselines": [y0 + cap + i * pitch for i in range(len(lines))], "cap": cap, "pitch": pitch,
            "width": max(measure(ln) for ln in lines), "overflow": True, "squeeze": 1.0}


# ------------------------------------------------------------------- style
def _snap(colour: np.ndarray) -> Tuple[int, ...]:
    """Lettering is printed in solid ink: pull near-black/white to the extreme."""
    lum = float(np.mean(colour))
    grey = float(np.ptp(colour)) < 30
    if lum < 128 and (grey or lum < 70):
        return tuple(0 for _ in colour)        # screentoned or dark grey lettering prints black
    if lum > 160 and grey:
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


def choose_role(block: TextBlock, area: dict, dominant: float) -> str:
    """Lettering style by where the text sits and how big it is.

    Speech in a balloon is dialogue (oversized: a shout).  Text outside
    balloons follows the usual English-manga pairing: captions and lines set
    on the art in an italic of the dialogue face, small asides in a
    handwriting face, very large text in a heavy display face.
    """
    q = block.font / max(1.0, dominant)
    if area["balloon"] and not area["boxy"]:
        return "shout" if q >= 1.5 else "dialogue"
    if q >= 1.8:
        return "title"
    if q <= 0.7 and not area["boxy"]:
        return "handwritten"
    return "narration"


# --------------------------------------------------------------- rendering
def _draw(shape, layout: dict, book: FontBook, stroke: float = 0.0) -> np.ndarray:
    """Alpha mask (float 0-1, window sized) of the laid-out text."""
    h, w = shape
    sq = float(layout.get("squeeze", 1.0))
    cw = int(np.ceil(w / sq)) if sq < 0.999 else w      # condensed: draw wide, then narrow it
    im = Image.new("L", (cw * SS, h * SS), 0)
    d = ImageDraw.Draw(im)
    face = book.face(layout["role"], layout["size"] * SS)
    sw = int(round(stroke * SS))
    anchor = "ls" if layout.get("align") == "left" else "ms"
    for line, base in zip(layout["lines"], layout["baselines"]):
        d.text((layout["x"] / sq * SS, base * SS), line, font=face, fill=255, anchor=anchor,
               stroke_width=sw, stroke_fill=255)
    alpha = np.asarray(im.reduce(SS), np.float32) / 255.0
    if cw != w:
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)
    return alpha


class Typesetter:
    def __init__(self, fonts_dir: Optional[Path] = None, log: Optional[Callable[[str], None]] = None):
        self.book = FontBook(fonts_dir, log)
        self._nobreak: frozenset = frozenset()

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
        # names are never hyphenated
        self._nobreak = frozenset(w.lower() for n in (getattr(page, "names", None) or [])
                                  for w in re.findall(r"[A-Za-z][A-Za-z'-]*", str(n.get("english", ""))))

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
            # a block of dots or marks says nothing about the lettering size
            src = b.font if re.search(r"[A-Za-z0-9]", b.translation) else max(b.font, 0.85 * dominant)
            area = find_area(clean, b, cell, 0.9 * src)
            role = override if override in ROLES else choose_role(b, area, dominant)
            look = read_style(page.work, b, area["bg"])
            caps = ROLE_CAPS[role]
            words = split_words(b.translation, caps, caps and self.book.unicase(role))
            # capitals about three quarters as tall as the original glyphs
            size_max = CAP_SHARE * src / self.book.cap(role) * float(b.style.get("scale", 1.0))
            if not caps:
                size_max *= 1.25
            size_max = float(np.clip(size_max, 6.0, 0.2 * max(H, W)))
            plans.append({"block": b, "area": area, "role": role, "look": look, "words": words,
                          "size_max": size_max, "size_min": max(5.0, 0.45 * size_max),
                          "cell": cell, "src": src, "loud": role in ("shout", "title") or src >= 1.25 * dominant})

        for p in plans:
            p["fit"] = self._fit(p, p["size_max"])
            # a cramped balloon: give up some of the margin before giving up size
            if p["area"]["balloon"] and p["fit"]["size"] < 0.8 * p["size_max"]:
                snug = dict(p, area=find_area(clean, p["block"], p["cell"], 0.9 * p["src"], margin=0.13))
                fit = self._fit(snug, p["size_max"])
                old = p["fit"]
                better = fit["size"] > old["size"] + 0.4 or (
                    fit["size"] >= old["size"] - 0.2 and fit.get("squeeze", 1.0) > old.get("squeeze", 1.0))
                if better and not fit.get("overflow"):
                    p.update(area=snug["area"], fit=fit, turn=snug["turn"], shape=snug["shape"])
        # Loud lettering filled its space in the original; let it do so again,
        # up to the footprint the Japanese had.
        for p in plans:
            if not p["loud"] or p["fit"].get("overflow") or p["fit"]["size"] < p["size_max"] - 0.3:
                continue
            b = p["block"]
            room = 0.9 * b.width * b.height
            size = p["size_max"]
            while size * 1.06 <= EMPHASIS_GROWTH * p["size_max"]:
                size *= 1.06
                fit = self._fit(p, size, size)
                if fit.get("overflow") or fit.get("squeeze", 1.0) < 1.0:
                    break
                height = (len(fit["lines"]) - 1) * fit["pitch"] + fit["cap"]
                if fit["width"] * height > room:
                    break
                p["fit"] = fit

        # ordinary dialogue is lettered at one size across the page
        peers = [p for p in plans if p["role"] == "dialogue" and not p["loud"] and not p["fit"].get("overflow")
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

    def _fit(self, p: dict, size_max: float, size_min: Optional[float] = None) -> dict:
        b, area = p["block"], p["area"]
        mask = area["mask"]
        ox, oy = area["origin"]
        p["turn"] = self._rotated(p)
        if p["turn"]:
            # a margin strip or spine: the English stays in the column the Japanese had
            pad = int(round(0.15 * b.font)) + 1
            keep = np.zeros_like(mask)
            keep[:, max(0, b.box[0] - ox - pad):max(0, b.box[2] - ox + pad)] = True
            mask = mask & keep
        ax, ay = _anchor(mask, area["origin"], b)
        left = None
        if p["turn"]:
            h, w = mask.shape
            mask = np.rot90(mask, 1)                     # lay the column on its side
            ax, ay = ay, w - 1 - ax
        elif not b.vertical and not area["balloon"]:
            left = float(b.box[0] - ox)                  # a caption keeps its left edge
        p["shape"] = mask.shape
        low = p["size_min"] if size_min is None else size_min
        words, role, nb = p["words"], p["role"], self._nobreak

        def attempt(lo, squeeze=1.0, hyphenate=False, left_x=left, one_line=False):
            return fit_text(words, self.book, role, mask, (ax, ay), size_max, lo, left_x,
                            max_lines_allowed=1 if one_line else 999, hyphenate=hyphenate,
                            squeeze=squeeze, nobreak=nb)

        fit = None
        if left is not None and b.n_lines == 1:
            # a one-line caption stays one line if a slightly smaller size does it
            fit = attempt(max(low, 0.72 * size_max), one_line=True)
        # the order a letterer works in: shrink a little, condense the letters,
        # shrink further, and only then break a word (never a name)
        for lo, squeeze, hyphenate in ((max(low, 0.7 * size_max), 1.0, False),
                                       (max(low, 0.7 * size_max), SQUEEZE, False),
                                       (low, SQUEEZE, False),
                                       (low, SQUEEZE, True),
                                       (low, SQUEEZE, "all")):   # a name, split between syllables
            if fit is None:
                fit = attempt(lo, squeeze, hyphenate)
        if fit is None and left is not None:             # no room from that edge: centre it instead
            fit = attempt(low, SQUEEZE, True, left_x=None)
        if fit is None:
            fit = _force_fit(words, self.book, role, mask.shape, (ax, ay), low)
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
                    "rotated": bool(p["turn"]), "squeeze": round(float(fit.get("squeeze", 1.0)), 2)}
