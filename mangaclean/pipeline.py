"""Page-level cleaning: detect lettering, build erase masks, erase.

Typical use::

    cleaner = Cleaner()
    page = cleaner.analyze(rgb)          # find the lettering (slow part #1)
    clean = cleaner.render(page)         # erase it          (slow part #2)

``analyze`` and ``render`` are separate so an interactive front end can let
the user switch regions on and off, or paint extra ones, and re-render
without running the detector again.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from .blocks import TextBlock, build_blocks
from .detect import TextDetector
from .inpaint import LamaInpainter, telea_inpaint
from .models import ensure_group, ensure_model
from .textmask import MaskSettings, Region, analyze_region, build_regions

QUALITY = {"fast": 512, "normal": 768, "high": 1280}


@dataclass
class Page:
    rgb: np.ndarray                         # original page, RGB uint8
    gray: bool                              # page carries no colour
    work: np.ndarray                        # what the analysis runs on: HxW if gray, else HxWx3
    regions: List[Region] = field(default_factory=list)
    blocks: List[TextBlock] = field(default_factory=list)   # units of text to read and translate
    protect: Optional[np.ndarray] = None    # bool HxW: pixels the user wants left alone
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]

    def masks(self, only_enabled: bool = True):
        """Page-sized (flat, art) erase masks."""
        h, w = self.rgb.shape[:2]
        flat = np.zeros((h, w), bool)
        art = np.zeros((h, w), bool)
        for r in self.regions:
            if only_enabled and not r.enabled:
                continue
            x1, y1, x2, y2 = r.box
            flat[y1:y2, x1:x2] |= r.flat
            art[y1:y2, x1:x2] |= r.art
        if self.protect is not None:
            flat &= ~self.protect
            art &= ~self.protect
        return flat, art


def is_grayscale(rgb: np.ndarray, tol: int = 10) -> bool:
    """True if the page has (practically) no colour, e.g. a B/W page saved as RGB."""
    step = max(1, int(max(rgb.shape[:2]) // 1500))
    s = rgb[::step, ::step].astype(np.int16)
    chroma = s.max(axis=2) - s.min(axis=2)
    return bool(np.percentile(chroma, 99.9) <= tol)


class Cleaner:
    def __init__(self, models_dir: Optional[Path] = None, threads: int = 0,
                 quality: str = "normal", settings: Optional[MaskSettings] = None,
                 inpaint: bool = True, log: Optional[Callable[[str], None]] = None):
        self.models_dir = models_dir
        self.threads = threads
        self.max_side = QUALITY.get(quality, QUALITY["normal"])
        self.settings = settings or MaskSettings()
        self.use_lama = inpaint
        self.log = log or (lambda s: None)
        self._detector: Optional[TextDetector] = None
        self._inpainter: Optional[LamaInpainter] = None
        self._ocr = None
        self._translators: Dict[str, object] = {}
        self._typesetter = None

    # -------------------------------------------------------------- models
    @property
    def detector(self) -> TextDetector:
        if self._detector is None:
            path = ensure_model("detector", self.models_dir, self.log)
            self._detector = TextDetector(str(path), threads=self.threads)
        return self._detector

    @property
    def inpainter(self) -> LamaInpainter:
        if self._inpainter is None:
            path = ensure_model("inpainter", self.models_dir, self.log)
            self._inpainter = LamaInpainter(str(path), threads=self.threads, max_side=self.max_side)
        return self._inpainter

    @property
    def ocr(self):
        if self._ocr is None:
            from .ocr import MangaOcr
            self._ocr = MangaOcr(ensure_group("ocr", self.models_dir, self.log), threads=self.threads)
        return self._ocr

    def translator(self, engine: str = "offline"):
        if engine not in self._translators:
            if engine == "claude":
                from .translate import ClaudeTranslator
                self._translators[engine] = ClaudeTranslator()
            elif engine == "offline":
                from .translate import OfflineTranslator
                self._translators[engine] = OfflineTranslator(
                    ensure_group("mt", self.models_dir, self.log), threads=self.threads)
            else:
                raise ValueError(f"unknown translation engine: {engine}")
        return self._translators[engine]

    @property
    def typesetter(self):
        if self._typesetter is None:
            from .typeset import Typesetter
            self._typesetter = Typesetter(log=self.log)
        return self._typesetter

    # ------------------------------------------------------------ analysis
    def analyze(self, rgb: np.ndarray) -> Page:
        """Find the lettering on a page (RGB uint8) and plan how to erase it."""
        rgb = np.ascontiguousarray(rgb[..., :3])
        gray = is_grayscale(rgb)
        work = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray else rgb
        page = Page(rgb=rgb, gray=gray, work=work)
        t = time.perf_counter()
        det = self.detector.detect(rgb)
        page.timings["detect"] = time.perf_counter() - t
        t = time.perf_counter()
        page.regions = build_regions(work, det, self.settings)
        page.blocks = build_blocks(det, self.settings)
        page.timings["regions"] = time.perf_counter() - t
        return page

    def add_manual(self, page: Page, mask: np.ndarray, mode: str = "auto") -> List[Region]:
        """Add hand-painted areas (bool HxW) as extra regions.

        ``mode``: 'auto' picks flat fill or inpainting per blob like the
        automatic regions; 'art' always inpaints exactly what was painted.
        """
        h, w = page.rgb.shape[:2]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        added: List[Region] = []
        next_id = max([r.id for r in page.regions], default=0) + 1
        for k in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[k])
            if area < 4:
                continue
            pad = int(max(12, 0.5 * min(bw, bh)))
            box = (max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad))
            m = lab[box[1]:box[3], box[0]:box[2]] == k
            if mode == "art":
                parts = [{"box": box, "kind": "art", "bg": (255,), "flat": np.zeros_like(m), "art": m}]
            else:
                parts = analyze_region(page.work, None, m, box, None, self.settings)
            for p in parts:
                both = p["flat"] | p["art"]
                if not both.any():
                    continue
                ys, xs = np.nonzero(both)
                px, py = p["box"][0], p["box"][1]
                tight = (px + int(xs.min()), py + int(ys.min()), px + int(xs.max()) + 1, py + int(ys.max()) + 1)
                reg = Region(id=next_id, box=p["box"], bbox=tight, confident=True, score=1.0,
                             font=0.0, kind=p["kind"], bg=p["bg"], flat=p["flat"], art=p["art"],
                             enabled=True, source="manual")
                next_id += 1
                page.regions.append(reg)
                added.append(reg)
        return added

    # ----------------------------------------------------------- rendering
    def render(self, page: Page, progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Erase all enabled regions. Returns the cleaned page as RGB uint8."""
        t = time.perf_counter()
        out = page.work.copy()
        protect = page.protect
        for r in page.regions:
            if not r.enabled or not r.flat.any():
                continue
            x1, y1, x2, y2 = r.box
            m = r.flat if protect is None else r.flat & ~protect[y1:y2, x1:x2]
            sub = out[y1:y2, x1:x2]
            if out.ndim == 2:
                sub[m] = r.bg[0]
            else:
                sub[m] = r.bg if len(r.bg) == 3 else (r.bg[0],) * 3
        page.timings["fill"] = time.perf_counter() - t

        t = time.perf_counter()
        _, art = page.masks()
        if art.any():
            rgb = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB) if out.ndim == 2 else out
            if self.use_lama:
                rgb = self.inpainter.inpaint(rgb, art, progress)
            else:
                rgb = telea_inpaint(rgb, art)
            out = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if out.ndim == 2 else rgb
        page.timings["inpaint"] = time.perf_counter() - t
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB) if out.ndim == 2 else out

    # --------------------------------------------------------- translation
    def read(self, page: Page, progress: Optional[Callable[[int, int], None]] = None) -> None:
        """OCR every text block that has not been read yet."""
        t = time.perf_counter()
        todo = [b for b in page.blocks if not b.text and not b.edited]
        for i, b in enumerate(todo):
            b.text, b.confidence = self.ocr.read_block(page.rgb, b)
            if progress:
                progress(i + 1, len(todo))
        page.timings["ocr"] = time.perf_counter() - t

    def translate(self, page: Page, engine: str = "offline",
                  progress: Optional[Callable[[int, int], None]] = None) -> None:
        """Read the page's text blocks and translate them (hand-edited blocks are kept)."""
        self.read(page)
        t = time.perf_counter()
        todo = [b for b in page.blocks if not b.edited]
        items = [{"id": b.id, "text": b.text, "confidence": b.confidence, "box": b.box,
                  "vertical": b.vertical} for b in todo]
        results = {r["id"]: r for r in self.translator(engine).translate_page(items, page.rgb, progress)}
        for b in todo:
            r = results.get(b.id)
            if r is not None:
                b.text, b.translation = r["text"], r["translation"]
        page.timings["translate"] = time.perf_counter() - t

    def typeset(self, page: Page, clean_rgb: np.ndarray) -> np.ndarray:
        """Letter the translations onto the cleaned page."""
        t = time.perf_counter()
        out = self.typesetter.typeset(page, clean_rgb)
        page.timings["typeset"] = time.perf_counter() - t
        return out

    def clean(self, rgb: np.ndarray, include_uncertain: bool = False):
        """One-shot convenience: returns ``(cleaned_rgb, page)``."""
        page = self.analyze(rgb)
        if include_uncertain:
            for r in page.regions:
                r.enabled = True
        return self.render(page), page


# -------------------------------------------------------------- debug view
def overlay(page: Page, show_disabled: bool = True) -> np.ndarray:
    """Original page with the erase masks tinted: green = flat fill,
    red = inpaint, orange = found but switched off."""
    vis = page.rgb.astype(np.float32).copy()
    for r in page.regions:
        if not r.enabled and not show_disabled:
            continue
        x1, y1, x2, y2 = r.box
        sub = vis[y1:y2, x1:x2]
        if r.enabled:
            sub[r.flat] = sub[r.flat] * 0.4 + np.array([0, 190, 60], np.float32) * 0.6
            sub[r.art] = sub[r.art] * 0.4 + np.array([235, 40, 40], np.float32) * 0.6
        else:
            both = r.flat | r.art
            sub[both] = sub[both] * 0.5 + np.array([255, 150, 0], np.float32) * 0.5
    vis = vis.astype(np.uint8)
    for r in page.regions:
        if not r.enabled and not show_disabled:
            continue
        x1, y1, x2, y2 = r.bbox
        col = (0, 90, 255) if r.enabled else (200, 90, 0)
        cv2.rectangle(vis, (x1 - 2, y1 - 2), (x2 + 1, y2 + 1), col, 1)
        cv2.putText(vis, str(r.id), (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return vis
