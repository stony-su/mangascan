"""Text blocks: the units that are read, translated and typeset.

The erase regions of :mod:`textmask` are built for *removing* lettering and
happily lump a whole margin of unrelated text together.  Translation needs
the opposite: one block per balloon / caption / line, with its reading
direction and glyph size.  Blocks come from the detector's own structure:

* every detected text block box is a block;
* text lines that belong to no box are clustered into blocks of their own
  (parallel neighbouring lines of similar size form a paragraph).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .textmask import MaskSettings, _cc, estimate_font


@dataclass
class TextBlock:
    id: int
    box: Tuple[int, int, int, int]      # tight box around the source lettering
    vertical: bool                      # source is written top-to-bottom
    font: float                         # glyph size of the source lettering (px)
    spans: List[Tuple[int, int]]        # extent of each line across the block, in reading order
    mask: np.ndarray                    # bool, box sized: source lettering pixels
    text: str = ""                      # what the OCR read
    confidence: float = 0.0             # how sure the OCR was (0-1)
    translation: str = ""
    style: dict = field(default_factory=dict)
    layout: Optional[dict] = None       # filled in by the typesetter
    enabled: bool = True                # draw the translation for this block
    edited: bool = False                # translation was changed by hand

    @property
    def n_lines(self) -> int:
        return max(1, len(self.spans))

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


def _runs(profile: np.ndarray, min_gap: int = 2) -> List[Tuple[int, int]]:
    """Start/end of the True runs in a 1-D profile, bridging tiny gaps."""
    idx = np.flatnonzero(profile)
    if idx.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > min_gap)
    starts = np.concatenate([[idx[0]], idx[cuts + 1]])
    ends = np.concatenate([idx[cuts], [idx[-1]]]) + 1
    return list(zip(starts.tolist(), ends.tolist()))


def measure(mask: np.ndarray, vertical: bool) -> Tuple[float, List[Tuple[int, int]]]:
    """Glyph size and the extent of every line, from the lettering's projection.

    For vertical text each column of glyphs is a hump in the profile along x;
    its width is the glyph size.  Returns ``(font, spans)`` where the spans
    are (start, end) offsets across the lines, in reading order.
    """
    profile = mask.sum(axis=0 if vertical else 1).astype(np.float64)
    if profile.max() <= 0:
        return 8.0, [(0, mask.shape[1] if vertical else mask.shape[0])]
    # ruby text and the fat fringes of neighbouring lines stay below this
    level = 0.3 * np.percentile(profile[profile > 0], 90)
    runs = _runs(profile > level, min_gap=1)
    if not runs:
        runs = _runs(profile > 0)
    widths = np.array([b - a for a, b in runs], np.float64)
    main = widths >= 0.6 * np.median(widths[widths >= 0.5 * widths.max()])
    font = float(np.median(widths[main]))
    spans = [r for r, keep in zip(runs, main) if keep]
    # a slanted or hand-drawn block has no clean valleys: one giant "line"
    rough = estimate_font(mask)
    if len(spans) == 1 and font > 2.2 * rough:
        font = rough
    if vertical:
        spans = spans[::-1]                 # columns are read right to left
    return max(4.0, font), spans


def _line_boxes(lines: np.ndarray, thresh: float) -> List[dict]:
    """Detected text-line strips, grown back to roughly the full line."""
    binary = (lines > thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 16:
            continue
        x, y, w, h = cv2.boundingRect(c)
        t = float(min(w, h))
        if t < 2:
            continue
        vertical = h >= w
        g, e = 0.55 * t, 0.5 * t            # the strip is about half as thick as the glyphs
        box = (x - g, y - e, x + w + g, y + h + e) if vertical else (x - e, y - g, x + w + e, y + h + g)
        out.append({"box": box, "vertical": vertical, "clear": max(w, h) > 1.5 * t,
                    "t": t, "length": float(max(w, h))})
    return out


def _overlap(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _area(a) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _cluster_lines(lines: List[dict]) -> List[List[dict]]:
    """Neighbouring parallel lines of similar size form one paragraph."""
    n = len(lines)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            a, b = lines[i], lines[j]
            if a["vertical"] != b["vertical"] or not 0.6 <= a["t"] / b["t"] <= 1.67:
                continue
            ax1, ay1, ax2, ay2 = a["box"]
            bx1, by1, bx2, by2 = b["box"]
            reach = 1.3 * max(a["t"], b["t"])
            if a["vertical"]:
                gap = max(ax1, bx1) - min(ax2, bx2)
                shared = min(ay2, by2) - max(ay1, by1)
                shorter = min(ay2 - ay1, by2 - by1)
            else:
                gap = max(ay1, by1) - min(ay2, by2)
                shared = min(ax2, bx2) - max(ax1, bx1)
                shorter = min(ax2 - ax1, bx2 - bx1)
            if gap <= reach and shared >= 0.3 * shorter:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    return list(groups.values())


def build_blocks(det, cfg: Optional[MaskSettings] = None) -> List[TextBlock]:
    """Turn detector output into text blocks, in reading order."""
    cfg = cfg or MaskSettings()
    seg = det.seg
    H, W = seg.shape
    n, lab, stats = _cc(seg > cfg.seg_thresh)
    if n <= 1:
        return []
    area = stats[:, cv2.CC_STAT_AREA]
    pmax = np.zeros(n, np.float32)
    flat = lab.ravel()
    sel = flat > 0
    np.maximum.at(pmax, flat[sel], seg.ravel()[sel])
    valid = (area >= cfg.min_area) & (pmax >= cfg.seed_thresh)
    valid[0] = False

    # --- candidate blocks: detector boxes, then paragraphs of leftover lines
    cands = []
    for (x1, y1, x2, y2) in det.boxes:
        pad = 0.06 * min(x2 - x1, y2 - y1) + 2
        cands.append({"box": (x1 - pad, y1 - pad, x2 + pad, y2 + pad), "lines": []})
    n_boxes = len(cands)
    loose = []
    for ln in _line_boxes(det.lines, cfg.line_thresh):
        best, best_ov = -1, 0.0
        for i in range(n_boxes):
            ov = _overlap(ln["box"], cands[i]["box"])
            if ov > best_ov:
                best, best_ov = i, ov
        if best >= 0 and best_ov >= 0.5 * _area(ln["box"]):
            cands[best]["lines"].append(ln)
        else:
            loose.append(ln)
    for group in _cluster_lines(loose):
        xs1, ys1, xs2, ys2 = zip(*[g["box"] for g in group])
        cands.append({"box": (min(xs1), min(ys1), max(xs2), max(ys2)), "lines": group})

    # --- give every lettering blob to the candidate that contains most of it
    owner = np.full(n, -1, np.int64)
    for k in np.flatnonzero(valid):
        bx, by, bw, bh = (int(v) for v in stats[k, :4])
        blob = (bx, by, bx + bw, by + bh)
        best, best_ov = -1, 0.0
        for i, c in enumerate(cands):
            ov = _overlap(blob, c["box"])
            if ov > best_ov:
                best, best_ov = i, ov
        if best >= 0 and best_ov >= 0.5 * bw * bh:
            owner[k] = best

    blocks: List[TextBlock] = []
    for i, c in enumerate(cands):
        members = np.flatnonzero(owner == i)
        if members.size == 0:
            continue
        x1 = int(stats[members, cv2.CC_STAT_LEFT].min())
        y1 = int(stats[members, cv2.CC_STAT_TOP].min())
        x2 = int((stats[members, cv2.CC_STAT_LEFT] + stats[members, cv2.CC_STAT_WIDTH]).max())
        y2 = int((stats[members, cv2.CC_STAT_TOP] + stats[members, cv2.CC_STAT_HEIGHT]).max())
        lut = np.zeros(n, bool)
        lut[members] = True
        mask = lut[lab[y1:y2, x1:x2]]
        if mask.sum() < 6:
            continue
        clear = [ln for ln in c["lines"] if ln["clear"]]
        if clear:
            v = sum(ln["length"] for ln in clear if ln["vertical"])
            h = sum(ln["length"] for ln in clear if not ln["vertical"])
            vertical = v >= h
        else:
            vertical = (y2 - y1) >= 0.8 * (x2 - x1)      # Japanese default: top to bottom
        font, spans = measure(mask, vertical)
        blocks.append(TextBlock(id=0, box=(x1, y1, x2, y2), vertical=bool(vertical),
                                font=font, spans=spans, mask=mask))

    # manga reading order: top to bottom in bands, right to left within a band
    band = max(1, H // 12)
    blocks.sort(key=lambda b: (b.box[1] // band, -b.box[2]))
    for i, b in enumerate(blocks):
        b.id = i + 1
    return blocks


def erased_fraction(block: TextBlock, flat: np.ndarray, art: np.ndarray) -> float:
    """How much of a block's lettering the (page sized) erase masks cover."""
    x1, y1, x2, y2 = block.box
    gone = (flat[y1:y2, x1:x2] | art[y1:y2, x1:x2]) & block.mask
    return float(gone.sum()) / max(1, int(block.mask.sum()))
