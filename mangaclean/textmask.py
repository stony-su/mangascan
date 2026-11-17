"""Turn raw detector output into per-region erase masks.

The detector gives a per-pixel text probability map.  This module

1. splits it into connected blobs and decides which of them are *confident*
   (backed up by a detected text block or text line) and which are merely
   *uncertain* (typically fragments of stylised logos / sound effects);
2. groups the blobs into regions (a balloon's worth of text, a caption ...);
3. for every region works out **how** it should be erased:

   ``flat``  the lettering sits on a uniform background (a speech balloon, a
             caption box, the page margin).  These pixels are simply painted
             with the background colour, which gives a perfectly clean result
             and cannot smear the balloon outline.
   ``art``   the lettering overlaps artwork and has to be reconstructed by
             the inpainting network.

All geometry is relative to the estimated font size of the region so the
same settings work for 800 px web rips and 3000 px scans alike.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MaskSettings:
    seg_thresh: float = 0.30      # text probability that counts as lettering
    seed_thresh: float = 0.50     # a blob needs at least one pixel this confident
    line_thresh: float = 0.30     # text-line probability that counts as a line
    min_area: int = 3             # ignore blobs smaller than this (pixels)
    flat_tol: int = 28            # max deviation from the background colour (0-255)
    ink_tol: int = 64             # max deviation from the lettering colour (0-255)
    flat_uniform: float = 0.60    # share of background pixels that must share one colour
    cover_flat: float = 0.97      # share of lettering that must be enclosed -> flat
    cover_mixed: float = 0.50     # below this the whole region is treated as art
    grow: float = 1.0             # multiplier for all mask dilations
    debug: Optional[list] = None  # collects per-region diagnostics when set


@dataclass
class Region:
    id: int
    box: Tuple[int, int, int, int]      # analysis window on the page (x1, y1, x2, y2)
    bbox: Tuple[int, int, int, int]     # tight box around the pixels that get erased
    confident: bool
    score: float                        # mean text probability of the lettering
    font: float                         # estimated glyph size in pixels
    kind: str                           # 'flat' | 'art' | 'mixed'
    bg: Tuple[int, ...]                 # fill colour for the flat part
    flat: np.ndarray                    # bool, window sized: paint with ``bg``
    art: np.ndarray                     # bool, window sized: inpaint
    enabled: bool = True
    source: str = "auto"                # 'auto' | 'manual'

    @property
    def area(self) -> int:
        return int(self.flat.sum() + self.art.sum())


# ----------------------------------------------------------------- helpers
def to_gray(work: np.ndarray) -> np.ndarray:
    return work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)


def _cc(binary: np.ndarray, connectivity: int = 8):
    return cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=connectivity)[:3]


def _dist_to(mask: np.ndarray) -> np.ndarray:
    """Distance of every pixel to the nearest True pixel of ``mask``."""
    if not mask.any():
        return np.full(mask.shape, 1e6, np.float32)
    return cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _disk(radius)).astype(bool)


def _cdist(win: np.ndarray, colour) -> np.ndarray:
    """Per-pixel distance (max over channels) to ``colour``; int16."""
    if win.ndim == 2:
        return np.abs(win.astype(np.int16) - int(np.ravel(colour)[0]))
    return np.abs(win.astype(np.int16) - np.asarray(colour, np.int16)).max(axis=2)


def _median_colour(win: np.ndarray, sel: np.ndarray) -> np.ndarray:
    px = win[sel]
    if px.size == 0:
        return np.array([255] * (1 if win.ndim == 2 else win.shape[2]), np.int16)
    if win.ndim == 2:
        return np.array([int(np.median(px))], np.int16)
    return np.median(px, axis=0).astype(np.int16)


def _dominant_colour(win: np.ndarray, sel: np.ndarray, tol: int) -> np.ndarray:
    """Most common colour among the selected pixels (robust to a second,
    differently coloured population - unlike a plain median)."""
    gray = to_gray(win)[sel]
    if gray.size == 0:
        return np.array([255] * (1 if win.ndim == 2 else win.shape[2]), np.int16)
    hist = np.bincount(gray, minlength=256).astype(np.float64)
    smooth = np.convolve(np.pad(hist, 6), np.ones(13), mode="valid")
    mode = int(np.argmax(smooth))
    px = win[sel][np.abs(gray.astype(np.int16) - mode) <= tol]
    if win.ndim == 2:
        return np.array([int(np.median(px))], np.int16)
    return np.median(px, axis=0).astype(np.int16)


def _mode_colour(win: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Most frequent value per channel among the selected pixels."""
    px = win[sel].reshape(int(sel.sum()), -1)
    out = []
    for c in range(px.shape[1]):
        hist = np.bincount(px[:, c], minlength=256).astype(np.float64)
        smooth = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
        smooth[0] = hist[0] + 0.5 * hist[1]
        smooth[255] = hist[255] + 0.5 * hist[254]
        out.append(int(np.argmax(smooth)))
    return np.array(out, np.int16)


def _otsu(values: np.ndarray) -> float:
    if values.size < 2 or values.min() == values.max():
        return float(values.mean()) if values.size else 127.0
    t, _ = cv2.threshold(values.reshape(-1, 1).astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    v, w = values[order], weights[order].astype(np.float64)
    c = np.cumsum(w)
    return float(v[min(len(v) - 1, np.searchsorted(c, c[-1] / 2.0))])


def _roughness(mask: np.ndarray) -> float:
    """Outline length of ``mask`` relative to the outline of its convex hull."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) >= 3]
    if not cnts:
        return 1.0
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    hp = cv2.arcLength(cv2.convexHull(np.concatenate(cnts)), True)
    return float(perim / hp) if hp > 0 else 1.0


def estimate_font(mask: np.ndarray, limit: float = 1e9) -> float:
    """Rough glyph size (pixels) of the lettering in ``mask``."""
    n, _, st = _cc(mask)
    if n <= 1:
        return 8.0
    size = np.minimum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]).astype(np.float64)
    return float(np.clip(_weighted_median(size, st[1:, cv2.CC_STAT_AREA]) * 1.15, 3.0, limit))


# ------------------------------------------------------- region analysis
def _art_mask(win: np.ndarray, seg_w: Optional[np.ndarray], seed: np.ndarray, font: float,
              cfg: MaskSettings) -> np.ndarray:
    """Erase mask for lettering that sits on artwork.

    Starts from the detected text pixels, completes strokes the network only
    partly covered, swallows a contrasting outline ("halo") around the glyphs
    if there is one, and pads the result a little.
    """
    if not seed.any():
        return np.zeros_like(seed)
    core = seed & (seg_w > 0.6) if seg_w is not None else seed
    if core.sum() < 10:
        core = seed
    tcol = _median_colour(win, core)

    # 1. complete strokes: same-coloured pixels attached to the detected ones
    ds = _dist_to(seed)
    ink = (_cdist(win, tcol) <= cfg.ink_tol) & (ds <= max(2.0, 0.25 * font))
    n, lab, _ = _cc(ink)
    lut = np.zeros(n, bool)
    lut[np.unique(lab[seed & ink])] = True
    lut[0] = False
    base = seed | lut[lab]

    # 1b. whole glyphs the detector skipped (long-vowel bars, small kana,
    #     punctuation): lettering-coloured shapes of glyph size that sit
    #     that sit in a gap *between* detected glyphs of the same column or
    #     row.  Requiring lettering on both sides keeps this from reaching
    #     into the artwork beside or beyond a line; hatching lines are far
    #     longer than a glyph and screentone dots far smaller.
    allink = _cdist(win, tcol) <= cfg.ink_tol
    n, lab, st = _cc(allink)
    if n > 1:
        area = st[:, cv2.CC_STAT_AREA]
        side = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
        sized = (side >= 0.25 * font) & (side <= 1.3 * font)
        sized[0] = False
        if sized.any():
            reach = max(3, int(round(2.2 * font)))
            fwd = np.zeros(2 * reach + 1, np.uint8)
            fwd[reach:] = 1                              # looks towards +axis
            back = fwd[::-1].copy()                      # looks towards -axis
            for _ in range(3):                           # a found glyph can bridge to the next
                b8 = base.astype(np.uint8)
                between = ((cv2.dilate(b8, fwd.reshape(-1, 1)) & cv2.dilate(b8, back.reshape(-1, 1)))
                           | (cv2.dilate(b8, fwd.reshape(1, -1)) & cv2.dilate(b8, back.reshape(1, -1))))
                hit = np.bincount(lab[between.astype(bool)], minlength=n)
                known = np.bincount(lab[base], minlength=n)
                glyph = sized & (hit >= 0.7 * area) & (known < 0.5 * area)
                if not glyph.any():
                    break
                base |= glyph[lab]

    # 2. outline / halo in a contrasting colour hugging the glyphs
    db = _dist_to(base)
    adj = (db > 0) & (db <= 2.5)
    if adj.sum() >= 12:
        hcol = _median_colour(win, adj)
        dh = _cdist(win, hcol)
        uniform = float((dh[adj] <= cfg.flat_tol).mean())
        contrast = int(np.abs(hcol - tcol).max())
        if uniform >= 0.6 and contrast >= 70:
            cand = (dh <= cfg.flat_tol) & (db <= max(3.0, 0.4 * font)) & ~base
            n, lab, _ = _cc(cand, 4)
            lut = np.zeros(n, bool)
            lut[np.unique(lab[cand & (db <= 1.5)])] = True
            lut[0] = False
            base |= lut[lab]

    # 3. pad for anti-aliasing / compression ringing and close the tiny gaps
    #    between strokes, which the inpainter would otherwise treat as known
    r = int(np.clip(round(0.10 * font * cfg.grow), 2, 10)) + 1
    art = _dilate(base, r)
    k = max(1, int(round(0.12 * font)))
    return cv2.morphologyEx(art.astype(np.uint8), cv2.MORPH_CLOSE, _disk(k)).astype(bool)


def _is_inverted(gw: np.ndarray, seg_w: Optional[np.ndarray], mask: np.ndarray,
                 dist: np.ndarray) -> bool:
    """True if the lettering is light on a dark background.

    The detector is most certain on the strokes themselves and less so on the
    fringe of each blob, so comparing the brightness of the two tells which
    side the lettering is on.  Without a probability map (hand-drawn regions)
    the pixels just outside the mask are taken as background instead.
    """
    if seg_w is not None:
        core = mask & (seg_w > 0.9)
        rim = mask & (seg_w <= 0.6)
        if core.sum() >= 4 and rim.sum() >= 4:
            return float(gw[core].mean()) - float(gw[rim].mean()) > 50.0
        return False
    ring = (dist > 0) & (dist <= 3.0)
    if ring.sum() < 8 or mask.sum() < 8:
        return False
    t = _otsu(gw[mask | ring])
    # the lettering is on whichever side of the split the surroundings are not
    return float(np.median(gw[ring])) <= t and float((gw[mask] > t).mean()) > 0.05


def _part(box, kind, bg, flat, art) -> dict:
    return {"box": box, "kind": kind, "bg": tuple(int(v) for v in bg), "flat": flat, "art": art}


def _paste(src: np.ndarray, src_box, dst_box) -> np.ndarray:
    """Copy the overlapping part of a window-sized mask into another window."""
    out = np.zeros((dst_box[3] - dst_box[1], dst_box[2] - dst_box[0]), src.dtype)
    ox1, oy1 = max(src_box[0], dst_box[0]), max(src_box[1], dst_box[1])
    ox2, oy2 = min(src_box[2], dst_box[2]), min(src_box[3], dst_box[3])
    if ox2 > ox1 and oy2 > oy1:
        piece = src[oy1 - src_box[1]:oy2 - src_box[1], ox1 - src_box[0]:ox2 - src_box[0]]
        out[oy1 - dst_box[1]:oy2 - dst_box[1], ox1 - dst_box[0]:ox2 - dst_box[0]] = piece
    return out


def analyze_region(work: np.ndarray, seg: Optional[np.ndarray], mask: np.ndarray,
                   box: Tuple[int, int, int, int], font: Optional[float], cfg: MaskSettings,
                   inverted: Optional[bool] = None, depth: int = 0) -> List[dict]:
    """Decide how the lettering ``mask`` inside window ``box`` gets erased.

    Returns a list of parts ``{box, kind, bg, flat, art}`` with the masks in
    the coordinates of each part's own ``box``.  Usually that is one part.  A
    label box that carries lettering of the opposite polarity (white digits
    in a black box among black-on-white text) becomes a part of its own, so
    the box survives and only its lettering is filled in.
    """
    x1, y1, x2, y2 = box
    H, W = work.shape[:2]
    win = work[y1:y2, x1:x2]
    gw = to_gray(win)
    seg_w = seg[y1:y2, x1:x2] if seg is not None else None
    empty = np.zeros(mask.shape, bool)
    total = int(mask.sum())
    if total == 0:
        return []
    if font is None:
        font = estimate_font(mask, 0.10 * max(H, W))

    r = int(np.clip(round(0.10 * font * cfg.grow), 2, 10))
    dist = _dist_to(mask)
    info = {"box": box, "font": round(font, 1), "depth": depth}
    if cfg.debug is not None:
        cfg.debug.append(info)

    def as_art(why, bg=(255,)):
        info["why"] = why
        return [_part(box, "art", bg, empty, _art_mask(win, seg_w, mask, font, cfg))]

    # --- split the pixels on and right around the lettering into "ink" and
    #     "paper"; which is which follows from the polarity of the lettering
    hood = dist <= max(3.0, 0.12 * font)
    if inverted is None:
        inverted = _is_inverted(gw, seg_w, mask, dist)
    t = _otsu(gw[hood])
    paper = hood & ((gw <= t) if inverted else (gw > t))
    info.update(inverted=inverted, otsu=t)
    if paper.sum() < 16:
        return as_art("no background pixels")
    bg = _dominant_colour(win, paper, cfg.flat_tol)
    dbg = _cdist(win, bg)
    same = dbg <= cfg.flat_tol
    uniform = float(same[paper].mean())
    info.update(bg=tuple(int(v) for v in bg), uniform=round(uniform, 2))
    if uniform < cfg.flat_uniform:
        return as_art("background not uniform", bg)

    # lettering pixels the detector is sure about (the fringe of a blob is
    # often just background that happens to be next to a stroke)
    ink = mask & ~same & ((gw > t) if inverted else (gw <= t))
    strong = ink & (seg_w > 0.6) if seg_w is not None else ink
    n_strong = int(strong.sum())

    # --- the connected background area around the lettering ("interior").
    #     Everything it completely encloses is a "hole".  Background-coloured
    #     pockets *inside* a hole (the counter of a glyph, a white patch in a
    #     black label) belong to that hole, not to the interior.
    n1, lab1, st1 = _cc(same, 4)
    cnt = np.bincount(lab1[hood & same], minlength=n1)
    cnt[0] = 0
    cands = [c for c in np.argsort(-cnt) if cnt[c] >= max(6, 0.03 * cnt.sum())]
    if not cands:
        return as_art("no background area", bg)

    def holes_of(inside):
        nn, ll, ss = _cc(~inside, 8)
        edge_ids = np.unique(np.concatenate([ll[0], ll[-1], ll[:, 0], ll[:, -1]]))
        hole = np.ones(nn, bool)
        hole[edge_ids] = False
        hole[0] = False
        return nn, ll, ss, hole

    interior = lab1 == cands[0]
    n2, lab2, st2, is_hole = holes_of(interior)
    for c in cands[1:]:
        cx, cy = int(st1[c, cv2.CC_STAT_LEFT]), int(st1[c, cv2.CC_STAT_TOP])
        row = lab1[cy, cx:cx + int(st1[c, cv2.CC_STAT_WIDTH])]
        px = cx + int(np.argmax(row == c))          # a pixel that belongs to c
        if is_hole[lab2[cy, px]]:
            continue                                # nested inside a hole
        interior |= lab1 == c
        n2, lab2, st2, is_hole = holes_of(interior)
    non = ~interior
    filled = interior | is_hole[lab2]
    cover = float((strong & filled).sum()) / n_strong if n_strong else 1.0
    info.update(cover=round(cover, 3))
    if cover < cfg.cover_mixed:
        return as_art("lettering not enclosed by background", bg)

    area2 = st2[:, cv2.CC_STAT_AREA]
    has_text = np.bincount(lab2[mask & non], minlength=n2) > 0
    near = (dist <= max(r + 1.0, 0.25 * font)) & non
    has_near = np.bincount(lab2[near], minlength=n2) > 0
    side = np.maximum(st2[:, cv2.CC_STAT_WIDTH], st2[:, cv2.CC_STAT_HEIGHT])
    small = side <= max(4.0, 0.8 * font)

    # --- a screentone / textured background is "uniform" between its dots,
    #     but erasing it flat would punch a white ghost into the pattern
    dmax = np.zeros(n2, np.int16)
    np.maximum.at(dmax, lab2[non], dbg[non])
    solid = (area2 >= max(3.0, (0.08 * font) ** 2)) & (dmax >= 60)
    islands = int((is_hole & ~has_text & has_near & solid).sum())
    info.update(islands=islands)
    if islands > max(10, 0.5 * int((is_hole & has_text).sum())):
        return as_art("textured background", bg)

    # --- a thick outline around lettering on artwork also looks "flat" up
    #     close.  Unlike a balloon it leaves no room around the lettering and
    #     its edge is as ragged as the glyphs themselves.
    band = (dist > 0.3 * font) & (dist <= 0.6 * font)
    room = float(interior[band].mean()) if band.any() else 1.0
    local = filled & (dist <= max(3.0, 0.5 * font))
    rough = _roughness(local) if room < 0.5 else 1.0
    info.update(room=round(room, 2), rough=round(rough, 2))
    if room < 0.5 and rough > 1.25:
        info["why"] = "outlined lettering on artwork"
        art = _dilate(local, 2) | _art_mask(win, seg_w, mask, font, cfg)
        return [_part(box, "art", bg, empty, art)]

    # --- what is each lettering-bearing hole?
    #     glyph      its ink is (almost) all detected lettering  -> erase it
    #     container  a shape that merely *carries* lettering of the opposite
    #                polarity (white digits in a black label)    -> keep the
    #                shape, clean the lettering inside it on its own
    #     object     something drawn that happens to have lettering on it
    #                -> leave it alone, inpaint just the lettering
    #     In a container the pixels the detector is surest about are on the
    #     *paper* side of this region while its margins are not lettering.
    parts: List[dict] = []
    keep = np.zeros(n2, bool)                  # holes that must not be wiped
    art_seed = np.zeros(mask.shape, bool)
    off_bg = non & ~same
    n_ink = np.bincount(lab2[off_bg], minlength=n2)
    n_ink_det = np.bincount(lab2[off_bg & _dilate(mask, 1)], minlength=n2)
    detected = n_ink_det / np.maximum(n_ink, 1)
    flipped = np.zeros(n2, bool)
    if seg_w is not None:
        g64 = gw.astype(np.float64)
        core = non & mask & (seg_w > 0.9)
        rim = non & (seg_w <= 0.6)
        nc = np.bincount(lab2[core], minlength=n2)
        nr = np.bincount(lab2[rim], minlength=n2)
        lc = np.bincount(lab2[core], weights=g64[core], minlength=n2) / np.maximum(nc, 1)
        lr = np.bincount(lab2[rim], weights=g64[rim], minlength=n2) / np.maximum(nr, 1)
        diff = (lr - lc) if inverted else (lc - lr)
        flipped = (nc >= 8) & (nr >= np.maximum(12, 0.1 * area2)) & (diff > 50.0)
    for h in np.flatnonzero(is_hole & has_text & (flipped | (detected < 0.7))):
        sub_w = mask & (lab2 == h)
        if flipped[h] and depth < 2:
            hx, hy = int(st2[h, cv2.CC_STAT_LEFT]), int(st2[h, cv2.CC_STAT_TOP])
            hw, hh = int(st2[h, cv2.CC_STAT_WIDTH]), int(st2[h, cv2.CC_STAT_HEIGHT])
            sub_font = estimate_font(sub_w, 0.10 * max(H, W))
            pad = int(max(12, round(1.2 * sub_font)))
            sb = (max(0, x1 + hx - pad), max(0, y1 + hy - pad),
                  min(W, x1 + hx + hw + pad), min(H, y1 + hy + hh + pad))
            sub = analyze_region(work, seg, _paste(sub_w, box, sb), sb, sub_font, cfg,
                                 not inverted, depth + 1)
            if sub and all(p["kind"] == "flat" for p in sub):
                keep[h] = True
                parts.extend(sub)
                continue
            if detected[h] >= 0.5:
                continue                        # a label that is all lettering: wipe it whole
        elif detected[h] >= 0.7:
            continue
        keep[h] = True
        art_seed |= sub_w
    info.update(kept=int(keep.sum()))

    # --- holes to erase: the lettering itself plus small islands right next
    #     to it (ruby text, dots, dakuten the detector may have skipped)
    erase_holes = (is_hole & ~keep & (has_text | (has_near & small)))[lab2]
    erasable = interior | erase_holes
    seedm = (mask & ~keep[lab2]) | erase_holes
    flat = _dilate(seedm, r) & erasable
    # wipe compression ringing a little further out (it never reaches beyond
    # one 8 px coding block), but stay off outlines and enclosed shapes
    r_far = int(round(min(max(r + 2, 0.35 * font * cfg.grow), max(r + 2, 9, 0.2 * font))))
    flat |= _dilate(seedm, r_far) & interior & ~_dilate(~erasable, 2)
    # fill with the paper colour found a little away from the lettering,
    # where compression ringing has not dirtied it (so the patch is invisible
    # even after a levels adjustment)
    clean = interior & (dist > r_far) & (dist <= r_far + max(8.0, 0.5 * font))
    if clean.sum() >= 30:
        bg = _mode_colour(win, clean)

    # --- lettering that is *not* enclosed (touches the outline / leaves the balloon)
    left = strong & ~filled
    if left.any():
        n3, lab3, st3 = _cc(left)
        big = st3[:, cv2.CC_STAT_AREA] >= max(6, 0.02 * max(1, n_strong))
        big[0] = False
        left = big[lab3]
    if cover >= cfg.cover_flat and left.sum() < max(12, 0.03 * n_strong):
        left = empty
    left = left | art_seed
    art = _art_mask(win, seg_w, left, font, cfg) & ~flat if left.any() else empty
    info["why"] = "enclosed by flat background"
    if flat.any() or art.any():
        parts.insert(0, _part(box, "mixed" if art.any() else "flat", bg, flat, art))
    return parts


# ------------------------------------------------------------ page level
def _line_polys(lines: np.ndarray, cfg: MaskSettings) -> List[np.ndarray]:
    """Text-line strips from the line map, grown back to roughly the full line."""
    binary = (lines > cfg.line_thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < 16:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(c)
        t = min(w, h)
        if t < 2:
            continue
        ext = min(1.5 * t, 0.5 * max(w, h))        # reach past both ends of the line
        size = (w + 2 * ext, h + 2 * t) if w >= h else (w + 2 * t, h + 2 * ext)
        pts = cv2.boxPoints(((cx, cy), size, ang))
        polys.append(np.round(pts).astype(np.int32))
    return polys


def build_regions(work: np.ndarray, det, cfg: Optional[MaskSettings] = None) -> List[Region]:
    """Group detected lettering into regions and compute their erase masks."""
    cfg = cfg or MaskSettings()
    H, W = work.shape[:2]
    seg = det.seg
    n, lab, stats = _cc(seg > cfg.seg_thresh)
    if n <= 1:
        return []

    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    bx, by = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    bw, bh = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    flat_lab = lab.ravel()
    textpix = flat_lab > 0
    psum = np.bincount(flat_lab, weights=seg.ravel(), minlength=n)
    pmax = np.zeros(n, np.float32)
    np.maximum.at(pmax, flat_lab[textpix], seg.ravel()[textpix])
    valid = (area >= cfg.min_area) & (pmax >= cfg.seed_thresh)
    valid[0] = False
    if not valid.any():
        return []

    # --- where the block / line detectors agree there is text
    zone = np.zeros((H, W), np.uint8)
    shapes = []
    for (x1, y1, x2, y2) in det.boxes:
        pad = 0.06 * min(x2 - x1, y2 - y1) + 2
        p1 = (int(max(0, x1 - pad)), int(max(0, y1 - pad)))
        p2 = (int(min(W - 1, x2 + pad)), int(min(H - 1, y2 + pad)))
        shapes.append(("rect", p1, p2))
        cv2.rectangle(zone, p1, p2, 1, -1)
    for poly in _line_polys(det.lines, cfg):
        shapes.append(("poly", poly))
        cv2.fillPoly(zone, [poly], 1)
    zsum = np.bincount(flat_lab, weights=zone.ravel(), minlength=n)
    confident = (zsum / np.maximum(area, 1)) >= 0.5

    # one representative pixel per blob, used to look its group up
    _, first = np.unique(flat_lab, return_index=True)
    rep_y, rep_x = np.divmod(first, W)

    size = np.minimum(bw, bh).astype(np.float64)
    max_font = 0.10 * max(H, W)
    font = np.clip(size * 1.15, 3.0, max_font)
    group = np.zeros(n, np.int64)

    def assign(sel: np.ndarray, with_zone: bool, offset: int) -> int:
        canvas = np.zeros((H, W), np.uint8)
        if with_zone:
            for s in shapes:
                if s[0] == "rect":
                    cv2.rectangle(canvas, s[1], s[2], 1, -1)
                else:
                    cv2.fillPoly(canvas, [s[1]], 1)
        idx = np.flatnonzero(sel)
        for i in idx:
            e = int(round(0.35 * font[i]))
            cv2.rectangle(canvas, (int(bx[i] - e), int(by[i] - e)),
                          (int(bx[i] + bw[i] - 1 + e), int(by[i] + bh[i] - 1 + e)), 1, -1)
        gn, glab = cv2.connectedComponents(canvas, connectivity=8)
        group[idx] = glab[rep_y[idx], rep_x[idx]] + offset
        return offset + gn

    for _ in range(2):          # second pass uses the font size of the whole group
        off = assign(valid & confident, True, 0)
        assign(valid & ~confident, False, off)
        for g in np.unique(group[valid]):
            m = valid & (group == g)
            font[m] = np.clip(_weighted_median(size[m], area[m]) * 1.15, 3.0, max_font)

    regions: List[Region] = []
    for g in np.unique(group[valid]):
        members = np.flatnonzero(valid & (group == g))
        f = float(font[members[0]])
        tx1, ty1 = int(bx[members].min()), int(by[members].min())
        tx2 = int((bx[members] + bw[members]).max())
        ty2 = int((by[members] + bh[members]).max())
        pad = int(max(12, round(1.2 * f)))
        box = (max(0, tx1 - pad), max(0, ty1 - pad), min(W, tx2 + pad), min(H, ty2 + pad))
        lut = np.zeros(n, bool)
        lut[members] = True
        mask = lut[lab[box[1]:box[3], box[0]:box[2]]]
        conf = bool(confident[members[0]])
        score = float(psum[members].sum() / area[members].sum())
        for p in analyze_region(work, seg, mask, box, f, cfg):
            both = p["flat"] | p["art"]
            if not both.any():
                continue
            ys, xs = np.nonzero(both)
            px, py = p["box"][0], p["box"][1]
            tight = (px + int(xs.min()), py + int(ys.min()),
                     px + int(xs.max()) + 1, py + int(ys.max()) + 1)
            regions.append(Region(
                id=0, box=p["box"], bbox=tight, confident=conf, score=score, font=f,
                kind=p["kind"], bg=p["bg"], flat=p["flat"], art=p["art"], enabled=conf))

    regions.sort(key=lambda r: (r.bbox[1] // max(1, H // 12), -r.bbox[2]))
    for i, r in enumerate(regions):
        r.id = i + 1
    return regions
