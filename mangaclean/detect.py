"""Text detection for manga pages.

Wraps the *comic-text-detector* network (ONNX).  For one 1024x1024 input the
network produces three things:

``blk``  YOLO-style text *block* boxes (one per paragraph / balloon of text)
``seg``  a per-pixel text probability map (the thing we mostly care about)
``det``  a DBNet text *line* probability map

Pages are letterboxed to the network size.  Very large or very elongated
pages (webtoon strips) are cut into overlapping tiles so small lettering is
not shrunk into oblivion, and the tile results are stitched back together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

NET_SIZE = 1024


@dataclass
class Detection:
    """Raw detector output, mapped back to page coordinates."""

    seg: np.ndarray                 # float32 HxW in [0, 1]: text-pixel probability
    lines: np.ndarray               # float32 HxW in [0, 1]: text-line probability
    boxes: np.ndarray               # float32 Nx4 (x1, y1, x2, y2) text block boxes
    scores: np.ndarray              # float32 N block confidences
    tiles: List[Tuple[int, int, int, int]] = field(default_factory=list)


def _letterbox(rgb: np.ndarray, size: int = NET_SIZE):
    """Resize keeping aspect ratio and pad bottom/right with black."""
    h, w = rgb.shape[:2]
    r = min(size / h, size / w)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (nw, nh), interpolation=interp) if (nw, nh) != (w, h) else rgb
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = resized
    return canvas, nw, nh


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain greedy non-maximum suppression. Returns kept indices."""
    if len(boxes) == 0:
        return np.zeros((0,), np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, np.int64)


def _tile_starts(length: int, tile: int, overlap: int) -> List[int]:
    if length <= tile:
        return [0]
    n = int(np.ceil((length - overlap) / (tile - overlap)))
    last = length - tile
    return sorted({int(round(i * last / (n - 1))) for i in range(n)})


class TextDetector:
    """comic-text-detector running on ONNX Runtime."""

    def __init__(self, model_path: str, threads: int = 0, conf_thresh: float = 0.4,
                 nms_thresh: float = 0.35):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        # Without this the convolutions hit denormal floats on flat page areas
        # and the network runs 5-8x slower on many CPUs.
        opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        names = [o.name for o in self.session.get_outputs()]
        # be robust to exports that order/label the outputs differently
        self._order = self._resolve_outputs(names)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def _resolve_outputs(self, names):
        shapes = {o.name: o.shape for o in self.session.get_outputs()}
        blk = seg = det = None
        for n in names:
            s = shapes[n]
            if len(s) == 3:
                blk = n
            elif len(s) == 4 and s[1] == 1:
                seg = n
            elif len(s) == 4 and s[1] == 2:
                det = n
        if None in (blk, seg, det):
            raise RuntimeError(f"Unexpected detector outputs: {shapes}")
        return [blk, seg, det]

    # ------------------------------------------------------------------ core
    def _run_tile(self, rgb: np.ndarray):
        """Run the network on one tile, results in tile pixel coordinates."""
        h, w = rgb.shape[:2]
        canvas, nw, nh = _letterbox(rgb)
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        blk, seg, det = self.session.run(self._order, {self.input_name: blob})

        seg = np.clip(seg[0, 0, :nh, :nw], 0.0, 1.0)
        line = np.clip(det[0, 0, :nh, :nw], 0.0, 1.0)
        if (nw, nh) != (w, h):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
            line = cv2.resize(line, (w, h), interpolation=cv2.INTER_LINEAR)

        blk = blk[0]
        conf = blk[:, 4] * blk[:, 5:].max(axis=1)
        sel = conf > self.conf_thresh
        blk, conf = blk[sel], conf[sel]
        sx, sy = w / nw, h / nh
        boxes = np.stack([
            (blk[:, 0] - blk[:, 2] / 2) * sx,
            (blk[:, 1] - blk[:, 3] / 2) * sy,
            (blk[:, 0] + blk[:, 2] / 2) * sx,
            (blk[:, 1] + blk[:, 3] / 2) * sy,
        ], axis=1) if len(blk) else np.zeros((0, 4), np.float32)
        return seg, line, boxes.astype(np.float32), conf.astype(np.float32)

    @staticmethod
    def plan_tiles(w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Split a page into pieces that each look like one ordinary page.

        The network was trained on whole pages squeezed into 1024 px, so what
        matters is not the pixel count but how big the lettering is relative
        to the piece it sees.  An ordinary portrait page of any resolution is
        therefore processed in one go; a double-page spread is cut into two
        squares and a webtoon strip into page-shaped slices.
        """
        if w > 1.3 * h:                       # spread / landscape
            tw, th = h, h
        elif h > 1.8 * w:                     # vertical strip
            tw, th = w, int(round(1.4 * w))
        else:
            return [(0, 0, w, h)]
        return [(x, y, min(x + tw, w), min(y + th, h))
                for y in _tile_starts(h, th, int(th * 0.2))
                for x in _tile_starts(w, tw, int(tw * 0.2))]

    def detect(self, rgb: np.ndarray) -> Detection:
        """Detect text on an RGB uint8 page."""
        h, w = rgb.shape[:2]
        tiles = self.plan_tiles(w, h)

        seg = np.zeros((h, w), np.float32)
        line = np.zeros((h, w), np.float32)
        all_boxes, all_scores = [], []
        for (x1, y1, x2, y2) in tiles:
            s, l, b, c = self._run_tile(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
            np.maximum(seg[y1:y2, x1:x2], s, out=seg[y1:y2, x1:x2])
            np.maximum(line[y1:y2, x1:x2], l, out=line[y1:y2, x1:x2])
            if len(b):
                b = b + np.array([x1, y1, x1, y1], np.float32)
                all_boxes.append(b)
                all_scores.append(c)

        if all_boxes:
            boxes = np.concatenate(all_boxes)
            scores = np.concatenate(all_scores)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
            keep = _nms(boxes, scores, self.nms_thresh)
            boxes, scores = boxes[keep], scores[keep]
        else:
            boxes = np.zeros((0, 4), np.float32)
            scores = np.zeros((0,), np.float32)
        return Detection(seg=seg, lines=line, boxes=boxes, scores=scores, tiles=tiles)
