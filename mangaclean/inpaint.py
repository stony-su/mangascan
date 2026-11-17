"""Reconstruct artwork underneath erased lettering.

Uses LaMa (the anime/manga fine-tune, exported to ONNX).  The network is not
run on the whole page: masked areas are clustered, every cluster is cut out
with some surrounding context, inpainted on its own and pasted back.  That is
both much faster on a CPU and sharper, because small lettering is processed
at its native resolution.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

MODULO = 8


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def plan_crops(mask: np.ndarray, link: int = 24, max_ctx: int = 128) -> List[dict]:
    """Cluster the mask and choose a context window for every cluster.

    Returns ``[{"box": (x1, y1, x2, y2), "mask": bool array (window sized)}]``.
    Lettering closer together than ``link`` pixels shares one window.
    """
    H, W = mask.shape
    if not mask.any():
        return []
    merged = cv2.dilate(mask.astype(np.uint8), _disk(link))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    lab = lab * mask                         # labels only on real mask pixels
    crops = []
    for k in range(1, n):
        ys, xs = np.nonzero(lab == k)
        if xs.size == 0:
            continue
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        ctx = int(np.clip(0.6 * min(x2 - x1, y2 - y1) + 32, 48, max_ctx))
        box = (max(0, x1 - ctx), max(0, y1 - ctx), min(W, x2 + ctx), min(H, y2 + ctx))
        crops.append({"box": box, "mask": lab[box[1]:box[3], box[0]:box[2]] == k})
    return crops


class LamaInpainter:
    """LaMa on ONNX Runtime with a small result cache."""

    def __init__(self, model_path: str, threads: int = 0, max_side: int = 768):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.log_severity_level = 3
        opts.add_session_config_entry("session.set_denormal_as_zero", "1")
        self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.max_side = int(max_side)
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ core
    def _forward(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run the network on one RGB uint8 crop. Returns RGB uint8."""
        h, w = mask.shape
        ph, pw = (-h) % MODULO, (-w) % MODULO
        if ph or pw:
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
        img = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        m = mask.astype(np.float32)[None, None]
        out = self.session.run(None, {"image": img, "mask": m})[0][0]
        if float(out.max()) > 2.0:           # some exports already answer in 0..255
            out = out / 255.0
        out = np.clip(out.transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return out[:h, :w]

    def inpaint_crop(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint one crop, shrinking it first if it is too large.

        The limit is on the pixel *count* (max_side squared), not on the
        longer side: run time grows with the area, and a long thin line of
        lettering should not be blurred just because it is wide.
        """
        h, w = mask.shape
        scale = min(1.0, (self.max_side * self.max_side / float(h * w)) ** 0.5)
        if scale < 0.999:
            sw, sh = max(MODULO, int(round(w * scale))), max(MODULO, int(round(h * scale)))
            small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
            smask = cv2.resize(mask.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_AREA) > 0
            smask = cv2.dilate(smask.astype(np.uint8), _disk(1)).astype(bool)
            out = self._forward(small, smask)
            return cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
        return self._forward(rgb, mask)

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray,
                progress: Optional[Callable[[int, int], None]] = None) -> np.ndarray:
        """Inpaint ``mask`` (bool HxW) in an RGB uint8 page. Returns a new array."""
        out = rgb.copy()
        crops = plan_crops(mask)
        for i, c in enumerate(crops):
            x1, y1, x2, y2 = c["box"]
            m = c["mask"]
            # everything masked inside the window is hidden from the network,
            # not just this cluster, so it never copies neighbouring lettering
            hide = mask[y1:y2, x1:x2]
            src = out[y1:y2, x1:x2]
            key = hashlib.sha1(np.packbits(hide).tobytes() + repr(c["box"]).encode()
                               + src[~hide].tobytes()).hexdigest()
            res = self._cache.get(key)
            if res is None:
                res = self.inpaint_crop(src, hide)
                if len(self._cache) > 64:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = res
            # feather the seam a little
            alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
            alpha = np.where(m, 1.0, alpha * cv2.dilate(m.astype(np.uint8), _disk(2)))[..., None]
            blend = src.astype(np.float32) * (1 - alpha) + res.astype(np.float32) * alpha
            out[y1:y2, x1:x2] = np.clip(blend + 0.5, 0, 255).astype(np.uint8)
            if progress:
                progress(i + 1, len(crops))
        return out


def telea_inpaint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Classical fallback when the LaMa model is not available."""
    return cv2.inpaint(rgb, mask.astype(np.uint8) * 255, 4, cv2.INPAINT_TELEA)
