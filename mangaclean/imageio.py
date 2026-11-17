"""Reading and writing page images."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif"}
ARCHIVE_EXTS = {".zip", ".cbz"}

Image.MAX_IMAGE_PIXELS = 400_000_000      # tall webtoon strips are legitimately huge


def natural_key(name: str):
    """Sort key that puts page2 before page10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def decode(im: Image.Image) -> Tuple[np.ndarray, dict]:
    """PIL image -> (RGB uint8 array, metadata needed to write it back)."""
    fmt = (im.format or "").upper()
    im = ImageOps.exif_transpose(im)
    meta = {"format": fmt, "mode": im.mode, "icc": im.info.get("icc_profile"),
            "dpi": im.info.get("dpi"), "alpha": None, "gray": False}
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(im).astype(np.float64)
        top = 65535.0 if arr.max() > 255 else 255.0
        im = Image.fromarray(np.clip(arr / top * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        rgba = np.asarray(im.convert("RGBA"))
        if rgba[..., 3].min() < 255:
            meta["alpha"] = rgba[..., 3].copy()
        rgb = rgba[..., :3]
    else:
        meta["gray"] = im.mode in ("L", "1")
        rgb = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rgb), meta


def load_image(path) -> Tuple[np.ndarray, dict]:
    with Image.open(path) as im:
        im.load()
        return decode(im)


def load_image_bytes(data: bytes) -> Tuple[np.ndarray, dict]:
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return decode(im)


def encode(rgb: np.ndarray, meta: Optional[dict] = None, fmt: str = "png") -> Tuple[bytes, str]:
    """RGB uint8 array -> (file bytes, extension). ``fmt``: png | jpg | webp | same."""
    meta = meta or {}
    fmt = fmt.lower()
    if fmt == "same":
        fmt = {"JPEG": "jpg", "MPO": "jpg", "WEBP": "webp"}.get(meta.get("format", ""), "png")
    if fmt == "jpeg":
        fmt = "jpg"
    alpha = meta.get("alpha")
    if meta.get("gray"):
        im = Image.fromarray(rgb[..., 0], "L")
    else:
        im = Image.fromarray(rgb, "RGB")
    if alpha is not None and fmt != "jpg" and alpha.shape == rgb.shape[:2]:
        im.putalpha(Image.fromarray(alpha, "L"))
    extra = {}
    if meta.get("icc"):
        extra["icc_profile"] = meta["icc"]
    if meta.get("dpi"):
        extra["dpi"] = tuple(float(v) for v in meta["dpi"])
    buf = io.BytesIO()
    if fmt == "jpg":
        im.save(buf, "JPEG", quality=95, subsampling=0, optimize=True, **extra)
    elif fmt == "webp":
        extra.pop("dpi", None)
        im.save(buf, "WEBP", lossless=True, method=4, **extra)
    else:
        fmt = "png"
        im.save(buf, "PNG", compress_level=6, **extra)
    return buf.getvalue(), "." + fmt


def save_image(path, rgb: np.ndarray, meta: Optional[dict] = None, fmt: str = "png") -> Path:
    data, ext = encode(rgb, meta, fmt)
    path = Path(path).with_suffix(ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def is_image(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTS


def list_images(folder: Path, recursive: bool = False, skip: Optional[Path] = None) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    skip = skip.resolve() if skip else None
    out = []
    for p in it:
        if not p.is_file() or not is_image(p):
            continue
        if skip and skip in p.resolve().parents:
            continue
        out.append(p)
    return sorted(out, key=lambda p: natural_key(p.relative_to(folder).as_posix()))


def iter_archive(path: Path) -> Iterator[Tuple[str, bytes]]:
    """Yield (inner name, bytes) for every image inside a .zip / .cbz."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist()
                 if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
                 and not Path(n).name.startswith(".") and "__MACOSX" not in n]
        for n in sorted(names, key=natural_key):
            yield n, zf.read(n)
