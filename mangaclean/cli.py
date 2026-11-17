"""Command line front end.

    python -m mangaclean                       open the review app in the browser
    python -m mangaclean chapter_12/           clean a whole folder
    python -m mangaclean page.jpg -o out/      clean one page
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from . import __version__
from .imageio import (ARCHIVE_EXTS, is_image, iter_archive, list_images, load_image,
                      load_image_bytes, save_image)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mangaclean",
        description="Remove the lettering from raw manga pages so they can be re-typeset. "
                    "Run without arguments to open the review app in your browser.")
    p.add_argument("inputs", nargs="*", help="image files, folders, or .zip/.cbz archives")
    p.add_argument("-o", "--output", metavar="DIR",
                   help="where to write the cleaned pages (default: a 'cleaned' folder next to the input)")
    p.add_argument("-f", "--format", choices=["png", "jpg", "webp", "same"], default="png",
                   help="output format (default: png, lossless). 'same' keeps the source format")
    p.add_argument("-r", "--recursive", action="store_true", help="look into sub-folders too")
    p.add_argument("--all", dest="include_uncertain", action="store_true",
                   help="also erase regions the detector is unsure about (stylised logos, "
                        "sound effects); by default those are left alone")
    p.add_argument("--no-inpaint", action="store_true",
                   help="use a simple classical fill instead of the neural inpainter for "
                        "lettering on artwork (much faster, visibly worse)")
    p.add_argument("--quality", choices=["fast", "normal", "high"], default="normal",
                   help="resolution limit for the neural inpainter (default: normal)")
    p.add_argument("--grow", type=float, default=1.0, metavar="X",
                   help="scale the safety margin around lettering (default 1.0; try 1.5 if "
                        "faint outlines are left behind)")
    p.add_argument("--mask", action="store_true", help="also save <name>_mask.png (white = erased)")
    p.add_argument("--debug", action="store_true",
                   help="also save <name>_overlay.png showing what was detected")
    p.add_argument("--skip-existing", action="store_true", help="do not redo pages already in the output folder")
    p.add_argument("--threads", type=int, default=0, metavar="N", help="CPU threads (default: all)")
    p.add_argument("--ui", action="store_true", help="open the review app (default when no input is given)")
    p.add_argument("--port", type=int, default=8765, help="port for the review app")
    p.add_argument("--no-browser", action="store_true", help="with --ui: do not open a browser window")
    p.add_argument("--version", action="version", version=f"mangaclean {__version__}")
    return p


def _expand(inputs: List[str], recursive: bool, output: Optional[Path]):
    """Yield jobs: (label, loader, destination path without extension)."""
    for raw in inputs:
        matches = [Path(m) for m in glob.glob(raw)] if any(c in raw for c in "*?[") else [Path(raw)]
        if not matches:
            print(f"! nothing matches {raw}", file=sys.stderr)
        for path in matches:
            if path.is_dir():
                out_dir = output or (path / "cleaned")
                for img in list_images(path, recursive, skip=out_dir):
                    rel = img.relative_to(path)
                    yield str(rel), (lambda p=img: load_image(p)), out_dir / rel.with_suffix("")
            elif path.suffix.lower() in ARCHIVE_EXTS and path.is_file():
                out_dir = (output or (path.parent / "cleaned")) / path.stem
                for name, data in iter_archive(path):
                    yield f"{path.name}:{name}", (lambda d=data: load_image_bytes(d)), \
                        out_dir / Path(name).with_suffix("")
            elif path.is_file() and is_image(path):
                out_dir = output or (path.parent / "cleaned")
                yield path.name, (lambda p=path: load_image(p)), out_dir / path.stem
            else:
                print(f"! skipping {path} (not an image, folder or archive)", file=sys.stderr)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ui or not args.inputs:
        from .server import serve
        return serve(port=args.port, open_browser=not args.no_browser, threads=args.threads,
                     quality=args.quality)

    from .pipeline import Cleaner, overlay
    from .textmask import MaskSettings

    output = Path(args.output) if args.output else None
    jobs = list(_expand(args.inputs, args.recursive, output))
    if not jobs:
        print("No images found.", file=sys.stderr)
        return 1

    cleaner = Cleaner(threads=args.threads, quality=args.quality,
                      settings=MaskSettings(grow=args.grow), inpaint=not args.no_inpaint,
                      log=lambda s: print(s, flush=True))
    failed = 0
    started = time.perf_counter()
    width = len(str(len(jobs)))
    for i, (label, loader, dest) in enumerate(jobs, 1):
        prefix = f"[{i:{width}d}/{len(jobs)}] {label}"
        if args.skip_existing and any(dest.with_suffix(e).exists() for e in (".png", ".jpg", ".webp")):
            print(f"{prefix}  (already done)")
            continue
        try:
            t = time.perf_counter()
            rgb, meta = loader()
            page = cleaner.analyze(rgb)
            if args.include_uncertain:
                for r in page.regions:
                    r.enabled = True
            clean = cleaner.render(page)
            saved = save_image(dest, clean, meta, args.format)
            if args.mask:
                flat, art = page.masks()
                Image.fromarray(((flat | art) * 255).astype(np.uint8), "L").save(
                    dest.parent / (dest.name + "_mask.png"))
            if args.debug:
                Image.fromarray(overlay(page)).save(dest.parent / (dest.name + "_overlay.png"))
            on = sum(r.enabled for r in page.regions)
            off = len(page.regions) - on
            note = f"{on} text region{'s' if on != 1 else ''} erased"
            if off:
                note += f", {off} uncertain left alone"
            print(f"{prefix}  {note}  {time.perf_counter() - t:.1f}s  -> {saved}")
        except Exception as exc:                      # keep going with the other pages
            failed += 1
            print(f"{prefix}  FAILED: {exc}", file=sys.stderr)
    total = time.perf_counter() - started
    print(f"Done: {len(jobs) - failed} page(s) in {total:.0f}s" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
