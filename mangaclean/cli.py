"""Command line front end.

    python -m mangaclean                       open the review app in the browser
    python -m mangaclean chapter_12/           clean a whole folder
    python -m mangaclean page.jpg -o out/      clean one page
    python -m mangaclean chapter_12/ -t        clean, translate and letter in English
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
                   help="where to write the results (default: a 'cleaned' folder next to the input, "
                        "or 'translated' with --translate)")
    p.add_argument("-t", "--translate", nargs="?", const="auto",
                   choices=["auto", "claude-code", "claude", "offline"], metavar="ENGINE",
                   help="also read the Japanese, translate it to English and letter the page. "
                        "ENGINE: 'claude-code' localises each page with Claude through your Claude "
                        "Code login; 'claude' does the same through the Anthropic API (API key); "
                        "'offline' is a literal rough draft made on this computer. Default 'auto': "
                        "Claude Code if it is installed, else offline. The Claude engines send each "
                        "page to Anthropic and use your account")
    p.add_argument("--notes", metavar="TEXT",
                   help="with --translate: what the translator should know about the series "
                        "(setting, who is who, preferred name spellings)")
    p.add_argument("--notes-file", metavar="FILE",
                   help="like --notes, read from a text file (a notes.txt inside a folder you "
                        "translate is used automatically)")
    p.add_argument("--script", action="store_true",
                   help="with --translate: also save <name>.txt with the Japanese and English of every block")
    p.add_argument("--font", action="append", metavar="[STYLE=]FONT",
                   help="with --translate: letter in this font. FONT is a name from --list-fonts "
                        "(e.g. \"Comic Neue Bold\"), a file in the fonts folder, or the path of a font file. "
                        "On its own it sets the dialogue font (captions then use its italic); STYLE= sets "
                        "another style: dialogue, captions, shouts, titles or asides. Repeat for several")
    p.add_argument("--list-fonts", action="store_true", help="list the fonts --font can use, and quit")
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
    p.add_argument("--ui", action="store_true", help="open the website (default when no input is given)")
    p.add_argument("--port", type=int, default=8765, help="port for the website")
    p.add_argument("--no-browser", action="store_true", help="with --ui: do not open a browser window")
    p.add_argument("--library", metavar="DIR",
                   help="with --ui: where the website keeps your chapters (default: the 'library' "
                        "folder next to the program)")
    p.add_argument("--version", action="version", version=f"mangaclean {__version__}")
    return p


def _expand(inputs: List[str], recursive: bool, output: Optional[Path], folder: str = "cleaned"):
    """Yield jobs: (label, loader, destination path without extension)."""
    for raw in inputs:
        matches = [Path(m) for m in glob.glob(raw)] if any(c in raw for c in "*?[") else [Path(raw)]
        if not matches:
            print(f"! nothing matches {raw}", file=sys.stderr)
        for path in matches:
            if path.is_dir():
                out_dir = output or (path / folder)
                skip_dirs = [out_dir, path / "cleaned", path / "translated"]
                for img in [i for i in list_images(path, recursive, skip=out_dir)
                            if not any(d.resolve() in i.resolve().parents for d in skip_dirs)]:
                    rel = img.relative_to(path)
                    yield str(rel), (lambda p=img: load_image(p)), out_dir / rel.with_suffix("")
            elif path.suffix.lower() in ARCHIVE_EXTS and path.is_file():
                out_dir = (output or (path.parent / folder)) / path.stem
                for name, data in iter_archive(path):
                    yield f"{path.name}:{name}", (lambda d=data: load_image_bytes(d)), \
                        out_dir / Path(name).with_suffix("")
            elif path.is_file() and is_image(path):
                out_dir = output or (path.parent / folder)
                yield path.name, (lambda p=path: load_image(p)), out_dir / path.stem
            else:
                print(f"! skipping {path} (not an image, folder or archive)", file=sys.stderr)


def _write_script(path: Path, label: str, blocks) -> None:
    """Plain-text script of a page: what was read and how it was translated."""
    lines = [f"# {label}", ""]
    for b in blocks:
        state = "" if b.translation.strip() else "   (not lettered: could not be read)"
        lines += [f"[{b.id}]  OCR confidence {b.confidence:.2f}{state}",
                  f"JP: {b.text}", f"EN: {b.translation}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


STYLE_WORDS = {"dialogue": "dialogue", "speech": "dialogue", "captions": "narration", "caption": "narration",
               "narration": "narration", "shouts": "shout", "shout": "shout", "titles": "title", "title": "title",
               "asides": "handwritten", "aside": "handwritten", "handwritten": "handwritten"}


def _list_fonts() -> int:
    from .models import default_fonts_dir
    from .typeset import ROLE_NAMES, FontCatalog
    try:
        sys.stdout.reconfigure(errors="replace")           # font names in any script
    except (AttributeError, ValueError):
        pass
    titles = {"free": "Free comic fonts (downloaded the first time they are used)",
              "yours": "Your fonts (the fonts folder, and professional comic fonts you installed)",
              "installed": "Installed on this computer"}
    entries = FontCatalog(default_fonts_dir()).entries()
    for group, title in titles.items():
        items = [e for e in entries if e["group"] == group]
        if not items:
            continue
        print(title + ":")
        for e in items:
            tag = ROLE_NAMES.get(e["use"], e["use"]) if e["use"] else "professional comic font" if e["pro"] else ""
            print(f"  {e['name']:<36} {tag}".rstrip())
        print()
    print('Letter with one: --font "Comic Neue Bold", or for another style --font shouts="Luckiest Guy".')
    return 0


def _parse_fonts(items: List[str]) -> Optional[dict]:
    """--font values -> {style: font id}; None (after saying why) if one is unknown."""
    from .models import default_fonts_dir
    from .typeset import ROLE_NAMES, FontCatalog
    catalog = FontCatalog(default_fonts_dir())
    fonts = {}
    for item in items:
        style, font = "dialogue", item
        key, sep, rest = item.partition("=")
        if sep and key.strip().lower() in STYLE_WORDS:
            style, font = STYLE_WORDS[key.strip().lower()], rest
        fid = catalog.find(font)
        if fid is None:
            print(f"No font called '{font.strip()}'. See --list-fonts for the ones you can use.", file=sys.stderr)
            return None
        fonts[style] = fid
        print(f"Lettering {ROLE_NAMES[style]} in {catalog.name_of(fid)}", flush=True)
    return fonts


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_fonts:
        return _list_fonts()
    if args.ui or not args.inputs:
        from .server import serve
        return serve(port=args.port, open_browser=not args.no_browser, threads=args.threads,
                     quality=args.quality, library=Path(args.library) if args.library else None)

    from .pipeline import Cleaner, overlay
    from .textmask import MaskSettings

    output = Path(args.output) if args.output else None
    jobs = list(_expand(args.inputs, args.recursive, output, "translated" if args.translate else "cleaned"))
    if not jobs:
        print("No images found.", file=sys.stderr)
        return 1

    fonts = None
    if args.font:
        if not args.translate:
            print("--font only matters with --translate (it chooses how the English is lettered).",
                  file=sys.stderr)
        fonts = _parse_fonts(args.font)
        if fonts is None:
            return 2

    cleaner = Cleaner(threads=args.threads, quality=args.quality,
                      settings=MaskSettings(grow=args.grow), inpaint=not args.no_inpaint,
                      log=lambda s: print(s, flush=True))
    context = None
    if args.translate:
        from .translate import PageContext, best_engine, engine_problems
        if args.translate == "auto":
            args.translate = best_engine()
        problem = engine_problems().get(args.translate)
        if problem:
            print(f"Cannot use the '{args.translate}' translator: {problem}.", file=sys.stderr)
            return 2
        notes = args.notes or ""
        if args.notes_file:
            notes = (notes + "\n" + Path(args.notes_file).read_text(encoding="utf-8")).strip()
        if not notes:                                   # a notes.txt in a chapter folder
            for raw in args.inputs:
                cand = Path(raw) / "notes.txt"
                if Path(raw).is_dir() and cand.is_file():
                    notes = cand.read_text(encoding="utf-8").strip()
                    print(f"Using the series notes in {cand}", flush=True)
                    break
        context = PageContext(notes=notes)
        print({"claude-code": "Translating with Claude through your Claude Code login: each page (image "
                              "and text) is sent to Anthropic and counts toward your Claude usage.",
               "claude": "Translating with the Claude API: each page (image and text) is sent to "
                         "Anthropic and billed to your API account.",
               "offline": "Translating on this computer (a literal rough draft)."}[args.translate],
              flush=True)
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
            result, lettered = clean, None
            if args.translate:
                cleaner.translate(page, args.translate, context=context)
                context.remember(label, page.scene,
                                 [{"japanese": b.text, "english": b.translation}
                                  for b in page.blocks if b.translation.strip()], page.names)
                result = cleaner.typeset(page, clean, fonts)
                lettered = sum(1 for b in page.blocks if b.layout)
                if args.script:
                    _write_script(dest.parent / (dest.name + ".txt"), label, page.blocks)
            saved = save_image(dest, result, meta, args.format)
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
            if lettered is not None:
                unread = sum(1 for b in page.blocks if not b.translation.strip())
                note += f"; {lettered} block{'s' if lettered != 1 else ''} lettered in English"
                if unread:
                    note += f", {unread} could not be read"
            print(f"{prefix}  {note}  {time.perf_counter() - t:.1f}s  -> {saved}")
        except Exception as exc:                      # keep going with the other pages
            failed += 1
            print(f"{prefix}  FAILED: {exc}", file=sys.stderr)
    total = time.perf_counter() - started
    print(f"Done: {len(jobs) - failed} page(s) in {total:.0f}s" + (f", {failed} failed" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
