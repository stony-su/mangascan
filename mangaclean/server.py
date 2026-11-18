"""Manga Cleaner website: a small web server plus the browser front end.

Everything stays on this machine - the server only listens on 127.0.0.1.
Work is kept in a library on disk (the ``library`` folder next to the
program): chapters of pages, grouped by series. Each page keeps its
original, the cleaned and lettered versions and every decision made on it,
so nothing is lost when the window is closed.

    /                       the library: chapters grouped by series
    /chapter/<id>           one chapter: pages, series notes, translate, download
    /chapter/<id>/edit      the page editor (clean and translate)
    /chapter/<id>/read      the reader
"""
from __future__ import annotations

import copy
import io
import itertools
import json
import mimetypes
import os
import pickle
import queue
import re
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
import webbrowser
import zipfile
import zlib
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import __version__
from .imageio import IMAGE_EXTS, encode, load_image_bytes, natural_key
from .pipeline import Cleaner, Page

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 300 * 1024 * 1024
MAX_FONT = 50 * 1024 * 1024
MAX_UNDO = 30
# which job goes first: edits someone is waiting for, then cleaning new pages,
# then a single page's translation, then whole chapters (translate, re-letter)
PRIORITY = {"warmup": 0, "render": 0, "typeset": 0, "analyze": 1, "translate": 1}
THUMB = (360, 510)
KEEP_LOADED = 40                    # page states held in memory at once
IDLE_BEFORE_UNLOAD = 120.0          # ...and only pages untouched this long are freed
MIME = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".json": "application/json",
        ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".ttf": "font/ttf",
        ".woff2": "font/woff2", ".txt": "text/plain"}


def default_library_dir() -> Path:
    env = os.environ.get("MANGACLEAN_LIBRARY")
    return Path(env) if env else Path(__file__).resolve().parent.parent / "library"


def _write(path: Path, data: bytes) -> None:
    """Write a file so that a crash never leaves half of it behind."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ------------------------------------------------------------------- pages
class PageState:
    """One page of a chapter, kept in its own folder of the library.

    The images live in files (``source.*``, ``clean.png``, ``english.png``);
    the erase regions and text blocks live in ``state.pkl`` and are only
    loaded when the page is worked on; everything the website lists is in
    ``meta.json``.
    """

    META = ("name", "source", "width", "height", "status", "stage", "error", "tstate", "tnote",
            "engine", "scene", "names", "usage", "seconds", "version", "counts", "script")

    def __init__(self, pid: str, chapter: str, folder: Path):
        self.id = pid
        self.chapter = chapter
        self.dir = folder
        self.name = ""
        self.source = "source.png"            # file name of the original upload
        self.meta: dict = {}                  # format details of the original (for export)
        self.width = 0
        self.height = 0
        self.status = "queued"                # queued | working | done | error
        self.stage = "Waiting"
        self.error: Optional[str] = None
        self.tstate = "none"                  # none | working | done | error
        self.tnote = ""                       # why translation failed, in plain words
        self.engine = ""
        self.scene = ""                       # what happens on the page (from the translator)
        self.names: list = []                 # names / terms and their English spelling
        self.usage: Optional[dict] = None     # tokens / cost of the last translation
        self.seconds = 0.0
        self.version = 0
        self.counts: dict = {}                # erased / skipped / blocks / lettered, for listings
        self.script: list = []                # the page's lines, for the next page's translator
        self.history: List[bytes] = []        # undo snapshots (this session only)
        self.job = ""                         # the last job queued for it (this session only)
        self.dirty = False                    # changed in memory, not saved yet
        self._regions = None
        self._blocks = None
        self._protect = None
        self._gray = False
        self._loaded = True                   # a new page has no saved state to load
        self.last_used = time.time()
        self.on_load: Optional[Callable[["PageState"], None]] = None

    @classmethod
    def load(cls, pid: str, chapter: str, folder: Path) -> "PageState":
        ps = cls(pid, chapter, folder)
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        for key in cls.META:
            if key in meta:
                setattr(ps, key, meta[key])
        ps._loaded = False
        return ps

    # -- the regions and blocks, loaded on first use
    def _load(self) -> None:
        self.last_used = time.time()
        if self._loaded:
            return
        self._loaded = True
        path = self.dir / "state.pkl"
        if path.exists():
            try:
                data = pickle.loads(path.read_bytes())
                self._regions, self._blocks = data.get("regions"), data.get("blocks")
                self._protect, self._gray = data.get("protect"), bool(data.get("gray", False))
            except Exception as exc:                          # noqa: BLE001
                print(f"Could not read the saved state of {self.name}: {exc}", flush=True)
        if self.on_load:
            self.on_load(self)

    def unload(self) -> bool:
        """Free the memory of a page nobody is working on (it is on disk)."""
        if not self._loaded or self.dirty or self.status in ("queued", "working")                 or time.time() - self.last_used < IDLE_BEFORE_UNLOAD:
            return False
        if not (self.dir / "state.pkl").exists():
            return False
        self._regions = self._blocks = self._protect = None
        self._loaded = False
        return True

    @property
    def regions(self):
        self._load()
        return self._regions

    @regions.setter
    def regions(self, value):
        self._load()
        self._regions = value

    @property
    def blocks(self):
        self._load()
        return self._blocks

    @blocks.setter
    def blocks(self, value):
        self._load()
        self._blocks = value

    @property
    def protect(self):
        self._load()
        return self._protect

    @protect.setter
    def protect(self, value):
        self._load()
        self._protect = value

    @property
    def gray(self):
        self._load()
        return self._gray

    @gray.setter
    def gray(self, value):
        self._load()
        self._gray = bool(value)

    # -- the image files
    def _path(self, what: str) -> Path:
        return self.dir / {"source": self.source, "clean": "clean.png", "english": "english.png",
                           "thumb": "thumb.jpg"}[what]

    def read(self, what: str) -> Optional[bytes]:
        path = self._path(what)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def write(self, what: str, data: Optional[bytes]) -> None:
        path = self._path(what)
        if data is None:
            path.unlink(missing_ok=True)
        else:
            _write(path, data)
        (self.dir / f"thumb_{what}.jpg").unlink(missing_ok=True)

    @property
    def src(self) -> bytes:
        return self.read("source") or b""

    @property
    def thumb(self) -> bytes:
        return self.read("thumb") or b""

    @property
    def clean(self) -> Optional[bytes]:
        return self.read("clean")

    @clean.setter
    def clean(self, data: Optional[bytes]) -> None:
        self.write("clean", data)

    @property
    def english(self) -> Optional[bytes]:
        return self.read("english")

    @english.setter
    def english(self, data: Optional[bytes]) -> None:
        self.write("english", data)

    @property
    def has_clean(self) -> bool:
        return self._path("clean").exists()

    @property
    def has_english(self) -> bool:
        return self._path("english").exists()

    # -- saving
    def _count(self) -> dict:
        regs, blocks = self._regions or [], self._blocks or []
        return {"erased": sum(1 for r in regs if r.enabled), "skipped": sum(1 for r in regs if not r.enabled),
                "blocks": len(blocks), "lettered": sum(1 for b in blocks if b.layout is not None)}

    def save(self) -> None:
        if self._loaded:
            self.counts = self._count()
            self.script = [{"japanese": b.text, "english": b.translation} for b in (self._blocks or [])
                           if b.enabled and b.translation.strip()]
            if self._regions is not None or self._blocks is not None or self._protect is not None:
                state = {"regions": self._regions, "blocks": self._blocks, "protect": self._protect,
                         "gray": self._gray}
                _write(self.dir / "state.pkl", pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
        meta = {key: getattr(self, key) for key in self.META}
        _write(self.dir / "meta.json", json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        self.dirty = False

    # -- what the website shows
    def summary(self) -> dict:
        counts = self._count() if self._loaded and self._regions is not None else (self.counts or {})
        return {
            "id": self.id, "chapter": self.chapter, "name": self.name,
            "width": self.width, "height": self.height,
            "status": self.status, "stage": self.stage, "error": self.error, "job": self.job,
            "version": self.version, "seconds": round(self.seconds, 1),
            "erased": counts.get("erased", 0), "skipped": counts.get("skipped", 0),
            "canUndo": bool(self.history),
            "tstate": self.tstate, "tnote": self.tnote, "engine": self.engine,
            "scene": self.scene,
            "cleaned": self.has_clean, "translated": self.has_english,
        }

    def detail(self) -> dict:
        d = self.summary()
        d["regions"] = [{
            "id": r.id, "bbox": [int(v) for v in r.bbox], "kind": r.kind,
            "enabled": bool(r.enabled), "confident": bool(r.confident), "source": r.source,
        } for r in (self.regions or [])]
        d["blocks"] = [{
            "id": b.id, "box": [int(v) for v in b.box], "japanese": b.text,
            "english": b.translation, "confidence": round(float(b.confidence), 2),
            "enabled": bool(b.enabled), "edited": bool(b.edited),
            "role": (b.layout or {}).get("role") or b.style.get("role_override") or "",
            "roleLocked": bool(b.style.get("role_override")),
            "scale": float(b.style.get("scale", 1.0)),
            "font": b.style.get("font", ""),                 # picked for this line only
            "fontName": (b.layout or {}).get("fontName", ""),  # what it is lettered in
            "size": (b.layout or {}).get("size"),
            "lettered": b.layout is not None,
            "overflow": bool((b.layout or {}).get("overflow")),
            "drawn": (b.layout or {}).get("box"),
        } for b in (self.blocks or [])]
        return d


# ---------------------------------------------------------------- chapters
class Chapter:
    def __init__(self, cid: str, folder: Path, title: str = "", series: str = "", notes: str = "",
                 created: Optional[float] = None, updated: Optional[float] = None,
                 pages: Optional[List[str]] = None, fonts: Optional[dict] = None):
        self.id = cid
        self.folder = folder
        self.title = title
        self.series = series
        self.notes = notes                     # used when the chapter belongs to no series
        self.fonts: dict = dict(fonts or {})   # style -> font id; likewise
        self.created = created or time.time()
        self.updated = updated or self.created
        self.pages: List[str] = list(pages or [])

    def to_json(self) -> dict:
        return {"id": self.id, "title": self.title, "series": self.series, "notes": self.notes,
                "fonts": self.fonts, "created": self.created, "updated": self.updated, "pages": self.pages}

    def save(self) -> None:
        _write(self.folder / "chapter.json", json.dumps(self.to_json(), ensure_ascii=False, indent=1).encode("utf-8"))

    def touch(self) -> None:
        self.updated = time.time()
        self.save()


class Library:
    """Series, chapters and pages on disk."""

    def __init__(self, root: Path, log: Callable[[str], None]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.chapters: "OrderedDict[str, Chapter]" = OrderedDict()
        self.pages: Dict[str, PageState] = {}
        self.series: Dict[str, dict] = {}       # name -> {"notes": ...}
        self.settings: dict = {}
        self._in_memory: "OrderedDict[str, None]" = OrderedDict()
        self._mem_lock = threading.Lock()
        self.load()

    # -- loading and saving
    def load(self) -> None:
        index = self.root / "library.json"
        if index.exists():
            try:
                data = json.loads(index.read_text(encoding="utf-8"))
                self.series = {str(k): dict(v) for k, v in (data.get("series") or {}).items()}
                self.settings = dict(data.get("settings") or {})
            except (ValueError, OSError, AttributeError) as exc:
                self.log(f"The library index could not be read ({exc}); chapters are still loaded.")
        found = []
        for folder in self.root.iterdir():
            info_file = folder / "chapter.json"
            if not folder.is_dir() or not info_file.exists():
                continue
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
                fonts = info.get("fonts") if isinstance(info.get("fonts"), dict) else {}
                ch = Chapter(folder.name, folder, info.get("title", ""), info.get("series", ""),
                             info.get("notes", ""), info.get("created"), info.get("updated"), fonts=fonts)
            except (ValueError, OSError) as exc:
                self.log(f"Skipping the chapter in {folder.name}: {exc}")
                continue
            for pid in info.get("pages", []):
                try:
                    ps = PageState.load(pid, ch.id, folder / pid)
                except (ValueError, OSError) as exc:
                    self.log(f"Skipping a page of '{ch.title}': {exc}")
                    continue
                ps.on_load = self._loaded
                self.pages[pid] = ps
                ch.pages.append(pid)
            found.append(ch)
        for ch in sorted(found, key=lambda c: c.created):
            self.chapters[ch.id] = ch

    def save_index(self) -> None:
        data = {"version": 1, "series": self.series, "settings": self.settings}
        _write(self.root / "library.json", json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))

    def _loaded(self, ps: PageState) -> None:
        """Keep only the most recently used page states in memory."""
        with self._mem_lock:
            self._in_memory[ps.id] = None
            self._in_memory.move_to_end(ps.id)
            if len(self._in_memory) <= KEEP_LOADED:
                return
            for pid in list(self._in_memory)[:-KEEP_LOADED // 2]:
                other = self.pages.get(pid)
                if other is None or other.unload():
                    self._in_memory.pop(pid, None)

    def touch(self, ps: PageState) -> None:
        ps.last_used = time.time()
        if ps._loaded:
            self._loaded(ps)

    # -- chapters
    def new_chapter(self, title: str, series: str = "") -> Chapter:
        cid = _new_id()
        folder = self.root / cid
        folder.mkdir(parents=True)
        ch = Chapter(cid, folder, (title or "").strip()[:200] or "Untitled chapter", (series or "").strip()[:200])
        ch.save()
        self.chapters[cid] = ch
        if ch.series and ch.series not in self.series:
            self.series[ch.series] = {"notes": ""}
            self.save_index()
        return ch

    def default_chapter(self) -> Chapter:
        """Where pages go when no chapter was named (older clients, scripts)."""
        for ch in self.chapters.values():
            if ch.title == "Unsorted pages" and not ch.series:
                return ch
        return self.new_chapter("Unsorted pages")

    def update_chapter(self, ch: Chapter, title: Optional[str] = None, series: Optional[str] = None) -> None:
        if title is not None and title.strip():
            ch.title = title.strip()[:200]
        if series is not None:
            old, new = ch.series, series.strip()[:200]
            if new != old:
                old_notes, old_fonts = self.notes_for(ch), self.fonts_for(ch)
                ch.series = new
                if new and new not in self.series:
                    self.series[new] = {"notes": ""}
                still_used = any(c.series == old for c in self.chapters.values() if c is not ch)
                # renaming a series (its only chapter moved): the notes and fonts go with it
                if old_notes and not self.notes_for(ch):
                    self.set_notes(ch, old_notes)
                if old_fonts and not self.fonts_for(ch):
                    self.set_fonts(ch, old_fonts)
                if old and not still_used:
                    self.series.pop(old, None)
                self.save_index()
        ch.touch()

    def delete_chapter(self, ch: Chapter) -> None:
        self.chapters.pop(ch.id, None)
        with self._mem_lock:
            for pid in ch.pages:
                self.pages.pop(pid, None)
                self._in_memory.pop(pid, None)
        if ch.series and not any(c.series == ch.series for c in self.chapters.values()) \
                and not self.series.get(ch.series, {}).get("notes") \
                and not self.series.get(ch.series, {}).get("fonts"):
            self.series.pop(ch.series, None)
            self.save_index()
        shutil.rmtree(ch.folder, ignore_errors=True)

    def chapter_pages(self, ch: Chapter) -> List[PageState]:
        return [self.pages[p] for p in ch.pages if p in self.pages]

    def chapter_of(self, ps: PageState) -> Optional[Chapter]:
        return self.chapters.get(ps.chapter)

    def notes_for(self, ch: Chapter) -> str:
        if ch.series:
            return self.series.get(ch.series, {}).get("notes", "")
        return ch.notes

    def set_notes(self, ch: Chapter, text: str) -> None:
        text = (text or "")[:4000]
        if ch.series:
            self.series.setdefault(ch.series, {})["notes"] = text
            self.save_index()
        else:
            ch.notes = text
            ch.save()

    def fonts_for(self, ch: Chapter) -> dict:
        """The fonts picked for a chapter (style -> font id); like the notes,
        they belong to its series."""
        if ch.series:
            return dict(self.series.get(ch.series, {}).get("fonts") or {})
        return dict(ch.fonts or {})

    def set_fonts(self, ch: Chapter, changes: dict) -> dict:
        """Pick fonts (style -> font id, or None to go back to automatic)."""
        fonts = self.fonts_for(ch)
        for role, fid in changes.items():
            if fid:
                fonts[role] = fid
            else:
                fonts.pop(role, None)
        if ch.series:
            entry = self.series.setdefault(ch.series, {"notes": ""})
            if fonts:
                entry["fonts"] = fonts
            else:
                entry.pop("fonts", None)
            self.save_index()
        else:
            ch.fonts = fonts
            ch.save()
        return fonts

    def series_chapters(self, ch: Chapter) -> List[Chapter]:
        if not ch.series:
            return [ch]
        same = [c for c in self.chapters.values() if c.series == ch.series]
        return sorted(same, key=lambda c: (natural_key(c.title), c.created))

    # -- pages
    def add_page(self, ch: Chapter, name: str, data: bytes) -> PageState:
        with Image.open(io.BytesIO(data)) as im:              # validates the upload
            fmt = (im.format or "PNG").lower()
            im = ImageOps.exif_transpose(im)
            width, height = im.size
            th = im.convert("RGB")
            th.thumbnail(THUMB)
            buf = io.BytesIO()
            th.save(buf, "JPEG", quality=82)
        ext = Path(name).suffix.lower()
        if ext not in IMAGE_EXTS:
            ext = "." + {"jpeg": "jpg"}.get(fmt, fmt)
        pid = _new_id()
        folder = ch.folder / pid
        folder.mkdir(parents=True)
        ps = PageState(pid, ch.id, folder)
        ps.name = Path(name).name[:200] or f"page{len(ch.pages) + 1}{ext}"
        ps.source = "source" + ext
        ps.width, ps.height = width, height
        _write(folder / ps.source, data)
        _write(folder / "thumb.jpg", buf.getvalue())
        ps.on_load = self._loaded
        ps.save()
        self.pages[pid] = ps
        ch.pages.append(pid)
        ch.touch()
        return ps

    def delete_page(self, ps: PageState) -> None:
        with self._mem_lock:
            self.pages.pop(ps.id, None)
            self._in_memory.pop(ps.id, None)
        ch = self.chapters.get(ps.chapter)
        if ch is not None and ps.id in ch.pages:
            ch.pages.remove(ps.id)
            ch.touch()
        shutil.rmtree(ps.dir, ignore_errors=True)


# --------------------------------------------------------------------- app
class App:
    def __init__(self, threads: int = 0, quality: str = "normal", library: Optional[Path] = None):
        from .translate import best_engine, engine_problems
        from .typeset import FontBook
        self.cleaner = Cleaner(threads=threads, quality=quality, log=self._log)
        self.lock = threading.RLock()
        self.jobs: "queue.PriorityQueue" = queue.PriorityQueue()
        self._seq = itertools.count()                  # first come, first served within a priority
        self._reletter: set = set()                    # pages waiting to be lettered again
        self._decoded: "OrderedDict[str, tuple]" = OrderedDict()   # small LRU of decoded pages
        self.model_note = ""
        # the website's own view of the fonts (names, previews); the lettering
        # itself happens in the worker with the typesetter's book
        catalog = self.cleaner.typesetter.book.catalog
        self.fontbook = FontBook(log=lambda s: print(s, flush=True), catalog=catalog)
        self._preview_lock = threading.Lock()
        threading.Thread(target=catalog.entries, daemon=True, name="fonts").start()
        self.lib = Library(library or default_library_dir(), self._log)
        saved = self.lib.settings.get("engine")
        self.engine = saved if saved and engine_problems().get(saved, "x") is None else best_engine()
        self.auto_translate = bool(self.lib.settings.get("autoTranslate", False))
        threading.Thread(target=self._worker, daemon=True, name="cleaner").start()
        self.jobs.put((0, next(self._seq), "warmup", None, None, None))
        self._resume()

    @property
    def pages(self) -> Dict[str, PageState]:
        return self.lib.pages

    # ------------------------------------------------------------ plumbing
    def _log(self, msg: str) -> None:
        self.model_note = msg.strip()
        print(msg, flush=True)

    def _resume(self) -> None:
        """Finish what the last session left half done."""
        for ch in list(self.lib.chapters.values()):
            for ps in self.lib.chapter_pages(ch):
                if ps.tstate == "working":
                    ps.tstate, ps.tnote = "error", "Interrupted when the app was closed. Translate the page again."
                if ps.status in ("queued", "working"):
                    if not (ps.dir / "state.pkl").exists():
                        self.submit("analyze", ps)
                    elif not ps.has_clean:
                        self.submit("render", ps)
                    else:
                        ps.status, ps.stage = "done", "Done"
                        ps.save()

    def save_settings(self) -> None:
        self.lib.settings.update({"engine": self.engine, "autoTranslate": self.auto_translate})
        self.lib.save_index()

    def _arrays(self, ps: PageState):
        """Decoded (rgb, work) for a page, kept for the few most recent pages."""
        hit = self._decoded.get(ps.id)
        if hit is not None:
            self._decoded.move_to_end(ps.id)
            return hit
        rgb, meta = load_image_bytes(ps.src)
        ps.meta = meta
        from .pipeline import is_grayscale
        gray = is_grayscale(rgb)
        work = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray else rgb
        self._decoded[ps.id] = (rgb, gray, work)
        while len(self._decoded) > 4:
            self._decoded.popitem(last=False)
        return self._decoded[ps.id]

    def _page(self, ps: PageState) -> Page:
        rgb, gray, work = self._arrays(ps)
        return Page(rgb=rgb, gray=gray, work=work, regions=ps.regions or [],
                    blocks=ps.blocks or [], protect=ps.protect)

    def _worker(self) -> None:
        while True:
            _, _, kind, ps, done, arg = self.jobs.get()
            try:
                if kind == "warmup":
                    _ = self.cleaner.detector
                    _ = self.cleaner.inpainter
                    self.model_note = ""
                elif ps.id in self.lib.pages:
                    if arg == "reletter":
                        with self.lock:              # a font change from now on queues it again
                            self._reletter.discard(ps.id)
                    self._run(kind, ps, arg)
            except Exception as exc:                          # noqa: BLE001
                if ps is not None and ps.id not in self.lib.pages:
                    pass                                      # deleted while it was being worked on
                else:
                    traceback.print_exc()
                    if ps is not None:
                        with self.lock:
                            ps.status, ps.error, ps.stage = "error", str(exc), "Failed"
                            ps.version += 1
                            ps.save()
            finally:
                if done is not None:
                    done.set()

    def _run(self, kind: str, ps: PageState, arg=None) -> None:
        """One job for one page.

        analyze    find the text, erase it (and translate, if that is switched on)
        render     erase again after the regions changed
        translate  read, translate and letter the page
        typeset    letter again after a translation, style or font was changed
        """
        t0 = time.perf_counter()
        with self.lock:
            ps.status, ps.error = "working", None
            ps.stage = {"analyze": "Finding text", "render": "Erasing",
                        "translate": "Reading the text", "typeset": "Lettering"}[kind]
            self.lib.touch(ps)
        if kind == "analyze":
            rgb, gray, work = self._arrays(ps)
            page = self.cleaner.analyze(rgb)
            with self.lock:
                ps.regions, ps.blocks, ps.gray = page.regions, page.blocks, page.gray
                ps.stage = "Erasing"
        page = self._page(ps)
        if kind in ("analyze", "render") or not ps.has_clean:
            clean = self.cleaner.render(page)
            buf = io.BytesIO()
            Image.fromarray(clean).save(buf, "PNG", compress_level=1)
            with self.lock:
                ps.clean = buf.getvalue()
        else:                                      # nothing is erased differently: reuse the cleaned page
            clean = np.asarray(Image.open(io.BytesIO(ps.clean)).convert("RGB"))

        want = kind == "translate" or (kind == "analyze" and self.auto_translate)
        if want:
            engine = arg or self.engine
            with self.lock:
                ps.tstate, ps.tnote, ps.engine = "working", "", engine
                ps.stage = "Reading the text"
            try:
                self.cleaner.read(page)
                with self.lock:
                    ps.stage = "Translating"
                self.cleaner.translate(page, engine, context=self._context(ps))
                with self.lock:
                    ps.tstate = "done"
                    ps.scene, ps.names, ps.usage = page.scene, page.names, page.usage
            except Exception as exc:                          # noqa: BLE001
                from .translate import TranslationUnavailable
                if not isinstance(exc, TranslationUnavailable):
                    traceback.print_exc()
                with self.lock:
                    ps.tstate, ps.tnote = "error", str(exc)
        # (a page whose last translation failed keeps the English it had: letter that again)
        if ps.tstate == "done" or (kind == "typeset" and ps.has_english):
            with self.lock:
                ps.stage = "Lettering"
                ch = self.lib.chapter_of(ps)
                fonts = self.lib.fonts_for(ch) if ch is not None else {}
            english = self.cleaner.typeset(page, clean, fonts)
            buf = io.BytesIO()
            Image.fromarray(english).save(buf, "PNG", compress_level=1)
            with self.lock:
                ps.english = buf.getvalue()
        with self.lock:
            ps.status, ps.stage = "done", "Done"
            if kind == "analyze":
                ps.seconds = time.perf_counter() - t0
            ps.version += 1
            if ps.id in self.lib.pages:
                ps.save()
                ch = self.lib.chapter_of(ps)
                if ch is not None:
                    ch.touch()

    def submit(self, kind: str, ps: PageState, wait: float = 0.0, arg=None,
               priority: Optional[int] = None) -> None:
        done = threading.Event()
        with self.lock:
            if ps.status != "working":
                ps.status = "queued"
            ps.job = kind
        self.jobs.put((PRIORITY[kind] if priority is None else priority, next(self._seq), kind, ps, done, arg))
        if wait:
            done.wait(wait)

    # --------------------------------------------------------- translation
    def _context(self, ps: PageState):
        """What the translator should know before reading ``ps``: the series
        notes, every name already settled in the series, and the story so far
        (the end of the previous chapter, then the pages before this one)."""
        from .translate import PageContext
        with self.lock:
            ch = self.lib.chapter_of(ps)
            if ch is None:
                return PageContext()
            ctx = PageContext(notes=self.lib.notes_for(ch))
            chapters = self.lib.series_chapters(ch)
            here = chapters.index(ch) if ch in chapters else 0
            if here > 0:                           # how the previous chapter ended
                prev = self.lib.chapter_pages(chapters[here - 1])
                for other in [p for p in prev if p.scene][-4:]:
                    ctx.previous.append({"name": other.name, "scene": other.scene, "lines": []})
            pages = self.lib.chapter_pages(ch)
            for other in pages[:pages.index(ps)] if ps in pages else []:
                if other.tstate == "done":
                    ctx.remember(other.name, other.scene, other.script, other.names)
            ctx.add_names(ps.names)
            for c in chapters:                     # names settled anywhere in the series stay settled
                for other in self.lib.chapter_pages(c):
                    ctx.add_names(other.names)
        return ctx

    def engine_problem(self, engine: str) -> Optional[str]:
        from .translate import engine_problems
        return engine_problems().get(engine, "unknown translator")

    def translate(self, ps: PageState, engine: Optional[str] = None) -> None:
        engine = engine or self.engine
        problem = self.engine_problem(engine)
        if problem:                              # say so plainly instead of failing mid-way
            with self.lock:
                ps.tstate = "error" if not ps.has_english else ps.tstate
                ps.tnote = f"That translator is not available: {problem}."
                ps.version += 1
                ps.save()
            return
        self.submit("translate", ps, wait=900, arg=engine)

    def translate_chapter(self, ch: Chapter, engine: Optional[str] = None, redo: bool = False):
        """Queue every page of a chapter, in reading order, so each page is
        translated knowing the ones before it. Returns (queued, problem)."""
        engine = engine or self.engine
        problem = self.engine_problem(engine)
        if problem:
            return 0, problem
        queued = 0
        with self.lock:
            pages = self.lib.chapter_pages(ch)
        for ps in pages:
            if redo or ps.tstate != "done":
                with self.lock:
                    ps.tstate, ps.tnote = "working", ""
                    ps.stage = "Waiting to be translated"
                self.submit("translate", ps, arg=engine, priority=2)
                queued += 1
        return queued, None

    # ------------------------------------------------------------- fonts
    def lettering(self, ch: Chapter) -> dict:
        """The chapter's font choices, and the font that letters each style."""
        from .typeset import ROLES
        with self.lock:
            choices = self.lib.fonts_for(ch)
        book = self.fontbook.using(choices)
        return {"fonts": choices, "fontsVersion": book.catalog.version,
                "lettering": {r: {"id": book.font_id(r), "name": book.name(r), "how": book.how(r)}
                              for r in ROLES}}

    def font_ids(self, choices: dict) -> dict:
        from .typeset import ROLES
        book = self.fontbook.using(choices)
        return {r: book.font_id(r) for r in ROLES}

    def parse_fonts(self, raw) -> dict:
        """A font change from the website: {style: font id, or "" for automatic}."""
        from .typeset import ROLES
        if not isinstance(raw, dict):
            raise ValueError("fonts must map styles to fonts")
        out = {}
        for role, fid in raw.items():
            if role not in ROLES:
                raise ValueError(f"unknown style: {role}")
            fid = str(fid or "").strip()
            if fid and not self.fontbook.catalog.exists(fid):
                raise ValueError("That font is not on this computer.")
            out[role] = fid or None
        return out

    def reletter_chapter(self, ch: Chapter, first: Optional[str] = None) -> int:
        """Letter the chapter's English pages again (its fonts changed), without
        translating anything again. ``first`` is the page someone is looking
        at: it goes before everything else. Returns how many pages."""
        with self.lock:
            pages = [p for p in self.lib.chapter_pages(ch) if p.has_english and p.tstate != "working"]
            for p in sorted(pages, key=lambda p: p.id != first):
                if p.id == first:
                    self.submit("typeset", p, arg="reletter", priority=0)
                elif p.id not in self._reletter:   # already waiting: it reads the newest fonts when it runs
                    self._reletter.add(p.id)
                    p.stage = "Waiting to be lettered again"
                    self.submit("typeset", p, arg="reletter", priority=2)
        return len(pages)

    def fonts_json(self) -> dict:
        from .typeset import ROLES
        cat = self.fontbook.catalog
        auto = {}
        for role in ROLES:
            fid, why = self.fontbook.auto(role)
            auto[role] = {"id": fid, "name": cat.name_of(fid), "why": why}
        return {"fonts": cat.public(), "version": cat.version, "auto": auto}

    def font_preview(self, fid: str, role: str, text: str = "") -> bytes:
        """PNG of a line lettered in a font, for the font picker."""
        from .typeset import render_sample
        book = self.fontbook.using({role: fid})
        if book.file(fid) is None:                 # a free font that could not be downloaded
            raise LookupError("This font could not be downloaded. Is the computer online?")
        with self._preview_lock:
            return render_sample(book, role, text)

    def add_font(self, name: str, data: bytes) -> dict:
        """Keep a font file someone added in the fonts folder; returns its catalogue entry."""
        from PIL import ImageFont

        from .typeset import FONT_EXTS, _display
        base = Path(name).name
        ext = Path(base).suffix.lower()
        if ext not in FONT_EXTS:
            raise ValueError("Choose a .ttf or .otf font file.")
        if not data or len(data) > MAX_FONT:
            raise ValueError(f"{base} is empty or too large for a font.")
        try:
            font = ImageFont.truetype(io.BytesIO(data), 24)
            font.getbbox("A")
            label = _display(*(str(v or "").strip() for v in font.getname()))
        except Exception:                                      # noqa: BLE001
            raise ValueError(f"{base} is not a font file this app can use.") from None
        catalog = self.fontbook.catalog
        same = next((e for e in catalog.entries() if e["name"].lower() == label.lower()), None)
        if same is None:
            stem = re.sub(r"[^\w\- .()]+", "", Path(base).stem).strip(" .") or "font"
            folder = self.fontbook.dir
            folder.mkdir(parents=True, exist_ok=True)
            target, k = folder / f"{stem}{ext}", 2
            while target.exists():                 # never overwrite another font of the same file name
                target, k = folder / f"{stem} ({k}){ext}", k + 1
            _write(target, data)
            same = catalog.get(target.name)
        if same is None:
            raise ValueError(f"{base} could not be added.")
        return {k: v for k, v in same.items() if not k.startswith("_")}

    def edit_block(self, ps: PageState, body: dict) -> bool:
        """Change one block's translation, style or whether it is lettered."""
        from .typeset import ROLES
        with self.lock:
            block = next((b for b in ps.blocks or [] if b.id == body.get("id")), None)
            if block is None:
                return False
            if "english" in body:
                text = str(body["english"])[:2000]
                if text != block.translation:
                    block.translation, block.edited = text, True
            if "enabled" in body:
                block.enabled = bool(body["enabled"])
            if "role" in body:
                role = body["role"]
                if role in ROLES:
                    block.style["role_override"] = role
                else:
                    block.style.pop("role_override", None)
            if "scale" in body:
                scale = min(3.0, max(0.3, float(body["scale"])))
                if abs(scale - 1.0) < 0.01:
                    block.style.pop("scale", None)
                else:
                    block.style["scale"] = scale
            if "font" in body:                     # checked against the catalogue by the caller
                fid = str(body["font"] or "").strip()
                if fid:
                    block.style["font"] = fid
                else:
                    block.style.pop("font", None)
            if ps.tstate == "none":
                ps.tstate = "done"                 # lettering typed in by hand counts
            ps.dirty = True
        self.submit("typeset", ps, wait=120)
        return True

    # ------------------------------------------------------------- actions
    def add(self, name: str, data: bytes, ch: Optional[Chapter] = None) -> PageState:
        with self.lock:
            ch = ch or self.lib.default_chapter()
            ps = self.lib.add_page(ch, name, data)
        self.submit("analyze", ps)
        return ps

    def delete_page(self, ps: PageState) -> None:
        with self.lock:
            self.lib.delete_page(ps)
            self._decoded.pop(ps.id, None)

    def _snapshot(self, ps: PageState) -> None:
        state = {
            "enabled": [r.enabled for r in ps.regions or []],
            "manual": [copy.copy(r) for r in ps.regions or [] if r.source == "manual"],
            "protect": None if ps.protect is None else zlib.compress(np.packbits(ps.protect).tobytes(), 1),
        }
        ps.history.append(pickle.dumps(state))
        del ps.history[:-MAX_UNDO]
        ps.dirty = True

    def undo(self, ps: PageState) -> None:
        with self.lock:
            if not ps.history:
                return
            state = pickle.loads(ps.history.pop())
            auto = [r for r in ps.regions or [] if r.source != "manual"]
            ps.regions = auto + state["manual"]
            for r, e in zip(ps.regions, state["enabled"]):
                r.enabled = e
            if state["protect"] is None:
                ps.protect = None
            else:
                bits = np.frombuffer(zlib.decompress(state["protect"]), np.uint8)
                ps.protect = np.unpackbits(bits)[: ps.width * ps.height].reshape(ps.height, ps.width).astype(bool)
            ps.dirty = True
        self.submit("render", ps, wait=120)

    def toggle(self, ps: PageState, rid: Optional[int], x: Optional[float], y: Optional[float],
               enabled: Optional[bool]) -> Optional[int]:
        with self.lock:
            regs = ps.regions or []
            target = None
            if rid is not None:
                target = next((r for r in regs if r.id == rid), None)
            elif x is not None and y is not None:
                xi, yi = int(x), int(y)
                best = None
                for r in regs:                      # exact hit on a mask wins
                    x1, y1, x2, y2 = r.box
                    if x1 <= xi < x2 and y1 <= yi < y2 and (r.flat | r.art)[yi - y1, xi - x1]:
                        best = r
                        break
                if best is None:                    # otherwise the smallest box around the point
                    inside = [r for r in regs if r.bbox[0] - 4 <= xi < r.bbox[2] + 4
                              and r.bbox[1] - 4 <= yi < r.bbox[3] + 4]
                    if inside:
                        best = min(inside, key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]))
                target = best
            if target is None:
                return None
            self._snapshot(ps)
            target.enabled = (not target.enabled) if enabled is None else bool(enabled)
            hit = target.id
        self.submit("render", ps, wait=120)
        return hit

    def set_all(self, ps: PageState, which: str, enabled: bool) -> None:
        with self.lock:
            self._snapshot(ps)
            for r in ps.regions or []:
                if which == "all" or (which == "uncertain" and not r.confident and r.source != "manual"):
                    r.enabled = enabled
        self.submit("render", ps, wait=120)

    def stroke(self, ps: PageState, points, radius: float, mode: str) -> None:
        h, w = ps.height, ps.width
        mask = np.zeros((h, w), np.uint8)
        pts = np.round(np.asarray(points, np.float64)).astype(np.int32).reshape(-1, 2)
        if len(pts) == 0:
            return
        rad = int(max(1, round(radius)))
        if len(pts) > 1:
            cv2.polylines(mask, [pts], False, 1, thickness=2 * rad, lineType=cv2.LINE_8)
        for p in (pts[0], pts[-1]):
            cv2.circle(mask, (int(p[0]), int(p[1])), rad, 1, -1)
        mask = mask.astype(bool)
        with self.lock:
            self._snapshot(ps)
            if mode == "keep":
                ps.protect = mask if ps.protect is None else (ps.protect | mask)
            else:
                if ps.protect is not None:
                    ps.protect = ps.protect & ~mask
                page = self._page(ps)
                self.cleaner.add_manual(page, mask, "art" if mode == "inpaint" else "auto")
                ps.regions = page.regions
        self.submit("render", ps, wait=180)

    def reset(self, ps: PageState) -> None:
        with self.lock:
            self._snapshot(ps)
            ps.regions = [r for r in ps.regions or [] if r.source != "manual"]
            for r in ps.regions:
                r.enabled = r.confident
            ps.protect = None
        self.submit("render", ps, wait=120)

    def overlay_png(self, ps: PageState) -> bytes:
        rgba = np.zeros((ps.height, ps.width, 4), np.uint8)
        with self.lock:
            for r in ps.regions or []:
                x1, y1, x2, y2 = r.box
                sub = rgba[y1:y2, x1:x2]
                if r.enabled:
                    sub[r.flat] = (0, 200, 90, 150)
                    sub[r.art] = (240, 60, 60, 150)
                else:
                    sub[r.flat | r.art] = (255, 170, 0, 140)
            if ps.protect is not None:
                rgba[ps.protect] = (70, 130, 255, 120)
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=1)
        return buf.getvalue()

    def thumb(self, ps: PageState, what: str = "original") -> bytes:
        """A small JPEG of a page; ``best`` is the most finished version."""
        if what == "best":
            what = "english" if ps.has_english else "clean" if ps.has_clean else "original"
        if what in ("english", "clean"):
            src, cache = ps.dir / f"{what}.png", ps.dir / f"thumb_{what}.jpg"
            try:
                if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
                    return cache.read_bytes()
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail(THUMB)
                    buf = io.BytesIO()
                    im.save(buf, "JPEG", quality=82)
                _write(cache, buf.getvalue())
                return buf.getvalue()
            except OSError:
                pass
        return ps.thumb

    def export(self, ps: PageState, fmt: str, what: str = "clean"):
        """(bytes, filename) of the cleaned or lettered page in the requested format."""
        src = ps.english if what == "english" and ps.has_english else ps.clean
        if src is None:
            raise ValueError("page is not ready yet")
        rgb = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
        if not ps.meta:
            self._arrays(ps)
        data, ext = encode(rgb, ps.meta, fmt)
        return data, Path(ps.name).stem + ext

    def zip_pages(self, pages: List[PageState], fmt: str, what: str, numbered: bool) -> bytes:
        buf = io.BytesIO()
        used = set()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for i, p in enumerate([p for p in pages if p.has_clean], 1):
                data, fname = self.export(p, fmt, what)
                if numbered:                       # keeps the reading order in any file browser
                    fname = f"{i:03d}_{fname}"
                base, k = fname, 2
                while fname in used:
                    fname = f"{Path(base).stem}_{k}{Path(base).suffix}"
                    k += 1
                used.add(fname)
                zf.writestr(fname, data)
        return buf.getvalue()

    # -------------------------------------------------------- the website
    def chapter_summary(self, ch: Chapter) -> dict:
        pages = self.lib.chapter_pages(ch)
        cover = pages[0] if pages else None
        return {
            "id": ch.id, "title": ch.title, "series": ch.series,
            "created": ch.created, "updated": ch.updated,
            "pages": len(pages),
            "cleaned": sum(1 for p in pages if p.has_clean),
            "translated": sum(1 for p in pages if p.has_english),
            "working": sum(1 for p in pages if p.status in ("queued", "working")),
            "errors": sum(1 for p in pages if p.status == "error" or p.tstate == "error"),
            "cover": cover.id if cover else None,
            "coverVersion": cover.version if cover else 0,
        }

    def chapter_detail(self, ch: Chapter) -> dict:
        d = self.chapter_summary(ch)
        d["notes"] = self.lib.notes_for(ch)
        d["pageList"] = [p.summary() for p in self.lib.chapter_pages(ch)]
        series = self.lib.series_chapters(ch) if ch.series else []
        at = next((i for i, c in enumerate(series) if c is ch), -1)
        d["prev"] = series[at - 1].id if at > 0 else None
        d["next"] = series[at + 1].id if 0 <= at < len(series) - 1 else None
        return d

    def settings_json(self) -> dict:
        from .translate import engine_problems
        return {"autoTranslate": self.auto_translate, "engine": self.engine, "engines": engine_problems()}


# ------------------------------------------------------------------ HTTP
ROUTES = (
    (re.compile(r"/"), "home.html"),
    (re.compile(r"/chapter/[0-9a-f]{6,32}/?"), "chapter.html"),
    (re.compile(r"/chapter/[0-9a-f]{6,32}/edit/?"), "index.html"),
    (re.compile(r"/chapter/[0-9a-f]{6,32}/read/?"), "reader.html"),
)


class Handler(BaseHTTPRequestHandler):
    app: App = None            # set by serve()
    protocol_version = "HTTP/1.1"
    server_version = "mangaclean/" + __version__

    def log_message(self, fmt, *args):      # keep the console quiet
        pass

    # -- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None,
              cache: bool = False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # versioned image URLs never change, so the browser may keep them
        self.send_header("Cache-Control", "private, max-age=31536000, immutable" if cache else "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, code: int, msg: str):
        self._json({"error": msg}, code)

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("upload too large")
        return self.rfile.read(n) if n else b""

    def _json_body(self) -> dict:
        try:
            data = json.loads(self._body() or b"{}")
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    def _local_only(self) -> bool:
        """Refuse requests that another web site makes this browser send."""
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            o = urllib.parse.urlparse(origin)
            if o.hostname not in ("127.0.0.1", "localhost"):
                return False
        return True

    def _attachment(self, data: bytes, name: str, ctype: str):
        quoted = urllib.parse.quote(name)
        self._send(200, data, ctype, {
            "Content-Disposition": f"attachment; filename=\"download\"; filename*=UTF-8''{quoted}"})

    # -- routing
    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")

    def _route(self, method: str):
        try:
            if not self._local_only():
                return self._error(403, "local requests only")
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            parts = [p for p in url.path.split("/") if p]
            if not parts or parts[0] != "api":
                return self._static(url.path) if method == "GET" else self._error(405, "method not allowed")
            return self._api(method, parts[1:], q)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            try:
                self._error(500, str(exc))
            except Exception:                                 # noqa: BLE001
                pass

    def _static(self, path: str):
        for pattern, page in ROUTES:
            if pattern.fullmatch(path):
                rel = page
                break
        else:
            if path in ("/edit", "/editor", "/index.html"):  # the old single-page address
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            rel = path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if WEB_DIR not in target.parents or not target.is_file():
            return self._error(404, "not found")
        ctype = MIME.get(target.suffix.lower()) or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith(("javascript", "json", "svg+xml")):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _chapter(self, cid: str) -> Optional[Chapter]:
        with self.app.lock:
            return self.app.lib.chapters.get(cid)

    def _upload(self, ch: Optional[Chapter], name: str, data: bytes):
        app = self.app
        added = []
        if name.lower().endswith((".zip", ".cbz")):
            try:
                zf = zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                return self._error(400, f"{name} is not a readable archive")
            with zf:
                names = [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n
                         and Path(n).suffix.lower() in IMAGE_EXTS and not Path(n).name.startswith(".")]
                for n in sorted(names, key=natural_key):
                    try:
                        added.append(app.add(Path(n).name, zf.read(n), ch).summary())
                    except Exception:                         # noqa: BLE001
                        pass
            if not added:
                return self._error(400, f"{name} holds no images this app can read")
        else:
            try:
                added.append(app.add(Path(name).name, data, ch).summary())
            except Exception:                                 # noqa: BLE001
                return self._error(400, f"{name} is not an image this app can read")
        return self._json({"added": added})

    def _api(self, method: str, parts: List[str], q: Dict[str, str]):
        app = self.app

        # ---- the library
        if parts == ["library"] and method == "GET":
            with app.lock:
                chapters = [app.chapter_summary(c) for c in app.lib.chapters.values()]
                series = sorted(set(c["series"] for c in chapters if c["series"]), key=natural_key)
            return self._json({"chapters": chapters, "series": series, "settings": app.settings_json(),
                               "note": app.model_note, "version": __version__})

        if parts == ["chapters"] and method == "POST":
            body = self._json_body()
            with app.lock:
                ch = app.lib.new_chapter(str(body.get("title", "")), str(body.get("series", "")))
                return self._json(app.chapter_detail(ch), 201)

        if len(parts) >= 2 and parts[0] == "chapters":
            ch = self._chapter(parts[1])
            if ch is None:
                return self._error(404, "no such chapter")
            rest = parts[2:]
            if not rest and method == "GET":
                with app.lock:
                    detail = app.chapter_detail(ch)
                detail["settings"] = app.settings_json()
                with app.lock:
                    detail["seriesList"] = sorted(set(c.series for c in app.lib.chapters.values() if c.series),
                                                  key=natural_key)
                detail.update(app.lettering(ch))
                return self._json(detail)
            if not rest and method == "POST":
                body = self._json_body()
                fonts = None
                if "fonts" in body:
                    try:
                        fonts = app.parse_fonts(body["fonts"])
                    except ValueError as exc:
                        return self._error(400, str(exc))
                # the fonts belong to the series: a new series (or a font change) may change them
                before = app.font_ids(app.lib.fonts_for(ch)) if fonts or "series" in body else None
                with app.lock:
                    if isinstance(body.get("order"), list):
                        order = [str(p) for p in body["order"]]
                        if sorted(order) != sorted(ch.pages):
                            return self._error(400, "the new order must list every page of the chapter once")
                        ch.pages = order
                        ch.touch()
                    if "title" in body or "series" in body:
                        app.lib.update_chapter(ch, body.get("title"), body.get("series"))
                    if "notes" in body:
                        app.lib.set_notes(ch, str(body["notes"]))
                    if fonts:
                        app.lib.set_fonts(ch, fonts)
                    after = app.lib.fonts_for(ch)
                relettering = 0
                if before is not None and app.font_ids(after) != before:
                    relettering = app.reletter_chapter(ch, first=str(body.get("first") or "") or None)
                with app.lock:
                    detail = app.chapter_detail(ch)
                detail.update(app.lettering(ch))
                detail["relettering"] = relettering
                return self._json(detail)
            if not rest and method == "DELETE":
                with app.lock:
                    for pid in ch.pages:
                        app._decoded.pop(pid, None)
                    app.lib.delete_chapter(ch)
                return self._json({"ok": True})
            if rest == ["upload"] and method == "POST":
                return self._upload(ch, q.get("name") or "page", self._body())
            if rest == ["translate"] and method == "POST":
                body = self._json_body()
                from .translate import ENGINES
                engine = body.get("engine")
                if engine is not None and engine not in ENGINES:
                    return self._error(400, "unknown engine")
                queued, problem = app.translate_chapter(ch, engine, bool(body.get("redo")))
                if problem:
                    return self._error(409, f"That translator is not available: {problem}.")
                with app.lock:
                    return self._json({"queued": queued, "chapter": app.chapter_detail(ch)})
            if rest == ["download.zip"] and method == "GET":
                what = q.get("what", "english")
                with app.lock:
                    pages = app.lib.chapter_pages(ch)
                data = app.zip_pages(pages, q.get("format", "png"), what, numbered=True)
                stem = re.sub(r'[\\/:*?"<>|]+', " ", f"{ch.series} {ch.title}".strip()).strip() or "chapter"
                suffix = "english" if what == "english" else "cleaned"
                return self._attachment(data, f"{stem} ({suffix}).zip", "application/zip")
            return self._error(404, "not found")

        # ---- the editor's view of a chapter (or of everything)
        if parts == ["state"] and method == "GET":
            with app.lock:
                ch = app.lib.chapters.get(q.get("chapter", "")) if q.get("chapter") else None
                if q.get("chapter") and ch is None:
                    return self._error(404, "no such chapter")
                pages = app.lib.chapter_pages(ch) if ch else [p for c in app.lib.chapters.values()
                                                               for p in app.lib.chapter_pages(c)]
                out = {"pages": [p.summary() for p in pages], "note": app.model_note, "version": __version__,
                       "settings": dict(app.settings_json(), notes=app.lib.notes_for(ch) if ch else "")}
                if ch:
                    out["chapter"] = {"id": ch.id, "title": ch.title, "series": ch.series,
                                      "notes": app.lib.notes_for(ch)}
            if ch:
                out["chapter"].update(app.lettering(ch))
            return self._json(out)

        # ---- fonts
        if parts == ["fonts"] and method == "GET":
            return self._json(app.fonts_json())
        if parts == ["fonts"] and method == "POST":
            try:
                entry = app.add_font(q.get("name") or "font.ttf", self._body())
            except ValueError as exc:
                return self._error(400, str(exc))
            return self._json(dict(app.fonts_json(), font=entry), 201)
        if parts == ["fonts", "preview"] and method == "GET":
            from .typeset import ROLES
            role, fid = q.get("role", "dialogue"), q.get("id", "")
            if role not in ROLES:
                return self._error(400, "unknown style")
            if not app.fontbook.catalog.exists(fid):
                return self._error(404, "no such font")
            try:
                png = app.font_preview(fid, role, q.get("text", "")[:200])
            except LookupError as exc:
                return self._error(409, str(exc))
            return self._send(200, png, "image/png", cache="v" in q)

        if parts == ["settings"] and method == "POST":
            body = self._json_body()
            with app.lock:
                if "autoTranslate" in body:
                    app.auto_translate = bool(body["autoTranslate"])
                from .translate import ENGINES
                if body.get("engine") in ENGINES:
                    app.engine = body["engine"]
                if "notes" in body and body.get("chapter"):
                    ch = app.lib.chapters.get(str(body["chapter"]))
                    if ch is not None:
                        app.lib.set_notes(ch, str(body["notes"]))
                app.save_settings()
                return self._json(app.settings_json())

        if parts == ["upload"] and method == "POST":
            ch = self._chapter(q["chapter"]) if q.get("chapter") else None
            if q.get("chapter") and ch is None:
                return self._error(404, "no such chapter")
            return self._upload(ch, q.get("name") or "page", self._body())

        if parts == ["download.zip"] and method == "GET":
            with app.lock:
                pages = [p for c in app.lib.chapters.values() for p in app.lib.chapter_pages(c)]
            what = q.get("what", "clean")
            data = app.zip_pages(pages, q.get("format", "png"), what, numbered=False)
            zip_name = "english_pages.zip" if what == "english" else "cleaned_pages.zip"
            return self._attachment(data, zip_name, "application/zip")

        # ---- one page
        if len(parts) >= 2 and parts[0] == "page":
            with app.lock:
                ps = app.lib.pages.get(parts[1])
            if ps is None:
                return self._error(404, "no such page")
            rest = parts[2:]
            cache = "v" in q
            if method == "DELETE" and not rest:
                app.delete_page(ps)
                return self._json({"ok": True})
            if method == "GET":
                if not rest:
                    with app.lock:
                        return self._json(ps.detail())
                if rest == ["thumb"]:
                    return self._send(200, app.thumb(ps, q.get("what", "original")), "image/jpeg", cache=cache)
                if rest == ["original"]:
                    src = ps.src
                    with Image.open(io.BytesIO(src)) as im:
                        fmt = (im.format or "").upper()
                        plain = fmt in ("PNG", "JPEG", "WEBP", "GIF") and im.getexif().get(0x0112, 1) == 1
                        if plain:
                            return self._send(200, src, Image.MIME.get(fmt, "application/octet-stream"), cache=True)
                        b = io.BytesIO()
                        ImageOps.exif_transpose(im).convert("RGB").save(b, "PNG", compress_level=1)
                        return self._send(200, b.getvalue(), "image/png", cache=True)
                if rest == ["clean"]:
                    data = ps.clean
                    if data is None:
                        return self._error(409, "not ready")
                    return self._send(200, data, "image/png", cache=cache)
                if rest == ["overlay"]:
                    return self._send(200, app.overlay_png(ps), "image/png")
                if rest == ["english"]:
                    data = ps.english
                    if data is None:
                        return self._error(409, "not translated yet")
                    return self._send(200, data, "image/png", cache=cache)
                if rest == ["download"]:
                    data, fname = app.export(ps, q.get("format", "png"), q.get("what", "clean"))
                    return self._attachment(data, fname, "application/octet-stream")
            if method == "POST":
                if ps.regions is None:
                    return self._error(409, "page is still being analysed")
                body = self._json_body()
                if rest == ["toggle"]:
                    hit = app.toggle(ps, body.get("region"), body.get("x"), body.get("y"), body.get("enabled"))
                    with app.lock:
                        return self._json({"hit": hit, "page": ps.detail()})
                if rest == ["stroke"]:
                    mode = body.get("mode", "erase")
                    if mode not in ("erase", "keep", "inpaint"):
                        return self._error(400, "bad mode")
                    app.stroke(ps, body.get("points") or [], float(body.get("radius", 8)), mode)
                elif rest == ["set"]:
                    app.set_all(ps, body.get("which", "uncertain"), bool(body.get("enabled", True)))
                elif rest == ["translate"]:
                    engine = body.get("engine")
                    from .translate import ENGINES
                    if engine is not None and engine not in ENGINES:
                        return self._error(400, "unknown engine")
                    app.translate(ps, engine)
                elif rest == ["block"]:
                    if body.get("font") and not app.fontbook.catalog.exists(str(body["font"])):
                        return self._error(400, "That font is not on this computer.")
                    if not app.edit_block(ps, body):
                        return self._error(404, "no such block")
                elif rest == ["undo"]:
                    app.undo(ps)
                elif rest == ["reset"]:
                    app.reset(ps)
                else:
                    return self._error(404, "unknown action")
                with app.lock:
                    return self._json({"page": ps.detail()})
        return self._error(404, "not found")


def serve(port: int = 8765, open_browser: bool = True, threads: int = 0, quality: str = "normal",
          library: Optional[Path] = None) -> int:
    Handler.app = App(threads=threads, quality=quality, library=library)
    httpd = None
    for candidate in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        print(f"Could not find a free port near {port}.")
        return 1
    httpd.daemon_threads = True
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    n = len(Handler.app.lib.chapters)
    print(f"Manga Cleaner is running at {url}\n"
          f"Library: {Handler.app.lib.root} ({n} chapter{'s' if n != 1 else ''})\n"
          f"Keep this window open while you work. Press Ctrl+C to quit.", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        httpd.server_close()
    return 0
