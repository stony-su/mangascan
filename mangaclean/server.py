"""Local review app: a tiny web server plus a browser front end.

Everything stays on this machine - the server only listens on 127.0.0.1.
Pages are cleaned automatically as they are added; the browser UI then lets
you switch detected regions on and off, paint extra areas to erase, protect
areas that must stay, compare with the original and download the result.
"""
from __future__ import annotations

import copy
import io
import json
import mimetypes
import queue
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
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .imageio import encode, load_image_bytes, natural_key
from .pipeline import Cleaner, Page

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD = 300 * 1024 * 1024
MAX_UNDO = 30


class PageState:
    def __init__(self, name: str, src: bytes):
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.src = src
        self.meta: dict = {}
        self.width = 0
        self.height = 0
        self.gray = False
        self.thumb: bytes = b""
        self.status = "queued"            # queued | working | done | error
        self.stage = "Waiting"
        self.error: Optional[str] = None
        self.regions: Optional[list] = None
        self.protect: Optional[np.ndarray] = None
        self.clean: Optional[bytes] = None    # PNG of the cleaned page
        self.version = 0
        self.history: List[bytes] = []
        self.seconds = 0.0

    def summary(self) -> dict:
        regs = self.regions or []
        return {
            "id": self.id, "name": self.name, "width": self.width, "height": self.height,
            "status": self.status, "stage": self.stage, "error": self.error,
            "version": self.version, "seconds": round(self.seconds, 1),
            "erased": sum(1 for r in regs if r.enabled),
            "skipped": sum(1 for r in regs if not r.enabled),
            "canUndo": bool(self.history),
        }

    def detail(self) -> dict:
        d = self.summary()
        d["regions"] = [{
            "id": r.id, "bbox": [int(v) for v in r.bbox], "kind": r.kind,
            "enabled": bool(r.enabled), "confident": bool(r.confident), "source": r.source,
        } for r in (self.regions or [])]
        return d


class App:
    def __init__(self, threads: int = 0, quality: str = "normal"):
        self.cleaner = Cleaner(threads=threads, quality=quality, log=self._log)
        self.pages: "OrderedDict[str, PageState]" = OrderedDict()
        self.lock = threading.RLock()
        self.jobs: "queue.Queue" = queue.Queue()
        self._decoded: "OrderedDict[str, tuple]" = OrderedDict()   # small LRU of decoded pages
        self.model_note = ""
        threading.Thread(target=self._worker, daemon=True, name="cleaner").start()
        self.jobs.put(("warmup", None, None))

    # ------------------------------------------------------------ plumbing
    def _log(self, msg: str) -> None:
        self.model_note = msg.strip()
        print(msg, flush=True)

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
        return Page(rgb=rgb, gray=gray, work=work, regions=ps.regions or [], protect=ps.protect)

    def _worker(self) -> None:
        while True:
            kind, ps, done = self.jobs.get()
            try:
                if kind == "warmup":
                    _ = self.cleaner.detector
                    _ = self.cleaner.inpainter
                    self.model_note = ""
                elif ps.id in self.pages:
                    self._run(kind, ps)
            except Exception as exc:                          # noqa: BLE001
                traceback.print_exc()
                if ps is not None:
                    with self.lock:
                        ps.status, ps.error, ps.stage = "error", str(exc), "Failed"
                        ps.version += 1
            finally:
                if done is not None:
                    done.set()

    def _run(self, kind: str, ps: PageState) -> None:
        t0 = time.perf_counter()
        with self.lock:
            ps.status, ps.error = "working", None
            ps.stage = "Finding text" if kind == "analyze" else "Erasing"
        if kind == "analyze":
            rgb, gray, work = self._arrays(ps)
            page = self.cleaner.analyze(rgb)
            with self.lock:
                ps.regions, ps.gray = page.regions, page.gray
                ps.stage = "Erasing"
        page = self._page(ps)
        out = self.cleaner.render(page)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "PNG", compress_level=1)
        with self.lock:
            ps.clean = buf.getvalue()
            ps.status, ps.stage = "done", "Done"
            ps.seconds = time.perf_counter() - t0 if kind == "analyze" else ps.seconds
            ps.version += 1

    def submit(self, kind: str, ps: PageState, wait: float = 0.0) -> None:
        done = threading.Event()
        with self.lock:
            if ps.status != "working":
                ps.status = "queued"
        self.jobs.put((kind, ps, done))
        if wait:
            done.wait(wait)

    # ------------------------------------------------------------- actions
    def add(self, name: str, data: bytes) -> PageState:
        ps = PageState(name, data)
        with Image.open(io.BytesIO(data)) as im:              # validates the upload
            from PIL import ImageOps
            im = ImageOps.exif_transpose(im)
            ps.width, ps.height = im.size
            th = im.convert("RGB")
            th.thumbnail((200, 280))
            b = io.BytesIO()
            th.save(b, "JPEG", quality=80)
            ps.thumb = b.getvalue()
        with self.lock:
            self.pages[ps.id] = ps
        self.submit("analyze", ps)
        return ps

    def _snapshot(self, ps: PageState) -> None:
        state = {
            "enabled": [r.enabled for r in ps.regions or []],
            "manual": [copy.copy(r) for r in ps.regions or [] if r.source == "manual"],
            "protect": None if ps.protect is None else zlib.compress(np.packbits(ps.protect).tobytes(), 1),
        }
        import pickle
        ps.history.append(pickle.dumps(state))
        del ps.history[:-MAX_UNDO]

    def undo(self, ps: PageState) -> None:
        import pickle
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

    def export(self, ps: PageState, fmt: str):
        """(bytes, filename) of the cleaned page in the requested format."""
        if ps.clean is None:
            raise ValueError("page is not ready yet")
        rgb = np.asarray(Image.open(io.BytesIO(ps.clean)).convert("RGB"))
        if not ps.meta:
            self._arrays(ps)
        data, ext = encode(rgb, ps.meta, fmt)
        return data, Path(ps.name).stem + ext


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    app: App = None            # set by serve()
    protocol_version = "HTTP/1.1"
    server_version = "mangaclean/" + __version__

    def log_message(self, fmt, *args):      # keep the console quiet
        pass

    # -- helpers
    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[Dict[str, str]] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
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
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            try:
                self._error(500, str(exc))
            except Exception:                                 # noqa: BLE001
                pass

    def _static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if WEB_DIR not in target.parents or not target.is_file():
            return self._error(404, "not found")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, target.read_bytes(), ctype)

    def _api(self, method: str, parts: List[str], q: Dict[str, str]):
        app = self.app
        if parts == ["state"] and method == "GET":
            with app.lock:
                pages = [p.summary() for p in app.pages.values()]
            return self._json({"pages": pages, "note": app.model_note, "version": __version__})

        if parts == ["upload"] and method == "POST":
            name = q.get("name") or "page"
            data = self._body()
            added = []
            if name.lower().endswith((".zip", ".cbz")):
                with zipfile.ZipFile(io.BytesIO(data)) as zf:
                    names = [n for n in zf.namelist() if not n.endswith("/") and "__MACOSX" not in n
                             and Path(n).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp",
                                                            ".tif", ".tiff", ".gif", ".avif", ".jfif")]
                    for n in sorted(names, key=natural_key):
                        try:
                            added.append(app.add(Path(n).name, zf.read(n)).summary())
                        except Exception:                     # noqa: BLE001
                            pass
            else:
                try:
                    added.append(app.add(Path(name).name, data).summary())
                except Exception:                             # noqa: BLE001
                    return self._error(400, f"{name} is not an image this app can read")
            return self._json({"added": added})

        if parts == ["download.zip"] and method == "GET":
            fmt = q.get("format", "png")
            buf = io.BytesIO()
            used = set()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                with app.lock:
                    pages = [p for p in app.pages.values() if p.clean is not None]
                for p in pages:
                    data, fname = app.export(p, fmt)
                    base, k = fname, 2
                    while fname in used:
                        fname = f"{Path(base).stem}_{k}{Path(base).suffix}"
                        k += 1
                    used.add(fname)
                    zf.writestr(fname, data)
            return self._attachment(buf.getvalue(), "cleaned_pages.zip", "application/zip")

        if len(parts) >= 2 and parts[0] == "page":
            with app.lock:
                ps = app.pages.get(parts[1])
            if ps is None:
                return self._error(404, "no such page")
            rest = parts[2:]
            if method == "DELETE" and not rest:
                with app.lock:
                    app.pages.pop(ps.id, None)
                    app._decoded.pop(ps.id, None)
                return self._json({"ok": True})
            if method == "GET":
                if not rest:
                    with app.lock:
                        return self._json(ps.detail())
                if rest == ["thumb"]:
                    return self._send(200, ps.thumb, "image/jpeg")
                if rest == ["original"]:
                    ctype = Image.MIME.get((ps.meta.get("format") or "").upper()) if ps.meta else None
                    with Image.open(io.BytesIO(ps.src)) as im:
                        fmt = (im.format or "").upper()
                        plain = fmt in ("PNG", "JPEG", "WEBP", "GIF") and im.getexif().get(0x0112, 1) == 1
                        ctype = Image.MIME.get(fmt, "application/octet-stream")
                        if plain:
                            return self._send(200, ps.src, ctype)
                        b = io.BytesIO()
                        from PIL import ImageOps
                        ImageOps.exif_transpose(im).convert("RGB").save(b, "PNG", compress_level=1)
                        return self._send(200, b.getvalue(), "image/png")
                if rest == ["clean"]:
                    if ps.clean is None:
                        return self._error(409, "not ready")
                    return self._send(200, ps.clean, "image/png")
                if rest == ["overlay"]:
                    return self._send(200, app.overlay_png(ps), "image/png")
                if rest == ["download"]:
                    data, fname = app.export(ps, q.get("format", "png"))
                    return self._attachment(data, fname, "application/octet-stream")
            if method == "POST":
                if ps.regions is None:
                    return self._error(409, "page is still being analysed")
                body = json.loads(self._body() or b"{}")
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
                elif rest == ["undo"]:
                    app.undo(ps)
                elif rest == ["reset"]:
                    app.reset(ps)
                else:
                    return self._error(404, "unknown action")
                with app.lock:
                    return self._json({"page": ps.detail()})
        return self._error(404, "not found")


def serve(port: int = 8765, open_browser: bool = True, threads: int = 0, quality: str = "normal") -> int:
    Handler.app = App(threads=threads, quality=quality)
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
    print(f"Manga Cleaner is running at {url}\nKeep this window open while you work. Press Ctrl+C to quit.",
          flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye.")
    finally:
        httpd.server_close()
    return 0
