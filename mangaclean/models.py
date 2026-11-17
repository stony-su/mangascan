"""Model and font files: where they live and how to fetch them.

Everything is downloaded once, on first use, and verified against a pinned
SHA-256.  Licences:

* comic-text-detector (dmMaze), ONNX export by mayocream            Apache-2.0
* LaMa fine-tuned on anime/manga (AnimeMangaInpainting), ONNX by ogkalu  Apache-2.0
* manga-ocr (kha-white), 2025 retrain by jzhang533, ONNX by l0wgear  Apache-2.0
* FuguMT ja-en (staka), ONNX export by Kadonox                       CC BY-SA 4.0
* Comic Neue, Bangers, Anton, Crimson Text, Patrick Hand (Google Fonts)  SIL OFL 1.1
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Optional

HF = "https://huggingface.co/{repo}/resolve/main/{path}"
GF = "https://raw.githubusercontent.com/google/fonts/main/{path}"


def _f(file: str, url: str, sha256: str, size: int) -> dict:
    return {"file": file, "url": url, "sha256": sha256, "size": size}


GROUPS: Dict[str, list] = {
    "detector": [_f(
        "comic-text-detector.onnx",
        HF.format(repo="mayocream/comic-text-detector-onnx", path="comic-text-detector.onnx"),
        "1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f", 94669756)],
    "inpainter": [_f(
        "lama-manga-dynamic.onnx",
        HF.format(repo="ogkalu/lama-manga-onnx-dynamic", path="lama-manga-dynamic.onnx"),
        "de31ffa5ba26916b8ea35319f6c12151ff9654d4261bccf0583a69bb095315f9", 206291843)],
    "ocr": [
        _f("manga-ocr/encoder_model.onnx",
           HF.format(repo="l0wgear/manga-ocr-2025-onnx", path="encoder_model.onnx"),
           "f87668ae0f62d6f032dac6b213e8c0fea84cd15895ac8cab624cc9a2f49d4a27", 22356885),
        _f("manga-ocr/decoder_model.onnx",
           HF.format(repo="l0wgear/manga-ocr-2025-onnx", path="decoder_model.onnx"),
           "6b1fb216d542c4b2a4fa5b9d7ae3522081eb85fb959d2cecd28055af956a8a5e", 118053454),
        _f("manga-ocr/vocab.txt",
           HF.format(repo="l0wgear/manga-ocr-2025-onnx", path="vocab.txt"),
           "344fbb6b8bf18c57839e924e2c9365434697e0227fac00b88bb4899b78aa594d", 24072),
    ],
    "mt": [
        _f("fugumt-ja-en/encoder_model.onnx",
           HF.format(repo="Kadonox/fugumt-ja-en-onnx", path="onnx/encoder_model.onnx"),
           "ce16e5f7cf1fde5dffbae8ed5a9c8d915de8084edc34ce5549a8e7c7271a5d2e", 142366928),
        _f("fugumt-ja-en/decoder_model.onnx",
           HF.format(repo="Kadonox/fugumt-ja-en-onnx", path="onnx/decoder_model.onnx"),
           "f3fef884dc2bcaba1e3a6f844261ad31be84b433487156d2c11c0e8b297c15ae", 167896798),
        _f("fugumt-ja-en/source.spm",
           HF.format(repo="Kadonox/fugumt-ja-en-onnx", path="source.spm"),
           "2b75a55d81ba5be08d5cc971eecbb03d8efaf66382f2a4f9f0a5fe2fca2b6741", 796876),
        _f("fugumt-ja-en/vocab.json",
           HF.format(repo="Kadonox/fugumt-ja-en-onnx", path="vocab.json"),
           "bd868a2a41f8e2fd818fd575bb3f8fa5f1986859017de2badd2aa13971ef8769", 861152),
    ],
}

# Default lettering fonts (all SIL Open Font Licence).  file -> path in google/fonts
FONTS: Dict[str, dict] = {
    "ComicNeue-Bold.ttf": _f("ComicNeue-Bold.ttf", GF.format(path="ofl/comicneue/ComicNeue-Bold.ttf"),
        "3e7e5fccfd7e0788f317b43312151c1bd5cf058c9697a8d83eac3939050bd61e", 55716),
    "ComicNeue-BoldItalic.ttf": _f("ComicNeue-BoldItalic.ttf", GF.format(path="ofl/comicneue/ComicNeue-BoldItalic.ttf"),
        "5c312c2a2fa64eee82f3b87fcfab8f3b12a5e59b043124401d322eb323cfbf16", 55928),
    "Bangers-Regular.ttf": _f("Bangers-Regular.ttf", GF.format(path="ofl/bangers/Bangers-Regular.ttf"),
        "4160a7311de9342674cce9160cde9fcbb30f48190397d86ff1b70b455af65824", 93148),
    "Anton-Regular.ttf": _f("Anton-Regular.ttf", GF.format(path="ofl/anton/Anton-Regular.ttf"),
        "a4ba3a92350ebb031da0cb47630ac49eb265082ca1bc0450442f4a83ab947cab", 170812),
    "CrimsonText-Bold.ttf": _f("CrimsonText-Bold.ttf", GF.format(path="ofl/crimsontext/CrimsonText-Bold.ttf"),
        "a3a0765fc5e8d0b49b540a23aefe0184887dd79f06a0bdf4db7035cea6befa93", 112796),
    "PatrickHand-Regular.ttf": _f("PatrickHand-Regular.ttf", GF.format(path="ofl/patrickhand/PatrickHand-Regular.ttf"),
        "0f173b3e6cb6d1af25babf7f0057c5ac4ee11f9992b0469bb817e967ef4ad0fc", 214772),
}

# kept for callers that only know the single-file names
MODELS = {name: files[0] for name, files in GROUPS.items() if len(files) == 1}


def _log_default(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def project_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    env = os.environ.get("MANGACLEAN_MODELS")
    return Path(env) if env else project_dir() / "models"


def default_fonts_dir() -> Path:
    env = os.environ.get("MANGACLEAN_FONTS")
    return Path(env) if env else project_dir() / "fonts"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(spec: dict, folder: Path, log: Callable[[str], None] = _log_default) -> Path:
    """Return the local path of one registry file, downloading it if needed."""
    path = Path(folder) / spec["file"]
    if path.exists() and path.stat().st_size == spec["size"]:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    big = spec["size"] > 5_000_000
    if big:
        log(f"Downloading {Path(spec['file']).name} ({spec['size'] / 1e6:.0f} MB) - first use only ...")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "mangaclean"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        done, last = 0, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(done * 100 / max(1, spec["size"]))
            if big and pct // 10 != last:
                last = pct // 10
                log(f"  {pct:3d}%")
    ok = tmp.stat().st_size == spec["size"]
    if ok and spec["sha256"]:
        ok = _sha256(tmp) == spec["sha256"]
    if not ok:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{spec['file']}: the download is incomplete or corrupt. Please try again.")
    os.replace(tmp, path)
    return path


def ensure_group(name: str, models_dir: Optional[Path] = None,
                 log: Callable[[str], None] = _log_default) -> Dict[str, Path]:
    """Download every file of a model group; returns {file name: path}."""
    folder = Path(models_dir) if models_dir else default_models_dir()
    return {Path(spec["file"]).name: fetch(spec, folder, log) for spec in GROUPS[name]}


def ensure_model(name: str, models_dir: Optional[Path] = None,
                 log: Callable[[str], None] = _log_default) -> Path:
    """Path of a single-file model, downloading it on first use."""
    folder = Path(models_dir) if models_dir else default_models_dir()
    return fetch(GROUPS[name][0], folder, log)


def ensure_font(file: str, fonts_dir: Optional[Path] = None,
                log: Callable[[str], None] = _log_default) -> Optional[Path]:
    """Path of a lettering font: a file the user put in the fonts folder, or
    one of the bundled defaults (downloaded on first use)."""
    folder = Path(fonts_dir) if fonts_dir else default_fonts_dir()
    path = folder / file
    if path.exists():
        return path
    spec = FONTS.get(file)
    if spec is None:
        return None
    try:
        return fetch(spec, folder, log)
    except Exception as exc:                       # offline: caller falls back to a system font
        log(f"Could not download the font {file}: {exc}")
        return None
