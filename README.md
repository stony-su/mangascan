# Manga Cleaner

Takes raw (untranslated) manga pages and removes the lettering, leaving blank
pages that are ready to be typeset in another language.

- Lettering in speech balloons, caption boxes and page margins is wiped with
  the exact background colour, so balloons come out perfectly clean and their
  outlines are never touched.
- Lettering that sits on top of artwork is redrawn by an inpainting model
  trained on manga.
- Everything runs on your own computer. No page ever leaves it.

## Start

Double-click **`Manga Cleaner.bat`**. The app opens in your browser.

1. Drop pages onto the window (images, a folder, or a `.zip` / `.cbz`).
2. Each page is cleaned automatically. A tick appears on its thumbnail.
3. Check the result, fix anything you do not like, and download.

Keep the black console window open while you work; closing it quits the app.

## Reviewing a page

| Tool | Key | What it does |
| --- | --- | --- |
| Pick | `V` | Click any region to switch its erasing on or off. |
| Erase | `E` | Paint over something that should also be removed. |
| Keep | `K` | Paint over something that must be left exactly as it was. |

Views: **Cleaned** (`1`), **Original** (`2`), **Detected** (`3`). Hold `C` to
peek at the original. `Ctrl+Z` undoes the last change.

**Dashed orange boxes** mark things the detector found but was not sure
about, usually stylised logos and big sound effects. They are left alone by
default, because half-erasing a logo looks worse than leaving it. Click one
to erase it, or use *Also erase the uncertain regions*.

In the **Detected** view, green areas are wiped with the background colour,
red areas are redrawn by the model, orange was found but left alone, and blue
is what you protected.

## Cleaning a whole folder without the app

Drag a folder (or images, or a `.zip` / `.cbz`) onto **`Clean Folder.bat`**.
The cleaned pages are written to a `cleaned` folder next to the input.

From a terminal:

```
.venv\Scripts\python -m mangaclean "D:\raws\chapter 12"
.venv\Scripts\python -m mangaclean page01.jpg page02.jpg -o out
.venv\Scripts\python -m mangaclean --help
```

Useful options:

| Option | Meaning |
| --- | --- |
| `-o DIR` | Where to write the results. |
| `-f png\|jpg\|webp\|same` | Output format. PNG (lossless) is the default. |
| `-r` | Include sub-folders. |
| `--all` | Also erase the uncertain regions (logos, sound effects). |
| `--quality fast\|normal\|high` | Resolution the redraw model works at. |
| `--grow 1.5` | Widen the margin around lettering if faint outlines remain. |
| `--mask` | Also save a black/white mask of what was erased. |
| `--debug` | Also save a picture showing what was detected. |

## What to expect

- **Balloons, captions, margin text:** clean. This is most of the lettering on
  most pages.
- **Lettering over artwork:** removed, and the art underneath is a plausible
  reconstruction, not the real drawing. Over hatching and gradients it is
  usually hard to spot. Over regular screentone, or where the lettering covered
  a face or a detailed object, expect to touch it up by hand.
- **Logos and large sound effects** are often only partly detected, so they are
  left alone unless you switch them on.
- **Speed** on a laptop CPU: about 2 seconds to find the lettering on a page,
  plus 2 to 10 seconds for each block of lettering that has to be redrawn.
  A page with only balloon text takes about 2 seconds.
- Normal pages, double-page spreads and long vertical strips are all handled.
  Output keeps the size, colour mode and transparency of the source.

## Requirements

Windows with Python 3.10 or newer. The first start installs four packages
into a private `.venv` folder here. The two model files (about 300 MB) live in
`models`; if they are missing they are downloaded on first use.

## How it works

1. **Find the lettering.** `comic-text-detector` produces a per-pixel text
   probability map plus text blocks and text lines.
2. **Group and judge it.** Text pixels are grouped into regions. A region
   backed by a detected block or line is erased; a region that is not is
   marked uncertain.
3. **Decide how to erase each region.** If the lettering is enclosed by one
   flat colour (a balloon, a box, the margin, white or black), it is painted
   with that colour. A label box carrying opposite-coloured text keeps its
   box. Outlined lettering, screentone and artwork are sent to the model.
4. **Redraw.** LaMa, fine-tuned on manga, fills in each remaining area using a
   crop of the surrounding art.

## Credits

- [comic-text-detector](https://github.com/dmMaze/comic-text-detector) by dmMaze
  (ONNX export by mayocream), Apache-2.0.
- [LaMa](https://github.com/advimman/lama) inpainting, manga fine-tune
  [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
  (ONNX export by ogkalu), Apache-2.0.
