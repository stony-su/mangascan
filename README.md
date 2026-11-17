# Manga Cleaner

Takes raw (untranslated) manga pages and

1. **cleans** them: removes the Japanese lettering, leaving blank balloons and
   repaired artwork, ready to be typeset;
2. optionally **translates** them: reads the Japanese, translates it to
   English and letters it back into the balloons in a matching style.

Everything runs on your own computer. No page leaves it unless you choose the
Claude translator.

## Start

Double-click **`Manga Cleaner.bat`**. Manga Cleaner opens in your browser as
a small website that runs only on this computer.

1. **Library.** Drop the raw pages of a chapter anywhere on the page
   (images, a folder, or a `.zip` / `.cbz`), or press *New chapter*. Each page
   is cleaned automatically as it arrives.
2. **Chapter.** Give the chapter its title and series, write a few lines
   *About this series*, and press *Translate chapter*. The pages are
   translated in reading order, so each one knows the story so far.
3. **Editor.** Click any page to check it closely and fix what you do not
   like (see *Cleaning a page* and *Translating a page* below).
4. **Read** the finished chapter in the built-in reader, or download it as a
   `.zip` of English or cleaned pages.

Keep the black console window open while you work; closing it quits the app.
Everything is saved as you go, so your chapters are still there next time.

## The library

- **Series.** Chapters with the same series name are grouped together and
  share one set of notes, so names and tone stay the same from chapter to
  chapter. Each chapter page links to the previous and next chapter.
- **Chapter page.** Drag pages to put them in order (or select a page and
  press `Alt` + an arrow key). Add more pages at any time, translate the
  whole chapter, download it, or delete it.
- **Reader.** Reads right to left like a printed manga (`D` switches the
  direction), one page at a time or as one long scroll (`M`). Switch between
  the English, cleaned and original pages with `V`, and go full screen with
  `F`. Turn pages with the arrow keys, the space bar, a click on either side
  of the page, or a swipe. A click or tap in the middle of the page hides the
  controls. The reader remembers where you stopped.
- **Where it is kept.** In the `library` folder next to `Manga Cleaner.bat`,
  one folder per chapter holding the original pages, the cleaned and English
  versions, and your edits. Back that folder up to keep your work. To keep the
  library somewhere else, start the app with `--library "D:\Manga\library"`
  or set the `MANGACLEAN_LIBRARY` environment variable.
- **Private.** The site listens only on this computer (`127.0.0.1`) and
  turns away requests that other websites open in your browser try to send
  it.

## Cleaning a page

| Tool | Key | What it does |
| --- | --- | --- |
| Pick | `V` | Click any region to switch its erasing on or off. |
| Erase | `E` | Paint over something that should also be removed. |
| Keep | `K` | Paint over something that must be left exactly as it was. |

Views: **Cleaned** (`1`), **Original** (`2`), **Detected** (`3`), **English**
(`4`). Hold `C` to peek at the original. `Ctrl+Z` undoes the last cleaning
change.

**Dashed orange boxes** mark things the detector found but was not sure
about, usually stylised logos and big sound effects. They are left alone by
default, because half-erasing a logo looks worse than leaving it. Click one
to erase it, or use *Also erase the uncertain regions*.

## Translating a page

The **Translate** tab lists every block of text on the page: the Japanese as
it was read, and the English that was lettered. For each block you can

- rewrite the English (`Ctrl+Enter` or click away to re-letter it),
- change the lettering style or make it larger or smaller,
- tick *leave blank* to keep that balloon empty.

Clicking a balloon on the page jumps to its line in the list. Blocks the
reader could not make out are outlined in dashed orange and left blank for
you to fill in; nothing is ever lettered from a guess.

If you bring a region's Japanese back in the Clean tab, its English
disappears with it. *Translate this page again* keeps the lines you edited.

### The translators

A page can only be translated well with its context: who is speaking, to
whom, what just happened, and how these characters talk. The Claude
translators **localise** rather than translate word for word. Each page is
handled in one go, with the picture, your notes on the series, the names
already settled and the story of the previous pages. Every line is written
the way an English-language writer would have given it to that character.

| | Claude (Claude Code login) | Claude (API key) | On this computer |
| --- | --- | --- | --- |
| Quality | Localised, consistent across pages. Also reads lettering the local reader could not. | Same. | Literal, balloon by balloon: a rough draft to rewrite. |
| Privacy | Each page (image and text) is sent to Anthropic. | Same. | Nothing leaves the computer. |
| Cost | Counts toward your Claude usage (about $0.07 of usage per page with Opus). | Billed to your API account. | Free. |
| Setup | None if Claude Code is installed and logged in. Chosen by default when it is. | Set `ANTHROPIC_API_KEY` (or run `ant auth login`) and restart the app. | None. |

Both Claude translators use Claude Opus 5.5 (`claude-opus-5-5`). On the API,
a request a safety classifier declines is re-run on Anthropic's
recommended fallback model (the `fallbacks` option). A page that still
cannot be translated is reported as such. If your Claude usage limit is
reached, the app says when it resets; the offline translator keeps working
meanwhile. For more thorough (slower) localisation, set the environment
variable `MANGACLEAN_CLAUDE_EFFORT=high` before starting the app.

**About this series.** On the chapter page (or in the editor's Translate
tab), tell the translator what it cannot see on one page: the setting and
tone, who is who, and how names should be spelled. For example: "a
historical war drama; soldiers talk rough, the king speaks formally; spell
the hero's name Shin". The notes belong to the series, so every chapter of
it uses them. When working without the app, put the same text in a
`notes.txt` inside the chapter folder, or pass `--notes`.

Translate pages in reading order (*Translate chapter* does this for you).
Each page then knows the story so far, including the end of the previous
chapter of the series, and names stay spelled the way they were settled on
earlier pages.

### How the lettering is chosen

- **Where:** inside the balloon the Japanese sat in, shaped to the balloon
  (lines are longer where it is wider). Without a balloon, where the
  original was, without crossing a panel border.
- **Size:** in proportion to the original, so loud text stays loud and
  asides stay small. Ordinary dialogue is lettered at one size across the
  page; shouted text grows to fill its balloon the way the original did.
- **Style:** the usual English-manga conventions. Balloon dialogue is set in
  an upright comic face in capitals, and oversized balloon text in a shout
  face. Captions and lines set on the artwork use the italic of the
  dialogue face, small asides a handwriting face, and big titles a heavy
  display face. The ink colour is copied (white on black stays white on
  black, grey screentoned lettering prints solid black). Text that was
  outlined over artwork is outlined again.
- **Fitting:** like a letterer, it first shrinks the text a little, then
  condenses the letters (to 88%), then shrinks further. Only then does it
  break a word. Names are never broken by English rules; at the very worst
  they are split between syllables (KARYO-/TEN).
- A tall single column outside a balloon is lettered sideways, reading
  downwards, and stays inside its strip.

### Fonts

**What the professionals use.** Licensed English editions are lettered in
commercial comic fonts. The most common is Comicraft's *Wild Words*,
described by a professional manga letterer as "probably the most prolific
font used in localized manga" (My Hero Academia, Blue Exorcist, Goodnight
Punpun). Others include Comicraft's *Comicrazy* (Demon Slayer), *Meanwhile*
(Jujutsu Kaisen), *Samaritan*, *Joe Kubert* and *Yada Yada Yada*, and
Blambot's *Anime Ace*, *Blambot Classic* (Chainsaw Man), *Back Issues* and
*Imaginary Friend* (Attack on Titan). Scanlation groups have long used
*Wild Words* and *Anime Ace* for dialogue, with a separate face for small
asides.

**Why they are not included.** All of these are paid fonts (Comicraft's
cost $35 to $395). Blambot's free licence covers independent comics, not
mainstream manga, and bundling them inside software needs a paid embedding
licence. They cannot be shipped or downloaded by this program.

**If you own one, it is used automatically.** Install the font in Windows,
or drop the file into the `fonts` folder. The app looks for Wild Words,
Anime Ace, Comicrazy, Meanwhile, Samaritan, Blambot Classic, Back Issues and
the other families above. It letters dialogue with the first one it finds,
and captions with its italic cut if you have that too.

**Otherwise: open-licensed defaults** (SIL Open Font License, downloaded on
first use). They were chosen by lettering real pages with 26 free comic
fonts side by side, measuring how large each could set the same balloons,
and judging the look against professional lettering.

| Style | Font | Why |
| --- | --- | --- |
| Dialogue | Balsamiq Sans Bold | Upright, heavy and compact like professional manga dialogue; fits narrow balloons; has ō/ū for names. |
| Captions | Balsamiq Sans Bold Italic | The usual pairing: the dialogue face in italics. |
| Shouts | Bangers | A classic condensed comic shout face. |
| Titles | Anton | Heavy display face for big captions and credits. |
| Asides | Patrick Hand | Handwritten look for small notes on the art. |

**Choosing fonts yourself.** Put the files in the `fonts` folder and name
them in `fonts/fonts.json`. For a variable font, add the weight after an
`@`:

```json
{ "dialogue": "CCWildWords-Regular.ttf", "narration": "CCWildWords-Italic.ttf",
  "shout": "MyLoudFont.otf@800" }
```

The styles are `dialogue`, `shout`, `title`, `narration` (captions) and
`handwritten` (asides). In comic fonts whose lower case is a second set of
capitals, the crossbar "I" is kept for the pronoun only, the way letterers
use it.

## Without the app

Drag a folder (or images, or a `.zip` / `.cbz`) onto

- **`Clean Folder.bat`** to clean the pages into a `cleaned` folder, or
- **`Translate Folder.bat`** to clean, translate and letter them into a
  `translated` folder, with a `.txt` script of every page.

From a terminal:

```
.venv\Scripts\python -m mangaclean "D:\raws\chapter 12"
.venv\Scripts\python -m mangaclean "D:\raws\chapter 12" --translate --script
.venv\Scripts\python -m mangaclean page01.jpg --translate offline -o out
.venv\Scripts\python -m mangaclean --help
```

| Option | Meaning |
| --- | --- |
| `-o DIR` | Where to write the results. |
| `-t`, `--translate [ENGINE]` | Also translate and letter the pages. ENGINE is `claude-code`, `claude` or `offline`; the default picks Claude Code when it is installed. |
| `--notes TEXT`, `--notes-file FILE` | With `--translate`: what the translator should know about the series. A `notes.txt` inside a folder you translate is used automatically. |
| `--script` | With `--translate`: save the Japanese and English of every block as text. |
| `-f png\|jpg\|webp\|same` | Output format. PNG (lossless) is the default. |
| `-r` | Include sub-folders. |
| `--all` | Also erase the uncertain regions (logos, sound effects). |
| `--quality fast\|normal\|high` | Resolution the redraw model works at. |
| `--grow 1.5` | Widen the margin around lettering if faint outlines remain. |
| `--mask` | Also save a black/white mask of what was erased. |
| `--debug` | Also save a picture showing what was detected. |

## What to expect

- **Cleaning balloons, captions, margin text:** clean. This is most of the
  lettering on most pages.
- **Cleaning lettering over artwork:** removed, and the art underneath is a
  plausible reconstruction, not the real drawing. Over regular screentone, or
  where the lettering covered a face or a detailed object, expect to touch it
  up by hand.
- **Reading:** ordinary balloon text is read reliably. Small outlined text
  over artwork, dense advertising strips and rare name kanji are where it
  goes wrong; unreadable blocks are left blank rather than guessed.
- **Translation:** with Claude, natural English that reads as if it was
  written that way, consistent from page to page. Proofread it as you would
  any translator's work. The offline translator gives a literal draft to
  rewrite. Sound effects and logos are not translated.
- **Speed** on a laptop CPU: a few seconds to clean a page, plus roughly
  30 seconds to read, translate (about 15 of them with Claude) and letter it.

## Requirements

Windows with Python 3.10 or newer. The first start installs the packages
listed in `requirements.txt` into a private `.venv` folder here. The model
files (about 750 MB in all) live in `models`; missing ones are downloaded on
first use.

## How it works

1. **Find the lettering.** `comic-text-detector` produces a per-pixel text
   probability map plus text blocks and text lines.
2. **Decide how to erase each region.** Lettering enclosed by one flat colour
   (a balloon, a box, the margin, white or black) is painted with that
   colour. Outlined lettering, screentone and artwork are redrawn by LaMa, an
   inpainting model fine-tuned on manga.
3. **Read.** `manga-ocr` reads each text block from the original page and
   reports how sure it is.
4. **Translate.** Claude localises the whole page from the picture, the
   text, your series notes, the settled names and the story so far, and
   writes down what happens on the page for the next page's translator.
   Without Claude, FuguMT translates each balloon on this computer.
5. **Letter.** The balloon interior is measured on the cleaned page and the
   words are fitted to its shape at the largest size that suits the original.

Font research sources: Sara Linsley's professional
[lettering wiki](https://github.com/saraoswald/lettering-tutorials/wiki/Fonts)
and [font examples](https://github.com/saraoswald/lettering-tutorials/wiki/Font-Examples),
Kyla Aiko's [Manga Font Directory](https://mangafonts.carrd.co/) of fonts
used in licensed English manga, the scanlation guides
[The Complete Idiot's Guide to Editing Manga](https://www.insidescanlation.com/etc/The-Idiots-Guide-to-Editing-Manga/guide/type/fonts)
and [Vorbis's typesetting guide](https://www.insidescanlation.com/etc/Vorbis-Typesetting-Guide/fonts.htm),
and [Blambot's licensing terms](https://blambot.com/pages/licensing).

## Credits

- [comic-text-detector](https://github.com/dmMaze/comic-text-detector) by dmMaze
  (ONNX export by mayocream), Apache-2.0.
- [LaMa](https://github.com/advimman/lama) inpainting, manga fine-tune
  [AnimeMangaInpainting](https://huggingface.co/dreMaz/AnimeMangaInpainting)
  (ONNX export by ogkalu), Apache-2.0.
- [manga-ocr](https://github.com/kha-white/manga-ocr) by kha-white, 2025
  retrain by jzhang533 (ONNX export by l0wgear), Apache-2.0.
- [FuguMT](https://huggingface.co/staka/fugumt-ja-en) by staka (ONNX export
  by Kadonox), CC BY-SA 4.0.
- Fonts from [Google Fonts](https://github.com/google/fonts): Balsamiq Sans,
  Bangers, Anton and Patrick Hand, SIL Open Font License 1.1.
