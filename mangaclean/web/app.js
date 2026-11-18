'use strict';
/* Manga Cleaner - browser front end.  Talks to the local Python server. */

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');
const IMAGE_RE = /\.(png|jpe?g|jfif|webp|bmp|tiff?|gif|avif|zip|cbz)$/i;

/* The editor always works inside one chapter of the library. */
const CHAPTER = (location.pathname.match(/^\/chapter\/([0-9a-f]+)\/edit/) || [])[1] || null;
const WANT_PAGE = new URLSearchParams(location.search).get('page');
if (!CHAPTER) location.replace('/');

const S = {
  pages: [],                 // summaries from the server
  note: '',
  current: null,             // id of the page on screen
  detail: null,              // regions etc. of that page
  shown: { id: null, version: -1 },
  img: { orig: null, clean: null, overlay: null, english: null },
  view: 'clean',             // clean | original | detected | english
  tab: 'clean',              // clean | translate   (right-hand panel)
  settings: { autoTranslate: false, engine: 'offline', notes: '', engines: {} },
  chapter: null,             // title, series and fonts of the chapter
  selBlock: null,            // id of the text block being edited
  blocksSig: '',
  tool: 'select',            // select | erase | keep
  brush: 18,                 // radius, in page pixels
  zoom: 1, ox: 0, oy: 0, fitted: true,
  mouse: null,               // last pointer position on the canvas (css px)
  hover: null,               // region under the pointer
  peek: false, space: false,
  stroke: null,              // brush stroke in progress
  pending: 0,                // edits waiting for the server
  uploading: null,           // {done, total}
  offline: false,
};

/* ------------------------------------------------------------- server */
async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (_) { /* not json */ }
  if (!res.ok) {
    const err = new Error(data.error || res.statusText || 'request failed');
    err.status = res.status;
    throw err;
  }
  return data;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('could not load ' + url));
    im.src = url;
  });
}

const cur = () => S.pages.find((p) => p.id === S.current) || null;
const busyPage = (p) => p && (p.status === 'queued' || p.status === 'working');
const canEdit = () => { const p = cur(); return !!p && p.status === 'done' && !!S.detail && S.pending === 0; };

let pollTimer = null;
let stateSig = '';
async function poll() {
  clearTimeout(pollTimer);
  try {
    const st = await api(`/api/state?chapter=${CHAPTER}`);
    if (st.chapter) {
      S.chapter = st.chapter;
      document.title = `${st.chapter.title} - Editor - Manga Cleaner`;
      $('crumbChapter').textContent = st.chapter.title;
      $('crumbChapter').href = `/chapter/${CHAPTER}`;
      $('readLink').href = `/chapter/${CHAPTER}/read`;
    }
    S.offline = false;
    const sig = JSON.stringify(st);
    if (sig !== stateSig) {
      stateSig = sig;
      S.pages = st.pages;
      S.note = st.note || '';
      if (st.settings) S.settings = st.settings;
      renderList();
      await syncCurrent();
    }
  } catch (err) {
    if (err.status === 404) { location.href = '/'; return; }
    S.offline = true;
  }
  updateUI();
  pollTimer = setTimeout(poll, S.pages.some(busyPage) || S.uploading ? 600 : 2000);
}

/* -------------------------------------------------------- current page */
let loadToken = 0;
async function syncCurrent() {
  if (!S.pages.length) { S.current = null; }
  else if (!cur()) {
    S.current = WANT_PAGE && S.pages.some((p) => p.id === WANT_PAGE) && !S.shown.id ? WANT_PAGE : S.pages[0].id;
  }
  const p = cur();
  const token = ++loadToken;
  if (!p) {
    S.detail = null; S.shown = { id: null, version: -1 };
    S.img = { orig: null, clean: null, overlay: null, english: null };
    renderBlocks(); draw(); return;
  }
  if (S.shown.id !== p.id) {
    S.shown = { id: p.id, version: -1 };
    S.detail = null; S.hover = null; S.stroke = null; S.selBlock = null;
    S.img = { orig: null, clean: null, overlay: null, english: null };
    S.fitted = true;
    renderBlocks(); draw();
  }
  try {
    if (!S.img.orig) {
      const im = await loadImage(`/api/page/${p.id}/original`);
      if (token !== loadToken) return;
      S.img.orig = im;
      if (S.fitted) fit();
      draw();
    }
    if (p.status === 'done' && S.shown.version !== p.version) {
      const [detail, clean] = await Promise.all([
        api(`/api/page/${p.id}`),
        loadImage(`/api/page/${p.id}/clean?v=${p.version}`),
      ]);
      if (token !== loadToken) return;
      const hadEnglish = !!S.img.english;
      S.detail = detail; S.img.clean = clean; S.img.overlay = null; S.img.english = null;
      S.shown.version = detail.version;
      if (S.view === 'detected') loadOverlay();
      if (detail.translated) {
        const eng = await loadImage(`/api/page/${p.id}/english?v=${detail.version}`);
        if (token !== loadToken) return;
        S.img.english = eng;
        if (!hadEnglish && S.tab === 'translate') setView('english');
      } else if (S.view === 'english') setView('clean');
      renderBlocks();
    }
  } catch (err) {
    if (token === loadToken) toast(err.message);
  }
  draw(); updateUI();
}

async function loadOverlay() {
  const p = cur();
  if (!p || !S.detail) return;
  const version = S.detail.version;
  try {
    const im = await loadImage(`/api/page/${p.id}/overlay?v=${version}`);
    if (cur() === p && S.detail && S.detail.version === version) { S.img.overlay = im; draw(); }
  } catch (_) { /* cosmetic only */ }
}

function selectPage(id) {
  history.replaceState(null, '', `${location.pathname}?page=${id}`);
  if (S.current === id) return;
  S.current = id;
  renderList();
  syncCurrent();
}

/* Send an edit, wait for the re-rendered page. */
async function edit(action, body) {
  const p = cur();
  if (!p || !canEdit()) return null;
  S.pending++; updateUI();
  try {
    const res = await api(`/api/page/${p.id}/${action}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
    });
    const i = S.pages.findIndex((q) => q.id === p.id);
    if (i >= 0 && res.page) S.pages[i] = Object.assign({}, S.pages[i], res.page, { regions: undefined });
    stateSig = '';
    return res;
  } catch (err) {
    toast(err.message); return null;
  } finally {
    S.pending--;
    renderList();
    await syncCurrent();
  }
}

/* ---------------------------------------------------------------- view */
function resizeCanvas() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { w: r.width, h: r.height, dpr };
}

function pageSize() {
  const p = cur();
  if (S.img.orig) return { w: S.img.orig.naturalWidth, h: S.img.orig.naturalHeight };
  return p ? { w: p.width, h: p.height } : null;
}

function fit() {
  const sz = pageSize();
  if (!sz) return;
  const { w, h } = resizeCanvas();
  const pad = 20;
  S.zoom = Math.max(0.02, Math.min((w - 2 * pad) / sz.w, (h - 2 * pad) / sz.h));
  S.ox = (w - sz.w * S.zoom) / 2;
  S.oy = (h - sz.h * S.zoom) / 2;
  S.fitted = true;
}

function zoomAt(cx, cy, factor) {
  const z = Math.min(16, Math.max(0.05, S.zoom * factor));
  S.ox = cx - (cx - S.ox) * (z / S.zoom);
  S.oy = cy - (cy - S.oy) * (z / S.zoom);
  S.zoom = z; S.fitted = false;
  draw(); updateStatus();
}

const toPage = (pt) => [(pt.x - S.ox) / S.zoom, (pt.y - S.oy) / S.zoom];

function regionAt(x, y) {
  let best = null, bestArea = Infinity;
  for (const r of (S.detail && S.detail.regions) || []) {
    const [x1, y1, x2, y2] = r.bbox;
    if (x >= x1 - 4 && x < x2 + 4 && y >= y1 - 4 && y < y2 + 4) {
      const a = (x2 - x1) * (y2 - y1);
      if (a < bestArea) { best = r; bestArea = a; }
    }
  }
  return best;
}

function draw() {
  const { w, h, dpr } = resizeCanvas();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const p = cur();
  if (!p || !S.img.orig) return;

  const english = S.view === 'english' && S.img.english;
  const showOriginal = S.peek || !(S.view === 'clean' || english) || !S.img.clean;
  const img = showOriginal ? S.img.orig : english ? S.img.english : S.img.clean;
  ctx.save();
  ctx.translate(S.ox, S.oy);
  ctx.scale(S.zoom, S.zoom);
  ctx.imageSmoothingEnabled = S.zoom < 1.5;
  ctx.imageSmoothingQuality = 'high';
  ctx.shadowColor = 'rgba(0,0,0,.45)'; ctx.shadowBlur = 18 / S.zoom;
  ctx.drawImage(img, 0, 0);
  ctx.shadowColor = 'transparent';
  if (S.view === 'detected' && !S.peek && S.img.overlay) ctx.drawImage(S.img.overlay, 0, 0);

  const px = 1 / S.zoom;                     // one screen pixel, in page units
  if (S.view === 'english' && S.detail && !S.peek) {
    for (const b of S.detail.blocks || []) {  // the block being edited, and blocks still blank
      const sel = b.id === S.selBlock;
      if (!sel && (b.lettered || !b.enabled)) continue;
      const [x1, y1, x2, y2] = b.box;
      ctx.lineWidth = (sel ? 2 : 1.5) * px;
      ctx.setLineDash(sel ? [] : [5 * px, 4 * px]);
      ctx.strokeStyle = sel ? 'rgba(80,160,255,.95)' : 'rgba(255,170,0,.95)';
      ctx.strokeRect(x1 - 4 * px, y1 - 4 * px, x2 - x1 + 8 * px, y2 - y1 + 8 * px);
    }
    ctx.setLineDash([]);
  } else if (S.detail && !S.peek && S.view !== 'original') {
    ctx.lineWidth = 1.5 * px;
    for (const r of S.detail.regions) {      // things that were found but left alone
      if (r.enabled) continue;
      const [x1, y1, x2, y2] = r.bbox;
      ctx.setLineDash([6 * px, 4 * px]);
      ctx.strokeStyle = 'rgba(255,170,0,.95)';
      ctx.strokeRect(x1 - 3 * px, y1 - 3 * px, x2 - x1 + 6 * px, y2 - y1 + 6 * px);
    }
    ctx.setLineDash([]);
  }
  if (S.hover && S.tool === 'select' && !S.peek && S.view !== 'original' && S.view !== 'english') {
    const [x1, y1, x2, y2] = S.hover.bbox;
    ctx.lineWidth = 2 * px;
    ctx.strokeStyle = S.hover.enabled ? 'rgba(80,160,255,.95)' : 'rgba(255,170,0,1)';
    ctx.strokeRect(x1 - 4 * px, y1 - 4 * px, x2 - x1 + 8 * px, y2 - y1 + 8 * px);
  }
  if (S.stroke && S.stroke.points.length) {
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = 2 * S.stroke.radius;
    ctx.strokeStyle = S.stroke.mode === 'keep' ? 'rgba(70,130,255,.5)' : 'rgba(240,60,60,.5)';
    ctx.beginPath();
    const pts = S.stroke.points;
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (const q of pts) ctx.lineTo(q[0], q[1]);
    if (pts.length === 1) ctx.lineTo(pts[0][0] + 0.01, pts[0][1]);
    ctx.stroke();
  }
  ctx.restore();

  if (S.mouse && S.tool !== 'select' && !S.space) {       // brush cursor
    ctx.beginPath();
    ctx.arc(S.mouse.x, S.mouse.y, Math.max(2, S.brush * S.zoom), 0, Math.PI * 2);
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.stroke();
    ctx.lineWidth = 1.5; ctx.strokeStyle = S.tool === 'keep' ? '#7fb0ff' : '#ff8a8a'; ctx.stroke();
  }
  if (S.hover && S.tool === 'select' && S.mouse && !S.peek && S.view !== 'original' && S.view !== 'english') {
    const text = S.hover.enabled ? 'Click to bring this text back' : 'Click to erase this';
    ctx.font = '12px system-ui, "Segoe UI", sans-serif';
    const tw = ctx.measureText(text).width + 14;
    const tx = Math.min(w - tw - 6, S.mouse.x + 14), ty = Math.min(h - 30, S.mouse.y + 16);
    ctx.fillStyle = 'rgba(0,0,0,.82)';
    ctx.beginPath(); ctx.roundRect(tx, ty, tw, 22, 6); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textBaseline = 'middle';
    ctx.fillText(text, tx + 7, ty + 11.5);
  }
}

/* ------------------------------------------------------------------ UI */
function renderList() {
  const list = $('pageList');
  const have = new Map([...list.children].map((li) => [li.dataset.id, li]));
  S.pages.forEach((p, i) => {
    let li = have.get(p.id);
    if (!li) {
      li = document.createElement('li');
      li.dataset.id = p.id;
      li.tabIndex = 0;
      li.innerHTML = '<img alt="" loading="lazy"><span class="badge"></span>' +
        '<button class="x" title="Remove this page" aria-label="Remove this page">&times;</button><span class="name"></span>';
      li.querySelector('img').src = `/api/page/${p.id}/thumb`;
      li.addEventListener('click', (e) => { if (!e.target.classList.contains('x')) selectPage(p.id); });
      li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectPage(p.id); } });
      li.querySelector('.x').addEventListener('click', () => removePage(p.id));
    }
    have.delete(p.id);
    if (list.children[i] !== li) list.insertBefore(li, list.children[i] || null);
    li.classList.toggle('on', p.id === S.current);
    li.querySelector('.name').textContent = p.name;
    li.title = p.name;
    const b = li.querySelector('.badge');
    b.className = 'badge' + (p.status === 'done' ? ' done' : p.status === 'error' ? ' error' : '');
    b.textContent = p.status === 'done' ? '✓' : p.status === 'error' ? '!' : p.status === 'working' ? '…' : String(i + 1);
    b.title = p.status === 'error' ? (p.error || 'Failed') : p.stage;
  });
  for (const li of have.values()) li.remove();
  const n = S.pages.length;
  $('pageCount').textContent = n ? `${n} page${n === 1 ? '' : 's'}` : 'No pages';
  $('clearAll').hidden = n < 2;
}

function updateStatus() {
  const p = cur();
  let text = 'Ready';
  if (S.offline) text = 'Lost contact with the app. Is its window still open?';
  else if (p) {
    const bits = [p.name, `${p.width} × ${p.height}`];
    if (p.status === 'done') {
      bits.push(`${p.erased} region${p.erased === 1 ? '' : 's'} erased`);
      if (p.skipped) bits.push(`${p.skipped} left alone (dashed orange)`);
      if (p.translated && S.detail && S.detail.blocks) {
        const done = S.detail.blocks.filter((b) => b.lettered).length;
        bits.push(`${done} of ${S.detail.blocks.length} text blocks lettered in English`);
      }
      if (p.seconds) bits.push(`${p.seconds}s`);
    } else if (p.status === 'error') bits.push('failed: ' + (p.error || 'unknown error'));
    else bits.push(p.stage);
    text = bits.join('  ·  ');
  }
  $('statusText').textContent = text;
  $('zoomText').textContent = p && S.img.orig ? Math.round(S.zoom * 100) + '%' : '';
}

function updateUI() {
  const p = cur();
  $('empty').hidden = S.pages.length > 0 || !!S.uploading;
  canvas.style.cursor = S.space ? 'grab' : S.tool !== 'select' ? 'none' : S.hover ? 'pointer' : 'default';

  let pill = '';
  if (S.uploading) pill = `Adding pages ${S.uploading.done + 1} of ${S.uploading.total}`;
  else if (busyPage(p)) pill = (S.note ? S.note : p.stage) + '…';
  else if (S.pending) pill = 'Updating…';
  $('pill').hidden = !pill;
  $('pillText').textContent = pill;

  const ready = canEdit();
  const eng = S.view === 'english';
  $('dlPage').disabled = !(p && p.status === 'done');
  $('dlAll').disabled = !S.pages.some((q) => q.status === 'done');
  $('dlPage').textContent = eng ? 'Download English page' : 'Download page';
  $('dlAll').textContent = eng ? 'Download chapter in English' : 'Download chapter';
  for (const b of $('viewSeg').children) if (b.dataset.view === 'english') b.disabled = !S.img.english;

  $('tabClean').hidden = S.tab !== 'clean';
  $('tabTranslate').hidden = S.tab !== 'translate';
  $('translateBtn').disabled = !ready;
  $('translateBtn').textContent = p && p.translated ? 'Translate this page again' : 'Translate this page';
  if (document.activeElement !== $('engine')) $('engine').value = S.settings.engine;
  $('autoTranslate').checked = !!S.settings.autoTranslate;
  const engines = S.settings.engines || {};
  const problem = engines[S.settings.engine];
  $('engineNote').textContent = problem ? `Not available: ${problem}.` : (ENGINE_NOTES[S.settings.engine] || '');
  $('engineNote').classList.toggle('bad', !!problem);
  for (const o of $('engine').options) o.disabled = !!engines[o.value];
  if (document.activeElement !== $('notes')) $('notes').value = S.settings.notes || '';
  const lettering = S.chapter && S.chapter.lettering;
  $('fontName').textContent = lettering ? lettering.dialogue.name : 'Automatic';
  $('fontBtn').disabled = !lettering;
  const scene = p && p.translated && p.scene;
  $('sceneNote').hidden = !scene;
  if (scene) $('sceneNote').textContent = 'How the translator read this page: ' + scene;
  const failed = p && p.tstate === 'error';
  $('tnote').hidden = !failed;
  if (failed) $('tnote').textContent = 'Translation failed: ' + (p.tnote || 'unknown error');
  const hasBlocks = !!(S.detail && S.detail.blocks && S.detail.blocks.length && p && (p.translated || p.tstate === 'done'));
  $('blockHead').hidden = !hasBlocks;
  $('blockHint').hidden = !hasBlocks;
  $('undoBtn').disabled = !(ready && S.detail.canUndo);
  $('resetBtn').disabled = !ready;
  $('brushRow').hidden = S.tool === 'select';

  const ub = $('uncertainBtn');
  const unsure = ((S.detail && S.detail.regions) || []).filter((r) => !r.confident && r.source !== 'manual');
  ub.hidden = !unsure.length;
  if (unsure.length) {
    const off = unsure.filter((r) => !r.enabled).length;
    ub.dataset.enable = off ? '1' : '0';
    ub.textContent = off
      ? `Also erase the ${off} uncertain region${off === 1 ? '' : 's'}`
      : 'Leave the uncertain regions alone';
    ub.disabled = !ready;
  }
  updateStatus();
}

let toastTimer = null;
function toast(msg, kind) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  t.className = 'toast' + (kind ? ' ' + kind : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === 'info' ? 4000 : 6000);
}

const ROLE_NAMES = { '': 'Auto', dialogue: 'Dialogue', shout: 'Shout', title: 'Heavy title', narration: 'Caption (italic)', handwritten: 'Aside (handwritten)' };
const ENGINE_NOTES = {
  'claude-code': 'Localises the whole page at once, using the picture, your notes and the story so far. Each page (image and text) is sent to Anthropic and counts toward your Claude usage.',
  claude: 'The same localisation through the Anthropic API: each page is sent to Anthropic and billed to your API account.',
  offline: 'Runs entirely on this computer, but translates word for word: a rough draft to rewrite.',
};

/* The list of text blocks in the Translate tab. Rebuilt only when the data
   changes, so typing in a box is never interrupted. */
function renderBlocks() {
  const list = $('blockList');
  const p = cur();
  const blocks = (S.detail && p && (p.translated || p.tstate === 'done') && S.detail.blocks) || [];
  const lettering = (S.chapter && S.chapter.lettering) || {};
  const sig = JSON.stringify([p && p.id, blocks, lettering]);
  if (sig === S.blocksSig) { markSelected(); return; }
  S.blocksSig = sig;
  list.textContent = '';
  for (const b of blocks) {
    const li = document.createElement('li');
    li.dataset.id = b.id;
    li.classList.toggle('off', !b.enabled);
    const flag = !b.english.trim() ? ['could not read', 'warn']
      : b.overflow ? ['does not fit', 'warn'] : b.edited ? ['edited', ''] : ['', ''];
    li.innerHTML = '<div class="head"><span class="num"></span><span class="ja" lang="ja"></span>' +
      '<span class="flag"></span></div><textarea spellcheck="true" aria-label="English for this block"></textarea>' +
      '<div class="row"><select aria-label="Lettering style"></select>' +
      '<button class="step" data-step="-1" title="Smaller" aria-label="Smaller">&minus;</button>' +
      '<span class="size"></span>' +
      '<button class="step" data-step="1" title="Larger" aria-label="Larger">+</button>' +
      '<label class="skip"><input type="checkbox"> leave blank</label></div>' +
      '<button class="fontpick" type="button"><span>Font</span><span class="fp-v"></span><span class="fp-own"></span></button>';
    li.querySelector('.num').textContent = b.id;
    li.querySelector('.ja').textContent = b.japanese || '\u2014';
    const fl = li.querySelector('.flag');
    fl.textContent = flag[0]; if (flag[1]) fl.classList.add(flag[1]);
    const ta = li.querySelector('textarea');
    ta.value = b.english;
    ta.placeholder = 'Type the English for this block';
    const save = () => { if (ta.value !== b.english) editBlock(b.id, { english: ta.value }); };
    ta.addEventListener('focus', () => selectBlock(b.id, false));
    ta.addEventListener('blur', save);
    ta.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); ta.blur(); } });
    const sel = li.querySelector('select');
    for (const [value, label] of Object.entries(ROLE_NAMES)) {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = value === '' && b.role && !b.roleLocked ? `Auto (${ROLE_NAMES[b.role] || b.role})` : label;
      sel.appendChild(o);
    }
    sel.value = b.roleLocked ? b.role : '';
    sel.addEventListener('change', () => editBlock(b.id, { role: sel.value }));
    li.querySelector('.size').textContent = b.size ? `${Math.round(b.size)}px` : '';
    for (const btn of li.querySelectorAll('.step')) {
      btn.addEventListener('click', () => editBlock(b.id, { scale: Math.round((b.scale * (btn.dataset.step === '1' ? 1.12 : 1 / 1.12)) * 100) / 100 }));
    }
    const skip = li.querySelector('.skip input');
    skip.checked = !b.enabled;
    skip.addEventListener('change', () => editBlock(b.id, { enabled: !skip.checked }));
    // the font of this line: the chapter's for its style, unless it was given its own
    const role = b.role || 'dialogue';
    const chapterFont = lettering[role];
    const shown = b.fontName || (b.font ? FontPicker.nameOf(b.font) : chapterFont ? chapterFont.name : 'Automatic');
    const fp = li.querySelector('.fontpick');
    fp.querySelector('.fp-v').textContent = shown;
    fp.querySelector('.fp-own').textContent = b.font ? 'this line only' : '';
    fp.classList.toggle('custom', !!b.font);
    fp.title = b.font ? 'This line has a font of its own. Click to change it.' : 'Give this line a font of its own';
    fp.setAttribute('aria-label', `Font of line ${b.id}: ${shown}${b.font ? ', this line only' : ''}. Change`);
    fp.addEventListener('click', () => FontPicker.open({
      mode: 'line', line: b.id, role, font: b.font, sample: b.english,
      autoId: chapterFont ? chapterFont.id : '', autoName: chapterFont ? chapterFont.name : '',
      onApply: (id) => editBlock(b.id, { font: id }),
    }));
    li.addEventListener('mousedown', () => selectBlock(b.id, false));
    list.appendChild(li);
  }
  markSelected();
}

function markSelected() {
  for (const li of $('blockList').children) li.classList.toggle('sel', Number(li.dataset.id) === S.selBlock);
}

function selectBlock(id, reveal) {
  S.selBlock = id;
  markSelected();
  if (reveal) {
    const li = [...$('blockList').children].find((n) => Number(n.dataset.id) === id);
    if (li) { li.scrollIntoView({ block: 'nearest' }); li.querySelector('textarea').focus({ preventScroll: true }); }
  }
  draw();
}

async function editBlock(id, change) {
  await edit('block', Object.assign({ id }, change));
}

function blockAt(x, y) {
  let best = null, bestArea = Infinity;
  for (const b of (S.detail && S.detail.blocks) || []) {
    for (const box of [b.drawn, b.box]) {
      if (!box) continue;
      const [x1, y1, x2, y2] = box;
      if (x >= x1 - 6 && x < x2 + 6 && y >= y1 - 6 && y < y2 + 6) {
        const a = (x2 - x1) * (y2 - y1);
        if (a < bestArea) { best = b; bestArea = a; }
      }
    }
  }
  return best;
}

function setTab(t) {
  S.tab = t;
  for (const b of $('tabs').children) b.classList.toggle('on', b.dataset.tab === t);
  if (t === 'translate' && S.img.english) setView('english');
  else if (t === 'clean' && S.view === 'english') setView('clean');
  if (t === 'translate') setTool('select');
  updateUI(); draw();
}

async function translatePage() {
  const p = cur();
  if (!p || !canEdit()) return;
  S.pending++; updateUI();
  try {
    const res = await api(`/api/page/${p.id}/translate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ engine: S.settings.engine }),
    });
    if (res.page && res.page.tstate === 'error') toast('Translation failed: ' + (res.page.tnote || 'unknown error'));
  } catch (err) { toast(err.message); }
  S.pending--; stateSig = '';
  await poll();
}

async function saveSettings(change) {
  Object.assign(S.settings, change);
  updateUI();
  try {
    await api('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(change) });
  } catch (err) { toast(err.message); }
  stateSig = '';
}

function setView(v) {
  if (v === 'english' && !S.img.english) v = 'clean';
  S.view = v;
  for (const b of $('viewSeg').children) b.classList.toggle('on', b.dataset.view === v);
  if (v === 'detected' && !S.img.overlay) loadOverlay();
  updateUI(); draw();
}

function setTool(t) {
  S.tool = t; S.hover = null;
  for (const b of $('toolSeg').children) b.classList.toggle('on', b.dataset.tool === t);
  updateUI(); draw();
}

function setBrush(v) {
  S.brush = Math.max(3, Math.min(120, Math.round(v)));
  $('brush').value = S.brush;
  $('brushOut').textContent = S.brush + ' px';
  draw();
}

/* ------------------------------------------------------------- uploads */
const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

async function addFiles(files) {
  files = [...files].filter((f) => IMAGE_RE.test(f.name));
  if (!files.length) { toast('No images found in what you added.'); return; }
  const key = (f) => f._path || f.webkitRelativePath || f.name;
  files.sort((a, b) => natural.compare(key(a), key(b)));
  S.uploading = { done: 0, total: files.length };
  updateUI();
  for (const f of files) {
    try {
      await api(`/api/upload?chapter=${CHAPTER}&name=` + encodeURIComponent(f.name), { method: 'POST', body: f });
    } catch (err) { toast(`${f.name}: ${err.message}`); }
    S.uploading.done++;
    updateUI();
    if (S.uploading.done === 1) poll();
  }
  S.uploading = null;
  poll();
}

async function filesFromDrop(dt) {
  const entries = [...dt.items].filter((i) => i.kind === 'file')
    .map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
  if (!entries.some((e) => e && e.isDirectory)) return [...dt.files];
  const out = [];
  async function walk(entry, path) {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      f._path = path + f.name; out.push(f);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e, path + entry.name + '/');
      }
    }
  }
  for (const e of entries) if (e) await walk(e, '');
  return out;
}

async function removePage(id) {
  try { await api(`/api/page/${id}`, { method: 'DELETE' }); } catch (err) { toast(err.message); }
  if (S.current === id) {
    const i = S.pages.findIndex((p) => p.id === id);
    const next = S.pages[i + 1] || S.pages[i - 1];
    S.current = next ? next.id : null;
  }
  S.pages = S.pages.filter((p) => p.id !== id);
  stateSig = '';
  renderList(); await syncCurrent(); poll();
}

function download(url) {
  const a = document.createElement('a');
  a.href = url; a.download = '';
  document.body.appendChild(a); a.click(); a.remove();
}

/* ------------------------------------------------------- pointer input */
let drag = null;
const pointOf = (e) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

canvas.addEventListener('pointerdown', (e) => {
  if (!cur() || !S.img.orig) return;
  canvas.setPointerCapture(e.pointerId);
  canvas.focus({ preventScroll: true });
  const pt = pointOf(e);
  const pan = e.button !== 0 || S.space;
  drag = { start: pt, ox: S.ox, oy: S.oy, moved: false, mode: pan ? 'pan' : S.tool === 'select' ? 'click' : 'paint' };
  if (drag.mode === 'paint') {
    if (!canEdit()) { drag = null; return; }
    const mode = S.tool === 'keep' ? 'keep' : $('forceArt').checked ? 'inpaint' : 'erase';
    S.stroke = { points: [toPage(pt)], radius: S.brush, mode };
    draw();
  }
});

canvas.addEventListener('pointermove', (e) => {
  const pt = pointOf(e);
  S.mouse = pt;
  if (drag) {
    const dx = pt.x - drag.start.x, dy = pt.y - drag.start.y;
    if (drag.mode === 'click' && Math.hypot(dx, dy) > 5) drag.mode = 'pan';
    if (drag.mode === 'pan') {
      S.ox = drag.ox + dx; S.oy = drag.oy + dy; S.fitted = false; drag.moved = true;
    } else if (drag.mode === 'paint' && S.stroke) {
      const q = toPage(pt), last = S.stroke.points[S.stroke.points.length - 1];
      if (Math.hypot(q[0] - last[0], q[1] - last[1]) >= Math.max(0.75, 1.5 / S.zoom)) S.stroke.points.push(q);
    }
  } else if (S.tool === 'select' && S.detail && S.view === 'english') {
    const [x, y] = toPage(pt);
    canvas.style.cursor = blockAt(x, y) ? 'pointer' : 'default';
  } else if (S.tool === 'select' && S.detail && S.view !== 'original') {
    const [x, y] = toPage(pt);
    const hit = regionAt(x, y);
    if (hit !== S.hover) { S.hover = hit; updateUI(); }
  } else if (S.hover) { S.hover = null; updateUI(); }
  draw();
});

async function endDrag(e) {
  const d = drag; drag = null;
  if (!d) return;
  if (d.mode === 'click' && S.view === 'english') {
    const [x, y] = toPage(pointOf(e));
    const b = blockAt(x, y);
    if (b) { if (S.tab !== 'translate') setTab('translate'); selectBlock(b.id, true); }
  } else if (d.mode === 'click' && canEdit() && S.view !== 'original') {
    const [x, y] = toPage(pointOf(e));
    const sz = pageSize();
    if (x >= 0 && y >= 0 && x < sz.w && y < sz.h && regionAt(x, y)) {
      S.hover = null;
      await edit('toggle', { x, y });
    }
  } else if (d.mode === 'paint' && S.stroke) {
    const s = S.stroke;
    const res = await edit('stroke', { points: s.points.map((q) => [Math.round(q[0] * 10) / 10, Math.round(q[1] * 10) / 10]), radius: s.radius, mode: s.mode });
    S.stroke = null;
    if (res && S.view === 'original') setView('clean');
    draw();
  }
}
canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', () => { drag = null; S.stroke = null; draw(); });
canvas.addEventListener('pointerleave', () => { if (!drag) { S.mouse = null; S.hover = null; updateUI(); draw(); } });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('dblclick', () => { if (S.tool === 'select' && !S.hover) { fit(); draw(); updateStatus(); } });
canvas.addEventListener('wheel', (e) => {
  if (!S.img.orig) return;
  e.preventDefault();
  const pt = pointOf(e);
  const step = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
  zoomAt(pt.x, pt.y, Math.exp(-step * (e.ctrlKey ? 0.01 : 0.0018)));
}, { passive: false });

/* ------------------------------------------------------------ keyboard */
const typing = (e) => /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) && e.target.type !== 'range' && e.target.type !== 'checkbox';
window.addEventListener('keyup', (e) => { if (typing(e)) e.stopImmediatePropagation(); }, true);

window.addEventListener('keydown', (e) => {
  if (typing(e)) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); if (canEdit() && S.detail.canUndo) edit('undo'); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === ' ') { if (e.target === document.body || e.target === canvas) e.preventDefault(); if (!S.space) { S.space = true; updateUI(); draw(); } return; }
  if (k === 'c') { if (!S.peek) { S.peek = true; draw(); } return; }
  if (k === 'v') setTool('select');
  else if (k === 'e') setTool('erase');
  else if (k === 'k') setTool('keep');
  else if (k === '1') setView('clean');
  else if (k === '2') setView('original');
  else if (k === '3') setView('detected');
  else if (k === '4') setView('english');
  else if (k === 'f' || k === '0') { fit(); draw(); updateStatus(); }
  else if (k === '[') setBrush(S.brush / 1.2 - 1);
  else if (k === ']') setBrush(S.brush * 1.2 + 1);
  else if (k === '+' || k === '=') { const r = canvas.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 1.25); }
  else if (k === '-') { const r = canvas.getBoundingClientRect(); zoomAt(r.width / 2, r.height / 2, 0.8); }
  else if (k === 'arrowleft' || k === 'arrowright' || k === 'pageup' || k === 'pagedown') {
    if (e.target.type === 'range') return;
    const i = S.pages.findIndex((p) => p.id === S.current);
    const j = i + (k === 'arrowright' || k === 'pagedown' ? 1 : -1);
    if (S.pages[j]) { e.preventDefault(); selectPage(S.pages[j].id); }
  }
});
window.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  if (k === ' ') { S.space = false; updateUI(); draw(); }
  if (k === 'c') { S.peek = false; draw(); }
});
window.addEventListener('blur', () => { S.space = false; S.peek = false; draw(); });

/* -------------------------------------------------------------- wiring */
$('addBtn').onclick = $('emptyAdd').onclick = () => $('fileInput').click();
$('emptyFolder').onclick = () => $('dirInput').click();
$('fileInput').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('dirInput').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('viewSeg').onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) setView(b.dataset.view); };
$('tabs').onclick = (e) => { const b = e.target.closest('button'); if (b) setTab(b.dataset.tab); };
$('translateBtn').onclick = translatePage;
$('engine').onchange = (e) => saveSettings({ engine: e.target.value });
$('autoTranslate').onchange = (e) => saveSettings({ autoTranslate: e.target.checked });
$('notes').addEventListener('change', (e) => saveSettings({ notes: e.target.value, chapter: CHAPTER }));
$('fontBtn').onclick = () => {
  if (!S.chapter || !S.chapter.lettering) return;
  FontPicker.open({
    mode: 'chapter', role: 'dialogue', choices: S.chapter.fonts || {}, series: S.chapter.series,
    translated: S.pages.filter((p) => p.translated).length,
    onApply: async (changes) => {
      // the page on screen is lettered first, the rest of the chapter after it
      const res = await api(`/api/chapters/${CHAPTER}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ fonts: changes, first: S.current }),
      });
      S.chapter = Object.assign({}, S.chapter, { fonts: res.fonts, lettering: res.lettering, fontsVersion: res.fontsVersion });
      S.blocksSig = '';
      renderBlocks(); updateUI();
      toast(res.relettering
        ? `Lettering ${res.relettering} page${res.relettering === 1 ? '' : 's'} again, this one first. Nothing is translated again.`
        : `Saved. Pages will be lettered in ${res.lettering.dialogue.name} when they are translated.`, 'info');
      stateSig = '';
      poll();
    },
  });
};
$('toolSeg').onclick = (e) => { const b = e.target.closest('button'); if (b) setTool(b.dataset.tool); };
$('brush').oninput = (e) => setBrush(+e.target.value);
$('undoBtn').onclick = () => edit('undo');
$('resetBtn').onclick = () => edit('reset');
$('uncertainBtn').onclick = (e) => edit('set', { which: 'uncertain', enabled: e.currentTarget.dataset.enable === '1' });
const what = () => (S.view === 'english' ? 'english' : 'clean');
$('dlPage').onclick = () => { const p = cur(); if (p) download(`/api/page/${p.id}/download?format=${$('fmt').value}&what=${what()}`); };
$('dlAll').onclick = () => download(`/api/chapters/${CHAPTER}/download.zip?format=${$('fmt').value}&what=${what()}`);

let clearArmed = null;
$('clearAll').onclick = async (e) => {
  const b = e.currentTarget;
  if (!clearArmed) {                              // ask once, without a blocking dialog
    b.textContent = 'Really remove all?';
    clearArmed = setTimeout(() => { clearArmed = null; b.textContent = 'Remove all'; }, 3000);
    return;
  }
  clearTimeout(clearArmed); clearArmed = null; b.textContent = 'Remove all';
  for (const p of [...S.pages]) { try { await api(`/api/page/${p.id}`, { method: 'DELETE' }); } catch (_) { /* gone already */ } }
  S.pages = []; S.current = null; stateSig = '';
  renderList(); await syncCurrent(); poll();
};

let dragDepth = 0;
window.addEventListener('dragenter', (e) => { if (e.dataTransfer && [...e.dataTransfer.types].includes('Files')) { dragDepth++; $('dropveil').hidden = false; } });
window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('dropveil').hidden = true; });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', async (e) => {
  e.preventDefault(); dragDepth = 0; $('dropveil').hidden = true;
  if (e.dataTransfer) addFiles(await filesFromDrop(e.dataTransfer));
});

new ResizeObserver(() => { if (S.fitted) fit(); draw(); updateStatus(); }).observe($('stage'));
setBrush(S.brush);
poll();
