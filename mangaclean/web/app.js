'use strict';
/* Manga Cleaner - browser front end.  Talks to the local Python server. */

const $ = (id) => document.getElementById(id);
const canvas = $('view');
const ctx = canvas.getContext('2d');
const IMAGE_RE = /\.(png|jpe?g|jfif|webp|bmp|tiff?|gif|avif|zip|cbz)$/i;

const S = {
  pages: [],                 // summaries from the server
  note: '',
  current: null,             // id of the page on screen
  detail: null,              // regions etc. of that page
  shown: { id: null, version: -1 },
  img: { orig: null, clean: null, overlay: null },
  view: 'clean',             // clean | original | detected
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
  if (!res.ok) throw new Error(data.error || res.statusText || 'request failed');
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
    const st = await api('/api/state');
    S.offline = false;
    const sig = JSON.stringify(st);
    if (sig !== stateSig) {
      stateSig = sig;
      S.pages = st.pages;
      S.note = st.note || '';
      renderList();
      await syncCurrent();
    }
  } catch (_) {
    S.offline = true;
  }
  updateUI();
  pollTimer = setTimeout(poll, S.pages.some(busyPage) || S.uploading ? 600 : 2000);
}

/* -------------------------------------------------------- current page */
let loadToken = 0;
async function syncCurrent() {
  if (!S.pages.length) { S.current = null; }
  else if (!cur()) { S.current = S.pages[0].id; }
  const p = cur();
  const token = ++loadToken;
  if (!p) {
    S.detail = null; S.shown = { id: null, version: -1 };
    S.img = { orig: null, clean: null, overlay: null };
    draw(); return;
  }
  if (S.shown.id !== p.id) {
    S.shown = { id: p.id, version: -1 };
    S.detail = null; S.hover = null; S.stroke = null;
    S.img = { orig: null, clean: null, overlay: null };
    S.fitted = true;
    draw();
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
      S.detail = detail; S.img.clean = clean; S.img.overlay = null;
      S.shown.version = detail.version;
      if (S.view === 'detected') loadOverlay();
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

  const showOriginal = S.peek || S.view !== 'clean' || !S.img.clean;
  const img = showOriginal ? S.img.orig : S.img.clean;
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
  if (S.detail && !S.peek && S.view !== 'original') {
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
  if (S.hover && S.tool === 'select' && !S.peek && S.view !== 'original') {
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
  if (S.hover && S.tool === 'select' && S.mouse && !S.peek && S.view !== 'original') {
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
  $('dlPage').disabled = !(p && p.status === 'done');
  $('dlAll').disabled = !S.pages.some((q) => q.status === 'done');
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
function toast(msg) {
  const t = $('toast');
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 6000);
}

function setView(v) {
  S.view = v;
  for (const b of $('viewSeg').children) b.classList.toggle('on', b.dataset.view === v);
  if (v === 'detected' && !S.img.overlay) loadOverlay();
  draw();
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
      await api('/api/upload?name=' + encodeURIComponent(f.name), { method: 'POST', body: f });
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
  if (d.mode === 'click' && canEdit() && S.view !== 'original') {
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
$('addFolderBtn').onclick = $('emptyFolder').onclick = () => $('dirInput').click();
$('fileInput').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('dirInput').onchange = (e) => { addFiles(e.target.files); e.target.value = ''; };
$('viewSeg').onclick = (e) => { const b = e.target.closest('button'); if (b) setView(b.dataset.view); };
$('toolSeg').onclick = (e) => { const b = e.target.closest('button'); if (b) setTool(b.dataset.tool); };
$('brush').oninput = (e) => setBrush(+e.target.value);
$('undoBtn').onclick = () => edit('undo');
$('resetBtn').onclick = () => edit('reset');
$('uncertainBtn').onclick = (e) => edit('set', { which: 'uncertain', enabled: e.currentTarget.dataset.enable === '1' });
$('dlPage').onclick = () => { const p = cur(); if (p) download(`/api/page/${p.id}/download?format=${$('fmt').value}`); };
$('dlAll').onclick = () => download(`/api/download.zip?format=${$('fmt').value}`);

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
