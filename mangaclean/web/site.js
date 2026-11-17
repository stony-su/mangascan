'use strict';
/* Helpers shared by the library, chapter and reader pages. */

const $ = (id) => document.getElementById(id);
const IMAGE_RE = /\.(png|jpe?g|jfif|webp|bmp|tiff?|gif|avif|zip|cbz)$/i;
const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = {};
  try { data = await res.json(); } catch (_) { /* not json */ }
  if (!res.ok) {
    const err = new Error(data.error || res.statusText || 'Request failed');
    err.status = res.status;
    throw err;
  }
  return data;
}

function postJSON(path, body) {
  return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
}

let toastTimer = null;
function toast(msg, kind) {
  const t = $('toast');
  if (!t) return;
  t.textContent = msg;
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, kind === 'info' ? 3500 : 7000);
}

function plural(n, one, many) { return `${n} ${n === 1 ? one : (many || one + 's')}`; }

/* Build an element. Text is always set as text, never as HTML. */
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children) if (c !== null && c !== undefined) node.append(c);
  return node;
}

const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch (_) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* private mode */ }
  },
};

function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return 'just now';
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  if (s < 2 * 86400) return 'yesterday';
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
}

/* ------------------------------------------------------------- uploads */
const fileKey = (f) => f._path || f.webkitRelativePath || f.name;

function sortFiles(files) {
  return [...files].filter((f) => IMAGE_RE.test(f.name)).sort((a, b) => natural.compare(fileKey(a), fileKey(b)));
}

/* Files of a drop, folders included. Also returns the name of a single
   dropped folder or archive, which makes a good chapter title. */
async function filesFromDrop(dt) {
  const items = [...(dt.items || [])].filter((i) => i.kind === 'file');
  const entries = items.map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
  let title = '';
  if (entries.length === 1 && entries[0] && entries[0].isDirectory) title = entries[0].name;
  if (!entries.some((e) => e && e.isDirectory)) {
    const files = [...dt.files];
    if (files.length === 1 && /\.(zip|cbz)$/i.test(files[0].name)) title = files[0].name.replace(/\.(zip|cbz)$/i, '');
    return { files: sortFiles(files), title };
  }
  const out = [];
  async function walk(entry, path) {
    if (entry.isFile) {
      const f = await new Promise((res, rej) => entry.file(res, rej));
      f._path = path + f.name;
      out.push(f);
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
  return { files: sortFiles(out), title };
}

function titleFromPicked(files) {
  const first = files[0];
  if (!first) return '';
  if (first.webkitRelativePath) return first.webkitRelativePath.split('/')[0];
  if (files.length === 1 && /\.(zip|cbz)$/i.test(first.name)) return first.name.replace(/\.(zip|cbz)$/i, '');
  return '';
}

/* Upload into a chapter, one file at a time, in reading order. */
async function uploadAll(chapterId, files, onProgress) {
  let added = 0;
  for (let i = 0; i < files.length; i++) {
    if (onProgress) onProgress(i, files.length);
    try {
      const res = await api(`/api/chapters/${chapterId}/upload?name=${encodeURIComponent(files[i].name)}`,
        { method: 'POST', body: files[i] });
      added += (res.added || []).length;
    } catch (err) {
      toast(`${files[i].name}: ${err.message}`);
    }
  }
  return added;
}

/* Whole-window drag and drop of files (not of things dragged inside the page). */
function dropAnywhere(veil, onDrop) {
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes('Files');
  let depth = 0;
  window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; veil.hidden = false; } });
  window.addEventListener('dragleave', (e) => { if (hasFiles(e)) { depth = Math.max(0, depth - 1); if (!depth) veil.hidden = true; } });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener('drop', async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    veil.hidden = true;
    const got = await filesFromDrop(e.dataTransfer);
    if (!got.files.length) { toast('No images found in what you dropped.'); return; }
    onDrop(got.files, got.title);
  });
}

function taskbar(text) {
  const bar = $('taskbar');
  if (!bar) return;
  bar.hidden = !text;
  if (text) $('taskText').textContent = text;
}

const ENGINE_NOTES = {
  'claude-code': 'Localises each page with the picture, your series notes and the story so far. Pages are sent to Anthropic and count toward your Claude usage.',
  claude: 'The same localisation through the Anthropic API: pages are sent to Anthropic and billed to your API account.',
  offline: 'Runs entirely on this computer, but translates word for word: a rough draft to rewrite.',
};
const ENGINE_NAMES = { 'claude-code': 'Claude (your Claude Code login)', claude: 'Claude (Anthropic API key)', offline: 'On this computer (rough draft)' };

function fillEngines(select, settings) {
  select.textContent = '';
  for (const [value, label] of Object.entries(ENGINE_NAMES)) {
    const problem = (settings.engines || {})[value];
    select.append(el('option', { value, disabled: !!problem, text: label + (problem ? ' (not available)' : '') }));
  }
  select.value = settings.engine;
}
