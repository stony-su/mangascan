'use strict';
/* Chapter page: the pages of one chapter, its notes, translation and downloads. */

const CID = (location.pathname.match(/^\/chapter\/([0-9a-f]+)/) || [])[1];
const C = { detail: null, sig: '', timer: null, tiles: new Map(), dragId: null, saving: 0 };

async function load() {
  clearTimeout(C.timer);
  try {
    const d = await api(`/api/chapters/${CID}`);
    const sig = JSON.stringify(d);
    C.detail = d;
    if (sig !== C.sig) { C.sig = sig; render(); }
  } catch (err) {
    if (err.status === 404) { showMissing(); return; }
    toast('Lost contact with the app. Is its window still open?');
  }
  const busy = C.detail && C.detail.pageList.some((p) => p.status === 'queued' || p.status === 'working');
  C.timer = setTimeout(load, busy ? 1500 : 6000);
}

function showMissing() {
  $('main').hidden = true;
  $('missing').hidden = false;
  document.title = 'Chapter not found - Manga Cleaner';
}

/* ---------------------------------------------------------------- header */
function render() {
  const d = C.detail;
  $('main').hidden = false;
  document.title = `${d.title} - Manga Cleaner`;
  if (document.activeElement !== $('title')) $('title').value = d.title;
  if (document.activeElement !== $('series')) $('series').value = d.series;
  $('crumbTitle').textContent = d.title;
  $('crumbSeries').textContent = d.series;
  $('crumbSeries').hidden = $('crumbSep').hidden = !d.series;

  const list = $('seriesList');
  list.textContent = '';
  for (const s of d.seriesList || []) list.append(el('option', { value: s }));

  const pages = d.pageList;
  const translating = pages.filter((p) => p.tstate === 'working').length;
  const cleaning = pages.filter((p) => (p.status === 'queued' || p.status === 'working') && p.tstate !== 'working').length;
  let summary = pages.length ? `${plural(pages.length, 'page')}, ${d.cleaned} cleaned, ${d.translated} in English` : 'No pages yet';
  if (translating) summary += `. Translating, ${plural(translating, 'page')} to go`;
  else if (cleaning) summary += `. Cleaning ${plural(cleaning, 'page')}`;
  $('summary').textContent = summary;

  $('siblingNav').hidden = !d.prev && !d.next;
  $('prevLink').hidden = !d.prev;
  $('nextLink').hidden = !d.next;
  if (d.prev) $('prevLink').href = `/chapter/${d.prev}`;
  if (d.next) $('nextLink').href = `/chapter/${d.next}`;

  $('editBtn').href = `/chapter/${CID}/edit`;
  $('readBtn').href = `/chapter/${CID}/read`;
  $('readBtn').setAttribute('aria-disabled', d.cleaned ? 'false' : 'true');
  $('editBtn').setAttribute('aria-disabled', pages.length ? 'false' : 'true');

  renderSide(d);
  renderGrid(pages);
}

function renderSide(d) {
  const settings = d.settings || {};
  if (document.activeElement !== $('engine')) fillEngines($('engine'), settings);
  const problem = (settings.engines || {})[settings.engine];
  $('engineNote').textContent = problem ? `Not available: ${problem}.` : (ENGINE_NOTES[settings.engine] || '');
  $('engineNote').classList.toggle('bad', !!problem);
  const translating = d.pageList.some((p) => p.tstate === 'working');
  $('translateBtn').disabled = !d.pages || !!problem || translating;
  $('translateBtn').textContent = translating ? 'Translating...' : 'Translate chapter';

  $('notesHead').textContent = d.series ? 'About this series' : 'Notes for the translator';
  $('notesHelp').textContent = d.series
    ? `Shared by every chapter of ${d.series}. The setting, tone, who is who and name spellings help the translator most.`
    : 'Add this chapter to a series to share notes and name spellings between chapters.';
  if (document.activeElement !== $('notes')) $('notes').value = d.notes || '';

  const fmt = $('fmt').value;
  $('dlEnglish').href = `/api/chapters/${CID}/download.zip?what=english&format=${fmt}`;
  $('dlClean').href = `/api/chapters/${CID}/download.zip?what=clean&format=${fmt}`;
  $('dlEnglish').setAttribute('aria-disabled', d.translated ? 'false' : 'true');
  $('dlClean').setAttribute('aria-disabled', d.cleaned ? 'false' : 'true');
}

/* ----------------------------------------------------------------- pages */
function pageState(p) {
  if (p.status === 'error') return ['Could not be cleaned', 'bad', p.error || ''];
  if (p.tstate === 'working') return [p.status === 'working' ? p.stage : 'Waiting to be translated', 'work', ''];
  if (p.status === 'queued' || p.status === 'working') return [p.status === 'working' ? p.stage : 'Waiting', 'work', ''];
  if (p.tstate === 'error') return ['Translation failed', 'bad', p.tnote || ''];
  if (p.translated) return ['In English', 'done', ''];
  if (p.cleaned) return ['Cleaned', '', ''];
  return ['Waiting', '', ''];
}

function makeTile(p) {
  const li = el('li', { class: 'tile', 'data-id': p.id, draggable: 'true', tabindex: '0' });
  const img = el('img', { alt: '', draggable: 'false', loading: 'lazy' });
  const link = el('a', { class: 'thumb', draggable: 'false' }, img);
  const num = el('span', { class: 'num' });
  const dot = el('span', { class: 'working-dot', hidden: true });
  const state = el('span', { class: 'state' });
  const x = el('button', { class: 'tile-x', type: 'button', text: 'Remove' });
  li.append(link, el('div', { class: 'tile-meta' }, num, dot, state, x));
  x.addEventListener('click', (e) => { e.preventDefault(); removePage(p.id, x); });
  li.addEventListener('dragstart', (e) => {
    C.dragId = p.id;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', p.id);
    requestAnimationFrame(() => li.classList.add('dragging'));
  });
  li.addEventListener('dragend', () => { C.dragId = null; li.classList.remove('dragging'); clearMarks(); });
  li.addEventListener('keydown', (e) => {
    if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
      e.preventDefault();
      moveBy(p.id, e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 1);
    }
  });
  return { li, img, link, num, dot, state, x, v: -1, what: '' };
}

function renderGrid(pages) {
  const grid = $('grid');
  $('emptyPages').hidden = pages.length > 0;
  grid.hidden = pages.length === 0;
  const seen = new Set();
  pages.forEach((p, i) => {
    let t = C.tiles.get(p.id);
    if (!t) { t = makeTile(p); C.tiles.set(p.id, t); }
    seen.add(p.id);
    if (grid.children[i] !== t.li) grid.insertBefore(t.li, grid.children[i] || null);
    const what = p.translated ? 'english' : p.cleaned ? 'clean' : 'original';
    if (t.v !== p.version || t.what !== what) {
      t.img.src = `/api/page/${p.id}/thumb?what=best&v=${p.version}-${what}`;
      t.v = p.version;
      t.what = what;
    }
    t.link.href = `/chapter/${CID}/edit?page=${p.id}`;
    t.link.setAttribute('aria-label', `Page ${i + 1}: open in the editor`);
    t.img.alt = `Page ${i + 1}`;
    t.num.textContent = i + 1;
    const [text, kind, title] = pageState(p);
    t.state.textContent = text;
    t.state.className = 'state' + (kind ? ' ' + kind : '');
    t.state.title = title;
    t.dot.hidden = kind !== 'work';
    t.x.setAttribute('aria-label', `Remove page ${i + 1}`);
  });
  for (const [id, t] of C.tiles) if (!seen.has(id)) { t.li.remove(); C.tiles.delete(id); }
}

/* ------------------------------------------------------------- reorder */
function clearMarks() {
  for (const t of C.tiles.values()) t.li.classList.remove('drop-before', 'drop-after');
}

function dropTarget(e) {
  const li = e.target.closest && e.target.closest('.tile');
  if (!li || li.dataset.id === C.dragId) return null;
  const r = li.getBoundingClientRect();
  return { li, after: e.clientX > r.left + r.width / 2 };
}

$('grid').addEventListener('dragover', (e) => {
  if (!C.dragId) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'move';
  clearMarks();
  const t = dropTarget(e);
  if (t) t.li.classList.add(t.after ? 'drop-after' : 'drop-before');
});
$('grid').addEventListener('drop', (e) => {
  if (!C.dragId) return;
  e.preventDefault();
  const t = dropTarget(e);
  clearMarks();
  if (!t) return;
  const order = C.detail.pageList.map((p) => p.id).filter((id) => id !== C.dragId);
  let at = order.indexOf(t.li.dataset.id);
  if (t.after) at += 1;
  order.splice(at, 0, C.dragId);
  saveOrder(order);
});

function moveBy(id, delta) {
  const order = C.detail.pageList.map((p) => p.id);
  const i = order.indexOf(id);
  const j = i + delta;
  if (i < 0 || j < 0 || j >= order.length) return;
  order.splice(i, 1);
  order.splice(j, 0, id);
  saveOrder(order).then(() => { const t = C.tiles.get(id); if (t) t.li.focus(); });
}

async function saveOrder(order) {
  const byId = new Map(C.detail.pageList.map((p) => [p.id, p]));
  C.detail.pageList = order.map((id) => byId.get(id));
  renderGrid(C.detail.pageList);                         // show the new order at once
  try {
    C.detail = Object.assign(C.detail, await postJSON(`/api/chapters/${CID}`, { order }));
    C.sig = '';
  } catch (err) { toast(err.message); }
  load();
}

/* ------------------------------------------------------------- actions */
function armed(button, label, armedLabel, action) {
  if (button.classList.contains('armed')) {
    clearTimeout(button._t);
    button.classList.remove('armed');
    button.textContent = label;
    action();
    return;
  }
  button.classList.add('armed');
  button.textContent = armedLabel;
  button._t = setTimeout(() => { button.classList.remove('armed'); button.textContent = label; }, 4000);
}

function removePage(id, button) {
  armed(button, 'Remove', 'Remove?', async () => {
    try { await api(`/api/page/${id}`, { method: 'DELETE' }); } catch (err) { toast(err.message); }
    C.sig = '';
    load();
  });
}

async function addFiles(files) {
  if (!files.length) { toast('No images found in what you chose.'); return; }
  await uploadAll(CID, files, (i, n) => taskbar(`Adding page ${i + 1} of ${n}`));
  taskbar('');
  C.sig = '';
  load();
}

async function saveChapter(change) {
  try {
    C.detail = Object.assign(C.detail, await postJSON(`/api/chapters/${CID}`, change));
    C.sig = '';
    render();
  } catch (err) { toast(err.message); }
}

$('title').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); if (e.key === 'Escape') { e.target.value = C.detail.title; e.target.blur(); } });
$('title').addEventListener('change', (e) => { const v = e.target.value.trim(); if (v && v !== C.detail.title) saveChapter({ title: v }); else e.target.value = C.detail.title; });
$('series').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); });
$('series').addEventListener('change', (e) => { if (e.target.value.trim() !== C.detail.series) saveChapter({ series: e.target.value.trim() }); });
$('notes').addEventListener('change', (e) => saveChapter({ notes: e.target.value }));
$('engine').addEventListener('change', async (e) => {
  try { await postJSON('/api/settings', { engine: e.target.value }); } catch (err) { toast(err.message); }
  C.sig = '';
  load();
});
$('fmt').value = store.get('mc.format', 'png');
$('fmt').addEventListener('change', (e) => { store.set('mc.format', e.target.value); renderSide(C.detail); });

$('translateBtn').addEventListener('click', async () => {
  const btn = $('translateBtn');
  btn.disabled = true;
  try {
    const res = await postJSON(`/api/chapters/${CID}/translate`, { engine: $('engine').value, redo: $('redo').checked });
    $('translateNote').textContent = res.queued
      ? `Translating ${plural(res.queued, 'page')} in reading order, so each page knows the story so far. You can keep working meanwhile.`
      : 'Every page is already in English. Tick the box above to translate them again.';
    $('redo').checked = false;
  } catch (err) {
    $('translateNote').textContent = err.message;
  }
  C.sig = '';
  load();
});

$('deleteBtn').addEventListener('click', (e) => {
  const n = C.detail ? C.detail.pages : 0;
  armed(e.currentTarget, 'Delete chapter', `Delete it and its ${plural(n, 'page')}? Click again`, async () => {
    try {
      await api(`/api/chapters/${CID}`, { method: 'DELETE' });
      location.href = '/';
    } catch (err) { toast(err.message); }
  });
});

$('addBtn').onclick = $('emptyFiles').onclick = () => $('fileInput').click();
$('emptyFolder').onclick = () => $('dirInput').click();
$('fileInput').onchange = (e) => { const f = sortFiles(e.target.files); e.target.value = ''; addFiles(f); };
$('dirInput').onchange = (e) => { const f = sortFiles(e.target.files); e.target.value = ''; addFiles(f); };
dropAnywhere($('dropveil'), (files) => addFiles(files));

if (!CID) showMissing();
else load();
