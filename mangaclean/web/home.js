'use strict';
/* Library page: chapters grouped by series. */

const H = { lib: null, sig: '', timer: null };

async function load() {
  clearTimeout(H.timer);
  try {
    const lib = await api('/api/library');
    $('offline').hidden = true;
    const sig = JSON.stringify([lib.chapters, lib.series]);
    H.lib = lib;
    if (sig !== H.sig) { H.sig = sig; render(); }
  } catch (_) {
    $('offline').hidden = false;
  }
  $('loading').hidden = true;
  const busy = H.lib && H.lib.chapters.some((c) => c.working > 0);
  H.timer = setTimeout(load, busy ? 2000 : 8000);
}

function render() {
  const chapters = H.lib.chapters;
  const list = $('seriesOptions');
  list.textContent = '';
  for (const s of H.lib.series) list.append(el('option', { value: s }));

  $('empty').hidden = chapters.length > 0;
  $('library').hidden = chapters.length === 0;
  if (!chapters.length) return;

  const groups = new Map();
  for (const c of chapters) {
    const key = c.series || '';
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(c);
  }
  const latest = (items) => Math.max(...items.map((c) => c.updated));
  const ordered = [...groups.entries()].sort((a, b) => {
    if (!a[0] !== !b[0]) return a[0] ? -1 : 1;          // chapters without a series last
    return latest(b[1]) - latest(a[1]);                 // most recently worked on first
  });
  const named = ordered.filter(([name]) => name).length;
  $('libCount').textContent = `${plural(chapters.length, 'chapter')}` + (named ? ` in ${plural(named, 'series', 'series')}` : '');

  const box = $('groups');
  box.textContent = '';
  for (const [name, items] of ordered) {
    items.sort((a, b) => natural.compare(a.title, b.title));
    const heading = name || (named ? 'Chapters without a series' : 'Chapters');
    box.append(el('section', { class: 'series' },
      el('header', { class: 'series-head' },
        el('h2', { text: heading }),
        el('p', { class: 'muted', text: `${plural(items.length, 'chapter')}, updated ${timeAgo(latest(items))}` })),
      el('div', { class: 'cards' }, ...items.map(card))));
  }
}

function card(c) {
  const href = `/chapter/${c.id}`;
  const cover = c.cover
    ? el('img', { src: `/api/page/${c.cover}/thumb?what=best&v=${c.coverVersion}`, alt: '', loading: 'lazy' })
    : el('span', { class: 'nocover', text: 'No pages yet' });
  const pct = c.pages ? Math.round((100 * c.translated) / c.pages) : 0;
  let status = '';
  let bad = false;
  if (c.working) status = `Working on ${plural(c.working, 'page')}`;
  else if (c.errors) { status = `${plural(c.errors, 'page')} need attention`; bad = true; }
  else if (c.pages && c.translated === c.pages) status = 'Ready to read';
  return el('article', { class: 'card' },
    el('a', { class: 'cover', href, 'aria-label': `Open ${c.title}` },
      cover, pct ? el('span', { class: 'bar', style: `width:${pct}%` }) : null),
    el('div', { class: 'card-body' },
      el('h3', {}, el('a', { href, text: c.title })),
      el('p', { class: 'muted', text: c.pages ? `${plural(c.pages, 'page')}, ${c.translated} in English` : 'No pages yet' }),
      status ? el('p', { class: 'status' + (bad ? ' bad' : ''), text: status }) : null,
      c.translated ? el('a', { class: 'btn small', href: `${href}/read`, text: 'Read' }) : null));
}

/* ------------------------------------------------------- new chapters */
function openNew(show) {
  $('newPanel').hidden = !show;
  if (show) { $('newTitle').focus(); }
}

async function startChapter(files, title) {
  if (!files.length) { toast('No images found in what you chose.'); return; }
  let ch;
  try {
    ch = await postJSON('/api/chapters', { title: title || 'New chapter', series: '' });
  } catch (err) { toast(err.message); return; }
  const added = await uploadAll(ch.id, files, (i, n) => taskbar(`Adding page ${i + 1} of ${n}`));
  taskbar('');
  if (!added) toast('None of those files could be read as manga pages.');
  location.href = `/chapter/${ch.id}`;
}

$('newBtn').onclick = () => openNew($('newPanel').hidden);
$('newCancel').onclick = () => openNew(false);
$('newForm').onsubmit = async (e) => {
  e.preventDefault();
  const title = $('newTitle').value.trim();
  if (!title) { $('newTitle').focus(); return; }
  try {
    const ch = await postJSON('/api/chapters', { title, series: $('newSeries').value.trim() });
    location.href = `/chapter/${ch.id}`;
  } catch (err) { toast(err.message); }
};
$('pickFiles').onclick = () => $('fileInput').click();
$('pickFolder').onclick = () => $('dirInput').click();
$('fileInput').onchange = (e) => { const f = [...e.target.files]; e.target.value = ''; startChapter(sortFiles(f), titleFromPicked(f)); };
$('dirInput').onchange = (e) => { const f = [...e.target.files]; e.target.value = ''; startChapter(sortFiles(f), titleFromPicked(f)); };
$('retry').onclick = load;
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('newPanel').hidden) openNew(false); });
dropAnywhere($('dropveil'), (files, title) => startChapter(files, title));
load();
