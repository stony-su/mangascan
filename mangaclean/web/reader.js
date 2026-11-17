'use strict';
/* Reader: one page at a time (right to left, like manga) or one long scroll. */

const CID = (location.pathname.match(/^\/chapter\/([0-9a-f]+)/) || [])[1];
const prefs = store.get('mc.reader', {});
const R = {
  d: null, pages: [], i: 0, sig: '', timer: null, atEnd: false, observer: null, swiped: false,
  mode: prefs.mode === 'strip' ? 'strip' : 'page',
  dir: prefs.dir === 'ltr' ? 'ltr' : 'rtl',
  version: ['english', 'clean', 'original'].includes(prefs.version) ? prefs.version : 'english',
};

function savePrefs() { store.set('mc.reader', { mode: R.mode, dir: R.dir, version: R.version }); }

/* The image of a page in the chosen version, falling back to the most finished one. */
function source(p) {
  if (R.version === 'english' && p.translated) return [`/api/page/${p.id}/english?v=${p.version}`, ''];
  if (R.version !== 'original' && p.cleaned) {
    return [`/api/page/${p.id}/clean?v=${p.version}`, R.version === 'english' ? 'Not translated yet, showing the cleaned page' : ''];
  }
  return [`/api/page/${p.id}/original`, R.version === 'original' ? '' : 'Not cleaned yet, showing the original'];
}

async function load(first) {
  clearTimeout(R.timer);
  try {
    const d = await api(`/api/chapters/${CID}`);
    R.d = d;
    const sig = JSON.stringify(d.pageList.map((p) => [p.id, p.version, p.translated, p.cleaned]));
    if (sig !== R.sig || first) { R.sig = sig; R.pages = d.pageList; refresh(first); }
  } catch (err) {
    if (err.status === 404) { location.href = '/'; return; }
    toast('Lost contact with the app. Is its window still open?');
  }
  const busy = R.pages.some((p) => p.status === 'queued' || p.status === 'working');
  R.timer = setTimeout(() => load(false), busy ? 4000 : 15000);
}

function refresh(first) {
  const d = R.d;
  document.title = `${d.title} - Reader - Manga Cleaner`;
  $('rTitle').textContent = d.title;
  $('rSeries').textContent = d.series || '';
  for (const id of ['back', 'endBack', 'emptyBack']) $(id).href = `/chapter/${CID}`;
  $('nextChapter').hidden = !d.next;
  if (d.next) $('nextChapter').href = `/chapter/${d.next}/read`;

  const any = R.pages.length > 0;
  $('rempty').hidden = any;
  $('foot').hidden = !any;
  if (!any) { $('pageView').hidden = true; $('strip').hidden = true; $('endCard').hidden = true; return; }

  const english = R.pages.some((p) => p.translated);
  if (first) {
    if (!english && R.version === 'english') R.version = 'clean';
    R.i = Math.min(Math.max(0, store.get(`mc.read.${CID}`, 0) | 0), R.pages.length - 1);
  }
  R.i = Math.min(R.i, R.pages.length - 1);
  for (const b of $('versionSeg').children) {
    b.classList.toggle('on', b.dataset.v === R.version);
    b.disabled = b.dataset.v === 'english' && !english;
  }
  for (const b of $('modeSeg').children) b.classList.toggle('on', b.dataset.m === R.mode);
  document.body.classList.toggle('paged', R.mode === 'page');
  $('dirBtn').textContent = R.dir === 'rtl' ? 'Right to left' : 'Left to right';
  $('dirBtn').hidden = R.mode === 'strip';
  $('slider').max = R.pages.length;
  $('slider').classList.toggle('rtl', R.dir === 'rtl' && R.mode === 'page');
  $('endTitle').textContent = `End of ${d.title}`;
  $('endText').textContent = d.next ? 'The next chapter of the series is ready to open.' : 'You have reached the last page.';
  if (R.mode === 'strip') showStrip(first); else showPage();
}

/* ------------------------------------------------------------ page mode */
function showPage() {
  $('strip').hidden = true;
  if (R.observer) { R.observer.disconnect(); R.observer = null; }
  if (R.atEnd) {
    $('pageView').hidden = true;
    $('endCard').hidden = false;
    $('counter').textContent = `${R.pages.length} / ${R.pages.length}`;
    $('pageNote').textContent = '';
    return;
  }
  $('endCard').hidden = true;
  $('pageView').hidden = false;
  const p = R.pages[R.i];
  const [url, note] = source(p);
  const img = $('pageImg');
  if (img.getAttribute('src') !== url) {
    img.classList.add('fading');
    img.onload = () => img.classList.remove('fading');
    img.src = url;
  }
  img.alt = `Page ${R.i + 1} of ${R.pages.length}`;
  showPosition(note);
  const nextSide = R.dir === 'rtl' ? 'zoneLeft' : 'zoneRight';
  const prevSide = R.dir === 'rtl' ? 'zoneRight' : 'zoneLeft';
  $(nextSide).setAttribute('aria-label', 'Next page');
  $(prevSide).setAttribute('aria-label', 'Previous page');
  for (const k of [1, 2, -1]) {                       // have the neighbours ready
    const q = R.pages[R.i + k];
    if (q) { const pre = new Image(); pre.src = source(q)[0]; }
  }
}

function showPosition(note) {
  $('counter').textContent = `${R.i + 1} / ${R.pages.length}`;
  $('slider').value = R.i + 1;
  $('pageNote').textContent = note || '';
  $('pageNote').title = note || '';
  store.set(`mc.read.${CID}`, R.i);
}

function go(delta) {
  if (R.mode !== 'page') return;
  if (R.atEnd) {
    if (delta < 0) { R.atEnd = false; showPage(); }
    return;
  }
  const j = R.i + delta;
  if (j < 0) return;
  if (j >= R.pages.length) { R.atEnd = true; showPage(); return; }
  R.i = j;
  showPage();
}

/* ----------------------------------------------------------- scroll mode */
function showStrip(first) {
  $('pageView').hidden = true;
  $('strip').hidden = false;
  $('endCard').hidden = false;
  const strip = $('strip');
  if (strip.childElementCount !== R.pages.length) {
    strip.textContent = '';
    R.pages.forEach((p, i) => strip.append(el('img', {
      'data-i': i, alt: `Page ${i + 1}`, loading: 'lazy', decoding: 'async',
      width: p.width || 850, height: p.height || 1200,
    })));
  }
  [...strip.children].forEach((img, i) => {
    const url = source(R.pages[i])[0];
    if (img.getAttribute('src') !== url) img.src = url;
  });
  if (R.observer) R.observer.disconnect();
  R.observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        R.i = Number(e.target.dataset.i);
        showPosition(source(R.pages[R.i])[1]);
      }
    }
  }, { rootMargin: '-45% 0px -45% 0px' });
  for (const img of strip.children) R.observer.observe(img);
  if (first) jumpStrip(R.i, false);
}

function jumpStrip(i, smooth) {
  const img = $('strip').children[i];
  if (img) img.scrollIntoView({ block: 'start', behavior: smooth && !matchMedia('(prefers-reduced-motion: reduce)').matches ? 'smooth' : 'auto' });
}

/* ------------------------------------------------------------- controls */
function setMode(mode) {
  R.mode = mode;
  R.atEnd = false;
  document.body.classList.remove('immersive');
  savePrefs();
  refresh(false);
  if (mode === 'strip') jumpStrip(R.i, false);
  else window.scrollTo(0, 0);
}

function setVersion(v) {
  if (v === 'english' && !R.pages.some((p) => p.translated)) return;
  R.version = v;
  savePrefs();
  refresh(false);
}

function toggleDir() {
  R.dir = R.dir === 'rtl' ? 'ltr' : 'rtl';
  savePrefs();
  refresh(false);
}

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else if (document.documentElement.requestFullscreen) document.documentElement.requestFullscreen().catch(() => {});
}

$('versionSeg').onclick = (e) => { const b = e.target.closest('button'); if (b && !b.disabled) setVersion(b.dataset.v); };
$('modeSeg').onclick = (e) => { const b = e.target.closest('button'); if (b) setMode(b.dataset.m); };
$('dirBtn').onclick = toggleDir;
$('again').onclick = () => { R.atEnd = false; R.i = 0; if (R.mode === 'strip') jumpStrip(0, true); else showPage(); };
$('zoneLeft').onclick = () => { if (!R.swiped) go(R.dir === 'rtl' ? 1 : -1); };
$('zoneRight').onclick = () => { if (!R.swiped) go(R.dir === 'rtl' ? -1 : 1); };
$('zoneMid').onclick = () => { if (!R.swiped) document.body.classList.toggle('immersive'); };
$('slider').addEventListener('input', (e) => {
  const i = Number(e.target.value) - 1;
  if (R.mode === 'strip') { jumpStrip(i, false); return; }
  R.atEnd = false;
  R.i = i;
  showPage();
});

let touch = null;
$('pageView').addEventListener('pointerdown', (e) => { touch = { x: e.clientX, y: e.clientY }; R.swiped = false; });
$('pageView').addEventListener('pointerup', (e) => {
  if (!touch) return;
  const dx = e.clientX - touch.x;
  const dy = e.clientY - touch.y;
  touch = null;
  if (Math.abs(dx) > 50 && Math.abs(dx) > 1.5 * Math.abs(dy)) {
    R.swiped = true;                                   // the click that follows is not a tap
    setTimeout(() => { R.swiped = false; }, 50);
    // pulling the page towards the reading direction turns it
    go((dx > 0) === (R.dir === 'rtl') ? 1 : -1);
  }
});

window.addEventListener('keydown', (e) => {
  if (e.target.id === 'slider' || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  if (R.mode === 'page') {
    if (k === 'ArrowLeft') { e.preventDefault(); go(R.dir === 'rtl' ? 1 : -1); return; }
    if (k === 'ArrowRight') { e.preventDefault(); go(R.dir === 'rtl' ? -1 : 1); return; }
    if (k === ' ' || k === 'PageDown') { e.preventDefault(); go(e.shiftKey ? -1 : 1); return; }
    if (k === 'PageUp') { e.preventDefault(); go(-1); return; }
    if (k === 'Home') { R.atEnd = false; R.i = 0; showPage(); return; }
    if (k === 'End') { R.atEnd = false; R.i = R.pages.length - 1; showPage(); return; }
  }
  const lower = k.toLowerCase();
  if (lower === 'f') toggleFullscreen();
  else if (lower === 'd' && R.mode === 'page') toggleDir();
  else if (lower === 'm') setMode(R.mode === 'page' ? 'strip' : 'page');
  else if (lower === 'v') {
    const order = ['english', 'clean', 'original'].filter((v) => v !== 'english' || R.pages.some((p) => p.translated));
    setVersion(order[(order.indexOf(R.version) + 1) % order.length]);
  }
});

if (!CID) location.href = '/';
else load(true);
