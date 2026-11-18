'use strict';
/* Font picker: choose the lettering font of a chapter (for each style) or of
   one line. Shared by the chapter page and the editor. Everything lives in
   this closure; the pages only see window.FontPicker. */
(function () {
  const ROLES = ['dialogue', 'narration', 'shout', 'title', 'handwritten'];
  const ROLE_LABEL = { dialogue: 'Dialogue', narration: 'Captions', shout: 'Shouts', title: 'Titles', handwritten: 'Asides' };
  const USE_LABEL = { dialogue: 'Dialogue', narration: 'Captions', shout: 'Shouts', title: 'Titles', handwritten: 'Asides', special: 'Special' };
  const GROUPS = [
    ['free', 'Free comic fonts'],
    ['yours', 'Your fonts'],
    ['installed', 'Installed on this computer'],
  ];
  const WHY = { default: 'the free default', pro: 'a professional font you own', 'fonts.json': 'set in fonts.json' };

  let catalog = null;          // {fonts, version, auto} from /api/fonts
  let dlg = null;              // the dialog, built on first use
  let st = null;               // what the open dialog is choosing
  let watcher = null;          // loads previews as they scroll into view

  const byId = (id) => (catalog ? catalog.fonts.find((f) => f.id === id) : null) || null;
  const q = (sel) => dlg.querySelector(sel);

  function node(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === 'class') n.className = v;
      else if (k === 'text') n.textContent = v;
      else n.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids) if (kid !== null && kid !== undefined) n.append(kid);
    return n;
  }

  function say(msg, kind) { if (typeof toast === 'function') toast(msg, kind); }

  async function getJSON(url, opts) {
    const res = await fetch(url, opts);
    let data = {};
    try { data = await res.json(); } catch (_) { /* not json */ }
    if (!res.ok) throw new Error(data.error || res.statusText || 'Request failed');
    return data;
  }

  async function load(force) {
    if (!catalog || force) catalog = await getJSON('/api/fonts');
    return catalog;
  }

  function nameOf(id) {
    const f = byId(id);
    if (f) return f.name;
    return String(id || '').replace(/^(system|user)\//, '').replace(/\.(ttf|otf|ttc)(@.*)?$/i, '');
  }

  function previewUrl(id, role, text, version) {
    const p = new URLSearchParams({ id: id || '', role: role || 'dialogue' });
    if (text) p.set('text', String(text).slice(0, 160));
    p.set('v', version || (catalog && catalog.version) || '0');
    return '/api/fonts/preview?' + p.toString();
  }

  /* What "Automatic" letters a style with, given the fonts picked so far. */
  function automatic(role) {
    if (st.mode === 'line') return { id: st.autoId, note: `Same as the rest of the chapter: ${st.autoName || nameOf(st.autoId)}` };
    const lead = st.draft.dialogue;
    if (role === 'narration' && lead) {
      const f = byId(lead);
      const id = (f && f.italic) || lead;
      return { id, note: `${nameOf(id)}, to match the dialogue font` };
    }
    const a = catalog.auto[role] || {};
    return { id: a.id, note: `${a.name || nameOf(a.id)}, ${WHY[a.why] || 'chosen for you'}` };
  }

  /* ------------------------------------------------------------ the dialog */
  function build() {
    dlg = node('dialog', { class: 'fontdlg', 'aria-labelledby': 'fdTitle' });
    // static markup only; every name and note is set as text below
    dlg.innerHTML = `
      <div class="fd">
        <header class="fd-head">
          <div class="fd-titles">
            <h2 id="fdTitle">Lettering font</h2>
            <p class="fd-sub" id="fdSub"></p>
          </div>
          <button class="btn ghost small" type="button" data-close>Close</button>
        </header>
        <div class="fd-roles" id="fdRolesWrap">
          <div class="seg" id="fdRoles" role="group" aria-label="Style to choose a font for"></div>
        </div>
        <div class="fd-tools">
          <input class="input" id="fdSearch" type="search" placeholder="Search fonts" aria-label="Search fonts" autocomplete="off" spellcheck="false">
          <button class="btn" type="button" id="fdAdd" title="Add a .ttf or .otf file you own, such as a professional comic font">Add a font file</button>
          <input type="file" id="fdFile" accept=".ttf,.otf,.ttc" hidden>
        </div>
        <div class="fd-list" id="fdList" role="radiogroup" aria-labelledby="fdTitle"></div>
        <footer class="fd-foot">
          <p class="fd-hint" id="fdHint"></p>
          <div class="fd-actions">
            <button class="btn ghost" type="button" data-close>Cancel</button>
            <button class="btn primary" type="button" id="fdApply">Use font</button>
          </div>
        </footer>
      </div>`;
    document.body.append(dlg);

    const roles = q('#fdRoles');
    for (const r of ROLES) roles.append(node('button', { type: 'button', 'data-role': r, 'aria-pressed': 'false', text: ROLE_LABEL[r] }));
    roles.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b || b.dataset.role === st.role) return;
      st.role = b.dataset.role;
      renderList(true);
    });
    for (const b of dlg.querySelectorAll('[data-close]')) b.addEventListener('click', () => dlg.close());
    q('#fdList').addEventListener('change', (e) => {
      if (e.target.name !== 'fdpick') return;
      if (st.mode === 'line') st.draftFont = e.target.value;
      else if (e.target.value) st.draft[st.role] = e.target.value;
      else delete st.draft[st.role];
      if (st.mode === 'chapter' && st.role === 'dialogue') markRoles();
      updateFoot();
    });
    // double-click (or Enter on a font) picks it and closes
    q('#fdList').addEventListener('dblclick', (e) => { if (e.target.closest('.fd-opt:not(.fd-broken)')) apply(); });
    q('#fdList').addEventListener('keydown', (e) => { if (e.key === 'Enter' && e.target.name === 'fdpick') { e.preventDefault(); apply(); } });
    let typing = null;
    q('#fdSearch').addEventListener('input', () => {
      clearTimeout(typing);
      typing = setTimeout(() => { st.query = q('#fdSearch').value; renderList(true); }, 120);
    });
    q('#fdAdd').addEventListener('click', () => q('#fdFile').click());
    q('#fdFile').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) addFont(f); });
    q('#fdApply').addEventListener('click', apply);
    dlg.addEventListener('close', () => { if (watcher) watcher.disconnect(); });
  }

  function option(value, title, tag, note, previewId, text, checked) {
    const input = node('input', { type: 'radio', name: 'fdpick', value });
    input.checked = checked;
    const img = node('img', { alt: '', width: '600', height: '120', 'data-src': previewUrl(previewId, st.role, text) });
    img.addEventListener('error', () => {
      const opt = img.closest('.fd-opt');
      if (!opt) return;
      opt.classList.add('fd-broken');
      opt.querySelector('input').disabled = true;
      img.replaceWith(node('span', { class: 'fd-chip-msg', text: 'Not available: it could not be downloaded.' }));
    });
    const label = node('label', { class: 'fd-opt' + (value ? '' : ' fd-auto') }, input,
      node('span', { class: 'fd-name', text: title }),
      tag ? node('span', { class: 'fd-tag', text: tag }) : null,
      note ? node('span', { class: 'fd-note', text: note }) : null,
      node('span', { class: 'fd-chip' }, img));
    return label;
  }

  function renderList(resetScroll) {
    const list = q('#fdList');
    const role = st.role;
    const current = st.mode === 'line' ? st.draftFont : (st.draft[role] || '');
    const text = st.mode === 'line' ? st.sample : '';
    const find = st.query.trim().toLowerCase();
    list.textContent = '';
    for (const b of q('#fdRoles').children) {
      b.classList.toggle('on', b.dataset.role === role);
      b.setAttribute('aria-pressed', String(b.dataset.role === role));
    }
    if (!find) {
      const a = automatic(role);
      list.append(option('', st.mode === 'line' ? 'Chapter font' : 'Automatic', '', a.note, a.id, text, current === ''));
    }
    let any = false;
    for (const [group, title] of GROUPS) {
      const fonts = catalog.fonts.filter((f) => f.group === group && (!find || f.name.toLowerCase().includes(find)));
      if (!fonts.length) continue;
      any = true;
      list.append(node('h3', { class: 'fd-group', text: title }));
      for (const f of fonts) {
        const tag = f.use ? USE_LABEL[f.use] : f.pro ? 'Pro comic font' : '';
        list.append(option(f.id, f.name, tag, f.group === 'free' ? f.note : '', f.id, text, current === f.id));
      }
    }
    if (!any && find) list.append(node('p', { class: 'fd-empty', text: 'No font has that name. Clear the search to see them all.' }));
    if (resetScroll) list.scrollTop = 0;
    watch();
    updateFoot();
  }

  function watch() {
    if (watcher) watcher.disconnect();
    const list = q('#fdList');
    const imgs = [...list.querySelectorAll('img[data-src]')];
    const show = (img) => { if (img.dataset.src) { img.src = img.dataset.src; delete img.dataset.src; } };
    if (!('IntersectionObserver' in window)) { imgs.forEach(show); return; }
    watcher = new IntersectionObserver((seen) => {
      for (const e of seen) if (e.isIntersecting) { show(e.target); watcher.unobserve(e.target); }
    }, { root: list, rootMargin: '400px 0px' });
    imgs.forEach((img) => watcher.observe(img));
  }

  /* Mark the styles whose font was changed in this dialog. */
  function markRoles() {
    for (const b of q('#fdRoles').children) {
      const r = b.dataset.role;
      b.classList.toggle('changed', (st.draft[r] || '') !== (st.initial[r] || ''));
    }
  }

  function changes() {
    if (st.mode === 'line') return st.draftFont !== st.initialFont ? { font: st.draftFont } : null;
    const out = {};
    for (const r of ROLES) if ((st.draft[r] || '') !== (st.initial[r] || '')) out[r] = st.draft[r] || '';
    return Object.keys(out).length ? out : null;
  }

  function updateFoot() {
    const c = changes();
    const n = c && st.mode === 'chapter' ? Object.keys(c).length : 1;
    const btn = q('#fdApply');
    btn.disabled = !c || st.busy;
    btn.textContent = st.busy ? 'Saving...' : n > 1 ? 'Use these fonts' : 'Use font';
    if (st.mode === 'chapter') markRoles();
  }

  async function apply() {
    const c = changes();
    if (!c || st.busy) { if (!c) dlg.close(); return; }
    st.busy = true;
    updateFoot();
    try {
      await st.onApply(st.mode === 'line' ? c.font : c);
      st.busy = false;
      dlg.close();
    } catch (err) {
      st.busy = false;
      updateFoot();
      q('#fdHint').textContent = err.message;
      q('#fdHint').classList.add('bad');
    }
  }

  async function addFont(file) {
    const btn = q('#fdAdd');
    btn.disabled = true;
    btn.textContent = 'Adding...';
    try {
      const res = await getJSON('/api/fonts?name=' + encodeURIComponent(file.name), { method: 'POST', body: file });
      catalog = { fonts: res.fonts, version: res.version, auto: res.auto };
      if (st.mode === 'line') st.draftFont = res.font.id; else st.draft[st.role] = res.font.id;
      st.query = '';
      q('#fdSearch').value = '';
      renderList(false);
      const pick = q(`#fdList input[value="${CSS.escape(res.font.id)}"]`);
      if (pick) { pick.focus({ preventScroll: true }); pick.closest('.fd-opt').scrollIntoView({ block: 'center' }); }
      say(`Added ${res.font.name}. It is chosen for ${st.mode === 'line' ? 'this line' : ROLE_LABEL[st.role].toLowerCase()}.`, 'info');
    } catch (err) {
      say(err.message);
    }
    btn.disabled = false;
    btn.textContent = 'Add a font file';
  }

  /* Open the picker.
       mode 'chapter': {choices: {style: id}, series, translated, role, onApply(changes)}
       mode 'line':    {font, role, sample, autoId, autoName, line, onApply(id)} */
  async function open(opts) {
    try { await load(); } catch (err) { say('Could not load the list of fonts: ' + err.message); return; }
    if (!dlg) build();
    const mode = opts.mode === 'line' ? 'line' : 'chapter';
    st = {
      mode,
      role: ROLES.includes(opts.role) ? opts.role : 'dialogue',
      initial: Object.assign({}, opts.choices || {}),
      draft: Object.assign({}, opts.choices || {}),
      initialFont: opts.font || '',
      draftFont: opts.font || '',
      sample: (opts.sample || '').trim(),
      autoId: opts.autoId || '',
      autoName: opts.autoName || '',
      query: '',
      busy: false,
      onApply: opts.onApply,
    };
    q('#fdSearch').value = '';
    q('#fdRolesWrap').hidden = mode === 'line';
    q('#fdTitle').textContent = mode === 'line' ? `Font for line ${opts.line || ''}`.trim() : 'Lettering font';
    q('#fdSub').textContent = mode === 'line'
      ? 'Only this line changes. The rest of the page keeps the chapter font.'
      : opts.series ? `Shared by every chapter of ${opts.series}.` : 'For this chapter.';
    const hint = q('#fdHint');
    hint.classList.remove('bad');
    hint.textContent = mode === 'line'
      ? 'The page is lettered again at once. Double-click a font to use it straight away.'
      : opts.translated
        ? `The ${opts.translated === 1 ? 'page' : opts.translated + ' pages'} already in English ${opts.translated === 1 ? 'is' : 'are'} lettered again in the new font. Nothing is translated again.`
        : 'Pages are lettered in these fonts when you translate them.';
    renderList(true);
    dlg.showModal();
    const checked = q('#fdList input:checked');
    if (checked) { checked.focus({ preventScroll: true }); checked.closest('.fd-opt').scrollIntoView({ block: 'nearest' }); }
  }

  window.FontPicker = { open, load, previewUrl, nameOf, ROLES, ROLE_LABEL };
})();
