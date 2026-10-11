'use strict';
/* Learning → Memory: browse, search, edit and teach what the AI remembers. Server data is only ever shown as text nodes. */
window.LearnMemory = (() => {
  const KINDS = [
    ['fact', 'Fact', 'Something true about you or your world'],
    ['preference', 'Preference', 'How you like things done'],
    ['project', 'Project', 'About something you are working on'],
    ['lesson', 'Lesson', 'A reminder learned from your 👎 feedback'],
    ['note', 'Note', 'Anything else worth keeping'],
    ['web', 'Web', 'Saved from a web search'],
    ['doc', 'Document', 'A piece of a document you taught'],
  ];
  const ADD_KINDS = ['fact', 'preference', 'project', 'note', 'lesson'];
  const SOURCES = { user: 'added by you', auto: 'auto-learned', model: 'saved by the AI', feedback: 'from feedback', web: 'from the web', import: 'imported' };
  const PAGE = 50, MAX_TEXT = 4000, MAX_DOC_CHARS = 2000000, MAX_IMPORT_BYTES = 50000000, IMPORT_BATCH = 5000;

  const st = { items: [], total: 0, kind: '', q: '', error: '', loading: false, seq: 0, previewSeq: 0, stats: null, mode: 'text' };
  const ui = {};
  const rows = new Map();

  const enc = encodeURIComponent;
  const kindInfo = (k) => KINDS.find((x) => x[0] === k) || [k, k, ''];
  const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : '');
  const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : null);
  const hostOf = (u) => { try { return new URL(u).host; } catch (e) { return u; } };
  const clip = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const cleanItems = (items) => (Array.isArray(items) ? items : []).filter((x) => x && Number.isInteger(x.id) && typeof x.text === 'string');
  const ctrlEnter = (e) => e.key === 'Enter' && (e.ctrlKey || e.metaKey);

  function kindSelect(kinds, value, label = 'Kind') {
    const all = kinds.includes(value) ? kinds : [...kinds, value];
    return h('select', { 'aria-label': label }, all.map((k) => h('option', { value: k, selected: k === value, title: kindInfo(k)[2] }, kindInfo(k)[1])));
  }

  async function busy(btn, fn) {
    btn.disabled = true;
    try { return await fn(); } catch (e) { fail(e); return undefined; } finally { btn.disabled = false; }
  }

  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
    const a = h('a', { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  /* ---------------------------------------------------------------- stats */
  async function loadStats() {
    try { st.stats = await api('memory/stats'); } catch (e) { st.stats = null; }
    renderStats();
  }

  function renderStats() {
    const s = st.stats, box = clear(ui.stats);
    fillKindFilter();
    if (!s) return;
    const tile = (n, label) => h('div', { class: 'la-stat' }, h('b', {}, String(n || 0)), h('span', {}, label));
    box.append(tile(s.total, s.total === 1 ? 'memory' : 'memories'), tile(s.pinned, 'pinned'));
    Object.entries(s.by_kind || {}).filter(([, n]) => n > 0)
      .forEach(([k, n]) => box.append(h('span', { class: 'badge', title: kindInfo(k)[2] }, `${kindInfo(k)[1]} ${n}`)));
    if (s.fts === false) box.append(h('span', { class: 'badge slow', title: 'This Python build has no SQLite FTS5, so memories are matched by simple word overlap.' }, 'basic search'));
  }

  function fillKindFilter() {
    const by = (st.stats && st.stats.by_kind) || {};
    const names = KINDS.map((k) => k[0]);
    Object.keys(by).forEach((k) => { if (!names.includes(k)) names.push(k); });
    const count = (n) => (st.stats ? ` (${n || 0})` : '');
    clear(ui.kind).append(h('option', { value: '' }, 'All kinds' + count(st.stats && st.stats.total)),
      names.map((k) => h('option', { value: k }, kindInfo(k)[1] + count(by[k]))));
    ui.kind.value = names.includes(st.kind) ? st.kind : '';
    syncClearButton();
  }

  function syncClearButton() {
    const n = st.kind ? ((st.stats && st.stats.by_kind) || {})[st.kind] : st.stats && st.stats.total;
    ui.clearBtn.textContent = st.kind ? `Delete all “${kindInfo(st.kind)[1]}”…` : 'Delete everything…';
    ui.clearBtn.disabled = !n;
  }

  function renderOffBanner() {
    const off = S.values && S.values.memory_enabled === false;
    clear(ui.off).hidden = !off;
    if (off) ui.off.append('Memory is switched off, so nothing here is added to your chats. ', h('button', { onclick: turnOn }, 'Turn it on'));
  }

  async function turnOn() {
    if (await setSetting('memory_enabled', true)) { toast('Memory is on', 'ok'); renderOffBanner(); }
  }

  /* ----------------------------------------------------------------- list */
  async function load(more = false) {
    const seq = ++st.seq;
    st.loading = true;
    if (!more) st.error = '';
    renderState();
    try {
      const offset = more ? st.items.length : 0;
      const d = await api(`memory/list?q=${enc(st.q)}&kind=${enc(st.kind)}&limit=${PAGE}&offset=${offset}`);
      if (seq !== st.seq) return;
      const fresh = cleanItems(d.items);
      if (more) {
        const known = new Set(st.items.map((x) => x.id));
        const add = fresh.filter((x) => !known.has(x.id));
        st.items = st.items.concat(add);
        addRows(add);
      } else {
        st.items = fresh;
        rows.clear();
        clear(ui.list);
        addRows(fresh);
      }
      st.total = Number(d.total) || 0;
    } catch (e) {
      if (seq !== st.seq) return;
      st.error = e.message;
    }
    st.loading = false;
    renderState();
  }

  function addRows(items) {
    items.forEach((it) => { const el = h('div', { class: 'lm-item', 'data-id': it.id }); rows.set(it.id, el); mount(el, it); ui.list.append(el); });
  }

  function renderState() {
    const box = clear(ui.state), foot = clear(ui.foot);
    if (st.error) {
      box.append(h('div', { class: 'note bad' }, 'Could not load memories: ' + st.error + ' ', h('button', { onclick: () => load() }, 'Try again')));
      return;
    }
    if (!st.items.length) {
      if (st.loading) box.append(h('div', { class: 'muted' }, 'Loading…'));
      else if (!st.q && !st.kind && !st.total) box.append(h('div', { class: 'la-empty' }, h('b', {}, 'Nothing remembered yet.'), h('div', {}, 'Add something above, say “remember that …” in Chat, press 🧠 under an answer, or teach a document.')));
      else box.append(h('div', { class: 'la-empty' }, 'No memories match your search.'));
      return;
    }
    foot.append(h('span', { class: 'muted small', role: 'status' }, `Showing ${st.items.length} of ${st.total}`));
    if (st.items.length < st.total) foot.append(h('button', { disabled: st.loading, onclick: () => load(true) }, 'Show more'));
  }

  function mount(el, it) {
    el.className = 'lm-item' + (it.enabled ? '' : ' off') + (it.pinned ? ' pinned' : '');
    const long = it.text.length > 280 || it.text.split('\n').length > 4;
    const text = h('div', { class: 'lm-text' + (long ? ' clamp' : '') }, it.text);
    const more = long ? h('button', { class: 'ghost small lm-more', 'aria-expanded': 'false', onclick: (e) => {
      const clamped = text.classList.toggle('clamp');
      e.currentTarget.textContent = clamped ? 'Show more' : 'Show less';
      e.currentTarget.setAttribute('aria-expanded', String(!clamped));
    } }, 'Show more') : null;
    const link = safeUrl(it.url);
    const meta = h('div', { class: 'lm-meta small muted' },
      h('span', { class: 'badge', title: kindInfo(it.kind)[2] }, kindInfo(it.kind)[1]),
      it.title ? h('b', { class: 'lm-title' }, it.title) : null,
      h('span', {}, SOURCES[it.source] || it.source),
      link ? h('a', { href: link, target: '_blank', rel: 'noopener noreferrer', title: link }, hostOf(link)) : (it.url ? h('span', { class: 'mono' }, clip(it.url, 60)) : null),
      it.uses > 0 ? h('span', { title: 'How many times this was added to a chat' }, `used ${it.uses}×`) : null,
      h('span', { title: 'Last changed' }, when(it.updated || it.created)));
    const pin = h('button', { class: 'ghost lm-pin' + (it.pinned ? ' on' : ''), 'aria-pressed': String(!!it.pinned), 'aria-label': 'Pinned',
      title: it.pinned ? 'Pinned: always remembered. Click to unpin' : 'Pin: always remember this, whatever you ask', onclick: () => change(el, it, { pinned: !it.pinned }) }, '📌');
    const sw = h('label', { class: 'switch', title: it.enabled ? 'On: can be added to chats. Click to switch off' : 'Off: never added to chats. Click to switch on' },
      h('input', { type: 'checkbox', 'aria-label': 'Use this memory', checked: !!it.enabled, onchange: (e) => change(el, it, { enabled: e.target.checked }) }), h('span'));
    clear(el).append(h('div', { class: 'lm-ctl' }, pin, sw), h('div', { class: 'lm-body' }, text, more, meta),
      h('div', { class: 'lm-actions' },
        h('button', { class: 'ghost', title: 'Edit', 'aria-label': 'Edit this memory', onclick: () => edit(el, it) }, '✏️'),
        h('button', { class: 'ghost', title: 'Delete', 'aria-label': 'Delete this memory', onclick: () => remove(it) }, '🗑')));
  }

  function edit(el, it) {
    el.className = 'lm-item editing';
    const text = h('textarea', { rows: 4, maxlength: MAX_TEXT, 'aria-label': 'Memory text' }, it.text);
    const kind = kindSelect(KINDS.map((k) => k[0]), it.kind);
    const title = h('input', { type: 'text', maxlength: 200, placeholder: 'Title (optional)', 'aria-label': 'Title', value: it.title || '' });
    const url = h('input', { type: 'text', maxlength: 2000, placeholder: 'Source link (optional)', 'aria-label': 'Source link', value: it.url || '' });
    const save = h('button', { class: 'primary' }, 'Save');
    const cancel = () => mount(el, it);
    const submit = () => busy(save, async () => {
      const changes = {}, t = text.value.trim(), u = url.value.trim();
      if (!t) { toast('A memory needs some text', 'err'); text.focus(); return; }
      if (u && !safeUrl(u)) { toast('The source link must start with http:// or https://', 'err'); url.focus(); return; }
      if (t !== it.text) changes.text = t;
      if (kind.value !== it.kind) changes.kind = kind.value;
      if (title.value.trim() !== (it.title || '')) changes.title = title.value.trim();
      if (u !== (it.url || '')) changes.url = u;
      if (!Object.keys(changes).length) { cancel(); return; }
      const next = await update(it, changes);
      if (next) { apply(next); loadStats(); }
    });
    save.addEventListener('click', submit);
    clear(el).append(h('div', { class: 'lm-edit', onkeydown: (e) => {
      if (e.key === 'Escape') cancel(); else if (ctrlEnter(e)) { e.preventDefault(); submit(); }
    } }, text,
    h('div', { class: 'row' }, h('div', {}, kind), h('div', { class: 'grow' }, title), h('div', { class: 'grow' }, url)),
    h('div', { class: 'row' }, save, h('button', { onclick: cancel }, 'Cancel'), h('span', { class: 'muted small' }, 'Ctrl+Enter saves · Esc cancels'))));
    text.focus();
    text.setSelectionRange(text.value.length, text.value.length);
  }

  async function update(it, fields) {
    try {
      const d = await api('memory/update', { id: it.id, ...fields });
      if (!d || !d.item) throw new Error('The server sent an unexpected reply');
      return d.item;
    } catch (e) { fail(e); return null; }
  }

  async function change(el, it, fields) {
    el.classList.add('busy');
    const next = await update(it, fields);
    if (next) { apply(next); loadStats(); } else mount(el, it);
  }

  function apply(next) {
    const i = st.items.findIndex((x) => x.id === next.id);
    if (i >= 0) st.items[i] = next;
    const el = rows.get(next.id);
    if (el) mount(el, next);
  }

  async function remove(it) {
    if (!(await confirmBox('Delete this memory?', clip(it.text, 240), true))) return;
    try { await api('memory/delete', { id: it.id }); } catch (e) { fail(e); return; }
    st.items = st.items.filter((x) => x.id !== it.id);
    st.total = Math.max(0, st.total - 1);
    const el = rows.get(it.id);
    if (el) el.remove();
    rows.delete(it.id);
    loadStats();
    if (!st.items.length && st.total > 0) load(); else renderState();
  }

  async function clearMemories() {
    const kind = st.kind, label = kind ? `every “${kindInfo(kind)[1]}” memory` : 'EVERY memory';
    if (!(await confirmBox('Delete memories?', `This permanently deletes ${label}. Export first if you might want them back.`, true))) return;
    const d = await api('memory/clear', kind ? { kind } : {});
    toast(`Deleted ${d.deleted || 0}`, 'ok');
    refresh();
  }

  /* ------------------------------------------------------ import / export */
  async function exportAll() {
    const d = await api('memory/export');
    const items = Array.isArray(d.items) ? d.items : [];
    if (!items.length) { toast('Nothing to export yet', 'err'); return; }
    const now = new Date().toISOString();
    download(`duckstoken-memory-${now.slice(0, 10)}.json`, JSON.stringify({ format: 'duckstoken-memory', version: 1, exported: now, items }, null, 2));
    toast(`Exported ${items.length} memories`, 'ok');
  }

  async function importFile(file) {
    if (file.size > MAX_IMPORT_BYTES) { toast('That file is too large to import', 'err'); return; }
    let data;
    try { data = JSON.parse(await file.text()); } catch (e) { toast('That file is not valid JSON', 'err'); return; }
    const list = Array.isArray(data) ? data : data && Array.isArray(data.items) ? data.items : null;
    const items = (list || []).filter((x) => x && typeof x === 'object' && typeof x.text === 'string');
    if (!items.length) { toast('No memories found in that file (expected an export from this page)', 'err'); return; }
    let added = 0;
    try {
      for (let i = 0; i < items.length; i += IMPORT_BATCH) added += Number((await api('memory/import', { items: items.slice(i, i + IMPORT_BATCH) })).added) || 0;
    } finally { refresh(); }
    toast(`Imported ${added} new ${added === 1 ? 'memory' : 'memories'}` + (added < items.length ? ` (${items.length - added} already known or skipped)` : ''), 'ok');
  }

  /* ----------------------------------------------------------- add / teach */
  function buildAdd() {
    const text = h('textarea', { rows: 2, maxlength: MAX_TEXT, placeholder: 'e.g. “I prefer metric units” or “My dog is called Bruno”', 'aria-label': 'New memory' });
    const kind = kindSelect(ADD_KINDS, 'fact');
    const pin = h('input', { type: 'checkbox' });
    const btn = h('button', { class: 'primary' }, 'Remember');
    const submit = () => busy(btn, async () => {
      const t = text.value.trim();
      if (!t) { text.focus(); return; }
      await api('memory/add', { text: t, kind: kind.value, pinned: pin.checked });
      text.value = ''; pin.checked = false;
      toast('Remembered', 'ok');
      refresh();
    });
    btn.addEventListener('click', submit);
    text.addEventListener('keydown', (e) => { if (ctrlEnter(e)) { e.preventDefault(); submit(); } });
    return h('div', { class: 'card' }, h('h2', {}, '➕ Add a memory'),
      h('div', { class: 'muted small' }, 'One short, self-contained sentence works best. Pinned memories are added to every question.'),
      text, h('div', { class: 'row lm-gap' }, kind, h('label', { class: 'row small' }, pin, 'Pin: always remember'), h('span', { class: 'grow' }), btn));
  }

  function buildTeach() {
    const title = h('input', { type: 'text', maxlength: 200, placeholder: 'Title (optional)', 'aria-label': 'Document title' });
    const text = h('textarea', { rows: 6, placeholder: 'Paste the text here…', 'aria-label': 'Document text' });
    const url = h('input', { type: 'text', maxlength: 2000, placeholder: 'https://example.com/page', 'aria-label': 'Web page address' });
    const file = h('input', { type: 'file', accept: '.txt,.md,.markdown,.rst,.csv,.tsv,.log,.json,text/plain', hidden: true, onchange: async (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (!f) return;
      if (f.size > MAX_DOC_CHARS) { toast('That file is too big (limit about 2 MB of text)', 'err'); return; }
      try {
        const body = await f.text();
        if (body.includes('\u0000')) { toast('That does not look like a text file', 'err'); return; }
        text.value = body;
        if (!title.value.trim()) title.value = f.name.replace(/\.[^.]+$/, '');
      } catch (err) { fail(err); }
    } });
    const textPane = h('div', {}, text, h('div', { class: 'row lm-gap' }, h('button', { onclick: () => file.click() }, 'Load a text file…'), file));
    const urlPane = h('div', { hidden: true }, url, h('div', { class: 'muted small' }, 'The page is downloaded by Duckstoken (private and local addresses are blocked), and only its readable text is kept.'));
    const modes = [['text', 'Paste text', textPane], ['url', 'Web page', urlPane]];
    const tabs = modes.map(([m, label]) => h('button', { class: 'la-chip', 'aria-pressed': String(m === st.mode), onclick: () => {
      st.mode = m;
      modes.forEach(([k, , pane], i) => { pane.hidden = k !== m; tabs[i].setAttribute('aria-pressed', String(k === m)); });
    } }, label));
    const go = h('button', { class: 'primary' }, 'Teach this');
    go.addEventListener('click', () => busy(go, async () => {
      const body = {};
      if (title.value.trim()) body.title = title.value.trim();
      if (st.mode === 'url') {
        if (!safeUrl(url.value.trim())) { toast('Enter a web address starting with http:// or https://', 'err'); url.focus(); return; }
        body.url = url.value.trim();
      } else {
        if (!text.value.trim()) { toast('Paste some text first', 'err'); text.focus(); return; }
        if (text.value.length > MAX_DOC_CHARS) { toast('That text is too long (limit about 2 million characters)', 'err'); return; }
        body.text = text.value;
        body.title = body.title || 'Untitled';
      }
      go.textContent = 'Reading…';
      try {
        const d = await api('memory/add_doc', body);
        const n = Number(d.chunks) || 0;
        toast(n ? `Learned “${d.title || body.title || 'document'}” in ${n} ${n === 1 ? 'piece' : 'pieces'}` : 'Already known: nothing new to add', 'ok');
        title.value = text.value = url.value = '';
        refresh();
      } finally { go.textContent = 'Teach this'; }
    }));
    return h('details', { class: 'card lm-fold' }, h('summary', {}, h('b', {}, '📄 Teach a document')),
      h('div', { class: 'muted small' }, 'Long text is split into pieces; only the pieces relevant to a question are shown to the AI.'),
      h('div', { class: 'row lm-gap' }, tabs), h('div', { class: 'lm-gap' }, title), h('div', { class: 'lm-gap' }, textPane, urlPane), h('div', { class: 'row lm-gap' }, go));
  }

  /* -------------------------------------------------------------- preview */
  async function preview() {
    const q = ui.pq.value.trim().slice(0, 2000);
    if (!q) { ui.pq.focus(); return; }
    const seq = ++st.previewSeq;
    ui.pbtn.disabled = true;
    clear(ui.pout).append(h('div', { class: 'muted' }, 'Looking…'));
    try {
      const d = await api('memory/preview?q=' + enc(q));
      if (seq === st.previewSeq) renderPreview(d);
    } catch (e) {
      if (seq === st.previewSeq) clear(ui.pout).append(h('div', { class: 'note bad' }, e.message));
    } finally { if (seq === st.previewSeq) ui.pbtn.disabled = false; }
  }

  function renderPreview(d) {
    const out = clear(ui.pout), ids = new Set((d.ids || []).map(Number)), ex = d.examples || {}, items = Array.isArray(d.items) ? d.items : [];
    const block = (title, hint, node) => h('div', { class: 'lm-pv' }, h('h3', {}, title), hint ? h('div', { class: 'muted small' }, hint) : null, node);
    const pre = (t) => h('pre', { class: 'lm-pre' }, t);
    const none = (t) => h('div', { class: 'la-empty' }, t);
    if (S.values && S.values.memory_enabled === false) out.append(h('div', { class: 'note warn' }, 'Memory is switched off, so in real chats none of the memory below would be added.'));
    out.append(block('Memory added to the prompt', `${ids.size} ${ids.size === 1 ? 'memory' : 'memories'} used`, d.text ? pre(d.text) : none('Nothing in memory matches this message, so nothing is added.')));
    if (items.length) {
      out.append(block('Best matches', 'Ranked by relevance. Pinned memories always come first.', h('ul', { class: 'lm-matches' }, items.map((it) => h('li', { class: ids.has(Number(it.id)) ? '' : 'skipped' },
        h('span', { class: 'badge' }, kindInfo(it.kind)[1]), it.pinned ? h('span', { class: 'badge good' }, 'pinned') : null,
        typeof it.score === 'number' ? h('span', { class: 'muted small mono', title: 'Relevance score' }, it.score.toFixed(2)) : null,
        ' ', clip(String(it.text), 160), ids.has(Number(it.id)) ? null : h('span', { class: 'muted small' }, ' (left out: size limit)'))))));
    }
    out.append(block('Examples and reminders from your feedback', `${(ex.example_ids || []).length} example(s) · ${(ex.lesson_ids || []).length} reminder(s)`,
      ex.text ? pre(ex.text) : none('No liked answers or 👎 notes are similar enough to this message.')));
    const wg = S.values && S.values.web_grounding;
    out.append(block('Web search', '', h('div', {}, d.web_would_search ? '🌐 A web search would run first, because this looks like it needs fresh facts.'
      : wg === 'off' ? 'Web research is off: nothing is searched.'
        : wg === 'tool' ? 'No automatic search; the AI may still decide to search by itself (tool mode).'
          : 'No search: this message does not look like it needs one.')));
  }

  function buildPreview() {
    ui.pq = h('input', { type: 'text', maxlength: 2000, placeholder: 'Type a message you might send…', 'aria-label': 'Message to preview', onkeydown: (e) => { if (e.key === 'Enter') preview(); } });
    ui.pbtn = h('button', { class: 'primary', onclick: preview }, 'Preview');
    ui.pout = h('div', { 'aria-live': 'polite' });
    return h('details', { class: 'card lm-fold' }, h('summary', {}, h('b', {}, '🔍 What will the AI see for a message?')),
      h('div', { class: 'muted small' }, 'Shows the memories, examples and reminders that would be added to the prompt. Nothing is sent to the model.'),
      h('div', { class: 'row lm-gap' }, h('div', { class: 'grow' }, ui.pq), ui.pbtn), ui.pout);
  }

  /* ---------------------------------------------------------------- shell */
  function refresh() { loadStats(); load(); }

  function init(el) {
    ui.off = h('div', { class: 'note warn', hidden: true });
    ui.stats = h('div', { class: 'la-stats', role: 'status' });
    const importer = h('input', { type: 'file', accept: '.json,application/json', hidden: true, onchange: (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (f) importFile(f).catch(fail);
    } });
    ui.clearBtn = h('button', { class: 'danger', onclick: (e) => busy(e.currentTarget, clearMemories) }, 'Delete everything…');
    ui.kind = h('select', { 'aria-label': 'Filter by kind', onchange: (e) => { st.kind = e.target.value; syncClearButton(); load(); } });
    ui.search = h('input', { type: 'text', maxlength: 200, placeholder: '🔍 Search memories…', 'aria-label': 'Search memories', oninput: debounce((e) => { st.q = e.target.value.trim(); load(); }, 250),
      onkeydown: (e) => { if (e.key === 'Escape' && e.target.value) { e.target.value = ''; st.q = ''; load(); } } });
    ui.state = h('div', { 'aria-live': 'polite' });
    ui.list = h('div', { class: 'lm-list' });
    ui.foot = h('div', { class: 'row lm-foot' });
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '🧠 What the AI remembers about you'),
        h('p', { class: 'muted' }, 'Short notes about you and your projects, plus documents you teach. The most relevant ones are quietly added to each question. Everything stays on this PC.'),
        ui.off, ui.stats,
        h('div', { class: 'row lm-gap' }, h('button', { onclick: (e) => busy(e.currentTarget, exportAll) }, 'Export JSON'),
          h('button', { onclick: () => importer.click() }, 'Import JSON…'), importer, h('span', { class: 'grow' }), ui.clearBtn)),
      buildAdd(), buildTeach(), buildPreview(),
      h('div', { class: 'card' }, h('div', { class: 'row' }, h('div', { class: 'grow' }, ui.search), h('div', {}, ui.kind),
        h('button', { class: 'ghost', title: 'Reload', 'aria-label': 'Reload', onclick: refresh }, '↻')),
      ui.state, ui.list, ui.foot));
  }

  function show() {
    renderOffBanner();
    refresh();
  }

  return { init, show };
})();
