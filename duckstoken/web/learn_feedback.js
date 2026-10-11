'use strict';
/* Learning → Feedback: every logged answer with its rating, correction and big-model answer. Server data is only ever shown as text nodes. */
window.LearnFeedback = (() => {
  const PAGE = 20, MAX_NOTE = 2000, MAX_CONVERSATION = 12, MAX_CONVERSATION_CHARS = 1500;
  const FILTERS = [
    { id: 'all', label: 'All', stat: 'total', param: null },
    { id: 'up', label: '👍 Liked', stat: 'up', param: ['rating', '1'] },
    { id: 'down', label: '👎 Disliked', stat: 'down', param: ['rating', '-1'] },
    { id: 'corrected', label: '✏️ Corrected', stat: 'corrected', param: ['corrected', '1'] },
    { id: 'teacher', label: '🐘 With big-model answer', stat: 'with_teacher', param: ['with_teacher', '1'] },
  ];
  const ROLES = { system: 'System', user: 'You', assistant: 'AI', tool: 'Tool' };

  const st = { items: [], total: 0, filter: 'all', q: '', page: 0, error: '', loading: false, seq: 0, stats: null };
  const ui = {};
  const open = new Set();
  const rows = new Map();

  const enc = encodeURIComponent;
  const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : '');
  const safeUrl = (u) => (/^https?:\/\//i.test(u || '') ? u : null);
  const flat = (s) => String(s || '').replace(/\s+/g, ' ').trim();
  const clip = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  const cleanItems = (items) => (Array.isArray(items) ? items : []).filter((x) => x && Number.isInteger(x.id));
  const str = (v) => (typeof v === 'string' ? v : '');

  /* ---------------------------------------------------------------- stats */
  async function loadStats() {
    try { st.stats = await api('feedback/stats'); } catch (e) { st.stats = null; }
    renderStats();
  }

  function buildFilters() {
    ui.chips = FILTERS.map((f) => {
      const num = h('b', {}, '–');
      return { f, num, btn: h('button', { class: 'la-stat la-filter', onclick: () => setFilter(f.id) }, num, h('span', {}, f.label)) };
    });
    renderStats();
    return ui.chips.map((c) => c.btn);
  }

  function renderStats() {
    ui.chips.forEach(({ f, btn, num }) => {
      num.textContent = st.stats ? String(st.stats[f.stat] || 0) : '–';
      btn.setAttribute('aria-pressed', String(st.filter === f.id));
    });
  }

  function setFilter(id) {
    st.filter = id;
    st.page = 0;
    renderStats();
    load();
  }

  function renderLoggingBanner() {
    const off = S.values && S.values.record_exchanges === false;
    clear(ui.off).hidden = !off;
    if (off) ui.off.append('Logging is switched off, so new answers are not being recorded (and 👍/👎 in Chat will not work). ', h('button', { onclick: turnOn }, 'Turn it on'));
  }

  async function turnOn() {
    if (await setSetting('record_exchanges', true)) { toast('Logging is on', 'ok'); renderLoggingBanner(); }
  }

  /* ----------------------------------------------------------------- list */
  async function load() {
    const seq = ++st.seq;
    st.loading = true;
    st.error = '';
    renderState();
    try {
      const f = FILTERS.find((x) => x.id === st.filter);
      const qs = [`limit=${PAGE}`, `offset=${st.page * PAGE}`, st.q ? 'q=' + enc(st.q) : '', f.param ? `${f.param[0]}=${f.param[1]}` : ''].filter(Boolean).join('&');
      const d = await api('feedback/list?' + qs);
      if (seq !== st.seq) return;
      st.total = Number(d.total) || 0;
      const items = cleanItems(d.items);
      if (!items.length && st.page > 0 && st.total > 0) { st.page = Math.ceil(st.total / PAGE) - 1; load(); return; }
      st.items = items;
      rows.clear();
      clear(ui.list).append(...items.map((it) => { const el = h('article'); rows.set(it.id, el); mount(el, it); return el; }));
    } catch (e) {
      if (seq !== st.seq) return;
      st.error = e.message;
    }
    st.loading = false;
    renderState();
  }

  function renderState() {
    const box = clear(ui.state), pager = clear(ui.pager);
    if (st.error) {
      box.append(h('div', { class: 'note bad' }, 'Could not load the log: ' + st.error + ' ', h('button', { onclick: load }, 'Try again')));
      return;
    }
    if (!st.items.length) {
      if (st.loading) box.append(h('div', { class: 'muted' }, 'Loading…'));
      else if (st.stats && !st.stats.total) box.append(h('div', { class: 'la-empty' }, h('b', {}, 'No answers logged yet.'), h('div', {}, 'Chat with the AI, then use 👍 / 👎 / ✏️ under its answers (or rate them here).')));
      else box.append(h('div', { class: 'la-empty' }, 'No answers match this filter.'));
      return;
    }
    const pages = Math.max(1, Math.ceil(st.total / PAGE)), first = st.page * PAGE + 1;
    pager.append(h('button', { disabled: st.loading || st.page === 0, onclick: () => { st.page--; load(); } }, '‹ Newer'),
      h('span', { class: 'muted small', role: 'status' }, `${first}–${first + st.items.length - 1} of ${st.total} · page ${st.page + 1} of ${pages}`),
      h('button', { disabled: st.loading || st.page + 1 >= pages, onclick: () => { st.page++; load(); } }, 'Older ›'));
  }

  function tags(it) {
    const tag = (cls, text, title) => h('span', { class: 'badge ' + cls, title }, text);
    const sources = Array.isArray(it.web_sources) ? it.web_sources.length : 0;
    return h('span', { class: 'lf-tags' },
      it.rating === 1 ? tag('good', '👍 liked') : it.rating === -1 ? tag('error', '👎 disliked') : null,
      it.correction ? tag('ok', '✏️ corrected', 'You wrote a better answer') : null,
      it.teacher_answer ? tag('lf-big', '🐘 big model', 'The big model also answered this') : null,
      it.used_tools ? tag('', '🔧 tools', 'The AI used tools') : null,
      sources ? tag('', `🌐 ${sources}`, 'Web sources were used') : null,
      h('span', { class: 'muted small' }, [when(it.created), it.model, it.tps ? `${Number(it.tps).toFixed(1)} tok/s` : ''].filter(Boolean).join(' · ')));
  }

  function mount(el, it) {
    const isOpen = open.has(it.id);
    el.className = 'lf-item' + (it.rating === 1 ? ' up' : it.rating === -1 ? ' down' : '') + (isOpen ? ' open' : '');
    const icon = (label, title, pressed, fn) => h('button', { class: 'ghost lf-ic' + (pressed ? ' on' : ''), title, 'aria-label': title, 'aria-pressed': pressed == null ? null : String(pressed), onclick: fn }, label);
    const head = h('button', { class: 'lf-head', 'aria-expanded': String(isOpen), onclick: () => toggle(el, it) },
      h('span', { class: 'lf-chev', 'aria-hidden': 'true' }, isOpen ? '▾' : '▸'),
      h('span', { class: 'lf-sum' },
        h('span', { class: 'lf-prompt' }, clip(flat(it.prompt), 200) || '(no question text)'),
        h('span', { class: 'lf-snippet' }, (it.correction ? '✏️ ' : '') + clip(flat(it.correction || it.answer), 240)),
        tags(it)));
    const actions = h('div', { class: 'lf-actions' },
      icon('👍', it.rating === 1 ? 'Remove 👍' : 'Good answer: show it as an example next time', it.rating === 1, () => rate(el, it, it.rating === 1 ? 0 : 1)),
      icon('👎', it.rating === -1 ? 'Edit or remove 👎' : 'Bad answer: say what was wrong', it.rating === -1, () => dislike(el, it)),
      icon('✏️', 'Write the answer you wanted', !!it.correction, () => correct(el, it)),
      icon('🗑', 'Delete this entry', null, () => remove(it)));
    clear(el).append(h('div', { class: 'lf-row' }, head, actions));
    if (isOpen) el.append(detail(it));
  }

  function toggle(el, it) {
    if (open.has(it.id)) open.delete(it.id); else open.add(it.id);
    mount(el, it);
    const head = el.querySelector('.lf-head');
    if (head) head.focus();
  }

  function detail(it) {
    const block = (label, text, cls = '') => (text ? h('div', { class: 'lf-block ' + cls }, h('div', { class: 'lf-label' }, label), h('div', { class: 'lf-text' }, text)) : null);
    const sources = (Array.isArray(it.web_sources) ? it.web_sources : []).filter((s) => s && typeof s === 'object');
    const convo = h('div', { class: 'lf-convo' });
    const more = h('button', { onclick: (e) => showConversation(it, convo, e.currentTarget) }, 'Show the conversation before this');
    return h('div', { class: 'lf-detail' },
      block('You asked', str(it.prompt)),
      block('The AI answered', str(it.answer)),
      block('Your correction', str(it.correction), 'good'),
      block(`Big-model answer${it.teacher_model ? ' (' + it.teacher_model + ')' : ''}`, str(it.teacher_answer), 'big'),
      block('Your note on the 👎', str(it.note), 'bad'),
      sources.length ? h('div', { class: 'lf-block' }, h('div', { class: 'lf-label' }, 'Web sources'),
        sources.map((s) => h('div', { class: 'small' }, safeUrl(s.url) ? h('a', { href: s.url, target: '_blank', rel: 'noopener noreferrer' }, str(s.title) || s.url) : str(s.title || s.url)))) : null,
      h('div', { class: 'row small muted' }, `${it.tokens || 0} tokens`, it.effort ? ` · reasoning ${it.effort}` : '', ` · entry #${it.id}`),
      h('div', {}, more), convo);
  }

  function textOf(m) {
    const c = m.content;
    let text = typeof c === 'string' ? c
      : Array.isArray(c) ? c.map((p) => (p && typeof p.text === 'string' ? p.text : p && p.type ? `[${String(p.type).slice(0, 20)}]` : '')).filter(Boolean).join('\n') : '';
    const calls = Array.isArray(m.tool_calls) ? m.tool_calls.map((x) => x && x.function && x.function.name).filter((n) => typeof n === 'string') : [];
    if (calls.length) text += (text ? '\n' : '') + `[called ${calls.join(', ')}]`;
    return clip(text, MAX_CONVERSATION_CHARS);
  }

  async function showConversation(it, box, btn) {
    btn.disabled = true;
    try {
      const d = await api('feedback/get?id=' + it.id);
      const msgs = Array.isArray(d.item && d.item.messages) ? d.item.messages.filter((m) => m && typeof m === 'object') : [];
      clear(box);
      if (!msgs.length) box.append(h('div', { class: 'muted small' }, 'No earlier messages were stored for this answer.'));
      else if (msgs.length > MAX_CONVERSATION) box.append(h('div', { class: 'muted small' }, `Showing the last ${MAX_CONVERSATION} of ${msgs.length} messages.`));
      msgs.slice(-MAX_CONVERSATION).forEach((m) => box.append(h('div', { class: 'lf-msg ' + (m.role === 'user' ? 'user' : '') },
        h('b', {}, ROLES[m.role] || 'Message'), h('div', { class: 'lf-text' }, textOf(m) || '(empty)'))));
      btn.remove();
    } catch (e) { fail(e); btn.disabled = false; }
  }

  /* -------------------------------------------------------------- actions */
  const controls = (el) => [...el.querySelectorAll('button')];
  const focusIndex = (el) => controls(el).indexOf(document.activeElement);

  function apply(next, focus) {
    const i = st.items.findIndex((x) => x.id === next.id);
    if (i < 0) return;
    st.items[i] = { ...st.items[i], ...next };
    const el = rows.get(next.id);
    mount(el, st.items[i]);
    if (focus >= 0 && controls(el)[focus]) controls(el)[focus].focus();
  }

  /* `focus` is the index of the button that was focused before the row was rebuilt, so keyboard users keep their place */
  async function send(el, path, body, focus = focusIndex(el)) {
    el.classList.add('busy');
    try {
      const d = await api(path, body);
      if (!d || !d.item) throw new Error('The server sent an unexpected reply');
      apply(d.item, focus);
      loadStats();
    } catch (e) { fail(e); } finally { el.classList.remove('busy'); }
  }

  const rate = (el, it, rating, note = '', focus) => send(el, 'feedback/rate', { id: it.id, rating, note }, focus);

  async function dislike(el, it) {
    const had = it.rating === -1, focus = focusIndex(el);
    const note = h('textarea', { rows: 4, maxlength: MAX_NOTE, 'aria-label': 'What was wrong?', placeholder: 'What was wrong? (optional)' }, had ? str(it.note) : '');
    const pending = modal(had ? 'Edit your 👎' : 'Bad answer',
      h('div', {}, h('div', { class: 'lf-ctx' }, clip(flat(it.prompt), 160)),
        h('p', { class: 'muted small' }, 'A note becomes a reminder the AI sees when you ask something similar.'), note),
      [{ label: 'Cancel', value: false }, ...(had ? [{ label: 'Remove 👎', value: 'clear' }] : []), { label: 'Save', value: true, primary: true }]);
    note.focus();
    const r = await pending;
    if (r === true) rate(el, it, -1, note.value.trim(), focus);
    else if (r === 'clear') rate(el, it, 0, '', focus);
  }

  async function correct(el, it) {
    const start = str(it.correction) || str(it.answer), focus = focusIndex(el);
    const ta = h('textarea', { rows: 12, class: 'lf-editor', 'aria-label': 'The answer you wanted' }, start);
    const pending = modal('Write the answer you wanted',
      h('div', {}, h('div', { class: 'lf-ctx' }, clip(flat(it.prompt), 160)),
        h('p', { class: 'muted small' }, 'Saved as a gold example: similar questions will be shown this answer, and it can become training data.'), ta),
      [{ label: 'Cancel', value: false }, ...(it.correction ? [{ label: 'Remove correction', value: 'clear' }] : []), { label: 'Save correction', value: true, primary: true }]);
    ta.focus();
    const r = await pending;
    if (r === 'clear') { send(el, 'feedback/correct', { id: it.id, correction: '' }, focus); return; }
    if (r !== true) return;
    const text = ta.value.trim();
    if (!text) toast('Write the answer first, or use “Remove correction”', 'err');
    else if (text === start.trim()) toast('Nothing changed yet. Edit the answer first', 'err');
    else send(el, 'feedback/correct', { id: it.id, correction: text }, focus);
  }

  async function remove(it) {
    if (!(await confirmBox('Delete this entry?', 'This removes the logged question and answer, including your rating and correction.', true))) return;
    try { await api('feedback/delete', { id: it.id }); } catch (e) { fail(e); return; }
    open.delete(it.id);
    toast('Deleted', 'ok');
    refresh();
  }

  /* ---------------------------------------------------------------- shell */
  function refresh() { loadStats(); load(); }

  function init(el) {
    ui.off = h('div', { class: 'note warn', hidden: true });
    ui.stats = h('div', { class: 'la-stats', role: 'group', 'aria-label': 'Filter by rating' }, buildFilters());
    ui.search = h('input', { type: 'text', maxlength: 200, placeholder: '🔍 Search questions, answers and notes…', 'aria-label': 'Search the log',
      oninput: debounce((e) => { st.q = e.target.value.trim(); st.page = 0; load(); }, 300),
      onkeydown: (e) => { if (e.key === 'Escape' && e.target.value) { e.target.value = ''; st.q = ''; st.page = 0; load(); } } });
    ui.state = h('div', { 'aria-live': 'polite' });
    ui.list = h('div', { class: 'lf-list' });
    ui.pager = h('div', { class: 'row lf-pager' });
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '👍 Your feedback on the AI’s answers'),
        h('p', { class: 'muted' }, '👍 and ✏️ answers are shown to the AI as examples when you ask something similar; 👎 notes become reminders. Rated, corrected and big-model answers can also become training data.'),
        ui.off, ui.stats),
      h('div', { class: 'card' }, h('div', { class: 'row' }, h('div', { class: 'grow' }, ui.search),
        h('button', { class: 'ghost', title: 'Reload', 'aria-label': 'Reload', onclick: refresh }, '↻')),
      ui.state, ui.list, ui.pager));
  }

  function show() {
    renderLoggingBanner();
    refresh();
  }

  return { init, show };
})();
