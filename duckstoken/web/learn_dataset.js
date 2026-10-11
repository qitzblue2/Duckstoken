'use strict';
/* Learning → Training data: review the examples Duckstoken derived from your chats, then download a training bundle.
   Server data is only ever shown as text nodes. */
window.LearnDataset = (() => {
  const PAGE = 20, MAX_ANSWER = 200000, CLAMP_CHARS = 700, CLAMP_LINES = 8, MAX_EARLIER_CHARS = 1500;
  const STATUS_TILES = [
    { id: '', label: 'All', stat: 'total' },
    { id: 'pending', label: 'To review', stat: 'pending' },
    { id: 'approved', label: 'Approved', stat: 'approved' },
    { id: 'rejected', label: 'Rejected', stat: 'rejected' },
  ];
  const KIND_TILES = [
    { id: 'sft', label: 'Example answers', stat: 'sft', title: 'A question with one good answer to copy' },
    { id: 'pref', label: 'Better vs worse', stat: 'pref', title: 'A question with a better answer and a worse one' },
  ];
  const KINDS = { sft: ['📘 example answer', 'A question with one good answer for the AI to copy'], pref: ['⚖️ better vs worse', 'The same question with a better and a worse answer'] };
  const SOURCES = { corrected: '✏️ you corrected it', teacher: '🐘 big-model answer', liked: '👍 you liked it' };
  const STATUS_BADGE = { pending: ['to review', 'ok'], approved: ['approved', 'good'], rejected: ['rejected', 'error'] };
  const ROLES = { system: 'System', user: 'You', assistant: 'AI' };

  const st = { items: [], total: 0, status: '', kind: '', q: '', page: 0, error: '', loading: false, seq: 0, stats: null, touched: false, exporting: false };
  const ui = {};
  const rows = new Map();

  const enc = encodeURIComponent;
  const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
  const str = (v) => (typeof v === 'string' ? v : '');
  const when = (ts) => (num(ts) > 0 ? new Date(num(ts) * 1000).toLocaleString() : '');
  const clip = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const cssWord = (s) => (/^[a-z_]{1,20}$/.test(s) ? s : '');
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
  /* Replaces the content of `el`; unlike a bare append(), skips null and false children. */
  const fill = (el, ...kids) => clear(el).append(...kids.flat(Infinity).filter((k) => k != null && k !== false));
  const cleanItems = (items) => (Array.isArray(items) ? items : []).filter((x) => x && Number.isInteger(x.id));

  function contentText(c) {
    if (typeof c === 'string') return c;
    return Array.isArray(c) ? c.map((p) => (p && typeof p.text === 'string' ? p.text : '')).filter(Boolean).join('\n') : '';
  }

  const messagesOf = (it) => (Array.isArray(it.messages) ? it.messages : []).filter((m) => m && typeof m === 'object');

  /* ----------------------------------------------------------------- stats */
  async function loadStats() {
    try { st.stats = await api('dataset/stats'); } catch (e) { st.stats = null; }
    renderStats();
  }

  function buildTiles(tiles, onclick) {
    return tiles.map((t) => {
      const count = h('b', {}, '–');
      return { t, count, btn: h('button', { class: 'lb-stat', title: t.title, onclick: () => onclick(t.id) }, count, h('span', {}, t.label)) };
    });
  }

  function renderStats() {
    const stat = (key) => (st.stats ? String(num(st.stats[key])) : '–');
    ui.statusTiles.forEach(({ t, count, btn }) => { count.textContent = stat(t.stat); btn.setAttribute('aria-pressed', String(st.status === t.id)); });
    ui.kindTiles.forEach(({ t, count, btn }) => { count.textContent = stat(t.stat); btn.setAttribute('aria-pressed', String(st.kind === t.id)); });
    renderExport();
  }

  function setStatusFilter(id) {
    st.touched = true;
    st.status = id;
    st.page = 0;
    renderStats();
    load();
  }

  function setKindFilter(id) {
    st.touched = true;
    st.kind = st.kind === id ? '' : id;
    st.page = 0;
    renderStats();
    load();
  }

  async function rebuild() {
    ui.rebuild.disabled = true;
    ui.rebuild.textContent = 'Rebuilding…';
    try {
      const d = await api('dataset/refresh', {});
      const added = num(d.added), total = num(d.total);
      toast(added ? `Found ${plural(added, 'new example', 'new examples')}. ${total} in total.` : `No new examples. ${total} in total.`, 'ok');
      refresh();
    } catch (e) { fail(e); }
    ui.rebuild.disabled = false;
    ui.rebuild.textContent = '🔄 Rebuild from my chats';
  }

  /* ------------------------------------------------------------------ list */
  async function load() {
    const seq = ++st.seq;
    st.loading = true;
    st.error = '';
    renderState();
    try {
      const qs = [`limit=${PAGE}`, `offset=${st.page * PAGE}`, st.status ? 'status=' + enc(st.status) : '', st.kind ? 'kind=' + enc(st.kind) : '', st.q ? 'q=' + enc(st.q) : ''].filter(Boolean).join('&');
      const d = await api('dataset/list?' + qs);
      if (seq !== st.seq) return;
      st.total = num(d.total);
      const items = cleanItems(d.items);
      if (!items.length && st.page > 0 && st.total > 0) { st.page = Math.ceil(st.total / PAGE) - 1; load(); return; }
      st.items = items;
      rows.clear();
      clear(ui.list).append(...items.map((it) => { const el = h('article'); rows.set(it.id, el); mount(el, it); return el; }));
    } catch (e) {
      if (seq !== st.seq) return;
      st.error = e.message || 'unknown error';
      st.items = [];
      rows.clear();
      clear(ui.list);
    }
    st.loading = false;
    renderState();
  }

  function renderState() {
    const box = clear(ui.state), pager = clear(ui.pager);
    ui.bulk.hidden = !st.items.length;
    ui.approveAll.textContent = `✓ Approve all ${st.items.length} on this page`;
    ui.rejectAll.textContent = `✗ Reject all ${st.items.length} on this page`;
    if (st.error) {
      box.append(h('div', { class: 'note bad' }, 'Could not load the training examples: ' + st.error + ' ', h('button', { onclick: load }, 'Try again')));
      return;
    }
    if (!st.items.length) {
      if (st.loading) box.append(h('div', { class: 'muted' }, 'Loading…'));
      else if (st.stats && !num(st.stats.total) && !st.q && !st.status && !st.kind) {
        box.append(h('div', { class: 'lb-empty' }, h('b', {}, 'No training examples yet.'),
          h('div', {}, 'Examples are made from answers you 👍, ✏️ corrected, or the big model answered. Rate a few answers in Chat, then press “Rebuild from my chats”.')));
      } else box.append(h('div', { class: 'lb-empty' }, 'No examples match this filter.'));
      return;
    }
    const pages = Math.max(1, Math.ceil(st.total / PAGE)), first = st.page * PAGE + 1;
    pager.append(h('button', { disabled: st.loading || st.page === 0, onclick: () => { st.page--; load(); } }, '‹ Newer'),
      h('span', { class: 'muted small', role: 'status' }, `${first}–${first + st.items.length - 1} of ${st.total} · page ${st.page + 1} of ${pages}`),
      h('button', { disabled: st.loading || st.page + 1 >= pages, onclick: () => { st.page++; load(); } }, 'Older ›'));
  }

  function refresh() {
    loadStats();
    load();
  }

  /* ----------------------------------------------------------------- cards */
  function longText(text) {
    if (text.length <= CLAMP_CHARS && text.split('\n').length <= CLAMP_LINES) return h('div', { class: 'lb-text' }, text);
    const body = h('div', { class: 'lb-text lb-clamp' }, text);
    const more = h('button', { class: 'lb-link', 'aria-expanded': 'false', onclick: () => {
      const collapsed = body.classList.toggle('lb-clamp');
      more.textContent = collapsed ? 'Show all' : 'Show less';
      more.setAttribute('aria-expanded', String(!collapsed));
    } }, 'Show all');
    return h('div', {}, body, more);
  }

  const block = (label, text, cls = '') => h('div', { class: 'ld-block ' + cls }, h('div', { class: 'ld-label' }, label), longText(text));

  function tags(it) {
    const tag = (cls, text, title) => h('span', { class: 'badge ' + cls, title }, text);
    const kind = KINDS[it.kind] || [str(it.kind), ''], status = STATUS_BADGE[it.status] || [str(it.status), ''];
    return h('div', { class: 'ld-tags' },
      tag('ld-kind', kind[0], kind[1]), tag(status[1], status[0]),
      SOURCES[it.source] ? tag('', SOURCES[it.source], 'Where the good answer came from') : null,
      it.edited ? tag('', '✎ edited by you', 'You changed this text, so rebuilding will not overwrite it') : null,
      h('span', { class: 'muted small' }, when(it.created)));
  }

  function mount(el, it) {
    const isPref = it.kind === 'pref', messages = messagesOf(it), earlier = messages.slice(0, -1);
    el.className = 'ld-item ' + cssWord(it.status);
    el.onkeydown = null;
    const action = (act, label, title, pressed, fn) => h('button', { class: 'ld-btn', 'data-act': act, title, 'aria-label': `${title} (example ${it.id})`,
      'aria-pressed': pressed == null ? null : String(pressed), onclick: fn }, label);
    fill(el,
      h('div', { class: 'ld-head' }, tags(it),
        h('div', { class: 'ld-actions' },
          action('approve', '✓ Approve', it.status === 'approved' ? 'Approved: click to move it back to “to review”' : 'Approve: put it in the training file', it.status === 'approved', () => setStatus(el, it, 'approved', 'approve')),
          action('reject', '✗ Reject', it.status === 'rejected' ? 'Rejected: click to move it back to “to review”' : 'Reject: keep it out of the training file', it.status === 'rejected', () => setStatus(el, it, 'rejected', 'reject')),
          action('edit', '✏️ Edit', 'Edit the answer text', null, () => edit(el, it)),
          action('delete', '🗑', 'Delete this example', null, () => remove(el, it)))),
      block('Question', lastUserText(messages) || '(no question text)'),
      earlier.length ? h('details', { class: 'ld-earlier' }, h('summary', {}, `Earlier in this conversation (${plural(earlier.length, 'message', 'messages')})`),
        earlier.map((m) => h('div', { class: 'ld-msg' }, h('b', {}, ROLES[m.role] || 'Message'), h('div', { class: 'lb-text' }, clip(contentText(m.content), MAX_EARLIER_CHARS) || '(empty)')))) : null,
      block(isPref ? 'Better answer (the AI should answer like this)' : 'Answer to learn from', str(it.chosen), 'good'),
      isPref ? block('Worse answer (the AI should avoid this)', str(it.rejected), 'bad') : null);
  }

  function lastUserText(messages) {
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'user') return contentText(messages[i].content);
    return '';
  }

  const focusAct = (el, act) => { const b = el && [...el.querySelectorAll('[data-act]')].find((x) => x.dataset.act === act); if (b) b.focus(); };

  /* Puts an updated item back on the page, or drops it when it no longer matches the status filter. */
  function settle(el, next, act) {
    const i = st.items.findIndex((x) => x.id === next.id);
    if (i < 0) return;
    if (!st.status || next.status === st.status) {
      st.items[i] = next;
      mount(el, next);
      focusAct(el, act);
      return;
    }
    const neighbour = el.nextElementSibling || el.previousElementSibling;
    st.items.splice(i, 1);
    rows.delete(next.id);
    el.remove();
    st.total = Math.max(0, st.total - 1);
    if (!st.items.length) { load(); return; }
    renderState();
    focusAct(neighbour, act);
  }

  async function setStatus(el, it, status, act) {
    const target = it.status === status ? 'pending' : status;
    el.classList.add('busy');
    try {
      const d = await api('dataset/set_status', { ids: [it.id], status: target });
      if (num(d.updated) < 1) { toast('This example no longer exists. Reloading the list.', 'err'); refresh(); return; }
      settle(el, { ...it, status: target }, act);
      loadStats();
    } catch (e) { fail(e); } finally { el.classList.remove('busy'); }
  }

  async function remove(el, it) {
    const text = 'The example is removed from the list. It can come back the next time you press “Rebuild from my chats” while its chat answer still qualifies. To keep it out of your training data for good, use ✗ Reject instead.';
    if (!(await confirmBox('Delete this example?', text, true))) return;
    try { await api('dataset/delete', { id: it.id }); } catch (e) { fail(e); return; }
    toast('Deleted', 'ok');
    const neighbour = el.nextElementSibling || el.previousElementSibling;
    st.items = st.items.filter((x) => x.id !== it.id);
    rows.delete(it.id);
    el.remove();
    st.total = Math.max(0, st.total - 1);
    if (!st.items.length) load(); else { renderState(); focusAct(neighbour, 'delete'); }
    loadStats();
  }

  function edit(el, it) {
    const isPref = it.kind === 'pref';
    const area = (label, value, rowCount) => h('textarea', { class: 'ld-editor', rows: rowCount, maxlength: MAX_ANSWER, 'aria-label': label }, value);
    const chosen = area(isPref ? 'Better answer' : 'Answer to learn from', str(it.chosen), 9);
    const rejected = isPref ? area('Worse answer', str(it.rejected), 7) : null;
    const err = h('div', { class: 'note bad', role: 'alert', hidden: true });
    const problem = (msg) => { err.textContent = msg; err.hidden = false; };
    const cancel = () => { mount(el, it); focusAct(el, 'edit'); };

    async function save(approve) {
      const body = { id: it.id, chosen: chosen.value.trim() };
      if (!body.chosen) { problem('The answer cannot be empty.'); return; }
      if (isPref) {
        body.rejected = rejected.value.trim();
        if (!body.rejected) { problem('The worse answer cannot be empty.'); return; }
        if (body.rejected === body.chosen) { problem('The better and the worse answer are the same.'); return; }
      }
      err.hidden = true;
      el.classList.add('busy');
      try {
        let next = it;
        if (body.chosen !== str(it.chosen).trim() || (isPref && body.rejected !== str(it.rejected).trim())) {
          const d = await api('dataset/edit', body);
          if (!d || !d.item) throw new Error('The server sent an unexpected reply');
          next = d.item;
        }
        if (approve && next.status !== 'approved') {
          if (num((await api('dataset/set_status', { ids: [it.id], status: 'approved' })).updated) < 1) throw new Error('This example no longer exists');
          next = { ...next, status: 'approved' };
        }
        toast(next === it ? 'Nothing changed' : approve ? 'Saved and approved' : 'Saved', 'ok');
        settle(el, next, 'edit');
        loadStats();
      } catch (e) { problem(e.message); } finally { el.classList.remove('busy'); }
    }

    fill(el,
      h('div', { class: 'ld-head' }, tags(it)),
      block('Question', lastUserText(messagesOf(it)) || '(no question text)'),
      h('div', { class: 'ld-label' }, isPref ? 'Better answer (the AI should answer like this)' : 'Answer to learn from'), chosen,
      isPref ? [h('div', { class: 'ld-label' }, 'Worse answer (the AI should avoid this)'), rejected] : null,
      err,
      h('div', { class: 'row ld-editbar' },
        h('button', { class: 'primary', onclick: () => save(false) }, 'Save'),
        h('button', { onclick: () => save(true) }, 'Save and approve'),
        h('button', { class: 'ghost', onclick: cancel }, 'Cancel'),
        h('span', { class: 'muted small' }, 'Esc cancels. Edited examples are never overwritten when you rebuild.')));
    el.onkeydown = (e) => { if (e.key === 'Escape') cancel(); };
    chosen.focus();
  }

  async function bulk(status) {
    const targets = st.items.filter((it) => it.status !== status);
    if (!targets.length) { toast(`Everything on this page is already ${status}.`); return; }
    const approve = status === 'approved', n = plural(targets.length, 'example', 'examples');
    const text = approve ? 'Every example on this page, including the ones you have not read, goes into the training file. You can still change each one afterwards.'
      : 'Every example on this page is kept out of the training file. You can still change each one afterwards.';
    if (!(await confirmBox(`${approve ? 'Approve' : 'Reject'} ${n}?`, text))) return;
    ui.bulk.querySelectorAll('button').forEach((b) => { b.disabled = true; });
    try {
      const d = await api('dataset/set_status', { ids: targets.map((it) => it.id), status });
      toast(`${plural(num(d.updated), 'example', 'examples')} ${status}`, 'ok');
      refresh();
    } catch (e) { fail(e); }
    ui.bulk.querySelectorAll('button').forEach((b) => { b.disabled = false; });
  }

  /* ---------------------------------------------------------------- export */
  const option = (key, title, help, checked) => {
    const box = h('input', { type: 'checkbox', checked, onchange: renderExport });
    ui.opt[key] = box;
    return h('label', { class: 'ld-opt' }, box, h('span', {}, h('b', {}, title), h('span', { class: 'muted small ld-help' }, help)));
  };

  function exportCount() {
    if (!st.stats) return null;
    return num(st.stats.approved) + (ui.opt.only_approved.checked ? 0 : num(st.stats.pending));
  }

  function renderExport() {
    const n = exportCount();
    ui.dl.disabled = st.exporting || n === 0;
    ui.count.textContent = n == null ? 'Could not read the number of examples.' : n === 0
      ? (ui.opt.only_approved.checked ? 'Nothing to download yet: approve some examples first.' : 'Nothing to download yet: there are no examples.')
      : `${plural(n, 'example', 'examples')} will be in the file.`;
    ui.unsafe.hidden = ui.opt.redact.checked;
  }

  function fileName(r) {
    const m = /filename="?([^";]+)"?/i.exec(r.headers.get('Content-Disposition') || '');
    const name = m ? m[1].replace(/[^\w.-]+/g, '_') : '';
    return /\.zip$/i.test(name) ? name : 'duckstoken-training.zip';
  }

  function saveBlob(blob, name) {
    const url = URL.createObjectURL(blob), a = h('a', { href: url, download: name, hidden: true });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  async function download() {
    if (!ui.opt.redact.checked && !(await confirmBox('Download without hiding personal details?', 'The file will contain your real email addresses, phone numbers, keys and other personal details exactly as they appear in your chats.', true))) return;
    st.exporting = true;
    renderExport();
    try {
      const flag = (key) => (ui.opt[key].checked ? 1 : 0);
      const r = await api(`dataset/export?only_approved=${flag('only_approved')}&redact=${flag('redact')}&include_reasoning=${flag('include_reasoning')}&include_script=${flag('include_script')}`, undefined, { raw: true });
      if (!r.ok) {
        let message = 'HTTP ' + r.status;
        try { message = (await r.json()).error || message; } catch (e) { /* not json */ }
        throw new Error(message);
      }
      const blob = await r.blob(), name = fileName(r);
      saveBlob(blob, name);
      toast(`Saved ${name} (${fmtBytes(blob.size)}). Read it before you upload it anywhere.`, 'ok');
    } catch (e) { fail(e); }
    st.exporting = false;
    renderExport();
  }

  function exportCard() {
    ui.opt = {};
    ui.count = h('span', { class: 'muted', role: 'status' });
    ui.dl = h('button', { class: 'primary', onclick: download }, '⬇ Download training file (.zip)');
    ui.unsafe = h('div', { class: 'note bad', hidden: true }, 'Personal details will NOT be hidden. Only use this file on your own PC.');
    const options = [
      option('only_approved', 'Only examples I approved', 'Leave this on. Turn it off to include examples that are still waiting for review. Rejected ones are never included.', true),
      option('redact', 'Hide personal details', 'Replaces email addresses, phone numbers, API keys and tokens, card-like numbers, IP addresses and home-folder names with placeholders such as <email>.', S.values.redact_pii !== false),
      option('include_reasoning', 'Include the AI’s reasoning', 'The “thinking” the model wrote before answering. It makes the file bigger and can reveal more about you. Off is the safe choice.', false),
      option('include_script', 'Include the starter training script and guide', 'Adds train_lora.py and README_TRAINING.md. The script is a template that has not been tested here, so expect to adjust it.', true)];
    return h('div', { class: 'card' }, h('h2', {}, '📦 Download a training file'),
      h('div', { class: 'ld-options' }, options), ui.unsafe,
      h('div', { class: 'note warn' }, h('b', {}, 'Privacy. '), 'The file contains your questions and the AI’s answers, which can include personal details, private code or secrets. “Hide personal details” catches common patterns but cannot catch everything, for example a name or an address written in a sentence. Read the file before you upload it anywhere. Duckstoken itself uploads nothing: the download goes to your own PC.'),
      h('div', { class: 'note' }, h('b', {}, 'What to expect. '), 'Training on your examples teaches the small model your style, formats and recurring tasks. It does not turn a small model into a big one: model size still sets the ceiling, and a handful of examples changes very little.'),
      h('div', { class: 'row' }, ui.dl, ui.count));
  }

  function guideCard() {
    return h('div', { class: 'card' }, h('details', { class: 'ld-guide' },
      h('summary', {}, h('b', {}, 'How do I actually train this?')),
      h('p', { class: 'muted' }, 'Your PC only collects and prepares the data. The training itself needs a much stronger graphics card than a normal PC has, so you rent one by the hour.'),
      h('ol', {},
        h('li', {}, h('b', {}, 'Collect good examples. '), 'Use Chat as usual: 👍 good answers, ✏️ fix weak ones, and let the big model answer hard questions. Press “Rebuild from my chats” above, then approve only what you would be happy for the AI to copy. One wrong approved answer teaches the model something wrong, so quality matters more than quantity.'),
        h('li', {}, h('b', {}, 'Download the training file '), 'from the box above. It is a .zip with sft.jsonl (the example conversations), pref.jsonl (better-vs-worse pairs, for advanced methods; the starter script does not use it), manifest.json (a summary), README_TRAINING.md and train_lora.py.'),
        h('li', {}, h('b', {}, 'Read it. '), 'Open sft.jsonl in a text editor and check that nothing private is in it before it leaves your PC.'),
        h('li', {}, h('b', {}, 'Rent a GPU. '), 'The starter script fine-tunes gpt-oss-20b with LoRA and needs an NVIDIA GPU with roughly 48 to 80 GB of memory. Any cloud GPU rental service that gives you a terminal will do. The script needs at least 20 examples; a few hundred is a more realistic start.'),
        h('li', {}, h('b', {}, 'Run the script. '), 'Upload the unzipped files, then run ', h('code', {}, 'python train_lora.py'), '. It is a template that has not been run inside Duckstoken, and the training libraries change often, so you may need to adjust it. The comments at the top of the file explain what to install.'),
        h('li', {}, h('b', {}, 'Convert the result. '), 'llama.cpp only loads adapters in GGUF format. The exact conversion command (convert_lora_to_gguf.py) is at the top of train_lora.py.'),
        h('li', {}, h('b', {}, 'Use it in Duckstoken. '), 'Copy the .gguf adapter to this PC, open Settings, set “LoRA adapter (.gguf)” to that file and restart the model. An adapter only fits the model it was trained on (here gpt-oss-20b); it cannot be loaded on a different model.')),
      h('p', { class: 'muted small' }, 'The README_TRAINING.md inside the zip repeats these steps in more detail.')));
  }

  /* ----------------------------------------------------------------- shell */
  function init(el) {
    ui.statusTiles = buildTiles(STATUS_TILES, setStatusFilter);
    ui.kindTiles = buildTiles(KIND_TILES, setKindFilter);
    ui.rebuild = h('button', { class: 'primary', onclick: rebuild }, '🔄 Rebuild from my chats');
    ui.search = h('input', { type: 'text', maxlength: 200, placeholder: '🔍 Search questions and answers…', 'aria-label': 'Search the examples',
      oninput: debounce((e) => { st.q = e.target.value.trim(); st.page = 0; load(); }, 300),
      onkeydown: (e) => { if (e.key === 'Escape' && e.target.value) { e.target.value = ''; st.q = ''; st.page = 0; load(); } } });
    ui.approveAll = h('button', { onclick: () => bulk('approved') }, '✓ Approve all on this page');
    ui.rejectAll = h('button', { onclick: () => bulk('rejected') }, '✗ Reject all on this page');
    ui.bulk = h('div', { class: 'row ld-bulk', hidden: true }, ui.approveAll, ui.rejectAll, h('span', { class: 'muted small' }, 'Acts on the examples shown on this page.'));
    ui.state = h('div', { 'aria-live': 'polite' });
    ui.list = h('div', { class: 'ld-list' });
    ui.pager = h('div', { class: 'row ld-pager' });
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '🎓 Training data'),
        h('p', { class: 'muted' }, 'Your 👍 answers, your ✏️ corrections and the big model’s answers can be turned into examples for fine-tuning a copy of the model, so it gets better at your kind of questions. Nothing is trained on this PC: you review the examples here, download a file, and train on a rented GPU when you want to.'),
        h('div', { class: 'lb-stats' },
          h('div', { class: 'lb-group', role: 'group', 'aria-label': 'Filter by review status' }, ui.statusTiles.map((x) => x.btn)),
          h('div', { class: 'lb-group', role: 'group', 'aria-label': 'Filter by kind of example' }, ui.kindTiles.map((x) => x.btn))),
        h('div', { class: 'row ld-rebuild' }, ui.rebuild, h('span', { class: 'muted small grow' }, 'Looks through your logged answers and adds new examples. Safe to press any time: examples you approved, rejected or edited are never changed.'))),
      h('div', { class: 'card' },
        h('div', { class: 'row' }, h('div', { class: 'grow' }, ui.search),
          h('button', { class: 'ghost', title: 'Reload', 'aria-label': 'Reload the examples', onclick: refresh }, '↻')),
        ui.bulk, ui.state, ui.list, ui.pager),
      exportCard(),
      guideCard());
    renderStats();
    renderState();
  }

  async function show() {
    await loadStats();
    if (!st.touched) st.status = st.stats && num(st.stats.pending) > 0 ? 'pending' : '';
    renderStats();
    load();
  }

  return { init, show };
})();
