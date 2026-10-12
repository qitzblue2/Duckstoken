'use strict';
const Chat = {
  chats: [], cur: null, busy: false, ctrl: null, always: new Set(), refs: {},
  init() {
    try { this.chats = JSON.parse(localStorage.getItem('duck.chats') || '[]'); } catch (e) { this.chats = []; }
    const el = $('#page-chat'), r = this.refs;
    r.list = h('div', { class: 'chat-list' });
    r.msgs = h('div', { class: 'chat-msgs' });
    r.input = h('textarea', { rows: 2, placeholder: 'Message the model…  (Enter = send, Shift+Enter = new line)',
      onkeydown: (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); this.send(); } } });
    r.send = h('button', { class: 'primary', onclick: () => (this.busy ? this.stop() : this.send()) }, 'Send');
    r.effort = h('select', { style: 'width:auto', onchange: (e) => setSetting('reasoning_effort', e.target.value) },
      ['low', 'medium', 'high'].map((x) => h('option', { value: x }, 'reasoning: ' + x)));
    r.tools = h('input', { type: 'checkbox', onchange: (e) => setSetting('tools_enabled', e.target.checked).then(() => this.syncHeader()) });
    r.toolsInfo = h('span', { class: 'muted small' });
    r.status = h('span', { class: 'badge' }, 'server stopped');
    el.append(
      h('div', { class: 'chat-side' }, h('button', { class: 'primary', onclick: () => this.newChat() }, '+ New chat'), r.list),
      h('div', { class: 'chat-main' },
        h('div', { class: 'chat-head' }, r.status, r.effort,
          h('label', { class: 'row small' }, h('span', { class: 'switch' }, r.tools, h('span')), 'Let AI use my PC'), r.toolsInfo,
          h('span', { style: 'margin-left:auto' }),
          h('button', { class: 'ghost', title: 'Ask the model what is worth remembering from this chat', onclick: () => this.extractMemories(this.chat, 'ask', false) }, '🧠 Suggest memories'),
          h('button', { class: 'ghost', onclick: () => this.exportChat() }, 'Export'),
          h('button', { class: 'ghost', onclick: () => this.deleteChat(this.cur) }, 'Delete')),
        r.msgs, h('div', { class: 'chat-input' }, r.input, r.send)));
    if (!this.chats.length) this.newChat(); else this.cur = this.chats[0].id;
    this.renderList(); this.renderMsgs(); this.pollBig();
  },
  show() { this.syncHeader(); r_focus(this.refs.input); },
  syncHeader() {
    const r = this.refs, v = S.values;
    r.effort.value = v.reasoning_effort; r.tools.checked = !!v.tools_enabled;
    const on = ['list_dir', 'read_file', 'find_files', 'system_info', 'write_file', 'run_command', 'fetch_url'].filter((t) => v['tool_' + t]);
    r.toolsInfo.textContent = v.tools_enabled ? `${v.full_pc_access ? 'FULL PC' : 'folders only'} · ${on.length} tools` : '';
  },
  update(st) {
    const s = st.server.state; this.refs.status.className = 'badge ' + s;
    this.refs.status.textContent = s === 'ready' ? 'model ready' : 'server ' + s;
  },
  get chat() { return this.chats.find((c) => c.id === this.cur); },
  save() { try { localStorage.setItem('duck.chats', JSON.stringify(this.chats.slice(0, 60))); } catch (e) { /* quota */ } },
  newChat() {
    const c = { id: Date.now().toString(36), title: 'New chat', messages: [] };
    this.chats.unshift(c); this.cur = c.id; this.save(); this.renderList(); this.renderMsgs();
  },
  async deleteChat(id) {
    if (!(await confirmBox('Delete this chat?', 'Its logged answers, queued big-model jobs and training examples (Learning tab) are deleted too.', true))) return;
    api('feedback/delete_chat', { chat_id: id }).catch(() => {});  // what Duckstoken logged for learning goes with it
    this.chats = this.chats.filter((c) => c.id !== id);
    if (!this.chats.length) return this.newChat();
    this.cur = this.chats[0].id; this.save(); this.renderList(); this.renderMsgs();
  },
  exportChat() {
    const blob = new Blob([JSON.stringify(this.chat, null, 2)], { type: 'application/json' });
    const a = h('a', { href: URL.createObjectURL(blob), download: (this.chat.title || 'chat') + '.json' }); a.click();
  },
  renderList() {
    clear(this.refs.list);
    this.chats.forEach((c) => this.refs.list.append(h('div', { class: 'item' + (c.id === this.cur ? ' active' : ''), onclick: () => { this.cur = c.id; this.renderList(); this.renderMsgs(); } },
      h('span', {}, c.title))));
  },
  renderMsgs() {
    clear(this.refs.msgs);
    if (!this.chat.messages.length) this.refs.msgs.append(h('div', { class: 'muted', style: 'margin:auto;text-align:center' },
      h('div', { style: 'font-size:40px' }, '🦆'), 'Start the model server on the Dashboard, then say hi.'));
    this.chat.messages.forEach((m) => this.refs.msgs.append(this.msgEl(m)));
    this.scroll(true);
  },
  scroll(force) { const m = this.refs.msgs; if (force || m.scrollTop + m.clientHeight >= m.scrollHeight - 80) m.scrollTop = m.scrollHeight; },
  msgEl(m) {
    const el = h('div', { class: 'msg ' + m.role });
    this.fill(el, m); el._m = m; return el;
  },
  fill(el, m) {
    clear(el);
    if (m.role === 'tool') {
      el.className = 'msg tool';
      el.append(h('details', {}, h('summary', { class: 'muted small' }, `🔧 ${m.name} result`), h('pre', { class: 'mono', style: 'white-space:pre-wrap;max-height:260px;overflow:auto' }, m.content)));
      return;
    }
    if (m.reasoning && S.values.show_reasoning) el.append(h('details', { class: 'think', open: !m.content && !m.done }, h('summary', {}, 'Reasoning'), h('div', {}, m.reasoning)));
    if (m.content) el.append(h('div', { class: 'body', html: m.role === 'user' ? '<p>' + escapeHtml(m.content).replace(/\n/g, '<br>') + '</p>' : renderMarkdown(m.content) }));
    (m.tool_calls || []).forEach((tc) => el.append(h('div', { class: 'toolcard' }, h('b', {}, '🔧 ' + tc.function.name), h('pre', {}, tc.function.arguments || ''))));
    if (m.stats) el.append(h('div', { class: 'stats' }, m.stats));
    if (m.ctx) el.append(this.ctxChips(m.ctx));
    if (m.big) el.append(this.bigBlock(m));
    if (m.error) el.append(h('div', { class: 'note bad' }, m.error));
    if (m.done && m.role === 'assistant' && m.content && !(m.tool_calls || []).length) el.append(this.actions(m, el));
    if (!m.content && !m.reasoning && !(m.tool_calls || []).length && !m.error && m.role === 'assistant') el.append(h('span', { class: 'muted' }, '…'));
  },
  apiMessages(chat = this.chat) {
    return chat.messages.filter((m) => !(m.role === 'assistant' && !m.content && !(m.tool_calls || []).length)).map((m) => {
      const o = { role: m.role, content: m.content || '' };
      if (m.tool_calls?.length) { o.tool_calls = m.tool_calls; if (m.reasoning) o.reasoning_content = m.reasoning; }
      if (m.role === 'tool') o.tool_call_id = m.tool_call_id;
      return o;
    });
  },
  stop() { this.ctrl?.abort(); },
  setBusy(b) { this.busy = b; this.refs.send.textContent = b ? '■ Stop' : 'Send'; this.refs.send.className = b ? 'danger' : 'primary'; },
  async send() {
    const text = this.refs.input.value.trim();
    if (!text || this.busy) return;
    const c = this.chat;
    if (!c.messages.length) c.title = text.slice(0, 40);
    c.messages.push({ role: 'user', content: text });
    this.refs.input.value = '';
    this.renderList(); this.renderMsgs(); this.save();
    this.setBusy(true);
    try {
      for (let round = 0; round < (S.values.max_tool_rounds || 8); round++) {
        const asst = { role: 'assistant', content: '', reasoning: '', tool_calls: [] };
        c.messages.push(asst);
        const el = this.msgEl(asst);
        if (this.cur === c.id) { this.refs.msgs.append(el); this.scroll(true); }  // the user may have opened another chat meanwhile
        let ok = true;
        try { await this.stream(c, asst, el); } catch (e) {
          ok = false;
          if (e.name !== 'AbortError') asst.error = e.message; else asst.stats = (asst.stats || '') + ' (stopped)';
        }
        asst.done = true; this.fill(el, asst); this.save();
        if (ok && !asst.tool_calls.length) this.record(c, asst, el);
        if (!ok || !asst.tool_calls.length) break;
        for (const tc of asst.tool_calls) {
          const out = await this.runTool(tc);
          const tm = { role: 'tool', tool_call_id: tc.id, name: tc.function.name, content: out };
          c.messages.push(tm);
          if (this.cur === c.id) { this.refs.msgs.append(this.msgEl(tm)); this.scroll(true); }
        }
        this.save();
      }
    } finally {
      this.setBusy(false); this.ctrl = null; this.renderList(); this.armIdleExtract(c);
      if (this.cur === c.id) this.renderMsgs();  // after switching away and back mid-answer, show the finished state
    }
  },
  async stream(c, asst, el) {
    this.ctrl = new AbortController();
    const res = await api('chat', { messages: this.apiMessages(c) }, { raw: true, signal: this.ctrl.signal });
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.error || 'HTTP ' + res.status); }
    let queued = false;
    const repaint = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; this.fill(el, asst); if (this.cur === c.id) this.scroll(); }); };
    const stats = (t) => { if (t) asst.stats = `${t.predicted_n ?? '?'} tokens · ${(t.predicted_per_second ?? 0).toFixed(1)} tok/s · prompt ${(t.prompt_per_second ?? 0).toFixed(0)} tok/s`; };
    const calls = [];
    const take = (delta) => {
      if (delta.content) asst.content += delta.content;
      if (delta.reasoning_content) asst.reasoning += delta.reasoning_content;
      (delta.tool_calls || []).forEach((t) => {
        const i = t.index ?? calls.length; const c = calls[i] || (calls[i] = { id: '', type: 'function', function: { name: '', arguments: '' } });
        if (t.id) c.id = t.id; if (t.function?.name) c.function.name += t.function.name;
        if (t.function?.arguments) c.function.arguments += t.function.arguments;
      });
    };
    if ((res.headers.get('content-type') || '').includes('application/json')) {
      const d = await res.json(); const msg = d.choices?.[0]?.message || {};
      take({ content: msg.content, reasoning_content: msg.reasoning_content, tool_calls: (msg.tool_calls || []).map((t, i) => ({ ...t, index: i })) });
      stats(d.timings);
    } else {
      const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
      for (;;) {
        const { value, done } = await reader.read(); if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim(); if (data === '[DONE]') continue;
          let j; try { j = JSON.parse(data); } catch (e) { continue; }
          if (j.duck_context) { asst.ctx = j.duck_context; repaint(); continue; }
          if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
          const ch = j.choices?.[0]; if (ch?.delta) take(ch.delta);
          if (j.timings) stats(j.timings);
          repaint();
        }
      }
    }
    asst.tool_calls = calls.filter(Boolean).map((c, i) => ({ ...c, id: c.id || 'call_' + Date.now().toString(36) + i }));
  },
  /* ---- learning: record exchange, ratings, corrections, refine, big model, memory ---- */
  safeUrl(u) { return /^https?:\/\//i.test(u || '') ? u : null; },
  ctxChips(ctx) {
    const chips = [], n = (a) => (a || []).length;
    if (n(ctx.memory_ids)) chips.push(h('span', { class: 'chip', title: 'Memories added to this answer' }, `🧠 ${n(ctx.memory_ids)} memor${n(ctx.memory_ids) === 1 ? 'y' : 'ies'}`));
    if (n(ctx.example_ids) || n(ctx.lesson_ids)) chips.push(h('span', { class: 'chip', title: 'Past liked answers / feedback notes used' }, `📚 ${n(ctx.example_ids)} example${n(ctx.example_ids) === 1 ? '' : 's'}${n(ctx.lesson_ids) ? ' + ' + n(ctx.lesson_ids) + ' reminder(s)' : ''}`));
    (ctx.remembered || []).forEach((r) => chips.push(h('span', { class: 'chip good', title: r }, '🧠 remembered: ' + r.slice(0, 50))));
    const src = ctx.web_sources || [];
    const box = h('div', { class: 'ctxchips' }, chips);
    if (src.length) box.append(h('details', { class: 'sources' }, h('summary', { class: 'chip' }, `🌐 ${src.length} web source${src.length === 1 ? '' : 's'}`),
      src.map((s, i) => h('div', { class: 'small' }, `[${i + 1}] `, this.safeUrl(s.url) ? h('a', { href: s.url, target: '_blank', rel: 'noopener noreferrer' }, s.title || s.url) : (s.title || s.url)))));
    if (ctx.web_error) box.append(h('span', { class: 'chip bad', title: ctx.web_error }, '🌐 search failed'));
    return box;
  },
  bigBlock(m) {
    const b = m.big;
    if (b.pending) return h('div', { class: 'bigblock' }, h('span', { class: 'muted small' }, `🐘 Waiting for the big model… (${b.status || 'queued'}) — see Learning → Big model`));
    if (b.error) return h('div', { class: 'bigblock' }, h('span', { class: 'note bad' }, '🐘 Big model failed: ' + b.error));
    return h('div', { class: 'bigblock' }, h('div', { class: 'small muted' }, `🐘 Big model answer${b.model ? ' (' + b.model + ')' : ''}`),
      h('div', { class: 'body', html: renderMarkdown(b.answer || '') }),
      h('div', { class: 'row small' }, h('button', { onclick: () => { m.content = b.answer; m.big = null; this.save(); this.renderMsgs(); } }, 'Use this answer instead')));
  },
  actions(m, el) {
    const row = h('div', { class: 'msg-actions' });
    const btn = (label, title, fn, cls) => row.append(h('button', { class: 'ghost small ' + (cls || ''), title, onclick: fn }, label));
    const up = m.rating === 1, down = m.rating === -1;
    btn('👍', 'Good answer — teaches the AI your taste', () => this.rate(m, up ? 0 : 1, '', el), up ? 'on' : '');
    btn('👎', 'Bad answer — optionally say why', () => this.dislike(m, el), down ? 'on' : '');
    btn('✏️', 'Write the answer you wanted', () => this.correct(m, el));
    btn('🔁', 'Critique and rewrite this answer (slower, often better)', () => this.refine(m, el));
    if (S.values.big_model_path) btn('🐘', 'Ask the big model in the background', () => this.askBig(m, el));
    btn('🧠', 'Remember something from this', () => this.remember(m));
    btn('📋', 'Copy', () => navigator.clipboard?.writeText(m.content).then(() => toast('Copied', 'ok')));
    return row;
  },
  async record(c, asst, el) {
    if (!S.values.record_exchanges) return;
    const idx = c.messages.indexOf(asst);
    const prior = c.messages.slice(0, idx).map((m) => { const o = { role: m.role, content: m.content || '' }; if (m.tool_calls?.length) o.tool_calls = m.tool_calls; if (m.role === 'tool') o.tool_call_id = m.tool_call_id; return o; });
    const st = (asst.stats || '').match(/^(\d+) tokens · ([\d.]+) tok\/s/);
    try {
      const r = await api('feedback/record', { chat_id: c.id, msg_index: idx, messages: prior, answer: asst.content, reasoning: asst.reasoning || '',
        effort: S.values.reasoning_effort, tokens: st ? +st[1] : 0, tps: st ? +st[2] : 0, used_tools: c.messages.slice(0, idx).some((x) => x.role === 'tool'),
        web_sources: (asst.ctx?.web_sources) || [] });
      asst.xid = r.id; this.save();
    } catch (e) { /* logging must never disturb chat */ }
  },
  async needX(m) { if (!m.xid) { toast('This answer was not logged (is “Keep a log” on?)', 'err'); return false; } return true; },
  async rate(m, rating, note, el) {
    if (!await this.needX(m)) return;
    try { await api('feedback/rate', { id: m.xid, rating, note: note || '' }); m.rating = rating; this.save(); this.fill(el, m); toast(rating === 1 ? 'Thanks — I’ll use this as an example' : rating === -1 ? 'Noted' : 'Rating cleared', 'ok'); } catch (e) { fail(e); }
  },
  async dislike(m, el) {
    if (!await this.needX(m)) return;
    if (m.rating === -1) return this.rate(m, 0, '', el);
    const note = h('textarea', { rows: 3, placeholder: 'What was wrong? (optional — becomes a reminder for similar questions)' });
    const ok = await modal('Bad answer', note, [{ label: 'Cancel', value: false }, { label: 'Save', value: true, primary: true }]);
    if (ok) this.rate(m, -1, note.value.trim(), el);
  },
  async correct(m, el) {
    if (!await this.needX(m)) return;
    const ta = h('textarea', { rows: 10, style: 'width:100%' }, m.content);
    const ok = await modal('Write the answer you wanted', h('div', {}, h('p', { class: 'muted small' }, 'Saved as a gold example: similar questions will be shown this answer, and it becomes training data.'), ta),
      [{ label: 'Cancel', value: false }, { label: 'Save correction', value: true, primary: true }]);
    if (ok && ta.value.trim()) { try { await api('feedback/correct', { id: m.xid, correction: ta.value.trim() }); toast('Saved', 'ok'); } catch (e) { fail(e); } }
  },
  async refine(m, el) {
    const c = this.chat, idx = c.messages.indexOf(m);
    toast('Refining… this runs the model twice per pass');
    try {
      const r = await api('refine', { messages: c.messages.slice(0, idx).filter((x) => x.role === 'user' || x.role === 'assistant').map((x) => ({ role: x.role, content: x.content || '' })), answer: m.content });
      const body = h('div', {}, h('p', { class: 'muted small' }, 'Critique'), h('pre', { class: 'log', style: 'height:auto;max-height:150px' }, r.critique || '(none)'),
        h('p', { class: 'muted small' }, 'Improved answer'), h('div', { class: 'body', html: renderMarkdown(r.answer) }));
      const ok = await modal('Refined answer', body, [{ label: 'Keep original', value: false }, { label: 'Replace', value: true, primary: true }]);
      if (ok) { m.content = r.answer; m.rating = 0; this.save(); this.fill(el, m); this.record(c, m, el); }
    } catch (e) { fail(e); }
  },
  async askBig(m, el) {
    if (!await this.needX(m)) return;
    try {
      const r = await api('queue/enqueue', { exchange_id: m.xid, chat_id: this.chat.id, reason: 'manual' });
      m.big = { pending: true, job: r.item.id, status: 'queued' }; this.save(); this.fill(el, m); this.pollBig();
      toast(S.values.big_run_mode === 'manual' ? 'Queued. Press “Run queue now” in Learning → Big model to start it.' : 'Queued for the big model', 'ok');
    } catch (e) { fail(e); }
  },
  pollBig() {
    if (this._bigTimer) return;
    this._bigTimer = setInterval(async () => {
      const pend = this.chats.flatMap((c) => c.messages).filter((x) => x.big?.pending);
      if (!pend.length) { clearInterval(this._bigTimer); this._bigTimer = null; return; }
      for (const m of pend) {
        try {
          const j = (await api('queue/get?id=' + m.big.job)).item; m.big.status = j.status;
          if (j.status === 'done') m.big = { answer: j.answer, model: j.model };
          else if (j.status === 'error' || j.status === 'cancelled') m.big = { error: j.error || j.status };
        } catch (e) { m.big = { error: e.message }; }
      }
      this.save(); if (document.querySelector('#page-chat.active')) this.renderMsgs();
    }, 8000);
  },
  async remember(m) {
    const ta = h('textarea', { rows: 4, placeholder: 'One short, self-contained fact or preference' }, (m.content || '').slice(0, 280));
    const ok = await modal('Remember this', ta, [{ label: 'Cancel', value: false }, { label: 'Save to memory', value: true, primary: true }]);
    if (ok && ta.value.trim()) { try { await api('memory/add', { text: ta.value.trim(), kind: 'note' }); toast('Remembered', 'ok'); } catch (e) { fail(e); } }
  },
  /* suggest memories ~2 min after the last answer (so it never competes with a live chat) */
  armIdleExtract(c) {
    clearTimeout(this._idle);
    const mode = S.values.memory_auto_extract;
    if (!S.values.memory_enabled || mode === 'off') return;
    this._idle = setTimeout(() => this.extractMemories(c, mode, true), 120000);
  },
  async extractMemories(c, mode, quiet) {
    const users = c.messages.filter((m) => m.role === 'user').length;
    if (users < 1 || c.extractedAt === c.messages.length || this.busy) return;
    if (S.status?.server?.state !== 'ready') { if (!quiet) toast('Start the model server first', 'err'); return; }
    c.extractedAt = c.messages.length; this.save();
    try {
      const r = await api('memory/extract', { messages: c.messages.filter((m) => (m.role === 'user' || m.role === 'assistant') && m.content).map((m) => ({ role: m.role, content: m.content })) });
      const cands = r.candidates || [];
      if (!cands.length) { if (!quiet) toast('Nothing worth remembering found', 'ok'); return; }
      if (mode === 'auto') { for (const t of cands) await api('memory/add', { text: t, kind: 'fact' }); toast(`Remembered ${cands.length} thing(s)`, 'ok'); return; }
      const boxes = cands.map((t) => [h('input', { type: 'checkbox', checked: true }), t]);
      const ok = await modal('Remember these?', h('div', {}, boxes.map(([cb, t]) => h('label', { class: 'row', style: 'align-items:flex-start;margin:6px 0' }, cb, h('span', { class: 'grow' }, t)))),
        [{ label: 'No thanks', value: false }, { label: 'Save selected', value: true, primary: true }]);
      if (ok) { let n = 0; for (const [cb, t] of boxes) if (cb.checked) { await api('memory/add', { text: t, kind: 'fact' }); n++; } if (n) toast(`Remembered ${n}`, 'ok'); }
    } catch (e) { if (!quiet) fail(e); }
  },
  async runTool(tc) {
    const name = tc.function.name, body = { name, arguments: tc.function.arguments || '{}' };
    try {
      let r = await api('tools/call', body);
      if (r.needs_approval) {
        if (r.no_always || !this.always.has(name)) {  // risky calls (writes, commands, secrets) are asked every time
          const buttons = [{ label: 'Deny', value: 'no', danger: true }, { label: 'Allow once', value: 'once', primary: true }];
          if (!r.no_always) buttons.push({ label: `Always allow “${name}” in this session`, value: 'always' });
          const d = await modal(r.sensitive ? '⚠ The AI wants to touch private data' : 'The AI wants to use your PC',
            h('div', {}, h('p', {}, h('b', {}, name)), h('pre', { class: 'log', style: 'height:auto;max-height:320px' }, r.description)), buttons);
          if (d !== 'once' && d !== 'always') return 'The user denied this action.';
          if (d === 'always') this.always.add(name);
        }
        r = await api('tools/call', { ...body, approved: true });
      }
      return r.output;
    } catch (e) { return 'Error: ' + e.message; }
  },
};
function r_focus(el) { setTimeout(() => el && el.focus(), 50); }
