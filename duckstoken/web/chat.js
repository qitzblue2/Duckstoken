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
          h('button', { class: 'ghost', onclick: () => this.exportChat() }, 'Export'),
          h('button', { class: 'ghost', onclick: () => this.deleteChat(this.cur) }, 'Delete')),
        r.msgs, h('div', { class: 'chat-input' }, r.input, r.send)));
    if (!this.chats.length) this.newChat(); else this.cur = this.chats[0].id;
    this.renderList(); this.renderMsgs();
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
  deleteChat(id) {
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
    if (m.error) el.append(h('div', { class: 'note bad' }, m.error));
    if (!m.content && !m.reasoning && !(m.tool_calls || []).length && !m.error && m.role === 'assistant') el.append(h('span', { class: 'muted' }, '…'));
  },
  apiMessages() {
    return this.chat.messages.filter((m) => !(m.role === 'assistant' && !m.content && !(m.tool_calls || []).length)).map((m) => {
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
        const el = this.msgEl(asst); this.refs.msgs.append(el); this.scroll(true);
        let ok = true;
        try { await this.stream(asst, el); } catch (e) {
          ok = false;
          if (e.name !== 'AbortError') asst.error = e.message; else asst.stats = (asst.stats || '') + ' (stopped)';
        }
        asst.done = true; this.fill(el, asst); this.save();
        if (!ok || !asst.tool_calls.length) break;
        for (const tc of asst.tool_calls) {
          const out = await this.runTool(tc);
          const tm = { role: 'tool', tool_call_id: tc.id, name: tc.function.name, content: out };
          c.messages.push(tm); this.refs.msgs.append(this.msgEl(tm)); this.scroll(true);
        }
        this.save();
      }
    } finally { this.setBusy(false); this.ctrl = null; this.renderList(); }
  },
  async stream(asst, el) {
    this.ctrl = new AbortController();
    const res = await api('chat', { messages: this.apiMessages() }, { raw: true, signal: this.ctrl.signal });
    if (!res.ok) { const d = await res.json().catch(() => ({})); throw new Error(d.error || 'HTTP ' + res.status); }
    let queued = false;
    const repaint = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; this.fill(el, asst); this.scroll(); }); };
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
          if (j.error) throw new Error(j.error.message || JSON.stringify(j.error));
          const ch = j.choices?.[0]; if (ch?.delta) take(ch.delta);
          if (j.timings) stats(j.timings);
          repaint();
        }
      }
    }
    asst.tool_calls = calls.filter(Boolean).map((c, i) => ({ ...c, id: c.id || 'call_' + Date.now().toString(36) + i }));
  },
  async runTool(tc) {
    const name = tc.function.name, body = { name, arguments: tc.function.arguments || '{}' };
    try {
      let r = await api('tools/call', body);
      if (r.needs_approval) {
        if (!this.always.has(name)) {
          const d = await modal('The AI wants to use your PC', h('div', {}, h('p', {}, h('b', {}, name)), h('pre', { class: 'log', style: 'height:auto;max-height:260px' }, r.description)),
            [{ label: 'Deny', value: 'no', danger: true }, { label: 'Allow once', value: 'once', primary: true }, { label: `Always allow “${name}” in this session`, value: 'always' }]);
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
