'use strict';
/* Learning page: shell with sub-tabs + overview. Each tab module (LearnMemory, LearnFeedback, LearnQueue,
   LearnDataset, LearnWeb) exposes init(container) and show() (and optionally hide()). */
const LearnOverview = {
  init(el) {
    this.refs = { stats: h('div', { class: 'grid' }), rows: h('div') };
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '🎓 How Duckstoken gets smarter for you'),
        h('p', {}, 'Your model’s weights never change on this PC — instead Duckstoken feeds it better context and collects data you can use to train it properly later:'),
        h('ol', {},
          h('li', {}, h('b', {}, 'Memory'), ' — facts and preferences are stored and the relevant ones are added to every question.'),
          h('li', {}, h('b', {}, 'Feedback'), ' — 👍 answers and your corrections become examples the model copies; 👎 notes become reminders.'),
          h('li', {}, h('b', {}, 'Web research'), ' — optional searches ground answers in current facts, with sources.'),
          h('li', {}, h('b', {}, 'Big model'), ' — hard questions can be answered slowly by your 120B in the background (even overnight).'),
          h('li', {}, h('b', {}, 'Training data'), ' — approved answers (yours + the big model’s) are exported to fine-tune the small model on a rented GPU.')),
        h('div', { class: 'note warn' }, 'Honest expectations: this makes a small model noticeably better at ', h('i', {}, 'your'), ' tasks, style and facts. It will not turn a 20B into a 120B in general — model size still sets the ceiling.')),
      this.refs.stats,
      h('div', { class: 'card', style: 'padding:0' }, h('div', { style: 'padding:12px 16px' }, h('h2', { style: 'margin:0' }, 'Main switches')), this.refs.rows));
  },
  async show() {
    clear(this.refs.rows);
    ['memory_enabled', 'record_exchanges', 'examples_enabled', 'lessons_enabled', 'smart_prompt', 'web_grounding', 'big_model_path', 'big_run_mode', 'overnight_review_count']
      .forEach((k) => this.refs.rows.append(settingRow(schemaOf(k), () => this.show())));
    const get = (p) => api(p).catch(() => ({}));
    const [m, f, q, d] = await Promise.all([get('memory/stats'), get('feedback/stats'), get('queue/status'), get('dataset/stats')]);
    const card = (title, big, sub, tab) => h('div', { class: 'card statcard', onclick: () => Learning.open(tab), style: 'cursor:pointer' },
      h('div', { class: 'muted small' }, title), h('div', { style: 'font-size:28px;font-weight:700' }, String(big ?? '—')), h('div', { class: 'muted small' }, sub));
    clear(this.refs.stats).append(
      card('Memories', m.total, m.fts === false ? 'basic search (no FTS5)' : `${m.pinned || 0} pinned`, 'memory'),
      card('Logged answers', f.total, `${f.up || 0} 👍 · ${f.down || 0} 👎 · ${f.corrected || 0} corrected`, 'feedback'),
      card('Big-model jobs', (q.counts || {}).queued, `${(q.counts || {}).done || 0} done · phase ${q.phase || '—'}`, 'queue'),
      card('Training examples', d.approved, `${d.pending || 0} awaiting review · ${d.total || 0} total`, 'dataset'));
  },
};

const LearnWeb = {
  init(el) {
    this.out = h('div'); this.q = h('input', { type: 'text', value: 'latest stable Python release', onkeydown: (e) => e.key === 'Enter' && this.test() });
    this.rows = h('div');
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '🌐 Web research'),
        h('p', { class: 'muted' }, 'Off by default. When on, search queries are sent to the provider you pick. Pages are fetched with SSRF protection (no local/private addresses).'),
        h('div', { class: 'row' }, h('div', { class: 'grow' }, this.q), h('button', { class: 'primary', onclick: () => this.test() }, 'Test search')), this.out),
      h('div', { class: 'card', style: 'padding:0' }, this.rows));
  },
  show() {
    clear(this.rows);
    ['web_grounding', 'web_provider', 'searxng_url', 'brave_api_key', 'web_max_results', 'web_fetch_pages', 'web_page_chars', 'web_cache_minutes', 'web_timeout', 'web_save_to_memory', 'confirm_web', 'web_allow_private']
      .forEach((k) => this.rows.append(settingRow(schemaOf(k), () => this.show())));
  },
  async test() {
    clear(this.out).append(h('span', { class: 'muted' }, 'Searching…'));
    try {
      const r = await api('web/test', { query: this.q.value });
      clear(this.out);
      if (!r.results.length) this.out.append(h('div', { class: 'note' }, 'No results.'));
      r.results.forEach((x, i) => this.out.append(h('div', { style: 'margin:8px 0' },
        h('div', {}, `[${i + 1}] `, /^https?:\/\//i.test(x.url) ? h('a', { href: x.url, target: '_blank', rel: 'noopener noreferrer' }, x.title || x.url) : (x.title || x.url)),
        h('div', { class: 'muted small' }, x.snippet))));
      this.out.append(h('div', { class: 'muted small' }, `Provider: ${r.provider} · ${r.seconds}s`));
    } catch (e) { clear(this.out).append(h('div', { class: 'note bad' }, e.message)); }
  },
};

const Learning = {
  tabs: [['overview', 'Overview', () => LearnOverview], ['memory', '🧠 Memory', () => window.LearnMemory], ['feedback', '👍 Feedback', () => window.LearnFeedback],
    ['queue', '🐘 Big model', () => window.LearnQueue], ['dataset', '🎓 Training data', () => window.LearnDataset], ['web', '🌐 Web', () => LearnWeb]],
  cur: 'overview', built: {}, panes: {}, bar: null,
  init() {
    const el = $('#page-learning');
    this.bar = h('div', { class: 'subtabs' }, this.tabs.map(([k, label]) => h('button', { 'data-tab': k, onclick: () => this.open(k) }, label)));
    el.append(this.bar);
    this.tabs.forEach(([k]) => { this.panes[k] = h('div', { class: 'subpane', 'data-pane': k }); el.append(this.panes[k]); });
    try { this.cur = localStorage.getItem('duck.learnTab') || 'overview'; } catch (e) { /* ignore */ }
  },
  show() { this.open(this.cur); },
  open(k) {
    if (!this.panes[k]) k = 'overview';
    const prev = this.tabs.find((t) => t[0] === this.cur)?.[2]();
    if (prev && prev.hide && this.cur !== k) prev.hide();
    this.cur = k;
    try { localStorage.setItem('duck.learnTab', k); } catch (e) { /* ignore */ }
    this.bar.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.tab === k));
    Object.entries(this.panes).forEach(([n, p]) => p.classList.toggle('active', n === k));
    const mod = this.tabs.find((t) => t[0] === k)[2]();
    if (!mod) { clear(this.panes[k]).append(h('div', { class: 'note bad' }, 'This section failed to load (script missing).')); return; }
    if (!this.built[k]) { this.built[k] = true; mod.init(this.panes[k]); }
    mod.show();
  },
};
