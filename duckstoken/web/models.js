'use strict';
const Models = {
  refs: {}, assets: null,
  init() {
    const el = $('#page-models'), r = this.refs;
    r.dir = h('span', { class: 'mono' });
    r.installed = h('div'); r.hfResult = h('div'); r.engine = h('div'); r.engines = h('div');
    r.repo = h('input', { type: 'text', placeholder: 'owner/name  (e.g. ggml-org/gpt-oss-120b-GGUF)', onkeydown: (e) => e.key === 'Enter' && this.listRepo() });
    r.backend = h('select', { style: 'width:auto' }, ['auto', 'cpu', 'cuda', 'vulkan'].map((b) => h('option', { value: b }, b)));
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '📁 Where models are stored'),
        h('div', { class: 'row' }, r.dir, h('button', { onclick: () => this.changeDir() }, 'Change…'),
          h('button', { onclick: () => api('open-folder', { path: S.values.models_dir || '' }).catch(fail) }, 'Open')),
        h('div', { class: 'note warn' }, 'Use your SSD / M.2. A 59 GB model on a hard drive is unusable when it is bigger than your RAM; on a fast SSD it works. The Dashboard shows which drive is which.')),
      h('div', { class: 'card' }, h('h2', {}, '📦 Installed models', h('button', { style: 'margin-left:auto', onclick: () => this.refresh() }, 'Refresh')), r.installed),
      h('div', { class: 'card' }, h('h2', {}, '⬇ Download from Hugging Face'),
        h('div', { class: 'row' }, S.hfQuick.map((q) => h('button', { onclick: () => { r.repo.value = q.repo; this.listRepo(); } }, q.label))),
        h('div', { class: 'row', style: 'margin-top:8px' }, h('div', { class: 'grow' }, r.repo), h('button', { onclick: () => this.listRepo() }, 'List files')),
        r.hfResult),
      h('div', { class: 'card' }, h('h2', {}, '⚙ Engine (llama.cpp)'),
        h('p', { class: 'muted' }, 'The engine runs the model. Duckstoken downloads the official llama.cpp build for Windows from GitHub.'),
        r.engines,
        h('div', { class: 'row' }, 'Backend:', r.backend, h('button', { onclick: () => this.findEngine() }, 'Find latest build'),
          h('button', { onclick: async () => { const p = await pickPath({ kind: 'file', ext: '.exe' }); if (p) { await setSetting('llama_server_path', p); this.refresh(); } } }, 'I already have llama-server.exe…')),
        r.engine));
  },
  async show() { this.refresh(); },
  async refresh() {
    this.refs.dir.textContent = S.values.models_dir || '(default) ~/Duckstoken/models';
    try {
      const d = await api('models'); this.refs.dir.textContent = d.dir;
      const t = clear(this.refs.installed);
      if (!d.models.length) t.append(h('span', { class: 'muted' }, 'No models yet — download one below.'));
      else t.append(h('table', {}, h('tbody', {}, d.models.map((m) => h('tr', {},
        h('td', {}, m.name, m.kind !== 'model' ? h('span', { class: 'tag' }, m.kind) : null), h('td', {}, fmtGB(m.size_gb) + (m.shards > 1 ? ` (${m.shards} parts)` : '')),
        h('td', {}, h('button', { class: S.values.model_path === m.path ? 'primary' : '', onclick: async () => { await setSetting('model_path', m.path); toast('Selected ' + m.name, 'ok'); this.refresh(); } }, S.values.model_path === m.path ? 'Selected' : 'Use'), ' ',
          h('button', { onclick: () => this.inspect(m) }, 'Check'), ' ',
          h('button', { class: 'ghost', onclick: () => this.del(m) }, 'Delete')))))));
      const e = clear(this.refs.engines);
      e.append(h('div', { class: 'kv', style: 'margin-bottom:10px' }, h('div', {}, 'In use'), h('div', { class: 'mono' }, S.values.llama_server_path || 'not set')));
      d.engines.forEach((x) => e.append(h('div', { class: 'row small' }, '• ' + x.name, h('button', { onclick: async () => { await setSetting('llama_server_path', x.path); this.refresh(); } }, 'Use this'))));
    } catch (er) { fail(er); }
  },
  async changeDir() {
    const p = await pickPath({ kind: 'dir', start: S.values.models_dir });
    if (p && await setSetting('models_dir', p)) { toast('Models folder set', 'ok'); this.refresh(); }
  },
  async inspect(m) {
    try { const r = await api('models/inspect', { path: m.path }); modal('Can I run ' + m.name + '?', renderFeasibility(r.feasibility, r.info)); } catch (e) { fail(e); }
  },
  async del(m) {
    if (!await confirmBox('Delete model?', `Permanently delete ${m.name} (${fmtGB(m.size_gb)})?`, true)) return;
    try { await api('models/delete', { path: m.path }); this.refresh(); } catch (e) { fail(e); }
  },
  async listRepo() {
    const repo = this.refs.repo.value.trim(), box = clear(this.refs.hfResult);
    if (!repo) return;
    box.append(h('span', { class: 'muted' }, 'Loading…'));
    try {
      const d = await api('hf/files?repo=' + encodeURIComponent(repo)); clear(box);
      if (!d.groups.length) box.append(h('span', { class: 'muted' }, 'No .gguf files in that repo.'));
      box.append(h('table', {}, h('tbody', {}, d.groups.map((g) => h('tr', {},
        h('td', {}, g.name, g.kind !== 'model' ? h('span', { class: 'tag' }, g.kind === 'draft' ? 'speculative draft — not a chat model' : g.kind) : null),
        h('td', {}, fmtGB(g.size_gb)),
        h('td', {}, h('button', { class: g.kind === 'model' ? 'primary' : '', onclick: () => this.download(repo, g) }, 'Download')))))));
    } catch (e) { clear(box).append(h('div', { class: 'note bad' }, e.message)); }
  },
  async download(repo, g) {
    if (g.size_gb > 20 && !await confirmBox('Large download', `${g.name} is ${fmtGB(g.size_gb)}.\nIt will be saved in:\n${S.values.models_dir || '~/Duckstoken/models'}\n\nMake sure that is on your SSD. Downloads can be resumed if interrupted.`)) return;
    try { await api('hf/download', { repo, files: g.files }); toast('Download started — see the Dashboard', 'ok'); } catch (e) { fail(e); }
  },
  async findEngine() {
    const box = clear(this.refs.engine); box.append(h('span', { class: 'muted' }, 'Asking GitHub…'));
    try {
      const d = await api('llama/releases?backend=' + this.refs.backend.value); clear(box);
      const rec = new Set(d.recommended.map((a) => a.name)); const checks = [];
      box.append(h('p', {}, `Latest: ${d.tag} · suggested backend for your PC: `, h('b', {}, d.backend.toUpperCase())));
      d.assets.forEach((a) => {
        const cb = h('input', { type: 'checkbox', checked: rec.has(a.name) }); checks.push([cb, a]);
        box.append(h('label', { class: 'row small' }, cb, a.name, h('span', { class: 'muted' }, fmtBytes(a.size))));
      });
      box.append(h('div', { style: 'margin-top:8px' }, h('button', { class: 'primary', onclick: async () => {
        const sel = checks.filter(([c]) => c.checked).map(([, a]) => a);
        try { await api('llama/install', { assets: sel, tag: d.tag, backend: d.backend }); toast('Installing — see the Dashboard', 'ok'); } catch (e) { fail(e); }
      } }, 'Install selected')));
      if (d.backend === 'cuda') box.append(h('div', { class: 'note' }, 'CUDA builds need the matching “cudart” zip too (pre-ticked). Old cards (GTX 750 = Maxwell) need a CUDA 12.x build; CUDA 13 dropped them.'));
    } catch (e) { clear(box).append(h('div', { class: 'note bad' }, e.message)); }
  },
};
