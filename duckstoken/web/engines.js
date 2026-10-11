'use strict';
/* "Inference engine" card on the Models page: pick llama.cpp (bundled), an external server, or the built-in engine. */
const EnginePanel = {
  refs: {}, info: null,
  init(container) {
    this.refs.body = h('div');
    container.append(h('div', { class: 'card' }, h('h2', {}, '⚙ Inference engine'),
      h('p', { class: 'muted' }, 'The engine is the program that actually runs the model. Pick one — Duckstoken talks to all of them the same way.'),
      this.refs.body));
  },
  async refresh() {
    try { this.info = await api('engine/info'); S.engineInfo = this.info; } catch (e) { fail(e); return; }
    this.render();
  },
  render() {
    const box = clear(this.refs.body), sel = this.info.selected, v = S.values;
    const blurb = {
      llamacpp: 'Fast native engine; runs the biggest models (gpt-oss-120b) from RAM + SSD. Bundled inside Duckstoken.',
      external: 'Use Ollama, LM Studio, vLLM or any OpenAI-compatible server you already run.',
      builtin: 'Duckstoken’s own pure-Python engine: no other program at all, but only small models (≈3B and below) and much slower.',
    };
    this.info.engines.forEach((e) => {
      const ok = e.available !== false;
      const status = e.id === 'llamacpp' ? (ok ? (e.bundled ? 'bundled ✔' : 'installed ✔') : 'not installed') : e.id === 'builtin' ? (ok ? 'ready ✔' : 'unavailable') : 'needs a running server';
      const radio = h('input', { type: 'radio', name: 'engine', checked: sel === e.id, disabled: !ok && e.id === 'builtin',
        onchange: async () => { if (await setSetting('engine', e.id)) { toast('Engine: ' + e.label, 'ok'); this.refresh(); } } });
      box.append(h('label', { class: 'row', style: 'align-items:flex-start;margin:8px 0;cursor:pointer' }, radio,
        h('span', { class: 'grow' }, h('b', {}, e.label), ' ', h('span', { class: 'badge ' + (ok ? 'good' : 'unknown') }, status), h('div', { class: 'muted small' }, blurb[e.id]),
          !ok && e.problem ? h('div', { class: 'small', style: 'color:var(--warn)' }, e.problem) : null)));
    });
    if (sel === 'external') box.append(this.externalForm());
    if (sel === 'builtin') box.append(this.builtinForm());
    if (sel === 'llamacpp') box.append(h('div', { class: 'muted small' }, 'In use: ',
      h('span', { class: 'mono' }, v.llama_server_path || (this.info.bundled ? '(bundled with Duckstoken)' : 'none yet — install a build below')),
      ' · GPU builds (CUDA/Vulkan) can be installed in the card below.'));
  },
  externalForm() {
    const v = S.values, out = h('div', { class: 'small', style: 'margin-top:6px' });
    const modelSel = h('select', { onchange: (e) => setSetting('external_model', e.target.value) }, h('option', { value: v.external_model || '' }, v.external_model || '(first model the server reports)'));
    const url = h('input', { type: 'text', value: v.external_url, onchange: (e) => setSetting('external_url', e.target.value) });
    const key = h('input', { type: 'password', value: v.external_api_key, placeholder: 'API key (optional)', onchange: (e) => setSetting('external_api_key', e.target.value) });
    const preset = h('select', { onchange: async (e) => { await setSetting('external_preset', e.target.value); this.render(); } },
      [['ollama', 'Ollama (port 11434)'], ['lmstudio', 'LM Studio (port 1234)'], ['custom', 'Custom URL']].map(([k, l]) => h('option', { value: k, selected: v.external_preset === k }, l)));
    const test = async () => {
      clear(out).append('Testing…');
      try {
        const r = await api('engine/test', {}); clear(out);
        if (!r.ok) { out.append(h('div', { class: 'note bad' }, r.error)); return; }
        out.append(h('div', { class: 'note good' }, `Connected — ${r.models.length} model(s) found.`));
        clear(modelSel).append(h('option', { value: '' }, '(first model the server reports)'), r.models.map((m) => h('option', { value: m, selected: v.external_model === m }, m)));
      } catch (e) { clear(out).append(h('div', { class: 'note bad' }, e.message)); }
    };
    return h('div', { class: 'note', style: 'margin-top:10px' },
      h('div', { class: 'row' }, h('div', { style: 'min-width:200px' }, preset), v.external_preset === 'custom' ? h('div', { class: 'grow' }, url) : null, h('div', { class: 'grow' }, key)),
      h('div', { class: 'row', style: 'margin-top:8px' }, h('div', { class: 'grow' }, modelSel), h('button', { onclick: test }, 'Test connection')), out,
      h('div', { class: 'muted small', style: 'margin-top:6px' }, 'Then press ▶ Start on the Dashboard to connect. Duckstoken never starts or stops that server.'));
  },
  builtinForm() {
    const list = h('div', { class: 'muted small' }, 'Loading model list…');
    api('engine/builtin/catalog').then((d) => {
      clear(list).append(h('table', {}, h('tbody', {}, d.models.map((m) => h('tr', {},
        h('td', {}, h('b', {}, m.label), h('div', { class: 'muted small' }, m.note || '')), h('td', {}, fmtGB(m.size_gb)),
        h('td', {}, h('button', { onclick: async () => {
          try { await api('hf/download', { repo: m.repo, files: [{ path: m.file, size: Math.round(m.size_gb * 2 ** 30) }] }); toast('Download started — see the Dashboard', 'ok'); } catch (e) { fail(e); } } }, 'Download')))))));
    }).catch((e) => clear(list).append(h('span', { class: 'note bad' }, e.message)));
    return h('div', { class: 'note', style: 'margin-top:10px' }, h('b', {}, 'Small models that work with the built-in engine'), list,
      h('div', { class: 'muted small', style: 'margin-top:6px' }, 'Pick the downloaded file as the model on the Dashboard, then ▶ Start. Expect a few tokens per second on a typical CPU.'));
  },
};
