'use strict';
/* GPT-OSS tab: an honest feasibility report for gpt-oss-120b / gpt-oss-20b on this PC or a what-if machine.
   Backend: POST /api/vgpu/gptoss. Nothing here downloads anything. */

const GPTOSS_VERDICTS = {
  comfortable: ['Comfortable', 'ok'], ok: ['Fits in memory', 'ok'], tight: ['Fits, but tight', 'tight'],
  paging: ['Loads, but pages from disk — slow', 'paging'], not_realistic: ['Not realistic on this PC', 'reject'],
};

const VGptOss = {
  refs: {}, report: null,
  init(el) {
    const r = this.refs;
    r.model = h('select', { onchange: () => this.run() }, h('option', { value: 'gpt-oss-120b' }, 'gpt-oss-120b (117B, ~59 GiB file)'), h('option', { value: 'gpt-oss-20b' }, 'gpt-oss-20b (21B, ~11 GiB file)'));
    r.ctx = h('input', { type: 'number', value: 4096, min: 256, max: 131072, step: 256, style: 'width:110px', onchange: () => this.run() });
    r.preset = h('select', { onchange: () => this.run() }, h('option', { value: '' }, 'This PC (detected)'));
    r.ram = h('input', { type: 'number', min: 1, placeholder: 'RAM GB', style: 'width:90px' });
    r.vram = h('input', { type: 'number', min: 0, placeholder: 'VRAM MB (0 = none)', style: 'width:150px' });
    r.disk = h('select', {}, ['', 'nvme', 'ssd', 'hdd'].map((k) => h('option', { value: k }, k ? 'Model on ' + k.toUpperCase() : 'Drive: as preset')));
    r.file = h('input', { type: 'text', placeholder: 'optional: path to a downloaded .gguf to validate', style: 'flex:1;min-width:260px' });
    r.out = h('div', { class: 'muted' }, 'Choose a model and press “Check”.');
    el.append(h('div', { class: 'card' }, h('h2', {}, '🦆 Can I run GPT-OSS here?'),
      h('p', { class: 'muted' }, 'The official gpt-oss-120b file is 63.4 GB (≈59 GiB; 117 B parameters, 128 experts, 4 used per token). Whether it is practical depends on RAM, the drive and the engine — not on hope. This report uses your detected hardware or a what-if machine, and labels every number as an estimate.'),
      h('div', { class: 'row' }, h('div', { class: 'grow' }, r.model), h('label', { class: 'small muted' }, 'Context ', r.ctx), h('div', { style: 'min-width:260px' }, r.preset),
        h('button', { class: 'primary', onclick: () => this.run() }, 'Check')),
      h('details', { style: 'margin-top:8px' }, h('summary', { class: 'muted small' }, 'Customize the what-if machine'),
        h('div', { class: 'row', style: 'margin-top:6px' }, r.ram, r.vram, r.disk, h('span', { class: 'muted small' }, 'Starts from the selected preset (default: the project owner’s PC).'))),
      h('div', { class: 'row', style: 'margin-top:8px' }, r.file, h('button', { onclick: async () => { const p = await pickPath({ kind: 'file', ext: '.gguf', start: S.values.models_dir }); if (p) { r.file.value = p; this.run(); } } }, 'Browse…'))),
      r.out);
  },
  async show() {
    if (!this.presetsLoaded) {
      try { (await api('vgpu/presets')).presets.forEach((p) => this.refs.preset.append(h('option', { value: p.name }, 'What if: ' + p.label))); this.presetsLoaded = true; } catch (e) { /* optional */ }
    }
    if (!this.report) this.run();
  },
  async run() {
    const r = this.refs, hw = {};
    if (r.ram.value) hw.ram_gb = parseFloat(r.ram.value);
    if (r.vram.value !== '') hw.vram_mb = parseInt(r.vram.value, 10);
    if (r.disk.value) hw.disk_kind = r.disk.value;
    clear(r.out).append(h('span', { class: 'muted' }, 'Calculating…'));
    try {
      const body = { model: r.model.value, ctx: parseInt(r.ctx.value, 10) || 4096, preset: r.preset.value || undefined };
      if (Object.keys(hw).length) body.hardware = hw;
      if (r.file.value.trim()) body.file_path = r.file.value.trim();
      this.report = await api('vgpu/gptoss', body); this.render(this.report);
    } catch (e) { clear(r.out).append(h('div', { class: 'note bad' }, e.message)); }
  },
  render(d) {
    const out = clear(this.refs.out), [label, cls] = GPTOSS_VERDICTS[d.verdict] || [d.verdict, 'tight'];
    out.append(h('div', { class: 'card' },
      h('div', { class: 'verdict ' + cls }, label, d.verdict === 'not_realistic' ? h('span', { class: 'muted small' }, ' · would not load with the memory available') : h('span', { class: 'muted small' }, ' · ', VU.range(d.tokens_per_s, 'tokens/s'), ' · ', d.speed_class || ''),
        d.what_if ? h('span', { class: 'vchip experimental' }, 'what-if: ' + d.what_if) : null),
      h('p', {}, d.explanation),
      d.recommendation ? h('div', { class: 'note ' + (d.recommendation.action === 'run' ? 'good' : 'warn') }, h('b', {}, d.recommendation.title),
        h('ul', {}, (d.recommendation.steps || []).map((x) => h('li', {}, x)))) : null));
    const rq = d.requirements, row = (n, v) => h('tr', {}, h('td', {}, n), h('td', { class: 'num' }, VU.gb(v, 2)));
    out.append(h('div', { class: 'grid' },
      h('div', { class: 'card' }, h('h2', {}, 'Memory required'), h('table', { class: 'compact' }, h('tbody', {},
        row('Weights', rq.weights_gb), row('KV cache (context ' + d.context + ')', rq.kv_cache_gb), row('Compute buffers', rq.compute_buffers_gb), row('Engine overhead', rq.runtime_overhead_gb),
        row('Kept for Windows & apps', rq.os_reserve_gb), row('Safety margin', rq.safety_margin_gb), h('tr', {}, h('td', {}, h('b', {}, 'Total')), h('td', { class: 'num' }, h('b', {}, VU.gb(rq.total_gb, 2))))))),
      h('div', { class: 'card' }, h('h2', {}, 'How much RAM would it take?'), h('table', { class: 'compact' },
        h('thead', {}, h('tr', {}, ['RAM', 'Outcome', 'Speed (estimate)', 'Paged from disk'].map((t) => h('th', {}, t)))),
        h('tbody', {}, (d.ram_table || []).map((x) => h('tr', { class: x.current ? 'best' : '' },
          h('td', {}, x.ram_gb + ' GB', x.current ? h('span', { class: 'vchip measured' }, 'this machine') : null),
          h('td', {}, VU.statusChip(x.outcome === 'fits' ? 'ok' : x.outcome === 'paging' ? 'warn' : 'fail'), ' ', h('span', { class: 'small' }, x.outcome)),
          h('td', { class: 'small' }, VU.n(x.tokens_per_s) ? [VU.range(x.tokens_per_s, 't/s'), h('div', { class: 'muted' }, x.speed_class)] : h('span', { class: 'unavail' }, x.speed_class || 'not loadable')),
          h('td', { class: 'num' }, VU.gb(x.spill_gb, 1))))),
        h('div', { class: 'muted small', style: 'margin-top:6px' }, '96–128 GB is the range where the whole model stays in memory. Smaller sizes either page from the drive or cannot load.')))));
    const g = d.gpu || {};
    out.append(h('div', { class: 'card' }, h('h2', {}, 'The graphics card'),
      g.present ? [(g.gpus || []).map((x) => h('div', {}, h('b', {}, x.name), ` — ${x.architecture || 'unknown architecture'}${x.compute_cap ? ' (compute ' + x.compute_cap + ')' : ''} · ${x.vram_gb} GB VRAM${x.legacy ? ' · legacy' : ''}`)),
        h('div', { class: 'note' }, g.benefit || ''),
        (g.can || []).map((t) => h('div', { class: 'small' }, '✔ ' + t)), (g.cannot || []).map((t) => h('div', { class: 'small' }, '✘ ' + t)),
        (g.notes || []).map((t) => h('div', { class: 'muted small' }, t))] : h('div', { class: 'muted' }, 'No GPU with usable VRAM detected: everything runs on the CPU from system RAM, which is the supported baseline.')));
    out.append(h('div', { class: 'card' }, h('h2', {}, 'Checks'), h('table', { class: 'compact' }, h('tbody', {}, (d.checks || []).map((c) =>
      h('tr', {}, h('td', {}, VU.statusChip(c.status)), h('td', {}, c.label), h('td', { class: 'small muted' }, c.detail)))))));
    if (d.alternative) {
      const a = d.alternative;
      out.append(h('div', { class: 'card' }, h('h2', {}, 'For comparison: ' + a.model), h('div', {}, VU.statusChip(a.verdict === 'ok' ? 'ok' : 'warn'), ` ${a.size_gb} GB file · `, VU.range(a.tokens_per_s, 'tokens/s'), h('span', { class: 'muted' }, ' · ' + (a.speed_class || '')))));
    }
    const dl = d.download;
    if (dl) out.append(h('div', { class: 'card' }, h('h2', {}, 'Download'), h('div', { class: 'small' }, `${dl.repo} · ${dl.file} · ${(dl.size_bytes / 1e9).toFixed(1)} GB`),
      h('div', { class: 'muted small' }, dl.note), h('div', { class: 'row', style: 'margin-top:6px' }, h('button', { onclick: () => go('models') }, 'Open the Models page (asks before downloading)'))));
    out.append(h('details', { class: 'assume' }, h('summary', {}, `Assumptions behind every number (${(d.assumptions || []).length})`),
      h('table', { class: 'compact' }, h('tbody', {}, (d.assumptions || []).map((x) => h('tr', {}, h('td', { class: 'mono' }, x.key), h('td', {}, String(x.value)), h('td', { class: 'muted small' }, x.explain)))))));
  },
};
window.VGptOss = VGptOss;
