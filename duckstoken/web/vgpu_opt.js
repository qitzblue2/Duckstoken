'use strict';
/* Optimizer tab: candidate configurations, benchmark runs, measured-vs-estimated comparison, history, apply / revert.
   Talks to /api/vgpu/{opt,bench,job,profile}/*. Only one job (benchmark or optimization) runs at a time on the backend. */

const OBJECTIVES = {
  balanced: ['Best balance', 'Speed per GB of memory: the fastest setup that does not squeeze the rest of your PC.'],
  max_speed: ['Maximum speed', 'The configuration with the highest measured generation speed.'],
  max_model_size: ['Largest usable model', 'The biggest model that still runs at your minimum acceptable speed.'],
};

const VOptimizer = {
  refs: {}, timer: null, candidates: [], lastJobId: null,
  init(el) {
    const r = this.refs;
    r.model = h('select', { onchange: () => { this.loadCandidates(); this.loadProfile(); } });
    r.objective = h('select', { onchange: () => { r.objNote.textContent = OBJECTIVES[r.objective.value][1]; this.loadCandidates(); } },
      Object.entries(OBJECTIVES).map(([k, v]) => h('option', { value: k }, v[0])));
    r.objNote = h('div', { class: 'muted small' });
    r.estOnly = h('input', { type: 'checkbox' });
    r.allModels = h('input', { type: 'checkbox', onchange: () => this.loadCandidates() });
    r.profile = h('div');
    r.cands = h('div');
    r.job = h('div');
    r.result = h('div');
    r.history = h('div');
    r.bench = h('div');
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '⚡ Automatic optimizer'),
        h('p', { class: 'muted' }, 'Tries a short list of safe configurations on your real hardware — always starting with a CPU-only baseline — and measures each one with the same prompt, seed and settings. Prompt processing and token generation are reported separately. Nothing is applied until you say so.'),
        h('div', { class: 'row' }, h('div', { class: 'grow' }, r.model),
          h('label', { class: 'small muted', style: 'min-width:220px' }, 'Goal ', r.objective)),
        r.objNote,
        h('div', { class: 'row small', style: 'margin:8px 0' },
          h('label', {}, r.allModels, ' Compare all installed models (for “largest usable model”)'),
          h('label', {}, r.estOnly, ' Estimate only — do not run anything')),
        r.profile),
      h('div', { class: 'card' }, h('h2', {}, '🧪 Candidate configurations'), r.cands,
        h('div', { class: 'row', style: 'margin-top:10px' },
          h('button', { class: 'primary', onclick: () => this.start() }, '▶ Run optimizer'),
          h('button', { onclick: () => this.startBench() }, 'Benchmark my current settings only'))),
      r.job, r.result,
      h('div', { class: 'card' }, h('h2', {}, '📊 Benchmarks & history ', h('button', { style: 'margin-left:auto', onclick: () => this.loadHistory() }, 'Refresh')), r.bench, r.history));
    r.objNote.textContent = OBJECTIVES.balanced[1];
  },
  show() {
    this.loadModels(); this.loadHistory(); this.pollJob();
    clearInterval(this.timer); this.timer = setInterval(() => this.pollJob(), 1500);
  },
  hide() { clearInterval(this.timer); this.timer = null; },

  async loadModels() {
    try {
      const d = await api('models'), sel = this.refs.model, cur = sel.value || S.values.model_path || '';
      clear(sel).append(h('option', { value: '' }, '— choose a model —'));
      d.models.filter((m) => m.kind === 'model').forEach((m) => sel.append(h('option', { value: m.path }, `${m.name}  (${fmtGB(m.size_gb)})`)));
      if (cur && ![...sel.options].some((o) => o.value === cur)) sel.append(h('option', { value: cur }, cur));
      sel.value = cur;
      this.loadCandidates(); this.loadProfile();
    } catch (e) { fail(e); }
  },
  async loadProfile() {
    const box = clear(this.refs.profile), path = this.refs.model.value;
    if (!path) return;
    try {
      const d = await api('vgpu/profile?model_path=' + encodeURIComponent(path));
      const st = d.status;
      if (st === 'missing') box.append(h('div', { class: 'note' }, 'No saved optimization for this model on this PC yet.'));
      else {
        const p = d.profile || {};
        box.append(h('div', { class: 'note ' + (st === 'fresh' ? 'good' : 'warn') },
          st === 'fresh' ? '✅ Saved settings match this PC, model and engine. ' : '⚠ The hardware, model or engine changed since the last optimization — re-run it. ',
          p.objective ? `Goal: ${OBJECTIVES[p.objective] ? OBJECTIVES[p.objective][0] : p.objective}. ` : '',
          p.measured && VU.n(p.measured.gen_tps) != null ? `Measured ${Number(VU.n(p.measured.gen_tps)).toFixed(2)} tokens/s. ` : '',
          h('button', { style: 'margin-left:8px', onclick: () => this.apply(d.key) }, 'Apply to settings'),
          p.previous ? h('button', { style: 'margin-left:6px', onclick: () => this.revert(d.key) }, 'Undo last apply') : null));
      }
    } catch (e) { box.append(h('span', { class: 'muted small' }, e.message)); }
  },
  async loadCandidates() {
    const box = clear(this.refs.cands), path = this.refs.model.value;
    if (!path) { box.append(h('div', { class: 'muted' }, 'Choose a model to see which configurations would be tried.')); return; }
    box.append(h('span', { class: 'muted' }, 'Planning…'));
    try {
      const d = await api('vgpu/opt/candidates', { model_path: path, objective: this.refs.objective.value });
      this.candidates = d.candidates; clear(box);
      if (d.note) box.append(h('div', { class: 'note' }, d.note));
      box.append(h('table', { class: 'compact' },
        h('thead', {}, h('tr', {}, ['', 'Configuration', 'Changes', 'Planner verdict', 'Estimated speed'].map((t) => h('th', {}, t)))),
        h('tbody', {}, d.candidates.map((c, i) => h('tr', {},
          h('td', {}, i === 0 ? h('span', { class: 'vchip measured', title: 'Every GPU setup is compared against this' }, 'baseline') : ''),
          h('td', {}, c.label, c.expected ? h('div', { class: 'muted small' }, c.expected) : null),
          h('td', { class: 'mono small' }, Object.entries(c.overrides || {}).map(([k, v]) => `${k}=${v}`).join(' ') || 'current settings'),
          h('td', {}, c.admitted ? VU.statusChip(c.verdict === 'ok' ? 'ok' : 'warn') : VU.statusChip('fail'), ' ',
            h('span', { class: 'small' }, c.admitted ? (c.verdict || '') : (c.reject_reason || 'rejected — will not be run'))),
          h('td', {}, VU.range(c.tokens_per_s, 't/s')))))));
      box.append(h('div', { class: 'muted small' }, 'Rejected candidates are never started, so they cannot run you out of memory.'));
    } catch (e) { clear(box).append(h('div', { class: 'note bad' }, e.message)); }
  },

  async confirmRun(title, extra) {
    const running = S.status && S.status.server && S.status.server.running;
    return confirmBox(title, 'Duckstoken will load the model on its own temporary server, send a fixed test prompt several times and unload it again.\n\n' +
      (running ? '• Your chat model is running and will be STOPPED first (two copies would not fit in memory).\n' : '') +
      '• This can take several minutes for big models and will use a lot of CPU/RAM; the PC may feel slow meanwhile.\n• Your settings are not changed.\n' + (extra || ''));
  },
  async start() {
    const path = this.refs.model.value;
    if (!path) { toast('Choose a model first', 'err'); return; }
    const est = this.refs.estOnly.checked;
    const body = { model_path: path, objective: this.refs.objective.value, estimate_only: est, all_models: this.refs.allModels.checked };
    if (!est && !(await this.confirmRun('Run the optimizer?', `• Up to ${this.candidates.filter((c) => c.admitted).length} configurations are tested.`))) return;
    try { await api('vgpu/opt/start', { ...body, confirm: !est, stop_server: true }); clear(this.refs.result); this.pollJob(); } catch (e) { fail(e); }
  },
  async startBench() {
    const path = this.refs.model.value;
    if (!path) { toast('Choose a model first', 'err'); return; }
    if (!(await this.confirmRun('Run a benchmark?'))) return;
    try { await api('vgpu/bench/start', { model_path: path, confirm: true, stop_server: true, label: 'manual' }); clear(this.refs.result); this.pollJob(); } catch (e) { fail(e); }
  },

  async pollJob() {
    try {
      const j = await api('vgpu/job'), box = this.refs.job;
      if (j.state === 'idle') { clear(box); return; }
      const pr = j.progress || {}, pct = pr.total ? Math.round(100 * (pr.step || 0) / pr.total) : null;
      clear(box).append(h('div', { class: 'card' }, h('h2', {}, j.state === 'running' ? '⏳ Running: ' + j.kind : j.state === 'done' ? '✅ Finished: ' + j.kind : j.state === 'cancelled' ? 'Cancelled: ' + j.kind : '⚠ ' + j.kind + ' ' + j.state),
        pct != null ? h('div', { class: 'progress' }, h('div', { style: `width:${pct}%` })) : null,
        h('div', { class: 'small', style: 'margin:6px 0' }, pr.message || ''),
        j.error ? h('div', { class: 'note bad' }, j.error) : null,
        j.state === 'running' ? h('button', { onclick: () => api('vgpu/job/cancel', {}).then(() => this.pollJob()).catch(fail) }, 'Cancel (stops safely, nothing is applied)') : null,
        (j.log || []).length ? h('details', {}, h('summary', { class: 'muted small' }, 'Log'), h('pre', { class: 'log', style: 'height:140px' }, j.log.join('\n'))) : null));
      if (j.state === 'done' && j.id !== this.lastJobId) {  // the finished job stays in the runner until the next one: act once per job id
        this.lastJobId = j.id;
        if (j.kind === 'optimization' || j.kind === 'benchmark') { this.renderResult(j.kind, j.result); this.loadHistory(); this.loadProfile(); }
        else if (j.kind === 'stage') toast(j.result && j.result.ok ? 'Model copied to the fast drive' : 'Copy finished', 'ok');
      }
    } catch (e) { /* page not active */ }
  },

  renderResult(kind, res) {
    const box = clear(this.refs.result);
    if (kind === 'benchmark') { box.append(h('div', { class: 'card' }, h('h2', {}, 'Benchmark result'), this.benchDetail(res))); return; }
    const best = res.best;
    const card = h('div', { class: 'card' }, h('h2', {}, '🏁 Optimizer result'));
    if (res.estimate_only || res.status === 'estimate_only') card.append(h('div', { class: 'note' }, 'Estimate only: nothing was run, so every number below is an estimate and cannot be applied.'));
    if (res.status === 'aborted') card.append(h('div', { class: 'note bad' }, 'Stopped early because memory pressure became critical. ' + (res.error || '')));
    if (best) {
      card.append(h('div', { class: 'verdict ok' }, '🏆 Best for “' + ((OBJECTIVES[res.objective] || [res.objective])[0]) + '”: ' + (best.label || best.candidate_id),
        VU.n(best.score) != null ? h('span', { class: 'muted small', title: (best.score && best.score.note) || '' }, ' score ' + Number(VU.n(best.score)).toFixed(3)) : null,
        h('span', { class: 'vchip ' + (best.basis === 'measured' ? 'measured' : 'estimated') }, best.basis)),
        h('div', { class: 'small' }, best.reason || ''),
        best.vs_baseline ? h('div', { class: 'note ' + (best.vs_baseline.distinguishable ? 'good' : 'warn') }, best.vs_baseline.note || '') : null,
        h('div', { class: 'mono small' }, Object.entries(best.patch || {}).filter(([k]) => k !== 'model_path').map(([k, v]) => `${k}=${v}`).join('  ') || 'no changes needed — your current settings are already the best tested'),
        h('div', { class: 'row', style: 'margin:8px 0' },
          best.basis === 'measured' && best.profile_key ? h('button', { class: 'primary', onclick: () => this.apply(best.profile_key) }, 'Apply these settings') : null,
          h('span', { class: 'muted small' }, 'Your settings stay unchanged until you press this.')));
    } else {
      card.append(h('div', { class: 'note bad' }, res.error || (res.notes || []).join(' ') || 'No configuration could be measured. Your settings were not changed.'));
    }
    (res.models || []).forEach((m) => {
      card.append(h('h3', {}, m.name + (m.size_gb ? ` (${fmtGB(m.size_gb)})` : '')), m.error ? h('div', { class: 'note bad' }, m.error) : null,
        h('table', { class: 'compact' },
          h('thead', {}, h('tr', {}, ['Configuration', 'Status', 'Estimated t/s', 'Measured generation t/s', 'Measured prompt t/s', 'Δ vs estimate', 'vs baseline'].map((t) => h('th', {}, t)))),
          h('tbody', {}, (m.candidates || []).map((c) => h('tr', { class: c.id === m.best_id ? 'best' : '' },
            h('td', {}, c.label, c.id === m.best_id ? ' 🏆' : '', h('div', { class: 'mono small muted' }, Object.entries(c.overrides || {}).map(([k, v]) => `${k}=${v}`).join(' '))),
            h('td', {}, VU.statusChip(c.status === 'ok' || c.status === 'estimated' ? 'ok' : c.status === 'rejected' || c.status === 'skipped' ? 'warn' : 'fail'), ' ', h('span', { class: 'small muted' }, c.status === 'ok' ? '' : (c.reason || c.reject_reason || c.status))),
            h('td', { class: 'num' }, VU.fmt(c.estimated_tps, '', 2)),
            h('td', { class: 'num' }, VU.n(c.measured_gen_tps) != null ? VU.fmt(c.measured_gen_tps, '', 2) : h('span', { class: 'unavail' }, 'not measured')),
            h('td', { class: 'num' }, VU.n(c.measured_prompt_tps) != null ? VU.fmt(c.measured_prompt_tps, '', 1) : '—'),
            h('td', { class: 'num' }, VU.n(c.delta) != null ? ((VU.n(c.delta) > 0 ? '+' : '') + Number(VU.n(c.delta)).toFixed(2)) : '—'),
            h('td', { class: 'num' }, c.speedup_vs_baseline != null ? '×' + c.speedup_vs_baseline : '—'))))));
      if ((m.quant_alternatives || []).length) card.append(h('details', {}, h('summary', { class: 'muted small' }, 'Other quantizations of this model (estimates only, never benchmarked)'),
        h('table', { class: 'compact' }, h('tbody', {}, m.quant_alternatives.map((q) => h('tr', {}, h('td', {}, q.quant), h('td', { class: 'num' }, VU.gb(q.file_gb)), h('td', {}, q.verdict || '')))))));
    });
    if ((res.reverted || []).length) card.append(h('h3', {}, 'Reverted — never applied'), res.reverted.map((x) => h('div', { class: 'note warn' }, `${x.candidate || x.id}: ${x.reason || x.status}`)));
    if ((res.notes || []).length) card.append(res.notes.map((n) => h('div', { class: 'muted small' }, n)));
    box.append(card);
  },
  benchDetail(b) {
    const w = (b.warm && b.warm.gen_tps) || {}, pp = (b.warm && b.warm.prompt_tps) || {}, tt = (b.warm && b.warm.ttft_s) || {};
    const cold = b.cold || {};
    const f = (v, d = 2) => { const x = VU.n(v); return x == null ? h('span', { class: 'unavail' }, 'n/a') : Number(x).toFixed(d); };
    const m = b.metrics || {};
    return h('div', {},
      h('div', { class: 'row' }, VU.statusChip(b.status === 'ok' ? 'ok' : 'fail'), h('b', {}, (b.model && b.model.name) || ''), h('span', { class: 'muted small' }, VU.when(b.created))),
      b.error ? h('div', { class: 'note bad' }, b.error) : null, VU.list(b.reasons, 'warn'),
      h('table', { class: 'compact' }, h('thead', {}, h('tr', {}, ['', 'Prompt processing (t/s)', 'Generation (t/s)', 'Time to first token (s)'].map((t) => h('th', {}, t)))),
        h('tbody', {},
          h('tr', {}, h('td', {}, 'Cold (first request after load)'), h('td', { class: 'num' }, f(cold.prompt_tps, 1)), h('td', { class: 'num' }, f(cold.gen_tps)), h('td', { class: 'num' }, f(cold.ttft_s))),
          h('tr', {}, h('td', {}, `Warm (mean of ${(b.warm && b.warm.runs ? b.warm.runs.length : 0)})`), h('td', { class: 'num' }, f(pp.mean, 1)), h('td', { class: 'num' }, f(w.mean), w.stdev != null ? h('span', { class: 'muted small' }, ' ±' + Number(w.stdev).toFixed(2)) : null), h('td', { class: 'num' }, f(tt.mean))))),
      h('div', { class: 'kv', style: 'margin-top:8px' },
        h('div', {}, 'Load time'), h('div', {}, VU.fmt(b.load && b.load.seconds, 's')),
        h('div', {}, 'Peak RAM (process)'), h('div', {}, VU.fmt(m.peak_rss_gb, 'GB')),
        h('div', {}, 'Peak VRAM'), h('div', {}, VU.fmt(m.peak_vram_mb, 'MB', 0)),
        h('div', {}, 'CPU / GPU util'), h('div', {}, VU.fmt(m.cpu_util_pct, '%', 0), ' / ', VU.fmt(m.gpu_util_pct, '%', 0)),
        h('div', {}, 'Page faults'), h('div', {}, VU.fmt(m.page_faults, '', 0)),
        h('div', {}, 'Host RAM bandwidth'), h('div', {}, VU.fmt(m.host_bandwidth_gbs, 'GB/s')),
        h('div', {}, 'GPU↔host transfer'), h('div', {}, VU.fmt(m.transfer_bandwidth_gbs, 'GB/s'))),
      b.bottleneck ? h('div', { class: 'note' }, h('b', {}, 'Bottleneck: '), (b.bottleneck.kind || '').replace(/_/g, ' '), ' — ', b.bottleneck.explanation || '') : null,
      h('details', {}, h('summary', { class: 'muted small' }, 'Exact settings, engine and reproducibility info'),
        h('pre', { class: 'report' }, JSON.stringify({ backend: b.backend, config: b.config, spec: b.spec, reproducibility: b.reproducibility, hardware: b.hardware }, null, 2))));
  },

  async apply(key) {
    if (!(await confirmBox('Apply the optimized settings?', 'These settings replace the matching ones in Settings. You can undo this from the same banner. If the model server is running it must be restarted to use them.'))) return;
    try { const d = await api('vgpu/opt/apply', { key, confirm: true }); toast('Applied ' + Object.keys(d.applied || {}).length + ' setting(s)', 'ok'); if (window.Settings) Settings.refresh(); this.loadProfile(); } catch (e) { fail(e); }
  },
  async revert(key) {
    try { await api('vgpu/opt/revert', { key }); toast('Previous settings restored', 'ok'); if (window.Settings) Settings.refresh(); this.loadProfile(); } catch (e) { fail(e); }
  },

  async loadHistory() {
    const bx = clear(this.refs.bench), hx = clear(this.refs.history);
    try {
      const [b, o] = await Promise.all([api('vgpu/bench/list?limit=30'), api('vgpu/opt/history?limit=15')]);
      if (!b.benchmarks.length) bx.append(h('div', { class: 'muted' }, 'No benchmarks yet. Nothing here is ever invented — results appear after a real run.'));
      else bx.append(h('table', { class: 'compact' }, h('thead', {}, h('tr', {}, ['When', 'Model', 'Label', 'Status', 'Gen t/s', 'Prompt t/s', ''].map((t) => h('th', {}, t)))),
        h('tbody', {}, b.benchmarks.map((x) => {
          const w = x.warm || {};
          return h('tr', {}, h('td', { class: 'small' }, VU.when(x.created)), h('td', {}, (x.model && x.model.name) || ''), h('td', { class: 'small' }, x.label || ''),
            h('td', {}, VU.statusChip(x.status === 'ok' ? 'ok' : x.status === 'rejected' ? 'warn' : 'fail')),
            h('td', { class: 'num' }, w.gen_tps && w.gen_tps.mean != null ? Number(VU.n(w.gen_tps.mean)).toFixed(2) : '—'), h('td', { class: 'num' }, w.prompt_tps && w.prompt_tps.mean != null ? Number(VU.n(w.prompt_tps.mean)).toFixed(1) : '—'),
            h('td', {}, h('button', { onclick: () => this.showBench(x.id) }, 'Details'), ' ', h('button', { onclick: async () => { if (await confirmBox('Delete this benchmark record?', 'Only the saved result is removed.')) { await api('vgpu/bench/delete', { id: x.id, confirm: true }).catch(fail); this.loadHistory(); } } }, '✕')));
        }))));
      if (o.history.length) hx.append(h('h3', {}, 'Optimizer runs'), h('table', { class: 'compact' }, h('thead', {}, h('tr', {}, ['When', 'Goal', 'Models', 'Result', 'Winner'].map((t) => h('th', {}, t)))),
        h('tbody', {}, o.history.map((x) => h('tr', {}, h('td', { class: 'small' }, VU.when(x.created)), h('td', {}, (OBJECTIVES[x.objective] || [x.objective])[0]), h('td', { class: 'small' }, (x.models || []).join(', ')),
          h('td', {}, VU.statusChip(x.status === 'ok' ? 'ok' : x.estimate_only ? 'warn' : 'fail'), ' ', h('span', { class: 'small muted' }, x.estimate_only ? 'estimate' : x.status)),
          h('td', {}, x.best ? (x.best.label || x.best.candidate_id) : h('span', { class: 'unavail' }, 'none'), x.best ? h('span', { class: 'vchip ' + (x.best.basis === 'measured' ? 'measured' : 'estimated') }, x.best.basis) : null))))));
    } catch (e) { bx.append(h('div', { class: 'note bad' }, e.message)); }
  },
  async showBench(id) {
    try { const b = await api('vgpu/bench/get?id=' + encodeURIComponent(id)); await modal('Benchmark details', this.benchDetail(b.benchmark)); } catch (e) { fail(e); }
  },
};
window.VOptimizer = VOptimizer;
