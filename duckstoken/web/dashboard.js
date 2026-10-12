'use strict';
function verdictBadge(v) {
  const label = { good: 'Should run well', ok: 'Fits in memory', slow: 'Will be slow', unusable: 'Not practical here' }[v] || v;
  return h('span', { class: 'badge ' + v }, label);
}
function renderFeasibility(f, info) {
  const box = h('div');
  if (info) {
    box.append(h('div', { class: 'row' }, h('b', {}, info.name), h('span', { class: 'muted small' },
      `${info.architecture || '?'} · ${fmtGB(info.size_gb)} · ${info.block_count} layers` +
      (info.is_moe ? ` · MoE ${info.expert_used_count}/${info.expert_count} experts` : ''))));
  }
  box.append(h('div', { style: 'margin:8px 0' }, verdictBadge(f.verdict), ' ',
    h('span', { class: 'badge ' + f.drive_kind }, `${f.drive} ${f.drive_kind}`)));
  box.append(h('div', {}, f.summary));
  f.warnings.forEach((w) => box.append(h('div', { class: 'note bad' }, w)));
  f.tips.forEach((w) => box.append(h('div', { class: 'note' }, w)));
  box.append(h('div', { class: 'muted small' }, 'Estimates are rough; the chat footer shows your real tokens/s.'));
  return box;
}

const Dashboard = {
  logNext: 0, refs: {}, models: [],
  init() {
    const el = $('#page-dashboard'), r = this.refs;
    r.state = h('span', { class: 'badge' }, 'stopped');
    r.info = h('span', { class: 'muted small' });
    r.model = h('select', { onchange: (e) => setSetting('model_path', e.target.value).then(() => this.check()) });
    r.err = h('div');
    r.start = h('button', { class: 'primary', onclick: () => this.control('start') }, '▶ Start');
    r.stop = h('button', { onclick: () => this.control('stop') }, '■ Stop');
    r.restart = h('button', { onclick: () => this.control('restart') }, '↻ Restart');
    r.cmd = h('div', { class: 'mono muted', style: 'word-break:break-all' });
    r.log = h('pre', { class: 'log' });
    r.feas = h('div', {}, h('span', { class: 'muted' }, 'Pick a model, then press “Check this model”.'));
    r.hw = h('div');
    r.jobs = h('div');
    r.banner = h('div');
    r.steps = h('div');
    el.append(r.banner, h('div', { class: 'card' }, h('h2', {}, '🚀 Getting started'), r.steps),
      h('div', { class: 'grid' },
        h('div', {},
          h('div', { class: 'card' }, h('h2', {}, '🧠 Model server ', r.state),
            r.modelRow = h('div', { class: 'row' }, h('div', { class: 'grow' }, r.model),
              h('button', { onclick: async () => { const p = await pickPath({ kind: 'file', ext: '.gguf', start: S.values.model_path || S.values.models_dir }); if (p) { await setSetting('model_path', p); this.loadModels(); } } }, 'Browse…')),
            r.engineNote = h('div', { class: 'muted small', style: 'margin-top:6px' }),
            h('div', { class: 'row', style: 'margin:10px 0' }, r.start, r.stop, r.restart,
              h('button', { onclick: () => this.autotune() }, '⚡ Auto-tune for my PC'), r.info),
            r.err,
            r.live = h('div', { class: 'vrow small', style: 'margin:6px 0' }),
            h('details', {}, h('summary', { class: 'muted small' }, 'Command line'), r.cmd),
            h('div', { style: 'height:8px' }), r.log),
          h('div', { class: 'card' }, h('h2', {}, '⬇ Downloads & installs'), r.jobs)),
        h('div', {},
          h('div', { class: 'card' }, h('h2', {}, '🔎 Can my PC run this model?',
            h('button', { style: 'margin-left:auto', onclick: () => this.check() }, 'Check this model')), r.feas),
          h('div', { class: 'card' }, h('h2', {}, '🖥 Hardware ',
            h('button', { style: 'margin-left:auto', onclick: () => this.loadHardware(true) }, 'Re-scan')), r.hw))));
  },
  async show() { this.loadModels(); this.loadHardware(); this.pullLogs(); },
  async loadModels() {
    try {
      const d = await api('models'); this.models = d.models.filter((m) => m.kind === 'model');
      const sel = this.refs.model; clear(sel);
      const cur = S.values.model_path;
      if (cur && !this.models.some((m) => m.path === cur)) sel.append(h('option', { value: cur }, cur));
      if (!cur) sel.append(h('option', { value: '' }, '— no model selected —'));
      this.models.forEach((m) => sel.append(h('option', { value: m.path }, `${m.name}  (${fmtGB(m.size_gb)})`)));
      sel.value = cur || '';
    } catch (e) { fail(e); }
  },
  async loadHardware(refresh) {
    try {
      const hw = await api('hardware' + (refresh ? '?refresh=1' : '')); S.hw = hw;
      const g = hw.gpus.length ? hw.gpus.map((x) => `${x.name} (${x.vram_mb >= 1024 ? (x.vram_mb / 1024).toFixed(0) + ' GB' : x.vram_mb + ' MB'} VRAM)`) : ['none detected'];
      const box = clear(this.refs.hw);
      box.append(h('div', { class: 'kv' },
        h('div', {}, 'CPU'), h('div', {}, `${hw.cpu.name} — ${hw.cpu.physical_cores}C/${hw.cpu.logical_cores}T${hw.cpu.avx2 ? ' · AVX2' : ' · no AVX2!'}${hw.cpu.avx512 ? ' · AVX-512' : ''}`),
        h('div', {}, 'RAM'), h('div', {}, `${fmtGB(hw.ram.total_gb)} total, ${fmtGB(hw.ram.avail_gb)} free` + (hw.ram.speed_mts ? ` · ${hw.ram.speed_mts} MT/s × ${hw.ram.sticks} stick(s) ≈ ${hw.ram.bandwidth_gbs} GB/s` : '')),
        h('div', {}, 'GPU'), h('div', {}, g.map((x) => h('div', {}, x))),
        h('div', {}, 'Best engine'), h('div', {}, hw.backend_suggestion.toUpperCase())),
        h('h3', {}, 'Drives'),
        h('table', {}, h('thead', {}, h('tr', {}, ['Drive', 'Type', 'Free / size', ''].map((t) => h('th', {}, t)))),
          h('tbody', {}, hw.disks.map((d) => h('tr', {},
            h('td', {}, d.drive, h('div', { class: 'muted small' }, d.model)),
            h('td', {}, h('span', { class: 'badge ' + d.kind }, d.kind === 'unknown' ? 'unknown' : d.kind.toUpperCase())),
            h('td', {}, `${fmtGB(d.free_gb)} / ${fmtGB(d.size_gb)}`),
            h('td', {}, h('button', { onclick: () => this.benchmark(d) }, 'Test speed'), ' ',
              h('button', { onclick: () => this.useDrive(d) }, 'Store models here')))))));
      if (hw.disks.some((d) => d.kind === 'hdd') && hw.disks.some((d) => d.kind === 'ssd' || d.kind === 'nvme'))
        box.append(h('div', { class: 'note warn' }, 'You have both SSD and HDD. Big models must live on the SSD — a hard drive makes models larger than your RAM unusably slow.'));
    } catch (e) { clear(this.refs.hw).append(h('span', { class: 'muted' }, 'Could not read hardware: ' + e.message)); }
  },
  async benchmark(d) {
    toast('Testing ' + d.drive + ' (writes a 256 MB temp file)…');
    const dir = d.drive.includes(':') ? d.drive + '\\' : d.drive;
    try { const r = await api('disk/benchmark', { path: dir }); toast(`${d.drive} write ${r.write_mb_s} MB/s, read ${r.read_mb_s} MB/s`, 'ok'); } catch (e) { fail(e); }
  },
  async useDrive(d) {
    const sep = d.drive.includes(':') ? '\\' : '/';
    const p = d.drive.replace(/[\\/]$/, '') + sep + 'Duckstoken' + sep + 'models';
    if (await setSetting('models_dir', p)) { toast('Models folder is now ' + p, 'ok'); this.loadModels(); }
  },
  async check() {
    const box = clear(this.refs.feas);
    if (!S.values.model_path) { box.append(h('span', { class: 'muted' }, 'No model selected yet.')); return; }
    box.append(h('span', { class: 'muted' }, 'Reading model…'));
    try { const r = await api('models/inspect', { path: S.values.model_path }); clear(box).append(renderFeasibility(r.feasibility, r.info)); }
    catch (e) { clear(box).append(h('div', { class: 'note bad' }, e.message)); }
  },
  async autotune() {
    try {
      const r = await api('tune', { apply: true });
      const m = await modal('Auto-tune applied', h('div', {}, h('p', {}, 'Engine suggestion: ' + r.backend.toUpperCase()),
        r.notes.map((n) => h('div', { class: 'note' }, n)),
        h('pre', { class: 'log', style: 'height:auto;max-height:220px' }, Object.entries(r.values).map(([k, v]) => `${k} = ${v}`).join('\n'))));
      this.check();
      window.Settings && Settings.refresh();
    } catch (e) { fail(e); }
  },
  async control(action, force) {
    try {
      this.refs.err.replaceChildren();
      const r = await api('server/' + action, force ? { force: true } : {});
      S.runningSnapshot = { ...S.values }; S.dirty.clear(); updateDirtyBanner();
      toast(action === 'stop' ? 'Stopped' : 'Starting… large models can take minutes to load', 'ok');
    } catch (e) {
      const adm = e.data && e.data.admission;
      if (adm) {
        const body = h('div', {}, h('p', {}, e.message), VU.list(adm.blockers, 'bad'), VU.list(adm.warnings, 'warn'), VU.list(adm.suggestions, ''),
          h('div', { class: 'muted small' }, 'This is an estimate from your hardware. You can still try, but the PC may become very slow or the model may fail to load.'));
        const go = await modal('This model may not load', body, [{ label: 'Cancel', value: false }, { label: 'Open the planner', value: 'plan' }, { label: 'Start anyway', value: true, danger: true }]);
        if (go === true) return this.control(action, true);
        if (go === 'plan') window.go('vgpu');
        return;
      }
      fail(e);
    }
  },
  async pullTelemetry() {
    try {
      const t = await api('vgpu/telemetry?n=2'), s = t.latest || {}, p = t.pressure || {}, box = clear(this.refs.live);
      const gpu = (s.gpus || [])[0], proc = s.process || {};
      box.append(h('span', { class: 'badge ' + ({ ok: 'good', elevated: 'slow', high: 'error', critical: 'error' }[p.level] || '') }, 'memory: ' + (p.level || '?')),
        h('span', {}, 'RAM ', VU.fmt(s.ram && s.ram.used_pct, '%', 0)), h('span', {}, 'CPU ', VU.fmt(s.cpu && s.cpu.util_pct, '%', 0)),
        gpu ? h('span', {}, 'GPU ', VU.fmt(gpu.util_pct, '%', 0), ' · VRAM ', VU.fmt(gpu.vram_used_mb, 'MB', 0)) : h('span', { class: 'unavail' }, 'GPU telemetry unavailable'),
        proc.rss_gb && proc.rss_gb.value != null ? h('span', {}, 'model ', VU.fmt(proc.rss_gb, 'GB')) : null);
    } catch (e) { clear(this.refs.live); }
  },
  async pullLogs() {
    try {
      const d = await api('server/logs?since=' + this.logNext);
      if (d.next < this.logNext) { clear(this.refs.log); }
      if (d.lines.length) {
        const log = this.refs.log, stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
        log.append(document.createTextNode(d.lines.join('\n') + '\n'));
        if (stick) log.scrollTop = log.scrollHeight;
      }
      this.logNext = d.next;
    } catch (e) { /* ignore */ }
  },
  steps() {
    const v = S.values, ok = (b) => h('span', { class: 'badge ' + (b ? 'good' : 'unknown') }, b ? '✔' : '•');
    const rows = [
      [!!v.llama_server_path || !!S.engineInfo?.bundled || v.engine !== 'llamacpp', 'Choose an engine (llama.cpp is bundled)', 'Models → Engine', 'models'],
      [this.models.length > 0, 'Download a model (gpt-oss-120b or 20b)', 'Models → Hugging Face', 'models'],
      [!!v.model_path, 'Select the model and press ⚡ Auto-tune, then ▶ Start', '', ''],
      [S.status?.server?.state === 'ready', 'Chat', 'Chat', 'chat']];
    const box = clear(this.refs.steps);
    if (rows.every((x) => x[0])) { this.refs.steps.parentElement.style.display = 'none'; return; }
    this.refs.steps.parentElement.style.display = '';
    rows.forEach(([done, text, label, page]) => box.append(h('div', { class: 'row', style: 'margin:4px 0' }, ok(done), h('span', { class: 'grow' }, text),
      page && !done ? h('button', { onclick: () => go(page) }, label) : null)));
  },
  update(st) {
    const r = this.refs, s = st.server;
    this.steps();
    const ext = st.engine === 'external';
    r.modelRow.style.display = ext ? 'none' : '';
    r.engineNote.textContent = { llamacpp: '', external: 'Using an external server — set it up under Models → Inference engine.', builtin: 'Using the built-in engine (small models only).' }[st.engine] || '';
    r.state.className = 'badge ' + s.state; r.state.textContent = s.state === 'loading' ? 'loading model…' : s.state;
    r.start.disabled = s.running; r.stop.disabled = !s.running; r.restart.disabled = !s.running;
    r.info.textContent = s.running ? `up ${fmtDur(s.uptime)} · ${s.url}` : '';
    r.cmd.textContent = s.cmdline;
    clear(r.err); if (s.error) r.err.append(h('div', { class: 'note bad' }, s.error));
    clear(r.jobs);
    if (!st.jobs.length) r.jobs.append(h('span', { class: 'muted' }, 'Nothing downloading.'));
    st.jobs.forEach((j) => {
      const pct = j.total ? Math.min(100, j.done / j.total * 100) : 0;
      r.jobs.append(h('div', { style: 'margin-bottom:12px' },
        h('div', { class: 'row' }, h('b', { class: 'grow' }, j.title), h('span', { class: 'badge ' + (j.status === 'done' ? 'good' : j.status === 'error' ? 'error' : '') }, j.status),
          j.status === 'running' ? h('button', { onclick: () => api('jobs/cancel', { id: j.id }) }, 'Cancel') : null),
        h('div', { class: 'progress' }, h('div', { style: `width:${pct}%` })),
        h('div', { class: 'muted small' }, j.status === 'running' ? `${fmtBytes(j.done)} / ${fmtBytes(j.total)} · ${fmtBytes(j.speed)}/s · ${j.message}` : (j.error || j.message))));
    });
  },
};
