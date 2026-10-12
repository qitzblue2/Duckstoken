'use strict';
/* Diagnostics tab: hardware report, compatibility matrix (supported / unsupported / experimental / unknown),
   engine capabilities, recent failures, logs, and a copyable plain-text report. Backend: /api/vgpu/{diagnostics,compat,logs}. */

const VDiag = {
  refs: {}, logTab: 'server', data: null,
  init(el) {
    const r = this.refs;
    r.hw = h('div'); r.compat = h('div'); r.engine = h('div'); r.fail = h('div'); r.log = h('pre', { class: 'log', style: 'height:300px' });
    r.logTabs = h('div', { class: 'subtabs', style: 'margin:0 0 8px' }, [['server', 'Model server'], ['big', 'Big model'], ['perf', 'Performance log'], ['app', 'Duckstoken']]
      .map(([k, t]) => h('button', { 'data-k': k, onclick: () => { this.logTab = k; this.loadLog(); } }, t)));
    r.text = h('pre', { class: 'report' }, 'Building report…');
    el.append(
      h('div', { class: 'card' }, h('div', { class: 'row' }, h('h2', { style: 'margin:0' }, '🩺 Diagnostics'), h('span', { class: 'grow' }),
        h('button', { onclick: () => this.load(true) }, '↻ Re-scan everything'),
        h('button', { class: 'primary', onclick: () => this.data && VU.copy(this.data.text) }, '📋 Copy diagnostic report')),
        h('div', { class: 'muted small', style: 'margin-top:6px' }, 'Values that cannot be read on this machine say “Unavailable” instead of a guess. The copied report hides API keys and tokens.')),
      h('div', { class: 'card' }, h('h2', {}, 'Hardware'), r.hw),
      h('div', { class: 'card' }, h('h2', {}, 'Compatibility ', h('span', { class: 'muted small' }, 'supported · experimental · unsupported · unknown')), r.compat),
      h('div', { class: 'card' }, h('h2', {}, 'Inference engine & backends'), r.engine),
      h('div', { class: 'card' }, h('h2', {}, 'Recent problems (failed estimates, rejected loads, crashes)'), r.fail),
      h('div', { class: 'card' }, h('h2', {}, 'Logs'), r.logTabs, r.log),
      h('div', { class: 'card' }, h('h2', {}, 'Copyable report'), r.text));
  },
  show() { this.load(false); this.loadLog(); },
  async load(refresh) {
    try {
      const d = await api('vgpu/diagnostics' + (refresh ? '?refresh=1' : '')); this.data = d; this.render(d);
    } catch (e) { clear(this.refs.hw).append(h('div', { class: 'note bad' }, e.message)); }
  },
  render(d) {
    const r = this.refs, hw = d.report.hardware || {}, cpu = hw.cpu || {}, ram = hw.ram || {};
    clear(r.hw).append(h('div', { class: 'kv' },
      h('div', {}, 'OS'), h('div', {}, hw.os || 'Unavailable'),
      h('div', {}, 'CPU'), h('div', {}, `${cpu.name || 'Unavailable'} — ${cpu.physical_cores || '?'}C/${cpu.logical_cores || '?'}T${cpu.avx2 ? ' · AVX2' : ' · no AVX2'}${cpu.avx512 ? ' · AVX-512' : ''}`),
      h('div', {}, 'RAM'), h('div', {}, `${fmtGB(ram.total_gb)} total · ${fmtGB(ram.avail_gb)} available` + (ram.speed_mts ? ` · ${ram.speed_mts} MT/s × ${ram.sticks} ≈ ${ram.bandwidth_gbs} GB/s` : ' · speed unavailable')),
      h('div', {}, 'GPU'), h('div', {}, (hw.gpus || []).length ? hw.gpus.map((g) => h('div', {}, `${g.name} · ${g.vram_mb} MB VRAM · driver ${g.driver || 'unavailable'}${g.compute_cap ? ' · compute ' + g.compute_cap : ''}`)) : 'None detected'),
      h('div', {}, 'Drives'), h('div', {}, (hw.disks || []).map((x) => h('div', {}, `${x.drive} ${x.kind} — ${fmtGB(x.free_gb)} free of ${fmtGB(x.size_gb)}${x.model ? ' (' + x.model + ')' : ''}`)))));
    const gc = d.compat || {};
    clear(r.compat);
    (gc.gpu || []).forEach((g) => r.compat.append(h('div', { class: 'note' + (g.legacy ? ' warn' : '') }, h('b', {}, g.name), ` — ${g.architecture || 'unknown architecture'}${g.compute_cap ? ' (compute ' + g.compute_cap + ')' : ''}${g.legacy ? ' · legacy GPU' : ''}`,
      (g.notes || []).map((n) => h('div', { class: 'small muted' }, n)))));
    if (gc.summary) r.compat.append(h('div', { class: 'muted small' }, gc.summary));
    r.compat.append(h('table', { class: 'compact' }, h('thead', {}, h('tr', {}, ['Feature', 'Status', 'Detail', 'How to verify'].map((t) => h('th', {}, t)))),
      h('tbody', {}, (gc.items || []).map((i) => h('tr', {}, h('td', {}, i.label), h('td', {}, VU.statusChip(i.status), h('div', { class: 'muted small' }, i.evidence || '')), h('td', { class: 'small' }, i.detail), h('td', { class: 'small muted' }, i.how_to_verify || ''))))));
    const p = d.probe || {};
    clear(r.engine).append(p.exists ? h('div', {},
      h('div', { class: 'kv' }, h('div', {}, 'Engine'), h('div', { class: 'mono' }, p.path), h('div', {}, 'Version'), h('div', {}, p.version ? `${p.version}${p.build ? ' (build ' + p.build + ')' : ''}${p.commit ? ' · ' + p.commit : ''}` : 'Unavailable'),
        h('div', {}, 'Backends'), h('div', {}, (p.backends || []).join(', ') || 'CPU only / unavailable'),
        h('div', {}, 'Devices'), h('div', {}, (p.devices || []).length ? p.devices.map((x) => h('div', {}, `${x.id}: ${x.name}${x.total_mb ? ' · ' + x.total_mb + ' MB' : ''}`)) : 'No GPU devices listed — inference runs on the CPU'),
        h('div', {}, 'Architectures'), h('div', {}, Object.entries(p.supports || {}).map(([k, v]) => h('span', { style: 'margin-right:12px' }, k + ' ', VU.statusChip(v === true ? 'supported' : v === false ? 'unsupported' : 'unknown'))))),
      (p.errors || []).map((e) => h('div', { class: 'note warn' }, e)))
      : h('div', { class: 'note' }, p.errors && p.errors.length ? p.errors.join(' ') : 'No llama.cpp engine found yet. Install it from Models → Engine (or pick another engine in Settings).'));
    clear(r.fail);
    if (!(d.failures || []).length) r.fail.append(h('div', { class: 'muted' }, 'Nothing has gone wrong in this session.'));
    (d.failures || []).slice().reverse().forEach((f) => r.fail.append(h('div', { class: 'note bad' }, h('b', {}, f.kind), ' · ', new Date((f.t || 0) * 1000).toLocaleTimeString(), h('div', {}, f.message))));
    r.text.textContent = d.text || '';
  },
  async loadLog() {
    const r = this.refs;
    r.logTabs.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.k === this.logTab));
    try { const d = await api('vgpu/logs?which=' + this.logTab + '&n=400'); r.log.textContent = (d.lines || []).join('\n') || '(empty)'; r.log.scrollTop = r.log.scrollHeight; }
    catch (e) { r.log.textContent = e.message; }
  },
};
window.VDiag = VDiag;
