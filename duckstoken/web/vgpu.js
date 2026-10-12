'use strict';
/* Virtual GPU page: shell with sub-tabs + shared helpers + the Memory tab.
   Other tabs (VOptimizer, VDiag, VGptOss) live in vgpu_opt.js / vgpu_diag.js and expose init(container), show(), hide().
   Every number from the backend is an "honest value": {value, status: measured|estimated|experimental|unavailable, source, note}. */

const SVGNS = 'http://www.w3.org/2000/svg';
function svg(tag, attrs, ...kids) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  kids.flat().forEach((c) => c && e.append(c));
  return e;
}

const VU = {
  isV: (v) => v && typeof v === 'object' && 'status' in v,
  n: (v) => (VU.isV(v) ? v.value : v),
  chip(v) {
    if (!VU.isV(v)) return null;
    return h('span', { class: 'vchip ' + v.status, title: [v.source, v.note].filter(Boolean).join(' — ') || v.status }, v.status);
  },
  /* number with unit and a status chip; "Unavailable" (with the reason as tooltip) when there is no value */
  fmt(v, unit, d = 1) {
    const x = VU.n(v);
    if (x == null || (typeof x === 'number' && !isFinite(x))) return h('span', { class: 'unavail', title: (v && v.note) || 'No data' }, 'Unavailable');
    return h('span', {}, (typeof x === 'number' ? x.toFixed(d) : String(x)) + (unit ? ' ' + unit : ''), VU.chip(v));
  },
  gb: (v, d = 1) => VU.fmt(v, 'GB', d),
  range(r, unit) {
    const v = Array.isArray(r) ? [VU.n(r[0]), VU.n(r[1])] : (VU.isV(r) && Array.isArray(r.value) ? r.value : null);
    if (!v || v[0] == null) return h('span', { class: 'unavail', title: (r && r.note) || '' }, 'Unavailable');
    return h('span', {}, `${v[0]}–${v[1]} ${unit}`, VU.chip(VU.isV(r) ? r : { status: 'estimated' }));
  },
  when(c) { const d = typeof c === 'number' ? new Date(c * 1000) : new Date(c); return isNaN(d) ? '' : d.toLocaleString(); },
  statusChip: (s) => h('span', { class: 'statuschip ' + s }, s),
  stat(label, content, sub) {
    return h('div', { class: 'vstat' }, h('div', { class: 'lab' }, label), h('div', { class: 'big' }, content), sub ? h('div', { class: 'lab' }, sub) : null);
  },
  /* stacked bar: parts [{label, gb, cls}] against `total` (physical or budget); anything past total is drawn red */
  bar(parts, total) {
    const sum = parts.reduce((a, p) => a + (p.gb || 0), 0);
    const scale = Math.max(total || 0, sum, 0.001);
    const box = h('div', { class: 'sbar' });
    let used = 0;
    parts.forEach((p) => {
      const w = Math.max(0, (p.gb || 0) / scale * 100);
      const over = total && used + (p.gb || 0) > total;
      box.append(h('div', { class: over ? 'over' : p.cls || 'a', style: `width:${w}%`, title: `${p.label}: ${(p.gb || 0).toFixed(2)} GB` }));
      used += p.gb || 0;
    });
    const lg = h('div', { class: 'legend' }, parts.map((p) => h('span', {}, h('i', { class: p.cls || 'a', style: 'background:currentColor' }), `${p.label} ${(p.gb || 0).toFixed(1)} GB`)));
    return h('div', {}, box, lg);
  },
  spark(values, { min = 0, max = null } = {}) {
    const v = values.filter((x) => x != null && isFinite(x));
    const box = h('div');
    if (v.length < 2) { box.append(h('div', { class: 'unavail small' }, 'collecting…')); return box; }
    const hi = max ?? Math.max(...v, 1), lo = min, W = 200, H = 40;
    const pts = v.map((x, i) => `${(i / (v.length - 1) * W).toFixed(1)},${(H - (x - lo) / Math.max(hi - lo, 1e-9) * H).toFixed(1)}`);
    box.append(svg('svg', { class: 'spark', viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none' },
      svg('polygon', { class: 'area', points: `0,${H} ${pts.join(' ')} ${W},${H}` }), svg('polyline', { points: pts.join(' ') })));
    return box;
  },
  list(items, cls) { return (items || []).map((t) => h('div', { class: 'note ' + (cls || '') }, t)); },
  async copy(text) {
    try { await navigator.clipboard.writeText(text); toast('Copied', 'ok'); } catch (e) { toast('Select the text and press Ctrl+C', 'err'); }
  },
};
window.VU = VU;

const TIER_COLORS = ['a', 'b', 'c', 'd'];

const VMemory = {
  refs: {}, timer: null, plan: null, presets: [], busy: false,
  init(el) {
    const r = this.refs;
    r.pressure = h('div');
    r.live = h('div', { class: 'vrow' });
    r.tiers = h('div', { class: 'tiers' });
    r.model = h('select', { onchange: () => this.runPlan() });
    r.ctx = h('input', { type: 'number', min: 256, step: 256, style: 'width:110px', onchange: () => this.runPlan() });
    r.preset = h('select', { onchange: () => this.runPlan() }, h('option', { value: '' }, 'This PC (detected)'));
    r.strict = h('input', { type: 'checkbox', onchange: () => this.runPlan() });
    r.planOut = h('div', {}, h('div', { class: 'muted' }, 'Pick a model to see where its weights, KV cache and buffers would live.'));
    r.sparks = h('div', { class: 'sparks' });
    r.cache = h('div');
    r.caps = h('div');
    el.append(
      r.pressure,
      h('div', { class: 'card' }, h('h2', {}, '📈 Live resources'), r.live, r.sparks),
      h('div', { class: 'card' }, h('h2', {}, '🧱 Memory tiers ', h('span', { class: 'muted small' }, 'shown separately on purpose — system RAM is not VRAM')), r.tiers),
      h('div', { class: 'card' }, h('h2', {}, '🧮 Will it fit? (planner)'),
        h('div', { class: 'row' }, h('div', { class: 'grow' }, r.model), h('label', { class: 'small muted' }, 'Context ', r.ctx),
          h('label', { class: 'small muted', style: 'min-width:240px' }, 'Hardware ', r.preset),
          h('button', { class: 'primary', onclick: () => this.runPlan() }, 'Check')),
        h('label', { class: 'small muted' }, r.strict, ' Use my current settings exactly (otherwise unset GPU/MoE knobs are auto-fitted to this hardware)'),
        r.planOut),
      h('div', { class: 'card' }, h('h2', {}, '📦 Quantization options for this model'), r.quants = h('div', { class: 'muted' }, 'Choose a model to compare quantizations.')),
      h('div', { class: 'card' }, h('h2', {}, '💾 Storage tiers: prewarm & staging cache'), r.cache),
      h('div', { class: 'card' }, h('h2', {}, '⚙ Active strategy & what the engine can expose'), r.caps));
  },
  show() {
    this.loadModels(); this.loadPresets(); this.loadCaps(); this.loadCache(); this.poll();
    clearInterval(this.timer); this.timer = setInterval(() => this.poll(), 2000);
  },
  hide() { clearInterval(this.timer); this.timer = null; },

  async loadModels() {
    try {
      const d = await api('models'); const sel = this.refs.model; const cur = sel.value || S.values.model_path || '';
      clear(sel).append(h('option', { value: '' }, '— choose a model —'));
      d.models.filter((m) => m.kind === 'model').forEach((m) => sel.append(h('option', { value: m.path }, `${m.name}  (${fmtGB(m.size_gb)})`)));
      if (cur && ![...sel.options].some((o) => o.value === cur)) sel.append(h('option', { value: cur }, cur));
      sel.value = cur;
      if (!this.refs.ctx.value) this.refs.ctx.value = S.values.ctx_size || 8192;
      if (cur && !this.plan) this.runPlan();
    } catch (e) { fail(e); }
  },
  async loadPresets() {
    if (this.presets.length) return;
    try {
      this.presets = (await api('vgpu/presets')).presets;
      this.presets.forEach((p) => this.refs.preset.append(h('option', { value: p.name }, 'What if: ' + p.label)));
    } catch (e) { /* what-if is optional */ }
  },

  async poll() {
    try {
      const t = await api('vgpu/telemetry?n=90');
      this.renderLive(t);
    } catch (e) { clear(this.refs.live).append(h('span', { class: 'unavail' }, 'Telemetry unavailable: ' + e.message)); }
  },
  renderLive(t) {
    const r = this.refs, s = t.latest || {}, p = t.pressure || {};
    clear(r.pressure).append(h('div', { class: 'pressure ' + (p.level || 'ok') },
      h('b', {}, 'Memory pressure: ' + (p.level || 'unknown')),
      h('span', { class: 'muted' }, p.avail_gb != null ? `${Number(p.avail_gb).toFixed(1)} GB available (${Number(p.avail_pct).toFixed(0)}%)` : ''),
      h('span', { class: 'grow muted small' }, (p.reasons || []).join(' · ')),
      t.enabled ? null : h('span', { class: 'muted small' }, 'Monitoring is off (Settings → Virtual GPU)')));
    const ram = s.ram || {}, cpu = s.cpu || {}, gpu = (s.gpus || [])[0], proc = s.process || {};
    clear(r.live).append(
      VU.stat('System RAM in use', VU.fmt(ram.used_pct, '%', 0), h('span', {}, VU.fmt(ram.avail_gb, 'GB free'))),
      VU.stat('CPU', VU.fmt(cpu.util_pct, '%', 0)),
      gpu ? VU.stat('GPU ' + (gpu.name || ''), VU.fmt(gpu.util_pct, '%', 0), h('span', {}, 'VRAM ', VU.fmt(gpu.vram_used_mb, 'MB', 0), ' / ', VU.fmt(gpu.vram_total_mb, 'MB', 0)))
        : VU.stat('GPU', h('span', { class: 'unavail', title: (s.sources && s.sources.gpu && s.sources.gpu.note) || '' }, 'Unavailable'),
          (s.sources && s.sources.gpu && s.sources.gpu.note) || 'No NVIDIA telemetry (AMD/Intel GPUs are not read)'),
      VU.stat('Model process memory', VU.fmt(proc.rss_gb, 'GB'), h('span', {}, 'peak ', VU.fmt(proc.peak_rss_gb, 'GB'), ' · page faults ', VU.fmt(proc.page_faults, '', 0))));
    const hist = t.history || [], col = (k) => hist.map((x) => x[k]);
    const sp = (label, k, unit, max) => h('div', {}, h('div', { class: 'lab' }, h('span', {}, label), h('span', {}, (hist.length && hist[hist.length - 1][k] != null) ? Number(hist[hist.length - 1][k]).toFixed(unit === '%' ? 0 : 1) + ' ' + unit : '—')), VU.spark(col(k), { max }));
    clear(r.sparks).append(sp('RAM used', 'ram_used_pct', '%', 100), sp('CPU', 'cpu_pct', '%', 100), sp('GPU', 'gpu_pct', '%', 100), sp('VRAM used', 'vram_used_mb', 'MB'), sp('Model process RSS', 'rss_gb', 'GB'));
    this.renderTiers(t);
  },

  tileFromHardware(t) {
    // physical capacities straight from the live sample when no plan has been made yet
    const hw = S.hw, s = t.latest || {};
    return { hw, s };
  },
  renderTiers(t) {
    const plan = this.plan, box = clear(this.refs.tiers), tiers = plan && plan.tiers;
    const s = t.latest || {}, hw = S.hw || {};
    const gpu = (hw.gpus || [])[0];
    const vramPhys = tiers ? tiers.vram.physical_gb : (gpu ? { value: gpu.vram_mb / 1024, status: 'measured', source: 'hardware scan' } : { value: 0, status: 'measured', source: 'hardware scan', note: 'no GPU detected' });
    const ramPhys = tiers ? tiers.ram.physical_gb : { value: hw.ram ? hw.ram.total_gb : null, status: 'measured', source: 'OS' };
    // VRAM
    const vi = tiers ? tiers.vram.items || [] : [];
    box.append(h('div', { class: 'tier' }, h('h3', {}, 'Tier 1 · GPU VRAM'),
      h('div', { class: 'cap' }, VU.gb(vramPhys)),
      h('div', { class: 'sub' }, gpu ? gpu.name : 'No GPU with usable VRAM detected — everything runs from system RAM.'),
      tiers ? [h('div', { class: 'small' }, 'Budget for this model: ', VU.gb(tiers.vram.budget_gb)), VU.bar(vi.map((x, i) => ({ label: x.label, gb: x.gb, cls: TIER_COLORS[i % 4] })), VU.n(tiers.vram.budget_gb))]
        : h('div', { class: 'muted small' }, 'Choose a model below to see its VRAM plan.'),
      h('div', { class: 'muted small' }, 'Highest bandwidth, smallest capacity. Reached through the PCIe link only.')));
    // RAM
    const ri = tiers ? tiers.ram.items || [] : [];
    box.append(h('div', { class: 'tier' }, h('h3', {}, 'Tier 2 · System RAM'),
      h('div', { class: 'cap' }, VU.gb(ramPhys)),
      h('div', { class: 'sub' }, 'Free now: ', VU.fmt(s.ram && s.ram.avail_gb, 'GB')),
      tiers ? [h('div', { class: 'small' }, 'Budget for this model: ', VU.gb(tiers.ram.budget_gb)), VU.bar(ri.map((x, i) => ({ label: x.label, gb: x.gb, cls: TIER_COLORS[i % 4] })), VU.n(tiers.ram.budget_gb))]
        : h('div', { class: 'muted small' }, 'Choose a model below to see its RAM plan.'),
      h('div', { class: 'muted small' }, 'Holds everything that does not fit in VRAM; the OS keeps its share.')));
    // storage
    const st = tiers && tiers.storage;
    box.append(h('div', { class: 'tier' }, h('h3', {}, 'Tier 3 · Memory-mapped storage'),
      h('div', { class: 'cap' }, st ? [VU.fmt(st.spill_gb, 'GB'), h('span', { class: 'small muted' }, ' paged from disk')] : h('span', { class: 'unavail' }, 'no plan yet')),
      h('div', { class: 'sub' }, st ? `${st.drive || '?'} · ${st.kind || 'unknown'} · ${st.free_gb != null ? Number(st.free_gb).toFixed(0) + ' GB free' : ''}` : ''),
      st ? [h('div', { class: 'small' }, 'Resident via mmap: ', VU.gb(st.mmap_resident_gb), ' · read speed ', VU.fmt(st.read_gbs, 'GB/s', 2)),
        st.model_on_slow_disk ? h('div', { class: 'note bad' }, 'The model sits on a slow drive: paging from it will be far slower than RAM.') : null] : null,
      h('div', { class: 'muted small' }, 'Not fast RAM: every page fault is a disk read (HDD ≈ 50× slower than SSD).')));
    // disk cache
    const dc = tiers && tiers.disk_cache;
    box.append(h('div', { class: 'tier' }, h('h3', {}, 'Tier 4 · Disk-backed cache'),
      h('div', { class: 'cap' }, dc && dc.enabled ? 'On' : 'Off'),
      h('div', { class: 'sub' }, (dc && dc.note) || 'Optional staging copy of a model on your SSD (below).'),
      h('div', { class: 'muted small' }, 'Bounded, recoverable, never deletes your originals.')));
  },

  async runPlan() {
    const r = this.refs, path = r.model.value;
    if (!path) { this.plan = null; return; }
    clear(r.planOut).append(h('span', { class: 'muted' }, 'Planning…'));
    try {
      const d = await api('vgpu/plan', { model_path: path, ctx_size: parseInt(r.ctx.value, 10) || undefined, preset: r.preset.value || undefined, use_settings: r.strict.checked });
      this.plan = d.plan; this.renderPlan(d); this.loadQuants(path);
      this.poll();
    } catch (e) { clear(r.planOut).append(h('div', { class: 'note bad' }, e.message)); }
  },
  renderPlan(d) {
    const p = d.plan, a = p.admission, rq = p.requirements, out = clear(this.refs.planOut);
    const label = { ok: 'Fits comfortably', tight: 'Fits, but tight', paging: 'Loads, but will page from disk', reject: 'Cannot load this configuration' }[a.verdict] || a.verdict;
    out.append(h('div', { class: 'verdict ' + a.verdict }, label,
      h('span', { class: 'muted small' }, ` headroom ${a.headroom_gb != null ? Number(a.headroom_gb).toFixed(1) : '?'} GB`),
      d.what_if ? h('span', { class: 'vchip experimental', title: 'Hardware is hypothetical' }, 'what-if: ' + d.what_if) : null));
    out.append(VU.list(a.blockers, 'bad'), VU.list(a.warnings, 'warn'), VU.list(a.suggestions, ''));
    const row = (name, v, hint) => h('tr', {}, h('td', {}, name, hint ? h('div', { class: 'muted small' }, hint) : null), h('td', { class: 'num' }, VU.gb(v, 2)));
    out.append(h('h3', {}, 'Requirements'),
      h('table', { class: 'compact' }, h('tbody', {},
        row('Model weights', rq.weights_gb, `${p.model.name} · ${p.model.quant || 'quant unknown'} · ${fmtGB(p.model.size_gb)} file`),
        row('KV cache', rq.kv_cache_gb, `context ${p.config.ctx_size} · K ${p.config.cache_type_k} / V ${p.config.cache_type_v}`),
        row('Compute buffers', rq.compute_buffers_gb), row('Engine overhead', rq.runtime_overhead_gb),
        row('Kept for the OS', rq.os_reserve_gb), row('Safety margin', rq.safety_margin_gb),
        h('tr', {}, h('td', {}, h('b', {}, 'Total')), h('td', { class: 'num' }, h('b', {}, VU.gb(rq.total_gb, 2)))))));
    const pl = p.placement || {};
    out.append(h('h3', {}, 'Placement'), h('div', { class: 'kv' },
      h('div', {}, 'GPU layers'), h('div', {}, `${pl.gpu_layers ?? '?'} of ${p.model.block_count || '?'}${pl.kv_on_gpu ? ' · KV cache on GPU' : ' · KV cache in RAM'}`),
      h('div', {}, 'MoE experts in RAM'), h('div', {}, p.model.is_moe ? `${pl.moe_experts_on_cpu_layers ?? 0} layers (${p.model.expert_used_count} of ${p.model.expert_count} experts used per token)` : 'not a mixture-of-experts model'),
      h('div', {}, 'Weights'), h('div', {}, `${Number(pl.weights_vram_gb || 0).toFixed(1)} GB VRAM · ${Number(pl.weights_ram_gb || 0).toFixed(1)} GB RAM · ${Number(pl.weights_spill_gb || 0).toFixed(1)} GB paged`)));
    const sp = p.speed || {};
    out.append(h('h3', {}, 'Expected speed'), h('div', {}, VU.range(sp.tokens_per_s, 'tokens/s'), h('span', { class: 'muted' }, ` · limited by ${(sp.bound_by || '?').replace('_', ' ')}`)),
      h('div', { class: 'muted small' }, sp.explanation || ''), h('div', { class: 'muted small' }, 'An estimate from memory bandwidth, not a measurement. Run the Optimizer for real numbers.'));
    if (d.bottleneck) out.append(h('div', { class: 'note' }, h('b', {}, 'Likely bottleneck: '), (d.bottleneck.kind || '').replace(/_/g, ' '), ' — ', d.bottleneck.explanation || ''));
    out.append(h('details', { class: 'assume' }, h('summary', {}, `Assumptions behind every number (${(p.assumptions || []).length})`),
      h('table', { class: 'compact' }, h('tbody', {}, (p.assumptions || []).map((x) => h('tr', {}, h('td', { class: 'mono' }, x.key), h('td', {}, String(x.value)), h('td', { class: 'muted small' }, x.explain)))))));
    if (!a.can_load) out.append(h('div', { class: 'muted small' }, 'You can still try: Dashboard → Start asks for confirmation and loads it anyway.'));
  },

  async loadQuants(path) {
    const box = clear(this.refs.quants);
    try {
      const d = await api('vgpu/quants?model_path=' + encodeURIComponent(path));
      box.append(h('div', { class: 'muted small' }, `This file is ${d.current || 'an unknown quantization'}. File sizes are estimates from bits per weight; quality notes are qualitative, not scores.`),
        h('table', { class: 'compact' }, h('thead', {}, h('tr', {}, ['Quantization', 'Bits/weight', 'File', 'RAM needed', 'Fits here?', 'Quality trade-off', 'How to get it'].map((t) => h('th', {}, t)))),
          h('tbody', {}, d.rows.map((x) => h('tr', { class: x.quant === d.current ? 'best' : '' },
            h('td', {}, h('b', {}, x.quant)), h('td', { class: 'num' }, Number(x.bits_per_weight).toFixed(2)), h('td', { class: 'num' }, VU.gb(x.file_gb)), h('td', { class: 'num' }, VU.gb(x.ram_gb)),
            h('td', {}, VU.statusChip(x.verdict === 'ok' ? 'ok' : x.verdict === 'reject' ? 'fail' : 'warn'), ' ', h('span', { class: 'small muted' }, x.verdict)),
            h('td', { class: 'small' }, x.quality_note), h('td', { class: 'small muted' }, x.conversion_required ? 'needs conversion' : (x.how_to_get || 'GGUF, loads directly')))))));
    } catch (e) { box.append(h('span', { class: 'muted small' }, e.message)); }
  },

  async loadCaps() {
    try {
      const d = await api('vgpu/capabilities'), box = clear(this.refs.caps);
      box.append(h('div', { class: 'note' }, h('b', {}, 'Active strategy: '), d.strategy.text));
      box.append(h('table', { class: 'compact' }, h('thead', {}, h('tr', {}, ['Capability', 'Available', 'How / limitation'].map((x) => h('th', {}, x)))),
        h('tbody', {}, d.capabilities.map((c) => h('tr', {}, h('td', {}, c.label), h('td', {}, VU.statusChip(c.available ? 'supported' : 'unsupported')),
          h('td', { class: 'small' }, c.available ? c.how : '', c.limitation ? h('div', { class: 'muted' }, c.limitation) : null))))));
    } catch (e) { clear(this.refs.caps).append(h('div', { class: 'note bad' }, e.message)); }
  },

  async loadCache() {
    const box = clear(this.refs.cache);
    try {
      const d = await api('vgpu/cache' + (this.refs.model.value ? '?model_path=' + encodeURIComponent(this.refs.model.value) : ''));
      const pw = d.prewarm || {};
      box.append(h('div', { class: 'muted small' }, 'Prewarm reads the model once, in order, into the OS file cache. That turns many slow random page faults into one fast sequential pass — it only helps if the model fits in free RAM, and it does not make a hard drive fast.'),
        h('div', { class: 'row', style: 'margin:8px 0' },
          h('button', { onclick: () => this.prewarm() }, '🔥 Prewarm the selected model'),
          pw.state === 'running' ? h('button', { onclick: () => api('vgpu/prewarm/cancel', {}).then(() => this.loadCache()) }, 'Cancel') : null,
          h('span', { class: 'muted small' }, pw.message || '')));
      const sg = d.staging;
      if (!sg.enabled) {
        box.append(h('div', { class: 'note' }, 'Staging cache is off. To use a fast drive as a cache tier, set a folder and a size limit in Settings → Virtual GPU & memory manager.'));
      } else {
        const used = (sg.usage.used_bytes || 0) / 2 ** 30;
        box.append(h('div', { class: 'kv' }, h('div', {}, 'Folder'), h('div', { class: 'mono' }, sg.dir), h('div', {}, 'Used'), h('div', {}, `${used.toFixed(1)} of ${Number(sg.budget_gb || 0).toFixed(0)} GB`)));
        if (d.stage_plan) {
          const sp = d.stage_plan;
          box.append(h('div', { class: 'note ' + (sp.worthwhile ? 'good' : '') }, sp.reason || (sp.worthwhile ? 'Copying this model to the fast drive would speed up paging.' : 'Not needed.')),
            sp.worthwhile ? h('button', { class: 'primary', onclick: () => this.stage(sp, sg.dir) }, `Copy to cache (${fmtGB(sp.size_gb)})…`) : null);
        }
        (sg.items || []).forEach((it) => box.append(h('div', { class: 'row small' }, h('span', { class: 'grow mono' }, it.name), h('span', { class: 'muted' }, fmtBytes(it.size)),
          h('button', { onclick: async () => { if (await confirmBox('Remove the staged copy?', 'Only the cached copy is deleted; the original model stays where it is.')) { await api('vgpu/cache/evict', { name: it.name, confirm: true }).catch(fail); this.loadCache(); } } }, 'Remove copy'))));
      }
      Object.entries(d.bounded || {}).forEach(([k, v]) => box.append(h('div', { class: 'muted small' }, `Cache “${k}”: ${v.hits} hits · ${v.misses} misses · ${v.evictions} evictions · ${v.items} items (${v.hit_rate == null ? 'no lookups yet' : Math.round(v.hit_rate * 100) + '% hit rate'})`)));
    } catch (e) { box.append(h('div', { class: 'note bad' }, e.message)); }
  },
  async prewarm() {
    const path = this.refs.model.value || S.values.model_path;
    if (!path) { toast('Choose a model first', 'err'); return; }
    try { await api('vgpu/prewarm/start', { model_path: path }); toast('Prewarming…', 'ok'); this.watchPrewarm(); } catch (e) { fail(e); }
  },
  async watchPrewarm() {
    const d = await api('vgpu/cache').catch(() => null);
    if (d) { this.loadCache(); if (d.prewarm && d.prewarm.state === 'running') setTimeout(() => this.watchPrewarm(), 1500); }
  },
  async stage(sp, dir) {
    const ok = await confirmBox('Copy the model to the fast drive?', `This writes ${fmtGB(sp.size_gb)} to ${dir || 'the cache folder'}. The original is not touched, and the copy is removed automatically if the cache needs room.`);
    if (!ok) return;
    try { await api('vgpu/cache/stage', { model_path: this.refs.model.value || S.values.model_path, confirm: true }); toast('Copying in the background — progress shows on the Optimizer tab’s job card', 'ok'); } catch (e) { fail(e); }
  },
};

const VGpu = {
  tabs: [['memory', '🧱 Memory', () => VMemory], ['optimizer', '⚡ Optimizer', () => window.VOptimizer], ['diag', '🩺 Diagnostics', () => window.VDiag], ['gptoss', '🦆 GPT-OSS 120B', () => window.VGptOss]],
  cur: 'memory', built: {}, panes: {}, bar: null,
  init() {
    const el = $('#page-vgpu');
    this.bar = h('div', { class: 'subtabs' }, this.tabs.map(([k, label]) => h('button', { 'data-tab': k, onclick: () => this.open(k) }, label)));
    el.append(this.bar);
    this.tabs.forEach(([k]) => { this.panes[k] = h('div', { class: 'subpane', 'data-pane': k }); el.append(this.panes[k]); });
    try { this.cur = localStorage.getItem('duck.vgpuTab') || 'memory'; } catch (e) { /* ignore */ }
  },
  show() { this.open(this.cur); },
  hide() { const m = this.tabs.find((t) => t[0] === this.cur)?.[2](); if (m && m.hide) m.hide(); },
  open(k) {
    if (!this.panes[k]) k = 'memory';
    const prev = this.tabs.find((t) => t[0] === this.cur)?.[2]();
    if (prev && prev.hide && this.cur !== k) prev.hide();
    this.cur = k;
    try { localStorage.setItem('duck.vgpuTab', k); } catch (e) { /* ignore */ }
    this.bar.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.tab === k));
    Object.entries(this.panes).forEach(([n, p]) => p.classList.toggle('active', n === k));
    const mod = this.tabs.find((t) => t[0] === k)[2]();
    if (!mod) { clear(this.panes[k]).append(h('div', { class: 'note bad' }, 'This section failed to load (script missing).')); return; }
    if (!this.built[k]) { this.built[k] = true; mod.init(this.panes[k]); }
    mod.show();
  },
};
