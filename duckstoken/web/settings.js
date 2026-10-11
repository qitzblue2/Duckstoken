'use strict';
function controlFor(spec, onChange) {
  const v = S.values[spec.key], t = spec.type;
  const commit = (val) => onChange(val);
  if (t === 'bool') {
    const cb = h('input', { type: 'checkbox', checked: !!v, onchange: async (e) => {
      if (spec.danger && e.target.checked && !(await confirmBox('Are you sure?', `“${spec.label}” gives the AI more power over your PC.\n\n${spec.help || ''}\n\nOnly enable this for models and prompts you trust.`, true))) { e.target.checked = false; return; }
      commit(e.target.checked); } });
    return h('label', { class: 'switch' }, cb, h('span'));
  }
  if (t === 'select') return h('select', { onchange: (e) => commit(e.target.value) }, spec.options.map((o) => h('option', { value: o, selected: String(v) === o }, o)));
  if (t === 'int' || t === 'float') {
    return h('input', { type: 'number', value: v, step: t === 'int' ? 1 : 0.05, min: spec.min, max: spec.max, onchange: (e) => commit(e.target.value === '' ? spec.default : Number(e.target.value)) });
  }
  if (t === 'list') return h('textarea', { rows: Math.min(8, Math.max(3, (v || []).length + 1)), onchange: (e) => commit(e.target.value) }, (v || []).join('\n'));
  if (t === 'text') return h('textarea', { rows: 3, onchange: (e) => commit(e.target.value) }, v || '');
  const inp = h('input', { type: t === 'password' ? 'password' : 'text', value: v || '', onchange: (e) => commit(e.target.value) });
  if (t === 'path') {
    const b = h('button', { onclick: async () => { const p = await pickPath({ kind: spec.kind === 'dir' ? 'dir' : 'file', ext: spec.key === 'model_path' ? '.gguf' : spec.key === 'llama_server_path' ? '.exe' : '', start: inp.value }); if (p) { inp.value = p; commit(p); } } }, 'Browse…');
    return h('div', { class: 'ctl', style: 'width:100%' }, inp, b);
  }
  return inp;
}

function settingRow(spec, rerender) {
  const isMod = () => JSON.stringify(S.values[spec.key]) !== JSON.stringify(spec.default);
  const row = h('div', { class: 'setting' + (spec.danger ? ' danger' : '') + (isMod() ? ' mod' : ''), 'data-key': spec.key });
  const ctl = controlFor(spec, async (val) => { if (await setSetting(spec.key, val)) { row.classList.toggle('mod', isMod()); if (spec.rerender) rerender(); } });
  row.append(
    h('div', {}, h('div', { class: 'lbl' }, spec.label, spec.live ? h('span', { class: 'tag' }, 'live') : null, spec.advanced ? h('span', { class: 'tag' }, 'advanced') : null,
      spec.flag ? h('span', { class: 'tag mono' }, spec.flag) : null), spec.help ? h('div', { class: 'help' }, spec.help) : null),
    h('div', { class: 'ctl' }, ctl, h('button', { class: 'ghost', title: 'Reset to default', onclick: async () => { await api('settings/reset', { keys: [spec.key] }).then((r) => { S.values = r.values; }); rerender(); } }, '↺')));
  return row;
}

const Settings = {
  refs: {}, advanced: false, filter: '',
  init() {
    const el = $('#page-settings'), r = this.refs;
    r.banner = h('div');
    r.search = h('input', { type: 'text', placeholder: '🔍 Search all settings…', oninput: (e) => { this.filter = e.target.value.toLowerCase(); this.render(); } });
    r.preset = h('select', {}, Object.keys(S.presets).map((n) => h('option', { value: n }, n)));
    r.profile = h('select', {});
    r.groups = h('div');
    r.adv = h('input', { type: 'checkbox', onchange: (e) => { this.advanced = e.target.checked; this.render(); } });
    el.append(r.banner, h('div', { class: 'search' }, r.search),
      h('div', { class: 'card' },
        h('div', { class: 'row' }, h('b', {}, 'Preset'), h('div', { class: 'grow' }, r.preset),
          h('button', { onclick: () => this.applyPreset() }, 'Apply'),
          h('button', { class: 'primary', onclick: () => Dashboard.autotune() }, '⚡ Auto-tune for my PC')),
        h('div', { class: 'muted small', id: 'preset-desc' }),
        h('div', { class: 'row', style: 'margin-top:10px' }, h('b', {}, 'Profile'), h('div', { class: 'grow' }, r.profile),
          h('button', { onclick: () => this.loadProfile() }, 'Load'), h('button', { onclick: () => this.saveProfile() }, 'Save current as…'),
          h('button', { class: 'ghost', onclick: () => this.delProfile() }, 'Delete')),
        h('div', { class: 'row', style: 'margin-top:10px' }, h('label', { class: 'row' }, r.adv, 'Show advanced settings'),
          h('span', { style: 'margin-left:auto' }),
          h('button', { onclick: () => this.exportAll() }, 'Export JSON'), h('button', { onclick: () => this.importAll() }, 'Import JSON'),
          h('button', { class: 'danger', onclick: () => this.resetAll() }, 'Reset everything'))),
      r.groups);
    r.preset.addEventListener('change', () => { $('#preset-desc').textContent = S.presets[r.preset.value] || ''; });
    $('#preset-desc').textContent = S.presets[r.preset.value] || '';
  },
  show() { this.refresh(); },
  async refresh() { try { const d = await api('settings'); S.values = d.values; S.profiles = d.profiles; } catch (e) { fail(e); } this.render(); },
  render() {
    const r = this.refs, open = new Set([...r.groups.querySelectorAll('details[open]')].map((d) => d.dataset.g));
    clear(r.profile); S.profiles.forEach((n) => r.profile.append(h('option', { value: n }, n)));
    if (!S.profiles.length) r.profile.append(h('option', { value: '' }, '— none saved —'));
    clear(r.groups);
    S.groups.filter(([g]) => g !== 'pc').forEach(([g, title]) => {
      const specs = S.schema.filter((s) => s.group === g && (this.advanced || !s.advanced || this.filter) &&
        (!this.filter || (s.label + ' ' + s.key + ' ' + (s.flag || '') + ' ' + (s.help || '')).toLowerCase().includes(this.filter)));
      if (!specs.length) return;
      const d = h('details', { class: 'group', 'data-g': g, open: this.filter || open.has(g) || (!open.size && g === 'memory') }, h('summary', {}, title + ` (${specs.length})`));
      specs.forEach((s) => d.append(settingRow(s, () => this.render())));
      r.groups.append(d);
    });
  },
  async applyPreset() { try { const r = await api('presets/apply', { name: this.refs.preset.value }); S.values = r.values; this.render(); toast('Preset applied', 'ok'); this.markDirty(); } catch (e) { fail(e); } },
  markDirty() { if (S.status?.server?.running) { S.schema.forEach((s) => { if (!s.live && JSON.stringify(S.runningSnapshot?.[s.key]) !== JSON.stringify(S.values[s.key])) S.dirty.add(s.key); }); updateDirtyBanner(); } },
  async saveProfile() {
    const box = h('input', { type: 'text', placeholder: 'e.g. Fast chat / Max context / Coding' });
    const ok = await modal('Save profile', box, [{ label: 'Cancel', value: false }, { label: 'Save', value: true, primary: true }]);
    if (ok && box.value.trim()) { try { const r = await api('profiles/save', { name: box.value.trim() }); S.profiles = r.profiles; this.render(); toast('Saved', 'ok'); } catch (e) { fail(e); } }
  },
  async loadProfile() { const n = this.refs.profile.value; if (!n) return; try { const r = await api('profiles/load', { name: n }); S.values = r.values; this.render(); toast('Loaded ' + n, 'ok'); this.markDirty(); } catch (e) { fail(e); } },
  async delProfile() { const n = this.refs.profile.value; if (!n || !await confirmBox('Delete profile?', n, true)) return; const r = await api('profiles/delete', { name: n }); S.profiles = r.profiles; this.render(); },
  exportAll() { const a = h('a', { href: URL.createObjectURL(new Blob([JSON.stringify(S.values, null, 2)], { type: 'application/json' })), download: 'duckstoken-settings.json' }); a.click(); },
  async importAll() {
    const f = h('input', { type: 'file', accept: '.json' });
    f.onchange = async () => { try { const vals = JSON.parse(await f.files[0].text()); const r = await api('settings/import', { values: vals }); S.values = r.values; this.render(); toast('Imported', 'ok'); } catch (e) { fail(e); } };
    f.click();
  },
  async resetAll() { if (!await confirmBox('Reset all settings?', 'Every setting (including paths) goes back to its default.', true)) return; const r = await api('settings/reset', {}); S.values = r.values; this.render(); },
};

/* PC access page: same rows, filtered to the 'pc' group + audit log */
const PCAccess = {
  refs: {},
  init() {
    const el = $('#page-pc'); this.refs.rows = h('div'); this.refs.audit = h('div');
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '🛡 What the AI can do on your PC'),
        h('p', {}, 'Local models can be given tools to list, read and edit files and run commands, like an assistant that works inside your computer. Everything is OFF by default and every change is logged.'),
        h('div', { class: 'note warn' }, 'Text the model reads (a web page, a document) can try to trick it into using tools. Keep “Ask me before…” switched on for writes and commands, and avoid full PC access with models or prompts you do not trust. Commands are not sandboxed — they run with your Windows permissions.')),
      h('div', { class: 'card', style: 'padding:0' }, this.refs.rows),
      h('div', { class: 'card' }, h('h2', {}, '📜 Audit log', h('button', { style: 'margin-left:auto', onclick: () => this.loadAudit() }, 'Refresh')), this.refs.audit));
  },
  show() { this.render(); this.loadAudit(); },
  render() {
    clear(this.refs.rows);
    S.schema.filter((s) => s.group === 'pc').forEach((s) => this.refs.rows.append(settingRow(s, () => this.render())));
  },
  async loadAudit() {
    try {
      const d = await api('tools/audit'), box = clear(this.refs.audit);
      if (!d.entries.length) { box.append(h('span', { class: 'muted' }, 'Nothing yet.')); return; }
      box.append(h('table', {}, h('thead', {}, h('tr', {}, ['Time', 'Tool', 'Details', 'Result'].map((x) => h('th', {}, x)))),
        h('tbody', {}, d.entries.map((e) => h('tr', {}, h('td', { class: 'small' }, e.time), h('td', {}, e.tool),
          h('td', { class: 'mono' }, Object.values(e.args).join(' · ').slice(0, 140)), h('td', {}, h('span', { class: 'badge ' + (e.ok ? 'good' : 'error') }, e.ok ? 'ok' : 'failed')))))));
    } catch (e) { fail(e); }
  },
};
