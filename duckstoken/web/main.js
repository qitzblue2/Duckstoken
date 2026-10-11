'use strict';
const pages = { dashboard: Dashboard, chat: Chat, models: Models, settings: Settings, pc: PCAccess };
let current = 'dashboard';

function go(name) {
  current = name;
  document.querySelectorAll('#nav button').forEach((b) => b.classList.toggle('active', b.dataset.page === name));
  document.querySelectorAll('.page').forEach((p) => p.classList.toggle('active', p.id === 'page-' + name));
  pages[name]?.show?.();
}

window.onDirty = () => {
  document.querySelectorAll('.restart-banner').forEach((b) => b.remove());
  if (!S.dirty.size || !S.status?.server?.running) return;
  const mk = () => h('div', { class: 'banner restart-banner' }, `${S.dirty.size} changed setting(s) need a server restart to take effect.`,
    h('button', { onclick: () => Dashboard.control('restart') }, 'Restart now'));
  [Dashboard.refs.banner, Settings.refs.banner].forEach((t) => t && t.append(mk()));
};

function helpPage() {
  $('#page-about').append(
    h('div', { class: 'card' }, h('h2', {}, '🦆 Running gpt-oss-120b on a modest PC'),
      h('ol', {},
        h('li', {}, 'Models → pick the SSD as storage (the Dashboard marks each drive SSD / NVMe / HDD).'),
        h('li', {}, 'Models → Engine → “Find latest build” → Install (CPU or Vulkan for a GTX 750; CUDA 12 also works).'),
        h('li', {}, 'Models → download gpt-oss-120b (≈59 GB) — or gpt-oss-20b (≈11 GB) for a fast daily driver.'),
        h('li', {}, 'Dashboard → pick the model → ⚡ Auto-tune → ▶ Start. The first load of a huge model takes minutes.'),
        h('li', {}, 'Chat. Expect ~1 token/s for the 120B from an NVMe SSD on 48 GB RAM, several tokens/s for the 20B.'))),
    h('div', { class: 'card' }, h('h2', {}, 'Why is it slow / how do I speed it up?'),
      h('ul', {},
        h('li', {}, 'Token generation is limited by memory bandwidth. Enable XMP/DOCP in BIOS so RAM runs at its rated speed and make sure sticks are in dual-channel slots.'),
        h('li', {}, 'If the model is bigger than RAM, Windows must read the rest from disk while generating. SSD ≫ HDD (≈50×).'),
        h('li', {}, 'Lower the context size, close browsers/games, set reasoning effort to “low”.'),
        h('li', {}, 'A GTX 750 (1–2 GB VRAM) barely helps; the CPU does the work. Auto-tune handles this.'))),
    h('div', { class: 'card' }, h('h2', {}, 'Files'),
      h('div', { class: 'muted' }, 'Settings: %APPDATA%\\Duckstoken\\settings.json · Server log: %APPDATA%\\Duckstoken\\llama-server.log · Tool audit log: %APPDATA%\\Duckstoken\\tool_audit.jsonl')));
}

async function tick() {
  try {
    const st = await api('status'); S.status = st;
    Dashboard.update(st); Chat.update(st);
    const b = $('#top-state'); b.className = 'badge ' + st.server.state; b.textContent = 'model: ' + st.server.state;
    if (st.server.running && !S.runningSnapshot) S.runningSnapshot = { ...S.values };
    if (!st.server.running) S.runningSnapshot = null;
    if (current === 'dashboard') Dashboard.pullLogs();
  } catch (e) { /* server gone */ }
}

async function boot() {
  const sch = await api('schema');
  S.schema = sch.schema; S.groups = sch.groups; S.presets = sch.presets; S.hfQuick = sch.hf_quick;
  const d = await api('settings'); S.values = d.values; S.profiles = d.profiles;
  document.documentElement.dataset.theme = S.values.theme;
  Dashboard.init(); Chat.init(); Models.init(); Settings.init(); PCAccess.init(); helpPage();
  document.querySelectorAll('#nav button').forEach((b) => b.addEventListener('click', () => go(b.dataset.page)));
  $('#quit-btn').addEventListener('click', async () => {
    if (await confirmBox('Quit Duckstoken?', 'This also stops the model server.')) { await api('app/quit', {}).catch(() => {}); document.body.innerHTML = '<p style="padding:40px;font:16px sans-serif">Duckstoken has stopped. You can close this tab.</p>'; }
  });
  Settings.render(); PCAccess.render();
  go('dashboard'); tick();
  let n = 0;
  setInterval(() => { tick(); if (current === 'dashboard' && ++n % Math.max(1, Math.round((S.values.hardware_poll_seconds || 5) / 2)) === 0) Dashboard.loadHardware(); }, 2000);
}
boot().catch((e) => { document.body.prepend(h('pre', { style: 'color:#f55;padding:20px' }, 'Failed to start UI: ' + e.message)); });
