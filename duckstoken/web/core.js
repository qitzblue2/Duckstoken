'use strict';
const TOKEN = document.querySelector('meta[name=duck-token]').content;
const S = { schema: [], groups: [], presets: {}, hfQuick: [], values: {}, profiles: [], hw: null, status: null,
            dirty: new Set(), runningSnapshot: null };
const $ = (sel, root = document) => root.querySelector(sel);

async function api(path, body, opts = {}) {
  const o = { method: body === undefined ? 'GET' : 'POST', headers: { 'X-Duck-Token': TOKEN }, ...opts };
  if (body !== undefined) { o.headers['Content-Type'] = 'application/json'; o.body = JSON.stringify(body); }
  const r = await fetch('/api/' + path, o);
  if (opts.raw) return r;
  let data = {};
  try { data = await r.json(); } catch (e) { /* not json */ }
  if (!r.ok) { const err = new Error(data.error || ('HTTP ' + r.status)); err.data = data; err.status = r.status; throw err; }
  return data;
}

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'value') e.value = v;
    else if (k === 'html') e.innerHTML = v;  // only ever fed escaped text (see renderMarkdown)
    else if (v === true) e.setAttribute(k, '');
    else e.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) { if (c == null || c === false) continue; e.append(c.nodeType ? c : document.createTextNode(String(c))); }
  return e;
}
/* The DOM's own append() prints "[object HTMLDivElement]" for an array and "null" for null. Everything in this UI is built
   with arrays and optional children, so make append() flatten and skip empty children the way h() does. */
(function () {
  const native = Element.prototype.append;
  Element.prototype.append = function (...kids) { return native.apply(this, kids.flat(Infinity).filter((c) => c != null && c !== false)); };
})();
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
const fmtGB = (x) => (x == null ? '?' : Number(x).toFixed(x >= 100 ? 0 : 1) + ' GB');
const fmtBytes = (b) => { b = Number(b) || 0; const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; while (b >= 1024 && i < 4) { b /= 1024; i++; } return b.toFixed(i > 1 ? 1 : 0) + ' ' + u[i]; };
const fmtDur = (s) => { s = Math.round(s); return s >= 3600 ? Math.floor(s / 3600) + 'h ' + Math.floor(s % 3600 / 60) + 'm' : s >= 60 ? Math.floor(s / 60) + 'm ' + (s % 60) + 's' : s + 's'; };

function toast(msg, kind = '') {
  const t = h('div', { class: 'toast ' + kind }, msg);
  $('#toasts').append(t);
  setTimeout(() => t.remove(), kind === 'err' ? 9000 : 4000);
}
const fail = (e) => toast(e.message || String(e), 'err');

function modal(title, body, buttons) {
  return new Promise((resolve) => {
    const root = $('#modal-root');
    const close = (v) => { bg.remove(); resolve(v); };
    const acts = h('div', { class: 'actions' }, (buttons || [{ label: 'OK', value: true, primary: true }]).map((b) =>
      h('button', { class: b.primary ? 'primary' : b.danger ? 'danger' : '', onclick: () => close(b.value) }, b.label)));
    const bg = h('div', { class: 'modal-bg', onclick: (e) => { if (e.target === bg) close(null); } },
      h('div', { class: 'modal', role: 'dialog' }, h('h2', {}, title), h('div', { class: 'scroll' }, body), acts));
    root.append(bg);
  });
}
const confirmBox = (title, text, danger = false) =>
  modal(title, h('div', { style: 'white-space:pre-wrap' }, text),
    [{ label: 'Cancel', value: false }, { label: 'Continue', value: true, primary: !danger, danger }]);

/* folder / file picker backed by /api/browse */
async function pickPath({ kind = 'dir', ext = '', start = '' } = {}) {
  let result = null, cur = start;
  const list = h('div'), pathIn = h('input', { type: 'text' });
  const body = h('div', {}, pathIn, h('div', { style: 'height:8px' }), list);
  async function go(p) {
    try {
      const d = await api('browse?path=' + encodeURIComponent(p) + (kind === 'file' ? '&files=1&ext=' + encodeURIComponent(ext) : ''));
      cur = d.path; pathIn.value = d.path; clear(list);
      if (d.drives.length) list.append(h('div', { class: 'row small' }, d.drives.map((x) => h('button', { onclick: () => go(x) }, x))));
      if (d.parent) list.append(h('div', { class: 'dir-item', onclick: () => go(d.parent) }, '⬑ ..'));
      d.dirs.forEach((n) => list.append(h('div', { class: 'dir-item', onclick: () => go(d.path.replace(/[\\/]$/, '') + (d.path.includes('\\') ? '\\' : '/') + n) }, '📁 ' + n)));
      d.files.forEach((f) => list.append(h('div', { class: 'dir-item', onclick: () => { result = d.path.replace(/[\\/]$/, '') + (d.path.includes('\\') ? '\\' : '/') + f.name; $('.modal-bg button.primary')?.click(); } }, '📄 ' + f.name + ' ', h('span', { class: 'muted small' }, fmtBytes(f.size)))));
    } catch (e) { fail(e); }
  }
  pathIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(pathIn.value); });
  go(start);
  const r = await modal(kind === 'dir' ? 'Choose a folder' : 'Choose a file', body,
    [{ label: 'Cancel', value: false }, { label: kind === 'dir' ? 'Use this folder' : 'Close', value: true, primary: true }]);
  if (!r) return null;
  return kind === 'dir' ? cur : result;
}

/* settings access */
const schemaOf = (key) => S.schema.find((s) => s.key === key);
async function setSetting(key, value, quiet = false) {
  try {
    const r = await api('settings/update', { values: { [key]: value } });
    S.values = r.values;
    const spec = schemaOf(key);
    if (S.status?.server?.running && !spec.live && JSON.stringify(S.runningSnapshot?.[key]) !== JSON.stringify(S.values[key])) S.dirty.add(key);
    else S.dirty.delete(key);
    if (key === 'theme') document.documentElement.dataset.theme = value;
    updateDirtyBanner();
    return true;
  } catch (e) { if (!quiet) fail(e); return false; }
}
function updateDirtyBanner() { window.onDirty && window.onDirty(); }

/* tiny, safe markdown: escape first, then format */
function escapeHtml(s) { return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
function renderMarkdown(src) {
  const parts = src.split(/```/);
  let out = '';
  parts.forEach((p, i) => {
    if (i % 2 === 1) {
      const nl = p.indexOf('\n');
      out += '<pre><code>' + escapeHtml(nl >= 0 ? p.slice(nl + 1) : p) + '</code></pre>';
    } else {
      let t = escapeHtml(p);
      t = t.replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
        .replace(/^### (.*)$/gm, '<h4>$1</h4>').replace(/^## (.*)$/gm, '<h3>$1</h3>').replace(/^# (.*)$/gm, '<h3>$1</h3>')
        .replace(/^\s*[-*] (.*)$/gm, '<li>$1</li>').replace(/(<li>.*<\/li>\n?)+/g, (m) => '<ul>' + m + '</ul>');
      out += t.split(/\n{2,}/).map((x) => (x.startsWith('<h') || x.startsWith('<ul') ? x : '<p>' + x.replace(/\n/g, '<br>') + '</p>')).join('');
    }
  });
  return out;
}
