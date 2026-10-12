'use strict';
/* Updates card (Help page): version, channel, check / download / install. Uses /api/update/*. */
const Updates = {
  refs: {},
  timer: null,
  init() {
    const r = this.refs;
    r.body = h('div', { class: 'upd-body' }, 'Loading…');
    r.card = h('div', { class: 'card' }, h('h2', {}, '⬆️ Updates'), r.body);
    $('#page-about').prepend(r.card);
    this.refresh();
  },
  async refresh() {
    try { this.render(await api('update/status')); } catch (e) { clear(this.refs.body).append(h('div', { class: 'muted' }, e.message)); }
  },
  async act(path, body) {
    try { this.render(await api('update/' + path, body || {})); } catch (e) { fail(e); this.refresh(); }
  },
  render(s) {
    const body = clear(this.refs.body);
    const cur = s.current, res = s.result, dl = s.download;
    const build = cur.build_sha ? ' · build ' + cur.build_sha.slice(0, 7) : ' · source checkout';
    body.append(h('div', {}, 'You have ', h('b', {}, 'v' + cur.version), build, cur.installer ? ' · installed' : cur.frozen ? ' · portable' : ''));
    body.append(h('div', { class: 'muted' }, 'Channel: ' + s.channel + (s.channel === 'off' ? ' (change it in Settings → App & updates)' : '') +
      (s.last_checked ? ' · last checked ' + new Date(s.last_checked * 1000).toLocaleString() : '')));
    if (s.last_error) body.append(h('div', { class: 'banner' }, 'Last check failed: ' + s.last_error));
    if (res && res.available) {
      body.append(h('div', { class: 'banner ok' }, '✅ ', h('b', {}, 'Update available: v' + (res.version || '?')),
        res.sha ? ' (build ' + res.sha.slice(0, 7) + ')' : ''));
      if (res.notes) body.append(h('pre', { class: 'upd-notes' }, res.notes));
    } else if (res) {
      body.append(h('div', { class: 'muted' }, res.reason === 'off' ? '' : 'You are up to date.'));
    }
    const acts = h('div', { class: 'row' });
    acts.append(h('button', { onclick: () => this.act('check') }, 'Check now'));
    if (res && res.available) {
      if (dl.state === 'downloading') {
        const pct = dl.total ? Math.round(100 * dl.done / dl.total) : 0;
        acts.append(h('span', {}, 'Downloading… ' + pct + '% (' + fmtBytes(dl.done) + ' / ' + fmtBytes(dl.total) + ')'),
          h('button', { onclick: () => this.act('cancel') }, 'Cancel'));
        clearTimeout(this.timer); this.timer = setTimeout(() => this.refresh(), 1000);
      } else if (s.downloaded) {
        acts.append(h('button', { class: 'primary', disabled: !s.can_apply, onclick: async () => {
          if (await confirmBox('Install update?', 'Duckstoken will close, install v' + res.version + ' (checked against the published SHA-256) and start again.'))
            this.act('apply', { confirm: true, restart: true });
        } }, 'Install & restart'));
        if (!s.can_apply) acts.append(h('span', { class: 'muted' }, 'Automatic install only works in the packaged Windows app.'));
      } else {
        acts.append(h('button', { class: 'primary', onclick: () => this.act('download') }, 'Download update'));
      }
      if (dl.error) acts.append(h('span', { class: 'muted' }, dl.error));
    }
    body.append(acts);
  },
};
