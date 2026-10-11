'use strict';
/* Learning → Big model: the queue of questions the slow, smart model answers in the background.
   Server data is only ever shown as text nodes. */
window.LearnQueue = (() => {
  const POLL_MS = 3000, LIST_LIMIT = 100, MAX_REVIEW = 200, DEFAULT_REVIEW = 10;
  const MODES = {
    manual: { label: 'Only when I press “Run queue now”', text: () => 'Jobs wait in the queue until you press “Run queue now”. Nothing runs by itself.' },
    immediate: { label: 'As soon as a job is queued', text: () => 'The big model starts as soon as a job is queued. It uses most of your memory and processor while it works, so Chat can feel slow or pause meanwhile.' },
    overnight: { label: 'Only during the night window', text: (s) => `Jobs wait for the night window (${s.overnight && s.overnight.start || '?'} to ${s.overnight && s.overnight.end || '?'}) and then run one after another. Leave the PC switched on overnight.` },
  };
  const PHASES = { idle: ['idle', ''], waiting: ['waiting', 'ok'], swapping: ['swapping models', 'loading'], loading_big: ['loading the big model', 'loading'], answering: ['answering', 'good'] };
  const STATUS = { queued: ['waiting', 'ok'], running: ['running', 'loading'], done: ['done', 'good'], error: ['failed', 'error'], cancelled: ['cancelled', ''] };
  const FILTERS = [['', 'All', null], ['queued', 'Waiting', 'queued'], ['running', 'Running', 'running'], ['done', 'Done', 'done'], ['error', 'Failed', 'error'], ['cancelled', 'Cancelled', 'cancelled']];
  const REASONS = { manual: 'you asked', downvote: 'you gave 👎', hard: 'looked hard', review: 'review' };
  const NO_ANSWER = { queued: 'Waiting in the queue…', running: 'The big model is working on this right now…', error: 'No answer: the job failed.', cancelled: 'No answer: the job was cancelled.', done: '(The big model sent back an empty answer.)' };

  const st = { status: null, statusError: '', statusSeq: 0, renderKey: '', jobs: [], jobsError: '', jobsLoaded: false, jobsSeq: 0, filter: '', sig: '', timer: null };
  const ui = {};
  const open = new Set();
  const details = new Map();
  const busy = new Set();

  const enc = encodeURIComponent;
  const num = (v) => (Number.isFinite(Number(v)) ? Number(v) : 0);
  const str = (v) => (typeof v === 'string' ? v : '');
  const when = (ts) => (num(ts) > 0 ? new Date(num(ts) * 1000).toLocaleString() : '');
  const flat = (s) => str(s).replace(/\s+/g, ' ').trim();
  const clip = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);
  const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const baseName = (p) => str(p).split(/[\\/]/).filter(Boolean).pop() || '';
  const cssWord = (s) => (/^[a-z_]{1,20}$/.test(s) ? s : '');
  /* Replaces the content of `el`; unlike a bare append(), skips null and false children. */
  const fill = (el, ...kids) => clear(el).append(...kids.flat(Infinity).filter((k) => k != null && k !== false));

  function counts() {
    const c = (st.status && st.status.counts) || {};
    return { queued: num(c.queued), running: num(c.running), done: num(c.done), error: num(c.error), cancelled: num(c.cancelled) };
  }

  function contentText(c) {
    if (typeof c === 'string') return c;
    return Array.isArray(c) ? c.map((p) => (p && typeof p.text === 'string' ? p.text : '')).filter(Boolean).join('\n') : '';
  }

  function lastUserText(messages) {
    for (let i = messages.length - 1; i >= 0; i--) if (messages[i].role === 'user') return contentText(messages[i].content);
    return '';
  }

  /* --------------------------------------------------------------- polling */
  async function refreshStatus() {
    const seq = ++st.statusSeq;
    let data = null, error = '';
    try { data = await api('queue/status'); } catch (e) { error = e.message || 'unknown error'; }
    if (seq !== st.statusSeq) return;
    if (data && typeof data === 'object') { st.status = data; st.statusError = ''; } else st.statusError = error || 'The server sent an unexpected reply';
    renderStatus();
    const sig = st.statusError ? 'error' : JSON.stringify([st.status.counts, st.status.current_job]);
    if (sig !== st.sig) { st.sig = sig; loadJobs(); }
  }

  /* After an action: re-read everything, including the job list. */
  function refreshAll() {
    st.sig = '';
    return refreshStatus();
  }

  function tick() {
    if (!ui.root.offsetParent) stop(); // the tab was left without hide() being called
    else if (!document.hidden) refreshStatus();
  }

  function start() { stop(); st.timer = setInterval(tick, POLL_MS); }
  function stop() { clearInterval(st.timer); st.timer = null; }

  /* ---------------------------------------------------------------- status */
  function phaseText(s, c) {
    switch (s.phase) {
      case 'answering': return s.current_job ? `Answering job #${num(s.current_job)}.` : 'Answering a question.';
      case 'loading_big': return 'Loading the big model. This can take several minutes.';
      case 'swapping': return 'Stopping your everyday model to free up memory.';
      case 'waiting': return `${plural(c.queued, 'job is', 'jobs are')} waiting. ${waitHint(s)}`;
      default: return c.queued ? `${plural(c.queued, 'job', 'jobs')} queued.` : 'Nothing to do. Jobs show up here when they are queued.';
    }
  }

  function waitHint(s) {
    if (s.run_mode === 'manual') return 'Press “Run queue now” to start.';
    if (s.run_mode === 'overnight') return `They start in the night window (${s.overnight && s.overnight.start || '?'} to ${s.overnight && s.overnight.end || '?'}).`;
    return 'They start shortly.';
  }

  function renderStatus() {
    const key = JSON.stringify([st.status, st.statusError]);
    renderChips();
    renderControls();
    if (key === st.renderKey) return; // keep focus and selections while nothing changed
    st.renderKey = key;
    const s = st.status, error = st.statusError ? h('div', { class: 'note bad' }, 'Could not read the big-model status: ' + st.statusError + ' ', h('button', { onclick: refreshAll }, 'Try again')) : null;
    clear(ui.mode);
    if (!s) {
      fill(ui.status, error || h('div', { class: 'muted' }, 'Loading…'));
      return;
    }
    const c = counts(), phase = PHASES[s.phase] || [flat(s.phase) || 'unknown', ''], model = baseName(s.big_model_path), night = s.overnight || {};
    const bigState = str(s.big_state) || 'stopped';
    fill(ui.status,
      error,
      h('div', { class: 'lq-phase' }, h('span', { class: 'badge ' + phase[1] }, phase[0]), h('span', {}, phaseText(s, c))),
      model ? null : h('div', { class: 'note warn' }, 'No big model chosen yet. ', h('button', { onclick: () => go('settings') }, 'Open Settings to pick one')),
      s.small_was_stopped ? h('div', { class: 'note' }, 'Your everyday model is paused while the big model works. It restarts by itself when the queue is done.') : null,
      s.last_error ? h('div', { class: 'note bad lq-wrap' }, 'Last problem: ' + str(s.last_error)) : null,
      h('div', { class: 'kv lq-kv' },
        h('div', {}, 'Big model'), h('div', { class: 'lq-wrap', title: str(s.big_model_path) }, model || 'not chosen'),
        h('div', {}, 'Its server'), h('div', {}, h('span', { class: 'badge ' + cssWord(bigState) }, bigState)),
        h('div', {}, 'Jobs'), h('div', {}, `${c.queued} waiting · ${c.running} running · ${c.done} done · ${c.error} failed · ${c.cancelled} cancelled`),
        night.enabled ? [h('div', {}, 'Night window'), h('div', {}, `${str(night.start)} to ${str(night.end)} `, h('span', { class: 'badge ' + (s.window_active ? 'good' : '') }, s.window_active ? 'open now' : 'closed'))] : null,
        num(night.review_count) > 0 ? [h('div', {}, 'Night review'), h('div', {}, `${plural(num(night.review_count), 'recent answer', 'recent answers')} are sent to the big model each night`)] : null));
    renderModes(ui.mode, s);
  }

  function renderModes(box, s) {
    const known = MODES[s.run_mode];
    const others = Object.keys(MODES).filter((m) => m !== s.run_mode);
    fill(box,
      known ? h('p', {}, h('b', {}, 'Currently: ' + known.label + '. '), known.text(s)) : null,
      h('div', { class: 'small' }, known ? 'Change to: ' : 'Set to: ',
        others.map((m, i) => [i ? ' · ' : '', h('button', { class: 'lb-link', onclick: () => setMode(m) }, MODES[m].label)])));
  }

  async function setMode(mode) {
    if (!(await setSetting('big_run_mode', mode))) return;
    toast('Saved', 'ok');
    await refreshAll();
  }

  function renderControls() {
    const s = st.status, c = counts(), hasModel = !!(s && s.big_model_path), finished = c.done + c.error + c.cancelled;
    const running = !!(s && s.run_now);
    ui.runBtn.disabled = busy.has(ui.runBtn) || !hasModel || c.queued === 0 || running;
    ui.runBtn.textContent = running ? 'Running the queue…' : '▶ Run queue now';
    ui.runBtn.title = !hasModel ? 'Pick a big model in Settings first' : c.queued === 0 ? 'Nothing is waiting in the queue' : 'Answer every waiting job now, whatever the run mode says';
    ui.reviewBtn.disabled = busy.has(ui.reviewBtn) || !hasModel;
    ui.reviewBtn.title = hasModel ? '' : 'Pick a big model in Settings first';
    ui.clearBtn.disabled = busy.has(ui.clearBtn) || finished === 0;
  }

  function renderChips() {
    const c = counts();
    ui.chips.forEach(({ id, label, count, btn }) => {
      btn.textContent = count && st.status ? `${label} (${c[count]})` : label;
      btn.setAttribute('aria-pressed', String(st.filter === id));
    });
  }

  /* --------------------------------------------------------------- actions */
  /* Runs an async action with the button disabled; failures become toasts. */
  async function act(btn, fn) {
    busy.add(btn);
    renderControls();
    try { await fn(); } catch (e) { fail(e); }
    busy.delete(btn);
    renderControls();
  }

  const runNow = () => act(ui.runBtn, async () => {
    await api('queue/run_now', {});
    toast('Started. The big model needs a moment to load, and its first answer can take a few minutes.', 'ok');
    await refreshAll();
  });

  function review() {
    const n = Math.max(1, Math.min(MAX_REVIEW, Math.floor(Number(ui.reviewN.value)) || DEFAULT_REVIEW));
    ui.reviewN.value = n;
    return act(ui.reviewBtn, async () => {
      const queued = num((await api('queue/review', { n })).queued);
      if (!queued) toast('Nothing new to queue: your latest answers already have a big-model answer, are already waiting, or the queue is full.');
      else toast(`${plural(queued, 'answer', 'answers')} queued for the big model.` + (st.status && st.status.run_mode === 'manual' ? ' Press “Run queue now” to start.' : ''), 'ok');
      await refreshAll();
    });
  }

  async function clearFinished() {
    const text = 'This clears the list of done, failed and cancelled jobs. A big-model answer that belongs to a logged answer stays saved with it (see Feedback). Answers to questions that have no logged answer are gone for good.';
    if (!(await confirmBox('Remove finished jobs?', text))) return;
    await act(ui.clearBtn, async () => {
      toast(`Removed ${plural(num((await api('queue/clear', {})).removed), 'job', 'jobs')}`, 'ok');
      await refreshAll();
    });
  }

  async function cancel(job) {
    try {
      const d = await api('queue/cancel', { id: job.id });
      toast(d.ok ? 'Cancelled' : 'Too late: that job has already started or finished.', d.ok ? 'ok' : '');
    } catch (e) { fail(e); }
    refreshAll();
  }

  /* ------------------------------------------------------------------ jobs */
  async function loadJobs() {
    const seq = ++st.jobsSeq;
    try {
      const d = await api('queue/list?limit=' + LIST_LIMIT + (st.filter ? '&status=' + enc(st.filter) : ''));
      if (seq !== st.jobsSeq) return;
      st.jobs = (Array.isArray(d.items) ? d.items : []).filter((j) => j && Number.isInteger(j.id));
      st.jobsError = '';
      const ids = new Set(st.jobs.map((j) => j.id));
      [...details.keys()].forEach((id) => { if (!ids.has(id)) details.delete(id); });
    } catch (e) {
      if (seq !== st.jobsSeq) return;
      st.jobsError = e.message || 'unknown error';
    }
    st.jobsLoaded = true;
    renderJobs();
  }

  function setFilter(id) {
    st.filter = id;
    renderChips();
    loadJobs();
  }

  const focusKey = () => (document.activeElement && ui.rows.contains(document.activeElement) ? document.activeElement.dataset.focus || '' : '');

  /* Puts the focus back where it was after the rows were rebuilt (a vanished Cancel button falls back to its row's toggle). */
  function restoreFocus(key) {
    const find = (k) => [...ui.rows.querySelectorAll('[data-focus]')].find((x) => x.dataset.focus === k);
    const el = key && (find(key) || find('t' + key.slice(1)));
    if (el) el.focus();
  }

  function renderJobs() {
    const keep = focusKey(), box = clear(ui.state);
    clear(ui.rows);
    ui.table.hidden = true;
    if (st.jobsError) {
      box.append(h('div', { class: 'note bad' }, 'Could not load the jobs: ' + st.jobsError + ' ', h('button', { onclick: loadJobs }, 'Try again')));
      return;
    }
    if (!st.jobsLoaded) { box.append(h('div', { class: 'muted' }, 'Loading…')); return; }
    if (!st.jobs.length) {
      box.append(st.filter
        ? h('div', { class: 'lb-empty' }, 'No jobs with this status.')
        : h('div', { class: 'lb-empty' }, h('b', {}, 'No jobs yet.'), h('div', {}, 'A job appears when you queue a review above, when a 👎 or a hard question is sent to the big model automatically (see Settings → Big model), or when you ask for a second opinion in Chat.')));
      return;
    }
    ui.table.hidden = false;
    st.jobs.forEach((j) => ui.rows.append(...jobRows(j)));
    if (st.jobs.length >= LIST_LIMIT) box.append(h('div', { class: 'muted small' }, `Showing the newest ${LIST_LIMIT} jobs.`));
    restoreFocus(keep);
  }

  function durationText(j) {
    const started = num(j.started), finished = num(j.finished);
    if (j.status === 'running' && started > 0) return 'running since ' + new Date(started * 1000).toLocaleTimeString();
    return started > 0 && finished >= started ? 'took ' + fmtDur(finished - started) : '';
  }

  function jobRows(j) {
    const isOpen = open.has(j.id), badge = STATUS[j.status] || [flat(j.status) || '?', ''], took = durationText(j);
    const row = h('tr', { class: 'lq-row' + (isOpen ? ' open' : '') },
      h('td', {}, h('span', { class: 'badge ' + badge[1] }, badge[0])),
      h('td', { class: 'lq-q' }, h('button', { class: 'ghost lq-toggle', 'data-focus': 't' + j.id, 'aria-expanded': String(isOpen), onclick: () => toggle(j) },
        h('span', { class: 'lq-chev', 'aria-hidden': 'true' }, isOpen ? '▾' : '▸'),
        h('span', { class: 'lq-prompt' }, clip(flat(j.prompt), 200) || '(no question text)'))),
      h('td', { class: 'lq-sm small muted' }, REASONS[j.reason] || flat(j.reason)),
      h('td', { class: 'small' }, h('div', {}, when(j.created)), took ? h('div', { class: 'muted' }, took) : null),
      h('td', { class: 'lq-act' }, j.status === 'queued' ? h('button', { 'data-focus': 'c' + j.id, onclick: () => cancel(j) }, 'Cancel') : null));
    if (!isOpen) return [row];
    const slot = h('div', { class: 'lq-detail' });
    fillDetail(slot, j);
    return [row, h('tr', { class: 'lq-detailrow' }, h('td', { colspan: 5 }, slot))];
  }

  function toggle(j) {
    if (open.has(j.id)) open.delete(j.id); else open.add(j.id);
    renderJobs();
  }

  async function fillDetail(slot, job) {
    const cached = details.get(job.id);
    if (cached && cached.status === job.status) { renderDetail(slot, job, cached.item); return; }
    if (cached) renderDetail(slot, job, cached.item); else clear(slot).append(h('div', { class: 'muted' }, 'Loading…'));
    try {
      const d = await api('queue/get?id=' + job.id);
      if (!d || !d.item || typeof d.item !== 'object') throw new Error('The server sent an unexpected reply');
      details.set(job.id, { status: job.status, item: d.item });
      if (slot.isConnected) renderDetail(slot, job, d.item);
    } catch (e) {
      if (slot.isConnected) clear(slot).append(h('div', { class: 'note bad' }, 'Could not load this job: ' + e.message + ' ', h('button', { onclick: () => fillDetail(slot, job) }, 'Try again')));
    }
  }

  function renderDetail(slot, job, item) {
    const messages = (Array.isArray(item.messages) ? item.messages : []).filter((m) => m && typeof m === 'object');
    const earlier = Math.max(0, messages.length - 1), big = str(item.answer), small = str(item.small_answer);
    const column = (title, text, missing, cls) => h('div', { class: 'lq-col ' + cls },
      h('div', { class: 'lq-label' }, title),
      h('div', { class: 'lq-text' + (missing ? ' muted' : ''), tabindex: 0, role: 'region', 'aria-label': title }, text));
    const meta = [`job #${job.id}`, when(job.created), REASONS[job.reason] ? 'reason: ' + REASONS[job.reason] : '', num(item.tokens) ? `${num(item.tokens)} tokens` : '',
      num(item.tps) ? `${num(item.tps).toFixed(2)} tokens/s` : '', item.exchange_id ? `belongs to logged answer #${num(item.exchange_id)}` : 'no logged answer attached'].filter(Boolean);
    fill(slot,
      item.error ? h('div', { class: 'note bad lq-wrap' }, 'The big model could not answer: ' + str(item.error)) : null,
      h('div', { class: 'lq-label' }, 'Question' + (earlier ? ` (${plural(earlier, 'earlier message is', 'earlier messages are')} not shown)` : '')),
      h('div', { class: 'lq-text lq-question', tabindex: 0, role: 'region', 'aria-label': 'Question' }, lastUserText(messages) || str(job.prompt) || '(no question text)'),
      h('div', { class: 'lq-compare' },
        column('Everyday model', small || '(No original answer was stored for this question.)', !small, ''),
        column('Big model' + (item.model ? ' · ' + str(item.model) : ''), big || NO_ANSWER[job.status] || '', !big, 'big')),
      h('div', { class: 'muted small' }, meta.join(' · ')));
  }

  /* ----------------------------------------------------------------- shell */
  function init(el) {
    ui.root = el;
    ui.status = h('div');
    ui.mode = h('div');
    ui.runBtn = h('button', { class: 'primary', onclick: runNow }, '▶ Run queue now');
    ui.clearBtn = h('button', { onclick: clearFinished, title: 'Remove done, failed and cancelled jobs from the list' }, 'Clear finished jobs');
    ui.reviewN = h('input', { type: 'number', min: 1, max: MAX_REVIEW, value: DEFAULT_REVIEW, class: 'lq-n', 'aria-label': 'How many recent answers to review',
      onkeydown: (e) => { if (e.key === 'Enter') review(); } });
    ui.reviewBtn = h('button', { onclick: review }, 'Queue review');
    ui.chips = FILTERS.map(([id, label, count]) => ({ id, label, count, btn: h('button', { class: 'lb-chip', onclick: () => setFilter(id) }, label) }));
    ui.state = h('div', { 'aria-live': 'polite' });
    ui.rows = h('tbody');
    ui.table = h('div', { class: 'lq-scroll', hidden: true }, h('table', { class: 'lq-table' },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'Status'), h('th', { scope: 'col' }, 'Question'), h('th', { scope: 'col', class: 'lq-sm' }, 'Why'), h('th', { scope: 'col' }, 'When'), h('th', { scope: 'col' }, h('span', { class: 'lq-sr' }, 'Actions')))),
      ui.rows));
    el.append(
      h('div', { class: 'card' }, h('h2', {}, '🐘 The big model'),
        h('p', { class: 'muted' }, 'Some questions are too hard for your everyday model. Here a bigger, smarter model answers them in the background, even overnight. Its answers are kept next to the everyday model’s so you can compare them, and they can become training data. It is slow: on a modest PC expect about a token per second, so a long answer can take many minutes.'),
        ui.status,
        h('h3', {}, 'When should queued jobs run?'), ui.mode,
        h('h3', {}, 'Actions'),
        h('div', { class: 'row' }, ui.runBtn, ui.clearBtn),
        h('div', { class: 'row lq-review' }, h('span', {}, 'Ask the big model to review my last'), ui.reviewN, h('span', {}, 'answers'), ui.reviewBtn),
        h('div', { class: 'muted small' }, 'Review sends your newest answers that have no big-model answer yet. Each one takes the big model several minutes.')),
      h('div', { class: 'card' },
        h('div', { class: 'row' }, h('h2', { class: 'grow' }, 'Jobs'),
          h('button', { class: 'ghost', title: 'Reload', 'aria-label': 'Reload the job list', onclick: refreshAll }, '↻')),
        h('div', { class: 'row lq-chips', role: 'group', 'aria-label': 'Filter jobs by status' }, ui.chips.map((c) => c.btn)),
        ui.state, ui.table));
    renderChips();
    renderControls();
    renderJobs();
  }

  function show() {
    st.renderKey = '';
    refreshAll();
    start();
  }

  return { init, show, hide: stop };
})();
