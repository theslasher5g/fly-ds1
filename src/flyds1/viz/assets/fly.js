/* flyds1 live panel.
 *
 * Reads layout.json once and state.json every refresh tick, then draws.  No
 * dependencies: a canvas, fetch, and one timer.  The deliberate design choice
 * is that the *server* sends numbers and the *browser* does all rasterising --
 * the previous viewer rendered PNGs inside the agent's 33 ms step budget, which
 * is both slower and less informative than this.
 *
 * History lives here rather than on the server: the page polls every frame
 * anyway, so it can accumulate its own sparklines for free.
 */
'use strict';

const REFRESH = window.FLY_REFRESH_MS || 100;
const HISTORY = Math.max(60, Math.round(20000 / REFRESH)); // ~20 s of samples

const COL = {
  ink: '#d8dde5', dim: '#8a93a2', edge: '#272d37', panel: '#1b1f26',
  warm: '#ffb347', cool: '#4fc3f7', good: '#7bd88f', bad: '#ff6b6b',
  grid: '#232932',
};

let layout = null;
let state = null;
let tick = 0;

// Running scale per named quantity.  Absolute rates depend on the network's
// gain, which changes between connectomes, so nothing here may assume a fixed
// range; each scale tracks a decaying peak so panels stay readable and still
// come back down after a spike instead of being pinned by one outlier.
const scales = {};
function scaleFor(name, value) {
  const v = Math.abs(value) || 0;
  const prev = scales[name] || 1e-6;
  const next = Math.max(v, prev * 0.995);
  scales[name] = next;
  return next;
}

const history = {};
function push(name, value) {
  const arr = history[name] || (history[name] = []);
  arr.push(Number.isFinite(value) ? value : 0);
  if (arr.length > HISTORY) arr.shift();
  return arr;
}

/* ---------------------------------------------------------------- helpers */

/** Size a canvas's backing store to its CSS box at device resolution.
 *
 * The HTML's width/height attributes are the *authored aspect ratio*, not a
 * pixel size -- CSS stretches each canvas to its grid column. Capture that
 * ratio once, before the first resize overwrites the attributes, then derive
 * the height from the measured width so nothing is drawn squashed. Drawing
 * happens in CSS pixels; the devicePixelRatio transform keeps it crisp on a
 * HiDPI display instead of blurry.
 */
const ratios = new WeakMap();
function fit(id) {
  const canvas = document.getElementById(id);
  if (!ratios.has(canvas)) ratios.set(canvas, canvas.height / canvas.width);
  const ratio = ratios.get(canvas);
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(canvas.getBoundingClientRect().width));
  const h = Math.max(1, Math.round(w * ratio));
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = h + 'px';
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function label(ctx, text, x, y, color, align, size) {
  ctx.fillStyle = color || COL.dim;
  ctx.textAlign = align || 'left';
  ctx.textBaseline = 'middle';
  ctx.font = (size || 10) + 'px ui-monospace, Menlo, Consolas, monospace';
  ctx.fillText(text, x, y);
}

/** Blue for negative, amber for positive, near-black for zero. */
function signedColor(v, scale) {
  const t = Math.max(-1, Math.min(1, v / (scale || 1)));
  const a = Math.abs(t);
  if (t >= 0) return `rgba(255, ${Math.round(179 - 60 * a)}, 71, ${0.15 + 0.85 * a})`;
  return `rgba(79, ${Math.round(195 - 40 * a)}, 247, ${0.15 + 0.85 * a})`;
}

function heat(t) {
  const a = Math.max(0, Math.min(1, t));
  return `rgba(${Math.round(60 + 195 * a)}, ${Math.round(80 + 99 * a)}, ${Math.round(120 - 49 * a)}, ${0.2 + 0.8 * a})`;
}

/* ------------------------------------------------------------------- eyes */

let eyeGeom = null;
function eyeGeometry() {
  if (eyeGeom || !layout || !layout.eyes) return eyeGeom;
  const sides = Object.keys(layout.eyes).sort();
  eyeGeom = sides.map((side) => {
    const pts = layout.eyes[side].coords;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const [x, y] of pts) {
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    return { side, pts, x0, x1, y0, y1 };
  });
  return eyeGeom;
}

function drawEyes() {
  const { ctx, w, h } = fit('eyes');
  const geom = eyeGeometry();
  if (!geom || !state.eyes) { label(ctx, 'no retinal data', 8, h / 2); return; }
  const channel = document.querySelector('input[name=eyech]:checked').value;

  const pad = 26;
  const boxW = (w - pad * 3) / geom.length;
  geom.forEach((eye, k) => {
    const values = (state.eyes[eye.side] || {})[channel];
    const ox = pad + k * (boxW + pad);
    const oy = 18;
    const boxH = h - oy - 14;
    ctx.strokeStyle = COL.edge;
    ctx.strokeRect(ox, oy, boxW, boxH);
    label(ctx, eye.side + ' eye' + (values ? '' : ' (n/a)'), ox, oy - 8, COL.dim);

    if (!values) return;
    const scale = scaleFor('eye_' + channel, Math.max(...values.map(Math.abs)));
    const spanX = (eye.x1 - eye.x0) || 1;
    const spanY = (eye.y1 - eye.y0) || 1;
    const span = Math.max(spanX, spanY);
    const size = Math.min(boxW, boxH) / span;
    const cx = ox + boxW / 2, cy = oy + boxH / 2;
    const r = Math.max(1.1, size * 0.52);
    for (let i = 0; i < eye.pts.length && i < values.length; i++) {
      const px = cx + (eye.pts[i][0] - (eye.x0 + eye.x1) / 2) * size;
      // screen y grows downward, elevation grows upward
      const py = cy - (eye.pts[i][1] - (eye.y0 + eye.y1) / 2) * size;
      ctx.fillStyle = signedColor(values[i], scale);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fill();
    }
    label(ctx, `${eye.pts.length} facets  scale +-${scale.toFixed(2)}`,
          ox + boxW, oy + boxH + 8, COL.dim, 'right');
  });
}

/* -------------------------------------------------------------- hemisphere */

function drawBalance() {
  const { ctx, w, h } = fit('balance');
  const rows = (state.stages || []).filter((s) => s.left || s.right || s.balance);
  const whole = state.balance || 0;

  // Big gauge: the headline "which half".
  const top = 34;
  const cx = w / 2;
  ctx.strokeStyle = COL.edge;
  ctx.beginPath(); ctx.moveTo(10, top); ctx.lineTo(w - 10, top); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, top - 10); ctx.lineTo(cx, top + 10); ctx.stroke();
  const x = cx + whole * (w / 2 - 12);
  ctx.fillStyle = whole >= 0 ? COL.warm : COL.cool;
  ctx.beginPath(); ctx.arc(x, top, 5, 0, Math.PI * 2); ctx.fill();
  label(ctx, 'LEFT', 10, 14, COL.cool);
  label(ctx, 'whole brain ' + (whole >= 0 ? '+' : '') + whole.toFixed(2), cx, 14, COL.ink, 'center');
  label(ctx, 'RIGHT', w - 10, 14, COL.warm, 'right');

  // Per stage, because a whole-brain average hides a lateral signal that
  // exists in the optic lobe and is gone by the descending neurons -- which is
  // this project's measured failure mode, so it gets its own row.
  const y0 = top + 22;
  const rowH = Math.max(12, (h - y0 - 6) / Math.max(1, rows.length));
  rows.forEach((s, i) => {
    const y = y0 + i * rowH + rowH / 2;
    label(ctx, s.name.slice(0, 12), 8, y, COL.dim);
    const bx = 108, bw = w - bx - 44, bcx = bx + bw / 2;
    ctx.strokeStyle = COL.grid;
    ctx.beginPath(); ctx.moveTo(bcx, y - rowH * 0.32); ctx.lineTo(bcx, y + rowH * 0.32); ctx.stroke();
    const b = s.balance || 0;
    ctx.fillStyle = b >= 0 ? COL.warm : COL.cool;
    const bar = b * (bw / 2);
    ctx.fillRect(bcx, y - 3, bar, 6);
    label(ctx, (b >= 0 ? '+' : '') + b.toFixed(2), w - 6, y, COL.dim, 'right');
  });
}

/* ------------------------------------------------------------------ threat */

function drawThreat() {
  const { ctx, w, h } = fit('threat');
  const t = state.threat_norm || 0;
  const cx = w / 2, cy = h - 22, r = Math.min(w / 2 - 16, h - 44);

  ctx.lineWidth = 12;
  ctx.strokeStyle = COL.grid;
  ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, 0); ctx.stroke();
  ctx.strokeStyle = t > 0.66 ? COL.bad : t > 0.33 ? COL.warm : COL.good;
  ctx.beginPath(); ctx.arc(cx, cy, r, Math.PI, Math.PI + Math.PI * t); ctx.stroke();
  ctx.lineWidth = 1;

  label(ctx, (100 * t).toFixed(0) + '%', cx, cy - r / 2, COL.ink, 'center', 20);
  label(ctx, 'raw ' + (state.threat || 0).toFixed(3), cx, cy + 12, COL.dim, 'center');
  const groups = (layout.types || []).filter((g) => g.panel === 'threat');
  label(ctx, groups.map((g) => `${g.name}(${g.n})`).join('  ') || 'no looming cells',
        cx, h - 6, COL.dim, 'center');
}

/* --------------------------------------------------------- population flow */

function drawFlow() {
  const { ctx, w, h } = fit('flow');
  const rows = state.stages || [];
  if (!rows.length) { label(ctx, 'no stages', 8, h / 2); return; }

  const pad = 10;
  const colW = (w - pad * 2) / rows.length;
  const baseY = h - 34;
  const topY = 26;
  const meanScale = scaleFor('flow_mean', Math.max(...rows.map((s) => s.mean)));
  const modScale = scaleFor('flow_mod', Math.max(...rows.map((s) => s.modulation)));

  rows.forEach((s, i) => {
    const x = pad + i * colW;
    const mid = x + colW / 2;

    // arrow to the next stage: this is a chain, not a set of independent bars
    if (i < rows.length - 1) {
      ctx.strokeStyle = COL.edge;
      ctx.beginPath();
      ctx.moveTo(x + colW - 6, topY - 12);
      ctx.lineTo(x + colW + 6, topY - 12);
      ctx.stroke();
    }

    const barW = Math.min(30, colW * 0.3);
    const mh = (s.mean / meanScale) * (baseY - topY);
    ctx.fillStyle = heat(s.active);
    ctx.fillRect(mid - barW - 2, baseY - mh, barW, mh);

    const dh = (s.modulation / modScale) * (baseY - topY);
    ctx.fillStyle = COL.good;
    ctx.fillRect(mid + 2, baseY - dh, barW, dh);

    label(ctx, s.name.replace('visual_projection', 'projection').slice(0, 11),
          mid, baseY + 10, COL.dim, 'center');
    label(ctx, `mean ${s.mean.toFixed(2)}`, mid, baseY + 22, COL.dim, 'center');
    label(ctx, `mod ${s.modulation.toFixed(3)}`, mid, baseY + 32, COL.good, 'center');
  });
  label(ctx, 'bar left = mean rate (brightness = active fraction)   bar right = modulation depth',
        pad, 12, COL.dim);
}

/* -------------------------------------------------------------- cell types */

function typeBars(canvasId, panels) {
  const { ctx, w, h } = fit(canvasId);
  const groups = (layout.types || []).filter((g) => panels.includes(g.panel));
  if (!groups.length) { label(ctx, 'not in this connectome', 8, h / 2); return; }
  const values = groups.map((g) => (state.types || {})[g.name] || 0);
  const scale = scaleFor('types_' + canvasId, Math.max(...values.map(Math.abs)));

  const rowH = Math.min(26, (h - 6) / groups.length);
  groups.forEach((g, i) => {
    const y = 4 + i * rowH + rowH / 2;
    label(ctx, g.name, 6, y, COL.ink);
    const bx = 58, bw = w - bx - 52;
    ctx.fillStyle = COL.grid;
    ctx.fillRect(bx, y - 5, bw, 10);
    const v = values[i];
    const frac = Math.max(-1, Math.min(1, v / scale));
    ctx.fillStyle = v >= 0 ? COL.warm : COL.cool;
    if (v >= 0) ctx.fillRect(bx, y - 5, bw * frac, 10);
    else ctx.fillRect(bx + bw * (1 + frac), y - 5, bw * -frac, 10);
    label(ctx, v.toFixed(2), w - 6, y, COL.dim, 'right');
  });
}

/* ------------------------------------------------------------ DN -> keys */

let dnOrder = null;
function dnRanking() {
  if (dnOrder || !layout || !layout.decoder) return dnOrder;
  const W = layout.decoder.weight;               // (n_actions, n_dn)
  const nDn = layout.decoder.shape[1];
  const total = new Array(nDn).fill(0);
  for (const row of W) for (let j = 0; j < nDn; j++) total[j] += Math.abs(row[j]);
  // Sort descending neurons by how much influence they have at all, so the
  // panel's top rows are the ones that can actually move a key. With >100 DNs
  // an unsorted column is unreadable.
  dnOrder = total.map((v, j) => [v, j]).sort((a, b) => b[0] - a[0]).map((p) => p[1]);
  return dnOrder;
}

function drawDn() {
  const { ctx, w, h } = fit('dn');
  const rates = state.dn;
  if (!rates) { label(ctx, 'no descending rates', 8, h / 2); return; }
  const order = dnRanking();
  const actions = layout.actions || [];
  const W = layout.decoder ? layout.decoder.weight : null;
  const held = new Set((state.action || []).flatMap((v, i) => (v ? [i] : [])));

  const leftX = 8, barW = 120;
  const rightX = w - 130;
  const topY = 16, botY = h - 10;
  const shown = order ? order.slice(0, Math.min(order.length, Math.floor((botY - topY) / 5))) : [];
  const dnY = (k) => topY + (k + 0.5) * ((botY - topY) / Math.max(1, shown.length));
  const scale = scaleFor('dn', Math.max(...rates.map(Math.abs)));

  label(ctx, `${rates.length} descending neurons (top ${shown.length} by influence)`,
        leftX, 8, COL.dim);
  label(ctx, 'keys', rightX, 8, COL.dim);

  // wiring first, so the bars draw over it
  if (W) {
    const actY = (i) => topY + (i + 0.5) * ((botY - topY) / Math.max(1, actions.length));
    const contributions = [];
    for (let i = 0; i < actions.length; i++) {
      for (let k = 0; k < shown.length; k++) {
        const j = shown[k];
        const c = Math.abs((W[i] || [])[j] || 0) * Math.abs(rates[j] || 0);
        if (c > 0) contributions.push([c, i, k]);
      }
    }
    contributions.sort((a, b) => b[0] - a[0]);
    // Only the strongest few per key: 136 x 13 lines is a grey rectangle, and
    // the question the panel answers is *which* neuron pressed the key.
    const perAction = {};
    const cMax = contributions.length ? contributions[0][0] : 1;
    for (const [c, i, k] of contributions) {
      perAction[i] = (perAction[i] || 0) + 1;
      if (perAction[i] > 5) continue;
      ctx.strokeStyle = `rgba(255, 179, 71, ${(0.08 + 0.72 * (c / cMax)).toFixed(3)})`;
      ctx.lineWidth = 0.5 + 1.8 * (c / cMax);
      ctx.beginPath();
      ctx.moveTo(leftX + barW + 2, dnY(k));
      ctx.bezierCurveTo((leftX + rightX) / 2, dnY(k), (leftX + rightX) / 2, actY(i), rightX - 4, actY(i));
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    actions.forEach((name, i) => {
      const y = actY(i);
      const on = held.has(i);
      ctx.fillStyle = on ? COL.good : COL.grid;
      ctx.fillRect(rightX, y - 7, 14, 14);
      label(ctx, name, rightX + 20, y, on ? COL.ink : COL.dim);
    });
  }

  shown.forEach((j, k) => {
    const y = dnY(k);
    const v = rates[j] || 0;
    ctx.fillStyle = COL.grid;
    ctx.fillRect(leftX, y - 1.6, barW, 3.2);
    ctx.fillStyle = COL.warm;
    ctx.fillRect(leftX, y - 1.6, barW * Math.min(1, Math.abs(v) / scale), 3.2);
  });
  const names = layout.dn_labels || [];
  if (shown.length && names.length) {
    label(ctx, `strongest: ${names[shown[0]] || shown[0]}`, leftX, h - 2, COL.dim);
  }
}

/* ----------------------------------------------------------------- series */

const SERIES = [
  { key: 'player_hp', color: COL.good, lo: 0, hi: 1 },
  { key: 'boss_hp', color: COL.bad, lo: 0, hi: 1 },
  { key: 'reward', color: COL.warm, lo: null, hi: null },
  { key: 'balance', color: COL.cool, lo: -1, hi: 1 },
  { key: 'threat_norm', color: '#c792ea', lo: 0, hi: 1 },
];

function drawSeries() {
  const { ctx, w, h } = fit('series');
  const rowH = h / SERIES.length;
  SERIES.forEach((s, i) => {
    const arr = history[s.key] || [];
    const y0 = i * rowH + 4, y1 = (i + 1) * rowH - 4;
    ctx.strokeStyle = COL.grid;
    ctx.beginPath(); ctx.moveTo(70, y1); ctx.lineTo(w - 6, y1); ctx.stroke();
    label(ctx, s.key, 6, (y0 + y1) / 2, s.color);
    if (arr.length < 2) return;

    let lo = s.lo, hi = s.hi;
    if (lo === null) {
      lo = Math.min(...arr); hi = Math.max(...arr);
      if (hi - lo < 1e-6) { hi = lo + 1e-6; }
    }
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    for (let k = 0; k < arr.length; k++) {
      const x = 70 + (w - 78) * (k / (HISTORY - 1));
      const t = (arr[k] - lo) / (hi - lo);
      const y = y1 - Math.max(0, Math.min(1, t)) * (y1 - y0);
      k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    label(ctx, arr[arr.length - 1].toFixed(3), w - 6, y0 + 6, COL.dim, 'right');
  });
}

/* ----------------------------------------------------------------- status */

const STATUS_KEYS = ['episode', 'step', 'state', 'player_hp', 'boss_hp', 'attempts',
                     'kills', 'deaths', 'waypoint', 'total_reward', 'steps_per_second'];

function drawStatus() {
  const lines = [];
  for (const key of STATUS_KEYS) {
    if (state[key] === undefined) continue;
    const v = state[key];
    lines.push(`<span class="k">${key.padEnd(17)}</span>${typeof v === 'number' ? v.toFixed(3) : v}`);
  }
  if (state.held_keys && state.held_keys.length) {
    lines.push(`<span class="k">${'held'.padEnd(17)}</span>${state.held_keys.join(' ')}`);
  }
  lines.push(`<span class="k">${'neurons'.padEnd(17)}</span>${layout.n_neurons}`);
  if (layout.missing_types && layout.missing_types.length) {
    lines.push(`<span class="k">${'absent types'.padEnd(17)}</span>${layout.missing_types.join(' ')}`);
  }
  document.getElementById('status').innerHTML = lines.join('\n');

  const banner = document.getElementById('banner');
  if (state.dry_run === true) { banner.className = 'dry'; banner.textContent = 'DRY RUN - no keys sent'; }
  else if (state.dry_run === false) { banner.className = 'live'; banner.textContent = 'LIVE INPUT'; }
  else { banner.className = ''; banner.textContent = ''; }
}

/* ------------------------------------------------------------------- loop */

function draw() {
  drawStatus();
  drawEyes();
  drawBalance();
  drawThreat();
  drawFlow();
  typeBars('types', ['motion_on', 'motion_off']);
  typeBars('flowcells', ['flow', 'medulla', 'input']);
  drawDn();
  drawSeries();
}

async function poll() {
  try {
    if (!layout) {
      const r = await fetch('layout.json?' + tick);
      if (r.ok) layout = await r.json();
    }
    const r = await fetch('state.json?' + tick);
    if (r.ok) {
      const next = await r.json();
      if (!next.waiting) {
        state = next;
        document.getElementById('waiting').hidden = true;
        document.getElementById('panels').hidden = false;
        push('reward', state.reward);
        push('balance', state.balance);
        push('threat_norm', state.threat_norm);
        if (state.player_hp !== undefined) push('player_hp', state.player_hp);
        if (state.boss_hp !== undefined) push('boss_hp', state.boss_hp);
        document.getElementById('subtitle').textContent =
          `episode ${state.episode} - step ${state.step}`;
        if (layout) draw();
      }
    }
    // The frame lags the state on purpose (it is published less often), so a
    // cache-busting counter that only advances with the state would pin a stale
    // image; use the step, which the publisher ties the frame to.
    const img = document.getElementById('frame');
    img.src = 'frame.png?' + Math.floor((state ? state.step : 0));
  } catch (e) {
    document.getElementById('subtitle').textContent = 'lost the run (' + e + ')';
  }
  tick += 1;
  setTimeout(poll, REFRESH);
}

window.addEventListener('resize', () => { if (layout && state) draw(); });
document.addEventListener('change', (e) => {
  if (e.target.name === 'eyech' && layout && state) draw();
});
poll();
