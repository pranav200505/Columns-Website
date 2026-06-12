/* ColumnLab — shared frontend behaviour */
(function () {
  'use strict';

  /* ---------- theme ---------- */
  const root = document.documentElement;
  const saved = localStorage.getItem('cl-theme');
  if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    root.setAttribute('data-theme', 'dark');
  }
  const themeBtn = document.getElementById('themeBtn');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      const dark = root.getAttribute('data-theme') === 'dark';
      if (dark) { root.removeAttribute('data-theme'); localStorage.setItem('cl-theme', 'light'); }
      else { root.setAttribute('data-theme', 'dark'); localStorage.setItem('cl-theme', 'dark'); }
    });
  }

  /* ---------- loading overlay on long-running form submits ---------- */
  const overlay = document.getElementById('busyOverlay');
  document.querySelectorAll('form[data-busy]').forEach(function (form) {
    form.addEventListener('submit', function () {
      if (!overlay) return;
      const msgs = (form.dataset.busyMsgs || 'Crunching the numbers…').split('|');
      const title = form.dataset.busyTitle || 'Computing';
      overlay.querySelector('h3').textContent = title;
      const p = overlay.querySelector('p');
      let i = 0;
      p.textContent = msgs[0];
      overlay.classList.add('open');
      if (msgs.length > 1) {
        const t = setInterval(function () {
          i = (i + 1) % msgs.length;
          p.textContent = msgs[i];
        }, 2600);
        window.addEventListener('pagehide', function () { clearInterval(t); });
      }
    });
  });

  /* ---------- lightbox ---------- */
  const lb = document.getElementById('lightbox');
  if (lb) {
    const lbImg = lb.querySelector('img');
    document.querySelectorAll('.plot-frame img, .hist-grid img').forEach(function (img) {
      img.addEventListener('click', function () {
        lbImg.src = img.src;
        lb.classList.add('open');
      });
    });
    lb.addEventListener('click', function () { lb.classList.remove('open'); });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') lb.classList.remove('open');
    });
  }

  /* ---------- probabilistic page: CoV toggles + N slider ---------- */
  document.querySelectorAll('[data-cov-toggle]').forEach(function (cb) {
    const row = document.getElementById(cb.dataset.covToggle);
    if (!row) return;
    const sync = function () { row.classList.toggle('disabled', !cb.checked); };
    cb.addEventListener('change', sync);
    sync();
  });
  const nSlider = document.getElementById('N_slider');
  const nValue = document.getElementById('N_value');
  const nInput = document.getElementById('N_input');
  if (nSlider && nValue && nInput) {
    const sync = function (v) { nValue.textContent = v; nInput.value = v; };
    nSlider.addEventListener('input', function () { sync(nSlider.value); });
    sync(nSlider.value);
  }

  /* ---------- PM curves page ---------- */
  const geomRadios = document.querySelectorAll('input[name="geometry"]');
  if (geomRadios.length) {
    const blocks = {
      rectangular: document.getElementById('rect_inputs'),
      circular: document.getElementById('circ_inputs'),
      jacketed: document.getElementById('jacket_inputs')
    };
    const puHint = document.getElementById('pu_hint');
    const puHints = {
      dimensional: 'Axial load in kN, e.g. 1000',
      nondimensional: 'Non-dimensional P* = P/(f_ck·area term), e.g. 0.2'
    };

    function currentGeom() {
      const r = document.querySelector('input[name="geometry"]:checked');
      return r ? r.value : 'rectangular';
    }

    function onGeomChange() {
      const g = currentGeom();
      Object.keys(blocks).forEach(function (k) {
        if (blocks[k]) blocks[k].classList.toggle('hidden', k !== g);
      });
      drawPreview();
    }
    geomRadios.forEach(function (r) { r.addEventListener('change', onGeomChange); });

    const plotTypeSel = document.querySelector('select[name="plot_type"]');
    if (plotTypeSel && puHint) {
      const syncHint = function () { puHint.textContent = puHints[plotTypeSel.value] || ''; };
      plotTypeSel.addEventListener('change', syncHint);
      syncHint();
    }

    /* ----- live cross-section preview (SVG) ----- */
    const svg = document.getElementById('sectionPreview');
    const NS = 'http://www.w3.org/2000/svg';

    function val(name, fallback) {
      const el = document.querySelector('[name="' + name + '"]');
      const v = el ? parseFloat(el.value) : NaN;
      return isFinite(v) && v > 0 ? v : fallback;
    }
    function intVal(name, fallback) {
      const v = Math.round(val(name, fallback));
      return v >= 1 ? v : fallback;
    }
    function el(tag, attrs) {
      const e = document.createElementNS(NS, tag);
      Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
      return e;
    }

    function rectBarPositions(b, D, cover, n) {
      const pos = [];
      const c = [[cover, cover], [b - cover, cover], [b - cover, D - cover], [cover, D - cover]];
      for (let i = 0; i < Math.min(4, n); i++) pos.push(c[i]);
      const rem = n - pos.length;
      const sides = [0, 0, 0, 0];
      for (let i = 0; i < rem; i++) sides[i % 4]++;
      const lin = function (a, b2, k) {
        const out = [];
        for (let i = 1; i <= k; i++) out.push(a + (b2 - a) * i / (k + 1));
        return out;
      };
      lin(cover, b - cover, sides[0]).forEach(function (x) { pos.push([x, cover]); });
      lin(cover, D - cover, sides[1]).forEach(function (y) { pos.push([b - cover, y]); });
      lin(cover, b - cover, sides[2]).forEach(function (x) { pos.push([x, D - cover]); });
      lin(cover, D - cover, sides[3]).forEach(function (y) { pos.push([cover, y]); });
      return pos;
    }

    function circBarPositions(cx, cy, R, n) {
      const pos = [];
      for (let i = 0; i < n; i++) {
        const t = 2 * Math.PI * i / n - Math.PI / 2;
        pos.push([cx + R * Math.cos(t), cy + R * Math.sin(t)]);
      }
      return pos;
    }

    function setStats(stats) {
      const wrap = document.getElementById('previewStats');
      if (!wrap) return;
      wrap.innerHTML = '';
      stats.forEach(function (s) {
        const d = document.createElement('div');
        d.className = 'stat-chip';
        d.innerHTML = '<div class="k">' + s[0] + '</div><div class="v">' + s[1] + '</div>';
        wrap.appendChild(d);
      });
    }

    function drawPreview() {
      if (!svg) return;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const g = currentGeom();
      const W = 300, H = 300, PAD = 26;
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

      const css = getComputedStyle(document.documentElement);
      const edge = css.getPropertyValue('--text').trim() || '#0f172a';
      const dash = css.getPropertyValue('--muted').trim() || '#64748b';
      const steel = '#dc2626';

      if (g === 'rectangular') {
        const b = val('b', 300), D = val('D', 600),
              cover = val('cover', 40), dia = val('bar_dia', 16),
              n = intVal('num_bars', 8);
        const s = Math.min((W - 2 * PAD) / b, (H - 2 * PAD) / D);
        const ox = (W - b * s) / 2, oy = (H - D * s) / 2;
        svg.appendChild(el('rect', { x: ox, y: oy, width: b * s, height: D * s, fill: 'none', stroke: edge, 'stroke-width': 2, rx: 2 }));
        svg.appendChild(el('rect', { x: ox + cover * s, y: oy + cover * s, width: (b - 2 * cover) * s, height: (D - 2 * cover) * s, fill: 'none', stroke: dash, 'stroke-width': 1, 'stroke-dasharray': '5 4' }));
        rectBarPositions(b, D, cover, n).forEach(function (p) {
          svg.appendChild(el('circle', { cx: ox + p[0] * s, cy: oy + p[1] * s, r: Math.max(2.5, dia * s / 2), fill: steel }));
        });
        const As = n * Math.PI * dia * dia / 4;
        setStats([
          ['Section', b.toFixed(0) + ' × ' + D.toFixed(0) + ' mm'],
          ['Bars', n + ' × ⌀' + dia.toFixed(0)],
          ['Steel area', (As / 100).toFixed(1) + ' cm²'],
          ['Steel %', (100 * As / (b * D)).toFixed(2) + ' %']
        ]);
      } else if (g === 'circular') {
        const D = val('D_circ', 600), cover = val('cover_c', 40),
              dia = val('bar_dia_c', 16), n = intVal('num_bars_c', 8);
        const s = Math.min(W, H - 2 * PAD) / D * 0.92;
        const cx = W / 2, cy = H / 2, R = D / 2 * s;
        svg.appendChild(el('circle', { cx: cx, cy: cy, r: R, fill: 'none', stroke: edge, 'stroke-width': 2 }));
        svg.appendChild(el('circle', { cx: cx, cy: cy, r: (D / 2 - cover) * s, fill: 'none', stroke: dash, 'stroke-width': 1, 'stroke-dasharray': '5 4' }));
        circBarPositions(cx, cy, (D / 2 - cover) * s, n).forEach(function (p) {
          svg.appendChild(el('circle', { cx: p[0], cy: p[1], r: Math.max(2.5, dia * s / 2), fill: steel }));
        });
        const As = n * Math.PI * dia * dia / 4;
        const Ag = Math.PI * D * D / 4;
        setStats([
          ['Diameter', D.toFixed(0) + ' mm'],
          ['Bars', n + ' × ⌀' + dia.toFixed(0)],
          ['Steel area', (As / 100).toFixed(1) + ' cm²'],
          ['Steel %', (100 * As / Ag).toFixed(2) + ' %']
        ]);
      } else {
        const Dc = val('D_core', 300), Bj = val('B_j', 500), Dj = val('D_j', 600),
              cc = val('core_cover', 30), cj = val('jacket_cover', 40),
              dc = val('core_bar_dia', 12), nc = intVal('core_num_bars', 6),
              dj = val('jacket_bar_dia', 16), nj = intVal('jacket_num_bars', 8);
        const s = Math.min((W - 2 * PAD) / Bj, (H - 2 * PAD) / Dj);
        const ox = (W - Bj * s) / 2, oy = (H - Dj * s) / 2;
        svg.appendChild(el('rect', { x: ox, y: oy, width: Bj * s, height: Dj * s, fill: 'none', stroke: edge, 'stroke-width': 2, rx: 2 }));
        const ccx = ox + Bj * s / 2, ccy = oy + Dj * s / 2;
        svg.appendChild(el('circle', { cx: ccx, cy: ccy, r: Dc / 2 * s, fill: 'none', stroke: dash, 'stroke-width': 1.4, 'stroke-dasharray': '6 4' }));
        circBarPositions(ccx, ccy, (Dc / 2 - cc) * s, nc).forEach(function (p) {
          svg.appendChild(el('circle', { cx: p[0], cy: p[1], r: Math.max(2, dc * s / 2), fill: steel }));
        });
        rectBarPositions(Bj, Dj, cj, nj).forEach(function (p) {
          svg.appendChild(el('circle', { cx: ox + p[0] * s, cy: oy + p[1] * s, r: Math.max(2.5, dj * s / 2), fill: steel }));
        });
        const As = nc * Math.PI * dc * dc / 4 + nj * Math.PI * dj * dj / 4;
        setStats([
          ['Jacket', Bj.toFixed(0) + ' × ' + Dj.toFixed(0) + ' mm'],
          ['Core ⌀', Dc.toFixed(0) + ' mm'],
          ['Total bars', (nc + nj) + ''],
          ['Steel area', (As / 100).toFixed(1) + ' cm²']
        ]);
      }
    }

    document.querySelectorAll('#pmForm input, #pmForm select').forEach(function (inp) {
      inp.addEventListener('input', drawPreview);
    });
    // redraw on theme change so stroke colours follow the palette
    new MutationObserver(drawPreview).observe(root, { attributes: true, attributeFilter: ['data-theme'] });

    onGeomChange();
  }
})();
