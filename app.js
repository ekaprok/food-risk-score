'use strict';

// Two files feed the page: the scores, and the trade matrix the supplier
// breakdown is summed from.
const SCORE_CSV = 'food_score.csv';
const TRADE_CSV = 'faostat/TradeMatrix.csv';

// The score CSV spells its headers out in full, e.g. "food_risk_weighted
// (Vulnerability_weighted x Criticality)". We find each one by the short name it
// starts with, so re-wording the explanation in brackets does not break the page.
const COLUMN_PREFIXES = {
  risk: 'food_risk_weighted (',
  riskSim: 'food_risk_weighted_sim',
  topSupplier: 'top_supplier (',
  topShare: 'top_supplier_share',
};

// A score is vulnerability x criticality, so it runs from 0 to 1. Every score
// shown on the page is banded by these three thresholds; `limit` is inclusive.
const RISK_BANDS = [
  { limit: 0.10, label: 'Low risk', icon: '✓', variable: '--band-low' },
  { limit: 0.20, label: 'Moderate risk', icon: '●', variable: '--band-moderate' },
  { limit: Infinity, label: 'High risk', icon: '▲', variable: '--band-high' },
];

// Three named suppliers plus "everyone else". More colours than this stop being
// reliably distinguishable, so the tail is folded into one grey slice.
const MAX_DONUT_SLICES = 3;

let scoreRows = [];   // the year rows of food_score.csv, blanks dropped
let tradeRows = [];   // every row of the trade matrix
let columns = {};     // short name -> the full header it was found under
const charts = {};    // canvas id -> the Chart currently drawn on it

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/** Parses a CSV at `url` into `{ rows, headers }`. */
function loadCsv(url) {
  return new Promise(function (resolve, reject) {
    Papa.parse(url, {
      download: true,
      header: true,
      skipEmptyLines: 'greedy',  // drops the ",,,,," separator lines
      complete: function (result) {
        resolve({ rows: result.data, headers: result.meta.fields });
      },
      error: reject,
    });
  });
}

/** The full header starting with `prefix`. */
function findColumn(headers, prefix) {
  const match = headers.find(function (header) { return header.startsWith(prefix); });
  if (!match) {
    throw new Error('food_score.csv has no column starting with "' + prefix + '"');
  }
  return match;
}

function start() {
  Promise.all([loadCsv(SCORE_CSV), loadCsv(TRADE_CSV)])
    .then(function (results) {
      const scores = results[0];
      Object.keys(COLUMN_PREFIXES).forEach(function (name) {
        columns[name] = findColumn(scores.headers, COLUMN_PREFIXES[name]);
      });
      scoreRows = scores.rows.filter(function (row) { return row.country; });
      tradeRows = results[1].rows;

      fillDropdown('country-select', 'country');
      fillDropdown('commodity-select', 'commodity');
      document.getElementById('country-select').addEventListener('change', render);
      document.getElementById('commodity-select').addEventListener('change', render);

      // Charts read their colours from the stylesheet, so redraw when the OS
      // switches between light and dark.
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', render);

      document.getElementById('status').hidden = true;
      document.getElementById('view').hidden = false;
      render();
    })
    .catch(function (error) {
      document.getElementById('status').textContent =
        'Could not load the data: ' + error.message;
    });
}

/** Fills a dropdown with every distinct value of `field`, in order. */
function fillDropdown(selectId, field) {
  const values = [];
  scoreRows.forEach(function (row) {
    if (values.indexOf(row[field]) === -1) values.push(row[field]);
  });
  document.getElementById(selectId).innerHTML = values.map(function (value) {
    return '<option>' + value + '</option>';
  }).join('');
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** The value of a CSS variable, e.g. css('--series-1'). */
function css(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

/** A score column of a row, as a number. */
function score(row, name) {
  return Number(row[columns[name]]);
}

function format(value) {
  return isFinite(value) ? value.toFixed(2) : '–';
}

/** A share as a percentage, never rounded down to a misleading "0%". */
function formatShare(share) {
  if (share > 0 && share < 0.005) return '<1%';
  return Math.round(share * 100) + '%';
}

function formatTonnes(value) {
  return Math.round(value).toLocaleString('en-GB') + ' t';
}

function bandFor(value) {
  if (!isFinite(value)) return { label: 'No score', icon: '–', variable: '--text-muted' };
  return RISK_BANDS.find(function (band) { return value <= band.limit; });
}

/** A round axis top that leaves a little headroom above the biggest value. */
function axisTop(values) {
  const biggest = Math.max.apply(null, values.filter(isFinite));
  return Math.max(0.10, Math.ceil((biggest * 1.1) / 0.05) * 0.05);
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

// Chart.js draws its own text on the canvas, out of reach of the stylesheet, so
// its base size is set here to match the doubled type in style.css.
Chart.defaults.font.size = 16;

/** Replaces whatever is on a canvas with a new chart. */
function draw(canvasId, config) {
  if (charts[canvasId]) charts[canvasId].destroy();
  charts[canvasId] = new Chart(document.getElementById(canvasId), config);
}

/** One series over the five years, on an axis shared with its twin. */
function drawTrend(canvasId, years, values, color, top) {
  draw(canvasId, {
    type: 'line',
    data: {
      labels: years,
      datasets: [{
        label: 'Risk score',
        data: values,
        borderColor: color,
        borderWidth: 2,
        tension: 0.25,
        pointBackgroundColor: color,
        pointBorderColor: css('--surface-1'),
        pointBorderWidth: 2,      // the surface ring keeps dots legible on the line
        pointRadius: 4,
        pointHoverRadius: 6,
        hitRadius: 12,            // a comfortable hover target, not a pinpoint
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },   // one series; the caption names it
        tooltip: {
          callbacks: {
            label: function (item) { return 'Risk score ' + format(item.parsed.y); },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { color: css('--gridline') },
          ticks: { color: css('--text-muted') },
        },
        y: {
          min: 0,
          max: top,
          grid: { color: css('--gridline') },
          border: { display: false },
          ticks: { color: css('--text-muted'), callback: format },
        },
      },
    },
  });
}

/** The supplier split for one year, plus its legend. */
function drawSuppliers(suppliers, year) {
  const holder = document.getElementById('donut-holder');
  const legend = document.getElementById('supplier-legend');
  const colors = [css('--series-1'), css('--series-2'), css('--series-3')];

  document.getElementById('donut-year').textContent = year;

  if (!suppliers.length) {
    holder.hidden = true;
    legend.innerHTML = '<li>The trade matrix records no flows for this year.</li>';
    if (charts['supplier-chart']) charts['supplier-chart'].destroy();
    delete charts['supplier-chart'];
    return;
  }

  const sliceColors = suppliers.map(function (supplier, index) {
    return supplier.isOther ? css('--series-other') : colors[index];
  });

  legend.innerHTML = suppliers.map(function (supplier, index) {
    return '<li><span class="swatch" style="--swatch: ' + sliceColors[index] + '"></span>'
      + supplier.name
      + '<span class="share">' + formatShare(supplier.share) + '</span></li>';
  }).join('');

  holder.hidden = false;

  draw('supplier-chart', {
    type: 'doughnut',
    data: {
      labels: suppliers.map(function (supplier) { return supplier.name; }),
      datasets: [{
        data: suppliers.map(function (supplier) { return supplier.tonnes; }),
        backgroundColor: sliceColors,
        borderColor: css('--surface-1'),
        borderWidth: 2,   // a gap in the surface colour, not a stroke round each slice
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '60%',
      plugins: {
        legend: { display: false },   // the HTML list below is the legend
        tooltip: {
          callbacks: {
            label: function (item) {
              const supplier = suppliers[item.dataIndex];
              return formatTonnes(supplier.tonnes) + ' (' + formatShare(supplier.share) + ')';
            },
          },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// The trade matrix
// ---------------------------------------------------------------------------

/** Who shipped the country this crop in this year, biggest first, tail folded
    into one "other" entry. Mirrors supplier_flows() in food_score.py: the
    suppliers are the reporters, the country is the partner. */
function suppliersFor(country, commodity, year) {
  const totals = new Map();

  tradeRows.forEach(function (row) {
    const matches = row['Partner Countries'] === country
      && row['Item'] === commodity
      && row['Element'] === 'Export quantity'
      && row['Year'] === String(year);
    if (!matches) return;

    const tonnes = Number(row['Value']);
    if (!isFinite(tonnes) || tonnes <= 0) return;

    const supplier = row['Reporter Countries'];
    totals.set(supplier, (totals.get(supplier) || 0) + tonnes);
  });

  const ranked = Array.from(totals, function (entry) {
    return { name: entry[0], tonnes: entry[1], isOther: false };
  }).sort(function (a, b) { return b.tonnes - a.tonnes; });

  const shown = ranked.slice(0, MAX_DONUT_SLICES);
  const tail = ranked.slice(MAX_DONUT_SLICES);
  if (tail.length) {
    shown.push({
      name: 'Other suppliers (' + tail.length + ')',
      tonnes: tail.reduce(function (sum, supplier) { return sum + supplier.tonnes; }, 0),
      isOther: true,
    });
  }

  const total = ranked.reduce(function (sum, supplier) { return sum + supplier.tonnes; }, 0);
  shown.forEach(function (supplier) {
    supplier.share = total ? supplier.tonnes / total : 0;
  });
  return shown;
}

// ---------------------------------------------------------------------------
// Putting the page together
// ---------------------------------------------------------------------------

/** Writes a score, its band and its wash into one of the two cards. */
function fillScoreCard(prefix, value, year) {
  const band = bandFor(value);
  const card = document.getElementById(prefix + '-card');
  card.style.setProperty('--band', 'var(' + band.variable + ')');
  document.getElementById(prefix + '-year').textContent = year;
  document.getElementById(prefix + '-score').textContent = format(value);
  document.getElementById(prefix + '-band-icon').textContent = band.icon;
  document.getElementById(prefix + '-band-label').textContent = band.label;
}

function render() {
  const country = document.getElementById('country-select').value;
  const commodity = document.getElementById('commodity-select').value;

  const rows = scoreRows.filter(function (row) {
    return row.country === country && row.commodity === commodity;
  });
  const yearRows = rows.filter(function (row) { return row.year !== 'AVERAGES'; });
  if (!yearRows.length) return;

  const latest = yearRows[yearRows.length - 1];
  const years = yearRows.map(function (row) { return row.year; });
  const baseline = yearRows.map(function (row) { return score(row, 'risk'); });
  const shocked = yearRows.map(function (row) { return score(row, 'riskSim'); });

  const baselineLatest = score(latest, 'risk');
  const shockLatest = score(latest, 'riskSim');

  fillScoreCard('baseline', baselineLatest, latest.year);
  fillScoreCard('shock', shockLatest, latest.year);

  document.getElementById('shock-subhead').textContent =
    latest[columns.topSupplier] + ' supplies ' + formatShare(score(latest, 'topShare'))
    + ' of ' + country + '\u2019s recorded ' + commodity + ' imports. ' + '\n'
    + 'If ' + latest[columns.topSupplier] + ' stopped providing ' + commodity + ', '
    + 'vulnerability would change from '
    + format(baselineLatest) + ' (' + bandFor(baselineLatest).label + ')'
    + ' to ' + format(shockLatest) + ' (' + bandFor(shockLatest).label + ').';

  // One axis top for both trends, so the two columns can be read against
  // each other rather than each against its own scale.
  const top = axisTop(baseline.concat(shocked));
  drawTrend('baseline-chart', years, baseline, css('--series-1'), top);
  drawTrend('shock-chart', years, shocked, css('--series-2'), top);

  drawSuppliers(suppliersFor(country, commodity, latest.year), latest.year);
}

start();
