let allRows = [];
let intentsList = [];
let pollingInterval = null;
let selectedIds = new Set();

let currentPage = 1;
let intentChart = null;

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

async function api(url, opts = {}) {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts
  });
  return res.json();
}

function toast(msg, type = 'success') {
  let el = document.querySelector('.toast');
  if (!el) {
    el = document.createElement('div');
    el.className = 'toast';
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.className = `toast ${type}`;
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => el.classList.remove('show'), 3500);
}

function debounce(fn, ms = 300) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// --- Load data ---
async function loadData() {
  const [dataRes, intentsRes] = await Promise.all([
    api('/api/data'),
    api('/api/intents')
  ]);

  allRows = dataRes.rows || [];
  intentsList = intentsRes || [];

  updateStats(dataRes.stats);
  populateIntentFilters();
  resetPagination();
  renderTable();
}

function updateStats(stats) {
  if (!stats) return;
  $('#statTotal').textContent = stats.total;
  $('#statPending').textContent = stats.pending;
  $('#statApproved').textContent = stats.approved;
  $('#statRejected').textContent = stats.rejected;
}

function populateIntentFilters() {
  const filterSelect = $('#filterIntent');
  const batchSelect = $('#batchIntentSelect');

  [filterSelect, batchSelect].forEach((sel, i) => {
    const currentVal = sel.value;
    const options = i === 0
      ? [{ value: 'all', label: 'All' }]
      : [{ value: '', label: '-- Chọn intent --' }];

    intentsList.forEach(intent => options.push({ value: intent, label: intent }));

    sel.innerHTML = options.map(o => `<option value="${o.value}">${o.label}</option>`).join('');
    sel.value = currentVal;
  });
}

function getFilteredRows() {
  const statusFilter = $('#filterStatus').value;
  const intentFilter = $('#filterIntent').value;
  const searchFilter = $('#filterSearch').value.toLowerCase().trim();

  return allRows.filter(row => {
    if (statusFilter !== 'all' && row.status !== statusFilter) return false;
    if (intentFilter !== 'all') {
      const matchIntent = (row.labeled_intent === intentFilter) ||
                          (row.snorkel_intent === intentFilter);
      if (!matchIntent) return false;
    }
    if (searchFilter) {
      const haystack = `${row.raw_text} ${row.cleaned_text} ${row.snorkel_intent}`.toLowerCase();
      if (!haystack.includes(searchFilter)) return false;
    }
    return true;
  });
}

// --- Pagination ---
function getPageSize() {
  const el = $('#pageSizeSelect');
  if (!el) return 25;
  const v = el.value;
  return v === 'all' ? Infinity : parseInt(v, 10);
}

function getTotalPages(filteredLen) {
  const size = getPageSize();
  return size === Infinity ? 1 : Math.max(1, Math.ceil(filteredLen / size));
}

function resetPagination() {
  currentPage = 1;
}

function updatePaginationInfo(filteredLen) {
  const size = getPageSize();
  const totalPages = getTotalPages(filteredLen);
  const start = filteredLen === 0 ? 0 : (currentPage - 1) * size + 1;
  const end = Math.min(currentPage * size, filteredLen);

  const startEl = $('#showingStart');
  const endEl = $('#showingEnd');
  const totalEl = $('#showingTotal');
  if (startEl) startEl.textContent = start;
  if (endEl) endEl.textContent = end;
  if (totalEl) totalEl.textContent = filteredLen;

  const footEl = $('#tableFoot');
  if (!footEl) return;
  if (filteredLen === 0 || totalPages <= 1) {
    footEl.style.display = 'none';
    return;
  }
  footEl.style.display = '';

  let html = '';
  const maxVisible = 5;
  let pages = [];
  if (totalPages <= maxVisible + 2) {
    for (let i = 1; i <= totalPages; i++) pages.push(i);
  } else {
    pages.push(1);
    let startP = Math.max(2, currentPage - 1);
    let endP = Math.min(totalPages - 1, currentPage + 1);
    if (currentPage <= 3) { startP = 2; endP = Math.min(maxVisible, totalPages - 1); }
    if (currentPage >= totalPages - 2) { startP = Math.max(2, totalPages - maxVisible + 1); endP = totalPages - 1; }
    if (startP > 2) pages.push('...');
    for (let i = startP; i <= endP; i++) pages.push(i);
    if (endP < totalPages - 1) pages.push('...');
    pages.push(totalPages);
  }

  const prevDisabled = currentPage === 1;
  const nextDisabled = currentPage === totalPages;
  html += `<button class="page-btn${prevDisabled ? ' disabled' : ''}" data-page="${prevDisabled ? '' : currentPage - 1}"${prevDisabled ? ' disabled' : ''}>&laquo; Prev</button>`;
  for (const p of pages) {
    if (p === '...') {
      html += `<span class="page-dots">...</span>`;
    } else {
      html += `<button class="page-btn${p === currentPage ? ' active' : ''}" data-page="${p}">${p}</button>`;
    }
  }
  html += `<button class="page-btn${nextDisabled ? ' disabled' : ''}" data-page="${nextDisabled ? '' : currentPage + 1}"${nextDisabled ? ' disabled' : ''}>Next &raquo;</button>`;

  const btnsEl = $('#paginationBtns');
  if (btnsEl) btnsEl.innerHTML = html;
}

// --- Render ---
function renderTable() {
  try {
    const filtered = getFilteredRows();
    const tbody = $('#tableBody');
    if (!tbody) { console.error('tableBody not found'); return; }

    const size = getPageSize();
    const totalPages = getTotalPages(filtered.length);
    if (currentPage > totalPages) currentPage = totalPages;

    const slice = size === Infinity
      ? filtered
      : filtered.slice((currentPage - 1) * size, currentPage * size);

    updatePaginationInfo(filtered.length);

    if (slice.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="loading">No data found</td></tr>';
      return;
    }

    tbody.innerHTML = slice.map(row => {
      const isSelected = selectedIds.has(row.id);
      const statusClass = row.status === 'approved' ? 'status-approved'
                        : row.status === 'rejected' ? 'status-rejected' : '';

      const statusBadge = `<span class="status-badge ${row.status}">${row.status}</span>`;

      let actionsHtml;
      if (row.status === 'approved') {
        actionsHtml = `
          <div class="action-btns">
            <span class="status-badge approved">Done</span>
            <button class="btn btn-small" onclick="resetRow('${row.id}')">Undo</button>
          </div>`;
      } else if (row.status === 'rejected') {
        actionsHtml = `
          <div class="action-btns">
            <span class="status-badge rejected">Rejected</span>
            <button class="btn btn-small" onclick="resetRow('${row.id}')">Undo</button>
          </div>`;
      } else {
        actionsHtml = `
          <div class="action-btns">
            <button class="btn btn-approve" onclick="approveRow('${row.id}')">Approve</button>
            <button class="btn btn-reject" onclick="rejectRow('${row.id}')">Reject</button>
          </div>`;
      }

      const displayIntent = row.status === 'approved' ? row.labeled_intent || row.snorkel_intent
                          : row.status === 'rejected' ? '—'
                          : row.snorkel_intent;

      const confValue = row.snorkel_confidence
        ? `${(parseFloat(row.snorkel_confidence) * 100).toFixed(0)}%`
        : '—';

      const snorkelConfidence = parseFloat(row.snorkel_confidence) || 0;
      const confClass = snorkelConfidence > 0.6 ? 'green'
                      : snorkelConfidence > 0.3 ? 'yellow'
                      : 'red';

      return `<tr class="${statusClass}" data-id="${row.id}">
        <td class="col-check"><input type="checkbox" class="row-checkbox" value="${row.id}" ${isSelected ? 'checked' : ''}></td>
        <td class="col-id">${row.id}</td>
        <td class="raw-cell" title="${escapeHtml(row.raw_text || '')}">${escapeHtml(row.raw_text || '')}</td>
        <td>
          ${row.status === 'pending'
            ? `<input class="text-edit" type="text" value="${escapeHtml(row.cleaned_text || '')}" id="textEdit_${row.id}" data-original="${escapeHtml(row.cleaned_text || '')}">`
            : `<span>${escapeHtml(row.corrected_text || row.cleaned_text || '')}</span>`}
        </td>
        <td>
          ${row.status === 'pending'
            ? `<select class="intent-select" id="intentSelect_${row.id}">
                ${intentsList.map(i => `<option value="${i}" ${i === row.snorkel_intent ? 'selected' : ''}>${i}</option>`).join('')}
               </select>`
            : `<span>${displayIntent}</span>`}
        </td>
        <td style="text-align:right"><span style="color:var(--${confClass})">${confValue}</span></td>
        <td>${statusBadge}</td>
        <td>${actionsHtml}</td>
      </tr>`;
    }).join('');

    updateBatchButtonState();
  } catch (e) {
    console.error('renderTable error:', e);
    const tbody = $('#tableBody');
    if (tbody) tbody.innerHTML = `<tr><td colspan="8" class="loading" style="color:var(--red)">Error rendering table: ${escapeHtml(e.message)}</td></tr>`;
  }
}

function escapeHtml(text) {
  if (!text) return '';
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// --- Actions ---
async function approveRow(id) {
  const select = $(`#intentSelect_${id}`);
  const textInput = $(`#textEdit_${id}`);
  if (!select) return;

  const intent = select.value;
  const correctedText = textInput ? textInput.value : '';

  if (!intent) {
    toast('Please select an intent', 'error');
    return;
  }

  await api('/api/label', {
    method: 'POST',
    body: JSON.stringify({ id, intent, corrected_text: correctedText })
  });

  selectedIds.delete(id);
  toast(`Row #${id} approved as "${intent}"`);
  await loadData();
}

async function rejectRow(id) {
  await api('/api/reject', {
    method: 'POST',
    body: JSON.stringify({ id })
  });

  selectedIds.delete(id);
  toast(`Row #${id} rejected`, 'info');
  await loadData();
}

async function resetRow(id) {
  await api('/api/reset', {
    method: 'POST',
    body: JSON.stringify({ id })
  });

  selectedIds.delete(id);
  toast(`Row #${id} reset`, 'info');
  await loadData();
}

// --- Batch ---
document.addEventListener('change', (e) => {
  if (e.target.classList.contains('row-checkbox')) {
    if (e.target.checked) selectedIds.add(e.target.value);
    else selectedIds.delete(e.target.value);
    updateBatchButtonState();
  }
});

$('#selectAll')?.addEventListener('change', (e) => {
  const checked = e.target.checked;
  $$('.row-checkbox').forEach(cb => {
    cb.checked = checked;
    if (checked) selectedIds.add(cb.value);
    else selectedIds.delete(cb.value);
  });
  updateBatchButtonState();
});

$('#selectAllHead')?.addEventListener('change', (e) => {
  $('#selectAll').checked = e.target.checked;
  $('#selectAll').dispatchEvent(new Event('change'));
});

function updateBatchButtonState() {
  const count = selectedIds.size;
  $('#batchApproveBtn').disabled = count === 0;
  $('#batchRejectBtn').disabled = count === 0;
}

$('#batchApproveBtn')?.addEventListener('click', async () => {
  const intent = $('#batchIntentSelect').value;
  if (!intent) { toast('Chọn intent trước khi batch approve', 'error'); return; }

  const ids = Array.from(selectedIds);
  await api('/api/batch', {
    method: 'POST',
    body: JSON.stringify({ ids, action: 'label', intent })
  });

  selectedIds.clear();
  toast(`Approved ${ids.length} rows as "${intent}"`);
  await loadData();
});

$('#batchRejectBtn')?.addEventListener('click', async () => {
  const ids = Array.from(selectedIds);
  await api('/api/batch', {
    method: 'POST',
    body: JSON.stringify({ ids, action: 'reject' })
  });

  selectedIds.clear();
  toast(`Rejected ${ids.length} rows`, 'info');
  await loadData();
});

// --- Filters ---
$('#filterStatus')?.addEventListener('change', () => { resetPagination(); renderTable(); });
$('#filterIntent')?.addEventListener('change', () => { resetPagination(); renderTable(); });
$('#filterSearch')?.addEventListener('input', debounce(() => { resetPagination(); renderTable(); }, 200));

// --- Pagination events ---
$('#pageSizeSelect')?.addEventListener('change', () => {
  resetPagination();
  renderTable();
});

document.addEventListener('click', (e) => {
  const btn = e.target.closest('.page-btn');
  if (!btn || btn.classList.contains('disabled')) return;
  const page = parseInt(btn.dataset.page, 10);
  if (isNaN(page)) return;
  currentPage = page;
  renderTable();
});

// --- Chart ---
async function loadChart() {
  const data = await api('/api/intent-distribution');
  if (!data || !data.labels || data.labels.length === 0) return;

  const ctx = document.getElementById('intentChart');
  if (!ctx) return;

  if (intentChart) {
    intentChart.destroy();
  }

  const colors = [
    '#6c5ce7', '#00b894', '#fdcb6e', '#e17055', '#0984e3',
    '#a29bfe', '#55efc4', '#ffeaa7', '#fab1a0', '#74b9ff',
    '#dfe6e9', '#fd79a8', '#00cec9', '#e84393', '#6c5ce7'
  ];

  intentChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        label: 'Số lượng',
        data: data.counts,
        backgroundColor: data.labels.map((_, i) => colors[i % colors.length]),
        borderColor: data.labels.map((_, i) => colors[i % colors.length]),
        borderWidth: 1,
        borderRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: function(context) {
              const total = data.total;
              const pct = ((context.parsed.y / total) * 100).toFixed(1);
              return `Tỉ lệ: ${pct}%`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b90a5', font: { size: 11 } }
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { color: '#8b90a5', stepSize: 1 }
        }
      }
    }
  });
}

// --- Export NLU ---
$('#exportBtn')?.addEventListener('click', async () => {
  $('#exportBtn').disabled = true;
  $('#exportBtn').textContent = 'Exporting...';

  const res = await api('/api/export-nlu', { method: 'POST' });

  $('#exportBtn').disabled = false;
  $('#exportBtn').textContent = 'Export to NLU';

  if (res.success) {
    const msg = `Exported ${res.total_examples} examples (${res.intents_added.length} intents)` +
                (res.backup_file ? `. Backup: ${res.backup_file}` : '') +
                (res.removed_from_csv ? `. Removed ${res.removed_from_csv} rows from CSV` : '');
    toast(msg, 'success');
    selectedIds.clear();
    await loadData();
    await loadChart();
  } else {
    toast(res.error || 'Export failed', 'error');
  }
});

// --- Export CSV ---
$('#exportCsvBtn')?.addEventListener('click', async () => {
  $('#exportCsvBtn').disabled = true;
  $('#exportCsvBtn').textContent = 'Exporting...';

  try {
    const res = await fetch('/api/export-csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!res.ok) {
      const err = await res.json();
      toast(err.error || 'Export CSV failed', 'error');
      return;
    }

    const blob = await res.blob();
    const disposition = res.headers.get('Content-disposition') || '';
    const match = disposition.match(/filename=(.+)/);
    const filename = match ? match[1] : 'reviewed_data.csv';

    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
    toast(`CSV exported: ${filename}`, 'success');
    selectedIds.clear();
    await loadData();
    await loadChart();
  } catch (e) {
    toast('Export CSV failed: ' + e.message, 'error');
  } finally {
    $('#exportCsvBtn').disabled = false;
    $('#exportCsvBtn').textContent = 'Export CSV';
  }
});

// --- Retrain ---
$('#retrainBtn')?.addEventListener('click', async () => {
  if ($('#retrainBtn').disabled) return;

  const res = await api('/api/retrain', { method: 'POST' });

  if (!res.success) {
    toast(res.error || 'Failed to start training', 'error');
    return;
  }

  toast('Training started!', 'info');
  $('#retrainBtn').disabled = true;
  $('#retrainBtn').textContent = 'Training...';
  $('#logConsole').innerHTML = '';
  startLogPolling();
});

function startLogPolling() {
  if (pollingInterval) clearInterval(pollingInterval);

  pollingInterval = setInterval(async () => {
    const status = await api('/api/train-status');

    const logEl = $('#logConsole');
    if (status.logs) {
      const lines = status.logs.split('\n').filter(l => l.trim());
      logEl.innerHTML = lines.map(l => `<div>${escapeHtml(l)}</div>`).join('');
      logEl.scrollTop = logEl.scrollHeight;
    }

    if (!status.running) {
      clearInterval(pollingInterval);
      pollingInterval = null;
      $('#retrainBtn').disabled = false;
      $('#retrainBtn').textContent = 'Retrain Model';

      if (status.error) {
        toast(`Training failed: ${status.error}`, 'error');
      } else if (status.metrics) {
        toast(`Training done! F1: ${status.metrics.f1_score.toFixed(4)}`, 'success');
      } else {
        toast('Training completed', 'info');
      }

      loadHistory();
    }
  }, 2000);
}

// --- History ---
async function loadHistory() {
  const history = await api('/api/train-history');
  const tbody = $('#historyBody');

  if (!history || history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="loading">No history yet</td></tr>';
    return;
  }

  tbody.innerHTML = history.slice().reverse().map(h => `
    <tr>
      <td>${h.timestamp}</td>
      <td>${h.duration_seconds}</td>
      <td style="color: var(--accent); font-weight: 600;">${h.f1_score?.toFixed(4) || '—'}</td>
      <td>${h.accuracy?.toFixed(4) || '—'}</td>
      <td>${h.precision?.toFixed(4) || '—'}</td>
      <td>${h.recall?.toFixed(4) || '—'}</td>
    </tr>
  `).join('');
}

// --- Clear log ---
$('#clearLogBtn')?.addEventListener('click', () => {
  $('#logConsole').innerHTML = '<span class="log-placeholder">Waiting for training...</span>';
});

// --- Init ---
async function init() {
  await loadData();
  await loadHistory();
  await loadChart();

  const status = await api('/api/train-status');
  if (status.running) {
    $('#retrainBtn').disabled = true;
    $('#retrainBtn').textContent = 'Training...';
    startLogPolling();
  }
}

document.addEventListener('DOMContentLoaded', init);
