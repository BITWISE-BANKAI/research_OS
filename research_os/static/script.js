/* ═══════════════════════════════════════════════════════
   RESEARCHOS — Frontend Logic (script.js)
   ═══════════════════════════════════════════════════════ */

'use strict';

// ── State ──────────────────────────────────────────────
const state = {
  currentPanel: 'search',
  chatMode: 'library',        // 'library' | 'single'
  indexedPapers: [],          // [{paper_id, title, authors, year, url}]
  chatHistory: [],            // [{role, content}]
  searchResults: [],          // latest search results
  selectedCompare: new Set(), // paper_ids selected for compare
  selectedDeep:    new Set(), // paper_ids selected for deep research
};

// ── Panel routing ───────────────────────────────────────
function switchPanel(name) {
  state.currentPanel = name;

  // Update nav buttons
  document.querySelectorAll('.mode-btn[id^="nav-"]').forEach(b => b.classList.remove('active'));
  const navBtn = document.getElementById('nav-' + name);
  if (navBtn) navBtn.classList.add('active');

  // Show/hide panels
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.remove('active');
    p.style.display = 'none';
  });
  const panel = document.getElementById('panel-' + name);
  if (panel) { panel.classList.add('active'); panel.style.display = 'flex'; }

  // Update topbar
  const titles = {
    search:  ['🔍 Search Papers',       'Discover research via Semantic Scholar'],
    chat:    ['💬 RAG Chat',            'Ask questions grounded in indexed evidence'],
    compare: ['⚖️ Compare Papers',      'Side-by-side paper analysis'],
    deep:    ['🧠 Deep Research',       'Iterative multi-phase research briefing'],
    mongo:   ['🗄️ JSON Memory',        'Inspect stored MongoDB JSON records & collections'],
  };
  if (titles[name]) {
    document.getElementById('topbar-title').textContent    = titles[name][0];
    document.getElementById('topbar-subtitle').textContent = titles[name][1];
  }

  // Populate selectors on relevant panels
  if (name === 'compare') populateCompareSelector();
  if (name === 'deep')    populateDeepSelector();
  if (name === 'chat')    populateSinglePaperSelect();
  if (name === 'mongo')   { loadMongoStats(); switchMongoTab(state.currentMongoTab || 'papers'); }
}


// ── Chat mode ───────────────────────────────────────────
function setChatMode(mode) {
  state.chatMode = mode;
  document.querySelectorAll('.mode-btn[id^="chat-mode-"]').forEach(b => b.classList.remove('active'));
  document.getElementById('chat-mode-' + mode).classList.add('active');

  const badge = document.getElementById('chat-mode-badge');
  const singleSel = document.getElementById('single-paper-selector');
  if (mode === 'library') {
    badge.textContent = '📚 LIBRARY MODE';
    singleSel.style.display = 'none';
  } else {
    badge.textContent = '📄 SINGLE PAPER MODE';
    singleSel.style.display = 'flex';
    populateSinglePaperSelect();
  }
}

function populateSinglePaperSelect() {
  const sel = document.getElementById('single-paper-select');
  sel.innerHTML = '<option value="">— Select paper —</option>';
  state.indexedPapers.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.paper_id;
    opt.textContent = p.title.length > 60 ? p.title.slice(0, 58) + '…' : p.title;
    sel.appendChild(opt);
  });
}

// ── Toast notifications ─────────────────────────────────
function toast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(24px)';
    el.style.transition = 'all 0.3s ease';
    setTimeout(() => el.remove(), 320);
  }, duration);
}

// ── Indexed papers ──────────────────────────────────────
async function loadIndexedPapers() {
  try {
    const res = await fetch('/api/papers');
    const data = await res.json();
    state.indexedPapers = data.papers || [];
    renderIndexedList();
  } catch (e) {
    console.error('Failed to load indexed papers:', e);
  }
}

function renderIndexedList() {
  const list = document.getElementById('indexed-list');
  if (state.indexedPapers.length === 0) {
    list.innerHTML = '<div class="empty-hint">No papers indexed yet.</div>';
    return;
  }
  list.innerHTML = state.indexedPapers.map(p => `
    <div class="indexed-item" title="${escHtml(p.title)}">
      <span class="item-title">${escHtml(p.title)}</span>
      <button class="delete-btn" title="Remove from index" onclick="deletePaper('${escHtml(p.paper_id)}', event)">✕</button>
    </div>
  `).join('');
}

async function deletePaper(paperId, event) {
  event.stopPropagation();
  try {
    const res = await fetch('/api/papers/' + encodeURIComponent(paperId), { method: 'DELETE' });
    if (res.ok) {
      state.indexedPapers = state.indexedPapers.filter(p => p.paper_id !== paperId);
      renderIndexedList();
      populateCompareSelector();
      populateDeepSelector();
      populateSinglePaperSelect();
      toast('Paper removed from index.', 'info');
    } else {
      toast('Failed to delete paper.', 'error');
    }
  } catch (e) {
    toast('Server error while deleting.', 'error');
  }
}

// ── Search ──────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && document.activeElement.id === 'search-q') doSearch();
});

async function doSearch() {
  const q = document.getElementById('search-q').value.trim();
  if (!q) { toast('Enter a search query.', 'error'); return; }

  const btn = document.getElementById('search-btn');
  const status = document.getElementById('search-status');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Searching…';
  status.textContent = 'Querying Semantic Scholar…';
  document.getElementById('results-grid').innerHTML = '';

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q }),
    });
    const data = await res.json();
    state.searchResults = data.papers || [];
    status.textContent = state.searchResults.length
      ? `Found ${state.searchResults.length} papers.`
      : (data.message || 'No results found.');
    renderSearchResults();
  } catch (e) {
    status.textContent = 'Search failed — is the server running?';
    toast('Search request failed.', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>🔍</span> Search';
  }
}

function renderSearchResults() {
  const grid = document.getElementById('results-grid');
  if (!state.searchResults.length) { grid.innerHTML = ''; return; }

  const indexedIds = new Set(state.indexedPapers.map(p => p.paper_id));

  grid.innerHTML = state.searchResults.map(p => {
    const isIndexed = indexedIds.has(p.paper_id);
    const authors = Array.isArray(p.authors) ? p.authors.slice(0, 3).join(', ') + (p.authors.length > 3 ? ' et al.' : '') : '';
    const fields  = Array.isArray(p.fields_of_study) ? p.fields_of_study.slice(0, 2) : [];
    return `
      <div class="paper-card" id="card-${escHtml(p.paper_id)}">
        ${isIndexed ? '<div class="badge badge-indexed" style="position:absolute;top:14px;right:14px;">✓ Indexed</div>' : ''}
        <div class="paper-title">${escHtml(p.title)}</div>
        <div class="paper-meta">
          ${p.year ? `<span class="badge badge-year">${p.year}</span>` : ''}
          <span class="badge badge-cite">📚 ${(p.citation_count || 0).toLocaleString()} citations</span>
          ${fields.map(f => `<span class="badge badge-field">${escHtml(f)}</span>`).join('')}
        </div>
        ${authors ? `<div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">${escHtml(authors)}</div>` : ''}
        <div class="paper-abstract">${escHtml(p.abstract || 'No abstract available.')}</div>
        <div class="paper-actions">
          ${!isIndexed ? `<button class="btn btn-primary btn-sm" id="idx-btn-${escHtml(p.paper_id)}" onclick="indexPaperById('${escHtml(p.paper_id)}')">⬇ Index Paper</button>` : '<button class="btn btn-glass btn-sm" disabled>✓ Already Indexed</button>'}
          ${p.url ? `<a href="${escHtml(p.url)}" target="_blank" class="btn btn-glass btn-sm">🔗 View on S2</a>` : ''}
          ${p.arxiv_id ? `<a href="https://arxiv.org/abs/${escHtml(p.arxiv_id)}" target="_blank" class="btn btn-glass btn-sm">📄 arXiv</a>` : ''}
        </div>
      </div>
    `;
  }).join('');
}

async function indexPaperById(paperId) {
  const paper = state.searchResults.find(p => p.paper_id === paperId) || { paper_id: paperId, title: paperId };
  await indexPaper(paper);
}

async function indexPaper(paper) {
  const pid = paper.paper_id;
  const btn = document.getElementById('idx-btn-' + pid);
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Indexing…'; }
  toast(`Indexing "${(paper.title || pid).slice(0, 50)}…" — this may take a moment.`, 'info', 6000);

  try {
    const res = await fetch('/api/index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_id: paper.paper_id, paper: paper }),
    });
    const data = await res.json();
    const msg = data.message || 'Done.';

    if (msg.toLowerCase().includes('error')) {
      toast(`⚠ ${msg}`, 'error', 7000);
      if (btn) { btn.disabled = false; btn.innerHTML = '⬇ Index Paper'; }
    } else {
      toast(`✓ ${msg}`, 'success');
      if (btn) { btn.outerHTML = '<button class="btn btn-glass btn-sm" disabled>✓ Indexed</button>'; }
      await loadIndexedPapers();
      populateCompareSelector();
      populateDeepSelector();
      populateSinglePaperSelect();
    }
  } catch (e) {
    toast('Indexing failed — server error.', 'error');
    if (btn) { btn.disabled = false; btn.innerHTML = '⬇ Index Paper'; }
  }
}

// ── Manual Paper Entry Modal ────────────────────────────
function openManualPaperModal() {
  const m = document.getElementById('manual-paper-modal');
  if (m) {
    m.style.display = 'flex';
    document.getElementById('manual-title').focus();
  }
}

function closeManualPaperModal() {
  const m = document.getElementById('manual-paper-modal');
  if (m) m.style.display = 'none';
}

async function submitManualPaper() {
  const title = document.getElementById('manual-title').value.trim();
  if (!title) {
    toast('Please enter a paper title.', 'error');
    document.getElementById('manual-title').focus();
    return;
  }

  const authorsRaw = document.getElementById('manual-authors').value.trim();
  const authors = authorsRaw ? authorsRaw.split(',').map(a => a.trim()).filter(Boolean) : ['Unknown Author'];
  const year = parseInt(document.getElementById('manual-year').value) || 2024;
  const abstract = document.getElementById('manual-abstract').value.trim() || 'No abstract provided.';
  const content = document.getElementById('manual-content').value.trim() || abstract;
  const url = document.getElementById('manual-url').value.trim() || '';

  const btn = document.getElementById('manual-submit-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Indexing…';
  toast('Ingesting and embedding manual paper…', 'info', 5000);

  try {
    const res = await fetch('/api/papers/manual', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, authors, year, abstract, content, url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Failed to index paper.');

    toast(`✓ ${data.message || 'Paper indexed successfully!'}`, 'success', 5000);
    closeManualPaperModal();

    // Reset inputs
    document.getElementById('manual-title').value = '';
    document.getElementById('manual-authors').value = '';
    document.getElementById('manual-abstract').value = '';
    document.getElementById('manual-content').value = '';
    document.getElementById('manual-url').value = '';

    await loadIndexedPapers();
    populateCompareSelector();
    populateDeepSelector();
    populateSinglePaperSelect();
  } catch (err) {
    toast(`Error: ${err.message}`, 'error', 6000);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '⬇ Index into Knowledge Base';
  }
}

// ── Chat ────────────────────────────────────────────────
function handleChatKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}


function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

function clearChat() {
  state.chatHistory = [];
  const msgs = document.getElementById('messages');
  msgs.innerHTML = `
    <div class="msg assistant">
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-bubble">Chat cleared. Ask me anything about your indexed papers!</div>
      </div>
    </div>`;
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  input.style.height = 'auto';
  input.disabled = true;
  document.getElementById('send-btn').disabled = true;

  // Add user message
  appendMessage('user', query);

  // Typing indicator
  const typingId = 'typing-' + Date.now();
  appendTyping(typingId);

  // Determine paper_ids filter
  let paperIds = null;
  if (state.chatMode === 'single') {
    const sel = document.getElementById('single-paper-select');
    if (sel.value) paperIds = [sel.value];
  }

  try {
    const body = { query, mode: state.chatMode, paper_ids: paperIds, k: 5 };
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    removeTyping(typingId);
    appendAssistantMessage(data);
  } catch (e) {
    removeTyping(typingId);
    appendMessage('assistant', '❌ Error: Could not reach the server. Is the backend running?');
    toast('Server connection failed.', 'error');
  } finally {
    input.disabled = false;
    document.getElementById('send-btn').disabled = false;
    input.focus();
  }
}

function appendMessage(role, content) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? '👤' : '🤖'}</div>
    <div class="msg-body">
    <div class="msg-bubble">${formatText(content)}</div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function appendAssistantMessage(data) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg assistant';

  // Build sources HTML
  let sourcesHtml = '';
  const sources = data.sources || [];
  if (sources.length > 0) {
    const chips = sources.map(s => {
      const link = s.arxiv_id
        ? `https://arxiv.org/abs/${s.arxiv_id}`
        : s.open_access_url || s.url || '#';
      const year = s.year ? ` (${s.year})` : '';
      return `<div class="source-chip">📄 <a href="${escHtml(link)}" target="_blank">${escHtml(s.title)}${year}</a></div>`;
    }).join('');
    sourcesHtml = `
      <div class="sources-block">
        <button class="sources-toggle" onclick="toggleSources(this)">▶ ${sources.length} Source${sources.length>1?'s':''}</button>
        <div class="sources-list" style="display:none;">${chips}</div>
      </div>`;
  }

  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-bubble prose-block">${renderMarkdown(data.answer || '(No answer generated)')}</div>
      ${sourcesHtml}
    </div>`;
  msgs.appendChild(div);
  // Render LaTeX in the newly inserted bubble
  const bubble = div.querySelector('.msg-bubble');
  if (bubble && window.renderMathInElement) {
    renderMathInElement(bubble, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$',  right: '$',  display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true }
      ],
      throwOnError: false
    });
  }
  msgs.scrollTop = msgs.scrollHeight;
}

function toggleSources(btn) {
  const list = btn.nextElementSibling;
  const shown = list.style.display !== 'none';
  list.style.display = shown ? 'none' : 'flex';
  btn.textContent = (shown ? '▶' : '▼') + btn.textContent.slice(1);
}

function appendTyping(id) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg assistant'; div.id = id;
  div.innerHTML = `
    <div class="msg-avatar">🤖</div>
    <div class="msg-body">
      <div class="msg-bubble">
        <div class="typing-bubble">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    </div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function removeTyping(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Compare ─────────────────────────────────────────────
function populateCompareSelector() {
  const container = document.getElementById('compare-selector');
  if (!state.indexedPapers.length) {
    container.innerHTML = '<div class="empty-hint">Index papers first to see them here.</div>';
    return;
  }
  container.innerHTML = state.indexedPapers.map(p => `
    <label class="selector-item ${state.selectedCompare.has(p.paper_id) ? 'selected' : ''}">
      <input type="checkbox" value="${escHtml(p.paper_id)}" ${state.selectedCompare.has(p.paper_id) ? 'checked' : ''}
        onchange="toggleCompareSelect('${escHtml(p.paper_id)}', this)" />
      <span>${escHtml(p.title.length > 65 ? p.title.slice(0, 63) + '…' : p.title)} ${p.year ? `(${p.year})` : ''}</span>
    </label>`).join('');
}

function toggleCompareSelect(pid, checkbox) {
  const label = checkbox.closest('.selector-item');
  if (checkbox.checked) { state.selectedCompare.add(pid); label.classList.add('selected'); }
  else                  { state.selectedCompare.delete(pid); label.classList.remove('selected'); }
}

async function doCompare() {
  const ids = [...state.selectedCompare];
  if (ids.length < 2) { toast('Select at least 2 papers to compare.', 'error'); return; }

  const btn = document.getElementById('compare-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Comparing…';

  try {
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_ids: ids }),
    });
    const data = await res.json();
    const resultBlock = document.getElementById('compare-result');
    const resultBody  = document.getElementById('compare-result-body');
    resultBody.textContent = data.comparison || 'No comparison generated.';
    resultBlock.style.display = 'block';
    resultBlock.scrollIntoView({ behavior: 'smooth' });
    toast('Comparison complete!', 'success');
  } catch (e) {
    toast('Comparison failed.', 'error');
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚖️</span> Compare Selected Papers';
  }
}

// ── Deep Research ───────────────────────────────────────
function populateDeepSelector() {
  const container = document.getElementById('deep-selector');
  if (!state.indexedPapers.length) {
    container.innerHTML = '<div class="empty-hint">Index papers first to select them here.</div>';
    return;
  }
  container.innerHTML = state.indexedPapers.map(p => `
    <label class="selector-item ${state.selectedDeep.has(p.paper_id) ? 'selected' : ''}">
      <input type="checkbox" value="${escHtml(p.paper_id)}" ${state.selectedDeep.has(p.paper_id) ? 'checked' : ''}
        onchange="toggleDeepSelect('${escHtml(p.paper_id)}', this)" />
      <span>${escHtml(p.title.length > 65 ? p.title.slice(0, 63) + '…' : p.title)} ${p.year ? `(${p.year})` : ''}</span>
    </label>`).join('');
}

function toggleDeepSelect(pid, checkbox) {
  const label = checkbox.closest('.selector-item');
  if (checkbox.checked) { state.selectedDeep.add(pid); label.classList.add('selected'); }
  else                  { state.selectedDeep.delete(pid); label.classList.remove('selected'); }
}

function setStep(stepId, status) {
  // status: '' | 'active' | 'done'
  const el = document.getElementById(stepId);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (status) el.classList.add(status);
}

async function doDeepResearch() {
  const query = document.getElementById('deep-query').value.trim();
  const ids    = [...state.selectedDeep];
  const maxIter = parseInt(document.getElementById('deep-max-iter').value);
  const thresh  = parseInt(document.getElementById('deep-threshold').value);

  if (!query) { toast('Enter a research topic.', 'error'); return; }
  if (!ids.length) { toast('Select at least one paper.', 'error'); return; }

  const btn = document.getElementById('deep-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Running…';

  // Show progress
  const progress = document.getElementById('deep-progress');
  progress.style.display = 'flex';
  document.getElementById('deep-result').style.display = 'none';
  ['step-retrieve','step-generate','step-evaluate','step-regen','step-final'].forEach(s => setStep(s, ''));
  setStep('step-retrieve', 'active');

  toast('Deep research started — this may take several minutes.', 'info', 8000);

  // Simulate step progression while waiting
  const stepTimers = [
    setTimeout(() => { setStep('step-retrieve', 'done'); setStep('step-generate', 'active'); }, 3000),
    setTimeout(() => { setStep('step-generate', 'done'); setStep('step-evaluate', 'active'); }, 15000),
    setTimeout(() => { setStep('step-evaluate', 'done'); setStep('step-regen', 'active'); }, 30000),
    setTimeout(() => { setStep('step-regen', 'done'); setStep('step-final', 'active'); }, 50000),
  ];

  try {
    const res = await fetch('/api/deep-research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query, paper_ids: ids,
        max_iterations: maxIter,
        score_threshold: thresh,
      }),
    });
    stepTimers.forEach(clearTimeout);
    ['step-retrieve','step-generate','step-evaluate','step-regen'].forEach(s => setStep(s, 'done'));
    setStep('step-final', 'done');

    const data = await res.json();

    // Score badge
    const score = data.final_score;
    const scoreBadge = document.getElementById('deep-score-badge');
    if (score !== null && score !== undefined) {
      const cls = score >= 8 ? 'score-high' : score >= 5 ? 'score-mid' : 'score-low';
      scoreBadge.innerHTML = `<span class="score-badge ${cls}">Score: ${score}/10</span>`;
    } else {
      scoreBadge.innerHTML = '';
    }

    const resultBody = document.getElementById('deep-result-body');
    resultBody.textContent = data.summary || 'No briefing generated.';
    document.getElementById('deep-result').style.display = 'block';
    document.getElementById('deep-result').scrollIntoView({ behavior: 'smooth' });
    toast(`Deep research complete! (${data.iterations_run || '?'} iteration(s), status: ${data.status || 'done'})`, 'success', 6000);
  } catch (e) {
    stepTimers.forEach(clearTimeout);
    toast('Deep research failed — server error.', 'error');
    console.error(e);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>🧠</span> Generate Briefing';
  }
}

// ── MongoDB JSON Storage Explorer ────────────────────────
let currentModalData = null;
let currentModalFilename = 'document.json';

async function updateMongoStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const badge = document.getElementById('mongo-status-badge');
    const text = document.getElementById('mongo-status-text');
    const dot = document.getElementById('mongo-dot');
    
    if (data.mongo_connected) {
      badge.style.background = 'rgba(16,185,129,0.15)';
      badge.style.border = '1px solid rgba(16,185,129,0.3)';
      badge.style.color = '#34d399';
      dot.style.background = '#10b981';
      text.textContent = 'MongoDB: Connected';
    } else {
      badge.style.background = 'rgba(245,158,11,0.15)';
      badge.style.border = '1px solid rgba(245,158,11,0.3)';
      badge.style.color = '#fbbf24';
      dot.style.background = '#f59e0b';
      text.textContent = 'JSON Store: Local Backup';
    }
  } catch (e) {
    console.warn('Health check failed', e);
  }
}

async function loadMongoStats() {
  try {
    const res = await fetch('/api/mongo/stats');
    const stats = await res.json();
    document.getElementById('stat-papers').textContent = stats.papers_count ?? 0;
    document.getElementById('stat-chunks').textContent = stats.chunks_count ?? 0;
    document.getElementById('stat-briefings').textContent = stats.briefings_count ?? 0;
    document.getElementById('stat-chats').textContent = stats.chat_count ?? 0;
  } catch (e) {
    console.error('Failed to load MongoDB stats:', e);
  }
}

async function switchMongoTab(tab) {
  state.currentMongoTab = tab;
  ['papers','briefings','history','comparisons'].forEach(t => {
    const btn = document.getElementById('tab-mongo-' + t);
    if (btn) {
      btn.className = (t === tab) ? 'btn btn-primary btn-sm' : 'btn btn-glass btn-sm';
    }
  });

  const titleEl = document.getElementById('mongo-content-title');
  const bodyEl = document.getElementById('mongo-content-body');
  bodyEl.innerHTML = '<div class="empty-hint"><span class="spinner"></span> Loading ' + tab + ' records…</div>';

  try {
    if (tab === 'papers') {
      titleEl.textContent = '📄 Paper Metadata JSON Collection';
      const res = await fetch('/api/papers');
      const data = await res.json();
      const papers = data.papers || [];
      if (!papers.length) {
        bodyEl.innerHTML = '<div class="empty-hint">No papers stored in MongoDB yet. Use the Search tab to index papers.</div>';
        return;
      }
      bodyEl.innerHTML = papers.map(p => `
        <div class="result-card" style="background:rgba(15,20,45,0.4);border:1px solid var(--glass-border);padding:14px;border-radius:10px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div>
              <div style="font-weight:600;font-size:14px;color:var(--text-primary);margin-bottom:4px;">${escHtml(p.title)}</div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px;">ID: <code>${escHtml(p.paper_id)}</code> · Year: ${p.year || 'N/A'} · Citations: ${(p.citation_count || 0).toLocaleString()}</div>
            </div>
            <div style="display:flex;gap:6px;flex-shrink:0;">
              <button class="btn btn-glass btn-sm" onclick="viewPaperChunks('${escHtml(p.paper_id)}', '${escHtml(p.title)}')">🧩 View Chunks</button>
              <button class="btn btn-glass btn-sm" onclick='showJsonModal("Paper: ${escHtml(p.title)}", ${JSON.stringify(p)}, "${escHtml(p.paper_id)}.json")'>🔍 Raw JSON</button>
              <button class="btn btn-glass btn-sm" onclick='downloadJson("${escHtml(p.paper_id)}.json", ${JSON.stringify(p)})'>⬇ Export</button>
            </div>
          </div>
        </div>
      `).join('');
    } else if (tab === 'briefings') {
      titleEl.textContent = '🧠 Deep Research Briefings JSON Collection';
      const res = await fetch('/api/mongo/briefings');
      const data = await res.json();
      const briefings = data.briefings || [];
      if (!briefings.length) {
        bodyEl.innerHTML = '<div class="empty-hint">No research briefings generated yet. Run a Deep Research task first.</div>';
        return;
      }
      bodyEl.innerHTML = briefings.map((b, i) => `
        <div class="result-card" style="background:rgba(15,20,45,0.4);border:1px solid var(--glass-border);padding:14px;border-radius:10px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;">
              <div style="font-weight:600;font-size:14px;color:var(--text-primary);margin-bottom:4px;">Topic: "${escHtml(b.query)}"</div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:8px;">
                Date: ${b.created_at || 'N/A'} · Score: <strong>${b.final_score ?? 'N/A'}/10</strong> · Iterations: ${b.iterations_run || 1} · Status: <code>${escHtml(b.status || 'done')}</code>
              </div>
              <div style="font-size:12px;color:var(--text-secondary);max-height:80px;overflow:hidden;text-overflow:ellipsis;">
                ${escHtml((b.summary || '').slice(0, 260))}…
              </div>
            </div>
            <div style="display:flex;gap:6px;flex-shrink:0;">
              <button class="btn btn-glass btn-sm" onclick='showJsonModal("Briefing: ${escHtml(b.query)}", ${JSON.stringify(b)}, "briefing_${i+1}.json")'>🔍 Raw JSON</button>
              <button class="btn btn-glass btn-sm" onclick='downloadJson("briefing_${i+1}.json", ${JSON.stringify(b)})'>⬇ Export</button>
            </div>
          </div>
        </div>
      `).join('');
    } else if (tab === 'history') {
      titleEl.textContent = '💬 Grounded QA Chat Logs JSON Collection';
      const res = await fetch('/api/mongo/history');
      const data = await res.json();
      const history = data.history || [];
      if (!history.length) {
        bodyEl.innerHTML = '<div class="empty-hint">No chat interactions logged yet. Ask questions in RAG Chat first.</div>';
        return;
      }
      bodyEl.innerHTML = history.map((h, i) => `
        <div class="result-card" style="background:rgba(15,20,45,0.4);border:1px solid var(--glass-border);padding:14px;border-radius:10px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;">
              <div style="font-weight:600;font-size:13px;color:var(--accent-cyan);margin-bottom:4px;">Q: ${escHtml(h.query)}</div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;">Time: ${h.created_at || 'N/A'} · Mode: ${escHtml(h.mode || 'library')} · Status: ${escHtml(h.status || 'success')}</div>
              <div style="font-size:12px;color:var(--text-secondary);max-height:70px;overflow:hidden;">${escHtml((h.answer || '').slice(0, 220))}…</div>
            </div>
            <div style="display:flex;gap:6px;flex-shrink:0;">
              <button class="btn btn-glass btn-sm" onclick='showJsonModal("QA Log #${i+1}", ${JSON.stringify(h)}, "chat_${i+1}.json")'>🔍 Raw JSON</button>
              <button class="btn btn-glass btn-sm" onclick='downloadJson("chat_${i+1}.json", ${JSON.stringify(h)})'>⬇ Export</button>
            </div>
          </div>
        </div>
      `).join('');
    } else if (tab === 'comparisons') {
      titleEl.textContent = '⚖️ Multi-Paper Comparisons JSON Collection';
      const res = await fetch('/api/mongo/comparisons');
      const data = await res.json();
      const comps = data.comparisons || [];
      if (!comps.length) {
        bodyEl.innerHTML = '<div class="empty-hint">No comparisons stored yet. Run Compare Papers to save one.</div>';
        return;
      }
      bodyEl.innerHTML = comps.map((c, i) => `
        <div class="result-card" style="background:rgba(15,20,45,0.4);border:1px solid var(--glass-border);padding:14px;border-radius:10px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">
            <div style="flex:1;">
              <div style="font-weight:600;font-size:13px;color:var(--text-primary);margin-bottom:4px;">Compared Papers: ${(c.paper_ids || []).length} papers</div>
              <div style="font-size:12px;color:var(--text-muted);margin-bottom:6px;">Date: ${c.created_at || 'N/A'}</div>
              <div style="font-size:12px;color:var(--text-secondary);max-height:70px;overflow:hidden;">${escHtml((c.comparison || '').slice(0, 220))}…</div>
            </div>
            <div style="display:flex;gap:6px;flex-shrink:0;">
              <button class="btn btn-glass btn-sm" onclick='showJsonModal("Comparison #${i+1}", ${JSON.stringify(c)}, "comparison_${i+1}.json")'>🔍 Raw JSON</button>
              <button class="btn btn-glass btn-sm" onclick='downloadJson("comparison_${i+1}.json", ${JSON.stringify(c)})'>⬇ Export</button>
            </div>
          </div>
        </div>
      `).join('');
    }
  } catch (e) {
    bodyEl.innerHTML = `<div class="empty-hint" style="color:var(--accent-rose)">Error loading ${tab}: ${e.message}</div>`;
  }
}

async function viewPaperChunks(paperId, paperTitle) {
  try {
    const res = await fetch('/api/mongo/chunks/' + encodeURIComponent(paperId));
    const data = await res.json();
    showJsonModal(`Chunks for: ${paperTitle} (${data.chunks?.length || 0} chunks)`, data, `${paperId}_chunks.json`);
  } catch (e) {
    toast('Failed to load chunks: ' + e.message, 'error');
  }
}

function refreshMongoView() {
  loadMongoStats();
  switchMongoTab(state.currentMongoTab || 'papers');
  updateMongoStatus();
}

function showJsonModal(title, obj, filename = 'document.json') {
  currentModalData = obj;
  currentModalFilename = filename;
  document.getElementById('json-modal-title').textContent = title;
  document.getElementById('json-modal-content').textContent = JSON.stringify(obj, null, 2);
  const modal = document.getElementById('json-modal');
  modal.style.display = 'flex';
}

function closeJsonModal() {
  document.getElementById('json-modal').style.display = 'none';
}

function downloadModalJson() {
  if (currentModalData) {
    downloadJson(currentModalFilename, currentModalData);
  }
}

function downloadJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  toast(`Downloaded ${filename}`, 'success');
}

// ── Utilities ────────────────────────────────────────────
function escHtml(str) {
  if (typeof str !== 'string') return '';
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

function renderMarkdown(text) {
  if (!text) return '';

  // Pre-process: protect LaTeX blocks from Markdown escaping
  // Replace $$ and $ expressions temporarily
  const mathBlocks = [];
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, m) => {
    mathBlocks.push({ display: true, math: m });
    return `%%MATH${mathBlocks.length - 1}%%`;
  });
  text = text.replace(/\$([^\$\n]+?)\$/g, (_, m) => {
    mathBlocks.push({ display: false, math: m });
    return `%%MATH${mathBlocks.length - 1}%%`;
  });

  // Configure marked
  if (window.marked) {
    marked.setOptions({ breaks: true, gfm: true });
  }

  // Render markdown
  let html = window.marked ? marked.parse(text) : text.replace(/\n/g, '<br/>');

  // Restore math blocks as KaTeX-ready delimiters
  html = html.replace(/%%MATH(\d+)%%/g, (_, idx) => {
    const b = mathBlocks[parseInt(idx)];
    return b.display ? `$$${b.math}$$` : `$${b.math}$`;
  });

  return html;
}

function formatText(text) {
  if (!text) return '';
  // Lightweight fallback for non-assistant messages (user bubbles)
  return escHtml(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,     '<em>$1</em>')
    .replace(/`(.+?)`/g,       '<code>$1</code>')
    .replace(/\n/g,            '<br/>');
}

function copyResult(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  navigator.clipboard.writeText(el.textContent)
    .then(() => toast('Copied to clipboard!', 'success'))
    .catch(() => toast('Copy failed.', 'error'));
}

// ── Init ─────────────────────────────────────────────────
(async function init() {
  await updateMongoStatus();
  await loadIndexedPapers();
  switchPanel('search');
})();

