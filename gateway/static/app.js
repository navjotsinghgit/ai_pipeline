
'use strict';


const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewArea = document.getElementById('preview-area');
const previewImg = document.getElementById('preview-img');
const previewMeta = document.getElementById('preview-meta');
const clearBtn = document.getElementById('clear-btn');
const analyzeBtn = document.getElementById('analyze-btn');
const progressCont = document.getElementById('progress-container');
const progressLabel = document.getElementById('progress-label');
const progressFill = document.getElementById('progress-fill');
const resultsSection = document.getElementById('results-section');
const statsRow = document.getElementById('stats-row');
const resultImg = document.getElementById('result-img');
const downloadLink = document.getElementById('download-link');
const groupsGrid = document.getElementById('groups-grid');
const jsonBlock = document.getElementById('json-block');
const toggleJson = document.getElementById('toggle-json');
const errorCard = document.getElementById('error-card');
const errorMsg = document.getElementById('error-msg');

const steps = {
  upload: document.getElementById('step-upload'),
  detect: document.getElementById('step-detect'),
  group: document.getElementById('step-group'),
  viz: document.getElementById('step-viz'),
};

let currentFile = null;

// drop zone
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) loadFile(fileInput.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith('image/')) { showError('Please upload a valid image file.'); return; }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewMeta.textContent = `${file.name} — ${(file.size / 1024).toFixed(1)} KB`;
    dropZone.style.display = 'none';
    previewArea.style.display = 'block';
    hideError();
    resetResults();
  };
  reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
  currentFile = null; fileInput.value = '';
  previewArea.style.display = 'none';
  dropZone.style.display = 'flex';
  resetResults();
  hideError();
  setStep('upload');
});


analyzeBtn.addEventListener('click', runPipeline);

async function runPipeline() {
  if (!currentFile) return;

  analyzeBtn.disabled = true;
  hideError();
  resetResults();
  showProgress('Uploading image…', 10);
  setStep('detect');

  const formData = new FormData();
  formData.append('image', currentFile);

  try {
    animateProgress(10, 40, 1200, 'Running detection…');
    setStep('detect');

    const res = await fetch('/analyze', { method: 'POST', body: formData });
    animateProgress(40, 75, 800, 'Grouping products…');
    setStep('group');

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    animateProgress(75, 100, 600, 'Generating visualization…');
    setStep('viz');

    await sleep(700);
    hideProgress();
    renderResults(data);

  } catch (err) {
    hideProgress();
    showError(err.message || 'Unknown error occurred.');
    setStep('upload');
  } finally {
    analyzeBtn.disabled = false;
  }
}


function renderResults(data) {
  resultsSection.style.display = 'block';
  resultsSection.classList.add('fade-in');

  // stats
  statsRow.innerHTML = '';
  const stats = [
    { value: data.total_products_detected, label: 'Products Detected' },
    { value: data.total_brand_groups, label: 'Brand Groups' },
    { value: `${data.processing_time_ms}ms`, label: 'Total Latency' },
    { value: `${data.detection_time_ms}ms`, label: 'Detection Time' },
  ];
  stats.forEach(s => {
    statsRow.insertAdjacentHTML('beforeend', `
      <div class="stat-card">
        <div class="stat-value">${s.value}</div>
        <div class="stat-label">${s.label}</div>
      </div>`);
  });

  // annotated image
  if (data.visualization_url) {
    resultImg.src = data.visualization_url + '?t=' + Date.now();
    downloadLink.href = data.visualization_url;
    downloadLink.style.display = '';
  }

  // brand groups
  groupsGrid.innerHTML = '';
  const groups = data.brand_groups || {};
  Object.entries(groups).sort().forEach(([gid, info]) => {
    const [b, g, r] = info.color;
    const hex = rgbToHex(r, g, b);
    groupsGrid.insertAdjacentHTML('beforeend', `
      <div class="group-chip">
        <div class="group-swatch" style="background:${hex}"></div>
        <span class="group-name">${gid}</span>
        <span class="group-count">${info.count} item${info.count !== 1 ? 's' : ''}</span>
      </div>`);
  });

  // JSON
  jsonBlock.textContent = JSON.stringify(data, null, 2);
}

toggleJson.addEventListener('click', () => {
  const hidden = jsonBlock.style.display === 'none';
  jsonBlock.style.display = hidden ? 'block' : 'none';
  toggleJson.textContent = hidden ? 'Hide JSON' : 'Show JSON';
});

//progress helpers
function showProgress(label, pct) {
  progressCont.style.display = 'flex';
  progressLabel.textContent = label;
  progressFill.style.width = pct + '%';
}
function hideProgress() { progressCont.style.display = 'none'; }

function animateProgress(from, to, durationMs, label) {
  progressLabel.textContent = label;
  const start = performance.now();
  function frame(now) {
    const t = Math.min((now - start) / durationMs, 1);
    progressFill.style.width = (from + (to - from) * easeOut(t)) + '%';
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
function easeOut(t) { return 1 - Math.pow(1 - t, 2); }

//pipeline step indicator
function setStep(active) {
  const order = ['upload', 'detect', 'group', 'viz'];
  const idx = order.indexOf(active);
  order.forEach((k, i) => {
    steps[k].classList.remove('active', 'done');
    if (i < idx) steps[k].classList.add('done');
    else if (i === idx) steps[k].classList.add('active');
  });
}

//error & reset
function showError(msg) {
  errorCard.style.display = 'flex';
  errorMsg.textContent = msg;
}
function hideError() { errorCard.style.display = 'none'; }

function resetResults() {
  resultsSection.style.display = 'none';
  statsRow.innerHTML = '';
  groupsGrid.innerHTML = '';
  jsonBlock.textContent = '';
  jsonBlock.style.display = 'none';
  toggleJson.textContent = 'Show JSON';
  resultImg.src = '';
}

//animated background (particle dots)
(function initCanvas() {
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  const particles = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  for (let i = 0; i < 55; i++) {
    particles.push({
      x: Math.random() * window.innerWidth,
      y: Math.random() * window.innerHeight,
      r: Math.random() * 1.5 + 0.5,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.5 + 0.1,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0) p.x = canvas.width;
      if (p.x > canvas.width) p.x = 0;
      if (p.y < 0) p.y = canvas.height;
      if (p.y > canvas.height) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(99,102,241,${p.alpha})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

//utilities
function rgbToHex(r, g, b) {
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
