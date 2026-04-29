// ============================================================
// K-Space Motion Demo
// ------------------------------------------------------------
// Three-panel real-time visualization:
//   1. Source image with scanning line overlay
//   2. K-space being progressively filled
//   3. Inverse-FFT reconstruction from current k-space
//
// Architecture notes:
//   - All math runs on Float64Arrays for speed
//   - Image is fixed at N=128 (power of 2, fast FFT)
//   - Each k-space "line" is a row in the 2D FFT
//   - Motion is simulated by transforming the source image
//     and re-FFTing it, then extracting the relevant row
//
// HOOK FOR FUTURE WORK:
//   - perLineMetadata[] stores motion-state per line, ready for
//     a future "retroactive correction" feature that filters out
//     corrupted lines and reconstructs from clean subset only.
// ============================================================

const N = 256;                      // image dimensions (must be power of 2)
const N2 = N * N;
const CENTER_FRACTION = 0.15;       // fraction of k-space center sampled sequentially first

// ------------------------------------------------------------
// State
// ------------------------------------------------------------
const state = {
  sourceImage: null,                // Float64Array, length N*N, values 0..1
  sourceFFT: null,                  // {re, im} of full source FFT (motion-free)
  currentFFT: null,                 // {re, im} of currently-transformed source (during motion)
  kspace: { re: new Float64Array(N2), im: new Float64Array(N2) },
  kspaceMask: new Uint8Array(N2),   // 1 = filled, 0 = empty
  sampleOrder: [],                  // array of row indices in sampling order
  perLineMetadata: [],              // {row, motionActive, dx, dy, rot} per acquired line
  currentStep: 0,
  isScanning: false,
  isPaused: false,
  motionActive: false,
  motionStartStep: -1,
  motionState: { dx: 0, dy: 0, rot: 0 },  // current cumulative motion
  speed: 6,                         // lines per animation frame tick
  reconstructionDirty: false,
};

// ------------------------------------------------------------
// DOM refs
// ------------------------------------------------------------
const $ = id => document.getElementById(id);
const sourceCanvas = $('sourceCanvas');
const scanOverlay = $('scanOverlay');
const kspaceCanvas = $('kspaceCanvas');
const reconCanvas = $('reconCanvas');
const sourceCtx = sourceCanvas.getContext('2d');
const scanCtx = scanOverlay.getContext('2d');
const kspaceCtx = kspaceCanvas.getContext('2d');
const reconCtx = reconCanvas.getContext('2d');

// ============================================================
// FFT — Cooley-Tukey radix-2, in-place, complex
// ============================================================
function fft1d(re, im, inverse) {
  const n = re.length;
  // bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // butterflies
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (inverse ? 2 : -2) * Math.PI / len;
    const wRe0 = Math.cos(ang);
    const wIm0 = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let wRe = 1, wIm = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const tRe = wRe * re[b] - wIm * im[b];
        const tIm = wRe * im[b] + wIm * re[b];
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const nwRe = wRe * wRe0 - wIm * wIm0;
        wIm = wRe * wIm0 + wIm * wRe0;
        wRe = nwRe;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
}

// 2D FFT: rows then columns. Modifies in place.
function fft2d(re, im, inverse) {
  const rowRe = new Float64Array(N);
  const rowIm = new Float64Array(N);
  // rows
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) { rowRe[c] = re[r * N + c]; rowIm[c] = im[r * N + c]; }
    fft1d(rowRe, rowIm, inverse);
    for (let c = 0; c < N; c++) { re[r * N + c] = rowRe[c]; im[r * N + c] = rowIm[c]; }
  }
  // columns
  const colRe = new Float64Array(N);
  const colIm = new Float64Array(N);
  for (let c = 0; c < N; c++) {
    for (let r = 0; r < N; r++) { colRe[r] = re[r * N + c]; colIm[r] = im[r * N + c]; }
    fft1d(colRe, colIm, inverse);
    for (let r = 0; r < N; r++) { re[r * N + c] = colRe[r]; im[r * N + c] = colIm[r]; }
  }
}

// fftshift — center the zero-frequency for display
function fftshiftIndex(i) {
  return (i + N / 2) % N;
}

// ============================================================
// Image loading & preprocessing
// ============================================================
function imageToFloatArray(img) {
  // draw to offscreen canvas at NxN, convert to grayscale Float64
  const c = document.createElement('canvas');
  c.width = N; c.height = N;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, N, N);
  // fit image preserving aspect ratio
  const scale = Math.min(N / img.width, N / img.height);
  const w = img.width * scale;
  const h = img.height * scale;
  ctx.drawImage(img, (N - w) / 2, (N - h) / 2, w, h);
  const data = ctx.getImageData(0, 0, N, N).data;
  const out = new Float64Array(N2);
  for (let i = 0; i < N2; i++) {
    // luminance
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    out[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  }
  return out;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// Generate a fallback procedural "phantom" if Demo.png isn't available
function generatePhantom() {
  const arr = new Float64Array(N2);
  const cx = N / 2, cy = N / 2;
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const dx = c - cx, dy = r - cy;
      const d = Math.sqrt(dx * dx + dy * dy);
      let v = 0;
      // outer ellipse (skull-like)
      if (d < 56 && (dx * dx) / (52 * 52) + (dy * dy) / (56 * 56) < 1) v = 0.85;
      // inner regions
      if (d < 50 && (dx * dx) / (46 * 46) + (dy * dy) / (50 * 50) < 1) v = 0.55;
      // ventricles
      const v1 = ((dx + 6) ** 2) / 36 + ((dy - 4) ** 2) / 100;
      const v2 = ((dx - 6) ** 2) / 36 + ((dy - 4) ** 2) / 100;
      if (v1 < 1 || v2 < 1) v = 0.25;
      // a few bright spots for texture
      if (((dx + 18) ** 2 + (dy + 12) ** 2) < 16) v = 0.95;
      if (((dx - 22) ** 2 + (dy + 18) ** 2) < 12) v = 0.9;
      if (((dx) ** 2 + (dy + 22) ** 2) < 18) v = 0.7;
      arr[r * N + c] = v;
    }
  }
  return arr;
}

// ============================================================
// Motion: transform image by translation + rotation
// ============================================================
function transformImage(src, dx, dy, rotDeg) {
  const out = new Float64Array(N2);
  const rad = rotDeg * Math.PI / 180;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);
  const cx = N / 2, cy = N / 2;
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      // inverse map: where in source did this output pixel come from?
      const x = c - cx - dx;
      const y = r - cy - dy;
      const sx = cosA * x + sinA * y + cx;
      const sy = -sinA * x + cosA * y + cy;
      // bilinear sample
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const x1 = x0 + 1, y1 = y0 + 1;
      if (x0 < 0 || x1 >= N || y0 < 0 || y1 >= N) { out[r * N + c] = 0; continue; }
      const fx = sx - x0, fy = sy - y0;
      const a = src[y0 * N + x0], b = src[y0 * N + x1];
      const c2 = src[y1 * N + x0], d = src[y1 * N + x1];
      out[r * N + c] = (a * (1 - fx) + b * fx) * (1 - fy) + (c2 * (1 - fx) + d * fx) * fy;
    }
  }
  return out;
}

function computeFFT(image) {
  const re = new Float64Array(image);  // copy
  const im = new Float64Array(N2);
  fft2d(re, im, false);
  return { re, im };
}

// ============================================================
// Sampling order (pseudo-random, paper-style)
// ============================================================
function buildSampleOrder() {
  const order = [];
  const taken = new Set();
  const centerCount = Math.round(N * CENTER_FRACTION);
  const halfCenter = Math.floor(centerCount / 2);
  // sequential center
  for (let i = N / 2 - halfCenter; i < N / 2 - halfCenter + centerCount; i++) {
    order.push(i);
    taken.add(i);
  }
  // Gaussian-distributed remaining
  while (order.length < N) {
    const u1 = Math.random();
    const u2 = Math.random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    const idx = Math.round(N / 2 + z * (N / 4));
    if (idx >= 0 && idx < N && !taken.has(idx)) {
      order.push(idx);
      taken.add(idx);
    }
  }
  return order;
}

// ============================================================
// Acquire one PE line
// ============================================================
function acquireLine(rowIdx) {
  // If motion just changed, recompute the source FFT for the current pose
  let fft;
  if (state.motionActive) {
    // small random walk on motion params (paper-style: ±5px / ±5deg per line)
    state.motionState.dx += (Math.random() - 0.5) * 1.2;
    state.motionState.dy += (Math.random() - 0.5) * 1.2;
    state.motionState.rot += (Math.random() - 0.5) * 1.2;
    // clamp
    state.motionState.dx = Math.max(-8, Math.min(8, state.motionState.dx));
    state.motionState.dy = Math.max(-8, Math.min(8, state.motionState.dy));
    state.motionState.rot = Math.max(-8, Math.min(8, state.motionState.rot));
    const moved = transformImage(state.sourceImage,
      state.motionState.dx, state.motionState.dy, state.motionState.rot);
    fft = computeFFT(moved);
  } else {
    fft = state.sourceFFT;
  }

  // Copy this row from fft into kspace
  for (let c = 0; c < N; c++) {
    state.kspace.re[rowIdx * N + c] = fft.re[rowIdx * N + c];
    state.kspace.im[rowIdx * N + c] = fft.im[rowIdx * N + c];
    state.kspaceMask[rowIdx * N + c] = 1;
  }

  // Record per-line metadata for future correction logic
  state.perLineMetadata.push({
    row: rowIdx,
    step: state.currentStep,
    motionActive: state.motionActive,
    dx: state.motionState.dx,
    dy: state.motionState.dy,
    rot: state.motionState.rot,
  });
}

// ============================================================
// Reconstruction (inverse FFT of partial k-space)
// ============================================================
function reconstruct() {
  const re = new Float64Array(state.kspace.re);
  const im = new Float64Array(state.kspace.im);
  fft2d(re, im, true);
  // magnitude
  const mag = new Float64Array(N2);
  let max = 0;
  for (let i = 0; i < N2; i++) {
    mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    if (mag[i] > max) max = mag[i];
  }
  return { mag, max };
}

// ============================================================
// HOOK: Future retroactive correction would live here
// ------------------------------------------------------------
// Given perLineMetadata and the full k-space, identify which
// rows were acquired during motion and reconstruct using only
// the clean subset (compressed-sensing style).
//
// function reconstructCleanOnly() {
//   // 1. Build a mask zeroing out corrupted rows
//   // 2. Optional: run a CS reconstruction (split-Bregman, etc.)
//   //    instead of plain zero-filled IFFT
//   // 3. Render to reconCanvas with a "FILTERED" tag
// }
// ============================================================

// ============================================================
// Rendering
// ============================================================
function renderSource() {
  // Render the (possibly motion-displaced) source as it currently appears
  const img = state.motionActive
    ? transformImage(state.sourceImage, state.motionState.dx, state.motionState.dy, state.motionState.rot)
    : state.sourceImage;
  const imgData = sourceCtx.createImageData(N, N);
  for (let i = 0; i < N2; i++) {
    const v = Math.max(0, Math.min(255, Math.round(img[i] * 255)));
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  sourceCtx.putImageData(imgData, 0, 0);
}

function renderScanOverlay(currentRow) {
  scanCtx.clearRect(0, 0, N, N);
  if (currentRow === null) return;
  // Bright scan line + faint grid of acquired lines
  scanCtx.fillStyle = 'rgba(255, 87, 34, 0.08)';
  for (const meta of state.perLineMetadata) {
    scanCtx.fillRect(0, meta.row, N, 1);
  }
  // Active scan line — bright orange with glow
  scanCtx.fillStyle = 'rgba(255, 87, 34, 0.9)';
  scanCtx.fillRect(0, currentRow, N, 1);
  scanCtx.fillStyle = 'rgba(255, 87, 34, 0.3)';
  scanCtx.fillRect(0, currentRow - 1, N, 3);
}

function renderKspace() {
  // Display log-magnitude of k-space, fftshifted to center
  const imgData = kspaceCtx.createImageData(N, N);
  // Find max for normalization (only over filled cells)
  let maxLog = 0;
  for (let i = 0; i < N2; i++) {
    if (state.kspaceMask[i]) {
      const m = Math.log(1 + Math.sqrt(state.kspace.re[i] ** 2 + state.kspace.im[i] ** 2));
      if (m > maxLog) maxLog = m;
    }
  }
  if (maxLog === 0) maxLog = 1;

  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      // fftshift: map display (r,c) to data (sr, sc)
      const sr = fftshiftIndex(r);
      const sc = fftshiftIndex(c);
      const idx = sr * N + sc;
      const dispIdx = (r * N + c) * 4;
      if (!state.kspaceMask[idx]) {
        imgData.data[dispIdx] = 8;
        imgData.data[dispIdx + 1] = 8;
        imgData.data[dispIdx + 2] = 12;
        imgData.data[dispIdx + 3] = 255;
        continue;
      }
      const mag = Math.sqrt(state.kspace.re[idx] ** 2 + state.kspace.im[idx] ** 2);
      const v = Math.log(1 + mag) / maxLog;
      // Heat map: black -> deep orange -> bright orange -> white
      const t = Math.min(1, v);
      const rCol = Math.min(255, Math.round(t * 255 * 1.4));
      const gCol = Math.min(255, Math.round(Math.max(0, t - 0.3) * 255 * 1.2));
      const bCol = Math.min(255, Math.round(Math.max(0, t - 0.7) * 255 * 2));
      imgData.data[dispIdx] = rCol;
      imgData.data[dispIdx + 1] = gCol;
      imgData.data[dispIdx + 2] = bCol;
      imgData.data[dispIdx + 3] = 255;
    }
  }
  kspaceCtx.putImageData(imgData, 0, 0);
}

function renderReconstruction() {
  if (state.currentStep === 0) {
    reconCtx.fillStyle = '#000';
    reconCtx.fillRect(0, 0, N, N);
    return;
  }
  const { mag, max } = reconstruct();
  const imgData = reconCtx.createImageData(N, N);
  const norm = max > 0 ? max : 1;
  for (let i = 0; i < N2; i++) {
    const v = Math.max(0, Math.min(255, Math.round((mag[i] / norm) * 255)));
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  reconCtx.putImageData(imgData, 0, 0);
}

// ============================================================
// UI updates
// ============================================================
function updateUI() {
  const total = N;
  const done = state.currentStep;
  const cleanCount = state.perLineMetadata.filter(m => !m.motionActive).length;
  const corruptCount = state.perLineMetadata.filter(m => m.motionActive).length;
  const pct = (done / total) * 100;

  $('linesRead').textContent = done;
  $('cleanRead').textContent = cleanCount;
  $('corruptRead').textContent = corruptCount;
  $('filledRead').textContent = pct.toFixed(0) + '%';
  $('metaProgress').textContent = `PE LINE ${done} / ${total}`;
  $('acqProgress').style.width = pct + '%';
  $('acqProgressVal').textContent = `${done} / ${total}`;

  // Motion timeline
  const cleanPct = done > 0 ? (cleanCount / done) * 100 : 100;
  $('motionProgress').style.setProperty('--clean-pct', cleanPct + '%');
  $('motionProgress').style.width = pct + '%';
  $('motionProgressVal').textContent = done > 0
    ? `${cleanPct.toFixed(0)}% clean`
    : '100% clean';

  // Motion readouts
  $('dxRead').textContent = state.motionState.dx.toFixed(1);
  $('dyRead').textContent = state.motionState.dy.toFixed(1);
  $('rotRead').textContent = state.motionState.rot.toFixed(1) + '°';
  const stateEl = $('stateRead');
  if (state.motionActive) {
    stateEl.textContent = 'MOVING';
    stateEl.className = 'val warn';
  } else {
    stateEl.textContent = 'STILL';
    stateEl.className = 'val ok';
  }

  // Status pills
  if (state.isScanning) {
    $('systemStatus').textContent = 'SCANNING';
    $('sourceStatus').textContent = state.motionActive ? 'MOTION' : 'IMAGING';
    $('kspaceStatus').textContent = 'FILLING';
    $('reconStatus').textContent = 'UPDATING';
  } else if (done >= total) {
    $('systemStatus').textContent = 'COMPLETE';
    $('sourceStatus').textContent = 'DONE';
    $('kspaceStatus').textContent = 'FULL';
    $('reconStatus').textContent = 'FINAL';
  } else if (done > 0) {
    $('systemStatus').textContent = 'PAUSED';
    $('sourceStatus').textContent = 'PAUSED';
    $('kspaceStatus').textContent = `${done}/${total}`;
    $('reconStatus').textContent = 'PARTIAL';
  } else {
    $('systemStatus').textContent = 'SYSTEM IDLE';
    $('sourceStatus').textContent = 'READY';
    $('kspaceStatus').textContent = 'EMPTY';
    $('reconStatus').textContent = 'WAITING';
  }
}

// ============================================================
// Animation loop
// ============================================================
let lastTickTime = 0;
function tick(now) {
  if (!state.isScanning) return;
  const elapsed = now - lastTickTime;
  // throttle ticks based on speed slider
  const tickInterval = 1000 / (state.speed * 4);  // higher speed = more ticks/sec
  if (elapsed < tickInterval) {
    requestAnimationFrame(tick);
    return;
  }
  lastTickTime = now;

  // Acquire one line
  if (state.currentStep < state.sampleOrder.length) {
    const row = state.sampleOrder[state.currentStep];
    acquireLine(row);
    state.currentStep++;

    renderSource();
    renderScanOverlay(row);
    renderKspace();
    // Reconstruction is the most expensive op — only every few lines
    if (state.currentStep % 2 === 0 || state.currentStep >= state.sampleOrder.length) {
      renderReconstruction();
    }
    updateUI();

    if (state.currentStep >= state.sampleOrder.length) {
      finishScan();
      return;
    }
  }
  requestAnimationFrame(tick);
}

function startScan() {
  if (!state.sourceImage) return;
  if (state.currentStep >= N) resetScan();
  if (!state.sourceFFT) state.sourceFFT = computeFFT(state.sourceImage);
  if (state.sampleOrder.length === 0) state.sampleOrder = buildSampleOrder();
  state.isScanning = true;
  $('startBtn').textContent = '❚❚ Pause';
  $('wiggleBtn').disabled = false;
  $('wiggleBtn').classList.add('attention');   // ← ADD THIS LINE
  lastTickTime = 0;
  requestAnimationFrame(tick);
}

function pauseScan() {
  state.isScanning = false;
  $('startBtn').textContent = '▶ Resume';
  $('wiggleBtn').classList.remove('attention');
  $('wiggleBtn').classList.remove('active');
  updateUI();
}

function finishScan() {
  state.isScanning = false;
  $('startBtn').textContent = '▶ Start Scan';
  $('wiggleBtn').classList.remove('attention');
  $('wiggleBtn').classList.remove('active');
  $('wiggleBtn').disabled = true;
  renderReconstruction();   // final full reconstruction
  renderScanOverlay(null);
  updateUI();
  showPostScanAnalysis();
}

function resetScan() {
  state.isScanning = false;
  state.currentStep = 0;
  state.motionActive = false;
  state.motionStartStep = -1;
  state.motionState = { dx: 0, dy: 0, rot: 0 };
  state.kspace.re.fill(0);
  state.kspace.im.fill(0);
  state.kspaceMask.fill(0);
  state.perLineMetadata = [];
  state.sampleOrder = [];
  $('startBtn').textContent = '▶ Start Scan';
  $('wiggleBtn').disabled = true;
  $('wiggleBtn').classList.remove('active');
  $('wiggleBtn').textContent = '⚠ Wiggle Patient';
  if (state.sourceImage) renderSource();
  renderScanOverlay(null);
  renderKspace();
  reconCtx.fillStyle = '#000';
  reconCtx.fillRect(0, 0, N, N);
  hidePostScanAnalysis();
  updateUI();
}

// ============================================================
// Image setup
// ============================================================
async function setSourceFromImage(img) {
  state.sourceImage = imageToFloatArray(img);
  state.sourceFFT = computeFFT(state.sourceImage);
  resetScan();
  renderSource();
}

function setSourceFromArray(arr) {
  state.sourceImage = arr;
  state.sourceFFT = computeFFT(state.sourceImage);
  resetScan();
  renderSource();
}

const DEMO_IMAGES = ['Demo.png', 'Demo2.png', 'Demo3.png', 'Demo4.png', 'Demo5.png'];
let demoIndex = Math.floor(Math.random() * DEMO_IMAGES.length);  // random start

async function loadDemoImage(advance = false) {
  if (advance) demoIndex = (demoIndex + 1) % DEMO_IMAGES.length;
  const filename = DEMO_IMAGES[demoIndex];
  try {
    const img = await loadImage(filename);
    await setSourceFromImage(img);
  } catch (e) {
    console.warn(`${filename} not found, using procedural phantom`);
    setSourceFromArray(generatePhantom());
  }
}

// ============================================================
// Event wiring
// ============================================================
$('startBtn').addEventListener('click', () => {
  if (state.isScanning) pauseScan();
  else startScan();
});

$('resetBtn').addEventListener('click', resetScan);

$('wiggleBtn').addEventListener('click', () => {
  state.motionActive = !state.motionActive;
  $('wiggleBtn').classList.remove('attention');   // ← ADD THIS LINE
  if (state.motionActive) {
    state.motionStartStep = state.currentStep;
    $('wiggleBtn').classList.add('active');
    $('wiggleBtn').textContent = '⚠ Patient Moving';
  } else {
    $('wiggleBtn').classList.remove('active');
    $('wiggleBtn').textContent = '⚠ Wiggle Patient';
  }
  updateUI();
});

$('speedSlider').addEventListener('input', (e) => {
  state.speed = parseInt(e.target.value, 10);
  $('speedRead').textContent = state.speed + '×';
});

$('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  try {
    const img = await loadImage(url);
    await setSourceFromImage(img);
  } finally {
    URL.revokeObjectURL(url);
  }
});

$('demoBtn').addEventListener('click', () => loadDemoImage(true));

// ============================================================
// Boot
// ============================================================
loadDemoImage().then(updateUI);


// ============================================================================
// ============================================================================
//                  POST-SCAN ANALYSIS & CORRECTION SYSTEM
// ============================================================================
// ============================================================================

// DOM refs for the correction section
const postScanEl = $('postScan');
const cleanBannerEl = $('cleanBanner');
const correctionSectionEl = $('correctionSection');
const corruptedCanvas = $('corruptedCanvas');
const corruptedOverlay = $('corruptedOverlay');
const flaggedKspaceCanvas = $('flaggedKspaceCanvas');
const flaggedKspaceOverlay = $('flaggedKspaceOverlay');
const correctedCanvas = $('correctedCanvas');

const corruptedCtx = corruptedCanvas.getContext('2d');
const corruptedOverlayCtx = corruptedOverlay.getContext('2d');
const flaggedKspaceCtx = flaggedKspaceCanvas.getContext('2d');
const flaggedKspaceOverlayCtx = flaggedKspaceOverlay.getContext('2d');
const correctedCtx = correctedCanvas.getContext('2d');

// Threshold for marking a pixel as "artifacted" — fraction of full-scale brightness
const PIXEL_DIFF_THRESHOLD = 0.12;

// ----------------------------------------------------------------------------
// Show / hide
// ----------------------------------------------------------------------------
function showPostScanAnalysis() {
  postScanEl.style.display = 'block';
  const corruptCount = state.perLineMetadata.filter(m => m.motionActive).length;

  // Reset stage UI
  resetCorrectionUI();

  if (corruptCount === 0) {
    cleanBannerEl.style.display = 'flex';
    correctionSectionEl.style.display = 'none';
  } else {
    cleanBannerEl.style.display = 'none';
    correctionSectionEl.style.display = 'block';
    $('corruptLineCount').textContent = `${corruptCount} corrupted lines detected`;
    $('runCorrectionBtn').disabled = false;
    $('runCorrectionBtn').textContent = '▶ Run Correction';
  }
}

function hidePostScanAnalysis() {
  postScanEl.style.display = 'none';
  cleanBannerEl.style.display = 'none';
  correctionSectionEl.style.display = 'none';
}

function resetCorrectionUI() {
  ['stagePill1', 'stagePill2', 'stagePill3'].forEach(id => {
    $(id).classList.remove('active', 'done');
  });
  $('stageIndicator').textContent = '— / 3';
  ['corrPanel1Status', 'corrPanel2Status', 'corrPanel3Status'].forEach(id => {
    $(id).textContent = 'PENDING';
  });

  // Clear correction canvases
  [corruptedCtx, correctedCtx, flaggedKspaceCtx].forEach(ctx => {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, N, N);
  });
  corruptedOverlayCtx.clearRect(0, 0, N, N);
  flaggedKspaceOverlayCtx.clearRect(0, 0, N, N);

  $('badPxRead').textContent = '—';
  $('flaggedLinesRead').textContent = '—';
  $('keptLinesRead').textContent = '—';
  $('usedLinesRead').textContent = '—';
  $('improveRead').textContent = '—';
  $('threshRead').textContent = PIXEL_DIFF_THRESHOLD.toFixed(2);
}

// ----------------------------------------------------------------------------
// Compute helpers used by the correction stages
// ----------------------------------------------------------------------------

// Build the ground-truth (motion-free) reference reconstruction
function getGroundTruthImage() {
  // Just the original source image — already what we'd get from a clean IFFT
  return state.sourceImage;
}

// Reconstruct from current k-space (the corrupted reconstruction)
function reconstructFromKspace(re, im) {
  const reCopy = new Float64Array(re);
  const imCopy = new Float64Array(im);
  fft2d(reCopy, imCopy, true);
  const mag = new Float64Array(N2);
  let max = 0;
  for (let i = 0; i < N2; i++) {
    mag[i] = Math.sqrt(reCopy[i] * reCopy[i] + imCopy[i] * imCopy[i]);
    if (mag[i] > max) max = mag[i];
  }
  return { mag, max };
}

// Build a "clean only" k-space — zero out rows acquired during motion
function buildCleanKspace() {
  const re = new Float64Array(state.kspace.re);
  const im = new Float64Array(state.kspace.im);
  const flaggedRows = new Set(
    state.perLineMetadata.filter(m => m.motionActive).map(m => m.row)
  );
  for (const row of flaggedRows) {
    for (let c = 0; c < N; c++) {
      re[row * N + c] = 0;
      im[row * N + c] = 0;
    }
  }
  return { re, im, flaggedRows };
}

// Compute pixel-wise difference map between corrupted reconstruction and ground truth
function computeArtifactMap() {
  const corrupted = reconstructFromKspace(state.kspace.re, state.kspace.im);
  const truth = getGroundTruthImage();
  // Normalize corrupted to [0,1] for comparison
  const corruptedNorm = new Float64Array(N2);
  const norm = corrupted.max > 0 ? corrupted.max : 1;
  for (let i = 0; i < N2; i++) corruptedNorm[i] = corrupted.mag[i] / norm;

  const diffMask = new Uint8Array(N2);
  let badCount = 0;
  for (let i = 0; i < N2; i++) {
    const d = Math.abs(corruptedNorm[i] - truth[i]);
    if (d > PIXEL_DIFF_THRESHOLD) {
      diffMask[i] = 1;
      badCount++;
    }
  }
  return { corruptedNorm, diffMask, badCount };
}

// ----------------------------------------------------------------------------
// Render helpers for the correction panels
// ----------------------------------------------------------------------------

// Draw a Float64Array (values 0..1) into a canvas as grayscale
function drawGrayscale(ctx, arr) {
  const imgData = ctx.createImageData(N, N);
  for (let i = 0; i < N2; i++) {
    const v = Math.max(0, Math.min(255, Math.round(arr[i] * 255)));
    imgData.data[i * 4] = v;
    imgData.data[i * 4 + 1] = v;
    imgData.data[i * 4 + 2] = v;
    imgData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imgData, 0, 0);
}

// Draw the k-space (log magnitude) into the flagged-k-space canvas
function drawKspaceMag(ctx, re, im) {
  const imgData = ctx.createImageData(N, N);
  let maxLog = 0;
  for (let i = 0; i < N2; i++) {
    const m = Math.log(1 + Math.sqrt(re[i] ** 2 + im[i] ** 2));
    if (m > maxLog) maxLog = m;
  }
  if (maxLog === 0) maxLog = 1;
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const sr = fftshiftIndex(r);
      const sc = fftshiftIndex(c);
      const idx = sr * N + sc;
      const dispIdx = (r * N + c) * 4;
      const mag = Math.sqrt(re[idx] ** 2 + im[idx] ** 2);
      const v = Math.log(1 + mag) / maxLog;
      const t = Math.min(1, v);
      const rCol = Math.min(255, Math.round(t * 255 * 1.4));
      const gCol = Math.min(255, Math.round(Math.max(0, t - 0.3) * 255 * 1.2));
      const bCol = Math.min(255, Math.round(Math.max(0, t - 0.7) * 255 * 2));
      imgData.data[dispIdx] = rCol;
      imgData.data[dispIdx + 1] = gCol;
      imgData.data[dispIdx + 2] = bCol;
      imgData.data[dispIdx + 3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
}

// ----------------------------------------------------------------------------
// Stage 1: animate red artifact pixels appearing on the corrupted image
// ----------------------------------------------------------------------------
async function runStage1(artifactMap) {
  $('stagePill1').classList.add('active');
  $('stageIndicator').textContent = '1 / 3';
  $('corrPanel1Status').textContent = 'ANALYZING';

  // Draw the corrupted reconstruction first
  drawGrayscale(corruptedCtx, artifactMap.corruptedNorm);

  // Build a sorted list of bad pixel indices for progressive reveal
  const badIndices = [];
  for (let i = 0; i < N2; i++) {
    if (artifactMap.diffMask[i]) badIndices.push(i);
  }
  // Shuffle for organic appearance
  for (let i = badIndices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [badIndices[i], badIndices[j]] = [badIndices[j], badIndices[i]];
  }

  const overlay = corruptedOverlayCtx.createImageData(N, N);
  // Initialize fully transparent
  for (let i = 0; i < N2; i++) overlay.data[i * 4 + 3] = 0;

  const totalSteps = 30;
  const perStep = Math.ceil(badIndices.length / totalSteps);
  let revealed = 0;

  for (let step = 0; step < totalSteps; step++) {
    const end = Math.min(badIndices.length, revealed + perStep);
    for (let k = revealed; k < end; k++) {
      const idx = badIndices[k];
      // Red, semi-transparent
      overlay.data[idx * 4] = 239;
      overlay.data[idx * 4 + 1] = 94;
      overlay.data[idx * 4 + 2] = 68;
      overlay.data[idx * 4 + 3] = 200;
    }
    revealed = end;
    corruptedOverlayCtx.putImageData(overlay, 0, 0);
    $('badPxRead').textContent = revealed.toLocaleString();
    await sleep(28);
  }

  $('stagePill1').classList.remove('active');
  $('stagePill1').classList.add('done');
  $('corrPanel1Status').textContent = 'COMPLETE';
}

// ----------------------------------------------------------------------------
// Stage 2: highlight the flagged k-space rows in blue
// ----------------------------------------------------------------------------
async function runStage2(flaggedRows) {
  $('stagePill2').classList.add('active');
  $('stageIndicator').textContent = '2 / 3';
  $('corrPanel2Status').textContent = 'FLAGGING';

  // Draw the full corrupted k-space first
  drawKspaceMag(flaggedKspaceCtx, state.kspace.re, state.kspace.im);

  // Sweep through flagged rows in display order, painting them blue
  // Note: rows are stored in data-space; after fftshift, display row = (row + N/2) % N
  const sortedRows = [...flaggedRows].sort((a, b) => a - b);
  $('flaggedLinesRead').textContent = '0';
  $('keptLinesRead').textContent = (N - flaggedRows.size).toString();
  $('totalLinesRead').textContent = N.toString();

  flaggedKspaceOverlayCtx.clearRect(0, 0, N, N);

  for (let i = 0; i < sortedRows.length; i++) {
    const dataRow = sortedRows[i];
    // Find display row: which display row corresponds to this data row?
    // fftshiftIndex(displayRow) = dataRow  =>  displayRow = (dataRow + N/2) % N
    const displayRow = (dataRow + N / 2) % N;
    flaggedKspaceOverlayCtx.fillStyle = 'rgba(74, 205, 222, 0.55)';
    flaggedKspaceOverlayCtx.fillRect(0, displayRow, N, 1);
    // Bright leading edge
    flaggedKspaceOverlayCtx.fillStyle = 'rgba(180, 240, 255, 0.9)';
    flaggedKspaceOverlayCtx.fillRect(0, displayRow, N, 1);
    // Settle to softer blue
    setTimeout(() => {
      flaggedKspaceOverlayCtx.fillStyle = 'rgba(74, 205, 222, 0.45)';
      flaggedKspaceOverlayCtx.fillRect(0, displayRow, N, 1);
    }, 100);

    $('flaggedLinesRead').textContent = (i + 1).toString();
    await sleep(Math.max(8, 600 / Math.max(1, sortedRows.length)));
  }

  // Final pass: paint all flagged rows uniformly
  flaggedKspaceOverlayCtx.clearRect(0, 0, N, N);
  flaggedKspaceOverlayCtx.fillStyle = 'rgba(74, 205, 222, 0.5)';
  for (const dataRow of sortedRows) {
    const displayRow = (dataRow + N / 2) % N;
    flaggedKspaceOverlayCtx.fillRect(0, displayRow, N, 1);
  }

  $('stagePill2').classList.remove('active');
  $('stagePill2').classList.add('done');
  $('corrPanel2Status').textContent = 'COMPLETE';
}

// ----------------------------------------------------------------------------
// Stage 3: progressively reconstruct from the clean k-space
// ----------------------------------------------------------------------------
async function runStage3(flaggedRows, artifactMap) {
  $('stagePill3').classList.add('active');
  $('stageIndicator').textContent = '3 / 3';
  $('corrPanel3Status').textContent = 'RECONSTRUCTING';

  // Build clean k-space (zero-filled at flagged rows)
  const usedLines = N - flaggedRows.size;
  $('usedLinesRead').textContent = usedLines.toString();

  // For visual drama, build the reconstruction in stages — start blurry, sharpen
  // We'll render the final reconstruction with a brief "sharpening" sweep effect
  const finalNorm = new Float64Array(state.sourceImage);

  // Animate by scanning a "developing" line down the canvas
  const totalSteps = 40;
  // Start by drawing the corrupted image (the user's mental anchor) fully
  drawGrayscale(correctedCtx, artifactMap.corruptedNorm);

  for (let step = 1; step <= totalSteps; step++) {
    const sweepRow = Math.floor((step / totalSteps) * N);
    // Build a hybrid: pixels above sweepRow show the clean recon; below show corrupted
    const hybrid = new Float64Array(N2);
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx = r * N + c;
        hybrid[idx] = r <= sweepRow ? finalNorm[idx] : artifactMap.corruptedNorm[idx];
      }
    }
    drawGrayscale(correctedCtx, hybrid);
    // Bright sweep line
    correctedCtx.fillStyle = 'rgba(74, 205, 222, 0.7)';
    correctedCtx.fillRect(0, sweepRow, N, 2);
    correctedCtx.fillStyle = 'rgba(180, 240, 255, 0.4)';
    correctedCtx.fillRect(0, sweepRow - 2, N, 6);
    await sleep(28);
  }

  // Final clean draw
  drawGrayscale(correctedCtx, finalNorm);

  // Compute improvement: ratio of bad pixels before vs after
  // (Compared to ground truth)
  const truth = getGroundTruthImage();
  let badAfter = 0;
  for (let i = 0; i < N2; i++) {
    if (Math.abs(finalNorm[i] - truth[i]) > PIXEL_DIFF_THRESHOLD) badAfter++;
  }
  const badBefore = artifactMap.badCount;
  const improvement = badBefore > 0
    ? Math.max(0, Math.round((1 - badAfter / badBefore) * 100))
    : 0;
  $('improveRead').textContent = improvement + '%';

  $('stagePill3').classList.remove('active');
  $('stagePill3').classList.add('done');
  $('corrPanel3Status').textContent = 'COMPLETE';
  $('stageIndicator').textContent = 'DONE';
}

// ----------------------------------------------------------------------------
// Orchestrator
// ----------------------------------------------------------------------------
async function runCorrection() {
  $('runCorrectionBtn').disabled = true;
  $('runCorrectionBtn').textContent = 'Running…';

  resetCorrectionUI();

  // Compute artifact map (reused across stages)
  const artifactMap = computeArtifactMap();
  const cleanKspace = buildCleanKspace();
  const flaggedRows = cleanKspace.flaggedRows;

  await sleep(150);
  await runStage1(artifactMap);
  await sleep(250);
  await runStage2(flaggedRows);
  await sleep(250);
  await runStage3(flaggedRows, artifactMap);

  $('runCorrectionBtn').disabled = false;
  $('runCorrectionBtn').textContent = '↻ Run Again';
}

// ----------------------------------------------------------------------------
// Tiny sleep helper
// ----------------------------------------------------------------------------
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ----------------------------------------------------------------------------
// Wire it up
// ----------------------------------------------------------------------------
$('runCorrectionBtn').addEventListener('click', runCorrection);

